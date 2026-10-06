"""Local feedback demonstrations and federated actor initialization.

Current-observation thermal feedback anchors imitation. Economic scheduling is a
separate causal forecast/controller path, not a second imitation planner.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from gridpfn.core.em_strategy import apply_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.predictive_features import observe
from gridpfn.paths import ROOT


class FeedbackTeacher:
    def __init__(self, scaler, target_temp=20, quota_aware=False, storage_aware=False):
        self.scaler, self.storage_aware = scaler, storage_aware
        index = scaler["col_to_scaler_idx"][4]
        self.low = float(scaler["min"][index])
        self.span = float(scaler["max"][index]) - self.low
        self.span = self.span or 1.0
        self.target_temp = target_temp
        self.quota_aware = quota_aware

    def __call__(self, states):
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != 17:
            raise ValueError("Feedback guidance requires the 17-value control state")
        outdoor = states[:, 7] * self.span + self.low
        indoor = states[:, 9] * 10 + 20
        ac = np.clip((0.7 * indoor + 0.3 * outdoor - self.target_temp) / 3, 0, 2.5)
        hour = states[:, 0] * 24
        if self.quota_aware:
            ac = np.maximum(ac, np.clip(states[:, 15] * 60 / np.maximum(24 - hour, 1), 0, 2.5))
        # Meet remaining charging demand uniformly before departure; the
        # environment still enforces connection, capacity and deadline limits.
        ev = np.where(hour < 8, np.clip(states[:, 11] * 24 / np.maximum(8 - hour, 1), 0, 6), 0)
        battery = np.zeros_like(ac)
        if self.storage_aware:

            def physical(col, value):
                index = self.scaler["col_to_scaler_idx"][col]
                low, high = self.scaler["min"][index], self.scaler["max"][index]
                return np.maximum(0, value * (high - low if high != low else 1) + low)

            # WM's current trace estimates its load; no future samples or oracle
            # schedule are consulted. The environment enforces actual storage bounds.
            wm = physical(7, states[:, 6]) * (
                (states[:, 13] > 0) | ((states[:, 12] > 0) & (hour >= 10))
            )
            net = physical(1, states[:, 2]) - physical(0, states[:, 3]) - ac - ev - wm
            battery = np.clip(net, -2.4, 2.4)
            battery = np.clip(battery, -states[:, 8] * 6.4 / 0.95, (1 - states[:, 8]) * 6.4 / 0.95)
        return np.stack((ac, ev, battery), axis=1).astype(np.float32)


def demonstration_tensors(client, states, targets):
    actor = client.fedavg_agent.actor_net
    features = actor.prepare_features(states).detach()
    targets = torch.as_tensor(targets, device=client.device)
    if hasattr(actor, "control_dt"):
        with torch.no_grad():
            targets = actor.project_action(features, targets)
    return features, targets


def prepare_demonstrations(client):
    """Local training-day coverage, with a private RNG independent of RL sampling."""
    agent = client.fedavg_agent
    if getattr(agent, "expert_memory", None) is not None:
        transitions = agent.expert_memory.buffer
        return demonstration_tensors(
            client,
            np.asarray([t[0] for t in transitions], dtype=np.float32),
            np.asarray([t[1][1] for t in transitions], dtype=np.float32),
        )
    path = None
    if getattr(agent, "demonstration_cache", None):
        real_data = client.train_data[: getattr(client, "real_train_count", len(client.train_data))]
        digest = hashlib.sha256(np.ascontiguousarray(real_data).tobytes())
        digest.update(
            json.dumps(
                {
                    "scaler": client.scaler,
                    "strategy": client.em_strategy,
                    "feedback_target": agent.feedback_target,
                    "quota_guidance": agent.quota_guidance,
                    "storage_guidance": agent.storage_guidance,
                    "predictive_context": getattr(
                        getattr(client, "predictive_context", None), "fingerprint", None
                    ),
                },
                sort_keys=True,
            ).encode()
        )
        for name in (
            "gridpfn/core/control_guidance.py",
            "gridpfn/core/environment.py",
            "gridpfn/core/utils/thermal_planning.py",
        ):
            digest.update((ROOT / name).read_bytes())
        path = Path(agent.demonstration_cache).resolve() / (digest.hexdigest() + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                states, targets = saved["states"], saved["targets"]
            print(f"[imitation-cache] hit={path.name}", flush=True)
            return demonstration_tensors(client, states, targets)
    rng = np.random.default_rng(100)
    teacher = agent.feedback_teacher
    states = []
    for index, day in enumerate(
        client.train_data[: getattr(client, "real_train_count", len(client.train_data))]
    ):
        env = HOME_ENERGY_MGNT(
            day, scaler=client.scaler, fixed_cost=client.fixed_cost, state_dim=17
        )
        apply_em_strategy(env, client.em_strategy)
        state = env.reset()
        for _ in range(env.max_step):
            states.append(observe(client, state, index, env.current_step).copy())
            control = teacher(state[None])[0]
            if rng.random() < 0.5:
                control = rng.uniform([0, 0, -2.4], [2.5, 6, 2.4])
            state, *_ = env.step((int(state[0] * 24 >= 10), control))
    states = np.asarray(states, dtype=np.float32)
    targets = teacher(states[:, :17])
    if path is not None:
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, states=states, targets=targets)
        try:
            # First writer wins: parallel seeds share identical frozen labels,
            # including any time-limited planner's feasible incumbent choices.
            os.link(temporary, path)
        except FileExistsError:
            with np.load(path, allow_pickle=False) as saved:
                states, targets = saved["states"], saved["targets"]
        finally:
            temporary.unlink()
        print(f"[imitation-cache] saved={path.name}", flush=True)
    return demonstration_tensors(client, states, targets)


def initialize_guided_actors(server):
    """Federated BC: each home retains its examples and sends actor weights only."""
    agents = [client.fedavg_agent for client in server.clients]
    rounds = getattr(agents[0], "bc_rounds", 0)
    expert_directory = getattr(agents[0], "oracle_demonstrations", None) or getattr(
        agents[0], "ppo_oracle_init", None
    )
    if expert_directory:
        if not rounds or agents[0].return_mode != "td":
            raise ValueError("Oracle-assisted training requires BC initialization and TD learning")
        from oracle.demonstrations import prepare_experts

        prepare_experts(server, expert_directory)
    for client, agent in zip(server.clients, agents, strict=True):
        if rounds or getattr(agent, "bc_weight", 0):
            agent.feedback_teacher = FeedbackTeacher(
                client.scaler,
                agent.feedback_target,
                getattr(agent, "quota_guidance", False),
                getattr(agent, "storage_guidance", False),
            )
    if not rounds:
        if getattr(agents[0], "residual_scale", None) is not None:
            raise ValueError("Residual control requires imitation initialization")
        return
    data = []
    for index, client in enumerate(server.clients, 1):
        print(f"[imitation] preparing home={index}/{len(server.clients)}", flush=True)
        data.append(prepare_demonstrations(client))
    generators = [
        torch.Generator(device=a.device).manual_seed(100 + i) for i, a in enumerate(agents)
    ]
    optimizers = [torch.optim.Adam(a.actor_net.parameters(), lr=a.bc_lr) for a in agents]
    for index in range(rounds):
        losses = []
        for agent, (features, targets), optimizer, generator in zip(
            agents, data, optimizers, generators, strict=True
        ):
            for _ in range(agent.bc_steps):
                rows = torch.randint(
                    len(features), (agent.batch_size,), device=agent.device, generator=generator
                )
                action = agent.actor_net.forward_features(features[rows])
                loss = (
                    (
                        (action - targets[rows])
                        / (agent.actor_net.action_max - agent.actor_net.action_min)
                    )
                    .square()
                    .mean()
                )
                optimizer.zero_grad(set_to_none=True)
                if hasattr(agent.actor_net, "discrete_fc4") and agent.expert_memory is not None:
                    choices = torch.as_tensor(
                        [agent.expert_memory.buffer[i][1][0] for i in rows.tolist()],
                        dtype=torch.long,
                        device=agent.device,
                    )
                    loss = loss + 0.05 * torch.nn.functional.cross_entropy(
                        agent.actor_net.discrete_features(features[rows]), choices
                    )
                elif agent.actor_update == "ppo":
                    # Same causal WM-at-10 guide used to collect feedback states.
                    choices = (features[rows, -17] * 24 >= 10).long()
                    loss = loss + 0.05 * torch.nn.functional.cross_entropy(
                        agent.actor_net.discrete_features(features[rows]), choices
                    )
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
        params = server._average_parameters(
            [
                {
                    name: value.detach().cpu().clone()
                    for name, value in a.actor_net.state_dict().items()
                }
                for a in agents
            ]
        )
        for agent in agents:
            agent.actor_net.load_state_dict(params)
        print(
            f"[imitation] round={index + 1}/{rounds} normalized_mse={np.mean(losses):.6f}",
            flush=True,
        )
    for agent in agents:
        if agent.residual_scale is not None:
            import copy

            agent.actor_net.enable_residual(agent.residual_scale)
            agent.actor_target_net = copy.deepcopy(agent.actor_net)
        agent.actor_target_net.load_state_dict(agent.actor_net.state_dict())
        # RL uses its own optimizer/rate and fresh moments after pretraining.
        agent.actor_optimizer.state.clear()
    if getattr(agents[0], "oracle_demonstrations", None):
        from oracle.demonstrations import initialize_expert_critics

        initialize_expert_critics(agents, data)
    elif getattr(agents[0], "ppo_oracle_init", None):
        # Expert actions initialize the actor only. Fresh policy rollouts supply
        # value targets and every PPO ratio; no expert replay survives into RL.
        for agent in agents:
            agent.expert_memory = agent.expert_returns = None
