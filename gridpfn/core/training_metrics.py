"""Streaming convergence metrics and read-only periodic FedAvg evaluation."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from gridpfn.core.em_strategy import P2P_TRADING, apply_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.model import precompute_embeddings
from gridpfn.core.predictive_features import observe
from gridpfn.core.utils.convergence import ValidationStopping, feasible, stability
from gridpfn.core.utils.rollout_metrics import (
    calculate_metrics,
    episode_data,
    new_episode_log,
    record_environment,
)
from gridpfn.core.utils.run_io import atomic_json


def mean_present(values):
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def refit_selection_contract(path, episode, data_period, home_ids):
    """Validate the earlier selection run before any fixed-budget refit training."""
    if path is None:
        raise ValueError("Monthly refit requires --refit_selection from its completed selection run")
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    selected = json.loads(raw)["best"]
    source_run = path.parent / "run.json"
    settings = json.loads(source_run.read_text())["settings"]
    status = json.loads((path.parent / "status.json").read_text())
    select_period = data_period.removesuffix("_refit") + "_select"
    if (
        not data_period.endswith("_refit")
        or status.get("state") != "completed"
        or settings.get("data_period") != select_period
        or settings.get("home_ids") != list(home_ids)
        or type(selected["episode"]) is not int
        or selected["episode"] != episode
        or episode < 0
        or selected.get("reward") is None
        or not np.isfinite(selected["reward"])
    ):
        raise ValueError("Refit budget, fold or cohort differs from prior completed selection")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "run_sha256": hashlib.sha256(source_run.read_bytes()).hexdigest(),
        "data_period": select_period,
        "selected_episode": episode,
        "checkpoint": "best",
    }


def save_refit_checkpoint(server, logger, episode, contract, data_period):
    """Save fixed-budget final heads without reading any validation observations."""
    from gridpfn.core.model import encoder_context_identity

    if episode != contract["selected_episode"]:
        raise ValueError("Refit checkpoint exceeds its previously selected budget")
    rule = "fixed previously selected budget; no refit evaluation or checkpoint selection"
    payload = {
        "episode": episode,
        "split": "refit",
        "refit": True,
        "data_period": data_period,
        "reward": None,
        "comfort_pct": None,
        "elec_cost": None,
        "feasible": None,
        "reference_metrics": None,
        "selection_rule": rule,
        "selection_source": contract,
        "home_ids": list(logger.home_ids),
        "encoder_context_sha256": encoder_context_identity(),
        "ac_service": "energy_quota"
        if server.em_strategy.get("ac_energy_quota", True)
        else "thermal",
        "kind": "policy heads; optimizer and replay not included",
        "clients": [],
    }
    for client in server.clients:
        agent = client.fedavg_agent
        payload["clients"].append(
            {
                "predictive_context_sha256": getattr(
                    getattr(client, "predictive_context", None), "fingerprint", None
                ),
                "actor": {
                    name: value.detach().cpu().clone()
                    for name, value in agent.actor_net.state_dict().items()
                },
                "critic": {
                    name: value.detach().cpu().clone()
                    for name, value in agent.critic_net.state_dict().items()
                },
            }
        )
    root = logger.path.parent
    directory = root / "checkpoints/latest"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "heads.tmp"
    torch.save(payload, temporary)
    temporary.replace(directory / "heads.pt")
    summary = {
        "episode": episode,
        "reward": None,
        "comfort_pct": None,
        "elec_cost": None,
        "feasible": None,
        "refit": True,
        "data_period": data_period,
        "selection_rule": rule,
        "selection_source": contract,
        "evaluation_performed": False,
    }
    atomic_json(root / "selection.json", {"latest": summary})
    atomic_json(root / "refit_summary.json", summary)
    logger.write({"kind": "refit_checkpoint", "episode": episode, "evaluation_performed": False})
    return summary


class MetricsLogger:
    def __init__(self, path, home_ids):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise ValueError(f"Existing live metrics in {self.path}; use a fresh run directory")
        self.home_ids = list(home_ids)
        self.started = time.monotonic()

    def write(self, record):
        record = {
            "timestamp": time.time(),
            "elapsed_seconds": time.monotonic() - self.started,
            **record,
        }
        # A complete newline terminates each record; live readers ignore a partial tail.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def training(self, model, episode, results, timings=None, scenario=None):
        homes = [
            {
                "home_id": home_id,
                "reward": float(reward),
                "actor_loss": None if actor is None else float(actor),
                "critic_loss": None if critic is None else float(critic),
            }
            for home_id, (reward, actor, critic) in zip(self.home_ids, results, strict=True)
        ]
        self.write(
            {
                "kind": "train",
                "model": model,
                "episode": episode,
                "homes": homes,
                **({"scenario": scenario} if scenario is not None else {}),
                **({"pipeline_seconds": timings} if timings is not None else {}),
                **{
                    key: mean_present(home[key] for home in homes)
                    for key in ("reward", "actor_loss", "critic_loss")
                },
            }
        )


class PeriodicEvaluator:
    """Evaluate current client policies on fixed test dates, without learning.

    Critic loss is deterministic one-step TD MSE using unchanged target networks
    and requested actions (sum of both errors for twin critics; no target noise).
    Actor loss is the value diagnostic -max Q(s, actor(s)), excluding any training
    imitation penalty or Q scaling. Neither is a classification loss. Repeated
    monitoring of test data should not be treated as untouched final validation.
    """

    def __init__(
        self,
        logger,
        interval,
        days=0,
        split="test",
        min_episodes=0,
        patience=0,
        min_delta=0.01,
        oracle_reference=None,
        strict_convergence=False,
        selection_reference=None,
    ):
        if patience and split != "validation":
            raise ValueError("Early stopping must use validation, never test")
        self.stopping = ValidationStopping(min_episodes, patience, min_delta)
        if strict_convergence and (patience < 4 or patience % 2):
            raise ValueError("Strict convergence requires an even patience of at least four")
        self.strict_convergence, self.validation_history = strict_convergence, []
        self.split = split
        self.best_reward = -float("inf")
        self.best_feasible_reward = -float("inf")
        self.reference_metrics = None
        self.reference_contract = None
        if selection_reference is not None:
            if split != "validation":
                raise ValueError("A service reference must use validation, never test")
            self.reference_contract = json.loads(Path(selection_reference).read_text())
            self.reference_metrics = {
                key: self.reference_contract[key] for key in ("comfort_pct", "elec_cost")
            }
            if not np.isfinite(list(self.reference_metrics.values())).all():
                raise ValueError("Service reference metrics must be finite")
        self.logger = logger
        self.interval = interval
        self.days = days
        self.oracle_reference = oracle_reference
        self.oracle_comparison = None

    def __call__(self, server, episode):
        if self.interval <= 0 or (episode % self.interval and episode != server.episode):
            return
        started = time.monotonic()
        print(f"[eval] FedAvg episode={episode} starting", flush=True)
        self.logger.write({"kind": "eval_start", "model": "fedavg", "episode": episode})
        # Evaluation must not change exploration, replay sampling, or RNG streams.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = sorted(
            {
                client.device.index
                if client.device.index is not None
                else torch.cuda.current_device()
                for client in server.clients
                if client.device.type == "cuda"
            }
        )
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                record = self.evaluate(server)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        record.update(
            kind="eval",
            model="fedavg",
            episode=episode,
            evaluation_seconds=time.monotonic() - started,
            policy="current client policies",
            split=self.split,
            learning_rule=getattr(server.clients[0].fedavg_agent, "actor_update", "q_gradient"),
        )
        self.logger.write(record)
        if self.split == "validation":
            checkpoint_started = time.monotonic()
            self.save_checkpoint(server, episode, record)
            self.logger.write(
                {
                    "kind": "checkpoint",
                    "episode": episode,
                    "seconds": time.monotonic() - checkpoint_started,
                }
            )
            self.stopping.update(episode, record, self.reference_metrics)
            evidence = {}
            if self.strict_convergence:
                self.validation_history.append(record)
                evidence = stability(
                    self.validation_history,
                    self.stopping.min_episodes,
                    self.stopping.patience,
                    self.stopping.min_delta,
                )
                self.stopping.stopped = evidence["stopped"]
            atomic_json(
                self.logger.path.parent / "convergence.json",
                {
                    "episode": episode,
                    "stopped": self.stopping.stopped,
                    "reason": "validation plateau" if self.stopping.stopped else "training",
                    "stale_checks": self.stopping.stale_checks,
                    "patience": self.stopping.patience,
                    "min_episodes": self.stopping.min_episodes,
                    "min_delta": self.stopping.min_delta,
                    "best_meaningful_feasible_reward": self.stopping.best
                    if np.isfinite(self.stopping.best)
                    else None,
                    **evidence,
                },
            )
        print(
            f"[eval] FedAvg episode={episode} days={record['days']} "
            f"actor_loss={record['actor_loss']:.6f} critic_loss={record['critic_loss']:.6f} "
            f"reward={record['reward']:.6f} task_success={record['task_success_pct']:.2f}% "
            f"seconds={record['evaluation_seconds']:.2f}",
            flush=True,
        )

    def before_broadcast(self, server, episode):
        """Paired greedy validation of the local actors before FedAvg replaces them."""
        if not self.interval or episode % self.interval:
            return
        started = time.monotonic()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = sorted({c.device.index for c in server.clients if c.device.type == "cuda"})
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                record = self.evaluate(server, include_losses=False)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        self.logger.write(
            {
                **record,
                "kind": "federation_local",
                "episode": episode,
                "evaluation_seconds": time.monotonic() - started,
            }
        )

    def save_checkpoint(self, server, episode, record):
        """Atomic diagnostic heads, not a claim of exact optimizer/replay resume."""
        from gridpfn.core.model import encoder_context_identity

        if self.reference_contract is not None and (
            self.reference_contract["home_ids"] != self.logger.home_ids
            or self.reference_contract["split"] != self.split
            or self.reference_contract["dates"] != record["dates"]
        ):
            raise ValueError("Service reference cohort, dates or split differs")
        if self.reference_metrics is None:
            self.reference_metrics = {key: record.get(key) for key in ("comfort_pct", "elec_cost")}
        reference = self.reference_metrics
        is_feasible = feasible(record, reference)
        labels = ["latest"]
        if episode == 0:
            labels.append("initial")
        if record["reward"] > self.best_reward:
            labels.append("best")
        if is_feasible and record["reward"] > self.best_feasible_reward:
            labels.append("best_feasible")
            self.best_feasible_reward = record["reward"]
        payload = {
            "episode": episode,
            "split": self.split,
            "reward": record["reward"],
            "comfort_pct": record.get("comfort_pct"),
            "elec_cost": record.get("elec_cost"),
            "reference_metrics": reference,
            "feasible": is_feasible,
            "selection_rule": "max validation reward; best_feasible requires electrical objective <= service reference and comfort >= reference minus 1 percentage point",
            "service_reference": "declared fixed validation reference"
            if self.reference_contract
            else "initial policy",
            "home_ids": self.logger.home_ids,
            "encoder_context_sha256": encoder_context_identity(),
            "ac_service": (
                "energy_quota" if server.em_strategy.get("ac_energy_quota", True) else "thermal"
            )
            if hasattr(server, "em_strategy")
            else None,
            "kind": "policy heads; optimizer and replay not included",
            "clients": [],
        }
        for client in server.clients:
            agent = client.fedavg_agent
            payload["clients"].append(
                {
                    "predictive_context_sha256": getattr(
                        getattr(client, "predictive_context", None), "fingerprint", None
                    ),
                    "actor": {
                        k: v.detach().cpu().clone() for k, v in agent.actor_net.state_dict().items()
                    },
                    "critic": {
                        k: v.detach().cpu().clone()
                        for k, v in agent.critic_net.state_dict().items()
                    },
                }
            )
        for label in labels:
            directory = self.logger.path.parent / "checkpoints" / label
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "heads.tmp"
            torch.save(payload, temporary)
            temporary.replace(directory / "heads.pt")
        self.best_reward = max(self.best_reward, record["reward"])
        selection_path = self.logger.path.parent / "selection.json"
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        for label in labels:
            selection[label] = {
                key: payload[key]
                for key in ("episode", "reward", "comfort_pct", "elec_cost", "feasible")
            }
        temporary = selection_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(selection, allow_nan=False) + "\n")
        temporary.replace(selection_path)

    def prepare_features(self, server):
        """Cache held-out inputs without fitting statistics or collecting labels."""
        dates = sorted(set.intersection(*(set(c.test_dates) for c in server.clients)))
        if self.days:
            dates = dates[: self.days]
        for client in server.clients:
            if getattr(client.fedavg_agent.actor_net, "feature_mode", "frozen") == "raw":
                continue
            rows = []
            for date in dates:
                env = HOME_ENERGY_MGNT(
                    client.test_data[client.test_dates.index(date)],
                    scaler=client.scaler,
                    fixed_cost=client.fixed_cost,
                    state_dim=getattr(client, "state_dim", 9),
                )
                apply_em_strategy(env, server.em_strategy)
                rows.extend(env._state_for_step(t) for t in range(env.max_step + 1))
            precompute_embeddings(np.asarray(rows), client.device)

    def evaluate(self, server, include_days=False, include_losses=True, include_actions=False):
        clients = server.clients
        if self.oracle_reference and self.oracle_comparison is None:
            from oracle.benchmark import OracleComparison

            self.oracle_comparison = OracleComparison(self.oracle_reference, server, self.split)
        shared_dates = sorted(set.intersection(*(set(c.test_dates) for c in clients)))
        if self.days:
            shared_dates = shared_dates[: self.days]
        if not shared_dates:
            raise ValueError("Periodic evaluation needs shared held-out dates")
        envs, logs, states = [], [], []
        for client in clients:
            indices = {date: i for i, date in enumerate(client.test_dates)}
            home_envs = [
                HOME_ENERGY_MGNT(
                    client.test_data[indices[date]],
                    scaler=client.scaler,
                    fixed_cost=client.fixed_cost,
                    state_dim=getattr(client, "state_dim", 9),
                )
                for date in shared_dates
            ]
            for env in home_envs:
                apply_em_strategy(env, server.em_strategy)
            if getattr(client.fedavg_agent.actor_net, "feature_mode", "frozen") != "raw":
                precompute_embeddings(
                    np.asarray(
                        [
                            env._state_for_step(t)
                            for env in home_envs
                            for t in range(env.max_step + 1)
                        ]
                    ),
                    client.device,
                )
            envs.append(home_envs)
            logs.append(
                [
                    new_episode_log(env)
                    | {
                        "energy_bill_without_dr": 0.0,
                        "peer_import_kwh": 0.0,
                        "peer_export_kwh": 0.0,
                    }
                    for env in home_envs
                ]
            )
            states.append(
                [
                    observe(client, env.reset(), date, 0)
                    for env, date in zip(home_envs, shared_dates, strict=True)
                ]
            )
        if len({(env.max_step, env.delta_t) for row in envs for env in row}) != 1:
            raise ValueError("Periodic evaluation requires aligned daily timesteps")
        transitions = [[] for _ in clients]
        action_records = [[[] for _ in shared_dates] for _ in clients] if include_actions else None
        p2p = P2P_TRADING.is_enabled(server.p2p_config)
        for _ in range(envs[0][0].max_step):
            controls = [
                greedy_actions(c.fedavg_agent, np.asarray(home_states))
                for c, home_states in zip(clients, states, strict=True)
            ]
            # Batch network inference over days, preserve independent P2P markets.
            for day in range(len(shared_dates)):
                pending, infos = [], []
                for i, client in enumerate(clients):
                    discrete, control = int(controls[i][day, 0]), controls[i][day, 1:]
                    env, log = envs[i][day], logs[i][day]
                    if include_actions:
                        action_records[i][day].append([discrete, *control.tolist()])
                    next_state, elec, _, comfort, done = env.step((discrete, control))
                    record_environment(log, env)
                    if getattr(client.fedavg_agent, "actor_update", "q_gradient") == "implicit":
                        discrete, control = client.fedavg_agent.replay_action(
                            env, (discrete, control)
                        )
                    pending.append((states[i][day], discrete, control.copy(), next_state, done))
                    infos.append(
                        {
                            "reward_elec": elec,
                            "reward_comf": comfort,
                            "net_load": env.net_load,
                            "pv_surplus": env.pv_surplus,
                            "price": env.price,
                            "export_price": env.export_price,
                            "delta_t": env.delta_t,
                        }
                    )
                adjustments, imports, exports = (np.zeros(len(clients)) for _ in range(3))
                if p2p:
                    adjustments, imports, exports = P2P_TRADING.compute_adjustments(
                        infos, P2P_TRADING.price(server.p2p_config)
                    )
                for i, info in enumerate(infos):
                    env, log = envs[i][day], logs[i][day]
                    env.exported_kwh += exports[i] * env.delta_t
                    transition = (
                        *pending[i][:3],
                        observe(
                            clients[i],
                            env._state_for_step(env.current_step),
                            shared_dates[day],
                            env.current_step,
                        ),
                        pending[i][4],
                    )
                    elec = info["reward_elec"] + adjustments[i]
                    reward = elec + info["reward_comf"]
                    log["episode_reward"] += reward
                    log["episode_elec_cost"] += elec
                    log["episode_comfort"] += info["reward_comf"]
                    log["peer_import_kwh"] += imports[i] * env.delta_t
                    log["peer_export_kwh"] += exports[i] * env.delta_t
                    peer_price = P2P_TRADING.price(server.p2p_config) if p2p else 0.0
                    log["energy_bill_without_dr"] += (
                        (max(0, env.net_load) - imports[i]) * env.price
                        - (max(0, -env.net_load) - exports[i]) * env.export_price
                        + peer_price * (imports[i] - exports[i])
                    ) * env.delta_t + env.fixed_cost / 30 / env.max_step
                    log["powers"]["import"][-1] = max(0.0, log["powers"]["import"][-1] - imports[i])
                    log["powers"]["export"][-1] = max(0.0, log["powers"]["export"][-1] - exports[i])
                    transitions[i].append((*transition, reward))
                    states[i][day] = transition[3]
        homes, day_records = [], []
        for i, client in enumerate(clients):
            actor_loss, critic_loss = (
                self.losses(client.fedavg_agent, transitions[i]) if include_losses else (None, None)
            )
            diagnostics = (
                self.value_diagnostics(client.fedavg_agent, transitions[i], len(shared_dates))
                if include_losses
                else {}
            )
            days = []
            for env, log in zip(envs[i], logs[i], strict=True):
                metrics = calculate_metrics(episode_data(log, env))
                ev = env.ev_energy_delivered + env.eps >= env.ev_required_energy
                wm = not env.wm_required or env.wm_completed
                days.append(
                    {
                        **metrics,
                        "p2p_kwh": log["peer_import_kwh"],
                        "peer_export_kwh": log["peer_export_kwh"],
                        "task_success_pct": 100.0 * (ev and wm),
                        "ev_success_pct": 100.0 * ev,
                        "wm_success_pct": 100.0 * wm,
                        "comfort_pct": 100.0 * (1.0 - metrics["temperature_violation_ratio"]),
                    }
                )
            day_records.append(
                [{**row, "day": date} for date, row in zip(shared_dates, days, strict=True)]
            )
            homes.append(
                {
                    "home_id": self.logger.home_ids[i],
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    **diagnostics,
                    **{key: mean_present(day[key] for day in days) for key in days[0]},
                }
            )
        record = {
            "days": len(shared_dates),
            "dates": shared_dates,
            "homes": homes,
            **({"day_records": day_records} if include_days else {}),
            **({"action_records": action_records} if include_actions else {}),
            **{
                key: mean_present(home[key] for home in homes)
                for key in homes[0]
                if key != "home_id"
            },
        }
        if self.oracle_comparison is not None:
            self.oracle_comparison.annotate(record)
        return record

    @staticmethod
    @torch.no_grad()
    def value_diagnostics(agent, transitions, days):
        """Q calibration against actual finite-day greedy returns, by date.

        Transitions are time-major, with independent days interleaved. Positive
        bias means Q exceeds observed return; it is not a causal diagnosis.
        """
        state, discrete, control, _, done, reward = zip(*transitions, strict=True)
        features = (
            agent.critic_net.prepare_features(np.asarray(state))
            if hasattr(agent.critic_net, "prepare_features")
            else torch.as_tensor(np.asarray(state), dtype=torch.float32, device=agent.device)
        )
        controls = torch.as_tensor(np.asarray(control), dtype=torch.float32, device=agent.device)
        forward = (
            agent.critic_net.forward_features
            if hasattr(agent.critic_net, "forward_features")
            else agent.critic_net
        )
        choices = torch.as_tensor(discrete, device=agent.device, dtype=torch.long)[:, None]
        q = (
            (
                agent.critic_net.value_features(features)
                if getattr(agent, "actor_update", "q_gradient") == "ppo"
                else forward(features, controls).gather(1, choices).squeeze(1)
            )
            .cpu()
            .numpy()
        )
        returns = np.zeros(len(reward))
        running = np.zeros(days)
        for index in range(len(reward) - 1, -1, -1):
            day = index % days
            running[day] = reward[index] + agent.gamma * running[day] * (1 - done[index])
            returns[index] = running[day]
        return {
            "q_return_bias": float(np.mean(q - returns)),
            "q_return_rmse": float(np.sqrt(np.mean((q - returns) ** 2))),
        }

    @staticmethod
    def losses(agent, transitions):
        state, discrete, control, next_state, done, reward = zip(*transitions, strict=True)

        def tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=agent.device)

        prepared = hasattr(agent.actor_net, "prepare_features")
        s = agent.actor_net.prepare_features(np.asarray(state)) if prepared else tensor(state)
        ns = (
            agent.actor_target_net.prepare_features(np.asarray(next_state))
            if prepared
            else tensor(next_state)
        )
        controls = tensor(control)
        actor = agent.actor_net.forward_features if prepared else agent.actor_net
        critic = agent.critic_net.forward_features if prepared else agent.critic_net
        target_actor = (
            agent.actor_target_net.forward_features if prepared else agent.actor_target_net
        )
        target_critic = (
            agent.critic_target_net.forward_features if prepared else agent.critic_target_net
        )
        if getattr(agent.critic_target_net, "twin", False):
            target_critic = agent.critic_target_net.minimum_features
        actions = torch.as_tensor(discrete, dtype=torch.long, device=agent.device).unsqueeze(1)
        if getattr(agent, "actor_update", "q_gradient") == "ppo":
            # Validation has no sampled latent/log-probability: report a value
            # TD diagnostic and categorical NLL, never pretend these are PPO losses.
            value = agent.critic_net.value_features(s)
            target = tensor(reward) + agent.gamma * (
                1 - tensor(done)
            ) * agent.critic_net.value_features(ns)
            categorical = torch.nn.functional.cross_entropy(
                agent.actor_net.discrete_features(s), actions[:, 0]
            )
            return float(categorical), float((value - target).square().mean())
        next_q = (
            agent.critic_net.value_features(ns)
            if getattr(agent, "actor_update", "q_gradient") == "implicit"
            else target_critic(ns, target_actor(ns)).max(1).values
        )
        target = tensor(reward) + agent.gamma * (1.0 - tensor(done)) * next_q
        q = critic(s, controls).gather(1, actions).squeeze(1)
        critic_loss = (q - target).square().mean().item()
        if getattr(agent.critic_net, "twin", False):
            _, second = agent.critic_net.both_features(s, controls)
            critic_loss += (second.gather(1, actions).squeeze(1) - target).square().mean().item()
        actor_loss = -critic(s, actor(s)).max(1).values.mean().item()
        return actor_loss, critic_loss


@torch.no_grad()
def greedy_actions(agent, states):
    if hasattr(agent.actor_net, "prepare_features"):
        features = agent.actor_net.prepare_features(states)
        control = agent.actor_net.forward_features(features)
        discrete = (
            agent.actor_net.discrete_features(features)
            if hasattr(agent.actor_net, "discrete_fc4")
            else agent.critic_net.forward_features(features, control)
        ).argmax(1)
    else:
        states = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
        control = agent.actor_net(states)
        discrete = agent.critic_net(states, control).argmax(1)
    return torch.cat((discrete[:, None], control), dim=1).cpu().numpy()
