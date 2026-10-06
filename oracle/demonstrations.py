"""Training-only expert replay, independently executed in the learning simulator."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from gridpfn.core.dataset import temp_price_path
from gridpfn.core.em_strategy import P2P_TRADING, apply_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.utils.agent_utils import ReplayBuffer
from gridpfn.core.utils.run_io import atomic_json, file_sha256

from .config import OracleConfig
from .runner import read_record


def validate_protocol(manifest, clients, strategy, p2p_config, training=True):
    """Reject held-out labels, a different ring, data or physical scenario."""
    if training and (manifest.get("split") != "train" or not manifest.get("training_labels")):
        raise ValueError(
            "Oracle demonstrations must have split=train; held-out labels are forbidden"
        )
    if manifest["home_ids"] != [c.home_id for c in clients]:
        raise ValueError("Oracle demonstrations require the identical home ring and order")
    scenario = OracleConfig(**manifest["scenario"])
    default = OracleConfig()
    operational = {
        "name",
        "data_dir",
        "output",
        "split",
        "home_ids",
        "days",
        "objectives",
        "workers",
        "time_limit",
        "gap",
    }
    if not training:
        # Evaluation may vary the declared peer tariff. The actual settlement
        # price is checked below; privileged training labels remain original-only.
        operational.update(
            ("peer_price", "flat_price_per_kwh", "hourly_prices_per_kwh", "tou_enabled")
        )
    if any(
        getattr(scenario, f.name) != getattr(default, f.name)
        for f in fields(scenario)
        if f.name not in operational
    ):
        raise ValueError(
            "Expert replay currently supports the unchanged original physical scenario"
        )
    if any(
        (training and c.state_dim != 17) or c.fixed_cost != scenario.fixed_cost for c in clients
    ):
        raise ValueError(
            "Expert replay requires the original fixed cost and 17-value control state"
        )
    expected_prices = scenario.hourly_prices_per_kwh
    if scenario.flat_price_per_kwh is not None:
        expected_prices = (scenario.flat_price_per_kwh,) * 24
    for client in clients:
        actual_prices = client.scaler.get("grid_prices")
        if (actual_prices is None) != (expected_prices is None) or (
            expected_prices is not None and not np.array_equal(actual_prices, expected_prices)
        ):
            raise ValueError("Oracle explicit grid tariff differs")
    for key, value in manifest["strategy"].items():
        actual = strategy.get(key)
        if key == "tou":
            if (
                actual is None
                or actual["enabled"] != value["enabled"]
                or actual["n_blocks"] != value["n_blocks"]
            ):
                raise ValueError("Oracle tariff configuration differs")
            if value["enabled"]:
                np.testing.assert_allclose(
                    actual["hourly_prices"], value["hourly_prices"], atol=1e-12, rtol=0
                )
        elif actual != value:
            raise ValueError(f"Oracle strategy differs: {key}")
    if (
        not strategy.get("ac_energy_quota", True)
        or not P2P_TRADING.is_enabled(p2p_config)
        or P2P_TRADING.price(p2p_config) != scenario.peer_price
    ):
        raise ValueError("Oracle AC quota or peer market differs")
    dates = manifest["dates"]
    if (
        not dates
        or len(set(dates)) != len(dates)
        or any(
            not set(dates).issubset(c.scaler["train_dates"] if training else c.test_dates)
            for c in clients
        )
    ):
        raise ValueError("Every expert date must belong to every home's training split")
    for name, expected in manifest["input_file_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError("Oracle home data changed")
    if file_sha256(temp_price_path) != manifest["price_weather_sha256"]:
        raise ValueError("Oracle weather/price data changed")


def prepare_experts(server, directory):
    """Keep each home's transitions private, after synchronized peer settlement."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "oracle.json").read_text())
    if json.loads((directory / "status.json").read_text())["state"] != "completed":
        raise ValueError("Oracle demonstrations must be completed and certified")
    validate_protocol(manifest, server.clients, server.em_strategy, server.p2p_config)
    objective = server.clients[0].fedavg_agent.expert_objective
    rows = [[] for _ in server.clients]
    for date in manifest["dates"]:
        record = read_record(directory / "days" / f"{date}.json", manifest)
        oracle = record["oracles"][objective]
        if not oracle["solution"]["certified"]:
            raise ValueError("An expert schedule is not certified")
        envs = [
            HOME_ENERGY_MGNT(
                c.train_data[c.scaler["train_dates"].index(date)],
                scaler=c.scaler,
                fixed_cost=c.fixed_cost,
                state_dim=17,
            )
            for c in server.clients
        ]
        for env in envs:
            apply_em_strategy(env, server.em_strategy)
        states = [env.reset() for env in envs]
        total = np.zeros(len(envs))
        for hour in range(24):
            pending, infos = [], []
            for env, state, solution in zip(
                envs, states, oracle["solution"]["solutions"], strict=True
            ):
                action = (
                    int(solution["wm_start"] == hour),
                    np.asarray(solution["controls"][hour], dtype=np.float32),
                )
                _, elec, _, comfort, done = env.step(action)
                np.testing.assert_allclose(
                    action[1], [env.power_AC, env.power_EV, env.power_BESS], atol=2e-5, rtol=0
                )
                pending.append((state, action, elec + comfort, done))
                infos.append(
                    dict(
                        net_load=env.net_load,
                        pv_surplus=env.pv_surplus,
                        price=env.price,
                        export_price=env.export_price,
                        delta_t=env.delta_t,
                    )
                )
            adjustment, _, exports = P2P_TRADING.compute_adjustments(
                infos, P2P_TRADING.price(server.p2p_config)
            )
            for i, env in enumerate(envs):
                env.exported_kwh += exports[i] * env.delta_t
                state, action, reward, done = pending[i]
                reward += adjustment[i]
                next_state = env._state_for_step(env.current_step)
                rows[i].append((state.copy(), action, reward, next_state.copy(), done))
                states[i] = next_state
                total[i] += reward
        np.testing.assert_allclose(
            total, [h["reward"] for h in oracle["audit"]["homes"]], atol=1e-4, rtol=1e-7
        )
    for client, transitions in zip(server.clients, rows, strict=True):
        agent = client.fedavg_agent
        agent.expert_memory = ReplayBuffer(len(transitions))
        for transition in transitions:
            agent.expert_memory.store_transition(*transition)
        returns, running = np.empty(len(transitions)), 0.0
        for i in range(len(transitions) - 1, -1, -1):
            running = transitions[i][2] + agent.gamma * running * (not transitions[i][4])
            returns[i] = running
        agent.expert_returns = returns.astype(np.float32)
    if server.metrics_logger is not None:
        atomic_json(
            server.metrics_logger.path.parent / "expert_provenance.json",
            {
                "oracle_manifest_sha256": file_sha256(directory / "oracle.json"),
                "split": "train",
                "dates": manifest["dates"],
                "home_ids": manifest["home_ids"],
                "objective": objective,
                "labels": "Perfect-foresight training demonstrations only; inference uses the learned heads",
                "transitions_per_home": len(rows[0]),
            },
        )


def initialize_expert_critics(agents, data):
    """Fit private Q heads to actual expert returns before online bootstrapping."""
    for home, (agent, (features, controls)) in enumerate(zip(agents, data, strict=True)):
        if agent.expert_memory is None:
            continue
        choices = torch.as_tensor(
            [t[1][0] for t in agent.expert_memory.buffer], dtype=torch.long, device=agent.device
        )
        returns = torch.as_tensor(agent.expert_returns, device=agent.device)
        optimizer = torch.optim.Adam(agent.critic_net.parameters(), lr=agent.bc_lr)
        generator = torch.Generator(device=agent.device).manual_seed(200 + home)
        for _ in range(agent.expert_critic_steps):
            rows = torch.randint(
                len(features), (agent.batch_size,), generator=generator, device=agent.device
            )
            first, second = agent.critic_net.both_features(features[rows], controls[rows])
            choice = choices[rows, None]
            loss = agent.td_loss(first.gather(1, choice).squeeze(1), returns[rows])
            if agent.critic_net.twin:
                loss += agent.td_loss(second.gather(1, choice).squeeze(1), returns[rows])
            if hasattr(agent.critic_net, "value_fc1"):
                loss += agent.td_loss(
                    agent.critic_net.value_features(features[rows]), returns[rows]
                )
            # A small expert-action classification term initializes the WM choice;
            # returns supervise only the action actually executed by the expert.
            loss += 0.1 * torch.nn.functional.cross_entropy(first, choices[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        agent.critic_target_net.load_state_dict(agent.critic_net.state_dict())
        print(f"[expert-critic] home={home + 1} updates={agent.expert_critic_steps}", flush=True)
