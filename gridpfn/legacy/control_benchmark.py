"""Certified AC comfort limits and causal controller rollouts under original constraints.

The thermal oracle uses realized daily weather solely as a diagnostic. It optimizes
the count of comfortable hours, not reward, and never supplies training labels.
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gridpfn.core.control_guidance import FeedbackTeacher
from gridpfn.core.dataset import home_data_dir, load_data
from gridpfn.core.em_strategy import (
    P2P_TRADING,
    apply_em_strategy,
    compose_em_strategy,
    make_em_strategy,
)
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.training_config import HOME_IDS
from gridpfn.core.utils.rollout_metrics import (
    calculate_metrics,
    episode_data,
    new_episode_log,
    record_environment,
)
from gridpfn.core.utils.run_io import atomic_json
from gridpfn.core.utils.thermal_planning import thermal_limit


def load_clients(home_ids=HOME_IDS, split="validation", data_dir=home_data_dir):
    data = load_data(
        data_dir,
        "*.csv",
        choose=[f"home_{h}" for h in home_ids],
        validation_days=14,
        split=split,
    )
    return [
        SimpleNamespace(train_data=t, test_data=v, test_dates=d, scaler=s, fixed_cost=5)
        for t, v, d, s in data
    ]


def rollout_controllers(clients, strategy, controllers):
    """Evaluate causal (discrete, continuous) controller callables with P2P."""
    dates = sorted(set.intersection(*(set(c.test_dates) for c in clients)))
    coordinated = hasattr(controllers, "dispatch")
    for client, controller in zip(
        clients, ([None] * len(clients) if coordinated else controllers), strict=True
    ):
        actor = getattr(controller, "actor_net", None)
        if actor is not None and actor.feature_mode != "raw":
            from gridpfn.core.model import precompute_embeddings

            states = []
            for date in dates:
                env = HOME_ENERGY_MGNT(
                    client.test_data[client.test_dates.index(date)],
                    scaler=client.scaler,
                    state_dim=17,
                )
                apply_em_strategy(env, strategy)
                states.extend(env._state_for_step(t) for t in range(env.max_step + 1))
            precompute_embeddings(np.asarray(states), actor.fc1.weight.device)
    home_records = [[] for _ in clients]
    for date in dates:
        if coordinated:
            controllers.start_day(date)
        envs = [
            HOME_ENERGY_MGNT(
                c.test_data[c.test_dates.index(date)],
                scaler=c.scaler,
                fixed_cost=c.fixed_cost,
                state_dim=17,
            )
            for c in clients
        ]
        for env in envs:
            apply_em_strategy(env, strategy)
        states = [e.reset() for e in envs]
        logs = [new_episode_log(e) for e in envs]
        overrides = np.zeros((len(envs), 3))
        energy = [
            dict(
                p2p=0.0,
                pv=0.0,
                curtailed=0.0,
                balance_error=0.0,
                bill=0.0,
                dr=0.0,
                local_pv=0.0,
            )
            for _ in clients
        ]
        for _ in range(envs[0].max_step):
            infos = []
            actions = (
                controllers.dispatch(states)
                if coordinated
                else [
                    controller(state) for controller, state in zip(controllers, states, strict=True)
                ]
            )
            for i, (env, action, log) in enumerate(zip(envs, actions, logs, strict=True)):
                _, elec, _, comfort, _ = env.step(action)
                actual = np.array([env.power_AC, env.power_EV, env.power_BESS])
                overrides[i] += np.abs(actual - action[1]) > 1e-5
                record_environment(log, env)
                infos.append(
                    dict(
                        reward_elec=elec,
                        reward_comf=comfort,
                        net_load=env.net_load,
                        pv_surplus=env.pv_surplus,
                        price=env.price,
                        export_price=env.export_price,
                        delta_t=env.delta_t,
                    )
                )
            adjustments, imports, exports = P2P_TRADING.compute_adjustments(infos, 0.1)
            for i, (env, log, info) in enumerate(zip(envs, logs, infos, strict=True)):
                env.exported_kwh += exports[i] * env.delta_t
                states[i] = env._state_for_step(env.current_step)
                log["episode_reward"] += info["reward_elec"] + adjustments[i] + info["reward_comf"]
                log["episode_elec_cost"] += info["reward_elec"] + adjustments[i]
                log["episode_comfort"] += info["reward_comf"]
                log["powers"]["import"][-1] -= imports[i]
                log["powers"]["export"][-1] -= exports[i]
                raw_net = (
                    env.fixed_load
                    + env.power_AC
                    + env.power_EV
                    + env.power_WM
                    + env.power_BESS
                    - env.pv_generation
                )
                curtailment = max(0.0, env.net_load - raw_net)
                consumed = raw_net + env.pv_generation
                balance = (
                    env.pv_generation
                    + log["powers"]["import"][-1]
                    + imports[i]
                    - consumed
                    - log["powers"]["export"][-1]
                    - exports[i]
                    - curtailment
                )
                energy[i]["p2p"] += imports[i] * env.delta_t
                energy[i]["pv"] += env.pv_generation * env.delta_t
                energy[i]["curtailed"] += curtailment * env.delta_t
                energy[i]["local_pv"] += (
                    max(
                        0,
                        env.pv_generation - curtailment - log["powers"]["export"][-1] - exports[i],
                    )
                    * env.delta_t
                )
                energy[i]["bill"] += (
                    log["powers"]["import"][-1] * env.price
                    - log["powers"]["export"][-1] * env.export_price
                    + 0.1 * (imports[i] - exports[i])
                ) * env.delta_t + env.fixed_cost / 30 / env.max_step
                if env.dr_limit is not None:
                    energy[i]["dr"] += (
                        -env.dr_penalty * max(0, env.net_load - env.dr_limit)
                        + env.dr_incentive * max(0, env.dr_limit - env.net_load)
                    ) * env.delta_t
                energy[i]["balance_error"] = max(energy[i]["balance_error"], abs(balance))
        for records, env, log, override, flow in zip(
            home_records, envs, logs, overrides, energy, strict=True
        ):
            metrics = calculate_metrics(episode_data(log, env))
            records.append(
                {
                    **metrics,
                    "day": date,
                    "comfort_pct": 100 * (1 - metrics["temperature_violation_ratio"]),
                    "task_success_pct": 100.0
                    * float(
                        metrics["ev_completion_ratio"] >= 1 - 1e-6 and metrics["wm_completed"] == 1
                    ),
                    "p2p_kwh": flow["p2p"],
                    "energy_bill_without_dr": flow["bill"],
                    "dr_net_credit": flow["dr"],
                    "billing_reconciliation_error": abs(
                        metrics["elec_cost"] - (flow["bill"] - flow["dr"])
                    ),
                    "pv_local_use_kwh": flow["local_pv"],
                    "pv_generated_kwh": flow["pv"],
                    "pv_curtailed_kwh": flow["curtailed"],
                    "pv_utilized_pct": 100 * (1 - flow["curtailed"] / flow["pv"])
                    if flow["pv"] > 1e-9
                    else 100.0,
                    "energy_balance_max_abs_kw": flow["balance_error"],
                    **{
                        f"{name}_override_pct": 100 * value / env.max_step
                        for name, value in zip(("ac", "ev", "battery"), override, strict=True)
                    },
                }
            )
    return {
        "dates": dates,
        "day_records": home_records,
        "homes": [
            {k: float(np.mean([d[k] for d in days])) for k in days[0] if k != "day"}
            for days in home_records
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/control_benchmark/benchmark.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    clients = load_clients()
    strategy = compose_em_strategy(make_em_strategy(), clients)
    limits = []
    dates = sorted(set.intersection(*(set(c.test_dates) for c in clients)))
    for home, client in zip(HOME_IDS, clients, strict=True):
        records = []
        for date in dates:
            env = HOME_ENERGY_MGNT(
                client.test_data[client.test_dates.index(date)], scaler=client.scaler, state_dim=17
            )
            apply_em_strategy(env, strategy)
            records.append({"day": date, **thermal_limit(env)})
        limits.append(
            {
                "home_id": home,
                "days": records,
                "comfort_limit_pct": float(np.mean([r["comfort_limit_pct"] for r in records])),
            }
        )
        print(f"[limit] home={home} comfort={limits[-1]['comfort_limit_pct']:.3f}%", flush=True)
    teachers = [FeedbackTeacher(c.scaler) for c in clients]
    feedback = rollout_controllers(
        clients, strategy, [lambda s, t=t: (int(s[0] * 24 >= 10), t(s[None])[0]) for t in teachers]
    )
    atomic_json(
        args.output,
        dict(
            home_ids=list(HOME_IDS),
            split="validation",
            ac_service="energy_quota",
            thermal_limits=limits,
            feedback=feedback,
            seconds=time.monotonic() - started,
            note="Oracle optimizes comfort count with realized weather; not reward or a causal policy.",
        ),
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
