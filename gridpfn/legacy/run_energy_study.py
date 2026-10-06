"""Chronological forecast selection and matched causal energy-scheduling ablations."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gridpfn.core.control_guidance import FeedbackTeacher
from gridpfn.core.dataset import home_data_dir
from gridpfn.core.economic_control import EconomicCoordinator
from gridpfn.core.em_strategy import compose_em_strategy, make_em_strategy
from gridpfn.core.forecasting import DirectForecaster, forecast_metrics, physical_series
from gridpfn.core.model import configure_embedding_device, heads_from_state
from gridpfn.core.training_config import HOME_IDS
from gridpfn.core.utils.agent_utils import safe_torch_load
from gridpfn.core.utils.run_io import atomic_json, backbone_identity, file_sha256
from gridpfn.experiments.run_experiment import snapshot
from gridpfn.legacy.control_benchmark import load_clients, rollout_controllers
from gridpfn.paths import ROOT


def forecast_bank(clients, output, args, stage):
    kinds = ("persistence", "seasonal", "trees", "tabpfn")
    banks = {kind: [] for kind in kinds}
    inner_banks = {kind: [] for kind in banks}
    inner_clients = []
    measurements = []
    for home_id, client in zip(HOME_IDS, clients, strict=True):
        train = physical_series(client.train_data, client.scaler)
        query = physical_series(client.test_data, client.scaler)
        cutoff = str(date.fromisoformat(client.scaler["train_end_exclusive"]) - timedelta(days=7))
        dates = np.asarray(client.scaler["train_dates"])
        fit_mask = dates < cutoff
        fit, check = train[fit_mask], train[~fit_mask]
        check_dates = dates[~fit_mask].tolist()
        inner_clients.append(
            SimpleNamespace(
                train_data=client.train_data[fit_mask],
                test_data=client.train_data[~fit_mask],
                test_dates=check_dates,
                scaler=client.scaler,
                fixed_cost=client.fixed_cost,
            )
        )

        row = {
            "home_id": home_id,
            "training_days": len(train),
            "selection_days": 7,
            "forecasts": {},
        }
        for kind in kinds:
            stage(f"forecast {kind}, home {home_id}")
            model = DirectForecaster(
                kind, args.context, args.estimators, args.seed, args.device
            ).fit(fit)
            inner = model.predict_table(check)
            inner_banks[kind].append(dict(zip(check_dates, inner, strict=True)))
            selection = forecast_metrics(inner, check)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = DirectForecaster(
                kind, args.context, args.estimators, args.seed, args.device
            ).fit(train)
            table = model.predict_table(query)
            banks[kind].append(dict(zip(client.test_dates, table, strict=True)))
            row["forecasts"][kind] = {
                "training_selection_rmse": selection,
                "evaluation_rmse": forecast_metrics(table, query),
                "fit_seconds": model.fit_seconds,
                "predict_seconds": model.predict_seconds,
            }
            np.savez_compressed(output / f"forecast_{home_id}_{kind}.npz", table=table)
            np.savez_compressed(output / f"inner_{home_id}_{kind}.npz", table=inner)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        measurements.append(row)
        atomic_json(output / "forecasts.json", measurements)
        print(f"[forecast] home={home_id} complete", flush=True)
    for collection in (banks, inner_banks):
        collection["blend"] = [
            {day: 0.5 * home[day] + 0.5 * seasonal[day] for day in home}
            for home, seasonal in zip(collection["tabpfn"], collection["seasonal"], strict=True)
        ]
    return banks, measurements, inner_banks, inner_clients


def forecast_signature(clients, args):
    digest = hashlib.sha256(ROOT / "gridpfn/core/forecasting.py".read_bytes())
    digest.update(json.dumps([args.context, args.estimators, args.seed, list(HOME_IDS)]).encode())
    for client in clients:
        digest.update(physical_series(client.train_data, client.scaler).tobytes())
        digest.update(physical_series(client.test_data, client.scaler).tobytes())
        digest.update(json.dumps([client.scaler["train_dates"], client.test_dates]).encode())
    return digest.hexdigest()


def checkpoint_controllers(run, device="cpu"):
    settings = json.loads((run / "run.json").read_text())["settings"]
    if settings.get("scaler_mode", "local") != "local":
        raise ValueError("Comparison checkpoint must use the same local training scales")
    payload = safe_torch_load(run / "checkpoints/best_feasible/heads.pt", device)
    if payload["home_ids"] != list(HOME_IDS):
        raise ValueError("Checkpoint cohort differs from the scheduling study")
    policies = []
    for saved in payload["clients"]:
        actor, critic = heads_from_state(saved["actor"], saved["critic"], device)

        def policy(state, actor=actor, critic=critic):
            with torch.no_grad():
                features = actor.prepare_features(np.asarray(state)[None])
                control = actor.forward_features(features)
                choice = int(critic.forward_features(features, control).argmax(1).item())
                return choice, control[0].cpu().numpy()

        policy.actor_net = actor
        policies.append(policy)
    return policies


def comparison(reference, candidate):
    a, b = reference["homes"], candidate["homes"]
    cost_a, cost_b = (np.mean([h["elec_cost"] for h in homes]) for homes in (a, b))
    reward_a, reward_b = (np.mean([h["reward"] for h in homes]) for homes in (a, b))
    comfort = np.array([h["comfort_pct"] for h in b]) - np.array([h["comfort_pct"] for h in a])
    return {
        "objective_reduction_per_home_day": float(cost_a - cost_b),
        "relative_objective_reduction": float((cost_a - cost_b) / max(abs(cost_a), 1e-8)),
        "reward_improvement": float(reward_b - reward_a),
        "minimum_home_comfort_delta_pp": float(comfort.min()),
        "minimum_ev_completion": min(h["ev_completion_ratio"] for h in b),
        "minimum_wm_completion": min(h["wm_completed"] for h in b),
        "max_energy_balance_error_kw": max(h["energy_balance_max_abs_kw"] for h in b),
        "energy_bill_reduction": float(
            np.mean([h["energy_bill_without_dr"] for h in a])
            - np.mean([h["energy_bill_without_dr"] for h in b])
        ),
        "gate": bool(
            cost_b < cost_a
            and reward_b > reward_a
            and comfort.min() >= -0.1
            and np.mean([h["energy_bill_without_dr"] for h in b])
            < np.mean([h["energy_bill_without_dr"] for h in a])
            and max(h["energy_balance_max_abs_kw"] for h in b) < 1e-8
            and all(h["ev_completion_ratio"] > 1 - 1e-6 and h["wm_completed"] == 1 for h in b)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/energy_study"))
    parser.add_argument("--context", type=int, default=1536)
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data_dir", type=Path, default=Path(home_data_dir))
    parser.add_argument("--reuse_forecasts", type=Path)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME=RUN",
        help="Optional completed selected RL checkpoints to compare on the same dates",
    )
    args = parser.parse_args()
    if args.context < 1 or args.estimators < 1 or args.days < 0:
        raise ValueError("Use positive context/ensemble budgets and nonnegative days")
    output = args.output.resolve()
    root = ROOT
    if not output.is_relative_to(root / "results"):
        raise ValueError("Study output must be inside the repository results/ directory")
    specifications = []
    names = {"feedback"}
    for specification in args.baseline:
        name, separator, path = specification.partition("=")
        if not separator or not name or name in names or name.endswith(("_local", "_peer")):
            raise ValueError("Use unique baseline names NAME=RUN, without _local/_peer suffixes")
        run = Path(path).resolve()
        if json.loads((run / "status.json").read_text()).get("state") != "completed":
            raise ValueError("Comparison checkpoint must be from a completed run")
        if not (run / "checkpoints/best_feasible/heads.pt").is_file():
            raise ValueError("Missing selected baseline checkpoint")
        specifications.append((name, run))
        names.add(name)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(3)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(0.4, args.device)
    configure_embedding_device(args.device)
    clients = load_clients(split=args.split, data_dir=args.data_dir)
    strategy = compose_em_strategy(make_em_strategy(), clients)
    snapshot(root, output)
    source_hashes = json.loads((output / "source_hashes.json").read_text())
    signature = forecast_signature(clients, args)
    atomic_json(
        output / "run.json",
        {
            "created_at": time.time(),
            "gpu": args.device,
            "seed": args.seed,
            "clients": len(clients),
            "home_ids": list(HOME_IDS),
            "episodes": 13 + len(args.baseline),
            "mode": "scheduling",
            "eval_step": 1,
            "eval_split": args.split,
            "settings": vars(args)
            | {
                "output": str(output),
                "data_dir": str(args.data_dir.resolve()),
                "reuse_forecasts": str(args.reuse_forecasts) if args.reuse_forecasts else None,
            },
            "note": "Evaluation pass index, not RL episodes. Frozen forecast fitting followed by exact-simulator controller ablations.",
        },
    )

    def stage(message):
        atomic_json(
            output / "status.json", {"state": "running", "pid": os.getpid(), "stage": message}
        )
        print(f"[study] {message}", flush=True)

    try:
        if args.reuse_forecasts:
            previous = json.loads((args.reuse_forecasts / "study_results.json").read_text())
            if previous.get("forecast_signature") != signature:
                raise ValueError(
                    "Forecast cache differs in data, cohort, source, seed or context settings"
                )
            measurements = json.loads((args.reuse_forecasts / "forecasts.json").read_text())
            atomic_json(output / "forecasts.json", measurements)
            for path in args.reuse_forecasts.glob("forecast_*.npz"):
                (output / path.name).write_bytes(path.read_bytes())
            banks = {kind: [] for kind in ("persistence", "seasonal", "trees", "tabpfn")}
            for home, client in zip(HOME_IDS, clients, strict=True):
                for kind in banks:
                    table = np.load(output / f"forecast_{home}_{kind}.npz")["table"]
                    banks[kind].append(dict(zip(client.test_dates, table, strict=True)))
            banks["blend"] = [
                {day: 0.5 * home[day] + 0.5 * seasonal[day] for day in home}
                for home, seasonal in zip(banks["tabpfn"], banks["seasonal"], strict=True)
            ]
            inner_selection = previous["training_cost_selection"]
        else:
            banks, measurements, inner_banks, inner_clients = forecast_bank(
                clients, output, args, stage
            )
            inner_strategy = compose_em_strategy(make_em_strategy(), inner_clients)
            inner_feedback = [FeedbackTeacher(c.scaler, 18.2) for c in inner_clients]
            reference = rollout_controllers(
                inner_clients,
                inner_strategy,
                [lambda s, t=t: (int(s[0] * 24 >= 10), t(s[None])[0]) for t in inner_feedback],
            )
            scores = {}
            for kind in inner_banks:
                stage(f"training-only scheduling selection {kind}")
                controller = EconomicCoordinator(inner_clients, inner_strategy, inner_banks[kind])
                score = rollout_controllers(inner_clients, inner_strategy, controller)
                scores[kind] = {
                    "mean_objective": float(np.mean([h["elec_cost"] for h in score["homes"]])),
                    "comparison": comparison(reference, score),
                    "homes": score["homes"],
                    "dates": score["dates"],
                }
            eligible = [kind for kind, score in scores.items() if score["comparison"]["gate"]]
            winner = (
                min(eligible, key=lambda kind: scores[kind]["mean_objective"])
                if eligible
                else "seasonal"
            )
            inner_selection = {
                "winner": winner,
                "scores": scores,
                "note": "Selected on the final seven training-calendar dates, before July controller evaluation; fallback seasonal if no candidate passes.",
            }
        banks["economic_selected"] = banks[inner_selection["winner"]]
        if args.days:
            dates = sorted(set.intersection(*(set(c.test_dates) for c in clients)))[: args.days]
            for c in clients:
                indices = [c.test_dates.index(date) for date in dates]
                c.test_data, c.test_dates = c.test_data[indices], dates
        results = {
            "home_ids": list(HOME_IDS),
            "split": args.split,
            "forecasts": measurements,
            "forecast_signature": signature,
            "training_cost_selection": inner_selection,
            "controllers": {},
            "comparisons": {},
            "settings": {
                "context": args.context,
                "estimators": args.estimators,
                "seed": args.seed,
                "strategy": strategy,
            },
            "source_hashes": source_hashes,
            "backbone": backbone_identity(),
            "data_hashes": {
                path.name: file_sha256(path) for path in sorted(args.data_dir.glob("*.csv"))
            },
            "runtime": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "scikit-learn", "torch", "tabpfn")
                },
            },
        }
        teachers = [FeedbackTeacher(c.scaler, 18.2) for c in clients]
        cases = [
            ("feedback", [lambda s, t=t: (int(s[0] * 24 >= 10), t(s[None])[0]) for t in teachers])
        ]
        for name, run in specifications:
            cases.append((name, checkpoint_controllers(run)))
            results.setdefault("baseline_checkpoints", {})[name] = {
                "path": str(run),
                "sha256": file_sha256(run / "checkpoints/best_feasible/heads.pt"),
            }
        for kind in banks:
            for peers in (False, True):
                cases.append(
                    (
                        f"{kind}_{'peer' if peers else 'local'}",
                        EconomicCoordinator(clients, strategy, banks[kind], peers=peers),
                    )
                )
        for index, (name, policy) in enumerate(cases, 1):
            stage(f"evaluate {name} ({index}/{len(cases)})")
            before = time.monotonic()
            record = rollout_controllers(clients, strategy, policy)
            record["evaluation_seconds"] = time.monotonic() - before
            if isinstance(policy, EconomicCoordinator):
                record["solver"] = {
                    "solves": policy.solves,
                    "failures": policy.failures,
                    "seconds": policy.solve_seconds,
                    "median_seconds": float(np.median(policy.solve_times)),
                    "p95_seconds": float(np.quantile(policy.solve_times, 0.95)),
                }
            results["controllers"][name] = record
            if name != "feedback":
                results["comparisons"][name] = comparison(
                    results["controllers"]["feedback"], record
                )
            atomic_json(output / "study_results.json", results)
            homes = [dict(home_id=h, **r) for h, r in zip(HOME_IDS, record["homes"], strict=True)]
            event = {
                "kind": "eval",
                "model": "fedavg",
                "episode": index,
                "split": args.split,
                "policy": name,
                "elapsed_seconds": time.monotonic() - started,
                "evaluation_seconds": record["evaluation_seconds"],
                "days": len(record["dates"]),
                "homes": homes,
                **{k: float(np.mean([h[k] for h in homes])) for k in homes[0] if k != "home_id"},
                "actor_loss": None,
                "critic_loss": None,
            }
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, allow_nan=False) + "\n")
            print(
                f"[eval] {name}: cost={event['elec_cost']:.4f} reward={event['reward']:.4f} comfort={event['comfort_pct']:.3f} peer={event['p2p_kwh']:.3f}",
                flush=True,
            )
        results["seconds"] = time.monotonic() - started
        results["peak_gpu_reserved_gb"] = (
            torch.cuda.max_memory_reserved(args.device) / 1024**3
            if args.device.startswith("cuda")
            else 0.0
        )
        atomic_json(output / "study_results.json", results)
        atomic_json(output / "status.json", {"state": "completed", "pid": os.getpid()})
    except BaseException as error:
        atomic_json(
            output / "status.json", {"state": "failed", "error": str(error), "pid": os.getpid()}
        )
        raise


if __name__ == "__main__":
    main()
