"""Solve and independently audit all shared home-days; no RL or GPU is used."""

import hashlib
import importlib.metadata
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gridpfn.core.dataset import load_data, temp_price_path
from gridpfn.core.em_strategy import apply_em_strategy, compose_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.utils.run_io import atomic_json, file_sha256
from gridpfn.experiments.audit_constraints import REFERENCE, reference_module

from .config import ROOT, OracleConfig
from .solver import comfort_frontier, replay_oracle, solve_oracle


def _inputs(config):
    for home in config.home_ids:
        path = config.data_dir / f"home_{home}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing home data: {path}")
    bundles = load_data(
        config.data_dir,
        "*.csv",
        choose=[f"home_{h}" for h in config.home_ids],
        validation_days=14,
        split="validation" if config.split == "train" else config.split,
        data_period=config.data_period,
    )
    clients = []
    for train, heldout, dates, scaler in bundles:
        if scaler["delta_t"] != 1:
            raise ValueError("Oracle scenarios require hourly data")
        arrays = []
        for source in (train, heldout):
            physical = source.copy()
            for column, index in scaler["col_to_scaler_idx"].items():
                if index is not None:
                    low, high = scaler["min"][index], scaler["max"][index]
                    physical[..., column] = source[..., column] * (high - low or 1) + low
            if config.flat_price_per_kwh is not None:
                physical[..., 3] = config.flat_price_per_kwh
            elif config.hourly_prices_per_kwh is not None:
                physical[..., 3] = np.asarray(config.hourly_prices_per_kwh)[
                    physical[..., 2].astype(int)
                ]
            else:
                physical[..., 3] *= config.base_price_multiplier
            arrays.append(physical)
        clients.append(
            SimpleNamespace(
                train_data=arrays[0],
                test_data=arrays[0] if config.split == "train" else arrays[1],
                test_dates=scaler["train_dates"] if config.split == "train" else dates,
            )
        )
    strategy = compose_em_strategy(
        {
            "tou": {"enabled": config.tou_enabled, "n_blocks": config.tou_blocks},
            "dr_limit": config.dr_limit if config.dr_enabled else None,
            "dr_penalty": config.dr_penalty,
            "dr_incentive": config.dr_incentive,
            "pv_curtail": config.export_cap_kwh if config.export_cap_enabled else None,
            "export_price": config.export_price,
        },
        clients,
    )
    return clients, strategy


def build_homes(days, config, strategy):
    """Construct configured simulators from physical 24×8 daily observations."""
    envs = []
    for day in days:
        env = HOME_ENERGY_MGNT(day, fixed_cost=config.fixed_cost, state_dim=17)
        apply_em_strategy(env, strategy)
        for name in (
            "temperature_min",
            "temperature_max",
            "ac_energy_quota",
            "alpha",
            "beta",
            "thermal_inertia",
            "thermal_gain",
            "max_power_AC",
            "max_power_EV",
            "max_power_BESS",
            "max_capacity_EV",
            "max_capacity_BESS",
            "efficiency_EV",
            "efficiency_BESS",
            "time_ini_EV",
            "time_end_EV",
            "time_ini_WM",
            "time_end_WM",
            "wm_duration",
        ):
            setattr(env, name, getattr(config, name))
        env.max_duration = config.wm_duration
        env.reset()
        env.indoor_temp = config.initial_temperature
        env.SoE_BESS = config.initial_battery_soe
        envs.append(env)
    return envs


def _solve_day(job):
    date, days, strategy, config = job[:4]
    envs = build_homes(days, config, strategy)
    time_limit, gap = config.time_limit, config.gap
    frontiers = job[4] if len(job) == 5 else None
    if frontiers is None:
        frontiers = [comfort_frontier(e, time_limit, gap) for e in envs]
    else:
        frontiers = [
            saved if saved is not None else comfort_frontier(env, time_limit, gap)
            for env, saved in zip(envs, frontiers, strict=True)
        ]
    result = {"date": date, "frontiers": frontiers, "oracles": {}}
    for objective in config.objectives:
        solution = solve_oracle(
            envs,
            objective,
            peer_price=config.peer_price,
            peers=config.peer_trading,
            time_limit=time_limit,
            gap=gap,
            frontiers=frontiers,
        )
        current = replay_oracle(envs, solution)
        original = replay_oracle(envs, solution, reference=True)
        for a, b in zip(current["homes"], original["homes"], strict=True):
            for key in (
                "reward",
                "elec_cost",
                "energy_bill_without_dr",
                "import",
                "export",
                "squared_violation",
            ):
                np.testing.assert_allclose(a[key], b[key], atol=1e-5, rtol=1e-7)
        result["oracles"][objective] = {"solution": solution, "audit": original}
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_record(path, protocol):
    """Reject corrupted dates and accidental mixing of different scenarios."""
    record = json.loads(path.read_text())
    receipt = record.pop("receipt_sha256", None)
    if receipt != _digest(record) or record.get("protocol_sha256") != _digest(protocol):
        raise ValueError(f"Oracle receipt differs for {path.name}")
    return record


def summarize(records, home_ids, objectives):
    result = {}
    keys = (
        "reward",
        "elec_cost",
        "energy_bill_without_dr",
        "import",
        "export",
        "p2p_kwh",
        "pv_local_use_kwh",
        "pv_curtailed_kwh",
        "comfort_pct",
        "strict_comfort_pct",
        "squared_violation",
        "ev_completion_ratio",
        "wm_completed",
        "final_battery_soe",
    )
    for objective in objectives:
        homes = []
        for i, home in enumerate(home_ids):
            daily = [row["oracles"][objective]["audit"]["homes"][i] for row in records]
            homes.append(
                {"home_id": home, **{k: float(np.mean([r[k] for r in daily])) for k in keys}}
            )
        certificates = [
            c for row in records for c in row["oracles"][objective]["solution"]["certificates"]
        ]
        result[objective] = {
            "homes": homes,
            "daily_home_metrics": [
                {"date": row["date"], "home_id": home,
                 "objective": -float(row["oracles"][objective]["audit"]["homes"][i]["reward"]),
                 **{key: float(row["oracles"][objective]["audit"]["homes"][i][key])
                    for key in keys}}
                for row in records for i, home in enumerate(home_ids)
            ],
            "bound_scope": (
                "Perfect-future bound on the combined objective only. Bill and comfort "
                "are components of this schedule, not independent lower bounds."
            ),
            "mean": {k: float(np.mean([h[k] for h in homes])) for k in keys},
            "certified": all(c["certified"] for c in certificates),
            "maximum_absolute_solver_gap": max(c["absolute_gap"] for c in certificates),
            "maximum_replay_difference": max(
                row["oracles"][objective]["audit"]["max_trajectory_difference"] for row in records
            ),
            "lower_bound_per_home_day": sum(
                row["oracles"][objective]["solution"]["lower_bound"] for row in records
            )
            / (len(records) * len(home_ids)),
            "upper_bound_per_home_day": sum(
                row["oracles"][objective]["solution"]["upper_bound"] for row in records
            )
            / (len(records) * len(home_ids)),
        }
    result["comfort_limits"] = {
        "maximum_in_band_pct": float(
            np.mean([f["maximum_comfort_pct"] for row in records for f in row["frontiers"]])
        ),
        "minimum_squared_violation": float(
            np.mean([f["minimum_squared_violation"] for row in records for f in row["frontiers"]])
        ),
        "home_days_with_unavoidable_discomfort": sum(
            f["minimum_squared_violation"] > 1e-5 for row in records for f in row["frontiers"]
        ),
    }
    return result


def run(config: OracleConfig, resume=False, frontier_reference=None):
    """Solve a scenario, audit every schedule, save receipts and return its summary."""
    clients, strategy = _inputs(config)
    dates = sorted(set.intersection(*(set(c.test_dates) for c in clients)))
    if config.days:
        dates = dates[: config.days]
    if not dates:
        raise ValueError("No shared complete dates")
    frontier_rows, frontier_receipt = {}, None
    if frontier_reference is not None:
        reference = Path(frontier_reference).resolve()
        saved = json.loads((reference / "oracle.json").read_text())
        if json.loads((reference / "status.json").read_text())["state"] != "completed":
            raise ValueError("Frontier reference must be completed and certified")
        order = [
            saved["home_ids"].index(home) if home in saved["home_ids"] else None
            for home in config.home_ids
        ]
        receipts = {}
        for date in dates:
            if date not in saved["dates"]:
                continue
            path = reference / "days" / f"{date}.json"
            record = read_record(path, saved)
            frontier_rows[date] = [record["frontiers"][i] if i is not None else None for i in order]
            receipts[date] = file_sha256(path)
        frontier_receipt = {
            "manifest_sha256": file_sha256(reference / "oracle.json"),
            "day_sha256": receipts,
        }
    digest = hashlib.sha256()
    for client in clients:
        digest.update(np.ascontiguousarray(client.train_data).tobytes())
        for date in dates:
            digest.update(
                np.ascontiguousarray(client.test_data[client.test_dates.index(date)]).tobytes()
            )
    original_hashes = {
        name: reference_module(name, REFERENCE)[1] for name in ("environment", "em_strategy")
    }
    source_hashes = {
        name: file_sha256(ROOT / name)
        for name in (
            "oracle/solver.py",
            "oracle/ipopt.opt",
            "oracle/runner.py",
            "oracle/config.py",
            "gridpfn/core/environment.py",
            "gridpfn/core/em_strategy.py",
            "gridpfn/core/dataset.py",
            "gridpfn/core/utils/thermal_planning.py",
        )
    }
    protocol = {
        "home_ids": list(config.home_ids),
        "scenario": config.settings(),
        "input_file_sha256": {
            str(config.data_dir / f"home_{h}.csv"): file_sha256(config.data_dir / f"home_{h}.csv")
            for h in config.home_ids
        },
        "price_weather_sha256": file_sha256(temp_price_path),
        "dates": dates,
        "split": config.split,
        "data_period": config.data_period,
        "objectives": list(config.objectives),
        "strategy": strategy,
        "time_limit": config.time_limit,
        "gap": config.gap,
        "reference_commit": REFERENCE,
        "reference_source_sha256": original_hashes,
        "source_sha256": source_hashes,
        "data_sha256": digest.hexdigest(),
        "frontier_reference": frontier_receipt,
        "solver": {
            name: importlib.metadata.version(name) for name in ("pyscipopt", "numpy", "scipy")
        },
        "perfect_foresight": True,
        "training_labels": config.split == "train",
        "discount": 1.0,
        "comfort_definition": "Per-home minimum sum of squared temperature-band violations; EV/WM service and preferred WM window enforced in comfort_first",
        "economic_definition": "Sum of original electrical costs including DR and peer settlement; peer payments cancel within the community",
    }
    manifest = config.output / "oracle.json"
    if resume:
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Resume differs in source, data, solver or oracle protocol")
        for date in dates:
            path = config.output / "days" / f"{date}.json"
            if path.exists():
                read_record(path, protocol)
    else:
        if config.output.exists():
            raise FileExistsError("Choose a fresh output directory or --resume")
        (config.output / "days").mkdir(parents=True)
        atomic_json(manifest, protocol)
    jobs = [
        (
            date,
            [c.test_data[c.test_dates.index(date)] for c in clients],
            strategy,
            config,
            frontier_rows.get(date),
        )
        for date in dates
        if not (config.output / "days" / f"{date}.json").exists()
    ]
    completed = len(dates) - len(jobs)
    started = time.monotonic()
    status = {"state": "running", "completed_dates": completed, "total_dates": len(dates)}
    atomic_json(config.output / "status.json", status)
    failures = []
    try:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            submitted = {executor.submit(_solve_day, job): job[0] for job in jobs}
            for future in as_completed(submitted):
                try:
                    record = future.result()
                except Exception as error:
                    # Retain every successful certificate even if another date fails.
                    failure = {
                        "date": submitted[future], "error": repr(error),
                        "protocol_sha256": _digest(protocol),
                        "seconds": time.monotonic() - started,
                    }
                    failures.append(failure)
                    directory = config.output / "errors"
                    directory.mkdir(exist_ok=True)
                    atomic_json(directory / f"{submitted[future]}-{time.time_ns()}.json", failure)
                    atomic_json(config.output / "status.json", {
                        **status, "failed_dates": [item["date"] for item in failures],
                    })
                    print(f"[oracle] FAILED {submitted[future]}: {error}", flush=True)
                    continue
                record["protocol_sha256"] = _digest(protocol)
                record["receipt_sha256"] = _digest(record)
                atomic_json(config.output / "days" / f"{record['date']}.json", record)
                completed += 1
                status = {
                    **status,
                    "completed_dates": completed,
                    "latest_date": record["date"],
                    "failed_dates": [item["date"] for item in failures],
                    "seconds": time.monotonic() - started,
                }
                atomic_json(config.output / "status.json", status)
                with (config.output / "progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(status) + "\n")
                print(
                    f"[oracle] {completed}/{len(dates)} dates: {record['date']} certified and replayed against main",
                    flush=True,
                )
        if failures:
            raise RuntimeError(
                f"{len(failures)} oracle dates failed; certificates and error receipts retained. "
                "No aggregate is published until every declared date succeeds."
            )
        records = [read_record(config.output / "days" / f"{date}.json", protocol) for date in dates]
        summary = summarize(records, config.home_ids, config.objectives)
        atomic_json(config.output / "summary.json", summary)
        atomic_json(
            config.output / "status.json",
            {**status, "state": "completed", "seconds": time.monotonic() - started},
        )
        return summary
    except BaseException as error:
        atomic_json(
            config.output / "status.json", {**status, "state": "failed", "error": repr(error),
             "failed_dates": [item["date"] for item in failures]}
        )
        raise
