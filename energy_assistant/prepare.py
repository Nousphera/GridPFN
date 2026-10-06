"""Produce auditable, fast-to-serve evidence from real local model execution."""

import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dataset import _construct_dataset, share_training_scale
from demo import generate_data
from em_strategy import compose_em_strategy, make_em_strategy
from forecasting import DirectForecaster, causal_queries, forecast_metrics, physical_series

from .explanations import grouped_shapley
from .simulation import ledger_row, replay

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root=ROOT):
    """Record the implementations used for preparation and live explanations."""
    root = Path(root)
    names = {
        "dataset.py",
        "environment.py",
        "forecasting.py",
        "economic_control.py",
        "em_strategy.py",
        "model.py",
    }
    for directory in ("gridpfn/core", "gridpfn/foundation_backends", "energy_assistant"):
        sources = list((root / directory).rglob("*.py"))
        if not sources:
            raise ValueError(f"Missing source implementations: {directory}")
        names.update(path.relative_to(root).as_posix() for path in sources)
    return {name: digest(root / name) for name in sorted(names)}


def fit_tabpfn(days, context=96):
    """Explicitly select 3.5, retain its context KV cache for repeated queries."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    x, y, _ = causal_queries(days)
    indices = np.random.default_rng(6).choice(len(x), min(context, len(x)), replace=False)
    predictor = DirectForecaster("tabpfn", context=context, estimators=1, device="cpu")
    started = time.monotonic()
    for column in range(3):
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5,
            n_estimators=1,
            device="cpu",
            random_state=6,
            fit_mode="fit_with_cache",
            n_preprocessing_jobs=1,
        )
        predictor.models.append(model.fit(x[indices], y[indices, column]))
    predictor.fitted = True
    predictor.fit_seconds = time.monotonic() - started
    return predictor, x[indices]


def recorded_bill(bundle, strategy, fixed_cost):
    """Account for supplied traces, without attributing shared solar to appliances."""
    _, heldout, dates, scaler = bundle
    tou = strategy.get("tou", {}).get("hourly_prices") or [0] * 24
    physical = heldout.copy()
    for column, index in scaler["col_to_scaler_idx"].items():
        if index is not None:
            low, high = scaler["min"][index], scaler["max"][index]
            physical[..., column] = physical[..., column] * ((high - low) or 1) + low
    days = []
    for date, values in zip(dates, physical, strict=True):
        rows = []
        for hour, row in enumerate(values):
            net = max(0, row[0]) + sum(max(0, row[k]) for k in (5, 6, 7)) - max(0, row[1])
            consumption = {
                name: float(max(0, row[col]))
                for name, col in (("everyday", 0), ("cooling", 5), ("car", 6), ("laundry", 7))
            }
            tariff = float(row[3] + tou[hour])
            rows.append(
                ledger_row(
                    hour,
                    max(0, net),
                    max(0, -net),
                    row[3] + tou[hour],
                    strategy.get("export_price", 0),
                    fixed_cost / 30 / 24,
                )
            )
            rows[-1].update(
                devices_kwh=consumption,
                devices_gross_cost={name: kwh * tariff for name, kwh in consumption.items()},
                solar_kwh=float(max(0, row[1])),
                solar_used_credit=float(min(sum(consumption.values()), max(0, row[1])) * tariff),
            )
        days.append({"date": date, "total": sum(r["net_cost"] for r in rows), "hours": rows})
    return {
        "evidence": "recorded",
        "days": days,
        "period_total": sum(d["total"] for d in days),
        "import_cost": sum(r["import_cost"] for d in days for r in d["hours"]),
        "export_credit": sum(r["export_credit"] for d in days for r in d["hours"]),
        "fixed_cost": sum(r["fixed_cost"] for d in days for r in d["hours"]),
        "note": "Accounting reconstructed from supplied appliance and solar traces, using the declared tariff; not a utility invoice. No battery, export cap, peer trading or demand-response settlement in this recorded-trace accounting. Simulation comparisons use a separate matched baseline.",
    }


def prepare(
    output,
    homes=(101,),
    days=1,
    context=96,
    backend="tabpfn",
    run=None,
    checkpoint="best_feasible",
    split="validation",
    history=False,
):
    import torch

    torch.set_num_threads(3)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Keep private evidence and model inputs inside this repository's results/")
    if (
        backend not in {"tabpfn", "seasonal"}
        or (days is not None and (type(days) is not int or days < 1))
        or not 16 <= context <= 1024
    ):
        raise ValueError("Invalid preparation budget")
    output.mkdir(parents=True, exist_ok=False)
    settings = None
    if run:
        from utils.run_io import read_logged_settings

        run = Path(run).resolve()
        settings = read_logged_settings(run / "logs/fedavg/train_settings.txt")
        data = Path(settings["path_data"]).resolve().parent
        cohort = settings["home_ids"]
        if not set(homes).issubset(cohort):
            raise ValueError("Choose homes present in the checkpoint training cohort")
        hashes = json.loads((run / "data_hashes.json").read_text())
        for home in cohort:
            path = data / "split_homes_clean" / f"home_{home}.csv"
            if hashes.get(path.name) != digest(path):
                raise ValueError("Household inputs have changed since training")
        provenance_path = data / "PROVENANCE.json"
        provenance = (
            json.loads(provenance_path.read_text())
            if provenance_path.exists()
            else {
                "kind": "research",
                "uses_private_data": True,
                "purpose": "Local research input; publication requires data permission.",
            }
        )
    else:
        data = output / "generated"
        generate_data(data)
        cohort = [101, 102, 103]
        if not set(homes).issubset(cohort):
            raise ValueError("Generated demo provides homes 101, 102 and 103")
        provenance = json.loads((data / "PROVENANCE.json").read_text())
    weather = data / "temp_price_newyork.csv"
    bundles = [
        _construct_dataset(
            pd.read_csv(data / "split_homes_clean" / f"home_{home}.csv"),
            temp_price_path=weather,
            validation_days=settings["validation_days"] if settings else 14,
            split=split,
            grid_prices=settings.get("grid_prices") if settings else None,
            data_period=settings.get("data_period", "legacy") if settings else "legacy",
        )
        for home in cohort
    ]
    if settings is None or settings.get("scaler_mode") == "shared":
        bundles = share_training_scale(bundles)
    clients = [SimpleNamespace(train_data=b[0], scaler=b[3]) for b in bundles]
    strategy = compose_em_strategy(
        settings["em_strategy"] if settings else make_em_strategy(), clients
    )
    if settings:
        from model import configure_embedding_device, validate_encoder_context

        configure_embedding_device(torch.device("cpu"), None, settings.get("encoder_context"))
        for home, client in zip(cohort, clients, strict=True):
            client.home_id = home
        validate_encoder_context(clients, strategy, settings.get("p2p_config"))
    fixed = float(settings["fixed_cost"]) if settings else 5.0
    evidence = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "currency": "USD",
        "split": split,
        "replay_scope": "recorded_history" if history else split,
        "data_period": settings.get("data_period", "legacy") if settings else "legacy",
        "execution": "Prepared real model inference and simulator replays; chat retrieves these results without running a new model fit.",
        "forecaster": "TabPFN-3.5" if backend == "tabpfn" else "Seasonal baseline (no TabPFN)",
        "model": {
            "backend": backend,
            "context_rows": context,
            "estimators": 1,
            "device": "cpu",
            "kv_cache": backend == "tabpfn",
            "thinking_mode": False,
            "tabpfn_package": importlib.metadata.version("tabpfn") if backend == "tabpfn" else None,
        },
        "strategy": strategy,
        "homes": {},
        "input_sha256": {
            "weather": digest(weather),
            **{f"home_{h}": digest(data / "split_homes_clean" / f"home_{h}.csv") for h in cohort},
        },
        "source_sha256": source_hashes(),
    }
    for home in homes:
        bundle = bundles[cohort.index(home)]
        if history:
            from .history import recorded_history_bundle

            bundle = recorded_history_bundle(
                bundle, data / "split_homes_clean" / f"home_{home}.csv", weather
            )
        train, heldout, dates, scaler = bundle
        replay_days = len(dates) if days is None else min(days, len(dates))
        train_physical = physical_series(train, scaler)
        heldout_physical = physical_series(heldout, scaler)
        live_input = output / f"home_{home}_context.npz"
        from .trained_forecast import fit_context, load_context

        trained_context = (
            load_context(settings, home, scaler["train_dates"]) if backend == "tabpfn" else None
        )
        extra = {}
        if trained_context is not None:
            extra = {
                "trained_X": trained_context[0],
                "trained_y": trained_context[1],
                "trained_spec": json.dumps(trained_context[2]),
            }
            evidence["model"].update(
                context_rows=trained_context[2]["context_rows"],
                estimators=trained_context[2]["estimators"],
                horizon=trained_context[2]["horizon"],
                context_source="Saved study X/y context",
            )
        np.savez_compressed(
            live_input,
            train=train_physical,
            days=heldout_physical,
            dates=np.asarray(dates),
            **extra,
        )
        print(f"Preparing home {home}: {backend}, {replay_days} replay day(s)", flush=True)
        predictor, background = (
            (
                fit_context(*trained_context)
                if trained_context
                else fit_tabpfn(train_physical, context)
            )
            if backend == "tabpfn"
            else (DirectForecaster("seasonal").fit(train_physical), None)
        )
        seasonal = DirectForecaster("seasonal").fit(train_physical)
        policy = None
        checkpoint_info = None
        if run:
            from .checkpoints import checkpoint_policy

            policy, checkpoint_info = checkpoint_policy(
                run, run / "checkpoints" / checkpoint / "heads.pt", home, bundle
            )
        record = {
            "home": home,
            "train_start": scaler["train_dates"][0],
            "train_end": scaler["train_dates"][-1],
            "training_days": len(train),
            "bill": recorded_bill(bundle, strategy, fixed),
            "dates": {},
            "checkpoint": checkpoint_info,
            "fit_seconds": predictor.fit_seconds,
            "live_context_sha256": digest(live_input),
            "forecast_context": trained_context[2]
            if trained_context
            else {
                "source": "Separate household explanatory fit",
                "horizon": 23,
                "context_rows": context,
            },
        }
        from .history import attach_history

        attach_history(
            record,
            bundle,
            data / "split_homes_clean" / f"home_{home}.csv",
            weather,
            strategy,
            fixed,
        )
        for index in range(len(dates) - replay_days, len(dates)):
            actual = heldout_physical[index : index + 1]
            table = predictor.predict_table(actual)[0]
            seasonal_table = seasonal.predict_table(actual)[0]
            origin, target = 12, 18
            x, _, indices = causal_queries(actual)
            query = x[np.flatnonzero((indices[:, 1] == origin) & (indices[:, 2] == target))[0]]
            explanation = (
                grouped_shapley(
                    lambda rows: np.maximum(0, predictor.models[0].predict(rows)),
                    query,
                    background[:8],
                )
                if backend == "tabpfn"
                else None
            )
            forecasts = {
                "evidence": "predicted",
                "origin_hour": origin,
                "target_hour": target,
                "observed_through": f"{dates[index]} {origin:02}:00",
                "rows": [
                    {
                        "hour": hour,
                        "load_kwh": float(table[origin, hour, 0]),
                        "solar_kwh": float(table[origin, hour, 1]),
                        "outdoor_c": float(table[origin, hour, 2]),
                        "seasonal_load_kwh": float(seasonal_table[origin, hour, 0]),
                    }
                    for hour in range(
                        origin, min(24, origin + getattr(predictor, "horizon", 23) + 1)
                    )
                ],
                "explanation": explanation,
                "retrospective_metrics": {
                    lead: value
                    for lead, value in forecast_metrics(table[None], actual).items()
                    if int(lead) <= getattr(predictor, "horizon", 23)
                },
                "seasonal_metrics": {
                    lead: value
                    for lead, value in forecast_metrics(seasonal_table[None], actual).items()
                    if int(lead) <= getattr(predictor, "horizon", 23)
                },
                "note": "Only observations through the origin enter each forecast. Explanation target is fixed household demand, excluding controlled appliances. Metrics use this replay day only; they are not a general benchmark.",
            }
            if history:
                forecasts["note"] += (
                    " Full-history replay uses the final saved model. Earlier training dates "
                    "are retrospective demonstrations, not held-out forecast evaluation."
                )
            candidates = {
                "feedback": replay(
                    bundle,
                    index,
                    strategy,
                    fixed_cost=fixed,
                    outlook_table=table,
                ),
                "seasonal": replay(
                    bundle,
                    index,
                    strategy,
                    table=seasonal_table,
                    fixed_cost=fixed,
                    outlook_table=table,
                ),
                "tabpfn" if backend == "tabpfn" else "seasonal_duplicate": replay(
                    bundle, index, strategy, table=table, fixed_cost=fixed, outlook_table=table
                ),
            }
            candidates.pop("seasonal_duplicate", None)
            if policy:
                candidates["personalized"] = replay(
                    bundle, index, strategy, policy=policy, fixed_cost=fixed, outlook_table=table
                )
            record["dates"][dates[index]] = {"forecast": forecasts, "scenarios": candidates}
        evidence["homes"][str(home)] = record
    text = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    (output / "evidence.json").write_text(text)
    (output / "evidence.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    print(f"Ready: {output / 'evidence.json'}", flush=True)
    return evidence
