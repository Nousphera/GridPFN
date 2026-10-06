"""Build private forward-only features once for repeated federated PPO episodes."""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from gridpfn.core.dataset import home_data_dir, load_data
from gridpfn.core.forecasting import DirectForecaster, causal_queries, physical_series
from gridpfn.core.predictive_features import PredictiveContext
from gridpfn.core.tabpfn_adaptation import PromptRegressor
from gridpfn.core.training_config import HOME_IDS
from gridpfn.core.utils.run_io import atomic_json


def numeric_features(days, dates, forecasts, widths=None, history=True):
    """Calendar/history plus 3h/6h physical forecasts; terminal features are zero."""
    import datetime

    result = np.zeros((len(days), 25, 17), dtype=np.float32)
    scale = np.array([6.0, 6.0, 10.0])
    for d, (values, date) in enumerate(zip(days, dates, strict=True)):
        weekday = datetime.date.fromisoformat(date).weekday()
        for t in range(24):
            if history:
                average = values[max(0, t - 3) : t + 1, :3].mean(0)
                average[2] -= 20
                result[d, t, :2] = np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)
                result[d, t, 2:5] = (values[t, :3] - values[max(0, t - 1), :3]) / scale
                result[d, t, 5:8] = average / scale
            if forecasts is not None:
                for j, lead in enumerate((3, 6)):
                    prediction = forecasts[d, t, t + 1 : min(24, t + lead + 1), :3]
                    if len(prediction):
                        mean = prediction.mean(0).copy()
                        mean[2] -= 20
                        result[d, t, 8 + j * 3 : 11 + j * 3] = mean / scale
                if widths is not None:
                    interval = widths[d, t, t + 1 : min(24, t + 4), :3]
                    if len(interval):
                        result[d, t, 14:] = interval.mean(0) / scale
    return result


def prediction_table(
    train, heldout, kind, seed, device, context, steps, uncertainty, score_days, max_lead=6
):
    if kind not in ("prompt", "prompt_base"):
        forecaster = DirectForecaster(
            kind, context=context, estimators=1, seed=seed, device=device
        ).fit(train)
        table = forecaster.predict_table(heldout, max_lead=max_lead)
        widths, calibration = None, {}
        if uncertainty and kind == "tabpfn":
            x, y, index = causal_queries(heldout, max_lead=3)
            widths = np.zeros_like(table)
            for j, model in enumerate(forecaster.models):
                batches = []
                for chunk in np.array_split(x, max(1, (len(x) + 255) // 256)):
                    prediction = model.predict(chunk, output_type="quantiles", quantiles=[0.1, 0.9])
                    values = (
                        np.column_stack(prediction)
                        if isinstance(prediction, list)
                        else np.asarray(prediction)
                    )
                    if values.shape != (len(chunk), 2):
                        raise ValueError(f"Unexpected quantile layout: {values.shape}")
                    batches.append(values)
                quantiles = np.concatenate(batches)
                widths[tuple(index.T) + (np.full(len(index), j),)] = (
                    quantiles[:, 1] - quantiles[:, 0]
                )
                scored = index[:, 0] < score_days
                calibration[str(j)] = {
                    "coverage80": float(
                        np.mean(
                            ((y[:, j] >= quantiles[:, 0]) & (y[:, j] <= quantiles[:, 1]))[scored]
                        )
                    ),
                    "mean_width": float(np.mean((quantiles[:, 1] - quantiles[:, 0])[scored])),
                }
        return (
            table,
            widths,
            {
                "fit_seconds": forecaster.fit_seconds,
                "predict_seconds": forecaster.predict_seconds,
                "calibration": calibration,
            },
        )
    x, y, index = causal_queries(train)
    queries, truth, coordinates = causal_queries(heldout, max_lead=max_lead)
    table = np.zeros((len(heldout), 24, 24, 4))
    adapted = []
    for j in range(3):
        model = PromptRegressor(
            device, seed, context=min(context, 64), steps=steps if kind == "prompt" else 0
        ).fit(x, y[:, j], index[:, 0])
        table[tuple(coordinates.T) + (np.full(len(coordinates), j),)] = model.predict(queries)
        adapted.append(model.metrics)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    table[tuple(coordinates.T) + (np.full(len(coordinates), 3),)] = heldout[
        coordinates[:, 0], coordinates[:, 1], 3
    ]
    table[..., :2] = table[..., :2].clip(0)
    return table, None, {"adaptation": adapted}


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    prior = None
    if args.resume:
        import json

        prior = json.loads((output / "preparation.json").read_text())
        for key in ("kind", "seed", "context", "prompt_steps", "max_lead"):
            if prior.get(key, 23 if key == "max_lead" else None) != getattr(args, key):
                raise ValueError(f"Cannot resume changed feature settings: {key}")
        if prior["home_ids"] != args.homes:
            raise ValueError("Cannot resume a different home cohort")
        previous_uncertainty = prior.get(
            "uncertainty", any(b.get("calibration") for h in prior["homes"] for b in h["blocks"])
        )
        if bool(previous_uncertainty) != args.uncertainty:
            raise ValueError("Cannot resume changed uncertainty features")
    bundles = load_data(
        args.path_data,
        "*.csv",
        choose=[f"home_{h}" for h in args.homes],
        validation_days=14,
        split="validation",
        scaler_mode="shared",
    )
    tests = load_data(
        args.path_data,
        "*.csv",
        choose=[f"home_{h}" for h in args.homes],
        validation_days=14,
        split="test",
        scaler_mode="shared",
    )
    # Shared scaling depends on the complete cohort, even for a diagnostic subset.
    if tuple(args.homes) != HOME_IDS:
        raise ValueError("Feature preparation requires the declared full ten-home cohort")
    records = []
    for home, bundle, test in zip(args.homes, bundles, tests, strict=True):
        train, validation, validation_dates, scaler = bundle
        test_days, test_dates = test[1:3]
        physical = physical_series(train, scaler)
        heldout = np.concatenate(
            (physical_series(validation, scaler), physical_series(test_days, scaler))
        )
        dates = scaler["train_dates"] + validation_dates + test_dates
        complete = np.concatenate((physical, heldout))
        if prior is not None and (output / f"home_{home}.npz").exists():
            context = PredictiveContext(output / f"home_{home}.npz", train, scaler)
            if context.dates != dates or context.width != 17:
                raise ValueError("Cannot resume changed held-out dates or feature schema")
            records.append(next(row for row in prior["homes"] if row["home_id"] == home))
            continue
        table = (
            None if args.kind in ("history", "padding") else np.zeros((len(complete), 24, 24, 4))
        )
        widths = np.zeros_like(table) if args.uncertainty and table is not None else None
        blocks = []
        if table is not None:
            # First week uses current-observation persistence; every later
            # training block is predicted from strictly earlier complete dates.
            for d in range(min(7, len(physical))):
                for t in range(23):
                    table[d, t, t + 1 :] = physical[d, t]
            for first in sorted(set((7, 21, 35, len(physical)))):
                if first > len(physical):
                    continue
                last = min(first + 14, len(physical))
                query = physical[first:last] if first < len(physical) else heldout
                if not len(query):
                    continue
                start = time.perf_counter()
                values, intervals, metrics = prediction_table(
                    physical[:first],
                    query,
                    args.kind,
                    args.seed,
                    args.device,
                    args.context,
                    args.prompt_steps,
                    args.uncertainty,
                    len(validation) if first == len(physical) else len(query),
                    args.max_lead,
                )
                destination = (
                    slice(first, last)
                    if first < len(physical)
                    else slice(len(physical), len(complete))
                )
                table[destination] = values
                if widths is not None and intervals is not None:
                    widths[destination] = intervals
                if first == len(physical):
                    # July only determines prediction-quality decisions.
                    _, truth, index = causal_queries(
                        query[: len(validation)], max_lead=args.max_lead
                    )
                    predictions = values[tuple(index.T)][:, :3]
                    metrics["validation_rmse"] = np.sqrt(
                        np.mean((predictions - truth[:, :3]) ** 2, axis=0)
                    ).tolist()
                blocks.append(
                    {
                        "context_dates": scaler["train_dates"][:first],
                        "query_dates": dates[first:last]
                        if first < len(physical)
                        else validation_dates,
                        "seconds": time.perf_counter() - start,
                        **metrics,
                    }
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        features = numeric_features(complete, dates, table, widths, history=args.kind != "padding")
        np.savez_compressed(
            output / f"home_{home}.npz",
            dates=np.asarray(dates),
            values=features,
            train_dates=np.asarray(scaler["train_dates"]),
            train_sha256=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
        )
        records.append({"home_id": home, "blocks": blocks})
        atomic_json(
            output / "preparation.json",
            {
                "kind": args.kind,
                "seed": args.seed,
                "context": args.context,
                "prompt_steps": args.prompt_steps,
                "max_lead": args.max_lead,
                "uncertainty": args.uncertainty,
                "home_ids": args.homes,
                "homes": records,
                "selection": "July only; August prediction scores excluded",
            },
        )
        print(f"[features] home={home} kind={args.kind} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--path_data", type=Path, default=home_data_dir)
    parser.add_argument(
        "--kind",
        choices=(
            "padding",
            "history",
            "persistence",
            "seasonal",
            "trees",
            "tabpfn",
            "prompt_base",
            "prompt",
        ),
        default="tabpfn",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--prompt_steps", type=int, default=12)
    parser.add_argument(
        "--max_lead",
        type=int,
        choices=range(6, 24),
        default=6,
        help="Compute only horizons consumed by the 3h/6h policy features.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed homes with matching settings and data.",
    )
    parser.add_argument("--homes", nargs="+", type=int, default=list(HOME_IDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.2, torch.device(args.device))
    build(args)


if __name__ == "__main__":
    main()
