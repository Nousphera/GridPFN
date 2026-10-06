"""One-seed matched foundation-model forecasting and home-control prototype."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from gridpfn.experiments.foundation_worker import digest, save_json
from gridpfn.paths import ROOT


def verify_inputs(root):
    manifest = json.loads((root / "cases.json").read_text())
    for row in manifest["homes"] + manifest["cases"]:
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError("Immutable study inputs changed")
    for name, expected in manifest.get("input_sources", {}).items():
        if digest(ROOT / name) != expected:
            raise ValueError("Original source inputs changed")
    return manifest


def verify_predictions(root, kind, manifest):
    output = root / "predictions" / kind
    contract = json.loads((output / "contract.json").read_text())
    adapter = ROOT / "gridpfn/foundation_backends" / f"{kind}_backend.py"
    expected = {
        "kind": kind,
        "cases_sha256": digest(root / "cases.json"),
        "worker_sha256": digest(ROOT / "gridpfn/experiments/foundation_worker.py"),
        "adapter_sha256": digest(adapter) if adapter.exists() else None,
    }
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Prediction worker identity differs from this study")
    status = json.loads((output / "status.json").read_text())
    if status.get("state") != "completed" or status.get("cases") != len(manifest["cases"]):
        raise ValueError("Prediction preparation is incomplete")
    for case in manifest["cases"]:
        path = output / f"{case['id']}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if any(receipt.get(k) != v for k, v in contract.items()):
            raise ValueError("Forecast receipt belongs to a different worker")
        if receipt["case_sha256"] != case["sha256"] or digest(path) != receipt["prediction_sha256"]:
            raise ValueError("Forecast provenance mismatch")


def trajectory_features(physical, dates, table, history_only=False, normalization=None):
    """Observed history + six hourly predictions + masks; invalid/terminal slots zero."""
    from datetime import date

    result = np.zeros((len(physical), 25, 32), dtype=np.float32)
    center = (
        np.array([0.0, 0.0, 20.0]) if normalization is None else np.asarray(normalization["mean"])
    )
    units = (
        np.array([6.0, 6.0, 10.0]) if normalization is None else np.asarray(normalization["std"])
    )
    if (
        center.shape != (3,)
        or units.shape != (3,)
        or not np.isfinite(units).all()
        or np.any(units <= 0)
    ):
        raise ValueError("Invalid forecast normalization")
    for day, (values, when) in enumerate(zip(physical, dates, strict=True)):
        weekday = date.fromisoformat(when).weekday()
        for t in range(24):
            average = values[max(0, t - 3) : t + 1, :3].mean(0).copy()
            average -= center
            result[day, t, :2] = np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)
            result[day, t, 2:5] = (values[t, :3] - values[max(0, t - 1), :3]) / units
            result[day, t, 5:8] = average / units
            for lead in range(1, 7):
                if t + lead >= 24:
                    continue
                result[day, t, 26 + lead - 1] = 1
                if not history_only:
                    predicted = table[day, t, t + lead].copy()
                    predicted -= center
                    result[day, t, 8 + (lead - 1) * 3 : 8 + lead * 3] = predicted / units
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite control inputs")
    return result


def training_normalization(arrays):
    rows = np.concatenate([np.asarray(x)[..., :3].reshape(-1, 3) for x in arrays])
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite training observations")
    return {
        "mean": rows.mean(axis=0).tolist(),
        "std": rows.std(axis=0).clip(1e-6).tolist(),
        "training_rows": len(rows),
        "training_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
        "scope": "Shared statistics from training observations only; no validation rows",
    }


def prepare(root, protocol):
    from gridpfn.core.dataset import home_data_dir, load_data
    from gridpfn.core.forecasting import causal_queries, physical_series

    cfg = json.loads(protocol.read_text())
    if cfg["horizon"] != 6 or len(cfg["home_ids"]) < 1:
        raise ValueError("This prototype requires the six-hour feature schema")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "cases.json").exists():
        old = verify_inputs(root)
        if old["protocol"] != cfg:
            raise ValueError("Protocol changed; choose a fresh output")
        for row in old["cases"]:
            if digest(root / row["path"]) != row["sha256"]:
                raise ValueError("Stored case changed")
        return
    bundles = load_data(
        home_data_dir,
        "*.csv",
        choose=[f"home_{h}" for h in cfg["home_ids"]],
        validation_days=14,
        split="validation",
        scaler_mode="shared",
        data_period=cfg.get("data_period", "legacy"),
    )
    test_bundles = None
    if cfg.get("data_period") == "full":
        test_bundles = load_data(
            home_data_dir,
            "*.csv",
            choose=[f"home_{h}" for h in cfg["home_ids"]],
            validation_days=14,
            split="test",
            scaler_mode="shared",
            data_period="full",
        )
    cases, homes = [], []
    (root / "cases").mkdir(exist_ok=True)
    for home, (train, valid, vdates, scaler) in zip(cfg["home_ids"], bundles, strict=True):
        dates = scaler["train_dates"] + vdates
        physical = np.concatenate((physical_series(train, scaler), physical_series(valid, scaler)))
        if test_bundles is not None:
            test_train, test, tdates, test_scaler = test_bundles[cfg["home_ids"].index(home)]
            if not np.array_equal(test_train, train):
                raise ValueError("Test and validation use different training inputs")
            physical = np.concatenate((physical, physical_series(test, test_scaler)))
            dates += tdates
        home_path = root / f"home_{home}.npz"
        np.savez_compressed(
            home_path,
            physical=physical,
            dates=dates,
            train_dates=scaler["train_dates"],
            train_sha256=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
        )
        homes.append(
            {
                "id": home,
                "path": home_path.name,
                "sha256": digest(home_path),
                "training_days": len(train),
                "validation_days": len(valid),
            }
        )
        blocks = cfg["blocks"]
        if cfg.get("data_period") == "full":
            blocks = [*blocks, len(train)]
        for block_index, first in enumerate(blocks):
            if first > len(train):
                raise ValueError("Forward split exceeds training dates")
            last = min(first + 14, len(train)) if first < len(train) else len(physical)
            if cfg.get("data_period") == "full":
                last = blocks[block_index + 1] if block_index + 1 < len(blocks) else len(physical)
            if dates[first - 1] >= dates[first]:
                raise ValueError("Nonchronological context")
            X, y, _ = causal_queries(physical[:first], max_lead=cfg["horizon"])
            query, _, indices = causal_queries(physical[first:last], max_lead=cfg["horizon"])
            chosen = np.random.default_rng(cfg["seed"]).choice(
                len(X), min(cfg["context_rows"], len(X)), replace=False
            )
            path = root / "cases" / f"home{home}_from{first}.npz"
            np.savez_compressed(
                path,
                X=X[chosen],
                y=y[chosen, :3],
                query=query,
                indices=indices,
                selected_context_indices=chosen,
            )
            cases.append(
                {
                    "id": path.stem,
                    "home": home,
                    "first": first,
                    "last": last,
                    "path": str(path.relative_to(root)),
                    "sha256": digest(path),
                    "context_dates": dates[:first],
                    "query_dates": dates[first:last],
                    "rows": len(chosen),
                    "queries": len(query),
                }
            )
    normalization = None
    if cfg.get("feature_scaling") == "shared_training_mean_std":
        normalization = training_normalization([physical_series(b[0], b[3]) for b in bundles])
    save_json(
        root / "cases.json",
        {
            "protocol": cfg,
            "normalization": normalization,
            "homes": homes,
            "cases": cases,
            "input_sources": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    *[home_data_dir / f"home_{h}.csv" for h in cfg["home_ids"]],
                    ROOT / "dataset/temp_price_newyork.csv",
                ]
            },
        },
    )
    print(
        f"Prepared {len(cases)} immutable chronological cases; period={cfg.get('data_period', 'legacy')}"
    )


def features(root, kind):
    manifest = verify_inputs(root)
    cfg = manifest["protocol"]
    if kind not in cfg["methods"]:
        raise ValueError("Unknown study method")
    output = root / "features" / kind
    output.mkdir(parents=True, exist_ok=True)
    fitted = kind not in {"history", "persistence"}
    if fitted:
        verify_predictions(root, kind, manifest)
    if (output / "manifest.json").exists():
        verify_features(root, kind)
        return
    forecasts, provenance = [], []
    for home in manifest["homes"]:
        path = root / home["path"]
        if digest(path) != home["sha256"]:
            raise ValueError("Home evaluation inputs changed")
        with np.load(path, allow_pickle=False) as z:
            physical, dates = z["physical"], z["dates"].tolist()
            train_dates, train_hash = z["train_dates"], str(z["train_sha256"])
        table = np.zeros((len(physical), 24, 24, 3), dtype=np.float64)
        for d in range(len(physical)):
            for t in range(23):
                table[d, t, t + 1 :] = physical[d, t, :3]
        for case in (c for c in manifest["cases"] if c["home"] == home["id"]):
            with np.load(root / case["path"], allow_pickle=False) as z:
                index = z["indices"]
            if fitted:
                result = root / "predictions" / kind / f"{case['id']}.npz"
                receipt = json.loads(result.with_suffix(".json").read_text())
                if (
                    receipt["case_sha256"] != case["sha256"]
                    or digest(result) != receipt["prediction_sha256"]
                ):
                    raise ValueError("Forecast provenance mismatch")
                with np.load(result, allow_pickle=False) as z:
                    prediction = z["prediction"]
                table[index[:, 0] + case["first"], index[:, 1], index[:, 2]] = prediction
                provenance.append(receipt)
            if case["first"] == home["training_days"] and kind != "history" and not cfg.get("refit"):
                truth = physical[index[:, 0] + case["first"], index[:, 2], :3]
                pred = table[index[:, 0] + case["first"], index[:, 1], index[:, 2]]
                if cfg.get("data_period") == "full":
                    validation_rows = index[:, 0] < home["validation_days"]
                    truth, pred = truth[validation_rows], pred[validation_rows]
                forecasts.append(
                    {
                        "home": home["id"],
                        "rmse": np.sqrt(np.mean((pred - truth) ** 2, axis=0)).tolist(),
                        "mae": np.mean(abs(pred - truth), axis=0).tolist(),
                        "query_count": len(truth),
                    }
                )
        values = trajectory_features(
            physical,
            dates,
            table,
            history_only=kind == "history",
            normalization=manifest.get("normalization"),
        )
        np.savez_compressed(
            output / f"home_{home['id']}.npz",
            values=values,
            dates=dates,
            train_dates=train_dates,
            train_sha256=train_hash,
        )
    save_json(
        output / "manifest.json",
        {
            "kind": kind,
            "protocol": cfg,
            "cases_sha256": digest(root / "cases.json"),
            "forecast_validation": forecasts,
            "predictors": provenance,
            "auxiliary_width": 32,
            "normalization": manifest.get("normalization"),
            "feature_builder_sha256": digest(__file__),
            "files": {p.name: digest(p) for p in output.glob("home_*.npz")},
        },
    )
    print(f"Built {kind} trajectory features")


def verify_features(root, kind):
    manifest = verify_inputs(root)
    cfg = manifest["protocol"]
    feature_root = root / "features" / kind
    receipt = json.loads((feature_root / "manifest.json").read_text())
    if (
        receipt["kind"] != kind
        or receipt["protocol"] != cfg
        or receipt["cases_sha256"] != digest(root / "cases.json")
    ):
        raise ValueError("Feature manifest belongs to a different study or method")
    for name, sha in receipt["files"].items():
        if digest(feature_root / name) != sha:
            raise ValueError("Controller inputs changed")
    if kind not in {"history", "persistence"}:
        verify_predictions(root, kind, manifest)
    return cfg


def train(root, kind):
    cfg = verify_features(root, kind)
    feature_root = root / "features" / kind
    out = root / "policies" / kind
    episodes = cfg.get("refit_budgets", {}).get(kind, cfg["episodes"])
    if (out / "status.json").exists():
        if json.loads((out / "status.json").read_text())["state"] == "completed":
            actual = json.loads((out / "run.json").read_text())["settings"]
            wanted = {
                "seed": cfg["seed"],
                "episode": episodes,
                "home_ids": cfg["home_ids"],
                "head_width": cfg["actor_width"],
                "value_width": cfg["value_width"],
                "feature_mode": "raw",
                "predictive_features": str(feature_root),
            }
            if any(actual.get(k) != v for k, v in wanted.items()):
                raise ValueError("Completed policy belongs to a different training contract")
            return
        raise ValueError("Incomplete policy run preserved for inspection")
    command = [
        sys.executable,
        "-m", "gridpfn.experiments.run_experiment",
        "--preset",
        "ppo",
        "--path_train",
        str(out),
        "--seed",
        str(cfg["seed"]),
        "--fixed_seed",
        str(cfg["seed"] * 100 + 1),
        "--home_ids",
        *map(str, cfg["home_ids"]),
        "--episode",
        str(episodes),
        "--feature_mode",
        "raw",
        "--embedding_weight",
        "1",
        "--head_width",
        str(cfg["actor_width"]),
        "--value_width",
        str(cfg["value_width"]),
        "--predictive_features",
        str(feature_root),
        "--validation_only",
        "--no-strict_convergence",
        "--patience",
        "0",
        "--cpu_threads",
        "1",
        "--gpu",
        "0",
    ]
    if cfg.get("data_period"):
        command += ["--data_period", cfg["data_period"]]
    command += cfg.get("training_options", [])
    if cfg.get("refit"):
        command += ["--refit_selection", str(Path(cfg["refit_selection_root"]) / kind / "selection.json")]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with (root / f"policy_{kind}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {kind} policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "features", "train"])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/foundation_prototype_20261006"
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/foundation_prototype.json")
    parser.add_argument(
        "--kind", choices=["history", "persistence", "trees", "tabpfn", "tabfm", "tabicl"]
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve(), args.protocol)
    elif args.kind is None:
        parser.error("--kind is required")
    else:
        globals()[args.action](args.output.resolve(), args.kind)


if __name__ == "__main__":
    main()
