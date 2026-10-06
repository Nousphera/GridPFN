"""Predict immutable chronological cases inside an isolated model environment."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from gridpfn.paths import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def make_model(kind, cfg, device):
    if kind == "trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=5, random_state=cfg["seed"], n_jobs=1
        )
    from gridpfn.foundation_backends import create_regressor

    return create_regressor(
        kind,
        context_size=cfg["context_rows"],
        n_estimators=cfg["n_estimators"],
        seed=cfg["seed"],
        device=device,
    )


def run(root, kind, device):
    manifest_path = root / "cases.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest["protocol"]
    if kind not in cfg["methods"] or kind in {"history", "persistence"}:
        raise ValueError("A fitted predictor is required")
    out = root / "predictions" / kind
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "gridpfn/foundation_backends" / f"{kind}_backend.py"
    identity = {
        "kind": kind,
        "device": device,
        "cases_sha256": digest(manifest_path),
        "worker_sha256": digest(__file__),
        "adapter_sha256": digest(source) if source.exists() else None,
    }
    contract = out / "contract.json"
    if contract.exists() and json.loads(contract.read_text()) != identity:
        raise ValueError("Worker identity changed; use a fresh study output")
    save_json(contract, identity)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        torch = None
    for case in manifest["cases"]:
        case_path = root / case["path"]
        if digest(case_path) != case["sha256"]:
            raise ValueError("Immutable context/query case changed")
        name = case["id"]
        result_path, receipt_path = out / f"{name}.npz", out / f"{name}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if digest(result_path) != receipt["prediction_sha256"]:
                raise ValueError("Prediction archive changed")
            continue
        with np.load(case_path, allow_pickle=False) as z:
            X, y, query = z["X"], z["y"], z["query"]
        columns, timings, models, checks = [], [], [], []
        for j in range(3):
            model = make_model(kind, cfg, device)
            start = time.perf_counter()
            model.fit(X, y[:, j])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prediction = np.concatenate(
                [
                    model.predict(chunk)
                    for chunk in np.array_split(query, max(1, (len(query) + 255) // 256))
                ]
            )
            prediction_seconds = time.perf_counter() - start
            if prediction.shape != (len(query),) or not np.isfinite(prediction).all():
                raise ValueError("Model did not return finite scalar forecasts")
            if case == manifest["cases"][0]:
                prefix = query[:8]
                alone = model.predict(prefix)
                altered = np.concatenate((prefix, query[-8:] * 100 + 1000))
                together = model.predict(altered)[: len(prefix)]
                error = float(np.max(np.abs(alone - together)))
                if not np.allclose(alone, together, atol=1e-4, rtol=1e-4):
                    raise ValueError(f"Future query features affect earlier forecasts: {error}")
                checks.append(
                    {"target": j, "max_prefix_difference": error, "atol": 1e-4, "rtol": 1e-4}
                )
            columns.append(prediction)
            timings.append(
                {"target": j, "fit_seconds": fit_seconds, "predict_seconds": prediction_seconds}
            )
            models.append(getattr(model, "metadata_", {"kind": kind, "trees": 128}))
            del model
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        prediction = np.column_stack(columns)
        prediction[:, :2] = prediction[:, :2].clip(0)
        np.savez_compressed(result_path, prediction=prediction)
        save_json(
            receipt_path,
            {
                **identity,
                "case_sha256": case["sha256"],
                "prediction_sha256": digest(result_path),
                "timing": timings,
                "models": models,
                "batch_causality": checks,
            },
        )
        print(f"{kind}: {name} completed ({len(query)} queries)", flush=True)
    save_json(
        out / "status.json", {"state": "completed", "cases": len(manifest["cases"]), **identity}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--kind", required=True, choices=["trees", "tabpfn", "tabfm", "tabicl"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.study.resolve(), args.kind, args.device)


if __name__ == "__main__":
    main()
