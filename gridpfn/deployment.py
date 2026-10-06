"""Verified local-reference home bundles; no alternate training loop or test scoring."""

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import tomllib

from gridpfn.paths import ROOT


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _reference(path, expected=None):
    path = Path(path).resolve(strict=True)
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Changed model dependency: {path}")
    return {"path": str(path), "sha256": actual}


def ensure_tabpfn_checkpoint():
    """Use the official authorized download path, then verify the study's pin."""
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import download_model, resolve_model_path

    from gridpfn.foundation_backends.tabpfn_backend import CHECKPOINT, CHECKPOINT_SHA256

    paths, _, _, which = resolve_model_path(None, "regressor", version="v3.5")
    if len(paths) != 1 or paths[0].name != CHECKPOINT or which != "regressor":
        raise RuntimeError("The installed TabPFN resolves a different checkpoint than this study")
    path = paths[0]
    if not path.exists():
        message = (
            "Official TabPFN checkpoint download failed. Complete Prior Labs model access "
            "and authentication using the installed TabPFN client's official flow, then retry."
        )
        try:
            result = download_model(
                path, version=ModelVersion.V3_5, which="regressor", model_name=CHECKPOINT
            )
        except Exception:
            raise RuntimeError(message) from None
        if result != "ok" or not path.is_file():
            raise RuntimeError(message)
    if not path.is_file() or _sha(path) != CHECKPOINT_SHA256:
        raise ValueError("Cached TabPFN checkpoint differs from the study's pinned SHA-256")
    return path


def load_model_bundle(directory):
    """Verify a trusted local model and dependencies; never unpickle or evaluate it."""
    directory = Path(directory).resolve()
    path = directory / "home_model.json"
    if _sha(path) != (directory / "model.sha256").read_text().strip():
        raise ValueError("Home model manifest changed")
    bundle = _json(path)
    if bundle.get("schema_version") != 1 or bundle.get("bundle_type") != "local_reference":
        raise ValueError("Unsupported model bundle")
    if bundle["home_id"] not in bundle["cohort_home_ids"]:
        raise ValueError("Model home is absent from its training cohort")
    for receipt in bundle["files"].values():
        _reference(receipt["path"], receipt["sha256"])
    return bundle


def _history_metadata(path, period):
    """Read date coordinates only; never materialize physical observations."""
    import numpy as np

    from gridpfn.core.dataset import period_bounds

    with np.load(path, allow_pickle=False) as saved:
        arrays = [saved[name] for name in ("dates", "train_dates")]
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("Home history dates must be one-dimensional")
        available, training = [list(map(str, values)) for values in arrays]
    for values in (available, training):
        if not values or values != sorted(set(values)):
            raise ValueError("Home history dates must be ordered and unique")
        if any(date.fromisoformat(value).isoformat() != value for value in values):
            raise ValueError("Home history dates must use ISO calendar dates")
    if not set(training).issubset(available):
        raise ValueError("Training dates are absent from available home history")
    training_set = set(training)
    heldout = [value for value in available if value not in training_set]
    bounds = period_bounds(period)
    if (
        not bounds["refit"]
        or not heldout
        or any(
            not bounds["train_start_inclusive"] <= value < bounds["train_end_exclusive"]
            for value in training
        )
        or any(
            not bounds["test_start_inclusive"] <= value < bounds["test_end_exclusive"]
            for value in heldout
        )
        or available != training + heldout
    ):
        raise ValueError("Home history partitions violate the monthly training/test cutoff")
    return {
        "available_dates": available,
        "training_dates": training,
        "evaluation_dates": heldout,
        "evaluation_split": "test",
        "training_end_exclusive": bounds["train_end_exclusive"],
        "evaluation_end_exclusive": bounds["test_end_exclusive"],
        "note": "Calendar coordinates, not scores. Earlier-date replay with this final model uses later training knowledge and is not held-out evaluation.",
    }


def export_home_bundles(run, output):
    """Export a completed fixed-budget refit without reading held-out observations."""
    from gridpfn.core.training_metrics import refit_selection_contract
    from gridpfn.core.utils.agent_utils import safe_torch_load
    from gridpfn.core.utils.run_io import read_logged_settings
    from gridpfn.foundation_backends.tabpfn_backend import CHECKPOINT, CHECKPOINT_SHA256

    run, output = Path(run).resolve(), Path(output).resolve()
    if _json(run / "status.json").get("state") != "completed":
        raise ValueError("Only a completed refit can be exported")
    metadata = _json(run / "run.json")
    settings = metadata["settings"]
    summary = _json(run / "refit_summary.json")
    period = settings["data_period"]
    if not period.endswith("_refit") or not settings.get("refit"):
        raise ValueError("Deployment requires a monthly train+validation refit")
    if summary.get("evaluation_performed") is not False or summary.get("reward") is not None:
        raise ValueError("Refit must have no selection or evaluation reward")
    if summary.get("episode") != settings["episode"] or summary.get("data_period") != period:
        raise ValueError("Refit summary disagrees with the budget or period")
    cohort = settings["home_ids"]
    contract = refit_selection_contract(
        settings["refit_selection"], settings["episode"], period, cohort
    )
    if contract != summary.get("selection_source"):
        raise ValueError("Prior selection changed since refitting")
    checkpoint = run / "checkpoints/latest/heads.pt"
    final_summary = _json(run / "final_summary.json")
    if (
        final_summary.get("evaluation_performed") is not False
        or final_summary.get("latest") != summary
    ):
        raise ValueError("Final receipt disagrees with the unevaluated refit")
    _reference(checkpoint, final_summary["checkpoint_sha256"])
    payload = safe_torch_load(checkpoint, "cpu")
    if (
        payload.get("home_ids") != cohort
        or len(payload.get("clients", [])) != len(cohort)
        or payload.get("episode") != settings["episode"]
        or payload.get("selection_source") != contract
        or payload.get("data_period") != period
        or payload.get("refit") is not True
        or payload.get("reward") is not None
    ):
        raise ValueError("Latest checkpoint does not match the completed refit")
    logged = read_logged_settings(run / "logs/fedavg/train_settings.txt")
    for key in ("home_ids", "data_period", "path_data", "predictive_features"):
        if logged.get(key) != settings.get(key):
            raise ValueError(f"Logged settings disagree: {key}")
    files = {
        "checkpoint": _reference(checkpoint),
        "run": _reference(run / "run.json"),
        "settings": _reference(run / "logs/fedavg/train_settings.txt"),
        "refit_summary": _reference(run / "refit_summary.json"),
        "final_summary": _reference(run / "final_summary.json"),
        "selection": _reference(contract["path"], contract["sha256"]),
        "selection_run": _reference(
            Path(contract["path"]).parent / "run.json", contract["run_sha256"]
        ),
        "data_receipt": _reference(run / "data_hashes.json"),
        "source_receipt": _reference(run / "source_hashes.json"),
    }
    data_hashes = _json(run / "data_hashes.json")
    for home in cohort:
        if f"home_{home}.csv" not in data_hashes:
            raise ValueError("Data receipt does not cover the shared-scaling cohort")
    for name, sha in data_hashes.items():
        files[f"data/{name}"] = _reference(Path(settings["path_data"]) / name, sha)
    source_hashes = _json(run / "source_hashes.json")
    for name, sha in source_hashes.items():
        files[f"source/{name}"] = _reference(Path(metadata["source_dir"]) / name, sha)
    files["weather"] = _reference(
        Path(settings["path_data"]).parent / "temp_price_newyork.csv",
        source_hashes["dataset/temp_price_newyork.csv"],
    )
    feature_root = Path(settings["predictive_features"])
    feature_manifest = _json(feature_root / "manifest.json")
    if feature_manifest["kind"] != "tabpfn":
        raise ValueError("Deployment requires the TabPFN forecasting arm")
    stage = feature_root.parent.parent
    cases = _json(stage / "cases.json")
    files["cases_receipt"] = _reference(stage / "cases.json", feature_manifest["cases_sha256"])
    files["feature_receipt"] = _reference(feature_root / "manifest.json")
    if cases["protocol"] != feature_manifest["protocol"]:
        raise ValueError("Forecast context and feature protocols differ")
    protocol = cases["protocol"]
    if protocol["data_period"] != period or protocol["home_ids"] != cohort or not protocol["refit"]:
        raise ValueError("Forecast context does not belong to this refit")
    prepared = []
    for i, home in enumerate(cohort):
        name = f"home_{home}.npz"
        home_record = next(h for h in cases["homes"] if h["id"] == home)
        training_days = home_record["training_days"]
        context_case = next(
            c for c in cases["cases"] if c["home"] == home and c["first"] == training_days
        )
        home_files = {
            **files,
            "predictive_features": _reference(feature_root / name, feature_manifest["files"][name]),
            "forecast_context": _reference(stage / context_case["path"], context_case["sha256"]),
            "home_history": _reference(stage / home_record["path"], home_record["sha256"]),
            "home_data": files[f"data/home_{home}.csv"],
        }
        history = _history_metadata(home_files["home_history"]["path"], period)
        if (
            len(history["training_dates"]) != training_days
            or context_case["context_dates"] != history["training_dates"]
        ):
            raise ValueError("Home history training dates differ from the forecast context")
        fingerprint = payload["clients"][i].get("predictive_context_sha256")
        if fingerprint != home_files["predictive_features"]["sha256"]:
            raise ValueError("Home forecast features do not match the checkpoint fingerprint")
        manifest = {
            "schema_version": 1,
            "bundle_type": "local_reference",
            "portable": False,
            "home_id": home,
            "cohort_home_ids": cohort,
            "source_run": str(run),
            "run_dir": str(run),
            "checkpoint_path": str(checkpoint),
            "checkpoint": "latest",
            "checkpoint_name": "latest",
            "data_period": period,
            "training_budget": settings["episode"],
            "selection_source": contract,
            "predictive_context_sha256": fingerprint,
            "files": home_files,
            "history": history,
            "forecast": {
                "backend": "tabpfn",
                "model_version": "3.5",
                "model_checkpoint": CHECKPOINT,
                "model_checkpoint_sha256": CHECKPOINT_SHA256,
                "targets": ["load", "pv", "temperature"],
                "fit": "Three independent scalar regressors on identical saved X; y columns 0,1,2",
                "seed": protocol["seed"],
                "context_rows": context_case["rows"],
                "context_limit": protocol["context_rows"],
                "n_estimators": protocol["n_estimators"],
                "horizon": protocol["horizon"],
                "context_dates": context_case["context_dates"],
                "feature_scaling": protocol["feature_scaling"],
                "raw_state_width": 17,
                "auxiliary_width": 32,
                "auxiliary_schema": "8 history/calendar + 18 hourly load/PV/temperature forecasts + 6 validity masks",
            },
            "scope": "Shared FedAvg actor; home-specific forecast context and local critic. Requires original checkout, cohort data and run artifacts. No formal privacy guarantee.",
            "assistant_contract": {
                "home": str(home),
                "run": str(run),
                "checkpoint": "latest",
                "data_period": period,
                "split": "test",
                "note": "Preparing test replay reveals held-out observations: defer until all research models are frozen. Pass data_period and validate separate raw/auxiliary widths. Reuse forecast_context X/y for the trained predictor; a separately fitted app forecaster is a different model.",
            },
        }
        target = output / f"home_{home}"
        if target.exists() and load_model_bundle(target) != manifest:
            raise ValueError(f"Refusing to replace a different home model: {target}")
        prepared.append((target, manifest))
    for target, manifest in prepared:
        if not target.exists():
            target.mkdir(parents=True)
            _write(target / "home_model.json", manifest)
            (target / "model.sha256").write_text(_sha(target / "home_model.json") + "\n")
    return [target for target, _ in prepared]


def train_from_config(config, *, export_only=False):
    """Delegate to the seasonal pipeline for TabPFN's final October fold."""
    with Path(config).expanduser().resolve().open("rb") as handle:
        cfg = tomllib.load(handle)
    if cfg.get("version") != 1:
        raise ValueError("Unsupported training configuration")
    study = cfg["study"]
    if study["method"] != "tabpfn" or study["test_month"] != 10:
        raise ValueError("Use the research CLI for methods or months other than TabPFN/October")
    protocol = _json((ROOT / study["protocol"]).resolve())
    protocol.update(methods=["tabpfn"], test_months=[10])
    if cfg["tariff"] != {
        "mode": "observed_training_fitted_tou",
        "tou_blocks": 5,
        "p2p_enabled": True,
        "dr_limit": 5.0,
        "pv_curtail": 2.5,
        "fixed_cost": 5.0,
        "export_price": 0.025,
        "p2p_price": 0.1,
        "dr_penalty": 0.5,
        "dr_incentive": 0.025,
    }:
        raise ValueError("Deployment retains the original simulator tariffs and fees")
    forbidden = {
        "--grid_prices",
        "--flat_grid_price",
        "--no-tou",
        "--tou",
        "--tou_blocks",
        "--p2p",
        "--no-p2p",
        "--dr_limit",
        "--pv_curtail",
        "--fixed_cost",
        "--export_price",
        "--p2p_price",
        "--dr_penalty",
        "--dr_incentive",
    }
    if any(
        str(option).split("=")[0] in forbidden for option in protocol.get("training_options", [])
    ):
        raise ValueError("The base protocol overrides the declared tariff contract")
    # Freeze actual runtime arguments, rather than relying on parser defaults.
    tariff = cfg["tariff"]
    tariff_options = ["--tou", "--p2p"]
    for name in (
        "tou_blocks",
        "dr_limit",
        "pv_curtail",
        "fixed_cost",
        "export_price",
        "p2p_price",
        "dr_penalty",
        "dr_incentive",
    ):
        tariff_options.extend((f"--{name}", str(tariff[name])))
    protocol["training_options"] = [*protocol.get("training_options", []), *tariff_options]
    output = (ROOT / study["output"]).resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Training output must be under this checkout's results directory")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "personalized_protocol.json"
    if frozen.exists() and _json(frozen) != protocol:
        raise ValueError("Training protocol changed; choose a new output directory")
    if not frozen.exists():
        if export_only:
            raise ValueError("No frozen personalized training protocol exists")
        _write(frozen, protocol)
    if not export_only:
        ensure_tabpfn_checkpoint()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gridpfn.experiments.seasonal_study",
                "--output",
                str(output),
                "--protocol",
                str(frozen),
                "--device",
                study["device"],
                # Unused by a single-method protocol; no alternate model install needed.
                "--tabfm-python",
                sys.executable,
                "--tabicl-python",
                sys.executable,
            ],
            cwd=ROOT,
            check=True,
        )
    return export_home_bundles(output / "folds/2019-10/refit/policies/tabpfn", output / "models")
