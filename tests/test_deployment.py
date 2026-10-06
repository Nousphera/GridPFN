"""Local model provenance and delegation, without training or held-out scoring."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gridpfn import deployment as dep
from gridpfn.core.training_metrics import refit_selection_contract


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def refit(tmp_path):
    stage = tmp_path / "refit"
    run = stage / "policies/tabpfn"
    selection = tmp_path / "select/selection.json"
    write(selection, {"best": {"episode": 0, "reward": -1.0}})
    write(selection.parent / "status.json", {"state": "completed"})
    write(
        selection.parent / "run.json",
        {"settings": {"data_period": "month_10_select", "home_ids": [27]}},
    )
    contract = refit_selection_contract(selection, 0, "month_10_refit", [27])
    data = tmp_path / "dataset/split_homes_clean"
    write(data / "home_27.csv", "fixture,not real observations")
    write(data.parent / "temp_price_newyork.csv", "fixture weather")
    source = run / "source"
    weather = source / "dataset/temp_price_newyork.csv"
    write(weather, "fixture weather")
    feature_root = stage / "features/tabpfn"
    write(feature_root / "home_27.npz", "fixture features")
    write(stage / "cases/home27_from153.npz", "fixture context")
    available = np.arange("2019-05-01", "2019-11-01", dtype="datetime64[D]").astype(str)
    training = available[available < "2019-10-01"]
    # Loading this object-valued entry would fail with allow_pickle=False:
    # the exporter must read only dates and never physical outcomes.
    np.savez(
        stage / "home_27.npz",
        dates=available,
        train_dates=training,
        physical=np.array([object()], dtype=object),
    )
    protocol = {
        "data_period": "month_10_refit",
        "home_ids": [27],
        "refit": True,
        "seed": 41,
        "context_rows": 1024,
        "n_estimators": 1,
        "horizon": 6,
        "feature_scaling": "fixed_physical_divisors",
    }
    write(
        stage / "cases.json",
        {
            "protocol": protocol,
            "homes": [
                {
                    "id": 27,
                    "training_days": 153,
                    "path": "home_27.npz",
                    "sha256": dep._sha(stage / "home_27.npz"),
                }
            ],
            "cases": [
                {
                    "home": 27,
                    "first": 153,
                    "rows": 1024,
                    "context_dates": training.tolist(),
                    "path": "cases/home27_from153.npz",
                    "sha256": dep._sha(stage / "cases/home27_from153.npz"),
                }
            ],
        },
    )
    write(
        feature_root / "manifest.json",
        {
            "kind": "tabpfn",
            "protocol": protocol,
            "cases_sha256": dep._sha(stage / "cases.json"),
            "files": {"home_27.npz": dep._sha(feature_root / "home_27.npz")},
        },
    )
    settings = {
        "data_period": "month_10_refit",
        "refit": True,
        "home_ids": [27],
        "episode": 0,
        "refit_selection": str(selection),
        "path_data": str(data),
        "predictive_features": str(feature_root),
    }
    write(run / "run.json", {"settings": settings, "source_dir": str(source)})
    write(run / "status.json", {"state": "completed"})
    write(run / "data_hashes.json", {"home_27.csv": dep._sha(data / "home_27.csv")})
    write(run / "source_hashes.json", {"dataset/temp_price_newyork.csv": dep._sha(weather)})
    write(
        run / "refit_summary.json",
        {
            "episode": 0,
            "data_period": "month_10_refit",
            "reward": None,
            "selection_source": contract,
            "evaluation_performed": False,
        },
    )
    logs = run / "logs/fedavg/train_settings.txt"
    logs.parent.mkdir(parents=True)
    logs.write_text("\n".join(f"{k} = {v!r}" for k, v in settings.items()))
    checkpoint = run / "checkpoints/latest/heads.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "home_ids": [27],
            "clients": [{"predictive_context_sha256": dep._sha(feature_root / "home_27.npz")}],
            "episode": 0,
            "data_period": "month_10_refit",
            "refit": True,
            "selection_source": contract,
            "reward": None,
        },
        checkpoint,
    )
    write(
        run / "final_summary.json",
        {
            "evaluation_performed": False,
            "latest": dep._json(run / "refit_summary.json"),
            "checkpoint_sha256": dep._sha(checkpoint),
        },
    )
    return run


def test_export_and_verify_bc_only_model_with_exact_forecast_context(refit, tmp_path):
    [directory] = dep.export_home_bundles(refit, tmp_path / "models")
    manifest = dep.load_model_bundle(directory)
    assert manifest["portable"] is False
    assert manifest["home_id"] == 27 and manifest["training_budget"] == 0
    assert manifest["checkpoint"] == "latest" and manifest["run_dir"] == str(refit)
    assert manifest["forecast"]["seed"] == 41
    assert manifest["forecast"]["auxiliary_width"] == 32
    history = manifest["history"]
    assert len(history["available_dates"]) == 184
    assert len(history["training_dates"]) == 153
    assert history["evaluation_dates"] == [f"2019-10-{day:02d}" for day in range(1, 32)]
    assert set(history["training_dates"]).isdisjoint(history["evaluation_dates"])
    assert Path(manifest["files"]["home_data"]["path"]).name == "home_27.csv"
    assert Path(manifest["files"]["forecast_context"]["path"]).name == "home27_from153.npz"
    assert dep.export_home_bundles(refit, tmp_path / "models") == [directory]
    # Existing application descriptor interface resolves the same home and heads.
    from hems_assistant import resolve_model

    run, home, checkpoint, _ = resolve_model(directory)
    assert (run, home, checkpoint) == (refit, 27, "latest")


@pytest.mark.parametrize(
    "artifact", ["checkpoint", "weather", "predictive_features", "forecast_context", "selection"]
)
def test_bundle_detects_dependency_changes(refit, tmp_path, artifact):
    [directory] = dep.export_home_bundles(refit, tmp_path / "models")
    path = Path(dep.load_model_bundle(directory)["files"][artifact]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed model dependency"):
        dep.load_model_bundle(directory)


def test_bundle_detects_manifest_change(refit, tmp_path):
    [directory] = dep.export_home_bundles(refit, tmp_path / "models")
    with (directory / "home_model.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="manifest changed"):
        dep.load_model_bundle(directory)


@pytest.mark.parametrize(
    "change",
    [
        "unordered",
        "duplicate",
        "overlap",
        "past_heldout",
        "future_heldout",
        "missing_from_available",
        "invalid_iso",
    ],
)
def test_export_rejects_tampered_calendar_partition(refit, tmp_path, change):
    stage = refit.parent.parent
    source = stage / "home_27.npz"
    with np.load(source, allow_pickle=False) as saved:
        available, training = saved["dates"].tolist(), saved["train_dates"].tolist()
    if change == "unordered":
        available[0], available[1] = available[1], available[0]
    elif change == "duplicate":
        available.append(available[-1])
    elif change == "overlap":
        training.append("2019-10-01")
    elif change == "past_heldout":
        training.pop(-1)
    elif change == "future_heldout":
        available.append("2019-11-01")
    elif change == "missing_from_available":
        available.pop(0)
    else:
        available[0] = training[0] = "2019-05-00"
    np.savez(source, dates=available, train_dates=training)
    cases = dep._json(stage / "cases.json")
    cases["homes"][0]["sha256"] = dep._sha(source)
    write(stage / "cases.json", cases)
    receipt_path = stage / "features/tabpfn/manifest.json"
    receipt = dep._json(receipt_path)
    receipt["cases_sha256"] = dep._sha(stage / "cases.json")
    write(receipt_path, receipt)
    with pytest.raises(ValueError):
        dep.export_home_bundles(refit, tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_export_rejects_evaluated_refit(refit, tmp_path):
    summary = dep._json(refit / "refit_summary.json")
    summary["reward"] = -1.0
    write(refit / "refit_summary.json", summary)
    with pytest.raises(ValueError, match="no selection or evaluation reward"):
        dep.export_home_bundles(refit, tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_export_rejects_checkpoint_changed_since_run_completion(refit, tmp_path):
    checkpoint = refit / "checkpoints/latest/heads.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed model dependency"):
        dep.export_home_bundles(refit, tmp_path / "models")


def test_export_rejects_features_not_bound_to_checkpoint(refit, tmp_path):
    checkpoint = refit / "checkpoints/latest/heads.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["clients"][0]["predictive_context_sha256"] = "wrong"
    torch.save(payload, checkpoint)
    summary = dep._json(refit / "final_summary.json")
    summary["checkpoint_sha256"] = dep._sha(checkpoint)
    write(refit / "final_summary.json", summary)
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        dep.export_home_bundles(refit, tmp_path / "models")


def test_launcher_delegates_single_tabpfn_october_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(dep, "ROOT", tmp_path)
    original = Path(__file__).parents[1] / "configs/gridpfn.toml"
    config = tmp_path / "gridpfn.toml"
    config.write_text(original.read_text())
    write(
        tmp_path / "configs/seasonal.json",
        {"methods": ["tabfm", "tabpfn"], "test_months": [6, 7, 8, 9, 10]},
    )
    calls = []
    checkpoint_checks = []
    monkeypatch.setattr(dep, "ensure_tabpfn_checkpoint", lambda: checkpoint_checks.append(True))
    monkeypatch.setattr(dep.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(dep, "export_home_bundles", lambda run, output: [output / "home_27"])
    assert dep.train_from_config(config) == [tmp_path / "results/personalized/models/home_27"]
    protocol = dep._json(tmp_path / "results/personalized/personalized_protocol.json")
    assert protocol["methods"] == ["tabpfn"] and protocol["test_months"] == [10]
    # Exercise the actual runtime parser on the frozen options, not a duplicated
    # expected command string. All economic options must be explicitly present.
    from gridpfn.core.training_config import parse_args

    options = protocol["training_options"]
    parsed = parse_args(["--preset", "ppo", *options])
    for name, value in {
        "tou_blocks": 5,
        "dr_limit": 5.0,
        "pv_curtail": 2.5,
        "fixed_cost": 5.0,
        "export_price": 0.025,
        "p2p_price": 0.1,
        "dr_penalty": 0.5,
        "dr_incentive": 0.025,
    }.items():
        assert f"--{name}" in options
        assert getattr(parsed, name) == value
    assert "--tou" in options and "--p2p" in options
    assert parsed.tou and parsed.p2p and parsed.grid_prices is None
    assert "gridpfn.experiments.seasonal_study" in calls[0][0][0]
    assert calls[0][1]["check"] is True
    dep.train_from_config(config, export_only=True)
    assert len(calls) == 1
    assert checkpoint_checks == [True]
    write(tmp_path / "configs/seasonal.json", {"seed": 9})
    with pytest.raises(ValueError, match="protocol changed"):
        dep.train_from_config(config)


@pytest.mark.parametrize(
    "override",
    [
        "--flat_grid_price=99",
        "--no-p2p",
        "--tou_blocks=2",
        "--dr_limit=99",
        "--pv_curtail=99",
    ],
)
def test_launcher_rejects_tariff_override(tmp_path, monkeypatch, override):
    monkeypatch.setattr(dep, "ROOT", tmp_path)
    config = Path(__file__).parents[1] / "configs/gridpfn.toml"
    write(tmp_path / "configs/seasonal.json", {"training_options": [override]})
    with pytest.raises(ValueError, match="tariff contract"):
        dep.train_from_config(config)


def test_launcher_rejects_changed_tariff_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(dep, "ROOT", tmp_path)
    original = Path(__file__).parents[1] / "configs/gridpfn.toml"
    config = tmp_path / "modified.toml"
    config.write_text(original.read_text().replace("p2p_enabled = true", "p2p_enabled = false"))
    write(tmp_path / "configs/seasonal.json", {})
    with pytest.raises(ValueError, match="original simulator tariffs"):
        dep.train_from_config(config)
    assert not (tmp_path / "results").exists()


@pytest.fixture
def checkpoint_cache(tmp_path, monkeypatch):
    import tabpfn.model_loading as loading

    from gridpfn.foundation_backends import tabpfn_backend

    path = tmp_path / tabpfn_backend.CHECKPOINT
    content = b"mock official model bytes"
    import hashlib

    monkeypatch.setattr(tabpfn_backend, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(
        loading, "resolve_model_path", lambda *a, **kw: ([path], [], [], "regressor")
    )
    return loading, path, content


def test_cached_checkpoint_is_verified_without_download(checkpoint_cache, monkeypatch):
    loading, path, content = checkpoint_cache
    path.write_bytes(content)
    monkeypatch.setattr(
        loading, "download_model", lambda *a, **kw: pytest.fail("Unexpected download")
    )
    assert dep.ensure_tabpfn_checkpoint() == path


def test_missing_checkpoint_uses_official_pinned_download(checkpoint_cache, monkeypatch):
    loading, path, content = checkpoint_cache
    from tabpfn.constants import ModelVersion

    def download(to, **kwargs):
        assert to == path
        assert kwargs == {
            "version": ModelVersion.V3_5,
            "which": "regressor",
            "model_name": path.name,
        }
        to.write_bytes(content)
        return "ok"

    monkeypatch.setattr(loading, "download_model", download)
    assert dep.ensure_tabpfn_checkpoint() == path


@pytest.mark.parametrize(
    "failure",
    [
        "tampered_cache",
        "tampered_download",
        "returned_error",
        "raised_error",
        "missing_file",
        "wrong_name",
    ],
)
def test_checkpoint_failures_are_clear_and_do_not_expose_credentials(
    checkpoint_cache, monkeypatch, failure
):
    loading, path, content = checkpoint_cache
    if failure == "tampered_cache":
        path.write_bytes(b"wrong")
    if failure == "wrong_name":
        monkeypatch.setattr(
            loading,
            "resolve_model_path",
            lambda *a, **kw: ([path.with_name("wrong.pt")], [], [], "regressor"),
        )

    def download(to, **kwargs):
        if failure in {"tampered_cache", "wrong_name"}:
            pytest.fail("Must not download over an existing or incorrectly resolved model")
        if failure == "raised_error":
            raise RuntimeError("fake-secret-must-not-appear")
        if failure == "returned_error":
            return [RuntimeError("fake-secret-must-not-appear")]
        if failure == "tampered_download":
            to.write_bytes(b"wrong")
        return "ok"

    monkeypatch.setattr(loading, "download_model", download)
    with pytest.raises((RuntimeError, ValueError)) as error:
        dep.ensure_tabpfn_checkpoint()
    assert "fake-secret" not in str(error.value)
    if failure in {"returned_error", "raised_error", "missing_file"}:
        assert "official flow" in str(error.value)
