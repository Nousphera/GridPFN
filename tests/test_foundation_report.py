"""Artifact-contract checks for the one-seed report, without model weights or CSVs."""

import hashlib
import json
import sys

import numpy as np
import pytest

import gridpfn.experiments.foundation_pipeline as foundation_pipeline
import gridpfn.experiments.foundation_report as foundation_report
import gridpfn.experiments.submission_report as submission_report


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def mutate(path, update):
    record = json.loads(path.read_text())
    update(record)
    write_json(path, record)


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Stub upstream verification only; exercise real report files and SHA bindings."""
    cfg = {
        "methods": ["history", "tabpfn"], "seed": 41, "home_ids": [27, 950],
        "actor_width": 256, "value_width": 64, "episodes": 1000, "scope": "test fixture",
    }
    dates = ["2019-07-18", "2019-07-19"]
    homes = [{"id": h, "training_days": 1, "path": f"home{h}.npz"} for h in cfg["home_ids"]]
    cases = [
        {"id": f"home{h}_from1", "home": h, "first": 1,
         "query_dates": dates, "path": f"case{h}.npz"}
        for h in cfg["home_ids"]
    ]
    input_sources = {
        f"dataset/split_homes_clean/home_{h}.csv": hashlib.sha256(f"home{h}".encode()).hexdigest()
        for h in cfg["home_ids"]
    }
    input_sources["dataset/temp_price_newyork.csv"] = hashlib.sha256(b"weather").hexdigest()
    manifest = {"protocol": cfg, "homes": homes, "cases": cases, "input_sources": input_sources}
    write_json(tmp_path / "cases.json", manifest)
    monkeypatch.setattr(foundation_report, "verify_inputs", lambda root: manifest)
    monkeypatch.setattr(foundation_report, "verify_features", lambda root, kind: None)
    monkeypatch.setattr(submission_report, "CORE_SOURCE", ("core.py",))
    current_source = tmp_path / "current_source"
    current_source.mkdir()
    (current_source / "core.py").write_text("same numerical implementation")
    monkeypatch.setattr(foundation_report, "ROOT", current_source)
    for home, case in zip(homes, cases, strict=True):
        np.savez(tmp_path / home["path"], physical=np.ones((3, 24, 3)))
        np.savez(tmp_path / case["path"], indices=np.array([[0, 0, 1], [1, 0, 1]]))
        prediction = tmp_path / "predictions/tabpfn"
        prediction.mkdir(parents=True, exist_ok=True)
        np.savez(prediction / f"{case['id']}.npz", prediction=np.ones((2, 3)))
        write_json(prediction / f"{case['id']}.json", {
            "timing": [{"fit_seconds": 0.1, "predict_seconds": 0.2}], "device": "cpu",
            "models": [{"backend": "tabpfn"}], "batch_causality": [],
        })
    for kind in cfg["methods"]:
        write_json(tmp_path / "features" / kind / "manifest.json", {"kind": kind})
        run = tmp_path / "policies" / kind
        settings = {
            "feature_mode": "raw", "embedding_weight": 1, "seed": 41, "fixed_seed": 4101,
            "home_ids": cfg["home_ids"], "head_width": 256, "value_width": 64,
            "episode": 1000, "ppo_shuffle_days": True, "synthetic_data": None,
            "bc_rounds": 60, "bc_weight": 0.0, "validation_only": True,
        }
        write_json(run / "status.json", {"state": "completed"})
        write_json(run / "run.json", {"settings": settings})
        write_json(run / "selection.json", {
            "initial": {"reward": -4, "episode": 0},
            "best": {"reward": -2, "episode": 800},
            "latest": {"reward": -3, "episode": 1000},
        })
        (run / "source").mkdir()
        (run / "source/core.py").write_text("same numerical implementation")
        (run / "source/dataset").mkdir()
        (run / "source/dataset/temp_price_newyork.csv").write_bytes(b"weather")
        write_json(run / "data_hashes.json", {
            f"home_{h}.csv": input_sources[f"dataset/split_homes_clean/home_{h}.csv"]
            for h in cfg["home_ids"]
        })
        (run / "metrics.jsonl").write_text("".join(
            json.dumps({"kind": "eval", "split": "validation", "dates": dates,
                        "homes": [{"home_id": h} for h in cfg["home_ids"]],
                        "episode": episode, "reward": reward}) + "\n"
            for episode, reward in [(0, -4), (800, -2), (1000, -3)]
        ))
        for checkpoint, reward, episode in [("best", -2, 800), ("latest", -3, 1000)]:
            heads = run / "checkpoints" / checkpoint / "heads.pt"
            heads.parent.mkdir(parents=True)
            heads.write_bytes(f"{kind}-{checkpoint}".encode())
            digest = hashlib.sha256(heads.read_bytes()).hexdigest()
            write_json(run / "evaluation" / f"validation_{checkpoint}.json", {
                "split": "validation", "dates": dates, "checkpoint_sha256": digest,
                "episode": episode, "homes": [
                    {"home_id": h, "reward": reward, "energy_bill_without_dr": 1,
                     "squared_violation": 20} for h in cfg["home_ids"]
                ],
            })
            if checkpoint == "best":
                write_json(run / "independent_audit.json", {"policy_replay": {
                    "split": "validation", "dates": dates, "checkpoint_sha256": digest,
                    "matched_policy_transitions": 96, "max_absolute_daily_difference": 0,
                }})
    return tmp_path


def test_complete_report_preserves_forecast_and_control_evidence(study):
    result = foundation_report.collect(study)
    assert result["dates"] == ["2019-07-18", "2019-07-19"]
    assert [r["id"] for r in result["rows"]] == ["history", "tabpfn"]
    tab = result["rows"][1]
    assert tab["forecast_rmse"] == [0, 0, 0]
    assert tab["forecast_query_count"] == 4
    assert tab["policy"]["selected_objective"] == 2
    assert tab["policy"]["final_objective"] == 3
    assert "One seed" in result["finding"]


@pytest.mark.parametrize("artifact,update,match", [
    ("status.json", lambda d: d.update(state="running"), "Incomplete policy"),
    ("run.json", lambda d: d["settings"].update(seed=42), "Unmatched executed settings"),
    ("evaluation/validation_best.json", lambda d: d.update(split="test"), "cohort/split"),
    ("evaluation/validation_best.json", lambda d: d.update(dates=["2019-08-01"]), "cohort/split"),
    ("evaluation/validation_best.json", lambda d: d["homes"].reverse(), "cohort/split"),
    ("evaluation/validation_best.json", lambda d: d.update(checkpoint_sha256="wrong"), "Checkpoint identity"),
    ("selection.json", lambda d: d["best"].update(reward=-99), "Selected best"),
    ("selection.json", lambda d: d["latest"].update(episode=999), "Selected latest"),
    ("independent_audit.json", lambda d: d["policy_replay"].update(split="test"), "audit identity"),
    ("independent_audit.json", lambda d: d["policy_replay"].update(checkpoint_sha256="wrong"), "audit identity"),
    ("independent_audit.json", lambda d: d["policy_replay"].update(matched_policy_transitions=95), "Incomplete or failed"),
    ("independent_audit.json", lambda d: d["policy_replay"].update(max_absolute_daily_difference=0.1), "Incomplete or failed"),
])
def test_rejects_incomplete_or_mismatched_artifacts(study, artifact, update, match):
    mutate(study / "policies/tabpfn" / artifact, update)
    with pytest.raises(ValueError, match=match):
        foundation_report.collect(study)


def test_rejects_different_numerical_source_between_methods(study):
    (study / "policies/tabpfn/source/core.py").write_text("changed training algorithm")
    with pytest.raises(ValueError, match="numerical code|source|Source"):
        foundation_report.collect(study)


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -1])
def test_rejects_invalid_replay_error_value(study, error):
    mutate(study / "policies/tabpfn/independent_audit.json",
           lambda d: d["policy_replay"].update(max_absolute_daily_difference=error))
    with pytest.raises(ValueError):
        foundation_report.collect(study)


def test_rejects_evaluation_episode_not_matching_selection(study):
    mutate(study / "policies/tabpfn/evaluation/validation_best.json",
           lambda d: d.update(episode=999))
    with pytest.raises(ValueError):
        foundation_report.collect(study)


def test_pipeline_preserves_virtualenv_interpreter_symlink(tmp_path, monkeypatch):
    """Resolving a venv Python symlink selects the base interpreter and loses its deps."""
    base = tmp_path / "base-python"
    base.write_text("not executed")
    interpreters = []
    for name in ["tabfm-env", "tabicl-env"]:
        path = tmp_path / name / "bin/python"
        path.parent.mkdir(parents=True)
        path.symlink_to(base)
        interpreters.append(path)
    output = tmp_path / "study"
    commands = []
    monkeypatch.setattr(foundation_pipeline, "prepare", lambda path, protocol: path.mkdir())
    monkeypatch.setattr(foundation_pipeline.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(sys, "argv", [
        "foundation_pipeline.py", "--output", str(output),
        "--tabfm-python", str(interpreters[0]), "--tabicl-python", str(interpreters[1]),
        "--device", "cpu",
    ])
    foundation_pipeline.main()
    workers = {c[c.index("--kind") + 1]: c for c in commands if c[1:3] == ["-m", "gridpfn.experiments.foundation_worker"]}
    assert workers["tabfm"][0] == str(interpreters[0].absolute())
    assert workers["tabicl"][0] == str(interpreters[1].absolute())
    assert workers["tabfm"][0] != str(base)
    assert json.loads((output / "pipeline_status.json").read_text())["state"] == "completed"


@pytest.mark.parametrize("artifact", ["data_hashes.json", "source/dataset/temp_price_newyork.csv"])
def test_rejects_input_provenance_drift(study, artifact):
    path = study / "policies/tabpfn" / artifact
    if path.suffix == ".json":
        mutate(path, lambda d: d.update({"home_27.csv": "wrong"}))
    else:
        path.write_text("changed weather")
    with pytest.raises(ValueError):
        foundation_report.collect(study)


def test_rejects_missing_final_training_evaluation(study):
    path = study / "policies/tabpfn/metrics.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError):
        foundation_report.collect(study)


def test_public_export_removes_local_paths_without_mutating_private_provenance():
    private = {"rows": [{"model": {
        "installed_source": {"url": "file:///private/local/env"},
        "installed_source_revision": "abc123", "checkpoint_sha256": "verified-sha",
    }}, {"model": None}]}
    public = foundation_report.public_evidence(private)
    assert "installed_source" not in public["rows"][0]["model"]
    assert public["rows"][0]["model"]["installed_source_revision"] == "abc123"
    assert public["rows"][0]["model"]["checkpoint_sha256"] == "verified-sha"
    assert "installed_source" in private["rows"][0]["model"]
