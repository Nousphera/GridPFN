"""Finalization ordering and immutable orchestration; no real test data opened."""

import json

import pytest

from gridpfn.experiments import finalize_study as final


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    manifest = {
        "protocol": {"home_ids": list(range(25))},
        "folds": [
            {
                "id": f"2019-{month:02d}",
                "select": f"folds/2019-{month:02d}/select",
                "refit": f"folds/2019-{month:02d}/refit",
                "oracle": f"oracles/2019-{month:02d}",
            }
            for month in range(6, 11)
        ],
    }
    (root / "study.json").write_text(json.dumps(manifest))
    events, configs = [], []

    def freeze(path):
        assert path == root
        events.append("freeze")
        return {"study_sha256": final.full_report.digest(root / "study.json")}

    def oracle(config, *, resume):
        events.append(f"oracle:{config.data_period}:{resume}")
        configs.append(config)

    monkeypatch.setattr(final.full_report, "freeze_selection", freeze)
    monkeypatch.setattr(final, "_run_oracle", oracle)
    monkeypatch.setattr(final.full_report, "evaluate", lambda path: events.append("evaluate"))
    monkeypatch.setattr(
        final.full_report, "collect", lambda path: events.append("collect") or {"valid": True}
    )
    monkeypatch.setattr(
        final.full_report,
        "export",
        lambda evidence: events.append("export") if evidence["valid"] else None,
    )

    def models(run, output):
        assert run == root / "folds/2019-10/refit/policies/tabpfn"
        assert output == root / "models"
        events.append("models")
        return [output / "home_0"]

    monkeypatch.setattr(final, "_export_homes", models)
    return root, manifest, events, configs


def test_freeze_all_before_oracles_and_export_only_after_valid_report(workflow):
    root, _, events, configs = workflow
    assert final.finalize(root) == [root / "models/home_0"]
    assert events == [
        "freeze",
        *[f"oracle:month_{m:02d}_refit:False" for m in range(6, 11)],
        "evaluate",
        "collect",
        "export",
        "models",
    ]
    from oracle.config import OracleConfig

    original = OracleConfig().settings()
    for cfg in configs:
        assert cfg.split == "test" and cfg.home_ids == tuple(range(25))
        assert cfg.objectives == ("paper_reward",) and cfg.workers == 4 and cfg.days == 0
        for key, value in original.items():
            if key not in {"output", "split", "data_period", "home_ids", "objectives"}:
                assert cfg.settings()[key] == value


def test_incomplete_refits_prevent_all_test_and_oracle_access(workflow, monkeypatch):
    root, _, events, _ = workflow

    def incomplete(path):
        events.append("freeze")
        raise ValueError("Incomplete refits")

    monkeypatch.setattr(final.full_report, "freeze_selection", incomplete)
    monkeypatch.setattr(
        final.full_report, "read_json", lambda path: pytest.fail("Read artifact before freeze")
    )
    with pytest.raises(ValueError, match="Incomplete"):
        final.finalize(root)
    assert events == ["freeze"]


def test_resumes_matching_original_oracle(workflow):
    root, manifest, events, _ = workflow
    from oracle.config import OracleConfig

    output = root / manifest["folds"][0]["oracle"]
    output.mkdir(parents=True)
    config = OracleConfig(
        output=output,
        split="test",
        data_period="month_06_refit",
        home_ids=tuple(range(25)),
        objectives=("paper_reward",),
        workers=2,
    )
    (output / "oracle.json").write_text(json.dumps({"scenario": config.settings()}))
    final.finalize(root, workers=2)
    assert events[1] == "oracle:month_06_refit:True"


@pytest.mark.parametrize(
    "kind", ["conflicting_settings", "unreceipted_output", "escape", "overlap"]
)
def test_preflights_all_outputs_without_running_earlier_folds(workflow, kind):
    root, manifest, events, _ = workflow
    fold = manifest["folds"][-1]
    if kind == "escape":
        fold["oracle"] = "../outside"
    elif kind == "overlap":
        fold["oracle"] = manifest["folds"][0]["oracle"]
    else:
        output = root / fold["oracle"]
        output.mkdir(parents=True)
        if kind == "conflicting_settings":
            (output / "oracle.json").write_text(json.dumps({"scenario": {"split": "train"}}))
    (root / "study.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        final.finalize(root)
    assert events == ["freeze"]


@pytest.mark.parametrize("failure", ["oracle", "evaluate", "collect", "export"])
def test_failure_never_exports_home_models(workflow, monkeypatch, failure):
    root, _, events, _ = workflow

    def fail(*args, **kwargs):
        events.append(f"failed:{failure}")
        raise ValueError("Unfinished comparison")

    target, name = (final, "_run_oracle") if failure == "oracle" else (final.full_report, failure)
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(ValueError, match="Unfinished"):
        final.finalize(root)
    assert "models" not in events
    if failure in {"oracle", "evaluate", "collect"}:
        assert "export" not in events


def test_help_does_not_open_study_or_invoke_freeze(monkeypatch, capsys):
    monkeypatch.setattr(
        final.full_report, "freeze_selection", lambda *_: pytest.fail("Unexpected freeze")
    )
    with pytest.raises(SystemExit) as exc:
        final.main(["--help"])
    assert exc.value.code == 0 and "--workers" in capsys.readouterr().out
