"""Fail-closed report contracts; no licensed inputs or model weights required."""

import copy

import numpy as np
import pytest

from gridpfn.experiments import full_report as report


def test_household_sd_is_sample_dispersion_not_standard_error():
    actual = report.summarize_homes([1, 3, 5])
    assert actual == {"mean": 3.0, "sd": 2.0, "values": [1.0, 3.0, 5.0]}
    with pytest.raises(ValueError, match="finite"):
        report.summarize_homes([1, np.nan])


def policy_record():
    dates = ["2019-06-01", "2019-06-02"]
    daily = []
    homes = []
    for home, offset in [(27, 0), (950, 10)]:
        rows = [
            {
                "day": day,
                "reward": -(offset + i + 1),
                **{key: offset + i + 1 for key in report.METRICS if key != "objective"},
            }
            for i, day in enumerate(dates)
        ]
        daily.append(rows)
        homes.append(
            {
                "home_id": home,
                "reward": -offset - 1.5,
                **{key: offset + 1.5 for key in report.METRICS if key != "objective"},
            }
        )
    return {"split": "test", "dates": dates, "homes": homes, "day_records": daily}


def test_policy_aggregation_uses_complete_daily_home_records():
    record = policy_record()
    actual = report.summarize_policy(record, [27, 950], record["dates"])
    assert actual["metrics"]["objective"]["mean"] == 6.5
    assert actual["metrics"]["objective"]["values"] == [1.5, 11.5]
    assert actual["metrics"]["objective"]["sd"] == pytest.approx(np.std([1.5, 11.5], ddof=1))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["day_records"][0].pop(),
        lambda r: r["day_records"][0][0].update(day="2019-06-02"),
        lambda r: r["homes"][0].update(reward=-999),
        lambda r: r["day_records"][0][0].update(reward=float("nan")),
        lambda r: r.update(split="validation"),
    ],
)
def test_policy_incomplete_or_corrupted_daily_records_fail(mutate):
    record = policy_record()
    mutate(record)
    with pytest.raises(ValueError):
        report.summarize_policy(record, [27, 950], ["2019-06-01", "2019-06-02"])


def monthly_fold(dates, values):
    homes = [
        {"home_id": h, **{key: value for key in report.METRICS}}
        for h, value in zip([27, 950], values, strict=True)
    ]
    return {
        "home_ids": [27, 950],
        "dates": dates,
        "rows": [{"id": "tabpfn", "label": "TabPFN-3.5", "policy": {"per_home": homes}}],
    }


def test_month_pooling_weights_days_within_home_then_weights_homes_equally():
    folds = [
        monthly_fold(["2019-06-01"], [1, 11]),
        monthly_fold(["2019-07-01", "2019-07-02", "2019-07-03"], [5, 15]),
    ]
    actual = report.pool_monthly_results(folds)["rows"][0]["policy"]["metrics"]["objective"]
    assert actual["values"] == [4.0, 14.0]
    assert actual["mean"] == 9.0
    assert actual["sd"] == pytest.approx(np.std([4, 14], ddof=1))


@pytest.mark.parametrize("kind", ["overlap", "reversed", "cohort", "missing_home", "method"])
def test_month_pooling_refuses_unequal_or_overlapping_folds(kind):
    first = monthly_fold(["2019-06-01"], [1, 11])
    second = monthly_fold(["2019-07-01"], [2, 12])
    if kind == "overlap":
        second["dates"] = first["dates"]
    elif kind == "reversed":
        first, second = second, first
    elif kind == "cohort":
        second["home_ids"] = [950, 27]
    elif kind == "missing_home":
        second["rows"][0]["policy"]["per_home"].pop()
    else:
        second["rows"][0]["id"] = "different"
    with pytest.raises(ValueError):
        report.pool_monthly_results([first, second])


def test_corrupt_metric_does_not_produce_a_plot_ready_result():
    first = monthly_fold(["2019-06-01"], [1, 11])
    changed = copy.deepcopy(first)
    changed["rows"][0]["policy"]["per_home"][0]["objective"] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        report.pool_monthly_results([changed])


def test_plot_exports_all_methods_and_oracle_with_dispersion_label(tmp_path):
    rows = []
    for index, kind in enumerate(report.METHODS + ("oracle",)):
        metrics = {key: report.summarize_homes([index + 1, index + 2]) for key in report.METRICS}
        rows.append(
            {
                "id": kind,
                "label": "Perfect-future oracle" if kind == "oracle" else report.LABELS[kind],
                "policy": {"metrics": metrics},
            }
        )
    report.plot_performance(
        {"home_ids": [27, 950], "rows": rows[:-1], "oracle": rows[-1]}, tmp_path / "plot"
    )
    for suffix in ("svg", "png", "pdf"):
        assert (tmp_path / f"plot.{suffix}").stat().st_size > 1000
    source = (tmp_path / "plot.svg").read_text()
    for row in rows:
        assert row["label"] in source
    assert "sample SD" in source
    assert "Only its combined objective is a bound" in source


@pytest.fixture
def frozen_study(tmp_path, monkeypatch):
    study = {"protocol": {"methods": list(report.METHODS), "home_ids": list(range(25))}}
    report.write_json(tmp_path / "study.json", study)
    rows = []
    for month in report.FOLDS:
        for kind in report.METHODS:
            run = f"folds/{month}/refit/policies/{kind}"
            heads = tmp_path / run / "checkpoints/latest/heads.pt"
            heads.parent.mkdir(parents=True)
            heads.write_bytes(f"{month}/{kind}: fixed final weights".encode())
            rows.append(
                {
                    "fold": month,
                    "id": kind,
                    "run": run,
                    "heads_sha256": report.digest(heads),
                    "recipe": {"episodes": 200},
                }
            )
    monkeypatch.setattr(report, "_validate_study", lambda root: (study, rows, {"core.py": "sha"}))
    return tmp_path, rows


def test_all_30_refits_archived_before_manifest_and_test(frozen_study):
    root, rows = frozen_study
    frozen = report.freeze_selection(root)
    assert len(frozen["methods"]) == 30
    assert len(list((root / "frozen_selection").glob("*/*/heads.pt"))) == 30
    for row in rows:
        assert (
            report.digest(root / "frozen_selection" / row["fold"] / row["id"] / "heads.pt")
            == row["heads_sha256"]
        )
    assert report.freeze_selection(root) == frozen


def test_unfrozen_preexisting_test_is_rejected(frozen_study):
    root, rows = frozen_study
    report.write_json(root / rows[0]["run"] / "evaluation/test_latest.json", {"reward": 1})
    with pytest.raises(ValueError, match="without a prior"):
        report.freeze_selection(root)
    assert not (root / "frozen_selection/manifest.json").exists()


def test_changed_recipe_after_freeze_is_rejected(frozen_study):
    root, rows = frozen_study
    report.freeze_selection(root)
    rows[0]["recipe"]["episodes"] = 300
    with pytest.raises(ValueError, match="Frozen selection changed"):
        report.freeze_selection(root)


def test_modified_archived_checkpoint_is_rejected(frozen_study):
    root, rows = frozen_study
    report.freeze_selection(root)
    first = rows[0]
    (root / "frozen_selection" / first["fold"] / first["id"] / "heads.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Archived final refit"):
        report.freeze_selection(root)


def test_evaluation_cannot_start_until_entire_freeze_exists(frozen_study, monkeypatch):
    root, _ = frozen_study
    called = []

    def subprocess_stub(command, **kwargs):
        assert (root / "frozen_selection/manifest.json").exists()
        assert len(list((root / "frozen_selection").glob("*/*/heads.pt"))) == 30
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        # Emulate only output existence; numerical contracts are checked by collect.
        report.write_json(command[-1], {})
        called.append(command)

    monkeypatch.setattr(report.subprocess, "run", subprocess_stub)
    report.evaluate(root)
    assert len(called) == 60


def test_incomplete_training_never_launches_test(tmp_path, monkeypatch):
    def incomplete(root):
        raise ValueError("Complete all selections and refits first")

    monkeypatch.setattr(report, "_validate_study", incomplete)
    monkeypatch.setattr(
        report.subprocess, "run", lambda *a, **k: pytest.fail("Test launched early")
    )
    with pytest.raises(ValueError, match="Complete all"):
        report.evaluate(tmp_path)


def test_pooled_forecast_rmse_uses_query_weighted_squared_errors():
    folds = []
    for count, error in [(1, 1), (3, 3)]:
        homes = [{"home_id": h, "queries": count, "rmse": [error] * 3} for h in [27, 950]]
        folds.append({"rows": [{"id": "tabpfn", "forecast": {"per_home": homes}}]})
    result = report.pool_forecasts(folds, "tabpfn", [27, 950])
    assert result["load"]["mean"] == pytest.approx(np.sqrt(7))
    assert result["per_home"][0]["queries"] == 4
    assert result["load"]["mean"] != 2.5  # Averaging rooted monthly errors is wrong.


def test_refit_proves_all_prior_train_and_validation_dates(tmp_path):
    select, refit = tmp_path / "select", tmp_path / "refit"
    select.mkdir()
    refit.mkdir()
    homes = [{"id": 27, "path": "home_27.npz"}]
    np.savez(
        select / "home_27.npz",
        train_dates=["2019-05-01", "2019-05-24"],
        dates=["2019-05-01", "2019-05-24", "2019-05-25", "2019-05-31"],
    )
    np.savez(
        refit / "home_27.npz", train_dates=["2019-05-01", "2019-05-24", "2019-05-25", "2019-05-31"]
    )
    report._verify_refit_dates(
        select, {"homes": homes}, refit, {"homes": homes}, "2019-06-01", "2019-05-25"
    )
    np.savez(
        refit / "home_27.npz", train_dates=["2019-05-01", "2019-05-24", "2019-05-25", "2019-06-01"]
    )
    with pytest.raises(ValueError, match="no test dates"):
        report._verify_refit_dates(
            select, {"homes": homes}, refit, {"homes": homes}, "2019-06-01", "2019-05-25"
        )


@pytest.fixture
def seasonal_artifacts(tmp_path, monkeypatch):
    """Actual on-disk stage/recipe/refit records; stub only upstream feature hashes."""
    from gridpfn.experiments import foundation_study

    current = tmp_path / "current_source"
    current.mkdir()
    (current / "core.py").write_text("same immutable numerical source")
    monkeypatch.setattr(report, "ROOT", current)
    monkeypatch.setattr(report, "CORE_SOURCE", ("core.py",))
    monkeypatch.setattr(
        foundation_study, "verify_inputs", lambda stage: report.read_json(stage / "cases.json")
    )
    monkeypatch.setattr(foundation_study, "verify_features", lambda stage, kind: None)
    root = tmp_path / "study"
    protocol = {
        "methods": list(report.METHODS),
        "home_ids": list(range(25)),
        "seed": 41,
        "episodes": 100,
        "actor_width": 256,
        "value_width": 64,
    }
    input_sources = {
        f"dataset/split_homes_clean/home_{h}.csv": str(h) for h in protocol["home_ids"]
    }
    weather = current / "weather.csv"
    weather.write_bytes(b"weather")
    input_sources["dataset/temp_price_newyork.csv"] = report.digest(weather)
    folds = []
    for month in report.FOLDS:
        start, end = report.month_bounds(month)
        valid_start = (report.date.fromisoformat(start) - report.timedelta(days=7)).isoformat()
        dates = report.dates_between(valid_start, start)
        fold = {
            "id": month,
            "select": f"folds/{month}/select",
            "refit": f"folds/{month}/refit",
            "recipe": f"folds/{month}/recipe.json",
            "oracle": f"oracles/{month}",
        }
        folds.append(fold)
        recipes = {}
        for phase in ("select", "refit"):
            stage = root / fold[phase]
            cfg = {**protocol, "data_period": f"month_{month[-2:]}_{phase}"}
            if phase == "refit":
                cfg["refit_budgets"] = {k: 100 for k in report.METHODS}
            homes = [{"id": h, "path": f"home_{h}.npz"} for h in protocol["home_ids"]]
            report.write_json(
                stage / "cases.json",
                {"protocol": cfg, "homes": homes, "input_sources": input_sources},
            )
            for home in homes:
                train_dates = ["2019-05-01"] if phase == "select" else ["2019-05-01", *dates]
                all_dates = train_dates + (
                    dates if phase == "select" else report.dates_between(start, end)
                )
                np.savez(stage / home["path"], train_dates=train_dates, dates=all_dates)
            for kind in report.METHODS:
                run = stage / "policies" / kind
                selected_run = root / fold["select"] / "policies" / kind
                settings = {
                    "feature_mode": "raw",
                    "embedding_weight": 1,
                    "actor_update": "ppo",
                    "seed": 41,
                    "fixed_seed": 4101,
                    "home_ids": protocol["home_ids"],
                    "episode": 100,
                    "head_width": 256,
                    "value_width": 64,
                    "data_period": cfg["data_period"],
                    "refit": phase == "refit",
                    "ppo_shuffle_days": True,
                    "synthetic_data": None,
                    "bc_rounds": 60,
                    "bc_weight": 0.0,
                    "strict_convergence": False,
                    "path_train": str(run),
                    "predictive_features": str(stage / "features" / kind),
                    "refit_selection": str(selected_run / "selection.json")
                    if phase == "refit"
                    else None,
                }
                report.write_json(run / "run.json", {"settings": settings})
                report.write_json(run / "status.json", {"state": "completed"})
                report.write_json(
                    run / "data_hashes.json",
                    {f"home_{h}.csv": str(h) for h in protocol["home_ids"]},
                )
                (run / "source/dataset").mkdir(parents=True)
                (run / "source/core.py").write_bytes((current / "core.py").read_bytes())
                (run / "source/dataset/temp_price_newyork.csv").write_bytes(weather.read_bytes())
                report.write_json(stage / "features" / kind / "manifest.json", {"kind": kind})
                checkpoint = (
                    run / f"checkpoints/{'best' if phase == 'select' else 'latest'}/heads.pt"
                )
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(f"{month}/{kind}/{phase}".encode())
                if phase == "select":
                    metrics = [
                        {
                            "kind": "eval",
                            "split": "validation",
                            "dates": dates,
                            "homes": [{"home_id": h} for h in protocol["home_ids"]],
                            "episode": n,
                            "reward": reward,
                        }
                        for n, reward in [(0, -4), (100, -1)]
                    ]
                    (run / "metrics.jsonl").write_text(
                        "\n".join(report.json.dumps(r) for r in metrics)
                    )
                    report.write_json(
                        run / "selection.json",
                        {
                            "initial": {"episode": 0, "reward": -4},
                            "best": {"episode": 100, "reward": -1},
                            "latest": {"episode": 100, "reward": -1},
                        },
                    )
                    report.write_json(
                        run / "convergence.json",
                        {"episode": 100, "stopped": False, "reason": "episode cap reached"},
                    )
                    recipes[kind] = {
                        "episodes": 100,
                        "selection_sha256": report.digest(run / "selection.json"),
                        "heads_sha256": report.digest(checkpoint),
                        "metrics_sha256": report.digest(run / "metrics.jsonl"),
                    }
                else:
                    source = {
                        "path": str(selected_run / "selection.json"),
                        "sha256": recipes[kind]["selection_sha256"],
                        "run_sha256": report.digest(selected_run / "run.json"),
                        "data_period": f"month_{month[-2:]}_select",
                        "selected_episode": 100,
                        "checkpoint": "best",
                    }
                    summary = {
                        "episode": 100,
                        "refit": True,
                        "evaluation_performed": False,
                        "data_period": cfg["data_period"],
                        "selection_source": source,
                        "reward": None,
                        "comfort_pct": None,
                        "elec_cost": None,
                        "feasible": None,
                    }
                    report.write_json(run / "refit_summary.json", summary)
                    report.write_json(run / "selection.json", {"latest": summary})
                    (run / "metrics.jsonl").write_text(
                        report.json.dumps({"kind": "refit_checkpoint", "episode": 100})
                    )
        report.write_json(root / fold["recipe"], {"methods": recipes})
    report.write_json(root / "study.json", {"protocol": protocol, "folds": folds})
    return root


def test_complete_seasonal_stage_contract_and_refit_freeze(seasonal_artifacts):
    root = seasonal_artifacts
    result = report.freeze_selection(root)
    assert len(result["methods"]) == 30
    assert result["methods"][0]["selection_dates"] == report.dates_between(
        "2019-05-25", "2019-06-01"
    )
    assert result["methods"][-1]["test_dates"][-1] == "2019-10-31"


@pytest.mark.parametrize(
    "mutation", ["missing_refit", "changed_budget", "refit_eval", "source", "selection"]
)
def test_actual_stage_contract_rejects_invalid_refit(seasonal_artifacts, mutation):
    root = seasonal_artifacts
    run = root / "folds/2019-06/refit/policies/tabpfn"
    if mutation == "missing_refit":
        report.write_json(run / "status.json", {"state": "running"})
    elif mutation == "changed_budget":
        path = run / "run.json"
        data = report.read_json(path)
        data["settings"]["episode"] = 200
        report.write_json(path, data)
    elif mutation == "refit_eval":
        (run / "metrics.jsonl").write_text(report.json.dumps({"kind": "eval", "reward": -1}))
    elif mutation == "source":
        (run / "source/core.py").write_text("changed")
    else:
        path = root / "folds/2019-06/recipe.json"
        data = report.read_json(path)
        data["methods"]["tabpfn"]["selection_sha256"] = "changed"
        report.write_json(path, data)
    with pytest.raises(ValueError):
        report.freeze_selection(root)
    assert not (root / "frozen_selection/manifest.json").exists()


@pytest.fixture
def oracle_fixture(tmp_path):
    from oracle.config import OracleConfig
    from oracle.runner import _digest

    dates = ["2019-06-01", "2019-06-02"]
    manifest = {
        "protocol": {"home_ids": [27, 950], "data_period": "month_06_refit"},
        "input_sources": {
            "dataset/split_homes_clean/home_27.csv": "27",
            "dataset/split_homes_clean/home_950.csv": "950",
            "dataset/temp_price_newyork.csv": "weather",
        },
    }
    protocol = {
        "home_ids": [27, 950],
        "data_period": "month_06_refit",
        "dates": dates,
        "split": "test",
        "perfect_foresight": True,
        "scenario": OracleConfig().settings(),
        "input_file_sha256": {"/local/home_27.csv": "27", "/local/home_950.csv": "950"},
        "price_weather_sha256": "weather",
        "source_sha256": {},
        "reference_commit": "original",
    }
    report.write_json(tmp_path / "oracle.json", protocol)
    for day in dates:
        homes = [
            {
                "reward": -1,
                "energy_bill_without_dr": 2,
                "strict_comfort_pct": 80,
                "comfort_pct": 85,
                "squared_violation": 3,
            },
            {
                "reward": -3,
                "energy_bill_without_dr": 4,
                "strict_comfort_pct": 90,
                "comfort_pct": 95,
                "squared_violation": 5,
            },
        ]
        record = {
            "date": day,
            "protocol_sha256": _digest(protocol),
            "oracles": {
                "paper_reward": {
                    "solution": {
                        "lower_bound": 4,
                        "upper_bound": 4,
                        "certificates": [
                            {"homes": [0], "certified": True},
                            {"homes": [1], "certified": True},
                        ],
                    },
                    "audit": {"verified": True, "homes": homes},
                }
            },
        }
        record["receipt_sha256"] = _digest(record)
        report.write_json(tmp_path / "days" / f"{day}.json", record)
    return tmp_path, manifest, dates


def test_oracle_bounds_and_strict_comfort_are_separate(oracle_fixture):
    root, manifest, dates = oracle_fixture
    actual = report.oracle_summary(root, manifest, dates, {})
    assert actual["bound"]["lower"] == actual["bound"]["upper"] == 2
    assert actual["policy"]["metrics"]["objective"]["values"] == [1, 3]
    assert actual["policy"]["metrics"]["comfort_pct"]["mean"] == 85
    assert actual["policy"]["per_home"][0]["solver_tolerant_comfort_pct"] == 85
    assert actual["policy"]["per_home"][0]["comfort_pct"] == 80
    assert "components" not in actual["bound"]


@pytest.mark.parametrize("mutation", ["uncertified", "missing_home", "bad_bound", "replay"])
def test_oracle_invalid_certificate_or_replay_is_rejected(oracle_fixture, mutation):
    from oracle.runner import _digest

    root, manifest, dates = oracle_fixture
    path = root / "days" / f"{dates[0]}.json"
    record = report.read_json(path)
    record.pop("receipt_sha256")
    oracle = record["oracles"]["paper_reward"]
    if mutation == "uncertified":
        oracle["solution"]["certificates"][0]["certified"] = False
    elif mutation == "missing_home":
        oracle["solution"]["certificates"].pop()
    elif mutation == "bad_bound":
        oracle["solution"]["lower_bound"] = 5
    else:
        oracle["audit"]["homes"][0]["reward"] = -10
    record["receipt_sha256"] = _digest(record)
    report.write_json(path, record)
    with pytest.raises(ValueError):
        report.oracle_summary(root, manifest, dates, {})
