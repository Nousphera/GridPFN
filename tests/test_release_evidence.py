"""Synthetic in-memory release-gate tests; no public or held-out scores are read."""

import calendar
import copy
import json
import statistics

import pytest

from gridpfn.release_evidence import METHODS, METRICS, MONTHS, load_evidence, validate_evidence


def summary(values):
    return {"values": values, "mean": statistics.mean(values), "sd": statistics.stdev(values)}


def policy(kind):
    homes = [
        {
            "home_id": i,
            "objective": (1 if kind == "oracle" else 2) + (i - 12) * 0.01,
            "energy_bill_without_dr": 0.5 + i * 0.01,
            "comfort_pct": 80 + i * 0.1,
            "violation_hours": 24 * (1 - (80 + i * 0.1) / 100),
            "squared_violation": 0.2 + i * 0.01,
        }
        for i in range(25)
    ]
    return {"per_home": homes, "metrics": {k: summary([h[k] for h in homes]) for k in METRICS}}


def forecast(days):
    homes = [{"home_id": i, "queries": days * 10, "rmse": [0.1 + i * 0.01] * 3} for i in range(25)]
    return {
        "per_home": homes,
        **{k: summary([h["rmse"][0] for h in homes]) for k in ("load", "pv", "temperature")},
    }


@pytest.fixture
def evidence():
    sha = "a" * 64
    folds = []
    for month in MONTHS:
        dates = [
            f"{month}-{day:02d}"
            for day in range(1, calendar.monthrange(2019, int(month[-2:]))[1] + 1)
            if f"{month}-{day:02d}" != "2019-07-29"
        ]
        rows = [
            {
                "id": kind,
                "label": kind,
                "policy": policy(kind),
                "forecast": None if kind == "history" else forecast(len(dates)),
                "selected_episodes": 0,
                "selection_completed_episode": 3000,
                "selection_budget_cap": 20000,
                "selection_convergence": {"stopped": True},
                "checkpoint_sha256": sha,
                "evaluation_sha256": sha,
                "audit": {
                    "split": "test",
                    "dates": dates,
                    "checkpoint_sha256": sha,
                    "checkpoint_episode": 0,
                    "reference_commit": "b" * 40,
                    "matched_policy_transitions": len(dates) * 25 * 24,
                    "max_absolute_daily_difference": 0,
                },
            }
            for kind in METHODS
        ]
        oracle = {
            "id": "oracle",
            "label": "oracle",
            "policy": policy("oracle"),
            "bound": {
                "lower": 1 - 1e-8,
                "upper": 1,
                "daily_lower": [1 - 1e-8] * len(dates),
                "daily_upper": [1] * len(dates),
            },
            "protocol_sha256": sha,
            "receipts": [{"date": day, "file_sha256": sha} for day in dates],
            "reference_commit": "b" * 40,
        }
        folds.append(
            {
                "id": month,
                "dates": dates,
                "home_ids": list(range(25)),
                "rows": rows,
                "oracle": oracle,
            }
        )
    pooled = copy.deepcopy(folds[0])
    pooled.pop("id")
    pooled.update(
        schema_version=2,
        folds=folds,
        protocol={"seed": 41, "home_ids": list(range(25)), "methods": list(METHODS)},
        dates=[day for f in folds for day in f["dates"]],
        frozen_selection_sha256=sha,
        frozen_at_utc="2026-10-06T10:00:00+00:00",
        numerical_source={"model.py": sha},
    )
    for row in pooled["rows"]:
        if row["id"] != "history":
            row["forecast"] = forecast(len(pooled["dates"]))
    return pooled


def test_complete_export_roundtrips_without_optional_dependencies(evidence, tmp_path):
    target = tmp_path / "performance.json"
    target.write_text(json.dumps(evidence))
    assert load_evidence(target) == evidence
    assert len(evidence["dates"]) == 152


@pytest.mark.parametrize(
    "corruption",
    [
        "fixture",
        "nested_fixture",
        "missing_method",
        "duplicate_method",
        "missing_month",
        "missing_day",
        "missing_home",
        "wrong_seed",
        "missing_freeze",
        "missing_source",
        "missing_checkpoint",
        "bad_mean",
        "bad_sd",
        "bad_values",
        "nan",
        "reversed_bounds",
        "wrong_daily_bounds",
        "missing_oracle_receipt",
        "bad_audit",
        "wrong_audit_checkpoint",
        "wrong_budget",
        "pooled_weight",
        "pooled_forecast",
        "forecast_mean",
        "forecast_negative",
        "comfort_hours",
    ],
)
def test_rejects_unpublishable_evidence(evidence, corruption):
    first = evidence["folds"][0]
    row = first["rows"][0]
    metric = row["policy"]["metrics"]["objective"]
    if corruption == "fixture":
        evidence["fixture"] = True
    elif corruption == "nested_fixture":
        row["fixture"] = True
    elif corruption == "missing_method":
        first["rows"].pop()
    elif corruption == "duplicate_method":
        first["rows"][1]["id"] = "history"
    elif corruption == "missing_month":
        evidence["folds"].pop()
    elif corruption == "missing_day":
        first["dates"].pop()
    elif corruption == "missing_home":
        row["policy"]["per_home"].pop()
    elif corruption == "wrong_seed":
        evidence["protocol"]["seed"] = 9
    elif corruption == "missing_freeze":
        evidence.pop("frozen_selection_sha256")
    elif corruption == "missing_source":
        evidence["numerical_source"] = {}
    elif corruption == "missing_checkpoint":
        row.pop("checkpoint_sha256")
    elif corruption == "bad_mean":
        metric["mean"] += 1
    elif corruption == "bad_sd":
        metric["sd"] /= 5
    elif corruption == "bad_values":
        metric["values"][0] += 1
    elif corruption == "nan":
        metric["sd"] = float("nan")
    elif corruption == "reversed_bounds":
        first["oracle"]["bound"]["lower"] = 2
    elif corruption == "wrong_daily_bounds":
        first["oracle"]["bound"]["daily_upper"][0] = 3
    elif corruption == "missing_oracle_receipt":
        first["oracle"]["receipts"].pop()
    elif corruption == "bad_audit":
        row["audit"]["matched_policy_transitions"] -= 24
    elif corruption == "wrong_audit_checkpoint":
        row["audit"]["checkpoint_sha256"] = "c" * 64
    elif corruption == "wrong_budget":
        row["selected_episodes"] = 4000
    elif corruption == "pooled_weight":
        target = evidence["rows"][0]["policy"]
        for h in target["per_home"]:
            h["objective"] += 1
        target["metrics"]["objective"] = summary([h["objective"] for h in target["per_home"]])
    elif corruption == "pooled_forecast":
        evidence["rows"][1]["forecast"]["per_home"][0]["queries"] += 1
    elif corruption == "forecast_mean":
        first["rows"][1]["forecast"]["load"]["mean"] += 0.1
    elif corruption == "forecast_negative":
        first["rows"][1]["forecast"]["per_home"][0]["rmse"][0] = -1
    elif corruption == "comfort_hours":
        row["policy"]["per_home"][0]["violation_hours"] += 1
    with pytest.raises(ValueError):
        validate_evidence(evidence)


def test_missing_file_is_failure_not_skip(tmp_path):
    with pytest.raises(ValueError, match="required"):
        load_evidence(tmp_path / "missing.json")


@pytest.mark.parametrize("contents", ["{", "[]", "null", '{"schema_version":2}'])
def test_malformed_export_is_failure(tmp_path, contents):
    target = tmp_path / "performance.json"
    target.write_text(contents)
    with pytest.raises(ValueError):
        load_evidence(target)


def test_packaging_cannot_use_old_evidence_as_release_gate(tmp_path, monkeypatch):
    from scripts import make_release

    (tmp_path / "site").mkdir()
    (tmp_path / "site/evidence.json").write_text("{}")
    monkeypatch.setattr(make_release, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["make_release"])

    def unexpected_git(*args, **kwargs):
        raise AssertionError("Artifact validation must precede Git/archive operations")

    monkeypatch.setattr(make_release.subprocess, "check_output", unexpected_git)
    with pytest.raises(ValueError, match="performance.json"):
        make_release.main()


def highlight_fold(name, bill, competitors, comfort=80, days=30):
    """Small presentation fixture; deliberately independent of released scores."""
    rows = []
    for method, cost, warmth in [
        ("tabpfn", bill, comfort),
        ("tabfm", competitors[0], 70),
        ("tabicl", competitors[1], 70),
    ]:
        rows.append(
            {
                "id": method,
                "policy": {
                    "metrics": {
                        "energy_bill_without_dr": {"mean": cost},
                        "comfort_pct": {"mean": warmth},
                    }
                },
            }
        )
    return {"id": name, "dates": list(range(days)), "rows": rows}


def test_highlight_requires_joint_gains_and_ranks_the_weaker_comparison():
    from scripts.plot_foundation_comparison import select_highlight

    evidence = {
        "folds": [
            highlight_fold("unbalanced", 90, (100, 200)),
            highlight_fold("balanced", 80, (100, 110)),
            highlight_fold("worse-comfort", 1, (100, 110), comfort=60),
            highlight_fold("short-window", 2, (100, 110), days=6),
        ]
    }
    assert select_highlight(evidence)["id"] == "balanced"


@pytest.mark.parametrize("comfort,baseline", [(70, 100), (60, 100), (80, 0), (80, 80)])
def test_highlight_refuses_to_claim_a_joint_win_when_none_exists(comfort, baseline):
    from scripts.plot_foundation_comparison import select_highlight

    evidence = {"folds": [highlight_fold("no-joint-win", 80, (100, baseline), comfort)]}
    with pytest.raises(ValueError, match="No recorded month"):
        select_highlight(evidence)
