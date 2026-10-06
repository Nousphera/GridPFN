"""Fail-closed, dependency-free checks for exported seasonal evidence.

This checks public receipts and arithmetic, not private source data or solver
certificates. ``full_report`` verifies those originals before exporting.
"""

import argparse
import calendar
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

METHODS = ("history", "persistence", "trees", "tabpfn", "tabfm", "tabicl")
MONTHS = tuple(f"2019-{month:02d}" for month in range(6, 11))
METRICS = (
    "objective",
    "energy_bill_without_dr",
    "comfort_pct",
    "violation_hours",
    "squared_violation",
)


def require(condition, message):
    if not condition:
        raise ValueError(f"Invalid seasonal evidence: {message}")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected):
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)


def sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "missing SHA256 receipt"
    )


def clean(value):
    if isinstance(value, dict):
        require(not any(value.get(k) for k in ("fixture", "synthetic", "is_fixture")), "UI fixture")
        for item in value.values():
            clean(item)
    elif isinstance(value, list):
        for item in value:
            clean(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite value")


def stats(record, values):
    require(len(values) == 25 and all(finite(v) for v in values), "incomplete household values")
    require(record["values"] == values, "summary differs from household records")
    require(close(record["mean"], statistics.mean(values)), "household mean mismatch")
    require(close(record["sd"], statistics.stdev(values)), "household sample SD mismatch")


def policy(row, homes):
    require(isinstance(row["label"], str) and row["label"], "missing method label")
    per_home = row["policy"]["per_home"]
    require([h["home_id"] for h in per_home] == homes, "household order/cohort mismatch")
    for key in METRICS:
        stats(row["policy"]["metrics"][key], [h[key] for h in per_home])
    for home in per_home:
        require(0 <= home["comfort_pct"] <= 100, "comfort outside percentage range")
        require(
            close(home["violation_hours"], 24 * (1 - home["comfort_pct"] / 100)),
            "comfort/violation hours mismatch",
        )
        require(home["squared_violation"] >= 0, "negative squared violation")


def forecasts(row, homes):
    forecast = row["forecast"]
    if row["id"] == "history":
        require(forecast is None, "history-only arm has forecast scores")
        return
    records = forecast["per_home"]
    require([h["home_id"] for h in records] == homes, "forecast household cohort mismatch")
    require(
        all(type(h["queries"]) is int and h["queries"] > 0 for h in records),
        "missing forecast queries",
    )
    require(
        all(len(h["rmse"]) == 3 and all(finite(v) and v >= 0 for v in h["rmse"]) for h in records),
        "invalid forecast RMSE",
    )
    for index, key in enumerate(("load", "pv", "temperature")):
        stats(forecast[key], [h["rmse"][index] for h in records])


def group(data, homes):
    require(data["home_ids"] == homes, "group cohort mismatch")
    require([row["id"] for row in data["rows"]] == list(METHODS), "incomplete methods")
    require(data["oracle"]["id"] == "oracle", "missing oracle")
    for row in data["rows"]:
        policy(row, homes)
        forecasts(row, homes)
    oracle = data["oracle"]
    policy(oracle, homes)
    lower, upper = oracle["bound"]["lower"], oracle["bound"]["upper"]
    require(finite(lower) and finite(upper) and lower <= upper + 1e-8, "invalid oracle bounds")
    require(
        math.isclose(
            oracle["policy"]["metrics"]["objective"]["mean"], upper, rel_tol=1e-6, abs_tol=1e-4
        ),
        "oracle replay differs from upper bound",
    )
    require(
        all(row["policy"]["metrics"]["objective"]["mean"] >= lower - 1e-5 for row in data["rows"]),
        "policy below perfect-future lower bound",
    )


def validate_evidence(data):
    """Validate complete five-month/25-home public export; return unchanged data."""
    try:
        clean(data)
        require(data["schema_version"] == 2, "wrong schema version")
        homes = data["home_ids"]
        require(
            len(homes) == 25
            and len(set(homes)) == 25
            and all(type(h) is int for h in homes)
            and homes == sorted(homes),
            "expected25homes",
        )
        require(data["protocol"]["home_ids"] == homes, "protocol cohort mismatch")
        require(data["protocol"]["methods"] == list(METHODS), "protocol methods mismatch")
        require(data["protocol"]["seed"] == 41, "unexpected seed")
        require([fold["id"] for fold in data["folds"]] == list(MONTHS), "incomplete months")
        sha(data["frozen_selection_sha256"])
        require(
            datetime.fromisoformat(data["frozen_at_utc"].replace("Z", "+00:00")).utcoffset()
            is not None,
            "freeze timestamp lacks timezone",
        )
        require(bool(data["numerical_source"]), "missing numerical source receipts")
        for value in data["numerical_source"].values():
            sha(value)
        group(data, homes)
        dates = []
        for fold in data["folds"]:
            month = int(fold["id"][-2:])
            expected = [
                f"{fold['id']}-{day:02d}"
                for day in range(1, calendar.monthrange(2019, month)[1] + 1)
                if f"{fold['id']}-{day:02d}" != "2019-07-29"
            ]
            require(fold["dates"] == expected, "incomplete common test dates")
            dates.extend(expected)
            group(fold, homes)
            oracle = fold["oracle"]
            sha(oracle["protocol_sha256"])
            require(
                [r["date"] for r in oracle["receipts"]] == expected, "missing oracle daily receipts"
            )
            for receipt in oracle["receipts"]:
                sha(receipt["file_sha256"])
            bounds = oracle["bound"]
            require(
                len(bounds["daily_lower"]) == len(expected) == len(bounds["daily_upper"]),
                "incomplete daily oracle bounds",
            )
            require(
                all(
                    finite(lo) and finite(hi) and lo <= hi + 1e-8
                    for lo, hi in zip(bounds["daily_lower"], bounds["daily_upper"], strict=True)
                ),
                "invalid daily oracle bounds",
            )
            for side in ("lower", "upper"):
                require(
                    close(bounds[side], statistics.mean(bounds["daily_" + side])),
                    "oracle bound aggregation mismatch",
                )
            for row in fold["rows"]:
                sha(row["checkpoint_sha256"])
                sha(row["evaluation_sha256"])
                require(
                    type(row["selected_episodes"]) is int
                    and 0
                    <= row["selected_episodes"]
                    <= row["selection_completed_episode"]
                    <= row["selection_budget_cap"],
                    "invalid selected training budget",
                )
                require(isinstance(row["selection_convergence"], dict), "missing stopping receipt")
                audit = row["audit"]
                require(
                    audit["split"] == "test"
                    and audit["dates"] == expected
                    and audit["checkpoint_sha256"] == row["checkpoint_sha256"]
                    and audit["checkpoint_episode"] == row["selected_episodes"]
                    and audit["reference_commit"] == oracle["reference_commit"]
                    and audit["matched_policy_transitions"] == len(expected) * 25 * 24
                    and finite(audit["max_absolute_daily_difference"])
                    and 0 <= audit["max_absolute_daily_difference"] <= 1e-8,
                    "invalid independent physics replay receipt",
                )
        require(data["dates"] == dates, "pooled dates mismatch")
        for index, row in enumerate(data["rows"] + [data["oracle"]]):
            monthly = [(fold["rows"] + [fold["oracle"]])[index] for fold in data["folds"]]
            for home_index, home in enumerate(row["policy"]["per_home"]):
                for key in METRICS:
                    expected = sum(
                        r["policy"]["per_home"][home_index][key] * len(f["dates"])
                        for r, f in zip(monthly, data["folds"], strict=True)
                    ) / len(dates)
                    require(close(home[key], expected), "pooled household day weights mismatch")
            if row["id"] not in ("history", "oracle"):
                for home_index, home in enumerate(row["forecast"]["per_home"]):
                    records = [r["forecast"]["per_home"][home_index] for r in monthly]
                    count = sum(r["queries"] for r in records)
                    require(home["queries"] == count, "pooled forecast query count mismatch")
                    for target in range(3):
                        expected = math.sqrt(
                            sum(r["rmse"][target] ** 2 * r["queries"] for r in records) / count
                        )
                        require(close(home["rmse"][target], expected), "pooled RMSE mismatch")
        for side in ("lower", "upper"):
            expected = sum(
                f["oracle"]["bound"][side] * len(f["dates"]) for f in data["folds"]
            ) / len(dates)
            require(close(data["oracle"]["bound"][side], expected), "pooled oracle bound mismatch")
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError(
            f"Invalid seasonal evidence: missing or malformed record ({error})"
        ) from error
    return data


def load_evidence(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Complete seasonal evidence is required: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read seasonal evidence: {path}") from error
    return validate_evidence(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("site/performance.json"))
    args = parser.parse_args()
    data = load_evidence(args.path)
    print(
        f"Verified seasonal export: 6 methods + oracle, 25 homes, {len(data['dates'])} days, 5 months."
    )


if __name__ == "__main__":
    main()
