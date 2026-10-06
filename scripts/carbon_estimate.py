"""Export an explicitly assumed grid-electricity footprint from frozen evaluations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from gridpfn.paths import ROOT
from gridpfn.release_evidence import load_evidence

METHODS = ("tabpfn", "tabfm", "tabicl")
FACTOR = {
    "region": "Netherlands (assumed scenario)",
    "year": 2019,
    "kg_co2_per_kwh": 0.37,
    "method": "CBS integral method, electricity delivered to the user; no extra loss multiplier",
    "source": "https://www.cbs.nl/nl-nl/achtergrond/2024/51/rendementen-en-co2-emissie-van-elektriciteitsproductie-in-nederland-update-2023",
    "table": "https://www.cbs.nl/-/media/_excel/2024/51/co2-emissie_energieverbruik_rendementen_elektriciteit_2023.xls",
}


def footprint_difference(rows, factor=FACTOR["kg_co2_per_kwh"]):
    """Positive means lower footprint; retain negative results without clipping."""
    imported = {row["id"]: row["import_kwh"] for row in rows}
    saving = (imported["tabfm"] + imported["tabicl"]) / 2 - imported["tabpfn"]
    return {"import_reduction_kwh": saving, "estimated_co2_reduction_kg": saving * factor}


def validate_carbon(data, evidence, performance_bytes):
    if data.get("schema_version") != 1 or data.get("factor") != FACTOR:
        raise ValueError("Unexpected carbon scenario")
    if data.get("performance_sha256") != hashlib.sha256(performance_bytes).hexdigest():
        raise ValueError("Carbon estimate belongs to different performance evidence")
    periods = data.get("periods", [])
    if [p["id"] for p in periods] != [f["id"] for f in evidence["folds"]]:
        raise ValueError("Carbon periods differ from the evaluated months")
    for period, fold in zip(periods, evidence["folds"], strict=True):
        expected = {row["id"]: row for row in fold["rows"]}
        if period.get("dates") != fold["dates"]:
            raise ValueError("Carbon period dates differ")
        if [row["id"] for row in period["rows"]] != list(METHODS):
            raise ValueError("Incomplete carbon comparison")
        for row in period["rows"]:
            if row.get("evaluation_sha256") != expected[row["id"]]["evaluation_sha256"]:
                raise ValueError("Carbon evaluation identity differs")
            if (
                not isinstance(row.get("import_kwh"), (int, float))
                or not math.isfinite(row["import_kwh"])
                or row["import_kwh"] < 0
            ):
                raise ValueError("Invalid grid import")
    return data


def load_carbon(path=ROOT / "site/carbon.json", performance=ROOT / "site/performance.json"):
    evidence = load_evidence(performance)
    return validate_carbon(
        json.loads(Path(path).read_text()), evidence, Path(performance).read_bytes()
    )


def estimate_for(data, period):
    if period == "pooled":
        rows = [
            {
                "id": method,
                "import_kwh": sum(
                    next(row["import_kwh"] for row in group["rows"] if row["id"] == method)
                    for group in data["periods"]
                ),
            }
            for method in METHODS
        ]
    else:
        rows = next(group["rows"] for group in data["periods"] if group["id"] == period)
    return footprint_difference(rows, data["factor"]["kg_co2_per_kwh"])


def build(study, performance=ROOT / "site/performance.json"):
    evidence = load_evidence(performance)
    output = {
        "schema_version": 1,
        "performance_sha256": hashlib.sha256(Path(performance).read_bytes()).hexdigest(),
        "factor": FACTOR,
        "interpretation": "Estimated grid-electricity footprint difference, not measured or marginal avoided emissions. The 2019 Netherlands annual-average delivered-electricity factor is assumed for every method and period. Original energy traces and dollar tariffs are unchanged; this is not a Dutch deployment. No emissions credit for exports, no embodied or model-compute emissions.",
        "periods": [],
    }
    for fold in evidence["folds"]:
        indexed = {row["id"]: row for row in fold["rows"]}
        group = {"id": fold["id"], "dates": fold["dates"], "rows": []}
        for method in METHODS:
            path = (
                Path(study)
                / "folds"
                / fold["id"]
                / "refit/policies"
                / method
                / "evaluation/test_latest.json"
            )
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != indexed[method]["evaluation_sha256"]:
                raise ValueError(f"Changed frozen evaluation: {fold['id']}/{method}")
            record = json.loads(raw)
            if (
                record["dates"] != fold["dates"]
                or [h["home_id"] for h in record["homes"]] != evidence["home_ids"]
            ):
                raise ValueError("Evaluation cohort or dates differ")
            days = record["day_records"]
            if len(days) != len(evidence["home_ids"]) or any(
                len(h) != len(fold["dates"]) for h in days
            ):
                raise ValueError("Incomplete daily energy accounting")
            imported = 0.0
            for household in days:
                for day in household:
                    if abs(day["p2p_kwh"]) > 1e-10:
                        raise ValueError("This export requires grid-only settlement")
                    if not math.isfinite(day["import"]) or day["import"] < 0:
                        raise ValueError("Invalid daily electricity import")
                    if not math.isclose(
                        day["import"] - day["export"], day["net_demand"], abs_tol=1e-7
                    ):
                        raise ValueError("Daily energy balance differs")
                    imported += day["import"]
            if not math.isclose(
                imported / len(days) / len(fold["dates"]), record["import"], abs_tol=1e-7
            ):
                raise ValueError("Daily imports do not reconcile with evaluation")
            group["rows"].append(
                {"id": method, "import_kwh": imported, "evaluation_sha256": digest}
            )
        output["periods"].append(group)
    return validate_carbon(output, evidence, Path(performance).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "site/carbon.json")
    args = parser.parse_args()
    result = build(args.study)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(estimate_for(result, "2019-06"), indent=2))
