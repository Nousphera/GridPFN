"""Read-only domain tools; every displayed number comes from an evidence bundle."""

import hashlib
import json
from pathlib import Path

LABELS = {
    "feedback": "Comfort feedback",
    "seasonal": "Seasonal forecast + scheduler",
    "tabpfn": "TabPFN-3.5 + scheduler",
    "personalized": "Your trained controller",
}


class EvidenceService:
    def __init__(self, directory, live=False):
        self.directory = Path(directory).resolve()
        raw = (self.directory / "evidence.json").read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if (self.directory / "evidence.sha256").read_text().strip() != self.sha256:
            raise ValueError("Evidence checksum mismatch; rebuild the bundle")
        self.data = json.loads(
            raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
        )
        if self.data.get("schema_version") != 1 or not self.data.get("homes"):
            raise ValueError("Unsupported or empty evidence bundle")
        self.live = None
        if live:
            from .live import LiveForecast

            if self.data["model"]["backend"] != "tabpfn" or any(
                "live_context_sha256" not in r for r in self.data["homes"].values()
            ):
                raise ValueError("Prepare a current TabPFN bundle before enabling live inference")
            self.live = LiveForecast(self)

    def select(self, home, date):
        record = self.data["homes"].get(str(home))
        if record is None or (
            date not in record["dates"] and date not in record.get("history_phase", {})
        ):
            raise ValueError("Choose a household and replay date available in this bundle")
        return record, record["dates"].get(date)

    def bind_home(self, home=None):
        """Restrict an application instance to one household, including its API."""
        if home is None:
            if len(self.data["homes"]) != 1:
                raise ValueError("Choose this app's home once at launch with --home HOME_ID")
            home = next(iter(self.data["homes"]))
        if str(home) not in self.data["homes"]:
            raise ValueError("The selected model does not contain this home")
        self.data["homes"] = {str(home): self.data["homes"][str(home)]}
        return str(home)

    def catalog(self):
        return {
            "homes": [
                {
                    "id": h,
                    "dates": sorted(set(r["dates"]) | set(r.get("history_phase", {}))),
                    "model_dates": sorted(r["dates"]),
                    "personalized": bool(r.get("checkpoint")),
                }
                for h, r in self.data["homes"].items()
            ],
            "forecaster": self.data["forecaster"],
            "provenance": self.data["provenance"],
            "execution": self.data["execution"],
            "live_forecast": self.live is not None,
            "created_at": self.data["created_at"],
            "sha256": self.sha256,
            "currency": self.data["currency"],
        }

    def overview(self, home):
        """Compact daily cash/comfort pairs from the same simulated controller."""
        from math import ceil

        from .presentation import LABELS as FRIENDLY_LABELS

        record = self.data["homes"].get(str(home))
        if record is None:
            raise ValueError("This home is unavailable")
        common = set.intersection(*(set(day["scenarios"]) for day in record["dates"].values()))
        preferred = [p for p in ("personalized", "tabpfn", "seasonal", "feedback") if p in common]
        overview = {
            "home": str(home),
            "currency": self.data["currency"],
            "evidence": "simulated",
            "default_plan": preferred[0],
            "comfort_scale": "The colour combines how far and how long room temperature is outside its target range. The same scale is used for every day and plan; it saturates at 24 degree-hours.",
            "plans": [
                {
                    "id": plan,
                    "label": FRIENDLY_LABELS[plan],
                    "days": [
                        {
                            "date": date,
                            "net_cash": -float(day["scenarios"][plan]["bill"]),
                            "comfort_pct": day["scenarios"][plan]["comfort_pct"],
                            "degree_hours": day["scenarios"][plan]["degree_hours"],
                            "comfort_band": min(
                                10, max(0, ceil(day["scenarios"][plan]["degree_hours"] / 2.4))
                            ),
                        }
                        for date, day in sorted(record["dates"].items())
                    ],
                }
                for plan in preferred
            ],
        }
        if record.get("history_bill"):
            overview["plans"].append(
                {
                    "id": "recorded",
                    "label": "Recorded use",
                    "days": [
                        {
                            "date": d["date"],
                            "net_cash": -d["total"],
                            "comfort_pct": None,
                            "degree_hours": None,
                            "comfort_band": None,
                            "phase": record["history_phase"][d["date"]],
                        }
                        for d in record["history_bill"]["days"]
                    ],
                }
            )
        return overview

    def inspect_home(self, home, date):
        record, _ = self.select(home, date)
        return {
            "home": home,
            "date": date,
            "evidence": "recorded",
            "data_kind": self.data["provenance"]["kind"],
            "training_days": record["training_days"],
            "training_period": [record["train_start"], record["train_end"]],
            "forecaster": self.data["forecaster"],
            "model": self.data["model"],
            "personalized_policy": record.get("checkpoint"),
            "execution": self.data["execution"],
            "devices": ["Cooling", "Electric vehicle", "Washing machine", "Home battery", "Solar"],
            "limits": "Research simulator, not a connected home. No device commands. No building-retrofit or insulation estimates. No guaranteed savings.",
        }

    def explain_bill(self, home, date):
        record, _ = self.select(home, date)
        bill = record.get("history_bill", record["bill"])
        day = next(d for d in bill["days"] if d["date"] == date)
        peak = max(day["hours"], key=lambda r: r["net_cost"])
        return {
            "home": home,
            "date": date,
            "evidence": "recorded",
            "currency": "USD",
            "total": day["total"],
            "hours": day["hours"],
            "highest_cost_hour": peak,
            "period": {
                "period_total":day["total"],
                **{key:sum(row[key] for row in day["hours"]) for key in ("import_cost", "export_credit", "fixed_cost")},
            },
            "period_days": 1,
            "period_start": date,
            "period_end": date,
            "formula": "Import kWh × import tariff − export kWh × export tariff + fixed charge",
            "note": bill["note"],
            "data_kind": self.data["provenance"]["kind"],
        }

    def forecast_and_explain(self, home, date):
        _, day = self.select(home, date)
        if day is None:
            return self.model_unavailable()
        if self.live:
            return self.live(home, date)
        _, day = self.select(home, date)
        return {"home": home, "date": date, "model": self.data["forecaster"], **day["forecast"]}

    def compare_schedules(self, home, date):
        _, day = self.select(home, date)
        if day is None:
            return self.model_unavailable()
        baseline = day["scenarios"]["feedback"]
        candidates = []
        for name, result in day["scenarios"].items():
            reasons = []
            if not result["ev_complete"] or not result["washer_complete"]:
                reasons.append("An appliance service requirement is unmet")
            if (
                result["comfort_pct"] < baseline["comfort_pct"] - 1e-6
                or result["degree_hours"] > baseline["degree_hours"] + 1e-6
            ):
                reasons.append("Comfort is worse than the matched feedback baseline")
            if result["terminal_battery_soc"] < baseline["terminal_battery_soc"] - 1e-6:
                reasons.append("Ends with less stored battery energy than the baseline")
            if result["solver_failures"]:
                reasons.append("Scheduler fell back after a failed solve")
            candidates.append(
                {
                    "id": name,
                    "label": LABELS[name],
                    **result,
                    "difference_from_feedback": baseline["bill"] - result["bill"],
                    "eligible": not reasons,
                    "reasons": reasons,
                }
            )
        eligible = [
            r
            for r in candidates
            if r["eligible"] and r["id"] != "feedback" and r["bill"] < baseline["bill"] - 1e-6
        ]
        best = min(eligible, key=lambda r: r["bill"])["id"] if eligible else None
        return {
            "home": home,
            "date": date,
            "evidence": "simulated",
            "candidates": candidates,
            "recommended": best,
            "decision": "Lower-cost candidate passes the declared comparison checks."
            if best
            else "No cheaper alternative passes every comparison check. Keep the baseline for this replay.",
            "limits": baseline["limits"],
            "interpretation": "Retrospective, single-day evidence. No monthly extrapolation or real-home savings claim. A baseline-relative comfort check is not a guarantee of absolute comfort.",
        }

    @staticmethod
    def model_unavailable():
        return {
            "available": False,
            "evidence": "recorded",
            "note": "Recorded costs are available for these dates, but no prepared model replay covers them. Choose a model date for forecasts and plan comparisons. Training history is not held-out evaluation.",
        }

    def export_plan(self, home, date):
        record, _ = self.select(home, date)
        return {
            "schema_version": 1,
            "home": home,
            "date": date,
            "evidence_sha256": self.sha256,
            "model": self.data["model"],
            "created_at": self.data["created_at"],
            "source_sha256": self.data.get("source_sha256", {}),
            "input_sha256": self.data.get("input_sha256", {}),
            "live_context_sha256": record.get("live_context_sha256"),
            "strategy": self.data.get("strategy", {}),
            "provenance": self.data["provenance"],
            "bill": self.explain_bill(home, date),
            "forecast": self.forecast_and_explain(home, date),
            "comparison": self.compare_schedules(home, date),
            "checkpoint": record.get("checkpoint"),
            "forecast_context": record.get("forecast_context"),
            "date_role":record.get("history_phase",{}).get(date),
            "actuation": False,
            "note": "Analysis report only. These schedules are historical simulation results, not instructions to home devices.",
        }

    def call(self, name, home, date):
        if name not in TOOL_NAMES:
            raise ValueError("Unknown energy tool")
        if name in {"appliance_breakdown", "schedule_comparison", "plan_day"}:
            from .insights import breakdown, day_outlook, schedule_comparison

            _, day = self.select(home, date)
            if day is None and name != "appliance_breakdown":
                return self.model_unavailable()
            return {
                "home": str(home),
                "date": date,
                **{
                    "appliance_breakdown": breakdown,
                    "schedule_comparison": schedule_comparison,
                    "plan_day": day_outlook,
                }[name](self, home, [date]),
            }
        return getattr(self, name)(home, date)


TOOL_NAMES = (
    "inspect_home",
    "explain_bill",
    "forecast_and_explain",
    "compare_schedules",
    "export_plan",
    "appliance_breakdown",
    "schedule_comparison",
    "plan_day",
)


def narrative(name, result):
    """Plain-language answer, with numbers taken only from the tool result."""
    from .presentation import presentation

    return presentation(name, result)["summary"]
