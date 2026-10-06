"""Calendar summaries over available days, with explicit gaps and fair comparisons."""

from copy import deepcopy
from datetime import date as calendar_date

from .presentation import LABELS


class PeriodService:
    """A request-scoped view; never changes the underlying evidence or caches."""

    def __init__(self, service, start, end, plan=None, hour=6):
        self.service, self.start, self.end = service, start, end or start
        self.plan, self.hour = plan, hour
        if plan not in {None, "personalized", "feedback", "seasonal", "tabpfn"} or hour not in {
            0,
            6,
            12,
            18,
        }:
            raise ValueError("Choose an available energy plan and planning time")
        first, last = calendar_date.fromisoformat(start), calendar_date.fromisoformat(self.end)
        if last < first or (last - first).days > 365:
            raise ValueError("Choose a period of up to one year, with the start before the end")
        self.data, self.sha256 = service.data, service.sha256

    def dates(self, home):
        record = self.data["homes"].get(str(home))
        if record is None:
            raise ValueError("Choose an available home")
        dates = sorted(
            d
            for d in set(record["dates"]) | set(record.get("history_phase", {}))
            if self.start <= d <= self.end
        )
        if not dates:
            raise ValueError("There are no results for this home in the selected period")
        return dates

    def select(self, home, _date):
        return self.service.select(home, self.dates(home)[0])

    def coverage(self, home):
        dates = self.dates(home)
        requested = (
            calendar_date.fromisoformat(self.end) - calendar_date.fromisoformat(self.start)
        ).days + 1
        return {
            "start": self.start,
            "end": self.end,
            "available_days": len(dates),
            "requested_days": requested,
            "dates": dates,
            "complete": len(dates) == requested,
        }

    def call(self, name, home, _date):
        dates = self.dates(home)
        coverage = self.coverage(home)
        if name not in {"inspect_home", "explain_bill", "appliance_breakdown", "export_plan"}:
            dates = [d for d in dates if d in self.data["homes"][str(home)]["dates"]]
            if not dates:
                return self.service.model_unavailable()
            coverage.update(
                available_days=len(dates),
                dates=dates,
                complete=len(dates) == coverage["requested_days"],
            )
        if name in {"appliance_breakdown", "schedule_comparison", "plan_day"}:
            from .insights import breakdown, day_outlook, schedule_comparison

            if name == "appliance_breakdown":
                result = breakdown(self.service, home, dates)
            elif name == "schedule_comparison":
                result = schedule_comparison(self.service, home, dates, self.plan)
            else:
                result = day_outlook(self.service, home, dates, self.plan, self.hour)
            return {**result, "home": str(home), "date": self.start, "coverage": coverage}
        if self.start == self.end:
            result = self.service.call(name, home, dates[0])
            if name == "export_plan":
                result.update(
                    {
                        tool: self.call(tool, home, self.start)
                        for tool in ("appliance_breakdown", "schedule_comparison", "plan_day")
                    }
                )
                result.update(selected_plan=self.plan, planning_hour=self.hour)
            return result
        if name == "inspect_home":
            result = self.service.inspect_home(home, dates[0])
        elif name == "explain_bill":
            results = [self.service.explain_bill(home, date) for date in dates]
            result = deepcopy(results[0])
            result["total"] = sum(r["total"] for r in results)
            result["daily_costs"] = [{"date": r["date"], "cost": r["total"]} for r in results]
            result["largest_day"] = max(result["daily_costs"], key=lambda r: r["cost"])
            result["period_days"] = len(dates)
            result["period_start"], result["period_end"] = self.start, self.end
            result["period"] = {
                "period_total": result["total"],
                **{
                    key: sum(row[key] for r in results for row in r["hours"])
                    for key in ("import_cost", "export_credit", "fixed_cost")
                },
            }
            result["hours"] = []  # Do not disguise a daily series as an hourly one.
        elif name == "forecast_and_explain":
            # Summaries use already-computed daily predictions, never 31 new live fits.
            results = [self.data["homes"][str(home)]["dates"][d]["forecast"] for d in dates]
            result = deepcopy(results[0])
            result["model"] = self.data["forecaster"]
            result["rows"] = [
                {
                    "hour": row["hour"],
                    **{
                        key: sum(r["rows"][i][key] for r in results) / len(results)
                        for key in row
                        if key != "hour"
                    },
                }
                for i, row in enumerate(result["rows"])
            ]
            if all(r["explanation"] for r in results):
                explanations = [r["explanation"] for r in results]
                result["explanation"] = deepcopy(explanations[0])
                for key in ("baseline_kwh", "prediction_kwh"):
                    result["explanation"][key] = sum(e[key] for e in explanations) / len(
                        explanations
                    )
                for i, row in enumerate(result["explanation"]["contributions"]):
                    row["kwh"] = sum(e["contributions"][i]["kwh"] for e in explanations) / len(
                        explanations
                    )
                e = result["explanation"]
                e["additivity_error"] = abs(
                    e["baseline_kwh"]
                    + sum(r["kwh"] for r in e["contributions"])
                    - e["prediction_kwh"]
                )
                e["model_queries"] = sum(e["model_queries"] for e in explanations)
                e["method"] = "Average of the separate daily grouped Shapley explanations"
            else:
                result["explanation"] = None
            result.pop("retrospective_metrics", None)
            result.pop("seasonal_metrics", None)
            result["note"] = (
                "Average of previously computed daily forecasts. Each uses only that day's readings through the forecast time. This is not a forecast of one continuous future period."
            )
        elif name == "compare_schedules":
            results = [self.service.compare_schedules(home, date) for date in dates]
            identifiers = set.intersection(
                *(set(r["id"] for r in day["candidates"]) for day in results)
            )
            candidates = []
            for name_id in [r["id"] for r in results[0]["candidates"] if r["id"] in identifiers]:
                rows = [next(r for r in day["candidates"] if r["id"] == name_id) for day in results]
                candidates.append(
                    {
                        "id": name_id,
                        "label": LABELS[name_id],
                        "bill": sum(r["bill"] for r in rows),
                        "difference_from_feedback": sum(
                            r["difference_from_feedback"] for r in rows
                        ),
                        "comfort_pct": sum(r["comfort_pct"] for r in rows) / len(rows),
                        "degree_hours": sum(r["degree_hours"] for r in rows),
                        "ev_complete": all(r["ev_complete"] for r in rows),
                        "washer_complete": all(r["washer_complete"] for r in rows),
                        "terminal_battery_soc": sum(r["terminal_battery_soc"] for r in rows)
                        / len(rows),
                        "eligible": all(r["eligible"] for r in rows),
                        "reasons": sorted(set(reason for r in rows for reason in r["reasons"])),
                        "days_failing_checks": sum(not r["eligible"] for r in rows),
                        "solver_failures": sum(r["solver_failures"] for r in rows),
                        "actions": [],
                        "daily_costs": [
                            {"date": date, "cost": r["bill"]}
                            for date, r in zip(dates, rows, strict=True)
                        ],
                    }
                )
            eligible = [
                r
                for r in candidates
                if r["id"] != "feedback" and r["eligible"] and r["difference_from_feedback"] > 1e-6
            ]
            best = min(eligible, key=lambda r: r["bill"])["id"] if eligible else None
            result = {
                "candidates": candidates,
                "recommended": best,
                "evidence": "simulated",
                "decision": "A lower-cost plan meets every daily comparison check."
                if best
                else "No lower-cost plan passes the checks on every available day.",
                "limits": results[0]["limits"],
                "interpretation": "Each day is tested separately with its own initial battery state. These totals do not represent continuous month-long operation. The same plan is compared over the same days; we do not pick a different winner after seeing each day's results.",
            }
        elif name == "export_plan":
            record, _ = self.select(home, self.start)
            result = {
                "schema_version": 1,
                "evidence_sha256": self.sha256,
                "model": self.data["model"],
                "provenance": self.data["provenance"],
                "source_sha256": self.data.get("source_sha256", {}),
                "input_sha256": self.data.get("input_sha256", {}),
                "bill": self.call("explain_bill", home, self.start),
                "forecast": self.call("forecast_and_explain", home, self.start),
                "comparison": self.call("compare_schedules", home, self.start),
                "checkpoint": record.get("checkpoint"),
                "actuation": False,
                "selected_plan": self.plan,
                "planning_hour": self.hour,
                **{
                    tool: self.call(tool, home, self.start)
                    for tool in ("appliance_breakdown", "schedule_comparison", "plan_day")
                },
                "note": "Review only. No device settings have been changed.",
            }
        else:
            raise ValueError("Unknown energy tool")
        return {**result, "home": str(home), "date": self.start, "period_selection": coverage}
