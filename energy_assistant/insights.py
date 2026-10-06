"""Grounded household questions: arithmetic, matched comparisons and dated outlooks."""

from .presentation import LABELS

DEVICES = {
    "everyday": "Everyday use",
    "cooling": "Cooling",
    "car": "Car charging",
    "laundry": "Laundry",
}
ACTION_COLUMNS = {
    "cooling": "ac_kw",
    "car": "ev_kw",
    "laundry": "washer_kw",
    "battery": "battery_kw",
}


def breakdown(service, home, dates):
    record = service.data["homes"][str(home)]
    bill = record.get("history_bill", record["bill"])
    days = [next(d for d in bill["days"] if d["date"] == date) for date in dates]
    hours = [r for d in days for r in d["hours"]]
    if any("devices_gross_cost" not in r for r in hours):
        return {
            "available": False,
            "note": "Appliance readings were not included in this saved report. Prepare the household again to add a breakdown.",
        }
    devices = [
        {
            "id": key,
            "label": label,
            "kwh": sum(r["devices_kwh"][key] for r in hours),
            "gross_cost": sum(r["devices_gross_cost"][key] for r in hours),
        }
        for key, label in DEVICES.items()
    ]
    adjustments = [
        {"label": "Solar used at home", "cost": -sum(r["solar_used_credit"] for r in hours)},
        {"label": "Solar exported", "cost": -sum(r["export_credit"] for r in hours)},
        {"label": "Fixed charges", "cost": sum(r["fixed_cost"] for r in hours)},
    ]
    total = sum(d["total"] for d in days)
    if (
        abs(sum(d["gross_cost"] for d in devices) + sum(a["cost"] for a in adjustments) - total)
        > 1e-6
    ):
        raise ValueError("The appliance breakdown does not reconcile with the bill")
    return {
        "available": True,
        "evidence": "recorded",
        "devices": devices,
        "exchange": None
        if any("import_kwh" not in r or "export_kwh" not in r for r in hours)
        else {
            "evidence": "recorded",
            "import_kwh": sum(r["import_kwh"] for r in hours),
            "export_kwh": sum(r["export_kwh"] for r in hours),
            "export_credit": sum(r["export_credit"] for r in hours),
            "hours": hours
            if len(days) == 1
            else [
                {
                    "hour": d["date"],
                    "import_kwh": sum(r["import_kwh"] for r in d["hours"]),
                    "export_kwh": sum(r["export_kwh"] for r in d["hours"]),
                }
                for d in days
            ],
        },
        "adjustments": adjustments,
        "total": total,
        "days": len(dates),
        "dates": dates,
        "note": "Appliances are priced at the grid tariff before solar. Shared solar savings and export credits are subtracted once, separately. This is cost accounting, not a claim that switching a device off would save exactly that amount.",
    }


def passes(candidate, baseline):
    return (
        candidate["ev_complete"]
        and candidate["washer_complete"]
        and candidate["comfort_pct"] >= baseline["comfort_pct"] - 1e-6
        and candidate["degree_hours"] <= baseline["degree_hours"] + 1e-6
        and candidate["terminal_battery_soc"] >= baseline["terminal_battery_soc"] - 1e-6
        and not candidate["solver_failures"]
    )


def schedule_comparison(service, home, dates, reference=None):
    days = [service.data["homes"][str(home)]["dates"][date] for date in dates]
    common = set.intersection(*(set(d["scenarios"]) for d in days))
    reference = reference or ("personalized" if "personalized" in common else "feedback")
    if reference not in common:
        raise ValueError("This plan is unavailable for the selected period")
    costs = {name: sum(d["scenarios"][name]["bill"] for d in days) for name in common}
    eligible = [
        name
        for name in common
        if name != reference
        and costs[name] < costs[reference] - 1e-6
        and all(passes(d["scenarios"][name], d["scenarios"][reference]) for d in days)
    ]
    best = min(eligible, key=lambda name: (costs[name], name)) if eligible else None
    alternative = best or reference

    def plan(name):
        rows = [d["scenarios"][name] for d in days]
        return {
            "id": name,
            "label": LABELS[name],
            "cost": costs[name],
            "comfort_pct": sum(r["comfort_pct"] for r in rows) / len(rows),
            "degree_hours": sum(r["degree_hours"] for r in rows),
            "days": [
                {
                    "date": date,
                    "actions": row["actions"],
                    "ledger": row["ledger"],
                    "bill": row["bill"],
                    "comfort_pct": row["comfort_pct"],
                    "terminal_battery_soc": row["terminal_battery_soc"],
                }
                for date, row in zip(dates, rows, strict=True)
            ],
        }

    record = service.data["homes"][str(home)]
    recorded_days = []
    for date in dates:
        day = next(d for d in record["bill"]["days"] if d["date"] == date)
        if all("devices_kwh" in r for r in day["hours"]):
            recorded_days.append(
                {
                    "date": date,
                    "bill": day["total"],
                    "actions": [
                        {
                            "hour": r["hour"],
                            "ac_kw": r["devices_kwh"]["cooling"],
                            "ev_kw": r["devices_kwh"]["car"],
                            "washer_kw": r["devices_kwh"]["laundry"],
                        }
                        for r in day["hours"]
                    ],
                }
            )
    return {
        "evidence": "simulated",
        "reference": plan(reference),
        "alternative": plan(alternative),
        "improvement_found": best is not None,
        "savings": costs[reference] - costs[alternative],
        "days": len(dates),
        "dates": dates,
        "recorded_days": recorded_days,
        "note": "Both columns are simulations, not observations of what you actually did. Costs use the prices observed on each date. The alternative is the cheapest tested plan passing every daily comfort, charging, laundry, battery and solver check against the selected plan. It is not a future-knowing optimum. Days are tested separately.",
    }


def day_outlook(service, home, dates, reference=None, hour=6):
    record = service.data["homes"][str(home)]
    days = []
    for date in dates:
        plans = record["dates"][date]["scenarios"]
        plan = reference or ("personalized" if "personalized" in plans else "feedback")
        outlook = plans.get(plan, {}).get("outlooks", {}).get(str(hour))
        if outlook is None:
            return {
                "available": False,
                "note": "This saved model report has no forecast-based schedule for that time. Prepare it again to add morning and afternoon planning. Historical comparisons remain available.",
            }
        ledger = outlook.get("ledger")
        exchange = (
            None
            if ledger is None
            else {
                "import_kwh": sum(r["import_kwh"] for r in ledger),
                "export_kwh": sum(r["export_kwh"] for r in ledger),
                "export_credit": sum(r["export_credit"] for r in ledger),
                "hours": ledger,
            }
        )
        days.append({"date": date, **outlook, "exchange": exchange})
    return {
        "available": True,
        "evidence": "predicted",
        "days": days,
        "home": str(home),
        "trading": {
            "peer_enabled": False,
            "note": "This single-home model exports to the grid. Neighbour-to-neighbour trades are not simulated, and no live offers or transactions are available.",
        },
        "origin_hour": hour,
        "reference_label": LABELS[plan],
        "forecast_horizon_hours": record.get("forecast_context", {}).get("horizon", 23),
        "note": "An outlook from the selected date and time, not a live connection to your home. After that time the calculation uses TabPFN forecasts, declared appliance needs and a simulated starting state. Prices hold the last observed base rate plus the known time-of-use tariff; they are not TabPFN price predictions. Battery charge and room temperature come from device physics. Actual conditions and costs can differ; these schedules are not commands or proven optimal plans.",
    }
