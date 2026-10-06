"""Plain-language views shared by the chat interface and MCP tools.

Calculations and decisions remain in the evidence service. This module changes
their presentation only; optional technical details always retain the source.
"""

FEATURE_NAMES = {
    "Recent household demand": "Recent electricity use",
    "Recent solar generation": "Recent solar generation",
    "Weather and current tariff": "Weather and electricity prices",
    "Time and forecast horizon": "Time of day and how far ahead we look",
}

REASONS = {
    "An appliance service requirement is unmet": "Charging or laundry would not finish.",
    "Comfort is worse than the matched feedback baseline": "Room temperatures are less comfortable.",
    "Ends with less stored battery energy than the baseline": "It leaves less energy in the battery.",
    "Scheduler fell back after a failed solve": "We could not finish calculating this schedule.",
}

LABELS = {
    "feedback": "Comfort-first plan",
    "seasonal": "Daily-pattern plan",
    "tabpfn": "TabPFN plan",
    "personalized": "Personalized plan",
}


def explanation_factors(explanation):
    if not explanation:
        return []
    return [
        {"label": FEATURE_NAMES.get(row["feature"], row["feature"]), "change_kwh": row["kwh"]}
        for row in explanation["contributions"]
    ]


def presentation(name, result):
    """A concise answer plus the minimum context needed to interpret it correctly."""
    if result.get("available") is False:
        return {
            "title": "About these dates",
            "summary": result["note"],
            "note": "Recorded readings remain available in the calendar.",
        }
    if name in {"appliance_breakdown", "schedule_comparison", "plan_day"}:
        return insight_presentation(name, result)
    if "period_selection" in result:
        return period_presentation(name, result)
    if name == "explain_bill":
        peak = result["highest_cost_hour"]
        return {
            "title": "Your electricity costs",
            "summary": (
                f"Electricity for this day comes to ${result['total']:.2f}. "
                f"The most expensive hour was {peak['hour']:02}:00–{peak['hour'] + 1:02}:00, "
                f"costing ${peak['net_cost']:.2f}. "
            ),
            "note": "Calculated from the home's energy readings and prices. Your utility's actual bill may include other charges.",
        }
    if name == "forecast_and_explain":
        explanation = result["explanation"]
        if not explanation:
            return {
                "title": "Expected electricity use",
                "summary": "This example uses a simple forecast based on past daily patterns. A TabPFN explanation is not available for this home yet.",
                "note": "This is an estimate of everyday electricity use, excluding the appliances whose schedules we can change.",
            }
        strongest = max(explanation_factors(explanation), key=lambda row: abs(row["change_kwh"]))
        if abs(strongest["change_kwh"]) < 0.005:
            influence = "None of the factors shown makes a large difference to this estimate."
        else:
            direction = "up" if strongest["change_kwh"] > 0 else "down"
            influence = f"{strongest['label']} pushes the estimate {direction} the most."
        return {
            "title": "What TabPFN expects",
            "summary": (
                f"TabPFN expects about {explanation['prediction_kwh']:.2f} kWh of everyday "
                f"electricity use between {result['target_hour']:02}:00 and {result['target_hour'] + 1:02}:00, "
                f"using readings up to {result['origin_hour']:02}:00. {influence}"
            ),
            "note": "These factors explain the estimate, not the cause of your bill. The forecast excludes appliances whose schedules we can change.",
            "factors": explanation_factors(explanation),
        }
    if name == "compare_schedules":
        if result["recommended"]:
            best = next(row for row in result["candidates"] if row["id"] == result["recommended"])
            summary = (
                f"The {LABELS[best['id']]} costs ${best['bill']:.2f} in our test—"
                f"${best['difference_from_feedback']:.2f} less than the comfort-first plan. "
                "Charging and laundry finish, with no worse room comfort and no less "
                "energy left in the battery."
            )
        else:
            summary = (
                "We didn't find a cheaper plan that also meets the charging, laundry, "
                "comfort and battery checks. There isn't enough evidence to recommend a change for this day."
            )
        return {
            "title": "Could a different plan cost less?",
            "summary": summary,
            "note": "These plans were tested on a past day. The savings are simulated, not a promise for your next bill.",
            "plans": [
                {
                    "id": row["id"],
                    "label": LABELS[row["id"]],
                    "concerns": [REASONS.get(reason, reason) for reason in row["reasons"]],
                }
                for row in result["candidates"]
            ],
        }
    if name == "export_plan":
        return {
            "title": "Your home energy report",
            "summary": "Your report is ready. It brings together the costs, forecast, explanation and plans we compared.",
            "note": "For review only. Nothing in your home has been changed.",
        }
    return {
        "title": "A little about this home",
        "summary": (
            f"We have {result['training_days']} days of earlier readings to help understand this home. "
            + (
                "TabPFN uses them to estimate upcoming electricity use. "
                if "TabPFN" in result["forecaster"]
                else "Its forecast uses patterns from earlier days. "
            )
            + "You can explore the costs, understand the forecast, or compare energy plans."
        ),
        "note": "This is a demonstration. It cannot change your home's devices.",
    }


def tool_response(service, name, home, date, include_details=False):
    """Small, reader-ready MCP results; full evidence is explicitly opt-in."""
    result = service.call(name, home, date)
    view = presentation(name, result)
    response = {
        "home": str(home),
        "date": date,
        "answer": view["summary"],
        "context": view["note"],
        "data_source": "Generated example home"
        if service.data["provenance"]["kind"] == "synthetic"
        else "Household research data",
    }
    if "period_selection" in result:
        response["selected_period"] = result["period_selection"]
    if result.get("available") is False:
        return response
    if name in {"appliance_breakdown", "schedule_comparison", "plan_day"}:
        response["facts"] = {
            key: value for key, value in result.items() if key not in {"note", "home", "date"}
        }
        if not include_details:
            if name == "schedule_comparison":
                response["facts"].pop("recorded_days", None)
                for key in ("reference", "alternative"):
                    response["facts"][key] = {k: v for k, v in result[key].items() if k != "days"}
            elif name == "plan_day" and result.get("available"):
                response["facts"]["days"] = [
                    {k: v for k, v in day.items() if k not in {"actions", "forecast"}}
                    | {
                        "device_windows": {
                            key: active_windows(day["actions"], column)
                            for key, column in (
                                ("cooling", "ac_kw"),
                                ("car", "ev_kw"),
                                ("laundry", "washer_kw"),
                            )
                        }
                    }
                    for day in result["days"]
                ]
    elif name == "explain_bill":
        response["costs"] = {
            "currency": result["currency"],
            "day_total": result["total"],
            "most_expensive_hour": result["highest_cost_hour"],
            "period_days": result["period_days"],
            "period_start": result["period_start"],
            "period_end": result["period_end"],
            "period_total": result["period"]["period_total"],
        }
        if "period_selection" in result:
            response["costs"].pop("day_total")
            response["costs"].pop("most_expensive_hour")
            response["costs"]["most_expensive_day"] = result["largest_day"]
    elif name == "forecast_and_explain":
        response["forecast"] = {
            "model": result["model"],
            "readings_through_hour": result["origin_hour"],
            "hour": result["target_hour"],
            "expected_kwh": result["explanation"]["prediction_kwh"]
            if result["explanation"]
            else None,
            "main_factors": view.get("factors", []),
        }
    elif name == "compare_schedules":
        response["recommended_plan"] = LABELS.get(result["recommended"])
        response["plans"] = [
            {
                "name": LABELS[row["id"]],
                "cost_usd": row["bill"],
                "room_comfort_percent": row["comfort_pct"],
                "charging_finished": row["ev_complete"],
                "laundry_finished": row["washer_complete"],
                "battery_remaining_percent": row["terminal_battery_soc"] * 100,
                "passes_comparison_checks": row["eligible"],
                "concerns": [REASONS.get(reason, reason) for reason in row["reasons"]],
            }
            for row in result["candidates"]
        ]
        response["comparison_notes"] = (
            "Comfort means time within the chosen temperature range. Checks compare against the "
            "comfort-first plan; passing does not necessarily mean comfortable temperatures all day. "
            "Costs exclude demand-response rewards and penalties. Each home is tested without trading energy with neighbours."
        )
    elif name == "inspect_home":
        response["home_overview"] = {
            "days_of_history": result["training_days"],
            "devices": result["devices"],
            "forecast_model": result["forecaster"],
            "personalized_plan_available": bool(result["personalized_policy"]),
        }
    else:
        response["report"] = result  # Export deliberately delivers the complete evidence.
    if include_details and name != "export_plan":
        response["details"] = result
    return response


def active_windows(actions, column):
    hours = [row["hour"] for row in actions if row.get(column, 0) > 0.01]
    spans = []
    for hour in hours:
        if spans and spans[-1][1] == hour:
            spans[-1][1] = hour + 1
        else:
            spans.append([hour, hour + 1])
    return (
        ", ".join(f"{start:02}:00–{end:02}:00" for start, end in spans)
        or "No run needed in this time window"
    )


def insight_presentation(name, result):
    title = {
        "appliance_breakdown": "Where the cost comes from",
        "schedule_comparison": "What could have changed?",
        "plan_day": "A plan for the rest of the day",
    }[name]
    if result.get("available") is False:
        return {
            "title": title,
            "summary": result["note"],
            "note": "No missing readings or model results have been invented.",
        }
    if name == "appliance_breakdown":
        largest = max(result["devices"], key=lambda r: r["gross_cost"])
        balance = f"${abs(result['total']):.2f} {'credit' if result['total'] < 0 else 'cost'}"
        summary = f"{largest['label']} is the largest appliance cost, at ${largest['gross_cost']:.2f} before solar. After solar credits and fixed charges, the net result is a {balance} across {result['days']} day(s)."
        extra = {}
    elif name == "schedule_comparison":
        a, b = result["reference"], result["alternative"]
        summary = (
            f"The {b['label']} cost ${result['savings']:.2f} less than your selected {a['label']} across {result['days']} day(s), with no worse comfort, completed appliance tasks and no less battery energy on every day."
            if result["improvement_found"]
            else "None of the other tested plans cost less while also matching your selected plan's comfort, appliance completion and remaining battery energy on every day. The comparison keeps your selected plan."
        )
        extra = {}
    else:
        days = result["days"]
        first = days[0]
        schedules = [
            {"device": label, "key": key, "when": active_windows(first["actions"], column)}
            for key, label, column in [
                ("cooling", "Cooling", "ac_kw"),
                ("car", "Car charging", "ev_kw"),
                ("laundry", "Laundry", "washer_kw"),
            ]
        ]
        total = sum(day["remaining_cost"] for day in days)
        forecast = first["forecast"]
        solar = max(forecast, key=lambda r: r["solar_kwh"])
        lowest = min(forecast, key=lambda r: r["price_usd_kwh"])
        summary = (
            f"From {result['origin_hour']:02}:00 on {first['date']}, {first['plan_label'].lower()} estimates ${first['remaining_cost']:.2f} for the rest of the day. "
            f"Solar is expected to peak around {solar['hour']:02}:00 ({solar['solar_kwh']:.2f} kWh); the lowest assumed rate is ${lowest['price_usd_kwh']:.3f}/kWh. "
            f"Cooling: {schedules[0]['when']}. Car charging: {schedules[1]['when']}. Laundry: {schedules[2]['when']}."
        )
        if len(days) > 1:
            summary = f"Across {len(days)} separate daily outlooks from {result['origin_hour']:02}:00, estimated remaining-day costs total ${total:.2f}. Choose a date below to inspect its schedule; there is no single repeated schedule for the entire period."
        extra = {"schedules": schedules}
    note = result["note"]
    if "coverage" in result and not result["coverage"]["complete"]:
        note = (
            f"Includes {result['coverage']['available_days']} of {result['coverage']['requested_days']} selected days. Missing days are excluded. "
            + note
        )
    return {"title": title, "summary": summary, "note": note, **extra}


def period_presentation(name, result):
    scope = result["period_selection"]
    view = presentation(
        name, {key: value for key, value in result.items() if key != "period_selection"}
    )
    count = scope["available_days"]
    if name == "explain_bill":
        largest = result["largest_day"]
        view.update(
            title="Electricity costs over this period",
            summary=(
                f"Electricity comes to ${result['total']:.2f} across the {count} available days. "
                f"The most expensive day was {largest['date']}, at ${largest['cost']:.2f}."
            ),
        )
    elif name == "forecast_and_explain" and result["explanation"]:
        view.update(
            title="The average daily forecast",
            summary=(
                f"Across {count} days, TabPFN's average estimate for {result['target_hour']:02}:00–"
                f"{result['target_hour'] + 1:02}:00 was {result['explanation']['prediction_kwh']:.2f} kWh. "
                f"Each daily forecast used readings only up to {result['origin_hour']:02}:00 on that day."
            ),
        )
        view["note"] = (
            "An average of separate daily forecasts, not a prediction for one continuous future period. The factors explain the estimates, not the causes of a bill."
        )
    elif name == "compare_schedules":
        if result["recommended"]:
            best = next(r for r in result["candidates"] if r["id"] == result["recommended"])
            view["summary"] = (
                f"Across {count} days tested, the {LABELS[best['id']]} cost ${best['bill']:.2f}—"
                f"${best['difference_from_feedback']:.2f} less than the comfort-first plan. "
                "It passed the charging, laundry, comfort and battery checks on every included day."
            )
        else:
            view["summary"] = (
                f"Across {count} days tested, no cheaper plan passed every daily charging, laundry, comfort and battery check. We can't recommend a change for this period."
            )
        view["note"] = (
            "Simulated savings. Each day was tested separately, rather than as one continuous month of operation. Future bills may differ."
        )
    if not scope["complete"]:
        view["note"] = (
            f"Results cover {count} of the {scope['requested_days']} selected days. Missing days are excluded, not estimated. "
            + view["note"]
        )
    return view
