"""Self-contained, printable reports from already verified assistant evidence."""

from html import escape

from .presentation import active_windows


def money(value):
    return f"${abs(value):.2f} credit" if value < 0 else f"${value:.2f}"


def table(headings, rows):
    head = "".join(f"<th>{escape(str(v))}</th>" for v in headings)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def render_report(report):
    bill = report["bill"]
    scope = report.get("period_selection", {})
    start = scope.get("start", report.get("date", bill.get("date", "")))
    end = scope.get("end", start)
    home = report.get("home", bill.get("home", ""))
    period = start if start == end else f"{start} — {end}"
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Your home energy report</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#244a40;background:#f5f8f3;margin:0}main{max-width:900px;margin:auto;padding:38px 24px}header{border-bottom:2px solid #176857;padding-bottom:20px}h1{margin:8px 0;font-size:32px}h2{font-size:20px}section{background:white;border:1px solid #dce7dc;border-radius:14px;padding:22px;margin:20px 0}small,.muted{color:#63776c}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px 8px;border-bottom:1px solid #e6ece5}.scroll{overflow:auto}strong.total{font-size:36px}footer{font-size:12px;overflow-wrap:anywhere}@media print{body{background:white}main{padding:0}section{break-inside:avoid}h2{break-after:avoid}}</style></head><body><main>",
        f"<header><small>GRIDPFN · POWERED BY TABPFN-3.5</small><h1>Your energy report</h1><p>Home {escape(str(home))} · {escape(period)}</p></header>",
        f"<section><h2>Your recorded energy costs</h2><strong class='total'>{money(bill['total'])}</strong><p class='muted'>Calculated from household readings and the declared tariff; not a utility invoice.</p>",
    ]
    if scope:
        parts.append(
            f"<p>{scope['available_days']} of {scope['requested_days']} selected days have readings. Missing days are not estimated.</p>"
        )
    breakdown = report.get("appliance_breakdown", {})
    if breakdown.get("available"):
        rows = [
            [r["label"], f"{r['kwh']:.1f} kWh", money(r["gross_cost"])]
            for r in breakdown["devices"]
        ]
        rows += [[r["label"], "", money(r["cost"])] for r in breakdown["adjustments"]]
        parts.append(table(["Source", "Energy", "Cost / credit"], rows))
    parts.append("</section>")
    comparison = report.get("comparison", {})
    if comparison.get("candidates"):
        from .presentation import LABELS

        rows = [
            [
                LABELS.get(r["id"], r["label"]),
                money(r["bill"]),
                f"{r['comfort_pct']:.1f}%",
                "Passed" if r["eligible"] else "Not passed",
            ]
            for r in comparison["candidates"]
        ]
        parts.append(
            "<section><h2>Compared energy plans</h2>"
            + table(["Simulated plan", "Cost", "Comfort", "Comparison checks"], rows)
            + "<p class='muted'>These are matched simulations, not your recorded appliance schedule. Passing checks means no worse than the reference; it does not guarantee perfect comfort or real-world savings.</p></section>"
        )
    forecast = report.get("forecast", {})
    explanation = forecast.get("explanation")
    if explanation:
        from .presentation import FEATURE_NAMES

        rows = [["Reference from training examples", f"{explanation['baseline_kwh']:.3f} kWh"]]
        rows += [
            [FEATURE_NAMES.get(r["feature"], r["feature"]), f"{r['kwh']:+.3f} kWh"]
            for r in explanation["contributions"]
        ]
        rows += [["TabPFN demand estimate", f"{explanation['prediction_kwh']:.3f} kWh"]]
        title = "Average daily demand forecast" if start != end else "Demand forecast"
        parts.append(
            f"<section><h2>{title}</h2><p>Issued at {forecast['origin_hour']:02}:00 for {forecast['target_hour']:02}:00. Demand excludes separately controlled appliances.</p>"
            + table(["Forecast influence", "Contribution"], rows)
            + "<p class='muted'>Grouped SHAP explains the prediction, not causes of a bill or the controller's actions. For a period, these are averages of separate daily forecasts.</p></section>"
        )
    planning = report.get("plan_day", {})
    if planning.get("available"):
        rows = []
        for day in planning["days"]:
            windows = [
                active_windows(day["actions"], key) for key in ("ac_kw", "ev_kw", "washer_kw")
            ]
            rows.append([day["date"], money(day["remaining_cost"]), *windows])
        parts.append(
            f"<section><h2>Plan from {planning['origin_hour']:02}:00</h2>"
            + table(["Day", "Estimated remaining cost", "Cooling", "Car", "Laundry"], rows)
            + "<p class='muted'>Plans use dated simulated states, TabPFN forecasts and assumed prices. Days start independently. Later-than-supported forecasts use persistence. No physical devices are controlled.</p></section>"
        )
    source = (
        "Generated household inputs"
        if report.get("provenance", {}).get("kind") == "synthetic"
        else "Local household research readings"
    )
    parts.append(
        f"<footer><p>{source}. Grid exports are included where available; peer-to-peer trading is not enabled in this assistant.</p><p>Reference: {escape(report['evidence_sha256'])}</p><p>Use your browser’s Print menu to save this report as a PDF.</p></footer></main></body></html>"
    )
    return "".join(parts)
