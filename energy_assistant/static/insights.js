"use strict";
function timeline(actions, label) {
  const svg = svgNode("svg", {
    viewBox: "0 0 620 174",
    role: "img",
    "aria-label": label,
    class: "device-timeline",
  });
  const rows = [
    ["Cooling", "ac_kw", "#559687"],
    ["Car", "ev_kw", "#7195b4"],
    ["Laundry", "washer_kw", "#b7a06d"],
    ["Battery", "battery_kw", "#8e88ac"],
  ].filter(([, key]) => actions.some((r) => Object.hasOwn(r, key)));
  const x = (h) => 91 + h * 21.2;
  [0, 6, 12, 18, 24].forEach((h) =>
    svg.append(
      svgNode(
        "text",
        { x: x(h), y: 13, "text-anchor": "middle" },
        `${String(h).padStart(2, "0")}:00`,
      ),
    ),
  );
  rows.forEach(([name, key, color], i) => {
    const y = 26 + i * 35;
    svg.append(svgNode("text", { x: 2, y: y + 17 }, name));
    const max = Math.max(0.01, ...actions.map((r) => Math.abs(r[key] || 0)));
    for (let h = 0; h < 24; h++) {
      const row = actions.find((r) => r.hour === h),
        value = row?.[key] || 0;
      const cell = svgNode("rect", {
        x: x(h),
        y,
        width: 19.5,
        height: 25,
        rx: 4,
        fill: row && Math.abs(value) > 0.01 ? color : "#eef1eb",
        opacity:
          row && Math.abs(value) > 0.01
            ? 0.35 + (0.65 * Math.abs(value)) / max
            : 1,
      });
      cell.append(
        svgNode(
          "title",
          {},
          row
            ? `${name}, ${String(h).padStart(2, "0")}:00 · ${Math.abs(value).toFixed(2)} kW${key === "battery_kw" ? (value > 0 ? " charging" : value < 0 ? " discharging" : " idle") : ""}`
            : "Before the planning time",
        ),
      );
      svg.append(cell);
      if (key === "battery_kw" && Math.abs(value) > 0.01)
        svg.append(
          svgNode(
            "text",
            { x: x(h) + 10, y: y + 17, "text-anchor": "middle", fill: "#fff" },
            value > 0 ? "+" : "−",
          ),
        );
    }
  });
  const scroll = node("div", undefined, "timeline-scroll");
  scroll.append(svg);
  return scroll;
}
function resultDateSelector(dates, onChange) {
  const wrapper = node("div", undefined, "result-date-selector"),
    label = node("label", "Day to inspect"),
    select = node("select");
  const id = `result-date-${document.querySelectorAll(".result-date-selector").length}`;
  select.id = id;
  label.htmlFor = id;
  dates.forEach((d) => {
    const o = node("option", prettyDate(d));
    o.value = d;
    select.append(o);
  });
  select.onchange = () => onChange(select.value);
  wrapper.append(label, select);
  return wrapper;
}
function renderInsightCard(tool, d, v) {
  const c = card(v.title, d.evidence || "recorded");
  if (d.available === false) {
    note(c, v.summary);
    return c;
  }
  if (tool === "appliance_breakdown") {
    const rows = node("div", undefined, "cost-breakdown"),
      extent = Math.max(0.01, ...d.devices.map((r) => r.gross_cost));
    for (const r of [...d.devices].sort(
      (a, b) => b.gross_cost - a.gross_cost,
    )) {
      const row = node("div", undefined, "cost-row"),
        label = node("span", r.label),
        value = node("strong", money(r.gross_cost)),
        track = node("div", undefined, "cost-track"),
        bar = node("i");
      bar.style.width = `${(100 * r.gross_cost) / extent}%`;
      track.append(bar);
      label.append(node("small", `${r.kwh.toFixed(1)} kWh`));
      row.append(label, value, track);
      rows.append(row);
    }
    c.append(rows);
    for (const r of d.adjustments) {
      const row = node("div", undefined, "cost-adjustment");
      row.append(node("span", r.label), node("strong", money(r.cost)));
      c.append(row);
    }
    const total = node("div", undefined, "cost-adjustment cost-total");
    total.append(node("span", "Total"), node("strong", money(d.total)));
    c.append(total);
    if (d.exchange) c.append(exchangeView(d.exchange));
    details(c, "How solar is counted", [v.note]);
    return c;
  }
  if (tool === "schedule_comparison") {
    const stat = node("div", undefined, "schedule-stats");
    for (const plan of [d.reference, d.alternative]) {
      const box = node("div");
      box.append(
        node("small", plan.label),
        node("strong", money(plan.cost)),
        node("span", `${plan.comfort_pct.toFixed(1)}% comfortable`),
      );
      stat.append(box);
    }
    c.append(stat);
    const body = node("div");
    function show(date) {
      body.replaceChildren();
      const first = d.reference.days.find((r) => r.date === date),
        second = d.alternative.days.find((r) => r.date === date);
      const recorded = d.recorded_days?.find((r) => r.date === date);
      if (recorded) {
        const panel = node("details", undefined, "timeline-panel");
        panel.open = true;
        panel.append(
          node("summary", `Recorded appliances · ${money(recorded.bill)}`),
          timeline(recorded.actions, `Recorded appliance use on ${date}`),
        );
        note(
          panel,
          "Recorded consumption priced with the declared tariff. Battery use is not measured here; compare savings between the two simulations below.",
        );
        body.append(panel);
      }
      for (const [plan, row] of [
        [d.reference, first],
        [d.alternative, second],
      ]) {
        const panel = node("div", undefined, "timeline-panel");
        panel.append(
          node("h4", `${plan.label} · ${money(row.bill)}`),
          timeline(
            row.actions,
            `${plan.label}: appliance operation on ${date}`,
          ),
        );
        body.append(panel);
      }
      body.append(
        node("div", "Observed electricity prices · $/kWh", "chart-label"),
        chart(
          first.ledger.map((r) => r.tariff),
          first.ledger.map((r) => r.hour),
          null,
          "line",
          "Observed tariff in dollars per kWh",
        ),
      );
      if (state.catalog?.live_forecast)
        body.append(forecastInspector(d.home || $("home").value, date, 12, 6));
    }
    if (d.dates.length > 1) c.append(resultDateSelector(d.dates, show));
    c.append(body);
    show(d.dates[0]);
    note(
      c,
      "Darker blocks mean higher power. Battery + charges; − supplies energy to the home.",
    );
    details(c, "What this comparison means", [v.note]);
    return c;
  }
  if (d.forecast_horizon_hours <= 6)
    note(
      c,
      "TabPFN looks up to six hours ahead. Later planning inputs hold the last forecast; they are less certain.",
    );
  const body = node("div");
  function show(date) {
    const day = d.days.find((row) => row.date === date);
    body.replaceChildren();
    const stats = node("div", undefined, "schedule-stats");
    for (const [label, value] of [
      ["Estimated remaining cost", money(day.remaining_cost)],
      ["Starting battery", `${(day.starting_battery_soc * 100).toFixed(0)}%`],
      ["Expected comfort", `${day.comfort_pct.toFixed(0)}%`],
    ]) {
      const box = node("div");
      box.append(node("small", label), node("strong", value));
      stats.append(box);
    }
    body.append(stats);
    body.append(
      node(
        "div",
        `${day.plan_label} · from ${String(day.origin_hour).padStart(2, "0")}:00 on ${prettyDate(date)}`,
        "chart-label",
      ),
      timeline(day.actions, "Forecast-based appliance schedule"),
    );
    const windows = node("div", undefined, "planning-windows");
    for (const [label, key] of [
      ["Cooling", "ac_kw"],
      ["Car charging", "ev_kw"],
      ["Laundry", "washer_kw"],
    ]) {
      const active = day.actions
          .filter((r) => r[key] > 0.01)
          .map((r) => r.hour),
        groups = [];
      for (const h of active) {
        if (groups.length && groups.at(-1)[1] === h) groups.at(-1)[1] = h + 1;
        else groups.push([h, h + 1]);
      }
      const row = node("div");
      row.append(
        node("strong", label),
        node(
          "span",
          groups.length
            ? groups
                .map(
                  ([a, b]) =>
                    `${String(a).padStart(2, "0")}:00–${String(b).padStart(2, "0")}:00`,
                )
                .join(", ")
            : key === "ev_kw" && day.origin_hour >= day.ev_deadline_hour
              ? "Charging window has ended"
              : "No run in this window",
        ),
      );
      windows.append(row);
    }
    body.append(windows);
    const plots = node("div", undefined, "outlook-plots");
    for (const [label, values, description] of [
      [
        "Expected solar · kWh",
        day.forecast.map((r) => r.solar_kwh),
        "TabPFN solar forecast in kWh",
      ],
      [
        "Assumed price · $/kWh",
        day.forecast.map((r) => r.price_usd_kwh),
        "Current base price held constant plus known time-of-use tariff",
      ],
      [
        "Projected battery · %",
        day.actions.map((r) => r.battery_soc * 100),
        "Simulated battery state of charge",
      ],
      [
        "Projected room temperature · °C",
        day.actions.map((r) => r.indoor_c),
        "Simulated room temperature",
      ],
    ]) {
      const box = node("div");
      box.append(
        node("div", label, "chart-label"),
        chart(
          values,
          day.forecast.map((r) => r.hour),
          null,
          "line",
          description,
        ),
      );
      plots.append(box);
    }
    body.append(plots);
    body.append(
      forecastInspector(
        d.home || $("home").value,
        date,
        day.origin_hour,
        d.forecast_horizon_hours,
        day.forecast,
      ),
      exchangeView(day.exchange),
    );
    note(
      body,
      `Comfort target: ${day.temperature_range.join("–")}°C. Charging ${day.ev_complete ? "finishes" : "does not finish"}; laundry ${day.washer_complete ? "finishes" : "does not finish"}. ${day.improvement_found ? "The forecast-based schedule passed the comparison checks." : "No cheaper forecast-based schedule passed every check, so this view keeps comfort-first scheduling."}`,
    );
  }
  if (d.days.length > 1)
    c.append(
      resultDateSelector(
        d.days.map((r) => r.date),
        show,
      ),
    );
  c.append(body);
  show(d.days[0].date);
  details(c, "What is predicted, and what is assumed?", [v.note]);
  return c;
}
function appendFollowups(answer, cards) {
  const tools = cards.map((c) => c.tool),
    wrap = node("div", undefined, "followups");
  wrap.setAttribute("aria-label", "Continue this conversation");
  const suggestions = tools.includes("appliance_breakdown")
    ? [
        [
          "Could I have spent less?",
          "Show side by side what I could have done differently to reduce my bill.",
        ],
        ["Help me plan my day", "When should I run the appliances today?"],
      ]
    : tools.includes("plan_day")
      ? [
          [
            "Why that forecast?",
            "Explain the TabPFN forecast and its influences.",
          ],
          [
            "Compare past schedules",
            "Show side by side what could have changed in my schedule.",
          ],
        ]
      : [
          [
            "Break down the cost",
            "Which appliances caused the bill? Give me a breakdown.",
          ],
          [
            "When should I charge?",
            "When should I charge my car and home battery?",
          ],
        ];
  for (const [label, question] of suggestions) {
    const button = node("button", `${label} ↗`);
    button.type = "button";
    button.onclick = () => ask(question);
    wrap.append(button);
  }
  answer.append(wrap);
}
