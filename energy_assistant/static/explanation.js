"use strict";
const influenceLabels = {
  "Recent household demand": "Recent use",
  "Recent solar generation": "Recent solar",
  "Weather and current tariff": "Weather & tariff",
  "Time and forecast horizon": "Time & horizon",
};
function influenceWaterfall(e) {
  const wrap = node("div", undefined, "influence-view"),
    factors = e.factors;
  const steps = [
    { label: "Past examples", start: 0, end: e.baseline, total: true },
  ];
  let running = e.baseline;
  for (const f of factors) {
    steps.push({
      label: influenceLabels[f.label] || f.label,
      start: running,
      end: running + f.value,
      value: f.value,
    });
    running += f.value;
  }
  steps.push({
    label: "This forecast",
    start: 0,
    end: e.prediction,
    total: true,
  });
  const minimum = Math.min(0, ...steps.flatMap((r) => [r.start, r.end])),
    maximum = Math.max(0.001, ...steps.flatMap((r) => [r.start, r.end])),
    span = maximum - minimum;
  const chart = node("div", undefined, "influence-waterfall");
  chart.setAttribute("role", "img");
  chart.setAttribute(
    "aria-label",
    `SHAP forecast explanation from ${e.baseline.toFixed(3)} to ${e.prediction.toFixed(3)} ${e.unit}`,
  );
  for (const r of steps) {
    const row = node(
        "div",
        undefined,
        `waterfall-row ${r.total ? "total" : ""}`,
      ),
      heading = node("div", undefined, "waterfall-label"),
      value = `${r.total ? r.end.toFixed(3) : (r.value >= 0 ? "+" : "") + r.value.toFixed(3)} ${e.unit}`;
    heading.append(node("span", r.label), node("strong", value));
    const track = node("div", undefined, "waterfall-track"),
      bar = node(
        "i",
        undefined,
        r.total ? "total" : r.value >= 0 ? "positive" : "negative",
      );
    bar.style.left = `${(100 * (Math.min(r.start, r.end) - minimum)) / span}%`;
    bar.style.width = `${(100 * Math.abs(r.end - r.start)) / span}%`;
    bar.title = `${r.label}: ${value}`;
    track.append(bar);
    row.append(heading, track);
    chart.append(row);
  }
  wrap.append(chart);
  note(
    wrap,
    "Brown raises the estimate · Green lowers it. These influences explain the forecast, not appliance choices or guaranteed savings.",
  );
  return wrap;
}
function forecastInspector(home, date, origin, horizon, forecast) {
  const panel = node(
      "details",
      undefined,
      "explanation-details forecast-inspector",
    ),
    summary = node("summary", "Why this forecast? · TabPFN SHAP");
  panel.append(summary);
  const controls = node("div", undefined, "influence-controls"),
    variable = node("select"),
    hour = node("select"),
    load = node("button", "Explain", "secondary");
  load.type = "button";
  variable.setAttribute("aria-label", "Forecast to explain");
  hour.setAttribute("aria-label", "Forecast hour");
  for (const [value, label] of [
    ["demand", "Everyday use"],
    ["solar", "Solar output"],
    ["temperature", "Outdoor temperature"],
  ]) {
    const o = node("option", label);
    o.value = value;
    variable.append(o);
  }
  for (let h = origin + 1; h <= Math.min(23, origin + horizon); h++) {
    const o = node("option", `${String(h).padStart(2, "0")}:00`);
    o.value = h;
    hour.append(o);
  }
  hour.value = String(Math.min(23, origin + Math.min(3, horizon)));
  controls.append(variable, hour, load);
  const result = node("div");
  result.setAttribute("aria-live", "polite");
  panel.append(controls, result);
  let request = 0;
  load.onclick = async () => {
    const current = ++request;
    load.disabled = true;
    result.replaceChildren(
      node("p", "Checking this home's forecast…", "card-note"),
    );
    try {
      const params = new URLSearchParams({
        origin,
        target: hour.value,
        variable: variable.value,
      });
      const response = await fetch(
        `/api/explanation/${encodeURIComponent(home)}/${encodeURIComponent(date)}?${params}`,
      );
      const e = await response.json();
      if (!response.ok)
        throw Error(e.detail || "The explanation is unavailable.");
      if (current !== request) return;
      result.replaceChildren();
      if (!e.available) {
        note(result, e.note);
        return;
      }
      note(
        result,
        `${prettyDate(date)} · readings through ${String(origin).padStart(2, "0")}:00 → forecast for ${String(e.target_hour).padStart(2, "0")}:00`,
      );
      result.append(forecastContextChart(e), influenceWaterfall(e));
      const key = {
          demand: "load_kwh",
          solar: "solar_kwh",
          temperature: "outdoor_c",
        }[e.target],
        saved = forecast?.find((r) => r.hour === e.target_hour)?.[key];
      if (saved !== undefined && Math.abs(saved - e.prediction) > 0.001)
        note(
          result,
          `Saved planning input: ${saved.toFixed(3)} ${e.unit}; recalculated here: ${e.prediction.toFixed(3)} ${e.unit}. This explanation describes the recalculated forecast.`,
        );
      details(result, "Method & checks", [
        `${e.method}. ${e.background_rows} training examples; reconstruction difference ${e.additivity_error.toExponential(1)} ${e.unit}.`,
        e.context_sha256
          ? "Uses the saved study's fitted context and pinned model weights. CPU execution can differ slightly from saved predictions."
          : "An explanatory household model; exact identity with the controller's forecast model has not been established.",
        e.caveat,
      ]);
    } catch (error) {
      result.replaceChildren(node("p", error.message, "card-note"));
    } finally {
      load.disabled = false;
    }
  };
  for (const select of [variable, hour])
    select.onchange = () => {
      request++;
      result.replaceChildren();
    };
  return panel;
}
function reasoningPanel(event, trace) {
  const panel = node(
    "details",
    undefined,
    "explanation-details answer-details",
  );
  panel.append(node("summary", "How we worked this out"));
  const tools = event.cards
      .filter((c) => c.data.available !== false)
      .map((c) => c.tool),
    planning = tools.includes("plan_day"),
    forecast = tools.includes("forecast_and_explain"),
    comparison = tools.some((t) =>
      ["compare_schedules", "schedule_comparison"].includes(t),
    );
  const steps = planning
    ? [
        ["Readings", "This home's history and selected time"],
        ["TabPFN", "Demand, solar and outdoor temperature"],
        ["Plan checks", "Cost, comfort, charging and battery"],
      ]
    : forecast
      ? [
          ["Home history", "Readings before the forecast"],
          ["TabPFN", "An estimate for a future hour"],
          ["SHAP", "Which input groups move that estimate"],
        ]
      : comparison
        ? [
            ["Same home", "Matched simulated conditions"],
            ["Tested plans", "Compare actions and observed tariffs"],
            ["Checks", "Cost and daily service requirements"],
          ]
        : [
            ["Recorded use", "Appliance and solar readings"],
            ["Tariff", "Import costs and export credits"],
            ["Your bill", "Add charges; subtract credits"],
          ];
  const flow = node("ol", undefined, "reasoning-flow");
  for (const [title, text] of steps) {
    const item = node("li");
    item.append(node("strong", title), node("span", text));
    flow.append(item);
  }
  panel.append(flow);
  if (planning)
    note(
      panel,
      "The selected plan is a simulation from a saved household state. TabPFN supplies forecasts; the scheduler chooses actions. Expand “Why this forecast?” in the plan to inspect demand, solar or temperature.",
    );
  if (comparison && !planning)
    note(
      panel,
      "A comparison shows tested outcomes; it does not attribute the controller's decisions to individual inputs.",
    );
  const audit = node("details", undefined, "explanation-details");
  audit.append(node("summary", "Calculation record"), trace);
  note(
    audit,
    `${event.mode === "llm" ? "The chat model chose the tools." : "Guided question routing chose the tools."} Reference: ${event.evidence_sha256.slice(0, 12)}.`,
  );
  panel.append(audit);
  return panel;
}
function exchangeView(exchange) {
  const panel = node(
    "details",
    undefined,
    "explanation-details energy-exchange",
  );
  panel.append(node("summary", "Home ↔ Grid · energy balance"));
  if (exchange) {
    const stats = node("div", undefined, "schedule-stats");
    for (const [label, value] of [
      ["Imported", `${exchange.import_kwh.toFixed(1)} kWh`],
      ["Exported", `${exchange.export_kwh.toFixed(1)} kWh`],
      ["Export credit", money(exchange.export_credit)],
    ]) {
      const box = node("div");
      box.append(node("small", label), node("strong", value));
      stats.append(box);
    }
    panel.append(
      stats,
      chart(
        exchange.hours.map((r) => r.export_kwh - r.import_kwh),
        exchange.hours.map((r) => r.hour),
        null,
        "bars",
        `${exchange.evidence === "recorded" ? "Recorded" : "Projected"} energy exchange: positive exports, negative imports, in kWh`,
      ),
    );
    note(
      panel,
      `Above zero: surplus sent to the grid. Below zero: energy bought. ${exchange.evidence === "recorded" ? "Calculated from recorded consumption and solar; battery flows are not measured." : "Projected flows for the displayed plan."}`,
    );
  } else
    note(
      panel,
      "Export credits are available in your bill breakdown. This saved plan has no projected grid-flow ledger; no extra trading income is assumed.",
    );
  note(
    panel,
    "Neighbour trading is not enabled for this single-home model. Grid credits are not peer-to-peer transactions, and this plan does not establish an optimal selling time.",
  );
  return panel;
}

function forecastContextChart(e) {
  const box = node("div", undefined, "forecast-context"),
    rows = e.recent_readings || [];
  if (!rows.length) return box;
  const values = [...rows.map((r) => r.value), e.prediction],
    lo = Math.min(...values),
    hi = Math.max(...values),
    pad = Math.max((hi - lo) * 0.15, 0.1),
    first = rows[0].hour;
  const X = (h) =>
      60 + ((h - first) / Math.max(1, e.target_hour - first)) * 500,
    Y = (v) => 133 - ((v - lo + pad) / (hi - lo + 2 * pad)) * 96;
  const svg = svgNode("svg", {
    viewBox: "0 0 620 182",
    class: "chart",
    role: "img",
    "aria-label": `Recorded ${e.target} through ${e.origin_hour}:00 and predicted value at ${e.target_hour}:00 in ${e.unit}`,
  });
  svg.append(
    svgNode("rect", {
      x: X(e.origin_hour),
      y: 22,
      width: X(e.target_hour) - X(e.origin_hour) + 18,
      height: 126,
      fill: "#edf4ee",
      rx: 8,
    }),
  );
  for (let i = 0; i < 3; i++) {
    const v = lo + ((hi - lo) * i) / 2;
    svg.append(
      svgNode(
        "text",
        { x: 48, y: Y(v) + 3, "text-anchor": "end" },
        v.toFixed(2),
      ),
    );
  }
  svg.append(
    svgNode("polyline", {
      points: rows.map((r) => `${X(r.hour)},${Y(r.value)}`).join(" "),
      class: "line",
    }),
  );
  for (const r of rows) {
    const dot = svgNode("circle", {
      cx: X(r.hour),
      cy: Y(r.value),
      r: 3,
      fill: "#176857",
    });
    dot.append(
      svgNode(
        "title",
        {},
        `${r.hour}:00 recorded: ${r.value.toFixed(3)} ${e.unit}`,
      ),
    );
    svg.append(dot);
  }
  const dot = svgNode("circle", {
    cx: X(e.target_hour),
    cy: Y(e.prediction),
    r: 6,
    fill: "#b47a43",
  });
  dot.append(
    svgNode(
      "title",
      {},
      `${e.target_hour}:00 predicted: ${e.prediction.toFixed(3)} ${e.unit}`,
    ),
  );
  svg.append(dot);
  for (const h of new Set([first, e.origin_hour, e.target_hour]))
    svg.append(
      svgNode(
        "text",
        { x: X(h), y: 167, "text-anchor": "middle" },
        `${String(h).padStart(2, "0")}:00`,
      ),
    );
  box.append(
    node("div", `Recorded → predicted · ${e.unit}`, "chart-label"),
    svg,
  );
  note(
    box,
    "Green: recent readings. Brown dot: TabPFN's estimate. Shaded area: future, with no future readings supplied.",
  );
  const inputs = node("details", undefined, "explanation-details");
  inputs.append(node("summary", "Readings behind this forecast"));
  const table = node("table"),
    head = node("tr");
  for (const label of [
    "Time",
    "Use · kWh",
    "Solar · kWh",
    "Outside · °C",
    "Base $/kWh",
  ])
    head.append(node("th", label));
  const thead = node("thead");
  thead.append(head);
  table.append(thead);
  const body = node("tbody");
  for (const r of e.observed_inputs || []) {
    const tr = node("tr");
    for (const v of [
      `${String(r.hour).padStart(2, "0")}:00`,
      r.demand_kwh.toFixed(3),
      r.solar_kwh.toFixed(3),
      r.outdoor_c.toFixed(1),
      r.base_price.toFixed(3),
    ])
      tr.append(node("td", v));
    body.append(tr);
  }
  table.append(body);
  const scroll = node("div", undefined, "table-wrap");
  scroll.append(table);
  inputs.append(scroll);
  note(
    inputs,
    "The model uses the latest reading, the previous reading and a recent average, plus time and forecast horizon. Weather is observed outdoor temperature, not an external weather forecast. Use excludes cooling, car charging and laundry.",
  );
  box.append(inputs);
  return box;
}
