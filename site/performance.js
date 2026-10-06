"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const methods = ["history", "persistence", "trees", "tabpfn", "tabfm", "tabicl"];
  const metricKeys = ["energy_bill_without_dr", "comfort_pct", "objective"];
  const measures = {
    energy_bill_without_dr: {label: "Electricity bill", unit: "$ / home / day · lower is better", digits: 3},
    comfort_pct: {label: "Comfortable time", unit: "% of hours in the temperature band · higher is better", digits: 1},
    objective: {label: "Combined objective", unit: "Original simulator cost / home / day · lower is better", digits: 3},
  };
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const format = (value, digits = 3) => finite(value) ? value.toFixed(digits) : "—";
  const node = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  };
  const svgNode = (tag, attributes) => {
    const result = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) result.setAttribute(key, String(value));
    return result;
  };
  const monthLabel = id => new Intl.DateTimeFormat("en", {month: "long", year: "numeric", timeZone: "UTC"}).format(new Date(`${id}-01T00:00:00Z`));
  const focused = new URLSearchParams(location.search).get("scope") === "foundations";
  const foundationIds = ["tabpfn", "tabfm", "tabicl"];
  let evidence = null;
  let metric = focused ? "energy_bill_without_dr" : "objective";
  let frame = 0;

  function validStats(stats, count) {
    return stats && finite(stats.mean) && finite(stats.sd) && stats.sd >= 0 &&
      Array.isArray(stats.values) && stats.values.length === count && stats.values.every(finite);
  }
  function validateGroup(group, homes) {
    if (!group || !Array.isArray(group.rows) || group.rows.length !== methods.length ||
        new Set(group.rows.map(row => row.id)).size !== methods.length ||
        methods.some(id => !group.rows.some(row => row.id === id)) ||
        !group.oracle || group.oracle.id !== "oracle" || !Array.isArray(group.dates) || !group.dates.length ||
        !Array.isArray(group.home_ids) || JSON.stringify(group.home_ids) !== JSON.stringify(homes)) {
      throw new Error("Incomplete comparison");
    }
    for (const row of [...group.rows, group.oracle]) {
      if (typeof row.label !== "string" || !metricKeys.every(key => validStats(row.policy?.metrics?.[key], homes.length))) throw new Error("Invalid household metrics");
    }
    if (!finite(group.oracle.bound?.lower) || !finite(group.oracle.bound?.upper) || group.oracle.bound.lower > group.oracle.bound.upper) throw new Error("Invalid oracle interval");
  }
  function validate(data) {
    if (!data || !Array.isArray(data.home_ids) || data.home_ids.length < 2 ||
        !Array.isArray(data.folds) || data.folds.length === 0 || !data.protocol ||
        new Set(data.folds.map(fold => fold.id)).size !== data.folds.length) throw new Error("Evidence unavailable");
    validateGroup(data, data.home_ids);
    data.folds.forEach(fold => { monthLabel(fold.id); validateGroup(fold, data.home_ids); });
    return data;
  }
  function selectedGroup() {
    const group = $("period").value === "pooled" ? evidence : evidence.folds.find(fold => fold.id === $("period").value);
    return focused ? {...group, rows: group.rows.filter(row => foundationIds.includes(row.id))} : group;
  }
  function chart(group) {
    const specification = measures[metric];
    const rows = [...group.rows, group.oracle];
    const all = rows.flatMap(row => { const s = row.policy.metrics[metric]; return [...s.values, s.mean - s.sd, s.mean + s.sd]; });
    if (metric === "objective") all.push(group.oracle.bound.upper);
    let low = Math.min(0, ...all);
    let high = Math.max(...all);
    const span = high - low || 1;
    high += span * 0.045;
    if (low < 0) low -= span * 0.025;
    const holder = $("chart");
    holder.replaceChildren();
    for (const row of rows) {
      const stats = row.policy.metrics[metric];
      const line = node("div", undefined, "chart-row");
      line.dataset.method = row.id;
      const label = node("div", row.id === "oracle" ? "Perfect future" : row.label, row.id === "tabpfn" ? "chart-label emphasized" : "chart-label");
      const svg = svgNode("svg", {class: "track", "aria-hidden": "true", focusable: "false"});
      const value = node("div", undefined, "chart-value");
      value.append(node("span", format(stats.mean, specification.digits)), node("span", ` ± ${format(stats.sd, specification.digits)}`, "sd"));
      line.append(label, svg, value); holder.append(line);
      const width = Math.max(30, svg.getBoundingClientRect().width);
      svg.setAttribute("viewBox", `0 0 ${width} 36`);
      const x = amount => 5 + (width - 10) * (amount - low) / (high - low);
      for (const tick of [low, (low + high) / 2, high]) svg.append(svgNode("line", {x1: x(tick), x2: x(tick), y1: 0, y2: 36, stroke: "#edf0e9", "stroke-width": 1}));
      if (metric === "objective") svg.append(svgNode("line", {x1: x(group.oracle.bound.upper), x2: x(group.oracle.bound.upper), y1: 0, y2: 36, stroke: "#b89d75", "stroke-width": 1, "stroke-dasharray": "3 3", class: "oracle-bound"}));
      const color = row.id === "tabpfn" ? "#235c42" : row.id === "oracle" ? "#96713e" : "#637e87";
      stats.values.forEach((amount, i) => {
        const offset = stats.values.length === 1 ? 0 : -7 + 14 * i / (stats.values.length - 1);
        svg.append(svgNode("circle", {cx: x(amount), cy: 18 + offset, r: 2.3, fill: color, opacity: 0.36, class: "household-dot"}));
      });
      svg.append(svgNode("line", {x1: x(stats.mean - stats.sd), x2: x(stats.mean + stats.sd), y1: 18, y2: 18, stroke: color, "stroke-width": 1.7}));
      for (const endpoint of [stats.mean - stats.sd, stats.mean + stats.sd]) svg.append(svgNode("line", {x1: x(endpoint), x2: x(endpoint), y1: 13, y2: 23, stroke: color, "stroke-width": 1.4}));
      const mean = x(stats.mean);
      svg.append(svgNode("polygon", {points: `${mean},12 ${mean + 6},18 ${mean},24 ${mean - 6},18`, fill: color, class: "mean-marker"}));
    }
    const axis = node("div", undefined, "axis");
    const labels = node("div", undefined, "axis-labels");
    [low, (low + high) / 2, high].forEach(tick => labels.append(node("span", format(tick, metric === "comfort_pct" ? 0 : 2))));
    axis.append(labels); holder.append(axis);
    $("metric-unit").textContent = specification.unit;
    $("oracle-note").textContent = metric === "objective"
      ? `Dashed line: perfect-future objective ${format(group.oracle.bound.upper, 4)}. Certified interval [${format(group.oracle.bound.lower, 6)}, ${format(group.oracle.bound.upper, 6)}]. This future information is unavailable to a deployed controller.`
      : "The oracle knows future traces. Its bill and comfort describe that schedule; they are not separate optimal bounds.";
  }
  function meanChart(group) {
    const rows = group.rows;
    const holder = $("chart"); holder.replaceChildren();
    const means = rows.map(row => row.policy.metrics[metric].mean);
    const span = Math.max(...means) - Math.min(...means) || 0.001;
    const rawStep = span / 3;
    const power = 10 ** Math.floor(Math.log10(rawStep));
    const step = Math.max(metric === "comfort_pct" ? 0.1 : 0.01,
      [1, 2, 5, 10].find(multiple => multiple * power >= rawStep) * power);
    const low = Math.floor(Math.min(...means) / step) * step;
    const high = Math.max(low + step, Math.ceil(Math.max(...means) / step) * step);
    const ticks = Array.from({length: Math.round((high - low) / step) + 1}, (_, i) => low + i * step);
    const tickDigits = Math.max(0, -Math.floor(Math.log10(step)));
    const digits = metric === "comfort_pct" ? 3 : 4;
    for (const row of [...rows, group.oracle]) {
      const stats = row.policy.metrics[metric];
      const line = node("div", undefined, "chart-row"); line.dataset.method = row.id;
      const label = node("div", row.label, "chart-label" + (row.id === "tabpfn" ? " emphasized" : ""));
      const value = node("div", undefined, "chart-value");
      value.append(node("span", format(stats.mean, digits)), node("span", ` ± ${format(stats.sd, 3)}`, "sd"));
      if (row.id === "oracle") {
        line.classList.add("oracle-separate");
        label.append(node("span", "Perfect-future reference", "reference"));
        line.append(label, node("span", "", "oracle-space"), value);
      } else {
        const svg = svgNode("svg", {viewBox: "0 0 600 36", class: "track", role: "img", "aria-label": `${row.label}: mean ${stats.mean}, household SD ${stats.sd}`});
        const x = amount => 12 + 576 * (amount - low) / (high - low);
        for (const tick of ticks) svg.append(svgNode("line", {x1:x(tick),x2:x(tick),y1:0,y2:36,stroke:"#edf0e9"}));
        const color = row.id === "tabpfn" ? "#126452" : "#748797";
        const marker = svgNode("circle", {cx:x(stats.mean),cy:18,r:7,fill:color,class:"foundation-mean"});
        marker.append(svgNode("title", {})); marker.firstChild.textContent = `${row.label}: ${format(stats.mean, digits)} ± ${format(stats.sd, 3)}`;
        svg.append(marker); line.append(label, svg, value);
      }
      holder.append(line);
    }
    const axis = node("div", undefined, "axis"); const labels = node("div", undefined, "axis-labels");
    ticks.forEach(tick => labels.append(node("span", (metric === "energy_bill_without_dr" ? "$" : "") + format(tick, tickDigits))));
    axis.append(labels); holder.append(axis);
    $("metric-unit").textContent = measures[metric].unit;
    $("oracle-note").textContent = "The oracle knows future traces. Only its combined objective bounds attainable performance; its bill and comfort are components of that schedule.";
  }
  function comparisons(group) {
    const tabpfn = group.rows.find(row => row.id === "tabpfn").policy.metrics[metric].mean;
    const holder = $("comparisons");
    holder.replaceChildren();
    const names = {tabfm: "TabFM", tabicl: "TabICLv2", trees: "Extra Trees"};
    const labels = {objective: "objective", energy_bill_without_dr: "bill", comfort_pct: "comfort"};
    for (const id of (focused ? ["tabfm", "tabicl"] : ["tabfm", "tabicl", "trees"])) {
      const baseline = group.rows.find(row => row.id === id).policy.metrics[metric].mean;
      const delta = tabpfn - baseline;
      const relative = metric !== "comfort_pct" && baseline > 0;
      const amount = relative ? delta / baseline * 100 : delta;
      const digits = focused ? (relative ? 2 : 3) : (metric === "comfort_pct" || relative ? 1 : 3);
      const unit = metric === "comfort_pct" ? " pp" : relative ? "%" : metric === "energy_bill_without_dr" ? " $/day" : " units";
      const magnitude = Math.abs(amount);
      const sign = amount < 0 ? "−" : amount > 0 ? "+" : "";
      const number = magnitude > 0 && magnitude < 0.5 * 10 ** -digits ? `<${10 ** -digits}` : format(magnitude, digits);
      const card = node("article", undefined, "comparison-card");
      const better = metric === "comfort_pct" ? delta > 0 : delta < 0;
      card.dataset.outcome = delta === 0 ? "equal" : better ? "better" : "worse";
      card.dataset.comparator = id;
      card.dataset.delta = String(delta);
      card.dataset.scale = metric === "comfort_pct" ? "percentage-points" : relative ? "percent" : "absolute";
      const value = node("p", undefined, "comparison-value");
      value.append(node("span", `${sign}${number}`), node("span", unit, "comparison-unit"));
      card.append(node("h3", `TabPFN vs ${names[id]}`), value);
      card.append(node("p", `${delta < 0 ? "Lower" : delta > 0 ? "Higher" : "Same"} ${labels[metric]}`, "comparison-direction"));
      holder.append(card);
    }
  }
  function tables(group) {
    const body = $("values"); body.replaceChildren();
    for (const row of [...group.rows, group.oracle]) {
      const tr = node("tr", undefined, row.id === "oracle" ? "reference-row" : row.id === "tabpfn" ? "highlight" : undefined);
      tr.dataset.method = row.id;
      tr.append(node("td", row.label));
      for (const key of metricKeys) {
        const stats = row.policy.metrics[key];
        tr.append(node("td", `${format(stats.mean, measures[key].digits)} ± ${format(stats.sd, measures[key].digits)}`));
      }
      body.append(tr);
    }
    const forecasts = $("forecast-values"); forecasts.replaceChildren();
    for (const row of group.rows) {
      const tr = node("tr"); tr.append(node("td", row.label));
      for (const key of ["load", "pv", "temperature"]) {
        const stats = row.forecast?.[key];
        tr.append(node("td", stats && finite(stats.mean) && finite(stats.sd) ? `${format(stats.mean)} ± ${format(stats.sd)}` : "—"));
      }
      tr.append(node("td", finite(row.selected_episodes) ? String(row.selected_episodes) : "Varies by month")); forecasts.append(tr);
    }
    $("budget-note").textContent = $("period").value === "pooled"
      ? "Choose a month to inspect its fixed refit budgets. Full stopping receipts and completed selection episodes are included in the evidence JSON."
      : "Budgets were selected on the previous week, then refitted using all earlier training and validation days. No checkpoint was selected on this test month. A plateau criterion does not prove a global optimum.";
  }
  function render() {
    if (!evidence) return;
    const group = selectedGroup();
    const period = $("period").value === "pooled" ? "June–October" : monthLabel(group.id);
    $("period-description").textContent = `${group.home_ids.length} homes · ${period} · ${group.dates.length} days`;
    if (!focused) $("table-caption").textContent = `${$("period").value === "pooled" ? "All months" : monthLabel(group.id)} · all methods · household mean ± sample SD`;
    comparisons(group);
    const showSpread = !focused;
    document.querySelector(".legend").textContent = showSpread ? "Households · mean ± household SD" : "Points: means · labels: mean ± household SD";
    if (showSpread) chart(group); else meanChart(group);
    if (focused) {
      $("oracle-note").textContent = "Oracle knows the future. Only its combined objective is a performance bound—not its bill or comfort separately.";
    } else tables(group);
  }
  if (focused) {
    document.body.classList.add("foundation-view");
    document.querySelector("h1").textContent = "Foundation models. Real tradeoffs.";
    document.querySelector(".lead").textContent = "TabPFN-3.5, TabFM and TabICLv2 · the same controller · a perfect-future reference.";
    const note = $("oracle-note");
    note.className = "oracle-note";
    document.querySelector(".result-card").append(note);
    document.querySelectorAll(".supporting").forEach(panel => panel.remove());
    document.querySelectorAll("[data-metric]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.metric === metric)));
  }
  $("period").addEventListener("change", render);
  document.querySelectorAll("[data-metric]").forEach(button => button.addEventListener("click", () => {
    metric = button.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    render();
  }));
  window.addEventListener("resize", () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(render); });
  fetch("performance.json", {cache: "no-store"})
    .then(response => { if (!response.ok) throw new Error("Results not published"); return response.json(); })
    .then(validate)
    .then(data => {
      evidence = data;
      data.folds.forEach(fold => { const option = node("option", monthLabel(fold.id)); option.value = fold.id; $("period").append(option); });
      const requestedPeriod = new URLSearchParams(location.search).get("period");
      if (data.folds.some(fold => fold.id === requestedPeriod)) $("period").value = requestedPeriod;
      $("period").disabled = false;
      document.querySelectorAll("[data-metric]").forEach(button => { button.disabled = false; });
      if (!focused) $("finding").textContent = typeof data.finding === "string" ? data.finding : "See the evidence file for the complete recorded comparison.";
      $("fixture-note").hidden = data.fixture !== true;
      $("downloads").hidden = focused || data.fixture === true;
      $("unavailable").hidden = true;
      $("results").hidden = false;
      render();
    })
    .catch(() => {
      evidence = null;
      $("results").hidden = true;
      $("unavailable").hidden = false;
      $("period").disabled = true;
      document.querySelectorAll("[data-metric]").forEach(button => { button.disabled = true; });
      $("status-title").textContent = "Results are not published yet";
      $("status-detail").textContent = "The complete comparison will appear after training and verification finish.";
    });
})();
