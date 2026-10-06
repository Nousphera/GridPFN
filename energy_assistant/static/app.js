"use strict";
const $ = (id) => document.getElementById(id);
const state = { catalog: null, busy: false, history: [] };
const calendar = new EnergyCalendar(() => updateExport());
function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function svgNode(tag, attrs, text) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
function svgBase(label) {
  const s = svgNode("svg", {
    viewBox: "0 0 620 190",
    role: "img",
    "aria-label": label,
    class: "chart",
  });
  return s;
}
function money(v) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(v);
}
function chart(
  values,
  labels,
  baseline = null,
  kind = "line",
  description = null,
) {
  const s = svgBase(
    description ||
      (kind === "bars"
        ? "Hourly cost in US dollars"
        : "Hourly fixed household demand in kWh"),
  );
  const max = Math.max(0.01, ...values, ...(baseline || [])) * 1.15,
    min = Math.min(0, ...values) * 1.15;
  const X = (i) => 43 + i * (555 / Math.max(1, values.length - 1)),
    Y = (v) => 155 - ((v - min) / (max - min)) * 134;
  for (let i = 0; i < 4; i++) {
    const value = min + ((max - min) * i) / 3;
    s.append(
      svgNode("line", {
        x1: 38,
        x2: 604,
        y1: Y(value),
        y2: Y(value),
        class: "axis",
      }),
    );
    s.append(
      svgNode(
        "text",
        { x: 32, y: Y(value) + 3, "text-anchor": "end" },
        value.toFixed(2),
      ),
    );
  }
  if (kind === "bars") {
    values.forEach((v, i) =>
      s.append(
        svgNode("rect", {
          x: X(i) - 7,
          y: Math.min(Y(v), Y(0)),
          width: 14,
          height: Math.max(1, Math.abs(Y(v) - Y(0))),
          rx: 3,
          fill: v >= 0 ? "#318670" : "#aec9b8",
        }),
      ),
    );
  } else {
    if (baseline)
      s.append(
        svgNode("polyline", {
          points: baseline.map((v, i) => `${X(i)},${Y(v)}`).join(" "),
          class: "line baseline",
        }),
      );
    s.append(
      svgNode("polyline", {
        points: values.map((v, i) => `${X(i)},${Y(v)}`).join(" "),
        class: "line",
      }),
    );
  }
  labels.forEach((v, i) => {
    if (i % Math.ceil(values.length / 6) === 0 || i === values.length - 1)
      s.append(
        svgNode(
          "text",
          { x: X(i), y: 178, "text-anchor": "middle" },
          String(v).includes("-")
            ? String(v).slice(5)
            : `${String(v).padStart(2, "0")}:00`,
        ),
      );
  });
  return s;
}
function card(title, evidence) {
  const c = node("div", undefined, "card"),
    h = node("div", undefined, "card-head");
  h.append(node("h3", title), node("span", evidence, `tag ${evidence}`));
  c.append(h);
  return c;
}
function note(c, text) {
  c.append(node("p", text, "card-note"));
}
function details(parent, title, paragraphs) {
  const panel = node("details", undefined, "explanation-details");
  panel.append(node("summary", title));
  for (const text of paragraphs.filter(Boolean)) note(panel, text);
  parent.append(panel);
}
function renderCard(tool, d, view) {
  const v = view || { title: "Your energy", note: "" };
  if (d.available === false) {
    const c = card(v.title, "recorded");
    note(c, v.summary);
    return c;
  }
  if (["appliance_breakdown", "schedule_comparison", "plan_day"].includes(tool))
    return renderInsightCard(tool, d, v);
  if (tool === "explain_bill") {
    const c = card(v.title, "recorded"),
      stat = node("div", money(d.total), "stat");
    stat.append(
      node(
        "small",
        d.period_selection
          ? `${d.period_selection.available_days} days with results`
          : `${d.date} · this day's cost`,
      ),
    );
    c.append(
      stat,
      d.daily_costs
        ? chart(
            d.daily_costs.map((r) => r.cost),
            d.daily_costs.map((r) => r.date),
            null,
            "bars",
            "Daily electricity cost in US dollars",
          )
        : chart(
            d.hours.map((r) => r.net_cost),
            d.hours.map((r) => r.hour),
            null,
            "bars",
          ),
    );
    const p = d.period;
    c.append(
      node(
        "div",
        `${d.period_days} days: ${money(p.import_cost)} electricity − ${money(p.export_credit)} solar credits + ${money(p.fixed_cost)} fixed charges = ${money(p.period_total)}`,
        "checkline",
      ),
    );
    note(c, v.note);
    details(c, "How the costs are calculated", [d.formula, d.note]);
    return c;
  }
  if (tool === "forecast_and_explain") {
    const c = card(v.title, "predicted");
    c.append(
      node("div", "Everyday electricity use · kWh per hour", "chart-label"),
    );
    c.append(
      chart(
        d.rows.map((r) => r.load_kwh),
        d.rows.map((r) => r.hour),
        d.rows.map((r) => r.seasonal_load_kwh),
      ),
    );
    const legend = node("div", undefined, "legend");
    legend.append(node("span", d.model), node("span", "Past daily patterns"));
    c.append(legend);
    const technical = [];
    if (d.explanation) {
      const e = d.explanation;
      c.append(
        node("div", "What nudges the forecast up or down?", "shap-title"),
      );
      note(
        c,
        `Start with ${e.baseline_kwh.toFixed(2)} kWh from past examples. These factors bring the estimate to ${e.prediction_kwh.toFixed(2)} kWh.`,
      );
      c.append(
        influenceWaterfall({
          baseline: e.baseline_kwh,
          prediction: e.prediction_kwh,
          unit: "kWh",
          factors: e.contributions.map((r) => ({
            label: r.feature,
            value: r.kwh,
          })),
        }),
      );
      technical.push(
        `${e.method}. ${e.model_queries} model queries; reconstruction error ${e.additivity_error.toExponential(1)} kWh.`,
        e.caveat,
      );
    }
    note(c, v.note);
    if (d.runtime)
      technical.push(
        `Computed locally in ${d.runtime.seconds.toFixed(1)} seconds. ${d.runtime.reused_fitted_context ? "Reused the fitted household context." : "Fitted a new household context."}`,
      );
    technical.push(d.note);
    details(c, "About this forecast", technical);
    const inspection = node("div"),
      dates = d.period_selection?.dates || [d.date];
    const show = (date) =>
      inspection.replaceChildren(
        forecastInspector(
          d.home,
          date,
          d.origin_hour,
          d.target_hour - d.origin_hour,
          dates.length === 1 ? d.rows : null,
        ),
      );
    if (dates.length > 1) c.append(resultDateSelector(dates, show));
    c.append(inspection);
    show(dates[0]);
    return c;
  }
  if (tool === "compare_schedules") {
    const c = card(v.title, "simulated"),
      wrap = node("div", undefined, "table-wrap"),
      table = node("table"),
      head = node("tr");
    [
      "Energy plan",
      d.period_selection ? "Period cost" : "Day cost",
      "Comfort",
      "Car / laundry",
      d.period_selection ? "Avg. battery" : "Battery left",
    ].forEach((t) => head.append(node("th", t)));
    const thead = node("thead");
    thead.append(head);
    table.append(thead);
    const tbody = node("tbody");
    for (const r of d.candidates) {
      const friendly = (v.plans || []).find((p) => p.id === r.id) || {
        label: r.label,
        concerns: r.reasons,
      };
      const tr = node(
          "tr",
          undefined,
          r.id === d.recommended ? "selected" : "",
        ),
        name = node("td", friendly.label);
      name.append(
        node(
          "small",
          r.id === "feedback"
            ? "Our starting point"
            : r.eligible
              ? r.id === d.recommended
                ? "Suggested for this test"
                : "Meets the comparison checks"
              : friendly.concerns.join(" "),
        ),
      );
      tr.append(
        name,
        node("td", money(r.bill)),
        node("td", `${r.comfort_pct.toFixed(0)}%`),
        node(
          "td",
          `${r.ev_complete ? "✓" : "✕"} / ${r.washer_complete ? "✓" : "✕"}`,
        ),
        node("td", `${(100 * r.terminal_battery_soc).toFixed(0)}%`),
      );
      tbody.append(tr);
    }
    table.append(tbody);
    wrap.append(table);
    c.append(wrap);
    note(
      c,
      "Comfort is the share of the day within the chosen temperature range. ✓ means charging or laundry finishes.",
    );
    note(c, v.note);
    const target =
      d.candidates.find((r) => r.id === d.recommended) ||
      d.candidates.find((r) => r.id === "tabpfn") ||
      d.candidates[0];
    const label =
      (v.plans || []).find((p) => p.id === target.id)?.label || target.label;
    const schedule = node("details", undefined, "explanation-details");
    schedule.append(
      node(
        "summary",
        `${d.period_selection ? "See daily costs" : "See when the car charges"} · ${label}`,
      ),
    );
    schedule.append(
      d.period_selection
        ? chart(
            target.daily_costs.map((r) => r.cost),
            target.daily_costs.map((r) => r.date),
            null,
            "bars",
            "Daily simulated cost in US dollars",
          )
        : chart(
            target.actions.map((r) => r.ev_kw),
            target.actions.map((r) => r.hour),
            null,
            "bars",
            "Car charging power in kilowatts",
          ),
    );
    c.append(schedule);
    if (state.catalog?.live_forecast) {
      const inspector = node("div"),
        dates = d.period_selection?.dates || [d.date];
      const show = (date) =>
        inspector.replaceChildren(forecastInspector(d.home, date, 12, 6));
      if (dates.length > 1) c.append(resultDateSelector(dates, show));
      c.append(inspector);
      show(dates[0]);
    }
    details(c, "What we checked", [
      "A suggested plan must cost less than our starting point, finish charging and laundry, keep comfort at least as good, and leave at least as much energy in the battery. Comfort can still be below 100% even when it is no worse than the starting point.",
      d.interpretation,
      d.limits,
    ]);
    return c;
  }
  if (tool === "export_plan") {
    const c = card(v.title, "recorded"),
      a = node("a", "↓ Download my report", "link-button");
    a.href = `/api/report/${encodeURIComponent(d.home)}/${encodeURIComponent(d.date)}?end_date=${encodeURIComponent(d.period_selection?.end || d.date)}`;
    if (d.selected_plan)
      a.href += `&plan=${encodeURIComponent(d.selected_plan)}`;
    a.href += `&hour=${d.planning_hour ?? 6}&format=html`;
    a.download = "";
    c.append(a);
    note(c, v.note);
    details(c, "Report reference", [`SHA-256 ${d.evidence_sha256}`, d.note]);
    return c;
  }
  const c = card(v.title, "recorded"),
    facts = node("div", undefined, "facts");
  for (const [name, value] of [
    ["Home history", `${d.training_days} days of readings`],
    ["Looks ahead with", d.forecaster],
    [
      "Data",
      d.data_kind === "synthetic"
        ? "Generated example data"
        : "Household research data",
    ],
    [
      "Personalized plan",
      d.personalized_policy ? "Available to compare" : "Not added yet",
    ],
  ]) {
    const f = node("div", value);
    f.prepend(node("small", name));
    facts.append(f);
  }
  c.append(facts);
  note(c, v.note);
  details(c, "About this home", [d.limits, d.execution]);
  return c;
}
function updateSelection() {
  const entry = state.catalog.homes.find((h) => h.id === $("home").value);
  calendar.setDates(entry.dates, entry.dates.at(-1));
}
function periodURL(home, date) {
  const params = new URLSearchParams({
    format: "html",
    end_date: $("end-date").value,
    hour: $("planning-hour").value,
  });
  if (calendar.plan && calendar.plan !== "recorded")
    params.set("plan", calendar.plan);
  return `/api/report/${encodeURIComponent(home)}/${encodeURIComponent(date)}?${params}`;
}
function updateExport() {
  state.history = [];
  $("export").href = periodURL($("home").value, $("date").value);
}
function busy(value) {
  state.busy = value;
  for (const id of [
    "send",
    "question",
    "calendar-toggle",
    "settings-open",
    "overview-prev",
    "overview-next",
    "overview-plan",
    "overview-compare",
    "planning-hour",
  ])
    $(id).disabled = value;
  document
    .querySelectorAll("[data-question]")
    .forEach((b) => (b.disabled = value));
  $("energy-overview").inert = value;
}
async function ask(message) {
  if (state.busy || !message.trim()) return;
  busy(true);
  $("error").textContent = "";
  const home = $("home").value,
    date = $("date").value,
    end_date = $("end-date").value;
  const user = node("div", undefined, "message user");
  user.append(node("p", message));
  $("conversation").append(user);
  const answer = node("div", undefined, "message"),
    label = node("div", undefined, "assistant-label");
  label.append(
    node("span", "⌂"),
    document.createTextNode(
      `GridPFN Home · ${home} · ${date === end_date ? prettyDate(date) : `${prettyDate(date)} – ${prettyDate(end_date)}`}`,
    ),
  );
  const trace = node("div", undefined, "trace");
  trace.setAttribute("role", "status");
  trace.setAttribute("aria-live", "polite");
  answer.append(label, trace);
  $("conversation").append(answer);
  answer.scrollIntoView({ block: "start", behavior: "smooth" });
  let finished = false;
  function event(e) {
    if (e.event === "error") throw Error(e.text);
    if (e.event === "status")
      trace.append(node("div", e.text, "trace-row pending"));
    if (e.event === "tool_start") {
      trace
        .querySelectorAll(".pending")
        .forEach((n) => (n.className = "trace-row done"));
      const r = node("div", e.text, "trace-row pending");
      r.dataset.tool = e.tool;
      trace.append(r);
    }
    if (e.event === "tool_done") {
      const r = trace.querySelector(`[data-tool="${e.tool}"]`);
      if (r) r.className = "trace-row done";
    }
    if (e.event === "answer") {
      finished = true;
      trace
        .querySelectorAll(".pending")
        .forEach((n) => (n.className = "trace-row done"));
      answer.append(node("div", e.text, "answer-text"));
      for (const c of e.cards)
        answer.append(renderCard(c.tool, c.data, c.presentation));
      if (e.notice) answer.append(node("p", e.notice, "tool-note"));
      answer.append(reasoningPanel(e, trace));
      appendFollowups(answer, e.cards);
      requestAnimationFrame(() =>
        answer.scrollIntoView({ block: "start", behavior: "smooth" }),
      );
      $("mode").textContent =
        e.mode === "llm"
          ? state.catalog.chat_provider === "local"
            ? "Local chat · Qwen3.5-2B"
            : "Chat mode"
          : "Guided mode";
    }
  }
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        home,
        date,
        end_date,
        history: state.history.slice(-4),
        plan: calendar.plan === "recorded" ? null : calendar.plan,
        hour: Number($("planning-hour").value),
      }),
    });
    if (!response.ok)
      throw Error(
        "The request could not be completed. Check the selected home and date.",
      );
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) if (line.trim()) event(JSON.parse(line));
    }
    if (buffer.trim()) event(JSON.parse(buffer));
    if (!finished)
      throw Error(
        "The response ended before the tools completed. Please try again.",
      );
    state.history.push(message);
  } catch (error) {
    trace
      .querySelectorAll(".pending")
      .forEach((n) => (n.className = "trace-row failed"));
    $("error").textContent = error.message;
  } finally {
    busy(false);
    $("question").focus({ preventScroll: true });
  }
}
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("question").value;
  $("question").value = "";
  ask(text);
});
document
  .querySelectorAll("[data-question]")
  .forEach((b) => b.addEventListener("click", () => ask(b.dataset.question)));
async function init() {
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw Error("Evidence could not be loaded.");
    state.catalog = await response.json();
    $("home").value = state.catalog.homes[0].id;
    $("home-name").textContent = `Home ${$("home").value}`;
    $("model").textContent = state.catalog.forecaster;
    $("mode").textContent =
      state.catalog.chat_provider === "local"
        ? "Local chat · Qwen3.5-2B"
        : state.catalog.agent_mode === "llm"
          ? "Chat mode · ready to connect"
          : "Guided mode";
    $("mode").title =
      state.catalog.agent_mode === "llm"
        ? "A configured language model helps choose how to answer."
        : "Built-in question routing; no language model connected.";
    updateSelection();
    const overviewResponse = await fetch("/api/overview");
    if (!overviewResponse.ok) throw Error("Calendar could not be loaded");
    calendar.setOverview(await overviewResponse.json());
    busy(false);
    $("export").classList.remove("disabled");
    $("export").removeAttribute("aria-disabled");
  } catch (error) {
    $("error").textContent =
      "We couldn’t load this home’s information. Please ask the person who set up the demo to check it.";
  }
}
init();

function providerFields() {
  const provider = $("chat-provider").value;
  $("provider-fields").hidden = ["local", "guided"].includes(provider);
  $("endpoint-field").hidden = ["openai", "anthropic"].includes(provider);
  $("provider-note").textContent =
    provider === "local"
      ? "Runs on this computer. Your questions stay here."
      : provider === "guided"
        ? "Uses built-in question matching. You can explore every energy feature without an LLM."
        : "Your provider receives your question. API use may be charged to your account.";
}
$("settings-open").onclick = async () => {
  try {
    const r = await fetch("/api/settings");
    if (!r.ok) throw Error("Settings are unavailable.");
    const d = await r.json();
    $("chat-provider").value = d.provider;
    $("chat-provider").querySelector('option[value="local"]').disabled =
      !d.local_available;
    $("chat-model").value = d.model;
    $("chat-endpoint").value = d.endpoint;
    $("chat-key").value = "";
    $("clear-chat-key").checked = false;
    $("settings-message").textContent = d.has_key
      ? "A key is available on this computer."
      : "";
    providerFields();
    $("settings-dialog").showModal();
  } catch (e) {
    $("error").textContent = e.message;
  }
};
$("settings-close").onclick = () => $("settings-dialog").close();
$("chat-provider").onchange = () => {
  providerFields();
  if (["openai", "anthropic"].includes($("chat-provider").value)) {
    $("chat-model").value = "";
    $("chat-endpoint").value =
      $("chat-provider").value === "openai"
        ? "https://api.openai.com/v1/responses"
        : "https://api.anthropic.com/v1/messages";
  }
  $("chat-key").value = "";
};
$("settings-form").onsubmit = async (e) => {
  e.preventDefault();
  $("settings-message").textContent = "Saving…";
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Settings-Token": state.catalog.settings_token,
      },
      body: JSON.stringify({
        provider: $("chat-provider").value,
        model: $("chat-model").value,
        endpoint: $("chat-endpoint").value,
        api_key: $("chat-key").value,
        clear_key: $("clear-chat-key").checked,
      }),
    });
    const d = await response.json();
    if (!response.ok) throw Error(d.detail || "Could not save settings.");
    state.catalog.chat_provider = d.provider;
    $("mode").textContent =
      d.provider === "local"
        ? "Local chat · Qwen3.5-2B"
        : d.provider === "guided"
          ? "Guided mode"
          : "Chat mode · ready to connect";
    $("chat-key").value = "";
    $("settings-message").textContent =
      "Saved. Your next question will use these settings.";
  } catch (e) {
    $("settings-message").textContent = e.message;
  }
};

$("overview-compare").onclick = () =>
  ask(
    "Show side by side what could have been done differently to reduce my bill.",
  );

$("planning-hour").onchange = () => updateExport();
