"use strict";
const utcDate = (iso) => new Date(`${iso}T00:00:00Z`);
const dateISO = (date) => date.toISOString().slice(0, 10);
const shiftDay = (iso, days) => {
  const d = utcDate(iso);
  d.setUTCDate(d.getUTCDate() + days);
  return dateISO(d);
};
const prettyDate = (iso) =>
  new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(utcDate(iso));
const periodLabel = (start, end) => {
  if (start === end) return prettyDate(start);
  if (start.slice(0, 7) === end.slice(0, 7))
    return `${Number(start.slice(8))}–${prettyDate(end)}`;
  if (start.slice(0, 4) === end.slice(0, 4))
    return `${prettyDate(start).slice(0, -5)} – ${prettyDate(end)}`;
  return `${prettyDate(start)} – ${prettyDate(end)}`;
};
const cashLabel = (cash) =>
  `${Math.abs(cash) < 0.005 ? "" : cash > 0 ? "+" : "−"}$${Math.abs(cash).toFixed(2)}`;
function decorateDay(button, day, iso) {
  if (!day) return;
  button.classList.add(
    "energy-day",
    day.comfort_band === null
      ? "comfort-unknown"
      : `comfort-${day.comfort_band}`,
  );
  const value = document.createElement("small");
  value.textContent = cashLabel(day.net_cash);
  button.append(value);
  const comfort =
    day.degree_hours === null
      ? `Comfort not measured · ${day.phase}`
      : day.degree_hours > 0
        ? `${day.comfort_pct.toFixed(1)}% of the day comfortable; ${day.degree_hours.toFixed(2)} degree-hours outside the target range`
        : "Comfortable all day";
  button.title = `${prettyDate(iso)} · ${cashLabel(day.net_cash)} ${day.net_cash > 0 ? "credit" : "cost"} · ${comfort}`;
  button.setAttribute("aria-label", button.title);
  if (day.degree_hours > 0) {
    const mark = document.createElement("span");
    mark.className = "comfort-mark";
    mark.textContent = "◔";
    mark.setAttribute("aria-hidden", "true");
    button.append(mark);
  }
}
class EnergyCalendar {
  constructor(onChange) {
    this.onChange = onChange;
    this.dates = [];
    this.mode = "day";
    document.getElementById("calendar-toggle").onclick = () => this.open();
    document.getElementById("calendar-close").onclick = () =>
      document.getElementById("calendar-dialog").close();
    document.getElementById("calendar-apply").onclick = () => this.apply();
    document.querySelectorAll("[data-period]").forEach(
      (b) =>
        (b.onclick = () => {
          this.mode = b.dataset.period;
          this.customStart = null;
          this.pick(this.anchor);
        }),
    );
    document.getElementById("calendar-prev").onclick = () => this.navigate(-1);
    document.getElementById("calendar-next").onclick = () => this.navigate(1);
    for (const id of ["calendar-month", "calendar-year"]) {
      document.getElementById(id).onchange = () => {
        this.anchor = `${document.getElementById("calendar-year").value}-${String(Number(document.getElementById("calendar-month").value) + 1).padStart(2, "0")}-01`;
        this.view = this.anchor;
        this.pick(this.anchor);
      };
    }
    for (const id of ["range-start", "range-end"]) {
      document.getElementById(id).onchange = () => {
        this.mode = "custom";
        this.start = document.getElementById("range-start").value;
        this.end = document.getElementById("range-end").value;
        this.render();
      };
    }
  }
  setDates(dates, initial) {
    this.dates = [...dates].sort();
    this.mode = "day";
    this.anchor = initial || this.dates[0];
    this.start = this.end = this.anchor;
    this.view = this.anchor;
    this.committed = { start: this.start, end: this.end, mode: this.mode };
    this.apply(false);
  }
  setOverview(data) {
    this.overview = data;
    this.overviewView = this.committed.start;
    this.plan = data.default_plan;
    const plans = document.getElementById("overview-plan");
    plans.replaceChildren();
    for (const p of data.plans) {
      const option = document.createElement("option");
      option.value = p.id;
      option.textContent = p.label;
      plans.append(option);
    }
    plans.value = this.plan;
    plans.onchange = () => {
      this.plan = plans.value;
      this.renderOverview();
      this.onChange();
    };
    document.getElementById("overview-method").textContent = data.comfort_scale;
    for (const [id, delta] of [
      ["overview-prev", -1],
      ["overview-next", 1],
    ])
      document.getElementById(id).onclick = () => {
        const d = utcDate(this.overviewView);
        d.setUTCDate(1);
        d.setUTCMonth(d.getUTCMonth() + delta);
        this.overviewView = dateISO(d);
        this.renderOverview();
      };
    document
      .getElementById("energy-overview")
      .setAttribute("aria-busy", "false");
    this.renderOverview();
  }
  planDays() {
    return this.overview?.plans.find((p) => p.id === this.plan)?.days || [];
  }
  renderOverview() {
    if (!this.overview) return;
    const view = utcDate(this.overviewView),
      year = view.getUTCFullYear(),
      month = view.getUTCMonth();
    document.getElementById("overview-month").textContent =
      new Intl.DateTimeFormat("en-GB", {
        month: "long",
        year: "numeric",
        timeZone: "UTC",
      }).format(view);
    const badge = document.getElementById("overview-evidence");
    badge.textContent = this.plan === "recorded" ? "Recorded" : "Simulated";
    badge.className = `tag ${this.plan === "recorded" ? "recorded" : "simulated"}`;
    const days = this.planDays(),
      lookup = new Map(days.map((d) => [d.date, d])),
      grid = document.getElementById("overview-days");
    grid.replaceChildren();
    const prefix = this.overviewView.slice(0, 7),
      included = days.filter((d) => d.date.startsWith(prefix));
    document.getElementById("overview-cash").textContent = included.length
      ? cashLabel(included.reduce((v, d) => v + d.net_cash, 0))
      : "—";
    document.getElementById("overview-comfort").textContent =
      this.plan === "recorded"
        ? "Not measured"
        : included.length
          ? `${included.filter((d) => d.degree_hours === 0).length} / ${included.length}`
          : "—";
    const offset = (new Date(Date.UTC(year, month, 1)).getUTCDay() + 6) % 7;
    for (let i = 0; i < offset; i++)
      grid.append(document.createElement("span"));
    for (
      let d = 1;
      d <= new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
      d++
    ) {
      const iso = dateISO(new Date(Date.UTC(year, month, d))),
        day = lookup.get(iso),
        button = document.createElement("button");
      button.type = "button";
      button.textContent = d;
      button.className = "calendar-day overview-day";
      button.disabled = !day;
      button.setAttribute("aria-label", `${prettyDate(iso)} · no results`);
      decorateDay(button, day, iso);
      button.classList.toggle(
        "overview-selected",
        iso >= this.committed.start && iso <= this.committed.end,
      );
      button.setAttribute(
        "aria-pressed",
        String(iso >= this.committed.start && iso <= this.committed.end),
      );
      button.onclick = () => {
        this.mode = "day";
        this.start = this.end = this.anchor = iso;
        this.apply(false);
      };
      grid.append(button);
    }
    const selected = lookup.get(this.committed.start),
      detail = document.getElementById("overview-detail");
    const selectedDays = days.filter(
      (d) => d.date >= this.committed.start && d.date <= this.committed.end,
    );
    if (this.committed.start !== this.committed.end) {
      detail.textContent = `${selectedDays.length} days selected · ${cashLabel(selectedDays.reduce((v, d) => v + d.net_cash, 0))} net energy balance`;
    } else if (selected) {
      detail.textContent = `${prettyDate(selected.date)} · ${cashLabel(selected.net_cash)} ${selected.net_cash > 0 ? "credit" : "cost"} · ${selected.degree_hours === null ? selected.phase : selected.degree_hours === 0 ? "Comfortable all day" : `${selected.comfort_pct.toFixed(0)}% of the day comfortable`}`;
    } else {
      detail.textContent =
        "No results for this plan on the selected date. Choose a highlighted day.";
    }
    document.getElementById("overview-compare").textContent =
      `Compare plans for this ${this.committed.start === this.committed.end ? "day" : "period"} ↗`;
  }
  open() {
    this.start = this.committed.start;
    this.end = this.committed.end;
    this.mode = this.committed.mode;
    this.anchor = this.start;
    this.view = this.start;
    this.customStart = null;
    this.render();
    document.getElementById("calendar-dialog").showModal();
  }
  navigate(delta) {
    const d = utcDate(this.view);
    d.setUTCDate(1);
    d.setUTCMonth(d.getUTCMonth() + delta);
    this.view = dateISO(d);
    this.render();
  }
  pick(iso) {
    this.anchor = iso;
    const d = utcDate(iso),
      year = d.getUTCFullYear(),
      month = d.getUTCMonth();
    if (this.mode === "week") {
      this.start = shiftDay(iso, -((d.getUTCDay() + 6) % 7));
      this.end = shiftDay(this.start, 6);
    } else if (this.mode === "month") {
      this.start = dateISO(new Date(Date.UTC(year, month, 1)));
      this.end = dateISO(new Date(Date.UTC(year, month + 1, 0)));
    } else if (this.mode === "year") {
      this.start = `${year}-01-01`;
      this.end = `${year}-12-31`;
    } else if (this.mode === "custom") {
      if (!this.customStart) {
        this.customStart = iso;
        this.start = this.end = iso;
      } else {
        [this.start, this.end] = [this.customStart, iso].sort();
        this.customStart = null;
      }
    } else this.start = this.end = iso;
    this.render();
  }
  render() {
    const view = utcDate(this.view),
      year = view.getUTCFullYear(),
      month = view.getUTCMonth();
    const months = document.getElementById("calendar-month");
    months.replaceChildren();
    for (let m = 0; m < 12; m++) {
      const option = document.createElement("option");
      option.value = m;
      option.textContent = new Intl.DateTimeFormat("en", {
        month: "long",
        timeZone: "UTC",
      }).format(new Date(Date.UTC(year, m, 1)));
      months.append(option);
    }
    months.value = month;
    const years = document.getElementById("calendar-year");
    years.replaceChildren();
    const first = Number(this.dates[0].slice(0, 4)),
      last = Number(this.dates.at(-1).slice(0, 4));
    for (let y = Math.min(first, year); y <= Math.max(last, year); y++) {
      const option = document.createElement("option");
      option.value = y;
      option.textContent = y;
      years.append(option);
    }
    years.value = year;
    document.querySelectorAll("[data-period]").forEach((b) => {
      b.classList.toggle("active", b.dataset.period === this.mode);
      b.setAttribute("aria-pressed", String(b.dataset.period === this.mode));
    });
    const grid = document.getElementById("calendar-days");
    grid.replaceChildren();
    const firstDay = new Date(Date.UTC(year, month, 1)),
      offset = (firstDay.getUTCDay() + 6) % 7;
    for (let i = 0; i < offset; i++)
      grid.append(document.createElement("span"));
    const available = new Set(this.dates);
    for (
      let day = 1;
      day <= new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
      day++
    ) {
      const iso = dateISO(new Date(Date.UTC(year, month, day))),
        button = document.createElement("button");
      button.type = "button";
      button.textContent = day;
      button.className = "calendar-day";
      button.disabled = !available.has(iso);
      button.title =
        prettyDate(iso) +
        (available.has(iso) ? " · results available" : " · no results");
      button.setAttribute("aria-label", button.title);
      button.classList.toggle("in-range", iso >= this.start && iso <= this.end);
      button.classList.toggle(
        "range-edge",
        iso === this.start || iso === this.end,
      );
      button.setAttribute(
        "aria-pressed",
        String(iso >= this.start && iso <= this.end),
      );
      button.onclick = () => this.pick(iso);
      grid.append(button);
      decorateDay(
        button,
        this.planDays().find((d) => d.date === iso),
        iso,
      );
    }
    document.getElementById("range-start").value = this.start;
    document.getElementById("range-end").value = this.end;
    const count = this.dates.filter(
      (d) => d >= this.start && d <= this.end,
    ).length;
    const days = (utcDate(this.end) - utcDate(this.start)) / 86400000 + 1;
    const valid = Number.isFinite(days) && days > 0 && days <= 366 && count > 0;
    document.getElementById("calendar-apply").disabled = !valid;
    document.getElementById("calendar-coverage").textContent = valid
      ? `${count} of ${days} selected days have results.${count < days ? " Missing days won't be estimated." : " All selected days are available."}`
      : "Choose a period of up to a year that includes available results.";
    document.getElementById("calendar-plan").textContent = this.overview
      ? `${this.overview.plans.find((p) => p.id === this.plan).label} · ${this.plan === "recorded" ? "recorded" : "simulated"} · USD`
      : "";
  }
  apply(close = true) {
    this.committed = { start: this.start, end: this.end, mode: this.mode };
    document.getElementById("date").value = this.start;
    document.getElementById("end-date").value = this.end;
    document.getElementById("calendar-selection").textContent = periodLabel(
      this.start,
      this.end,
    );
    document.getElementById("calendar-selection-mode").textContent = {
      day: "Day",
      week: "Week",
      month: "Month",
      year: "Year",
      custom: "Custom period",
    }[this.mode];
    if (close) document.getElementById("calendar-dialog").close();
    this.renderOverview();
    this.onChange();
  }
}
