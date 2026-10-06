"""Verify the authentic seasonal evidence viewer at desktop and mobile widths."""

import argparse
import json
import math
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# Support both direct script execution and python -m scripts.<name>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridpfn.release_evidence import validate_evidence

METRICS = ("energy_bill_without_dr", "comfort_pct", "objective")


def check_page(page, data, width):
    checks = []
    page.set_viewport_size({"width": width, "height": 1000})
    page.locator("#results").wait_for(state="visible")
    assert page.locator("#fixture-note").is_hidden()
    assert page.locator("#downloads").is_visible()
    assert page.locator("#period option").count() == 6
    page.locator("#numeric-details summary").click()
    for period in ["pooled"] + [fold["id"] for fold in data["folds"]]:
        page.select_option("#period", period)
        group = data if period == "pooled" else next(f for f in data["folds"] if f["id"] == period)
        for metric in METRICS:
            page.locator(f'[data-metric="{metric}"]').click()
            assert page.locator("#values tr").count() == 7
            assert page.locator(".chart-row").count() == 7
            assert page.locator(".household-dot").count() == 175
            assert page.locator(".mean-marker").count() == 7
            assert page.locator(".oracle-bound").count() == (7 if metric == "objective" else 0)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            tabpfn = next(r for r in group["rows"] if r["id"] == "tabpfn")["policy"]["metrics"][
                metric
            ]["mean"]
            assert page.locator(".comparison-card").count() == 3
            for comparator in ("tabfm", "tabicl", "trees"):
                baseline = next(r for r in group["rows"] if r["id"] == comparator)["policy"][
                    "metrics"
                ][metric]["mean"]
                delta = tabpfn - baseline
                relative = metric != "comfort_pct" and baseline > 0
                amount = delta / baseline * 100 if relative else delta
                digits = 1 if relative or metric == "comfort_pct" else 3
                card = page.locator(f'[data-comparator="{comparator}"]')
                assert math.isclose(float(card.get_attribute("data-delta")), delta, abs_tol=1e-10)
                scale = (
                    "percentage-points"
                    if metric == "comfort_pct"
                    else "percent"
                    if relative
                    else "absolute"
                )
                assert card.get_attribute("data-scale") == scale
                unit = (
                    " pp"
                    if metric == "comfort_pct"
                    else "%"
                    if relative
                    else " $/day"
                    if metric == "energy_bill_without_dr"
                    else " units"
                )
                displayed = card.locator(".comparison-value").inner_text()
                assert displayed.endswith(unit)
                number = displayed.removesuffix(unit)
                sign = "−" if amount < 0 else "+" if amount > 0 else ""
                assert not sign or number.startswith(sign)
                number = number.removeprefix(sign)
                if number.startswith("<"):
                    assert 0 < abs(amount) < float(number[1:])
                else:
                    assert abs(float(number) - abs(amount)) <= 0.50001 * 10**-digits
                direction = "Lower" if delta < 0 else "Higher" if delta > 0 else "Same"
                assert card.locator(".comparison-direction").inner_text().startswith(direction)
            for row in group["rows"] + [group["oracle"]]:
                digits = 1 if metric == "comfort_pct" else 3
                stats = row["policy"]["metrics"][metric]
                mean = f"{stats['mean']:.{digits}f}"
                expected = f"{mean} ± {stats['sd']:.{digits}f}"
                assert (
                    page.locator(f'#values tr[data-method="{row["id"]}"] td')
                    .nth(METRICS.index(metric) + 1)
                    .inner_text()
                    == expected
                )
                assert (
                    page.locator(
                        f'.chart-row[data-method="{row["id"]}"] .chart-value span'
                    ).first.inner_text()
                    == mean
                )
            checks.append(
                {"width": width, "period": period, "metric": metric, "values_match": True}
            )
    page.locator("#numeric-details summary").click()
    page.select_option("#period", "pooled")
    page.locator('[data-metric="objective"]').click()
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767", help="Static site base URL")
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--output", type=Path, default=Path("wiki/seasonal/browser"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    with sync_playwright() as playwright:
        options = {"headless": True}
        if args.browser:
            options["executable_path"] = str(args.browser)
        browser = playwright.chromium.launch(**options)
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.request.get(base + "/performance.json")
        if not response.ok:
            raise ValueError("Complete seasonal evidence is required before browser release checks")
        data = validate_evidence(response.json())
        for width in [390, 768, 1440]:
            page.set_viewport_size({"width": width, "height": 1000})
            page.goto(base + "/performance.html", wait_until="networkidle")
            checks.extend(check_page(page, data, width))
            page.screenshot(path=str(args.output / f"performance-{width}.png"), full_page=True)
        for name in ("performance.svg", "performance.pdf"):
            result = page.request.get(base + "/" + name)
            assert result.ok and result.body(), f"Missing figure download: {name}"
        browser.close()
    assert not errors, errors
    (args.output / "verification.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "javascript_errors": errors,
                "real_evidence": True,
                "frozen_selection_sha256": data["frozen_selection_sha256"],
            },
            indent=2,
        )
    )
    print(f"{len(checks)} responsive seasonal checks passed; no JavaScript errors.")


if __name__ == "__main__":
    main()
