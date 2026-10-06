"""Capture a 20-second walkthrough of the real app; generated evidence only.

Start the example app, then run this with --url. Dependencies: Playwright/Chromium,
ffmpeg. Numeric answers are never mocked; model waits are omitted between frames.
"""

import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def record(url, output):
    with urlopen(url + "/api/catalog", timeout=30) as response:
        catalog = json.load(response)
    if catalog["provenance"]["kind"] != "synthetic":
        raise ValueError("Public recordings must use generated household evidence")
    output.mkdir(parents=True, exist_ok=False)
    frames = output / "frames"
    frames.mkdir()
    count = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1180, "height": 820}, reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_selector(".overview-day.energy-day")
        # Stable calendar date for the shipped generated-data example.
        page.locator('.overview-day[aria-label^="22 Jul 2019"]').click()

        def capture(n):
            nonlocal count
            for _ in range(n):
                page.screenshot(path=str(frames / f"{count:04d}.png"))
                count += 1

        def ask(question):
            field = page.locator("#question")
            field.fill("")
            for i in range(1, 9):
                field.fill(question[: round(len(question) * i / 8)])
                capture(1)
            page.locator("#send").click()
            expect(page.locator("#send")).to_be_enabled(timeout=180000)
            assert not page.locator("#error").inner_text()
            answer = page.locator(".message:not(.user)").last
            answer.scroll_into_view_if_needed()
            page.wait_for_timeout(250)
            return answer

        # 0–4s: day/week selection and honest coverage.
        capture(12)
        page.locator("#calendar-toggle").click()
        page.locator('[data-period="week"]').click()
        capture(18)
        page.locator("#calendar-apply").click()
        capture(18)
        # 4–8s: real tool-backed cost accounting.
        answer = ask("Which appliances caused my electricity bill this week?")
        answer.locator(".card").first.scroll_into_view_if_needed()
        capture(40)
        # 8–12s: a dated forecast-based appliance plan.
        page.locator("#calendar-toggle").click()
        page.locator('[data-period="day"]').click()
        page.locator("#calendar-apply").click()
        answer = ask("When should I cool my home and do laundry?")
        answer.locator(".card").first.scroll_into_view_if_needed()
        capture(40)
        # 12–16s: real solar attribution, linked to observed inputs.
        inspector = answer.locator(".forecast-inspector").first
        inspector.locator("summary").first.click()
        inspector.get_by_label("Forecast to explain", exact=True).select_option("solar")
        inspector.get_by_role("button", name="Explain", exact=True).click()
        expect(inspector.locator(".influence-waterfall")).to_be_visible(timeout=180000)
        inspector.scroll_into_view_if_needed()
        capture(24)
        inspector.locator(".influence-waterfall").scroll_into_view_if_needed()
        capture(24)
        # 16–20s: inspect alternatives and take away the actual report.
        answer = ask("Compare energy plans for this day.")
        answer.locator(".card").first.scroll_into_view_if_needed()
        capture(28)
        with page.expect_download(timeout=180000) as info:
            page.locator("#export").click()
        download = info.value
        download.save_as(output / "example-report.html")
        assert "Your energy report" in (output / "example-report.html").read_text()
        capture(12)
        assert count == 240
        assert not errors, errors
        browser.close()
    (output / "capture.json").write_text(
        json.dumps(
            {
                "frames": count,
                "fps": 12,
                "duration_seconds": 20,
                "evidence_sha256": catalog["sha256"],
                "data": "generated",
                "real_tool_responses": True,
                "waits_shortened": True,
                "page_errors": errors,
            },
            indent=2,
        )
    )
    return frames


def render(frames, destination):
    destination.mkdir(parents=True, exist_ok=True)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    # A separate video caption band; the app's own UI is unchanged.
    phrases = [
        "Explore a week of costs and comfort",
        "See which appliances cost the most",
        "Plan cooling and laundry with TabPFN",
        "Inspect the solar forecast with SHAP",
        "Compare plans. Take your report.",
    ]
    overlay = "pad=iw:ih+58:0:58:color=0x173f35"
    for i, phrase in enumerate(phrases):
        overlay += f",drawtext=fontfile={font}:text='{phrase}':fontcolor=white:fontsize=23:x=22:y=10:enable='gte(t,{i * 4})*lt(t,{(i + 1) * 4})'"
    overlay += f",drawtext=fontfile={font}:text='Generated home / real inference / edited waits':fontcolor=0xbcd5c7:fontsize=12:x=22:y=38"
    mp4 = destination / "gridpfn-assistant.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-i",
            str(frames / "%04d.png"),
            "-vf",
            overlay,
            "-threads",
            "2",
            "-c:v",
            "libx264",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-filter_complex",
            "fps=10,scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-threads",
            "2",
            "-loop",
            "0",
            str(destination / "gridpfn-assistant.gif"),
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8774")
    parser.add_argument("--output", type=Path, default=Path("results/assistant_walkthrough"))
    parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    render(record(args.url, args.output), args.assets)
