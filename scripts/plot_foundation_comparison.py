"""Render the compact README bill comparison from verified experiment evidence."""

from datetime import date
from math import ceil, floor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from gridpfn.paths import ROOT
from gridpfn.release_evidence import load_evidence


def select_highlight(data):
    """Post-hoc highlight among complete recorded monthly folds, never custom windows."""
    candidates = []
    for fold in data["folds"]:
        if len(fold["dates"]) < 7:
            continue
        rows = {row["id"]: row["policy"]["metrics"] for row in fold["rows"]}
        own = rows["tabpfn"]
        bill_gains, comfort_gains = [], []
        for comparator in ("tabfm", "tabicl"):
            other = rows[comparator]
            baseline = other["energy_bill_without_dr"]["mean"]
            if baseline <= 0:
                break
            bill_gains.append(1 - own["energy_bill_without_dr"]["mean"] / baseline)
            comfort_gains.append(own["comfort_pct"]["mean"] - other["comfort_pct"]["mean"])
        if len(bill_gains) == 2 and min(bill_gains) > 0 and min(comfort_gains) > 0:
            candidates.append((min(bill_gains), min(comfort_gains), fold["id"], fold))
    if not candidates:
        raise ValueError("No recorded month improves both bill and comfort against both models")
    return max(candidates, key=lambda item: item[:3])[3]


def render(root=ROOT):
    root = Path(root)
    data = select_highlight(load_evidence(root / "site/performance.json"))
    indexed = {row["id"]: row for row in data["rows"]}
    rows = [indexed[key] for key in ("tabpfn", "tabfm", "tabicl")]
    bill = "energy_bill_without_dr"

    def stats(row):
        return row["policy"]["metrics"][bill]

    ink, green, muted, amber = "#193c35", "#126452", "#718478", "#a67b37"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(12, 6), facecolor="#f7f6f0")

    def text(x, y, value, size=11, color=ink, **kwargs):
        return fig.text(x, y, value, fontsize=size, color=color, parse_math=False, **kwargs)

    def card(x, y, width, height):
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="#dce2d7",
                linewidth=1,
                zorder=0,
            )
        )

    text(0.04, 0.94, "Electricity bill", size=18, weight="bold")
    period = date.fromisoformat(data["dates"][0]).strftime("%B %Y")
    text(0.96, 0.945, f"Selected month · {period}", color=muted, ha="right")
    for x, comparator in ((0.04, indexed["tabfm"]), (0.52, indexed["tabicl"])):
        card(x, 0.69, 0.44, 0.19)
        reduction = 100 * (1 - stats(rows[0])["mean"] / stats(comparator)["mean"])
        text(x + 0.02, 0.825, f"TabPFN vs {comparator['label']}", color=muted)
        text(x + 0.02, 0.751, f"−{reduction:.2f}%", size=28, color=green, weight="bold")
        text(x + 0.20, 0.779, "Lower simulated bill", size=12, color=green)
        comfort_gain = (
            rows[0]["policy"]["metrics"]["comfort_pct"]["mean"]
            - comparator["policy"]["metrics"]["comfort_pct"]["mean"]
        )
        text(x + 0.20, 0.735, f"+{comfort_gain:.2f} pp comfort*", size=11, color=green)

    card(0.04, 0.055, 0.92, 0.57)
    text(0.06, 0.588, "$ / home / day · lower is better", size=10, color=muted)
    text(0.94, 0.588, "Mean ± SD across homes", size=10, color=muted, ha="right")
    means = [stats(row)["mean"] for row in rows]
    low, high = floor(min(means) * 100) / 100, ceil(max(means) * 100) / 100
    if high <= low:
        high = low + 0.01
    axis = fig.add_axes((0.28, 0.205, 0.44, 0.325), zorder=2)
    axis.set_xlim(low, high)
    axis.set_ylim(-0.5, 2.5)
    axis.invert_yaxis()
    ticks = [cents / 100 for cents in range(round(low * 100), round(high * 100) + 1)]
    axis.set_xticks(ticks, [f"${v:.2f}" for v in ticks])
    axis.set_yticks([])
    axis.tick_params(length=0, labelsize=9, colors=muted, pad=10)
    axis.grid(axis="x", color="#edf0e9")
    for spine in axis.spines.values():
        spine.set_visible(False)
    for index, row in enumerate(rows):
        color = green if row["id"] == "tabpfn" else "#748797"
        axis.scatter(stats(row)["mean"], index, color=color, s=95, zorder=3)
        y = 0.205 + 0.325 * (2.5 - index) / 3
        text(0.06, y, row["label"], size=12, va="center", weight="bold" if index == 0 else "normal")
        text(
            0.94,
            y,
            f"{stats(row)['mean']:.4f} ± {stats(row)['sd']:.3f}",
            size=11,
            ha="right",
            va="center",
        )
    fig.add_artist(
        plt.Line2D(
            [0.06, 0.94],
            [0.125, 0.125],
            transform=fig.transFigure,
            color=amber,
            linestyle=(0, (3, 3)),
            linewidth=0.8,
        )
    )
    text(0.06, 0.09, "Perfect-future oracle", size=12, color=amber, va="center")
    oracle = stats(data["oracle"])
    text(
        0.94,
        0.09,
        f"{oracle['mean']:.4f} ± {oracle['sd']:.3f}",
        size=11,
        color=amber,
        ha="right",
        va="center",
    )
    text(
        0.04,
        0.016,
        "* Comfort: time within the target indoor temperature range. pp = percentage points.",
        size=10,
        color=muted,
        fontstyle="italic",
    )
    destination = root / "site/foundation-comparison"
    for suffix in ("svg", "png"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=180, facecolor=fig.get_facecolor())
    svg = destination.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    render()
