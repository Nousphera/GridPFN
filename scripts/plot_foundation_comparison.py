"""Render a focused, fully sourced foundation-model comparison for the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridpfn.paths import ROOT
from gridpfn.release_evidence import load_evidence


def render(root=ROOT):
    root = Path(root)
    data = load_evidence(root / "site/performance.json")
    indexed = {row["id"]: row for row in data["rows"]}
    rows = [indexed[k] for k in ("tabpfn", "tabicl", "tabfm")]
    oracle = data["oracle"]
    bill = "energy_bill_without_dr"

    def metric(row, key):
        return row["policy"]["metrics"][key]

    colors = ["#126452", "#678396", "#8a91a1"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10})
    fig = plt.figure(figsize=(12, 6.4), facecolor="#ffffff")
    fig.text(
        0.06,
        0.935,
        "Lower bills among the foundation models",
        fontsize=22,
        weight="bold",
        color="#193c35",
    )
    fig.text(
        0.06, 0.885, "25 homes · 152 test days · June–October 2019", fontsize=11, color="#62716d"
    )
    for x, key in ((0.06, "tabfm"), (0.52, "tabicl")):
        reduction = 100 * (1 - metric(rows[0], bill)["mean"] / metric(indexed[key], bill)["mean"])
        fig.text(x, 0.775, f"{reduction:.2f}%", fontsize=30, weight="bold", color="#126452")
        fig.text(x + 0.155, 0.79, "lower simulated bill", fontsize=12, color="#193c35")
        fig.text(x + 0.155, 0.754, f"vs {indexed[key]['label']}", fontsize=10, color="#62716d")
    left = fig.add_axes((0.16, 0.28, 0.31, 0.34))
    means = [metric(row, bill)["mean"] for row in rows]
    span = max(means) - min(means) or 0.001
    for i, (row, color) in enumerate(zip(rows, colors, strict=True)):
        stats = metric(row, bill)
        left.scatter(stats["mean"], i, color=color, s=90, zorder=3)
        left.annotate(
            f"${stats['mean']:.4f}",
            (stats["mean"], i),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=color,
            weight="bold",
        )
    left.set_xlim(min(means) - 0.42 * span, max(means) + 0.42 * span)
    left.set_ylim(2.55, -0.65)
    left.set_yticks(range(3), [row["label"] for row in rows])
    left.set_xticks([1.556, 1.560, 1.564, 1.568])
    left.set_xlabel("Mean bill ($ / home / day) · zoomed scale", labelpad=12, fontsize=9)
    left.set_title("Electricity bill ↓", loc="left", pad=23, weight="bold", fontsize=12)
    right = fig.add_axes((0.67, 0.28, 0.26, 0.34))
    bound = oracle["bound"]["upper"]
    all_rows = rows + [oracle]
    for i, (row, color) in enumerate(zip(all_rows, colors + ["#b28036"], strict=True)):
        gap = metric(row, "objective")["mean"] - bound
        right.hlines(i, 0, gap, color=color, linewidth=3)
        right.scatter(gap, i, color=color, s=65, zorder=3)
        right.annotate(
            f"{gap:.4f}",
            (gap, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=color,
        )
    right.set_xlim(-0.08, 3.4)
    right.set_ylim(3.55, -0.65)
    right.set_yticks(
        range(4), [row["label"] if row["id"] != "oracle" else "Oracle" for row in all_rows]
    )
    right.set_xticks([0, 1, 2, 3])
    right.set_xlabel("Combined objective above oracle · lower is better", labelpad=12, fontsize=9)
    right.set_title(
        "Distance to perfect foresight ↓", loc="left", pad=23, weight="bold", fontsize=12
    )
    for axis in (left, right):
        axis.grid(axis="x", color="#e7edeb", zorder=0)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=0, labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
    deviations = " · ".join(f"{row['label']}: ${metric(row, bill)['sd']:.3f}" for row in rows)
    fig.text(
        0.06, 0.16,
        "Household bill spread (SD): higher means more variation, not better performance.",
        fontsize=12, fontstyle="italic", color="#62716d",
    )
    fig.text(0.06, 0.12, deviations + " / day", fontsize=11, color="#62716d")
    fig.text(
        0.06,
        0.075,
        f"Oracle bill: ${metric(oracle, bill)['mean']:.4f} mean · ${metric(oracle, bill)['sd']:.3f} household SD / day",
        fontsize=12,
        fontstyle="italic",
        parse_math=False,
        color="#8c682f",
    )
    fig.text(
        0.06,
        0.03,
        "Oracle knows the future. Only its combined objective is a performance bound.",
        fontsize=12,
        fontstyle="italic",
        color="#62716d",
    )
    destination = root / "site/foundation-comparison"
    for suffix in ("svg", "png"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=180, facecolor=fig.get_facecolor())
    svg = destination.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    render()
