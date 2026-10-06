"""Export all frozen submission comparisons into one audited, compact evidence bundle."""

import argparse
import json
from pathlib import Path

import numpy as np

from gridpfn.core.utils.run_io import atomic_json, file_sha256
from gridpfn.paths import ROOT
from oracle.runner import _digest, read_record

METRICS = (
    "objective",
    "energy_bill_without_dr",
    "squared_violation",
    "violation_hours",
    "peak_violation_degrees",
    "task_success_pct",
)
SHORT = {"tabpfn": "TabPFN + PPO", "mlp": "MLP + PPO", "mlp_capacity": "Capacity-matched MLP + PPO"}
COLORS = {"tabpfn": "#235c42", "mlp": "#526f61", "mlp_capacity": "#35687e", "oracle": "#80612f"}
CORE_SOURCE = (
    "gridpfn/experiments/train.py", "gridpfn/core/training_config.py", "gridpfn/core/model.py", "gridpfn/core/environment.py", "gridpfn/core/em_strategy.py",
    "gridpfn/core/dataset.py", "gridpfn/core/server.py", "gridpfn/core/client.py", "gridpfn/core/batched_learning.py", "gridpfn/core/evaluate.py",
    "gridpfn/core/training_metrics.py", "gridpfn/core/agents/onpolicy.py", "gridpfn/core/control_guidance.py", "gridpfn/core/economic_control.py",
)


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite evidence")
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0,
        "values": values.tolist(),
    }


def metric(record, name):
    return -record["reward"] if name == "objective" else record[name]


def training_contract(output, cfg):
    """Check executed settings and inputs, not only the intended protocol."""
    allowed = {
        "path_train", "seed", "fixed_seed", "feature_mode", "embedding_weight",
        "head_width", "value_width",
    }
    expected = {
        "tabpfn": ("hybrid", 0.1, 64, 64),
        "mlp": ("raw", 1.0, 64, 64),
        "mlp_capacity": ("raw", 1.0, 2913, 3513),
    }
    reference, input_hashes = None, None
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            run = output / f"{arm}_seed{seed}"
            settings = json.loads((run / "run.json").read_text())["settings"]
            if (
                settings["seed"] != seed
                or settings["fixed_seed"] != seed * 100 + 1
                or settings["episode"] != cfg["episodes"]
                or settings["home_ids"] != cfg["home_ids"]
                or not settings["validation_only"]
                or settings["actor_update"] != "ppo"
                or settings["bc_rounds"] != 60
            ):
                raise ValueError("Executed training differs from the frozen study")
            actual = tuple(settings[key] for key in (
                "feature_mode", "embedding_weight", "head_width", "value_width"
            ))
            if actual != expected[arm]:
                raise ValueError("Executed representation differs from the declared arm")
            shared = {k: v for k, v in settings.items() if k not in allowed}
            inputs = json.loads((run / "data_hashes.json").read_text())
            if reference is None:
                reference, input_hashes = shared, inputs
            elif reference != shared or input_hashes != inputs:
                raise ValueError("Unmatched training settings or input files")
    # Paths are machine-specific and carry no scientific meaning in a public bundle.
    public = {k: v for k, v in reference.items() if k not in {
        "path_data", "embedding_cache", "gpu", "head_device"
    }}
    return {"shared_settings": public, "input_sha256": input_hashes}


def collect(output):
    protocol_path = ROOT / "configs/submission.json"
    cfg = json.loads(protocol_path.read_text())
    digest = file_sha256(protocol_path)
    status = json.loads((output / "status.json").read_text())
    frozen = json.loads((output / "protocol.json").read_text())
    if status != {"state": "completed", "runs": 9, "protocol_sha256": digest}:
        raise ValueError("Finish every frozen comparison before publication")
    if frozen["sha256"] != digest or frozen["protocol"] != cfg:
        raise ValueError("Frozen protocol differs from the published configuration")
    executed = training_contract(output, cfg)
    core_hashes = {name: file_sha256(ROOT / name) for name in CORE_SOURCE}
    provenance, methods, backbone = {}, [], None
    all_dates = None
    for arm, spec in cfg["arms"].items():
        primary, gated, curves, episodes, audits, details = [], [], [], [], [], []
        for seed in cfg["seeds"]:
            run = output / f"{arm}_seed{seed}"
            if arm == "tabpfn":
                identity = json.loads((run / "backbone.json").read_text())
                if backbone is not None and identity != backbone:
                    raise ValueError("TabPFN runs used different pretrained weights")
                backbone = identity
            source_hashes = json.loads((run / "source_hashes.json").read_text())
            for name, source_digest in core_hashes.items():
                if (
                    source_hashes[name] != source_digest
                    or file_sha256(run / "source" / name) != source_digest
                ):
                    raise ValueError(f"Published numerical code differs from executed code: {name}")
            results = []
            for checkpoint in (cfg["primary_checkpoint"], cfg["secondary_checkpoint"]):
                path = run / "evaluation" / f"test_{checkpoint}.json"
                record = json.loads(path.read_text())
                checkpoint_path = run / "checkpoints" / checkpoint / "heads.pt"
                if record["checkpoint_sha256"] != file_sha256(checkpoint_path):
                    raise ValueError("Checkpoint no longer matches its evaluation")
                if [h["home_id"] for h in record["homes"]] != cfg["home_ids"]:
                    raise ValueError("Different home order")
                if all_dates is None:
                    all_dates = record["dates"]
                if record["dates"] != all_dates or len(all_dates) != 31:
                    raise ValueError("Different evaluation dates")
                for name in METRICS:
                    expected = np.mean([metric(h, name) for h in record["homes"]])
                    if not np.isclose(metric(record, name), expected, atol=1e-10):
                        raise ValueError(f"Unreconciled household mean: {arm}/{seed}/{name}")
                provenance[str(path.relative_to(ROOT))] = file_sha256(path)
                results.append(record)
            records = [
                json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
            ]
            validation = [r for r in records if r["kind"] == "eval"]
            if validation[-1]["episode"] != cfg["episodes"]:
                raise ValueError("Training budget incomplete")
            chosen = max(validation, key=lambda r: r["reward"])
            if chosen["episode"] != results[0]["episode"]:
                raise ValueError("Primary checkpoint was not selected by July reward")
            audit = json.loads((run / "independent_audit.json").read_text())["policy_replay"]
            if (
                audit["matched_policy_transitions"] != 7440
                or audit["checkpoint_episode"] != chosen["episode"]
            ):
                raise ValueError("Independent replay does not cover the selected policy")
            if audit["max_absolute_daily_difference"] > 1e-5:
                raise ValueError("Independent physical replay mismatch")
            primary.append(results[0])
            gated.append(results[1])
            audits.append(audit)
            episodes.append(chosen["episode"])
            curves.append(
                {
                    "seed": seed,
                    "episodes": [r["episode"] for r in validation],
                    "best_objective": np.minimum.accumulate(
                        [-r["reward"] for r in validation]
                    ).tolist(),
                }
            )
            details.append(
                {
                    "seed": seed,
                    "selected_episode": chosen["episode"],
                    "checkpoint_sha256": results[0]["checkpoint_sha256"],
                    "day_means": {
                        name: np.asarray(
                            [
                                [metric(day, name) for day in home]
                                for home in results[0]["day_records"]
                            ]
                        )
                        .mean(0)
                        .tolist()
                        for name in METRICS
                    },
                    "source_manifest_sha256": file_sha256(run / "source_hashes.json"),
                    "data_manifest_sha256": file_sha256(run / "data_hashes.json"),
                    "runtime": json.loads((run / "runtime.json").read_text())["packages"],
                }
            )
        methods.append(
            {
                "id": arm,
                "label": spec["label"],
                "short_label": SHORT[arm],
                "metrics": {
                    name: summarize([metric(r, name) for r in primary]) for name in METRICS
                },
                "gated_sensitivity": {
                    name: summarize([metric(r, name) for r in gated]) for name in METRICS
                },
                "selected_episodes": episodes,
                "curves": curves,
                "audits": audits,
                "runs": details,
            }
        )
    oracle_dir = ROOT / "results/oracle/original_test"
    manifest = json.loads((oracle_dir / "oracle.json").read_text())
    if manifest["home_ids"] != cfg["home_ids"] or manifest["dates"] != all_dates:
        raise ValueError("Oracle cohort mismatch")
    rows, certificates = [], []
    for day in all_dates:
        rec = read_record(oracle_dir / "days" / f"{day}.json", manifest)
        item = rec["oracles"]["paper_reward"]
        solution, audit = item["solution"], item["audit"]
        if not solution["certified"] or not audit["verified"]:
            raise ValueError("Uncertified oracle")
        certificates.append(
            {
                "date": day,
                "lower": solution["lower_bound"] / len(cfg["home_ids"]),
                "upper": solution["upper_bound"] / len(cfg["home_ids"]),
                # read_record validates and removes the receipt before returning.
                "receipt_sha256": _digest(rec),
            }
        )
        for home in audit["homes"]:
            temperatures = np.asarray(home["temperatures"])
            # Oracle replay stores post-action temperatures, one per hour.
            if temperatures.shape != (24,):
                raise ValueError("Unexpected oracle temperature trace")
            deviation = np.maximum(np.maximum(18 - temperatures, temperatures - 22), 0)
            rows.append(
                {
                    "objective": -home["reward"],
                    "energy_bill_without_dr": home["energy_bill_without_dr"],
                    "squared_violation": home["squared_violation"],
                    "violation_hours": float((deviation > 1e-6).sum()),
                    "peak_violation_degrees": float(deviation.max()),
                    "task_success_pct": 100.0
                    * float(
                        home["ev_completion_ratio"] >= 1 - 1e-6 and home["wm_completed"] >= 1 - 1e-6
                    ),
                }
            )
    oracle = {
        "id": "oracle",
        "label": "Perfect-foresight oracle",
        "short_label": "Future-knowing oracle",
        "metrics": {name: summarize([np.mean([r[name] for r in rows])]) for name in METRICS},
        "daily_bounds": certificates,
        "mean_objective_interval": {
            "lower": float(np.mean([c["lower"] for c in certificates])),
            "upper": float(np.mean([c["upper"] for c in certificates])),
        },
        "note": "Numerically certified simulator objective. Component metrics are not separate bounds. Violation hours allow 1e-6 C solver tolerance.",
    }
    if not np.isclose(
        oracle["metrics"]["objective"]["mean"],
        oracle["mean_objective_interval"]["upper"],
        atol=1e-6, rtol=0,
    ):
        raise ValueError("Oracle replay differs from the certified objective")
    lower_bounds = np.asarray([c["lower"] for c in certificates])
    for method in methods:
        for run in method["runs"]:
            if np.any(np.asarray(run["day_means"]["objective"]) < lower_bounds - 1e-5):
                raise ValueError(f"Policy violates an oracle lower bound: {method['id']}/{run['seed']}")
    tab = methods[0]["metrics"]["objective"]["mean"]
    lower = oracle["metrics"]["objective"]["mean"]
    comparisons = {
        m["id"]: 100
        * (m["metrics"]["objective"]["mean"] - tab)
        / abs(m["metrics"]["objective"]["mean"])
        for m in methods[1:]
    }
    differences = [
        f"{abs(change):.2f}% {'lower' if change >= 0 else 'higher'} than {SHORT[arm]}"
        for arm, change in comparisons.items()
    ]
    finding = (
        "TabPFN's mean combined objective is " + " and ".join(differences) + ". "
        f"Its gap to the future-knowing oracle is {tab - lower:.3f} per home/day."
    )
    return {
        "schema_version": 1,
        "protocol_sha256": digest,
        "protocol": cfg,
        "executed_contract": executed,
        "core_source_sha256": core_hashes,
        "backbone": backbone,
        "methods": methods,
        "oracle": oracle,
        "dates": all_dates,
        "finding": finding,
        "objective_reduction_percent": comparisons,
        "provenance": provenance,
    }


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "#f7f6f0",
            "axes.facecolor": "#f7f6f0",
            "text.color": "#173f31",
            "axes.labelcolor": "#173f31",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.28, top=0.74, wspace=0.42)
    fig.text(
        0.055,
        0.94,
        "Does TabPFN improve a matched home-energy controller?",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.055,
        0.865,
        "10 homes · 3 seeds per method · 8,000 simulated training days/home · July selection → August scoring",
        fontsize=11,
    )
    for ax, key, title in zip(
        axes[:2],
        ["objective", "energy_bill_without_dr"],
        ["Combined objective ↓", "Electricity bill ($/home/day) ↓"],
        strict=True,
    ):
        for i, method in enumerate(data["methods"]):
            values = method["metrics"][key]["values"]
            ax.bar(i, np.mean(values), color=COLORS[method["id"]], width=0.58, alpha=0.8)
            ax.scatter(i + np.array([-0.10, 0, 0.10]), values, s=24, color="#203b30", zorder=3)
            ax.text(
                i,
                max(values) + 0.025 * max(1, max(values)),
                f"{np.mean(values):.3f}",
                ha="center",
                fontsize=10,
            )
        ref = data["oracle"]["metrics"][key]["mean"]
        ax.axhline(ref, color=COLORS["oracle"], ls="--", lw=1.5)
        ax.text(
            0.98,
            ref,
            " Oracle",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="#8f713e",
            fontsize=9,
        )
        ax.set_xticks([0, 1, 2], ["TabPFN", "MLP", "Wide MLP"])
        ax.set_title(title, loc="left", fontsize=12, pad=15)
        ax.set_ylim(0, max(max(m["metrics"][key]["values"]) for m in data["methods"]) * 1.22)
    ax = axes[2]
    for method in data["methods"]:
        x = method["curves"][0]["episodes"]
        y = np.asarray([c["best_objective"] for c in method["curves"]])
        ax.plot(x, y.mean(0), color=COLORS[method["id"]], label=method["short_label"])
        ax.fill_between(x, y.min(0), y.max(0), color=COLORS[method["id"]], alpha=0.13)
    ax.set_title("Learning with the same controller", loc="left", fontsize=12, pad=15)
    ax.set_ylabel("Best July validation objective ↓")
    ax.set_xlabel("Training days per home")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c7cfc3")
        ax.tick_params(colors="#657467")
    fig.text(
        0.055,
        0.13,
        "Dots: every seed. Learning bands: seed range. Oracle knows future traces; only its combined objective is a lower bound.",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.075,
        "August has historical development exposure. Simulation results do not establish real-home savings, SOTA or oracle parity.",
        fontsize=10,
        color="#657467",
    )
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(ROOT / "site" / f"results.{ext}", dpi=190)
    svg_path = ROOT / "site/results.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def write_summary(data):
    rows = data["methods"] + [data["oracle"]]
    table = [
        "| August, mean per home/day | Objective ↓ | Bill ($) ↓ | Squared discomfort ↓ |",
        "|---|---:|---:|---:|",
    ]
    for method in rows:
        numbers = [
            method["metrics"][key]["mean"]
            for key in ["objective", "energy_bill_without_dr", "squared_violation"]
        ]
        table.append(
            f"| {method['short_label']} | " + " | ".join(f"{v:.3f}" for v in numbers) + " |"
        )
    text = (
        "**Completed: nine matched runs; all primary policies independently replayed.**\n\n"
        "![Complete matched comparison](site/results.svg)\n\n"
        + "\n".join(table)
        + "\n\n" + data["finding"]
        + "\n\n[All seeds, tradeoffs and uncertainty](docs/RESULTS.md) · "
        "[Machine-readable evidence](site/evidence.json)"
    )
    readme = ROOT / "README.md"
    existing = readme.read_text()
    before, rest = existing.split("<!-- RESULTS:START -->")
    _, after = rest.split("<!-- RESULTS:END -->")
    readme.write_text(before + "<!-- RESULTS:START -->\n" + text + "\n<!-- RESULTS:END -->" + after)
    detailed = [
        "# Final matched results",
        "",
        data["finding"],
        "",
        "All three seeds are retained. Values below are means ± sample seed SD. "
        "Seeds share the same ten-home cohort; these are not population confidence intervals.",
        "",
        "| Method | Objective ↓ | Bill ($) ↓ | Squared discomfort ↓ | Violation hours ↓ | Peak deviation (°C) ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in rows:
        values = []
        for key in METRICS[:-1]:
            stat = method["metrics"][key]
            values.append(
                f"{stat['mean']:.4f}" + (f" ± {stat['sd']:.4f}" if method["id"] != "oracle" else "")
            )
        detailed.append(f"| {method['short_label']} | " + " | ".join(values) + " |")
    detailed.extend(
        [
            "",
            "## Sensitivity to the historical service gate",
            "",
            "The primary checkpoint maximizes July validation reward under all original "
            "physical constraints. The optional historical gate also restricts electrical "
            "cost and comfortable-hour loss relative to each initial policy. It is not a "
            "constraint imposed on the oracle.",
            "",
            "| Method | Primary objective | Gated objective | Primary selected episodes |",
            "|---|---:|---:|---|",
        ]
    )
    for method in data["methods"]:
        detailed.append(
            f"| {method['short_label']} | {method['metrics']['objective']['mean']:.4f} | "
            f"{method['gated_sensitivity']['objective']['mean']:.4f} | "
            + ", ".join(map(str, method["selected_episodes"]))
            + " |"
        )
    detailed.extend(["", "## What the evidence supports", ""])
    for arm, change in data["objective_reduction_percent"].items():
        direction = "lower" if change >= 0 else "higher"
        detailed.append(
            f"- TabPFN's mean objective is **{abs(change):.2f}% {direction}** than {SHORT[arm]}."
        )
    transitions = sum(a["matched_policy_transitions"] for m in data["methods"] for a in m["audits"])
    error = max(a["max_absolute_daily_difference"] for m in data["methods"] for a in m["audits"])
    detailed.extend(
        [
            f"- Independent original-physics replays cover **{transitions:,} transitions**; "
            f"maximum daily metric difference is {error:.3g}.",
            "- The oracle knows future traces. Its combined objective is numerically bounded; "
            "its bill and comfort components are not independent optima. Oracle violation "
            "hours allow 1e-6 °C solver tolerance; policy hours use the original strict bounds.",
            f"- The certified mean objective interval is "
            f"**[{data['oracle']['mean_objective_interval']['lower']:.8f}, "
            f"{data['oracle']['mean_objective_interval']['upper']:.8f}]** per home/day.",
            "- The oracle has more out-of-band hours but less squared discomfort. "
            "The reward penalizes squared temperature deviation, not violation duration; "
            "minimizing one does not minimize the other.",
            "- August was used in earlier development. No SOTA, "
            "real-home savings, privacy or oracle-parity claim is established.",
            "",
            "[Protocol](protocol.md) · [Reproduce](REPRODUCE.md) · "
            "[All seeds, dates, hashes and bounds](../site/evidence.json)",
        ]
    )
    (ROOT / "docs/RESULTS.md").write_text("\n".join(detailed) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/submission_20261006")
    args = parser.parse_args()
    data = collect(args.input)
    plot(data)
    write_summary(data)
    atomic_json(ROOT / "site/evidence.json", data)
    print(data["finding"])


if __name__ == "__main__":
    main()
