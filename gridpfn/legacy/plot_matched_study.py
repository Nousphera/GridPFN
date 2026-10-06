"""Compact, seed-paired RL branch comparisons with explicit convergence evidence."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

from gridpfn.paths import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from gridpfn.core.utils.run_io import atomic_json, file_sha256
from gridpfn.legacy.control_benchmark import load_clients
from gridpfn.legacy.run_matched_study import ARMS, evaluate_reward_selection

LABELS = ("Original MLP", "Current MLP", "Current TabPFN")
COLORS = ("#9a6677", "#d59a38", "#168f9c")


def report_identity():
    root = ROOT
    return {
        name: file_sha256(root / name)
        for name in (
            "gridpfn/legacy/plot_matched_study.py",
            "gridpfn/legacy/run_matched_study.py",
            "gridpfn/core/model.py",
            "gridpfn/core/training_metrics.py",
            "gridpfn/legacy/control_benchmark.py",
            "gridpfn/core/em_strategy.py",
            "gridpfn/core/dataset.py",
            "gridpfn/core/environment.py",
        )
    }


def paired_difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError("Paired results require equal nonempty one-dimensional seed arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Paired results must be finite")
    difference = right - left
    count = len(difference)
    sd = float(np.std(difference, ddof=1)) if count > 1 else 0.0
    radius = float(t.ppf(0.975, count - 1) * sd / np.sqrt(count)) if count > 1 else None
    return {
        "n": count,
        "mean": float(np.mean(difference)),
        "seed_sd": sd,
        "ci95": [float(np.mean(difference) - radius), float(np.mean(difference) + radius)]
        if radius is not None
        else None,
    }


def render_scheduling(study, scheduling, destination):
    """Compare the fitted forecast controllers with the newly trained RL arms."""
    manifest = json.loads((study / "study.json").read_text())
    cases = {arm: [] for arm in ARMS}
    cases.update(tabpfn_mpc=[], trees_mpc=[])
    provenance = []
    clients = load_clients(split="test", data_dir=manifest["data_dir"])
    for seed in manifest["seeds"]:
        dates = None
        for arm in ARMS:
            summary = json.loads((study / f"{arm}_seed{seed}" / "final_summary.json").read_text())
            physical = summary["evaluations"]["best_feasible"]["physical"]
            if dates is not None and physical["dates"] != dates:
                raise ValueError("RL test dates differ")
            dates = physical["dates"]
            cases[arm].append(physical)
        run = scheduling / f"seed{seed}" / "study_results.json"
        forecasts = json.loads(run.read_text())
        hashes = {Path(name).name: value for name, value in forecasts["data_hashes"].items()}
        if (
            forecasts["split"] != "test"
            or forecasts["home_ids"] != manifest["protocol"]["home_ids"]
        ):
            raise ValueError("Forecast study cohort/split differs")
        if any(hashes[name] != value for name, value in manifest["data_hashes"].items()):
            raise ValueError("Forecast study home data differ")
        archived = scheduling / f"seed{seed}" / "source" / "gridpfn/legacy/run_energy_study.py"
        spec = importlib.util.spec_from_file_location("forecast_receipt", archived)
        receipt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(receipt)
        settings = forecasts["settings"]
        # The forecast signature hashes reconstructed load/PV/weather/price
        # arrays, dates and the archived forecaster, not just the home CSVs.
        if (
            receipt.forecast_signature(clients, SimpleNamespace(**settings))
            != forecasts["forecast_signature"]
        ):
            raise ValueError("Forecast training/test physical traces or dates differ")
        for name, policy in (("tabpfn_mpc", "tabpfn_peer"), ("trees_mpc", "trees_peer")):
            physical = forecasts["controllers"][policy]
            if physical["dates"] != dates:
                raise ValueError("Forecast study test dates differ")
            cases[name].append(physical)
        provenance.append(
            {
                key: forecasts[key]
                for key in (
                    "settings",
                    "source_hashes",
                    "backbone",
                    "data_hashes",
                    "runtime",
                    "forecast_signature",
                )
            }
        )
    names = (*LABELS, "TabPFN MPC", "Tree MPC")
    ticks = ("Original", "Current\nMLP", "TabPFN\nRL", "TabPFN\nMPC", "Tree\nMPC")
    colors = (*COLORS, "#4b9870", "#7b79a3")
    specs = (
        ("energy_bill_without_dr", "Bill before DR ($/home/day)"),
        ("import", "Grid purchases (kWh/home/day)"),
        ("comfort_pct", "Comfort (%)"),
        ("pv_local_use_kwh", "PV retained locally (kWh/day)"),
        ("reward", "Daily reward"),
        ("elec_cost", "Objective including DR ($/day)"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6), layout="constrained")
    outcomes = {}
    for ax, (key, title) in zip(list(axes.flat)[:6], specs, strict=True):
        outcomes[key] = {}
        for i, ((case, rows), color) in enumerate(zip(cases.items(), colors, strict=True)):
            values = [float(np.mean([home[key] for home in row["homes"]])) for row in rows]
            outcomes[key][case] = values
            ax.bar(
                i,
                np.mean(values),
                yerr=np.std(values, ddof=1 if len(values) > 1 else 0),
                color=color,
                capsize=2,
                width=0.65,
            )
            ax.scatter(np.full(len(values), i), values, color="#263238", s=9, zorder=3)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=7)
        ax.set_title(title, loc="left", fontsize=9)
    homes = manifest["protocol"]["home_ids"]
    for ax, key, title in (
        (axes[1, 2], "comfort_pct", "TabPFN MPC − original comfort (pp)"),
        (axes[1, 3], "energy_bill_without_dr", "TabPFN MPC − original bill ($/day)"),
    ):
        baseline = np.asarray([[h[key] for h in r["homes"]] for r in cases["original"]])
        candidate = np.asarray([[h[key] for h in r["homes"]] for r in cases["tabpfn_mpc"]])
        changes = candidate - baseline
        ax.bar(
            range(len(homes)),
            changes.mean(0),
            color=colors[3],
            yerr=changes.std(0, ddof=1 if len(changes) > 1 else 0),
            capsize=2,
        )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(len(homes)), homes, rotation=65, fontsize=7)
        ax.set_title(title, loc="left", fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    convergence = [
        json.loads((study / f"{arm}_seed{seed}" / "convergence.json").read_text())["stopped"]
        for arm in ARMS
        for seed in manifest["seeds"]
    ]
    fig.suptitle(
        f"Matched physical scenario · 10 homes · 31 August dates · RL plateaus: {sum(convergence)}/{len(convergence)}\nRL: {len(manifest['seeds'])} training seeds; MPC: {len(manifest['seeds'])} forecast-context seeds; bars = seed SD",
        fontsize=10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=170)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    atomic_json(
        destination.with_suffix(".json"),
        {
            "outcomes": outcomes,
            "labels": dict(zip(cases, names, strict=True)),
            "rl_protocol": manifest["protocol"],
            "forecast_provenance": provenance,
            "report_source_hashes": report_identity(),
            "all_rl_converged": manifest.get("all_converged", False),
            "homes": {case: [row["homes"] for row in rows] for case, rows in cases.items()},
            "note": "Historical fitted forecast controllers reused after cohort/date/data checks. Physics independently audited. MPC changes both the scheduler and thermal policy; this is not an embedding-only gain. Forecast seeds are not RL retraining seeds.",
        },
    )


def render(study, destination):
    manifest = json.loads((study / "study.json").read_text())
    runs, records = {}, {}
    for arm in ARMS:
        runs[arm], records[arm] = [], []
        for seed in manifest["seeds"]:
            run = study / f"{arm}_seed{seed}"
            summary = run / "final_summary.json"
            if not summary.exists():
                raise ValueError(f"Unfinished run: {run}")
            runs[arm].append(json.loads(summary.read_text()))
            records[arm].append(
                [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            )
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6), layout="constrained")
    plt.rcParams.update({"font.size": 9})
    curves = (
        ("train", "reward", "Training reward"),
        ("eval", "reward", "Validation reward"),
        ("train", "critic_loss", "Critic training loss"),
        ("train", "actor_loss", "Actor training objective"),
    )
    for ax, (kind, metric, title) in zip(axes[0], curves, strict=True):
        for arm, label, color in zip(ARMS, LABELS, COLORS, strict=True):
            traces = []
            for events in records[arm]:
                events = [r for r in events if r["kind"] == kind and r.get(metric) is not None]
                # Training is averaged in fixed 50-episode bins, without interpolating stopped runs.
                bins = {}
                for event in events:
                    episode = (
                        event["episode"]
                        if kind == "eval"
                        else ((event["episode"] - 1) // 50 + 1) * 50
                    )
                    bins.setdefault(episode, []).append(event[metric])
                traces.append({key: float(np.mean(value)) for key, value in bins.items()})
            # Thin seed traces expose all stopping points; the mean/band uses only
            # episodes with every seed present, without extending completed runs.
            for trace in traces:
                episodes = sorted(trace)
                ax.plot(
                    episodes, [trace[x] for x in episodes], color=color, alpha=0.2, linewidth=0.5
                )
                if episodes:
                    ax.scatter(episodes[-1], trace[episodes[-1]], marker="|", color=color, s=24)
            shared = sorted(set.intersection(*(set(trace) for trace in traces)))
            if not shared:
                continue
            values = np.asarray([[trace[x] for x in shared] for trace in traces])
            mean, sd = values.mean(0), values.std(0, ddof=1 if len(values) > 1 else 0)
            ax.plot(shared, mean, color=color, label=label, linewidth=1.4)
            ax.fill_between(shared, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Episode")
        if "loss" in metric:
            ax.set_yscale("symlog", linthresh=1)
    outcomes = {}
    metric_specs = (
        ("energy_bill_without_dr", "Test bill before DR ($/day)"),
        ("comfort_pct", "Test comfort (%)"),
        ("import", "Grid purchases (kWh/day)"),
    )
    for ax, (key, title) in zip(axes[1, :3], metric_specs, strict=True):
        outcomes[key] = {}
        for i, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
            values = [
                np.mean([h[key] for h in run["evaluations"]["best_feasible"]["physical"]["homes"]])
                for run in runs[arm]
            ]
            outcomes[key][arm] = [float(v) for v in values]
            ax.bar(
                i,
                np.mean(values),
                yerr=np.std(values, ddof=1 if len(values) > 1 else 0),
                color=color,
                capsize=3,
                width=0.6,
            )
            ax.scatter(np.full(len(values), i), values, color="#263238", s=12, zorder=3)
        ax.set_xticks(range(3), ("Original", "Current\nMLP", "Current\nTabPFN"), fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
    ax = axes[1, 3]
    homes = manifest["protocol"]["home_ids"]
    base = np.asarray(
        [
            [
                h["energy_bill_without_dr"]
                for h in run["evaluations"]["best_feasible"]["physical"]["homes"]
            ]
            for run in runs["original"]
        ]
    )
    tab = np.asarray(
        [
            [
                h["energy_bill_without_dr"]
                for h in run["evaluations"]["best_feasible"]["physical"]["homes"]
            ]
            for run in runs["current_tabpfn"]
        ]
    )
    delta = tab - base
    ax.bar(
        range(len(homes)),
        delta.mean(0),
        color=COLORS[2],
        yerr=delta.std(0, ddof=1 if len(delta) > 1 else 0),
        capsize=2,
    )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(range(len(homes)), homes, rotation=65, fontsize=7)
    ax.set_title("TabPFN − original bill by home", loc="left", fontsize=10)
    ax.set_ylabel("$/home/day; below zero is better")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8, frameon=False)
    all_converged = all(run["convergence"]["stopped"] for arm in ARMS for run in runs[arm])
    passed = sum(run["convergence"]["stopped"] for arm in ARMS for run in runs[arm])
    status = f"validation plateaus: {passed}/{len(manifest['seeds']) * len(ARMS)} runs"
    if not all_converged:
        status += "; remaining runs hit cap"
    fig.suptitle(
        f"Matched FedAvg · {len(homes)} homes · {len(manifest['seeds'])} training seeds · {status}\nFixed August dates; bands/error bars = seed SD; losses are not comparable objectives",
        fontsize=10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=170)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    reward_selection = {}
    for arm in ARMS:
        for seed in manifest["seeds"]:
            path = study / f"{arm}_seed{seed}" / "reward_selection.json"
            value = json.loads(path.read_text()) if path.exists() else None
            if value:
                value["physical"].pop("day_records", None)
            reward_selection[arm, seed] = value
    portable = {
        "protocol": manifest["protocol"],
        "original_revision": manifest["original_revision"],
        "current_revision": manifest["current_revision"],
        "data_hashes": manifest["data_hashes"],
        "price_weather_sha256": manifest["price_weather_sha256"],
        "all_converged": all_converged,
        "inputs_matched": manifest.get("inputs_matched", False),
        "runtime": manifest.get("runtime", {}),
        "verification": json.loads((study / "verification.json").read_text())
        if (study / "verification.json").exists()
        else None,
        "settings": {
            arm: json.loads((study / f"{arm}_seed{manifest['seeds'][0]}" / "run.json").read_text())[
                "settings"
            ]
            for arm in ARMS
        },
        "model_capacity": {
            arm: json.loads(
                (study / f"{arm}_seed{manifest['seeds'][0]}" / "model_capacity.json").read_text()
            )
            for arm in ARMS
        },
        "backbones": {
            str(seed): json.loads(
                (study / f"current_tabpfn_seed{seed}" / "backbone.json").read_text()
            )
            for seed in manifest["seeds"]
        },
        "source_hashes": json.loads((study / "source_hashes.json").read_text()),
        "report_source_hashes": report_identity(),
        "original_source_hashes": manifest["original_source_hashes"],
        "outcomes": outcomes,
        "paired_differences": {
            key: {
                **{arm: paired_difference(values["original"], values[arm]) for arm in ARMS[1:]},
                "tabpfn_minus_current_mlp": paired_difference(
                    values["current_mlp"], values["current_tabpfn"]
                ),
            }
            for key, values in outcomes.items()
        },
        "runs": {
            arm: [
                {
                    "seed": r["seed"],
                    "training_seconds": r["training_seconds"],
                    "convergence": r["convergence"],
                    "selected_episode": r["evaluations"]["best_feasible"]["episode"],
                    "checkpoint_sha256": r["evaluations"]["best_feasible"]["checkpoint_sha256"],
                    "homes": r["evaluations"]["best_feasible"]["physical"]["homes"],
                    "reward_selection_sensitivity": reward_selection[arm, r["seed"]],
                    "checkpoint_comparison": {
                        label: {
                            "episode": value["episode"],
                            "metrics": {
                                key: float(np.mean([h[key] for h in value["physical"]["homes"]]))
                                for key in (
                                    "reward",
                                    "comfort_pct",
                                    "energy_bill_without_dr",
                                    "import",
                                )
                            },
                        }
                        for label, value in r["evaluations"].items()
                    },
                }
                for r in runs[arm]
            ]
            for arm in ARMS
        },
    }
    atomic_json(destination.with_suffix(".json"), portable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scheduling", type=Path, help="Optional audited forecast study on the same inputs."
    )
    parser.add_argument(
        "--reward-sensitivity",
        action="store_true",
        help="Also replay the unconstrained best validation reward checkpoints.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Sensitivity replay device.")
    args = parser.parse_args()
    destination = args.output or args.study / "comparison"
    if args.reward_sensitivity:
        evaluate_reward_selection(args.study.resolve(), args.gpu)
    render(args.study.resolve(), destination)
    if args.scheduling:
        render_scheduling(
            args.study.resolve(),
            args.scheduling.resolve(),
            destination.with_name(destination.name + "_scheduling"),
        )


if __name__ == "__main__":
    main()
