"""Publish certified oracle bounds and matched federated actor-critic gaps."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridpfn.core.utils.run_io import atomic_json, file_sha256

from .config import OracleConfig
from .runner import read_record

ARMS = ("original", "current_mlp", "current_tabpfn")
LABELS = ("Original\nMLP", "Current\nMLP", "TabPFN\nRL", "Reward\noracle", "Service\noracle")
COLORS = ("#9a6677", "#d59a38", "#168f9c", "#4b9870", "#6779a3")


def load_oracle(directory):
    protocol = json.loads((directory / "oracle.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    if status["state"] != "completed":
        raise ValueError("Publish only a completed, certified oracle run")
    rows = [
        read_record(directory / "days" / f"{date}.json", protocol) for date in protocol["dates"]
    ]
    for date, row in zip(protocol["dates"], rows, strict=True):
        if row["date"] != date or any(
            not value["solution"]["certified"] or not value["audit"]["verified"]
            for value in row["oracles"].values()
        ):
            raise ValueError("An oracle date lacks a solver certificate or independent replay")
    return protocol, rows, json.loads((directory / "summary.json").read_text())


def matched_policies(study, protocol, rows):
    """Reject mismatched physics, source data, home ordering and evaluation dates."""
    manifest = json.loads((study / "study.json").read_text())
    execution = {
        "name",
        "data_dir",
        "output",
        "split",
        "home_ids",
        "days",
        "objectives",
        "workers",
        "time_limit",
        "gap",
    }
    defaults = OracleConfig().settings()
    if any(protocol["scenario"][k] != v for k, v in defaults.items() if k not in execution):
        raise ValueError(
            "RL policies must be re-evaluated under this changed scenario before comparison"
        )
    hashes = {Path(name).name: digest for name, digest in protocol["input_file_sha256"].items()}
    if (
        protocol["split"] != "test"
        or manifest["protocol"]["home_ids"] != protocol["home_ids"]
        or not manifest["inputs_matched"]
        or protocol["price_weather_sha256"] != manifest["price_weather_sha256"]
        or hashes != manifest["data_hashes"]
    ):
        raise ValueError("Oracle and RL data/cohort differ")
    cases, provenance = {}, []
    for name in ("gridpfn/core/environment.py", "gridpfn/core/em_strategy.py", "gridpfn/core/dataset.py"):
        if file_sha256(study / "source" / name) != protocol["source_sha256"][name]:
            raise ValueError("Oracle and RL physical equations or preprocessing differ")
    for arm in ARMS:
        cases[arm] = []
        for seed in manifest["seeds"]:
            directory = study / f"{arm}_seed{seed}"
            summary_path = directory / "final_summary.json"
            selected = json.loads(summary_path.read_text())["evaluations"]["best_feasible"]
            physical = selected["physical"]
            if physical["dates"] != protocol["dates"]:
                raise ValueError("Oracle and RL evaluation dates differ")
            for i, records in enumerate(physical["day_records"]):
                if [r["day"] for r in records] != protocol["dates"]:
                    raise ValueError("RL home-day ordering differs")
                for day, oracle in zip(records, rows, strict=True):
                    solution = oracle["oracles"]["paper_reward"]["solution"]
                    # August factorizes, giving an independent bound for each home.
                    if solution["coupled_market"]:
                        raise ValueError("Per-home RL regret requires a factorized oracle market")
                    certificate = next(c for c in solution["certificates"] if c["homes"] == [i])
                    if -day["reward"] < certificate["lower_bound"] - 2e-4:
                        raise AssertionError("RL performance contradicts a certified oracle bound")
            cases[arm].append(physical["homes"])
            provenance.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "selected_episode": selected["episode"],
                    "summary_sha256": file_sha256(summary_path),
                    "checkpoint_sha256": selected["checkpoint_sha256"],
                }
            )
    return cases, provenance


def render(run, validation, study, output):
    protocol, rows, summary = load_oracle(run)
    if protocol["objectives"] != ["paper_reward", "comfort_first"]:
        raise ValueError("The comparison requires both oracle definitions")
    cases, provenance = matched_policies(study, protocol, rows)
    cases.update(
        {
            objective: [summary[objective]["homes"]]
            for objective in ("paper_reward", "comfort_first")
        }
    )
    outcomes = {}
    for case, seeds in cases.items():
        outcomes[case] = {
            key: [float(np.mean([home[key] for home in homes])) for homes in seeds]
            for key in ("reward", "elec_cost", "energy_bill_without_dr", "import")
        }
        outcomes[case]["discomfort_penalty"] = [
            float(np.mean([-home["reward"] - home["elec_cost"] for home in homes]))
            for homes in seeds
        ]
    frontiers = [
        {
            "home_id": home,
            "maximum_in_band_pct": float(
                np.mean([r["frontiers"][i]["maximum_comfort_pct"] for r in rows])
            ),
            "minimum_squared_violation": float(
                np.mean([r["frontiers"][i]["minimum_squared_violation"] for r in rows])
            ),
            "unavoidable_discomfort_days": sum(
                r["frontiers"][i]["minimum_squared_violation"] > 1e-5 for r in rows
            ),
        }
        for i, home in enumerate(protocol["home_ids"])
    ]
    tabpfn = np.array([[-h["reward"] for h in seed] for seed in cases["current_tabpfn"]])
    optimum = np.array([-h["reward"] for h in cases["paper_reward"][0]])
    gaps = tabpfn - optimum
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 5.7), layout="constrained")
    specs = (
        ("reward", "Paper objective (−reward / home-day)", -1),
        ("energy_bill_without_dr", "Bill before DR ($ / home-day)", 1),
        ("discomfort_penalty", "Original discomfort penalty / home-day", 1),
        ("import", "Grid purchases (kWh / home-day)", 1),
    )
    for ax, (key, title, sign) in zip(list(axes.flat)[:4], specs, strict=True):
        for i, (case, color) in enumerate(zip(cases, COLORS, strict=True)):
            values = sign * np.array(outcomes[case][key])
            ax.bar(i, values.mean(), color=color, width=0.65)
            if len(values) > 1:
                ax.scatter(np.full(len(values), i), values, color="#263238", s=11, zorder=3)
        ax.set_xticks(range(5), LABELS, fontsize=7)
        ax.set_title(title, loc="left", fontsize=9)
        ax.axhline(0, color="#777", linewidth=0.6)
    axis = axes[1, 1]
    axis.bar(range(len(frontiers)), gaps.mean(0), color=COLORS[2])
    axis.errorbar(
        range(len(frontiers)),
        gaps.mean(0),
        yerr=gaps.std(0, ddof=1),
        fmt="none",
        color="#263238",
        capsize=2,
    )
    axis.set_title("TabPFN RL − reward oracle (mean ± seed SD)", loc="left", fontsize=9)
    axis.set_xticks(range(len(frontiers)), protocol["home_ids"], rotation=60, fontsize=7)
    axis = axes[1, 2]
    learned = np.mean([[h["comfort_pct"] for h in s] for s in cases["current_tabpfn"]], axis=0)
    x = np.arange(len(frontiers))
    axis.bar(x - 0.18, learned, width=0.36, color=COLORS[2], label="TabPFN RL")
    axis.bar(
        x + 0.18,
        [f["maximum_in_band_pct"] for f in frontiers],
        width=0.36,
        color=COLORS[3],
        label="Maximum hours oracle",
    )
    axis.set_xticks(x, protocol["home_ids"], rotation=60, fontsize=7)
    axis.set_ylim(0, 105)
    axis.set_title("In-band hours (%) · separate comfort objective", loc="left", fontsize=9)
    axis.legend(frameon=False, fontsize=7, loc="lower left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.16)
        axis.set_axisbelow(True)
    fig.suptitle(
        "10 homes · 31 August days · same original constraints\n"
        "RL dots: 3 training seeds · oracles: realized future data · lower is better in first five panels",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    validation_protocol, validation_rows, validation_summary = (
        load_oracle(validation) if validation is not None else (None, [], None)
    )
    report = {
        "protocol": protocol,
        "summary": summary,
        "frontiers": frontiers,
        "outcomes": outcomes,
        "tabpfn_paper_objective_gap_per_home": gaps.mean(0).tolist(),
        "matched_provenance": provenance,
        "daily_certificates": {
            split: [
                {
                    "date": row["date"],
                    "severity_maximum_gap": max(
                        f["severity_certificate"]["absolute_gap"] for f in row["frontiers"]
                    ),
                    "oracles": {
                        key: {
                            "lower_bound": value["solution"]["lower_bound"],
                            "upper_bound": value["solution"]["upper_bound"],
                            "certified": value["solution"]["certified"],
                            "original_replay_verified": value["audit"]["verified"],
                            "maximum_replay_difference": value["audit"][
                                "max_trajectory_difference"
                            ],
                        }
                        for key, value in row["oracles"].items()
                    },
                }
                for row in records
            ]
            for split, records in (("test", rows), ("validation", validation_rows))
        },
        "validation": {
            "protocol": validation_protocol,
            "summary": validation_summary,
            "coupled_market_dates": [
                r["date"]
                for r in validation_rows
                if r["oracles"]["paper_reward"]["solution"]["coupled_market"]
            ],
        },
        "note": "Certified numerical perfect-foresight bounds, not deployable policies. Service oracle minimizes each home's squared discomfort, not the count of in-band hours. Bills across different comfort objectives are not equal-service savings. August was previously accessed in development. Original reward includes DR and WM discomfort; it is undiscounted.",
    }
    atomic_json(output.with_suffix(".json"), report)


def render_scenario(run, output):
    """Plot daily community means and shaded home SD for any configured scenario."""
    protocol, rows, summary = load_oracle(run)
    specs = (
        ("energy_bill_without_dr", "Bill before DR ($ / home-day)"),
        ("elec_cost", "Electrical cost including DR ($ / home-day)"),
        ("squared_violation", "Squared thermal discomfort (°C² / home-day)"),
        ("comfort_pct", "Comfortable hours (%) · 1e−6°C tolerance"),
        ("import", "Grid purchases (kWh / home-day)"),
        ("p2p_kwh", "Peer imports (kWh / home-day)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 5.7), layout="constrained")
    for axis, (key, title) in zip(axes.flat, specs, strict=True):
        for i, objective in enumerate(protocol["objectives"]):
            values = np.array(
                [[h[key] for h in r["oracles"][objective]["audit"]["homes"]] for r in rows]
            )
            mean, sd = values.mean(1), values.std(1)
            x = np.arange(len(rows))
            color = COLORS[3 + i]
            if len(rows) == 1:
                axis.errorbar(
                    x,
                    mean,
                    yerr=sd,
                    color=color,
                    fmt="o",
                    capsize=3,
                    label=objective.replace("_", " "),
                )
            else:
                axis.plot(x, mean, color=color, label=objective.replace("_", " "), linewidth=1.3)
                axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
        indices = sorted({0, len(rows) // 2, len(rows) - 1})
        axis.set_xticks(indices, [rows[i]["date"][5:] for i in indices], fontsize=8)
        axis.set_title(title, loc="left", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.16)
    axes[0, 0].legend(frameon=False, fontsize=8)
    dispersion = "points: mean ± home SD" if len(rows) == 1 else "lines: home mean · shade: home SD"
    fig.suptitle(
        f"{protocol['scenario']['name']} · {len(protocol['home_ids'])} homes · {len(rows)} days\n"
        f"Certified perfect foresight · {dispersion}",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    atomic_json(output.with_suffix(".json"), {"protocol": protocol, "summary": summary})


def render_actor_critic(run, study, validation, output):
    """Publish policy duration/severity, cheap causal control and certified bounds."""
    from types import SimpleNamespace

    from gridpfn.core.control_guidance import FeedbackTeacher
    from gridpfn.core.em_strategy import compose_em_strategy, make_em_strategy
    from gridpfn.legacy.control_benchmark import load_clients, rollout_controllers
    from gridpfn.legacy.plot_matched_study import paired_difference

    from .benchmark import OracleComparison

    protocol, oracle_days, summary = load_oracle(run)
    validation_protocol, _, validation_summary = load_oracle(validation)
    manifest = json.loads((study / "study.json").read_text())
    if manifest["state"] != "completed" or len(manifest["runs"]) != 2 * len(manifest["seeds"]):
        raise ValueError("Both learning arms and all planned seeds must finish")
    clients = load_clients(protocol["home_ids"], split=protocol["split"])
    for home, client in zip(protocol["home_ids"], clients, strict=True):
        client.home_id, client.state_dim = home, 17
    strategy = compose_em_strategy(make_em_strategy() | {"ac_energy_quota": True}, clients)
    server = SimpleNamespace(
        clients=clients, em_strategy=strategy, p2p_config={"enabled": True, "price": 0.1}
    )
    comparison = OracleComparison(run, server, protocol["split"])
    if (
        validation_protocol["home_ids"] != protocol["home_ids"]
        or validation_protocol["split"] != "validation"
    ):
        raise ValueError("Validation oracle cohort or split differs")
    validation_clients = load_clients(protocol["home_ids"])
    for home, client in zip(protocol["home_ids"], validation_clients, strict=True):
        client.home_id, client.state_dim = home, 17
    validation_comparison = OracleComparison(
        validation,
        SimpleNamespace(
            clients=validation_clients, em_strategy=strategy, p2p_config=server.p2p_config
        ),
        "validation",
    )
    cases, curves, provenance = {"previous_q": [], "raw": [], "tabpfn": []}, {}, []
    for arm in ("raw", "tabpfn"):
        curves[arm] = []
        for seed in manifest["seeds"]:
            directory = study / f"{arm}_seed{seed}"
            settings = json.loads((directory / "run.json").read_text())
            record = json.loads((directory / "evaluation/test_best_feasible.json").read_text())
            convergence = json.loads((directory / "convergence.json").read_text())
            audit = json.loads((directory / "constraint_audit.json").read_text())
            checkpoint = directory / "checkpoints/best_feasible/heads.pt"
            hashes = json.loads((directory / "data_hashes.json").read_text())
            if (
                not convergence["stopped"]
                or settings["home_ids"] != protocol["home_ids"]
                or record["dates"] != protocol["dates"]
                or record["split"] != protocol["split"]
                or record["checkpoint_sha256"] != file_sha256(checkpoint)
                or hashes != {Path(k).name: v for k, v in protocol["input_file_sha256"].items()}
                or audit["reference_commit"] != protocol["reference_commit"]
                or audit["matched_policy_transitions"] != len(protocol["dates"]) * len(clients) * 24
            ):
                raise ValueError(
                    "Learning provenance, dates, data, checkpoint or independent replay differ"
                )
            comparison.annotate(record)
            cases[arm].append(record["homes"])
            curves[arm].append(
                [
                    json.loads(line)
                    for line in (directory / "metrics.jsonl").read_text().splitlines()
                    if json.loads(line)["kind"] == "eval"
                ]
            )
            for row in curves[arm][-1]:
                validation_comparison.annotate(row)
            provenance.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "selected_episode": record["episode"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "evaluation_sha256": file_sha256(
                        directory / "evaluation/test_best_feasible.json"
                    ),
                    "source_hashes": json.loads((directory / "source_hashes.json").read_text()),
                    "settings": settings["settings"],
                    "convergence": convergence,
                    "constraint_audit": audit,
                }
            )
    previous_provenance = []
    for seed in manifest["seeds"]:
        path = study / f"previous_q_seed{seed}.json"
        record = json.loads(path.read_text())
        if record["dates"] != protocol["dates"] or record["split"] != protocol["split"]:
            raise ValueError("Previous policy dates differ")
        comparison.annotate(record)
        cases["previous_q"].append(record["homes"])
        previous_provenance.append(
            {
                "seed": seed,
                "selected_episode": record["episode"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "evaluation_sha256": file_sha256(path),
            }
        )
    teachers = [FeedbackTeacher(c.scaler, quota_aware=True, storage_aware=True) for c in clients]
    baseline = rollout_controllers(
        clients, strategy, [lambda s, t=t: (int(s[0] * 24 >= 10), t(s[None])[0]) for t in teachers]
    )
    if baseline["dates"] != protocol["dates"]:
        raise ValueError("Causal control dates differ")
    cases["causal"] = [baseline["homes"]]
    oracle_homes = []
    for i, home in enumerate(summary["paper_reward"]["homes"]):
        home = dict(home)
        peaks, degree_hours = [], []
        for row in oracle_days:
            temp = np.asarray(row["oracles"]["paper_reward"]["audit"]["homes"][i]["temperatures"])
            violation = np.maximum(np.maximum(18 - temp, temp - 22), 0)
            peaks.append(float(violation.max()))
            degree_hours.append(float(violation.sum()))
        home.update(
            violation_hours=24 * (1 - home["comfort_pct"] / 100),
            peak_violation_degrees=float(np.mean(peaks)),
            degree_hours=float(np.mean(degree_hours)),
        )
        oracle_homes.append(home)
    cases["oracle"] = [oracle_homes]
    metrics = (
        "reward",
        "energy_bill_without_dr",
        "elec_cost",
        "comfort_pct",
        "squared_violation",
        "violation_hours",
        "degree_hours",
        "peak_violation_degrees",
        "import",
    )
    outcomes = {
        arm: {key: [float(np.mean([h[key] for h in homes])) for homes in seeds] for key in metrics}
        for arm, seeds in cases.items()
    }
    paired = {
        reference: {
            key: paired_difference(outcomes[reference][key], outcomes["tabpfn"][key])
            for key in metrics
        }
        for reference in ("previous_q", "raw")
    }
    public = {
        "home_ids": protocol["home_ids"],
        "dates": protocol["dates"],
        "seeds": manifest["seeds"],
        "input_file_sha256": {Path(k).name: v for k, v in protocol["input_file_sha256"].items()},
        "price_weather_sha256": protocol["price_weather_sha256"],
        "protocol": "Train June1-July17; select July18-31; evaluate August after selection. August was historically accessed. Same original physics/reward. Frozen TabPFN; PPO is a new learning algorithm.",
        "uncertainty": "Initialization seeds with common fixed_seed42 sampling; error bars: seed SD. Equal hidden width is not equal parameter count. Three initializations provide limited uncertainty.",
        "comfort": "Reward sums squared deviation at every hour. Violation hours, degree-hours and daily peak measure different tradeoffs. Oracle duration uses1e-6C numerical tolerance; learned controls use strict original bounds.",
        "outcomes": outcomes,
        "paired_tabpfn_minus_reference": paired,
        "homes": cases,
        "validation_curves": {
            arm: [
                [
                    {
                        k: r[k]
                        for k in (
                            "episode",
                            "reward",
                            "comfort_pct",
                            "elec_cost",
                            "energy_bill_without_dr",
                            "oracle_regret",
                            "excess_squared_violation",
                            "q_return_rmse",
                        )
                    }
                    for r in rows
                ]
                for rows in seeds
            ]
            for arm, seeds in curves.items()
        },
        "runs": provenance,
        "previous_policy_provenance": previous_provenance,
        "oracle": {
            "manifest_sha256": file_sha256(run / "oracle.json"),
            "validation_manifest_sha256": file_sha256(validation / "oracle.json"),
            "bounds": summary["paper_reward"],
            "physical_minimum": summary["comfort_limits"],
        },
        "baseline": "Causal quota-aware20C thermal control, uniform remaining EV demand, WM-at10, observed-PV self-consumption; no realized future samples or oracle labels.",
    }
    labels = ("Previous\nQ-RL", "Raw\nPPO", "TabPFN\nPPO", "Causal\ncontrol", "Reward\noracle")
    colors = ("#94a3b8", "#d59a38", "#168f9c", "#52667a", "#4b9870")
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.1), layout="constrained")
        axis = axes[0, 0]
        for arm, color in (("raw", colors[1]), ("tabpfn", colors[2])):
            for records in curves[arm]:
                axis.plot(
                    [r["episode"] for r in records],
                    [-r["reward"] for r in records],
                    color=color,
                    alpha=0.25,
                    lw=0.7,
                )
            shared = sorted(
                set.intersection(*({r["episode"] for r in rows} for rows in curves[arm]))
            )
            values = np.array(
                [
                    [-next(r for r in rows if r["episode"] == e)["reward"] for e in shared]
                    for rows in curves[arm]
                ]
            )
            axis.plot(shared, values.mean(0), color=color, label=arm, lw=1.5)
            axis.fill_between(
                shared,
                values.mean(0) - values.std(0),
                values.mean(0) + values.std(0),
                color=color,
                alpha=0.15,
            )
        axis.axhline(
            -validation_summary["paper_reward"]["mean"]["reward"],
            color=colors[4],
            ls="--",
            lw=1,
            label="oracle",
        )
        axis.set(title="July learning · paper objective ↓", xlabel="Training episodes")
        axis.legend(frameon=False, fontsize=7)
        specs = (
            ("reward", "August paper objective ↓", -1),
            ("energy_bill_without_dr", "Bill before DR ($/home-day) ↓", 1),
            ("squared_violation", "Squared discomfort (°C²/home-day) ↓", 1),
            ("violation_hours", "Violation hours/day ↓ · oracle tolerance", 1),
            ("peak_violation_degrees", "Mean daily peak deviation (°C) ↓", 1),
        )
        for axis, (key, title, sign) in zip(list(axes.flat)[1:], specs, strict=True):
            for i, (arm, color) in enumerate(zip(cases, colors, strict=True)):
                values = sign * np.asarray(outcomes[arm][key])
                axis.bar(
                    i,
                    values.mean(),
                    color=color,
                    width=0.64,
                    yerr=values.std(ddof=1) if len(values) > 1 else 0,
                    error_kw={"capsize": 2, "elinewidth": 0.8},
                )
            axis.set(title=title, xticks=range(5), xticklabels=labels)
            axis.grid(axis="y", alpha=0.12)
        fig.suptitle(
            "Federated actor–critic · 10 homes · original constraints · three initializations · bars/shade: seed SD",
            fontsize=10,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".png"), dpi=180)
        fig.savefig(output.with_suffix(".pdf"))
        plt.close(fig)
    atomic_json(output.with_suffix(".json"), public)
    return public
