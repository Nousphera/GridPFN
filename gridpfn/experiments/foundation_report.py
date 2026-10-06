"""Verify and export the complete one-seed cross-foundation prototype."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from gridpfn.experiments.foundation_study import verify_features, verify_inputs
from gridpfn.experiments.foundation_worker import digest, save_json
from gridpfn.paths import ROOT

LABELS = {
    "history": "History only",
    "persistence": "Persistence",
    "trees": "Extra Trees",
    "tabpfn": "TabPFN-3.5",
    "tabfm": "TabFM",
    "tabicl": "TabICLv2",
}


def _verify_run_provenance(run, manifest, core_names):
    """Bind archived training inputs and executed code to the reported study."""
    expected_inputs = {
        f"home_{home}.csv": manifest["input_sources"][
            f"dataset/split_homes_clean/home_{home}.csv"
        ]
        for home in manifest["protocol"]["home_ids"]
    }
    actual_inputs = json.loads((run / "data_hashes.json").read_text())
    if actual_inputs != expected_inputs:
        raise ValueError("Archived training data differ from prepared study inputs")
    weather = "dataset/temp_price_newyork.csv"
    if digest(run / "source" / weather) != manifest["input_sources"][weather]:
        raise ValueError("Archived weather differs from prepared study inputs")
    core = {name: digest(run / "source" / name) for name in core_names}
    if any(value != digest(ROOT / name) for name, value in core.items()):
        raise ValueError("Current evaluation numerical code differs from training snapshot")
    return core


def _verify_logged_selection(run, selected, cfg, dates):
    """Check initial/final identities and the actual July reward maximizer."""
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    evaluations = [row for row in records if row.get("kind") == "eval"]
    episodes = [row["episode"] for row in evaluations]
    if (
        not episodes
        or episodes[0] != 0
        or episodes[-1] != cfg["episodes"]
        or episodes != sorted(set(episodes))
    ):
        raise ValueError("Incomplete or unordered validation history")
    for row in evaluations:
        if (
            row["split"] != "validation"
            or row["dates"] != dates
            or [home["home_id"] for home in row["homes"]] != cfg["home_ids"]
            or not np.isfinite(row["reward"])
        ):
            raise ValueError("Validation log cohort, split or reward differs")
    expected = {
        "initial": evaluations[0],
        "latest": evaluations[-1],
        "best": max(evaluations, key=lambda row: row["reward"]),
    }
    for label, row in expected.items():
        actual = selected[label]
        if actual["episode"] != row["episode"] or not np.isclose(
            actual["reward"], row["reward"], atol=1e-12, rtol=0
        ):
            raise ValueError(f"Selected {label} checkpoint differs from validation history")


def evaluate(root):
    from gridpfn.experiments.audit_constraints import audit_policy

    cfg = verify_inputs(root)["protocol"]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for kind in cfg["methods"]:
        verify_features(root, kind)
        run = root / "policies" / kind
        if json.loads((run / "status.json").read_text())["state"] != "completed":
            raise ValueError("Complete all policy runs before comparison")
        for checkpoint in ["best", "latest"]:
            target = run / "evaluation" / f"validation_{checkpoint}.json"
            if not target.exists():
                with (root / "evaluation.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m", "gridpfn.experiments.evaluate_checkpoint",
                            str(run),
                            "--checkpoint",
                            checkpoint,
                            "--split",
                            "validation",
                            "--gpu",
                            "0",
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
        target = run / "independent_audit.json"
        if not target.exists():
            save_json(
                target,
                {"policy_replay": audit_policy(run, checkpoint="best", split="validation", gpu=0)},
            )
        print(f"{kind}: July evaluation and independent replay complete", flush=True)


def collect(root):
    from gridpfn.experiments.submission_report import CORE_SOURCE

    manifest = verify_inputs(root)
    cfg = manifest["protocol"]
    common_dates = sorted(
        set.intersection(
            *[
                set(c["query_dates"])
                for c in manifest["cases"]
                if c["first"]
                == next(h["training_days"] for h in manifest["homes"] if h["id"] == c["home"])
            ]
        )
    )
    rows = []
    reference_settings = None
    reference_core = None
    for kind in cfg["methods"]:
        verify_features(root, kind)
        run = root / "policies" / kind
        if json.loads((run / "status.json").read_text())["state"] != "completed":
            raise ValueError("Incomplete policy")
        settings = json.loads((run / "run.json").read_text())["settings"]
        wanted = {
            "feature_mode": "raw",
            "embedding_weight": 1,
            "seed": cfg["seed"],
            "fixed_seed": cfg["seed"] * 100 + 1,
            "home_ids": cfg["home_ids"],
            "head_width": cfg["actor_width"],
            "value_width": cfg["value_width"],
            "episode": cfg["episodes"],
            "ppo_shuffle_days": True,
            "synthetic_data": None,
            "bc_rounds": 60,
            "bc_weight": 0.0,
            "validation_only": True,
        }
        if any(settings.get(k) != v for k, v in wanted.items()):
            raise ValueError(f"Unmatched executed settings: {kind}")
        shared = {
            k: v for k, v in settings.items() if k not in {"path_train", "predictive_features"}
        }
        core = _verify_run_provenance(run, manifest, CORE_SOURCE)
        if reference_settings is not None and (
            shared != reference_settings or core != reference_core
        ):
            raise ValueError("Methods differ in training settings or numerical code")
        reference_settings, reference_core = shared, core
        selected = json.loads((run / "selection.json").read_text())
        _verify_logged_selection(run, selected, cfg, common_dates)
        policy = {}
        checked = []
        for checkpoint in ["best", "latest"]:
            path = run / "evaluation" / f"validation_{checkpoint}.json"
            d = json.loads(path.read_text())
            if (
                d["split"] != "validation"
                or d["dates"] != common_dates
                or [h["home_id"] for h in d["homes"]] != cfg["home_ids"]
            ):
                raise ValueError("Evaluation cohort/split mismatch")
            if d["checkpoint_sha256"] != digest(run / "checkpoints" / checkpoint / "heads.pt"):
                raise ValueError("Checkpoint identity mismatch")
            if d["episode"] != selected[checkpoint]["episode"]:
                raise ValueError("Checkpoint episode mismatch")
            objective = -float(np.mean([h["reward"] for h in d["homes"]]))
            if not np.isclose(objective, -selected[checkpoint]["reward"], atol=1e-9):
                raise ValueError("Selected checkpoint differs from fresh evaluation")
            policy["selected_objective" if checkpoint == "best" else "final_objective"] = objective
            checked.append(
                {
                    "checkpoint": checkpoint,
                    "evaluation_sha256": digest(path),
                    "heads_sha256": d["checkpoint_sha256"],
                }
            )
            if checkpoint == "best":
                policy.update(
                    selected_episode=d["episode"],
                    bill=float(np.mean([h["energy_bill_without_dr"] for h in d["homes"]])),
                    squared_discomfort=float(np.mean([h["squared_violation"] for h in d["homes"]])),
                )
        if selected["latest"]["episode"] != cfg["episodes"]:
            raise ValueError("Unequal completed interaction budget")
        policy["initial_objective"] = -selected["initial"]["reward"]
        audit = json.loads((run / "independent_audit.json").read_text())["policy_replay"]
        if (
            audit["split"] != "validation"
            or audit["dates"] != common_dates
            or audit["checkpoint_sha256"] != checked[0]["heads_sha256"]
        ):
            raise ValueError("Physical audit identity mismatch")
        if (
            audit["matched_policy_transitions"] != len(common_dates) * len(cfg["home_ids"]) * 24
            or not np.isfinite(audit["max_absolute_daily_difference"])
            or audit["max_absolute_daily_difference"] < 0
            or audit["max_absolute_daily_difference"] > 1e-8
        ):
            raise ValueError("Incomplete or failed original-physics replay")
        squared_errors = []
        receipts = []
        for case in manifest["cases"]:
            home = next(h for h in manifest["homes"] if h["id"] == case["home"])
            if case["first"] != home["training_days"] or kind == "history":
                continue
            with np.load(root / home["path"], allow_pickle=False) as z:
                physical = z["physical"]
            with np.load(root / case["path"], allow_pickle=False) as z:
                index = z["indices"]
            included = np.array([case["query_dates"][i] in common_dates for i in index[:, 0]])
            truth = physical[index[:, 0] + case["first"], index[:, 2], :3]
            if kind == "persistence":
                prediction = physical[index[:, 0] + case["first"], index[:, 1], :3]
            else:
                with np.load(
                    root / "predictions" / kind / f"{case['id']}.npz", allow_pickle=False
                ) as z:
                    prediction = z["prediction"]
            squared_errors.append((prediction[included] - truth[included]) ** 2)
        if kind not in {"history", "persistence"}:
            receipts = [
                json.loads((root / "predictions" / kind / f"{c['id']}.json").read_text())
                for c in manifest["cases"]
            ]
        rows.append(
            {
                "id": kind,
                "label": LABELS[kind],
                "policy": policy,
                "forecast_rmse": np.sqrt(np.concatenate(squared_errors).mean(0)).tolist()
                if squared_errors
                else None,
                "forecast_query_count": sum(len(x) for x in squared_errors),
                "inference": {
                    "fit_seconds": sum(t["fit_seconds"] for r in receipts for t in r["timing"]),
                    "predict_seconds": sum(
                        t["predict_seconds"] for r in receipts for t in r["timing"]
                    ),
                    "device": receipts[0]["device"] if receipts else "No fitted predictor",
                },
                "model": receipts[0]["models"][0] if receipts else None,
                "batch_causality": [c for r in receipts for c in r["batch_causality"]],
                "checks": checked,
                "independent_audit": audit,
                "feature_manifest_sha256": digest(root / "features" / kind / "manifest.json"),
            }
        )
    tab = next(r for r in rows if r["id"] == "tabpfn")
    best = min(rows, key=lambda r: r["policy"]["selected_objective"])
    finding = (
        f"TabPFN's July-selected objective is {tab['policy']['selected_objective']:.4f}; "
        f"the lowest observed value is {best['policy']['selected_objective']:.4f} ({best['label']}). "
        "One seed and repeated validation selection do not establish a general ranking."
    )
    return {
        "schema_version": 1,
        "scope": cfg["scope"],
        "protocol": cfg,
        "dates": common_dates,
        "normalization": manifest.get("normalization"),
        "rows": rows,
        "finding": finding,
        "timing_scope": "Sum of 24 target fits/prediction calls across 8 chronological blocks, includes cold startup; hardware differs and other CPU work overlapped. Not a controlled latency benchmark.",
        "cases_sha256": digest(root / "cases.json"),
        "numerical_source": reference_core,
        "actor_parameters": 14088,
        "private_value_parameters": 3265,
        "limitations": [
            "Two homes and one seed; no confidence interval, held-out superiority or oracle-parity claim.",
            "July selects checkpoints and reports results; August is not evaluated in this prototype.",
            "Forecast scalers use all training dates as offline preprocessing, not strict online cold-start adaptation.",
            "Pretrained sizes, inference devices and internal model preprocessing differ; context and policy budgets match.",
            "TabFM pretrained weights are noncommercial/nonproduction and not included.",
        ],
    }


def public_evidence(data):
    """Keep checkpoint provenance while omitting local editable-install paths."""
    result = json.loads(json.dumps(data))
    for row in result["rows"]:
        model = row.get("model")
        if model and "installed_source" in model:
            model.pop("installed_source")
    return result


def plot(data, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), gridspec_kw={"height_ratios": [1, 1.05]})
    rows = data["rows"]
    pred = [r for r in rows if r["forecast_rmse"] is not None]
    for j, ax in enumerate(axes.flat[:3]):
        ax.barh(
            [r["label"] for r in pred],
            [r["forecast_rmse"][j] for r in pred],
            color=["#235c42" if r["id"] == "tabpfn" else "#638797" for r in pred],
            height=0.57,
        )
        ax.invert_yaxis()
        ax.set_title(
            ["Load RMSE (kWh)", "Solar RMSE (kWh)", "Temperature RMSE (°C)"][j],
            loc="left",
            weight="bold",
        )
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    ax = axes.flat[3]
    y = np.arange(len(rows))
    ax.scatter(
        [r["policy"]["selected_objective"] for r in rows],
        y,
        c=["#235c42" if r["id"] == "tabpfn" else "#638797" for r in rows],
        s=40,
        label="July-selected",
    )
    ax.scatter(
        [r["policy"]["final_objective"] for r in rows],
        y,
        c="#ab7650",
        marker="x",
        s=42,
        label="After 1,000 episodes",
    )
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_title("Control objective · lower is better", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.15)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Forecast quality and scheduling quality are different tests",
        x=0.06,
        ha="left",
        weight="bold",
        fontsize=17,
    )
    fig.text(
        0.06,
        0.915,
        "Same labeled contexts · same 256-unit actor / 64-unit critic · two homes · one seed",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.025,
        "13 shared July dates · validation-selected development results, not an untouched test · no uncertainty interval from one seed",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.99, 0.89), h_pad=2.1, w_pad=3)
    for extension in ["svg", "png", "pdf"]:
        fig.savefig(destination.with_suffix("." + extension), dpi=180)
    path = destination.with_suffix(".svg")
    path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()
    root = args.study.resolve()
    if args.evaluate:
        evaluate(root)
    data = collect(root)
    save_json(root / "report.json", data)
    if args.export:
        plot(data, ROOT / "site/foundation-results")
        save_json(ROOT / "site/foundation-evidence.json", public_evidence(data))
    print(data["finding"])


if __name__ == "__main__":
    main()
