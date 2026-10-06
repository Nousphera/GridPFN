"""Frozen-recipe seasonal evaluation: monthly forward refits, never test selection.

Run only after every method's selection and final refit has completed in all folds.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("history", "persistence", "trees", "tabpfn", "tabfm", "tabicl")
LABELS = dict(
    zip(
        METHODS,
        ("History only", "Persistence", "Extra Trees", "TabPFN-3.5", "TabFM", "TabICLv2"),
        strict=True,
    )
)
FOLDS = tuple(f"2019-{month:02d}" for month in range(6, 11))
CORE_SOURCE = (
    "gridpfn/experiments/train.py",
    "gridpfn/core/training_config.py",
    "gridpfn/core/model.py",
    "gridpfn/core/environment.py",
    "gridpfn/core/em_strategy.py",
    "gridpfn/core/dataset.py",
    "gridpfn/core/server.py",
    "gridpfn/core/client.py",
    "gridpfn/core/batched_learning.py",
    "gridpfn/core/evaluate.py",
    "gridpfn/core/training_metrics.py",
    "gridpfn/core/agents/onpolicy.py",
    "gridpfn/core/control_guidance.py",
    "gridpfn/core/economic_control.py",
    "gridpfn/core/predictive_features.py",
    "gridpfn/experiments/evaluate_checkpoint.py",
    "gridpfn/experiments/audit_constraints.py",
)
METRICS = (
    "objective",
    "energy_bill_without_dr",
    "comfort_pct",
    "violation_hours",
    "squared_violation",
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dates_between(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return [(first + timedelta(days=i)).isoformat() for i in range((last - first).days)]


def month_bounds(month):
    first = date.fromisoformat(month + "-01")
    end = first + timedelta(days=calendar.monthrange(first.year, first.month)[1])
    return first.isoformat(), end.isoformat()


def summarize_homes(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite values for at least two households")
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "values": values.tolist(),
    }


def _safe_path(root, value):
    root = Path(root).resolve()
    target = (root / value).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Study artifact path escapes the study directory")
    return target


def _dates_in_stage(stage, manifest, start, end):
    groups = []
    for home in manifest["homes"]:
        with np.load(stage / home["path"], allow_pickle=False) as z:
            groups.append({str(d) for d in z["dates"] if start <= str(d) < end})
    dates = sorted(set.intersection(*groups))
    if not dates:
        raise ValueError("No common complete household dates in the declared fold")
    return dates


def _verify_selection(run, cfg, dates):
    selected = read_json(run / "selection.json")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    evaluations = [r for r in rows if r.get("kind") == "eval"]
    episodes = [r["episode"] for r in evaluations]
    if (
        not episodes
        or episodes[0] != 0
        or episodes[-1] > cfg["episodes"]
        or episodes != sorted(set(episodes))
    ):
        raise ValueError("Incomplete or unordered pre-month validation history")
    for row in evaluations:
        if (
            row["split"] != "validation"
            or row["dates"] != dates
            or [h["home_id"] for h in row["homes"]] != cfg["home_ids"]
            or not np.isfinite(row["reward"])
        ):
            raise ValueError("Selection used a different split, cohort, or invalid reward")
    convergence = read_json(run / "convergence.json")
    if convergence.get("episode") != episodes[-1]:
        raise ValueError("Stopping evidence differs from the completed validation history")
    if episodes[-1] < cfg["episodes"] and not convergence.get("stopped"):
        raise ValueError("Early termination has no established stopping-criterion receipt")
    settings = read_json(run / "run.json")["settings"]
    if settings.get("strict_convergence"):
        from gridpfn.core.utils.convergence import stability

        verified = stability(
            evaluations, settings["min_episodes"], settings["patience"], settings["min_delta"]
        )
        if convergence.get("stopped") != verified["stopped"]:
            raise ValueError(
                "Stopping receipt disagrees with the recorded strict stability criterion"
            )
    expected = {
        "initial": evaluations[0],
        "latest": evaluations[-1],
        "best": max(evaluations, key=lambda r: r["reward"]),
    }
    for key, row in expected.items():
        if selected[key]["episode"] != row["episode"] or not np.isclose(
            selected[key]["reward"], row["reward"], atol=1e-12, rtol=0
        ):
            raise ValueError("Selection is not the logged pre-month reward maximizer")
    return selected


def _run_provenance(stage, manifest, kind):
    run = stage / "policies" / kind
    expected = {
        f"home_{h}.csv": manifest["input_sources"][f"dataset/split_homes_clean/home_{h}.csv"]
        for h in manifest["protocol"]["home_ids"]
    }
    if read_json(run / "data_hashes.json") != expected:
        raise ValueError("Training household inputs differ from frozen study inputs")
    weather = "dataset/temp_price_newyork.csv"
    if digest(run / "source" / weather) != manifest["input_sources"][weather]:
        raise ValueError("Training weather differs from frozen study inputs")
    core = {name: digest(run / "source" / name) for name in CORE_SOURCE}
    if core != {name: digest(ROOT / name) for name in CORE_SOURCE}:
        raise ValueError("Evaluation numerical source differs from the training snapshot")
    return core


def _verify_refit_dates(select, select_manifest, refit, refit_manifest, start, valid_start):
    for a, b in zip(select_manifest["homes"], refit_manifest["homes"], strict=True):
        if a["id"] != b["id"]:
            raise ValueError("Selection and refit household order differs")
        with np.load(select / a["path"], allow_pickle=False) as z:
            before = list(map(str, z["train_dates"]))
            validation = [str(d) for d in z["dates"] if valid_start <= str(d) < start]
        with np.load(refit / b["path"], allow_pickle=False) as z:
            after = list(map(str, z["train_dates"]))
        if (
            not before
            or not after
            or any(d >= valid_start for d in before)
            or any(d >= start for d in after)
            or after != sorted(set(before + validation))
        ):
            raise ValueError(
                "Refit must use all prior training plus validation dates, and no test dates"
            )


def _verify_refit_summary(summary, chosen, settings, selected_run):
    expected_source = {
        "path": str((selected_run / "selection.json").resolve()),
        "sha256": chosen["selection_sha256"],
        "run_sha256": digest(selected_run / "run.json"),
        "data_period": settings["data_period"].removesuffix("_refit") + "_select",
        "selected_episode": chosen["episodes"],
        "checkpoint": "best",
    }
    if (
        summary.get("episode") != chosen["episodes"]
        or summary.get("refit") is not True
        or summary.get("evaluation_performed") is not False
        or summary.get("data_period") != settings["data_period"]
        or summary.get("selection_source") != expected_source
        or any(
            summary.get(k) is not None for k in ("reward", "comfort_pct", "elec_cost", "feasible")
        )
    ):
        raise ValueError("Refit summary violates the fixed-budget, no-evaluation contract")


def _validate_study(root):
    from gridpfn.experiments.foundation_study import verify_features, verify_inputs

    study = read_json(root / "study.json")
    protocol = study["protocol"]
    if (
        protocol["methods"] != list(METHODS)
        or len(protocol["home_ids"]) != 25
        or len(set(protocol["home_ids"])) != 25
        or [f["id"] for f in study["folds"]] != list(FOLDS)
    ):
        raise ValueError(
            "Expected all six methods, 25 homes, and five chronological June–October folds"
        )
    # Do not read any test score, including the oracle, until all 60 runs finish.
    for fold in study["folds"]:
        for phase in ("select", "refit"):
            stage = _safe_path(root, fold[phase])
            for kind in METHODS:
                path = stage / "policies" / kind / "status.json"
                if not path.exists() or read_json(path).get("state") != "completed":
                    raise ValueError(
                        f"Complete all selections and refits first: {fold['id']}/{phase}/{kind}"
                    )
    rows, reference_core = [], None
    for fold in study["folds"]:
        select, refit = (_safe_path(root, fold[phase]) for phase in ("select", "refit"))
        select_manifest, refit_manifest = verify_inputs(select), verify_inputs(refit)
        start, end = month_bounds(fold["id"])
        valid_start = (date.fromisoformat(start) - timedelta(days=7)).isoformat()
        selection_dates = _dates_in_stage(select, select_manifest, valid_start, start)
        test_dates = _dates_in_stage(refit, refit_manifest, start, end)
        recipe_path = _safe_path(root, fold["recipe"])
        recipe = read_json(recipe_path)
        shared = selected_shared = None
        for stage, manifest, phase in (
            (select, select_manifest, "select"),
            (refit, refit_manifest, "refit"),
        ):
            if manifest["protocol"]["home_ids"] != protocol["home_ids"] or manifest["protocol"][
                "methods"
            ] != list(METHODS):
                raise ValueError("A seasonal stage changed the declared household or method cohort")
            if manifest["protocol"]["data_period"] != f"month_{start[5:7]}_{phase}":
                raise ValueError("A stage uses the wrong chronological data period")
        _verify_refit_dates(select, select_manifest, refit, refit_manifest, start, valid_start)
        for kind in METHODS:
            for stage in (select, refit):
                verify_features(stage, kind)
            selected_run, run = select / "policies" / kind, refit / "policies" / kind
            selected_settings = read_json(selected_run / "run.json")["settings"]
            select_comparable = {
                k: v
                for k, v in selected_settings.items()
                if k not in {"path_train", "predictive_features"}
            }
            if selected_shared is not None and select_comparable != selected_shared:
                raise ValueError(
                    "Selection stages differ in settings beyond their forecast features"
                )
            selected_shared = select_comparable
            selection = _verify_selection(
                selected_run, select_manifest["protocol"], selection_dates
            )
            chosen = recipe["methods"][kind]
            expected_recipe = {
                "episodes": selection["best"]["episode"],
                "selection_sha256": digest(selected_run / "selection.json"),
                "heads_sha256": digest(selected_run / "checkpoints/best/heads.pt"),
                "metrics_sha256": digest(selected_run / "metrics.jsonl"),
            }
            if any(chosen.get(k) != v for k, v in expected_recipe.items()):
                raise ValueError("Frozen recipe differs from its pre-month selection evidence")
            if refit_manifest["protocol"]["refit_budgets"][kind] != chosen["episodes"]:
                raise ValueError("Refit budget is not the validation-selected episode budget")
            settings = read_json(run / "run.json")["settings"]
            wanted = {
                "feature_mode": "raw",
                "embedding_weight": 1,
                "actor_update": "ppo",
                "seed": protocol["seed"],
                "fixed_seed": protocol["seed"] * 100 + 1,
                "home_ids": protocol["home_ids"],
                "episode": chosen["episodes"],
                "head_width": refit_manifest["protocol"]["actor_width"],
                "value_width": refit_manifest["protocol"]["value_width"],
                "data_period": f"month_{start[5:7]}_refit",
                "refit": True,
                "ppo_shuffle_days": True,
                "synthetic_data": None,
                "bc_rounds": 60,
                "bc_weight": 0.0,
            }
            if any(settings.get(k) != v for k, v in wanted.items()):
                raise ValueError(
                    f"Executed refit differs from the frozen recipe: {fold['id']}/{kind}"
                )
            if (
                Path(settings["refit_selection"]).resolve()
                != (selected_run / "selection.json").resolve()
            ):
                raise ValueError("Refit references a different selection stage")
            comparable = {
                k: v
                for k, v in settings.items()
                if k not in {"path_train", "predictive_features", "episode", "refit_selection"}
            }
            if shared is not None and comparable != shared:
                raise ValueError(
                    "Refitted methods differ beyond selected budget and forecast features"
                )
            shared = comparable
            for stage, manifest in ((select, select_manifest), (refit, refit_manifest)):
                core = _run_provenance(stage, manifest, kind)
                if reference_core is not None and core != reference_core:
                    raise ValueError("Numerical code differs across seasonal stages")
                reference_core = core
            # A refit cannot evaluate or select on any held-out data.
            if (run / "metrics.jsonl").exists():
                metrics = [
                    json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
                ]
                if any(
                    r.get("kind", "").startswith("eval") or r.get("kind") == "federation_local"
                    for r in metrics
                ):
                    raise ValueError(
                        "Refit performed evaluation instead of following its fixed budget"
                    )
            summary = read_json(run / "refit_summary.json")
            _verify_refit_summary(summary, chosen, settings, selected_run)
            if read_json(run / "selection.json") != {"latest": summary}:
                raise ValueError(
                    "Refit performed checkpoint selection beyond its fixed final checkpoint"
                )
            rows.append(
                {
                    "fold": fold["id"],
                    "id": kind,
                    "run": str(run.relative_to(root)),
                    "select": str(selected_run.relative_to(root)),
                    "recipe_sha256": digest(recipe_path),
                    "recipe": chosen,
                    "selection_convergence": read_json(selected_run / "convergence.json"),
                    "selection_completed_episode": selection["latest"]["episode"],
                    "selection_budget_cap": select_manifest["protocol"]["episodes"],
                    "selection_dates": selection_dates,
                    "test_dates": test_dates,
                    "refit_summary_sha256": digest(run / "refit_summary.json"),
                    "refit_summary": summary,
                    "run_sha256": digest(run / "run.json"),
                    "status_sha256": digest(run / "status.json"),
                    "heads_sha256": digest(run / "checkpoints/latest/heads.pt"),
                    "cases_sha256": digest(refit / "cases.json"),
                    "features_sha256": digest(refit / "features" / kind / "manifest.json"),
                }
            )
    return study, rows, reference_core


def freeze_selection(root):
    """Copy and seal all 30 final refit checkpoints before reading any test score."""
    root = Path(root).resolve()
    study, rows, core = _validate_study(root)
    path = root / "frozen_selection/manifest.json"
    contract = {
        "schema_version": 2,
        "study_sha256": digest(root / "study.json"),
        "protocol": study["protocol"],
        "methods": rows,
        "numerical_source": core,
    }
    if path.exists():
        existing = read_json(path)
        if {k: v for k, v in existing.items() if k != "frozen_at_utc"} != contract:
            raise ValueError("Frozen selection changed; preserve the test and create a new study")
    else:
        if list(root.glob("folds/*/refit/policies/*/evaluation/test*.json")) or list(
            root.glob("folds/*/refit/policies/*/test_audit.json")
        ):
            raise ValueError("Test artifacts exist without a prior all-fold selection freeze")
        for row in rows:
            target = path.parent / row["fold"] / row["id"] / "heads.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != row["heads_sha256"]:
                raise ValueError("Conflicting partially archived refit checkpoint")
            if not target.exists():
                shutil.copyfile(root / row["run"] / "checkpoints/latest/heads.pt", target)
        existing = {**contract, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(path, existing)
    for row in rows:
        if digest(path.parent / row["fold"] / row["id"] / "heads.pt") != row["heads_sha256"]:
            raise ValueError("Archived final refit checkpoint changed")
    return existing


def evaluate(root):
    root = Path(root).resolve()
    frozen = freeze_selection(root)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for row in frozen["methods"]:
        run = root / row["run"]
        target = run / "evaluation/test_latest.json"
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            code = (
                "import sys; from pathlib import Path; import torch; "
                "from gridpfn.experiments.evaluate_checkpoint import evaluate_saved; "
                "from gridpfn.experiments.full_report import write_json; torch.set_num_threads(1); "
                "write_json(Path(sys.argv[3]),evaluate_saved(Path(sys.argv[1]),Path(sys.argv[2]),'test',0))"
            )
            with (root / "test_evaluation.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        code,
                        str(run),
                        str(root / "frozen_selection" / row["fold"] / row["id"] / "heads.pt"),
                        str(target),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        target = run / "test_audit.json"
        if not target.exists():
            code = (
                "import sys; from pathlib import Path; import torch; "
                "from gridpfn.experiments.audit_constraints import audit_policy; "
                "from gridpfn.experiments.full_report import write_json; torch.set_num_threads(1); "
                "write_json(Path(sys.argv[2]),{'policy_replay':audit_policy(Path(sys.argv[1]),checkpoint='latest',split='test',gpu=0)})"
            )
            with (root / "test_evaluation.log").open("a") as log:
                subprocess.run(
                    [sys.executable, "-c", code, str(run), str(target)],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print(
            f"{row['fold']}/{row['id']}: fixed-budget test and original-physics audit complete",
            flush=True,
        )
    freeze_selection(root)


def summarize_policy(record, home_ids, dates):
    """Average complete daily observations within each home before aggregation."""
    if (
        record["split"] != "test"
        or record["dates"] != dates
        or [h["home_id"] for h in record["homes"]] != home_ids
        or len(record["day_records"]) != len(home_ids)
    ):
        raise ValueError("Policy evaluation has different homes, dates, or split")
    per_home = []
    for home, days, mean in zip(home_ids, record["day_records"], record["homes"], strict=True):
        if len(days) != len(dates) or [d["day"] for d in days] != dates:
            raise ValueError("Missing, duplicated, or reordered household evaluation dates")
        metrics = {}
        for key in METRICS:
            values = np.asarray(
                [-d["reward"] if key == "objective" else d[key] for d in days], dtype=float
            )
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite household daily metric")
            metrics[key] = float(values.mean())
            reported = -mean["reward"] if key == "objective" else mean[key]
            if not np.isclose(metrics[key], reported, atol=1e-9, rtol=1e-9):
                raise ValueError("Reported household mean differs from daily records")
        per_home.append({"home_id": home, **metrics})
    return {
        "per_home": per_home,
        "metrics": {key: summarize_homes([h[key] for h in per_home]) for key in METRICS},
    }


def pool_monthly_results(folds):
    """Pool daily outcomes within each household, then describe household spread.

    Each fold contains ``dates``, ``home_ids``, and method ``rows`` with
    ``policy.per_home``. Months are weighted by their number of evaluated days;
    houses remain equally weighted. This is not a confidence interval.
    """
    if not folds:
        raise ValueError("No completed monthly folds")
    home_ids = folds[0]["home_ids"]
    method_ids = [r["id"] for r in folds[0]["rows"]]
    seen_dates = set()
    last_date = None
    for fold in folds:
        dates = fold["dates"]
        if not dates or dates != sorted(set(dates)) or seen_dates.intersection(dates):
            raise ValueError("Monthly folds contain missing, unordered, or overlapping dates")
        if last_date is not None and dates[0] <= last_date:
            raise ValueError("Monthly folds must be in chronological evaluation order")
        if fold["home_ids"] != home_ids or [r["id"] for r in fold["rows"]] != method_ids:
            raise ValueError("Monthly folds differ in methods or household cohort")
        for row in fold["rows"]:
            if [h["home_id"] for h in row["policy"]["per_home"]] != home_ids:
                raise ValueError("Monthly household records are missing or reordered")
        seen_dates.update(dates)
        last_date = dates[-1]
    counts = np.array([len(f["dates"]) for f in folds])
    output = []
    for position, kind in enumerate(method_ids):
        homes = []
        for index, home in enumerate(home_ids):
            values = {}
            for key in METRICS:
                daily_means = [f["rows"][position]["policy"]["per_home"][index][key] for f in folds]
                if not np.isfinite(daily_means).all():
                    raise ValueError("Nonfinite monthly household metric")
                values[key] = float(np.average(daily_means, weights=counts))
            homes.append({"home_id": home, **values})
        output.append(
            {
                "id": kind,
                "label": folds[0]["rows"][position]["label"],
                "policy": {
                    "per_home": homes,
                    "metrics": {key: summarize_homes([h[key] for h in homes]) for key in METRICS},
                },
            }
        )
    return {
        "home_ids": home_ids,
        "dates": sorted(seen_dates),
        "rows": output,
        "aggregation": "Day-weighted monthly means within each household, then equal-weight household mean and sample SD; household heterogeneity, not a confidence interval.",
    }


def plot_performance(data, destination):
    """Publication figures: every method, individual homes, mean and household SD."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = list(data["rows"])
    if data.get("oracle") is not None:
        rows.append(data["oracle"])
    if not rows:
        raise ValueError("No verified methods to plot")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.8), sharey=True)
    specs = [
        ("energy_bill_without_dr", "Electricity bill ↓", "$ / home / day"),
        ("comfort_pct", "Comfortable time ↑", "% of hours in the temperature band"),
        ("objective", "Combined objective ↓", "Original simulator cost / home / day"),
    ]
    for axis, (key, title, unit) in zip(axes, specs, strict=True):
        for i, row in enumerate(rows):
            values = np.asarray(row["policy"]["metrics"][key]["values"], dtype=float)
            stats = summarize_homes(values)
            color = (
                "#235c42"
                if row["id"] == "tabpfn"
                else "#9a713e"
                if row["id"] == "oracle"
                else "#627c86"
            )
            jitter = np.linspace(-0.13, 0.13, len(values))
            axis.scatter(
                values, i + jitter, s=12, color=color, alpha=0.35, edgecolors="none", zorder=2
            )
            axis.errorbar(
                stats["mean"],
                i,
                xerr=stats["sd"],
                fmt="D",
                ms=5.5,
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=3.5,
                zorder=3,
            )
        axis.set_title(title, loc="left", pad=14)
        axis.set_xlabel(unit, fontsize=8, labelpad=11)
        axis.set_yticks(range(len(rows)), [r["label"] for r in rows])
        axis.grid(axis="x", color="#dfe4df", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0, pad=9)
        if key == "comfort_pct":
            axis.set_xlim(-3, 103)
        if key == "objective" and data.get("oracle") is not None:
            axis.axvline(
                data["oracle"]["policy"]["metrics"][key]["mean"],
                color="#9a713e", linestyle="--", linewidth=1, alpha=0.7,
            )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Seasonal home-energy control", x=0.075, y=0.97, ha="left", fontsize=19, weight="bold"
    )
    fig.text(
        0.075,
        0.91,
        f"{len(data['home_ids'])} homes · June–October 2019",
        fontsize=10,
        color="#4f5e54",
    )
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="#627c86",
            alpha=0.5,
            markersize=4,
            label="Household average",
        ),
        Line2D(
            [], [], marker="D", color="#627c86", markersize=5, label="Household mean ± sample SD"
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.067, 0.093),
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    fig.text(
        0.075,
        0.065,
        "Oracle: perfect future. Only its combined objective is a bound.",
        fontsize=8,
        color="#586257",
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.79, bottom=0.26, wspace=0.34)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=200, facecolor="white")
    plt.close(fig)


def forecast_metrics(stage, manifest, kind, dates):
    """Per-home RMSE on only this fold's test dates, with equal-home summaries."""
    if kind == "history":
        return None
    homes = []
    for home in manifest["homes"]:
        cases = [
            c
            for c in manifest["cases"]
            if c["home"] == home["id"] and c["first"] == home["training_days"]
        ]
        if len(cases) != 1:
            raise ValueError("Expected exactly one final refit forecasting case per home")
        case = cases[0]
        with np.load(stage / home["path"], allow_pickle=False) as z:
            physical = z["physical"]
        with np.load(stage / case["path"], allow_pickle=False) as z:
            index = z["indices"]
        mask = np.asarray([case["query_dates"][i] in dates for i in index[:, 0]])
        if {case["query_dates"][i] for i in index[mask, 0]} != set(dates):
            raise ValueError("Forecast queries do not cover all evaluated dates")
        truth = physical[index[:, 0] + case["first"], index[:, 2], :3]
        if kind == "persistence":
            prediction = physical[index[:, 0] + case["first"], index[:, 1], :3]
        else:
            with np.load(
                stage / "predictions" / kind / f"{case['id']}.npz", allow_pickle=False
            ) as z:
                prediction = z["prediction"]
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid held-out forecast prediction shape or values")
        homes.append(
            {
                "home_id": home["id"],
                "queries": int(mask.sum()),
                "rmse": np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2, axis=0)).tolist(),
            }
        )
    return summarize_forecasts(homes)


def summarize_forecasts(homes):
    if not homes or any(h["queries"] <= 0 or len(h["rmse"]) != 3 for h in homes):
        raise ValueError("Missing household forecast errors")
    return {
        "per_home": homes,
        **{
            key: summarize_homes([h["rmse"][i] for h in homes])
            for i, key in enumerate(("load", "pv", "temperature"))
        },
        "aggregation": "RMSE within each household's test queries; then equal-weight household mean and sample SD.",
    }


def pool_forecasts(folds, kind, home_ids):
    if kind == "history":
        return None
    homes = []
    for index, home in enumerate(home_ids):
        records = [
            next(r for r in f["rows"] if r["id"] == kind)["forecast"]["per_home"][index]
            for f in folds
        ]
        if any(r["home_id"] != home for r in records):
            raise ValueError("Forecast household ordering differs across months")
        weights = np.array([r["queries"] for r in records])
        mse = np.asarray([r["rmse"] for r in records]) ** 2
        homes.append(
            {
                "home_id": home,
                "queries": int(weights.sum()),
                "rmse": np.sqrt(np.average(mse, axis=0, weights=weights)).tolist(),
            }
        )
    return summarize_forecasts(homes)


def oracle_summary(path, manifest, dates, numerical_source):
    """Verify certificates and reconstruct per-household means from daily replays."""
    from oracle.config import OracleConfig
    from oracle.runner import read_record

    protocol = read_json(path / "oracle.json")
    cfg = manifest["protocol"]
    if (
        protocol["home_ids"] != cfg["home_ids"]
        or protocol["dates"] != dates
        or protocol["split"] != "test"
        or protocol["data_period"] != cfg["data_period"]
        or protocol.get("perfect_foresight") is not True
    ):
        raise ValueError("Oracle is not the same held-out household/date scenario")
    expected = {
        f"home_{h}.csv": manifest["input_sources"][f"dataset/split_homes_clean/home_{h}.csv"]
        for h in cfg["home_ids"]
    }
    if {Path(k).name: v for k, v in protocol["input_file_sha256"].items()} != expected:
        raise ValueError("Oracle household inputs differ")
    if (
        protocol["price_weather_sha256"]
        != manifest["input_sources"]["dataset/temp_price_newyork.csv"]
    ):
        raise ValueError("Oracle weather inputs differ")
    # Algorithmic resource settings may differ; physical/economic settings may not.
    default = OracleConfig().settings()
    ignore = {
        "name",
        "data_dir",
        "output",
        "split",
        "data_period",
        "home_ids",
        "days",
        "objectives",
        "workers",
        "time_limit",
        "gap",
    }
    if any(protocol["scenario"].get(k) != v for k, v in default.items() if k not in ignore):
        raise ValueError("Oracle changed original physical or economic settings")
    for name, expected_hash in protocol["source_sha256"].items():
        if name in numerical_source and numerical_source[name] != expected_hash:
            raise ValueError("Oracle and learned policies used different simulator source")
    home_days = [[] for _ in cfg["home_ids"]]
    lower, upper, receipts = [], [], []
    for day in dates:
        source = path / "days" / f"{day}.json"
        record = read_record(source, protocol)
        if record["date"] != day:
            raise ValueError("Oracle date receipt mismatch")
        oracle = record["oracles"]["paper_reward"]
        solution, audit = oracle["solution"], oracle["audit"]
        certificates = solution["certificates"]
        covered = [index for certificate in certificates for index in certificate["homes"]]
        if (
            not audit["verified"]
            or not certificates
            or sorted(covered) != list(range(len(cfg["home_ids"])))
            or not all(c["certified"] for c in certificates)
        ):
            raise ValueError(
                "Oracle optimum is uncertified, incomplete, or failed original-simulator replay"
            )
        lo, hi = solution["lower_bound"], solution["upper_bound"]
        if not np.isfinite([lo, hi]).all() or lo > hi + 1e-8:
            raise ValueError("Invalid oracle objective interval")
        if len(audit["homes"]) != len(home_days):
            raise ValueError("Oracle is missing household replays")
        replayed = -sum(float(h["reward"]) for h in audit["homes"])
        if not np.isclose(replayed, hi, atol=1e-4, rtol=1e-6):
            raise ValueError("Certified oracle objective differs from independent replay")
        for i, h in enumerate(audit["homes"]):
            # Strict band membership matches policy scoring. The solver-tolerant
            # count is also retained, so boundary sensitivity remains inspectable.
            strict = float(h["strict_comfort_pct"])
            home_days[i].append(
                {
                    "objective": -float(h["reward"]),
                    "energy_bill_without_dr": float(h["energy_bill_without_dr"]),
                    "comfort_pct": strict,
                    "violation_hours": 24 * (1 - strict / 100),
                    "squared_violation": float(h["squared_violation"]),
                    "solver_tolerant_comfort_pct": float(h["comfort_pct"]),
                }
            )
        lower.append(float(lo) / len(home_days))
        upper.append(float(hi) / len(home_days))
        receipts.append({"date": day, "file_sha256": digest(source)})
    homes = [
        {
            "home_id": home,
            **{
                key: float(np.mean([d[key] for d in daily]))
                for key in (*METRICS, "solver_tolerant_comfort_pct")
            },
        }
        for home, daily in zip(cfg["home_ids"], home_days, strict=True)
    ]
    return {
        "id": "oracle",
        "label": "Perfect-future oracle",
        "policy": {
            "per_home": homes,
            "metrics": {key: summarize_homes([h[key] for h in homes]) for key in METRICS},
        },
        "bound": {
            "lower": float(np.mean(lower)),
            "upper": float(np.mean(upper)),
            "daily_lower": lower,
            "daily_upper": upper,
            "scope": "Perfect-future bound on the combined objective only; component metrics are not independent bounds.",
        },
        "receipts": receipts,
        "protocol_sha256": digest(path / "oracle.json"),
        "reference_commit": protocol["reference_commit"],
        "comfort_definition": "Strict temperature-band membership, identical to learned-policy reporting. Solver-tolerant comfort (1e-6°C) is separately retained per household.",
    }


def collect(root, oracle_root=None):
    from gridpfn.experiments.foundation_study import verify_inputs

    root = Path(root).resolve()
    frozen = freeze_selection(root)
    study = read_json(root / "study.json")
    # Export only complete comparisons. In particular, do not inspect oracle
    # scores while a learned arm is missing its test result or independent audit.
    for row in frozen["methods"]:
        run = root / row["run"]
        if (
            not (run / "evaluation/test_latest.json").exists()
            or not (run / "test_audit.json").exists()
        ):
            raise ValueError(
                "Finish every frozen monthly test and independent replay before export"
            )
    folds = []
    for fold in study["folds"]:
        stage = _safe_path(root, fold["refit"])
        manifest = verify_inputs(stage)
        cfg = manifest["protocol"]
        selected_rows = [r for r in frozen["methods"] if r["fold"] == fold["id"]]
        dates = selected_rows[0]["test_dates"]
        rows = []
        for selected in selected_rows:
            kind = selected["id"]
            run = root / selected["run"]
            path = run / "evaluation/test_latest.json"
            record = read_json(path)
            if (
                record["checkpoint_sha256"] != selected["heads_sha256"]
                or record["episode"] != selected["recipe"]["episodes"]
                or record.get("validation_reward") is not None
            ):
                raise ValueError("Test used a different checkpoint or a validation-selected refit")
            policy = summarize_policy(record, cfg["home_ids"], dates)
            audit = read_json(run / "test_audit.json")["policy_replay"]
            if (
                audit["split"] != "test"
                or audit["dates"] != dates
                or audit["checkpoint_sha256"] != selected["heads_sha256"]
                or audit["checkpoint_episode"] != selected["recipe"]["episodes"]
                or audit["matched_policy_transitions"] != len(dates) * len(cfg["home_ids"]) * 24
                or not np.isfinite(audit["max_absolute_daily_difference"])
                or not 0 <= audit["max_absolute_daily_difference"] <= 1e-8
            ):
                raise ValueError(
                    "Independent original-physics replay failed or belongs to another checkpoint"
                )
            receipts = (
                []
                if kind in {"history", "persistence"}
                else [
                    read_json(stage / "predictions" / kind / f"{c['id']}.json")
                    for c in manifest["cases"]
                ]
            )
            rows.append(
                {
                    "id": kind,
                    "label": LABELS[kind],
                    "policy": policy,
                    "forecast": forecast_metrics(stage, manifest, kind, dates),
                    "selected_episodes": selected["recipe"]["episodes"],
                    "selection_completed_episode": selected["selection_completed_episode"],
                    "selection_budget_cap": selected["selection_budget_cap"],
                    "selection_convergence": selected["selection_convergence"],
                    "checkpoint_sha256": selected["heads_sha256"],
                    "evaluation_sha256": digest(path),
                    "audit": audit,
                    "inference": {
                        "fit_seconds": sum(t["fit_seconds"] for r in receipts for t in r["timing"]),
                        "predict_seconds": sum(
                            t["predict_seconds"] for r in receipts for t in r["timing"]
                        ),
                        "device": receipts[0]["device"] if receipts else "No fitted predictor",
                    },
                    "model": receipts[0]["models"][0] if receipts else None,
                }
            )
        oracle_path = (
            Path(oracle_root) / fold["id"] if oracle_root else _safe_path(root, fold["oracle"])
        )
        oracle = oracle_summary(oracle_path, manifest, dates, frozen["numerical_source"])
        for row in rows:
            if row["audit"]["reference_commit"] != oracle["reference_commit"]:
                raise ValueError("Policy and oracle use different original-simulator references")
            record = read_json(stage / "policies" / row["id"] / "evaluation/test_latest.json")
            daily = -np.mean([[r["reward"] for r in d] for d in record["day_records"]], axis=0)
            if np.any(daily < np.asarray(oracle["bound"]["daily_lower"]) - 1e-5):
                raise ValueError(
                    "A learned daily objective falls below the certified perfect-future lower bound"
                )
        folds.append(
            {
                "id": fold["id"],
                "home_ids": cfg["home_ids"],
                "dates": dates,
                "selection_dates": selected_rows[0]["selection_dates"],
                "rows": rows,
                "oracle": oracle,
            }
        )
    with_oracle = [{**f, "rows": f["rows"] + [f["oracle"]]} for f in folds]
    pooled = pool_monthly_results(with_oracle)
    oracle = pooled["rows"].pop()
    weights = [len(f["dates"]) for f in folds]
    oracle["bound"] = {
        key: float(np.average([f["oracle"]["bound"][key] for f in folds], weights=weights))
        for key in ("lower", "upper")
    }
    oracle["bound"]["scope"] = (
        "Combined objective only; perfect future is unavailable to a deployable controller."
    )
    for row in pooled["rows"]:
        row["forecast"] = pool_forecasts(folds, row["id"], pooled["home_ids"])
    tab = next(r for r in pooled["rows"] if r["id"] == "tabpfn")
    best = min(pooled["rows"], key=lambda r: r["policy"]["metrics"]["objective"]["mean"])
    objective = tab["policy"]["metrics"]["objective"]["mean"]
    finding = (
        f"TabPFN's seasonal combined objective is {objective:.4f}; the lowest observed learned-policy mean is "
        f"{best['policy']['metrics']['objective']['mean']:.4f} ({best['label']}). "
        f"Its gap to the perfect-future upper bound is {objective - oracle['bound']['upper']:.4f}. "
        "These are one-seed outcomes on the same 25 homes; household spread is not statistical evidence of general superiority."
    )
    return {
        "schema_version": 2,
        **pooled,
        "oracle": oracle,
        "folds": folds,
        "protocol": study["protocol"],
        "finding": finding,
        "frozen_selection_sha256": digest(root / "frozen_selection/manifest.json"),
        "frozen_at_utc": frozen["frozen_at_utc"],
        "numerical_source": frozen["numerical_source"],
        "selection_rule": "Select an episode budget on the last seven available calendar days before each test month; refit on all prior training plus validation dates; evaluate the final refit checkpoint without test selection.",
        "limitations": [
            "One seed; same households across time; household SD is heterogeneity, not a confidence interval or evidence of unseen-household generalization.",
            "Monthly validation stopping is an operational plateau criterion, not a proof of global optimizer convergence.",
            "Forecast model pretraining, native preprocessing, devices, and compute differ; contexts and policy recipes are matched.",
            "Within-home forecasts use observed history available by the decision time; measured simulation outcomes are not deployed savings.",
            "Raw household CSVs require authorized access and are not redistributed; code licensing does not grant dataset rights.",
            "TabFM weights retain their research-only license and are not included.",
        ],
    }


def public_evidence(data):
    """Retain identities while removing machine-local paths from model metadata."""

    def clean(value):
        if isinstance(value, dict):
            return {
                k: clean(v)
                for k, v in value.items()
                if k not in {"installed_source", "checkpoint_path", "model_path"}
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and value.startswith(("/home/", "/tmp/")):
            return "local artifact; see recorded SHA256"
        return value

    return clean(data)


def performance_markdown(data):
    def display(row, key):
        value = row["policy"]["metrics"][key]
        return f"{value['mean']:.4f} ± {value['sd']:.4f}"

    rows = data["rows"] + [data["oracle"]]
    text = [
        "# Seasonal performance",
        "",
        data["finding"],
        "",
        "![Complete seasonal comparison](../site/performance.svg)",
        "",
        f"All {len(data['home_ids'])} available households, one seed, {len(data['dates'])} common complete evaluation dates, and five forward monthly folds (June–October 2019). "
        "May supplies initial history. For each month, the previous seven calendar days select a training budget; the final policy is refitted from scratch on all earlier training and validation dates. "
        "The final refit checkpoint is evaluated without further selection. October never trains a reported model.",
        "",
        "## Complete comparison",
        "",
        "Values are mean ± sample SD across household means. Each home is averaged over its evaluated days first; all homes then receive equal weight. "
        "SD describes household heterogeneity. It is **not** a confidence interval, variation over random seeds, or evidence about unseen households.",
        "",
        "| Method | Objective ↓ | Bill ($/home/day) ↓ | Comfortable time (%) ↑ | Violation hours/day ↓ | Squared discomfort ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append("| " + " | ".join([row["label"], *[display(row, k) for k in METRICS]]) + " |")
    text += [
        "",
        "The oracle has perfect knowledge of future traces and obeys the original physical constraints. "
        f"Its certified combined-objective interval is [{data['oracle']['bound']['lower']:.8f}, {data['oracle']['bound']['upper']:.8f}] per home/day. "
        "Only the combined objective is bounded: its bill and comfort are components of that schedule, not independent optimum bounds. "
        "Strict temperature-band membership is used for both learned policies and oracle; solver-tolerant oracle comfort is retained separately in each monthly record.",
        "",
        "## Forecast errors",
        "",
        "RMSE is computed over each household's held-out forecast queries, then summarized with equal household weights. "
        "Month pooling uses query counts within each home before taking its RMSE; it does not average already-rooted errors across months. "
        "Final policies refit from the full eligible pool of prior complete training and validation days. Foundation predictors use the protocol's capped context sampled from available prior labels, rather than every historical row. "
        "Input units are kWh for load/solar and °C for temperature after inverse scaling.",
        "",
        "| Method | Load RMSE | Solar RMSE | Temperature RMSE |",
        "|---|---:|---:|---:|",
    ]
    for row in data["rows"]:
        forecast = row["forecast"]
        values = (
            [
                f"{forecast[k]['mean']:.4f} ± {forecast[k]['sd']:.4f}"
                for k in ("load", "pv", "temperature")
            ]
            if forecast
            else ["—"] * 3
        )
        text.append("| " + " | ".join([row["label"], *values]) + " |")
    text += [
        "",
        "## Monthly outcomes and training budgets",
        "",
        "Every method and month is retained. Early termination requires the declared validation stability criterion; reaching the episode cap does not establish convergence. "
        "Neither outcome proves a global optimum. The selected refit budget can be shorter than the completed selection run.",
        "",
        "| Test month | Method | Selection episodes / cap | Final refit episodes | Selection termination | Test objective |",
        "|---|---|---:|---:|---|---:|",
    ]
    for fold in data["folds"]:
        for row in fold["rows"]:
            text.append(
                f"| {fold['id']} | {row['label']} | {row['selection_completed_episode']} / {row['selection_budget_cap']} | "
                f"{row['selected_episodes']} | {row['selection_convergence']['reason']} | {display(row, 'objective')} |"
            )
    text += [
        "",
        "## Reproduce and inspect",
        "",
        "```bash",
        "python -m gridpfn.experiments.full_report PATH_TO_SEASONAL_STUDY --evaluate --export",
        "```",
        "",
        "The command refuses incomplete comparisons, checks source/settings/input identities, freezes all 30 refit checkpoints, then evaluates each test month and independently replays the original simulator. "
        "It refuses test artifacts created before that all-fold freeze. The output keeps the checkpoint, evaluation, source and oracle receipt hashes.",
        "",
        f"Selection freeze: `{data['frozen_selection_sha256']}` at `{data['frozen_at_utc']}`.",
        "",
        "Download [evidence JSON](../site/performance.json), [SVG](../site/performance.svg), or [PDF](../site/performance.pdf). "
        "Monthly and pooled household values, forecast errors, stopping receipts and model provenance are included in JSON.",
        "",
        "## Access and scope",
        "",
        "The real-data experiment requires authorized household inputs. See [dataset acquisition and restrictions](../dataset/README.md). "
        "Raw CSVs, trained policy weights, and third-party foundation weights are not redistributed with these figures. A synthetic demonstration checks operation but cannot reproduce the reported real-data scores. "
        "Apache-2.0 code licensing does not override dataset or model-weight terms; TabFM retains research-only weight restrictions.",
        "",
    ]
    text += [f"- {item}" for item in data["limitations"]]
    return "\n".join(text) + "\n"


def export(data, root=ROOT):
    root = Path(root)
    (root / "site").mkdir(exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    public = public_evidence(data)
    plot_performance(public, root / "site/performance")
    (root / "docs/PERFORMANCE.md").write_text(performance_markdown(public))
    # Publish the machine-readable completion artifact last.
    write_json(root / "site/performance.json", public)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument(
        "--oracle",
        type=Path,
        help="Optional directory of oracle month folders (2019-06 ... 2019-10); defaults to study.json paths.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Freeze all final refits, then evaluate and audit each held-out month.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Verify completed evaluations and export all-method evidence and figures.",
    )
    args = parser.parse_args(argv)
    if args.evaluate:
        evaluate(args.study)
    if args.export:
        evidence = collect(args.study, args.oracle)
        export(evidence)
        print(evidence["finding"])
    if not args.evaluate and not args.export:
        frozen = freeze_selection(args.study)
        print(f"Frozen {len(frozen['methods'])} final refit checkpoints; no test results read.")


if __name__ == "__main__":
    main()
