"""Run one reproducible FedAvg experiment, then evaluate validation-selected heads."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gridpfn.core.training_config import parse_args
from gridpfn.core.utils.run_io import atomic_json, backbone_identity
from gridpfn.paths import ROOT


def snapshot(root, output):
    source = output / "source"
    for directory, children, files in os.walk(root):
        children[:] = [
            d
            for d in children
            if d not in {".git", "results", "wiki", "__pycache__", ".venv", ".ruff_cache"}
            and not d.startswith(".venv-")
        ]
        for name in files:
            if (
                name.endswith(".py")
                or name in {"dashboard.html", "requirements.txt", "pyproject.toml"}
                or Path(directory).relative_to(root).parts[:1] == ("reference",)
            ):
                path = Path(directory) / name
                target = source / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    (source / "dataset").mkdir(exist_ok=True)
    weather = root / "dataset/temp_price_newyork.csv"
    if weather.exists():
        shutil.copy2(weather, source / "dataset/temp_price_newyork.csv")
    atomic_json(
        output / "source_hashes.json",
        {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
    )
    return source


def main():
    args = parse_args()
    has_validation = args.data_period != "legacy" or bool(args.validation_days)
    if args.run_models != ["fedavg"] or (
        not args.refit and (not has_validation or not args.eval_step)
    ):
        raise ValueError("Final experiment requires FedAvg and periodic chronological validation")
    root = ROOT
    output = Path(args.path_train).expanduser().resolve()
    # Reject output nested in source modules (also prevents recursive snapshots).
    if not output.is_relative_to(root / "results"):
        raise ValueError("Experiment output must be inside the repository results/ directory")
    output.mkdir(parents=True, exist_ok=False)
    source = snapshot(root, output)
    # The child runs in its immutable source snapshot, so input paths must be absolute.
    shared_paths = (
        "embedding_cache",
        "encoder_context",
        "demonstration_cache",
        "predictive_features",
        "oracle_reference",
        "synthetic_data",
        "ppo_oracle_init",
        "selection_reference",
        "refit_selection",
    )
    for name in shared_paths:
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    if args.synthetic_data:
        shutil.copy2(
            Path(args.synthetic_data) / "manifest.json", output / "synthetic_manifest.json"
        )
    command = [
        sys.executable,
        "-u",
        "-m", "gridpfn.experiments.train",
        *sys.argv[1:],
        "--path_train",
        str(output),
        "--path_data",
        str(Path(args.path_data).resolve()),
        "--skip_final_evaluation",
    ]
    for name in shared_paths:
        if getattr(args, name):
            command.extend((f"--{name}", getattr(args, name)))
    atomic_json(
        output / "run.json",
        {
            "created_at": time.time(),
            "models": args.run_models,
            "seed": args.seed,
            "gpu": args.gpu,
            "clients": len(args.home_ids),
            "home_ids": args.home_ids,
            "episodes": args.episode,
            "eval_step": args.eval_step,
            "eval_split": None if args.refit else "validation",
            "sparsity": 0,
            "ac_service": args.ac_service,
            "source_dir": str(source),
            "command": command,
            "settings": vars(args),
        },
    )
    inputs = [Path(args.path_data) / f"home_{home}.csv" for home in args.home_ids]
    atomic_json(
        output / "data_hashes.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    )
    atomic_json(
        output / "runtime.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "tabpfn", "numpy", "scipy", "pandas", "scikit-learn")
            },
        },
    )
    started = time.monotonic()
    try:
        with (output / "gridpfn.experiments.train.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=source)
            atomic_json(output / "status.json", {"state": "running", "pid": process.pid})
            code = process.wait()
            if code:
                raise RuntimeError(f"Training exited with {code}; inspect train.log")
        training_seconds = time.monotonic() - started
        if args.feature_mode != "raw":
            atomic_json(output / "backbone.json", backbone_identity())
        if args.refit:
            records = [
                json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
            ]
            if any(
                row.get("kind", "").startswith("eval")
                or row.get("kind") == "federation_local"
                for row in records
            ):
                raise ValueError("Refit unexpectedly evaluated or selected on observations")
            summary = json.loads((output / "refit_summary.json").read_text())
            if (
                summary["episode"] != args.episode
                or summary["reward"] is not None
                or summary["evaluation_performed"]
            ):
                raise ValueError("Refit summary violates the fixed-budget no-evaluation contract")
            checkpoint = output / "checkpoints/latest/heads.pt"
            atomic_json(
                output / "final_summary.json",
                {
                    "training_seconds": training_seconds,
                    "split": "refit",
                    "refit": True,
                    "evaluation_performed": False,
                    "initial": None,
                    "best": None,
                    "latest": summary,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            )
            atomic_json(
                output / "status.json",
                {"state": "completed", "split": "refit", "seconds": time.monotonic() - started},
            )
            print(f"Completed fixed-budget refit without evaluation: {output}")
            return
        if args.validation_only:
            records = [
                json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
            ]
            validation = [r for r in records if r["kind"] == "eval"]
            eligible = json.loads((output / "selection.json").read_text()).get("best_feasible")
            selected = eligible["episode"] if eligible is not None else None
            convergence = json.loads((output / "convergence.json").read_text())
            if not convergence["stopped"]:
                convergence["reason"] = "episode cap reached; plateau not established"
                atomic_json(output / "convergence.json", convergence)
            atomic_json(
                output / "final_summary.json",
                {
                    "training_seconds": training_seconds,
                    "split": "validation",
                    "convergence": convergence,
                    "initial": validation[0],
                    "selected": next((r for r in validation if r["episode"] == selected), None),
                    "best": max(validation, key=lambda r: r["reward"]),
                    "feasible_selection": eligible is not None,
                    "latest": validation[-1],
                },
            )
            subprocess.run(
                [sys.executable, "-m", "gridpfn.experiments.live_plot", str(output), "--once"],
                cwd=source,
                check=True,
            )
            atomic_json(
                output / "status.json",
                {
                    "state": "completed",
                    "split": "validation",
                    "seconds": time.monotonic() - started,
                },
            )
            print(f"Completed validation-only run: {output}")
            return
        atomic_json(output / "status.json", {"state": "evaluating", "pid": os.getpid()})
        eligible = json.loads((output / "selection.json").read_text()).get("best_feasible")
        labels = ("initial", "best_feasible", "latest") if eligible else ("initial", "latest")
        for label in labels:
            with (output / "evaluation.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m", "gridpfn.experiments.evaluate_checkpoint",
                        str(output),
                        "--checkpoint",
                        label,
                        "--gpu",
                        str(args.gpu),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=source,
                    check=True,
                )
        results = {
            label: json.loads((output / "evaluation" / f"test_{label}.json").read_text())
            for label in labels
        }

        def compact(row):
            return {k: v for k, v in row.items() if k not in {"homes", "day_records", "dates"}}

        convergence = json.loads((output / "convergence.json").read_text())
        if not convergence["stopped"]:
            convergence["reason"] = "episode cap reached; plateau not established"
            atomic_json(output / "convergence.json", convergence)
        atomic_json(
            output / "final_summary.json",
            {
                "training_seconds": training_seconds,
                "convergence": convergence,
                "initial": compact(results["initial"]),
                "selected": compact(results["best_feasible"]) if eligible else None,
                "feasible_selection": eligible is not None,
                "latest": compact(results["latest"]),
            },
        )
        # Rendering is an explicit end-of-run operation; live polling never redraws images.
        subprocess.run(
            [sys.executable, "-m", "gridpfn.experiments.live_plot", str(output), "--once"],
            cwd=source,
            check=True,
        )
        atomic_json(
            output / "status.json",
            {"state": "completed", "pid": os.getpid(), "seconds": time.monotonic() - started},
        )
    except BaseException as exc:
        atomic_json(
            output / "status.json", {"state": "failed", "pid": os.getpid(), "error": str(exc)}
        )
        raise
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
