"""Run the frozen nine-run submission comparison; evaluation follows all training."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from gridpfn.core.utils.run_io import atomic_json, file_sha256
from gridpfn.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/submission.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/submission_20261006")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    cfg = json.loads(args.protocol.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"protocol": cfg, "sha256": file_sha256(args.protocol)}
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError("Protocol changed; use a new output directory")
    atomic_json(manifest, frozen)
    jobs = [(seed, arm) for seed in cfg["seeds"] for arm in cfg["arms"]]

    def train(job):
        seed, arm = job
        run = output / f"{arm}_seed{seed}"
        if (run / "status.json").exists():
            if json.loads((run / "status.json").read_text())["state"] == "completed":
                return run
            raise RuntimeError(f"Incomplete run preserved: {run}. Inspect before retrying.")
        command = [
            sys.executable,
            "-m", "gridpfn.experiments.run_experiment",
            "--preset",
            cfg["preset"],
            "--path_train",
            str(run),
            "--seed",
            str(seed),
            "--fixed_seed",
            str(seed * 100 + 1),
            "--episode",
            str(cfg["episodes"]),
            "--gpu",
            str(args.gpu),
            "--home_ids",
            *map(str, cfg["home_ids"]),
            *cfg["shared"],
            *cfg["arms"][arm]["options"],
        ]
        with (output / f"{run.name}.log").open("a") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Training complete: {run.name}", flush=True)
        return run

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(train, jobs))
    atomic_json(
        output / "training_complete.json",
        {
            "protocol_sha256": frozen["sha256"],
            "runs": {r.name: file_sha256(r / "selection.json") for r in runs},
        },
    )
    for run in runs:
        for checkpoint in [cfg["primary_checkpoint"], cfg["secondary_checkpoint"]]:
            target = run / "evaluation" / f"test_{checkpoint}.json"
            if not (run / "checkpoints" / checkpoint / "heads.pt").exists():
                raise RuntimeError(f"Missing required checkpoint: {run.name}/{checkpoint}")
            if target.exists():
                continue
            with (output / "evaluation.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m", "gridpfn.experiments.evaluate_checkpoint",
                        str(run),
                        "--checkpoint",
                        checkpoint,
                        "--gpu",
                        str(args.gpu),
                        "--oracle_reference",
                        str(ROOT / "results/oracle/original_test"),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        audit = run / "independent_audit.json"
        if not audit.exists():
            with (output / "audit.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m", "gridpfn.experiments.audit_constraints",
                        "--policy_run",
                        str(run),
                        "--checkpoint",
                        cfg["primary_checkpoint"],
                        "--gpu",
                        str(args.gpu),
                        "--output",
                        str(audit),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print(f"Evaluation and independent replay complete: {run.name}", flush=True)
    atomic_json(
        output / "status.json",
        {"state": "completed", "runs": len(runs), "protocol_sha256": frozen["sha256"]},
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
