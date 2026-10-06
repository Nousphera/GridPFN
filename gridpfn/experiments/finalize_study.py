"""Freeze every seasonal refit, certify oracles, evaluate, publish, and export homes."""

import argparse
from pathlib import Path

from gridpfn.experiments import full_report


def _run_oracle(config, *, resume):
    from oracle.runner import run

    return run(config, resume=resume)


def _export_homes(run, output):
    from gridpfn.deployment import export_home_bundles

    return export_home_bundles(run, output)


def _overlap(a, b):
    return a.is_relative_to(b) or b.is_relative_to(a)


def finalize(study, *, workers=4):
    """Resume the existing numerical pipelines only after the all-fold seal.

    There is deliberately no partial-fold, alternate-tariff, or oracle-objective
    override. Failed or incomplete comparisons cannot publish evidence or models.
    """
    root = Path(study).resolve()
    if type(workers) is not int or workers < 1:
        raise ValueError("Oracle workers must be a positive integer")
    # This must precede importing/running the oracle, inspecting its artifacts,
    # evaluating any policy, or reading any held-out score.
    frozen = full_report.freeze_selection(root)
    manifest = full_report.read_json(root / "study.json")
    if frozen["study_sha256"] != full_report.digest(root / "study.json"):
        raise ValueError("Study manifest changed after the all-fold freeze")

    from oracle.config import OracleConfig

    protected = [root / "frozen_selection", root / "models"]
    for fold in manifest["folds"]:
        protected.extend(full_report._safe_path(root, fold[key]) for key in ("select", "refit"))
    plans = []
    for fold in manifest["folds"]:
        output = full_report._safe_path(root, fold["oracle"])
        if output == root or any(_overlap(output, path) for path in protected):
            raise ValueError("Oracle output overlaps protected study artifacts")
        if any(_overlap(output, cfg.output) for cfg, _ in plans):
            raise ValueError("Oracle fold output paths overlap")
        month = int(fold["id"].split("-")[1])
        config = OracleConfig(
            output=output,
            split="test",
            data_period=f"month_{month:02d}_refit",
            home_ids=tuple(manifest["protocol"]["home_ids"]),
            objectives=("paper_reward",),
            workers=workers,
        )
        resume = output.exists()
        if resume:
            receipt = output / "oracle.json"
            if not receipt.is_file():
                raise ValueError(f"Existing oracle output has no resumable receipt: {output}")
            if full_report.read_json(receipt).get("scenario") != config.settings():
                raise ValueError(f"Existing oracle settings differ; preserve its output: {output}")
        plans.append((config, resume))

    # Preflight every output before starting any expensive work. The runner also
    # verifies its complete source/data/solver protocol and each daily receipt.
    for config, resume in plans:
        _run_oracle(config, resume=resume)
    full_report.evaluate(root)
    evidence = full_report.collect(root)
    full_report.export(evidence)
    final_fold = next(fold for fold in manifest["folds"] if fold["id"] == "2019-10")
    run = full_report._safe_path(root, final_fold["refit"]) / "policies/tabpfn"
    return _export_homes(run, root / "models")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Complete six-method June–October study directory")
    parser.add_argument("--workers", type=int, default=4, help="Oracle CPU workers (default: 4)")
    args = parser.parse_args(argv)
    for directory in finalize(args.study, workers=args.workers):
        print(f"Verified home model: {directory}")


if __name__ == "__main__":
    main()
