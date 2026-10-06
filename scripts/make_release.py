"""Package explicitly reviewed source/assets after verified evidence and a clean commit.

The allowlist is the primary boundary. Known-key scanning is secondary and does
not establish that arbitrary credentials can be detected in reviewed content.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# Support both direct script execution and python -m scripts.<name>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridpfn.paths import ROOT
from gridpfn.release_evidence import load_evidence

# Reviewed current source and assets, including compatibility entrypoints and the
# independently maintained app. New tracked paths require an explicit review here;
# never regenerate this list automatically while packaging.
RELEASE_PATHS = frozenset(
    """
.github/workflows/pages.yml
.github/workflows/verify.yml
.github/workflows/assistant.yml
.gitignore
LICENSE
NOTICE
README.md
configs/foundation_prototype.json
configs/foundation_prototype_normalized.json
configs/gridpfn.toml
configs/seasonal.json
configs/submission.json
control_guidance.py
dashboard.html
dataset.py
dataset/README.md
demo.py
docs/ENERGY_ASSISTANT.md
docs/GOAL.md
docs/MODEL_BUNDLE.md
docs/PERFORMANCE.md
docs/REPRODUCE.md
docs/SUBMISSION.md
docs/architecture.md
docs/assets/gridpfn-training.png
docs/assets/gridpfn-assistant.gif
docs/assets/gridpfn-assistant.mp4
docs/energy-assistant-process.svg
docs/gridpfn-system.svg
docs/protocol.md
docs/tabpfn_capabilities.md
economic_control.py
em_strategy.py
energy_assistant/README.md
energy_assistant/__init__.py
energy_assistant/__main__.py
energy_assistant/agent.py
energy_assistant/checkpoints.py
energy_assistant/explanations.py
energy_assistant/history.py
energy_assistant/insights.py
energy_assistant/live.py
energy_assistant/local_llm.py
energy_assistant/mcp_server.py
energy_assistant/periods.py
energy_assistant/prepare.py
energy_assistant/presentation.py
energy_assistant/report.py
energy_assistant/service.py
energy_assistant/settings.py
energy_assistant/simulation.py
energy_assistant/static/app.js
energy_assistant/static/calendar.js
energy_assistant/static/explanation.js
energy_assistant/static/index.html
energy_assistant/static/insights.js
energy_assistant/static/style.css
energy_assistant/trained_forecast.py
energy_assistant/web.py
environment.py
forecasting.py
gridpfn/__init__.py
gridpfn/core/__init__.py
gridpfn/core/agents/agent.py
gridpfn/core/agents/agent_feddmpq.py
gridpfn/core/agents/agent_fedfit.py
gridpfn/core/agents/agent_pffdst.py
gridpfn/core/agents/agent_sparse.py
gridpfn/core/agents/implicit.py
gridpfn/core/agents/onpolicy.py
gridpfn/core/batched_learning.py
gridpfn/core/client.py
gridpfn/core/control_guidance.py
gridpfn/core/dataset.py
gridpfn/core/economic_control.py
gridpfn/core/em_strategy.py
gridpfn/core/environment.py
gridpfn/core/evaluate.py
gridpfn/core/forecasting.py
gridpfn/core/model.py
gridpfn/core/predictive_features.py
gridpfn/core/schedule.py
gridpfn/core/server.py
gridpfn/core/synthetic_days.py
gridpfn/core/tabpfn_adaptation.py
gridpfn/core/training_config.py
gridpfn/core/training_metrics.py
gridpfn/core/utils/agent_utils.py
gridpfn/core/utils/convergence.py
gridpfn/core/utils/feature_cache.py
gridpfn/core/utils/original_learning.py
gridpfn/core/utils/plot_utils.py
gridpfn/core/utils/plots.py
gridpfn/core/utils/preprocess.py
gridpfn/core/utils/rollout_metrics.py
gridpfn/core/utils/run_io.py
gridpfn/core/utils/thermal_planning.py
gridpfn/deployment.py
gridpfn/experiments/__init__.py
gridpfn/experiments/audit_constraints.py
gridpfn/experiments/build_predictive_features.py
gridpfn/experiments/evaluate_checkpoint.py
gridpfn/experiments/finalize_study.py
gridpfn/experiments/foundation_pipeline.py
gridpfn/experiments/foundation_report.py
gridpfn/experiments/foundation_study.py
gridpfn/experiments/foundation_worker.py
gridpfn/experiments/full_report.py
gridpfn/experiments/live_plot.py
gridpfn/experiments/monitor_run.py
gridpfn/experiments/run_experiment.py
gridpfn/experiments/seasonal_study.py
gridpfn/experiments/submission_report.py
gridpfn/experiments/submission_study.py
gridpfn/experiments/train.py
gridpfn/foundation_backends/__init__.py
gridpfn/foundation_backends/tabfm_backend.py
gridpfn/foundation_backends/tabicl_backend.py
gridpfn/foundation_backends/tabpfn_backend.py
gridpfn/legacy/__init__.py
gridpfn/legacy/control_benchmark.py
gridpfn/legacy/plot_matched_study.py
gridpfn/legacy/run_energy_study.py
gridpfn/legacy/run_matched_study.py
gridpfn/paths.py
gridpfn/release_evidence.py
hems_assistant.py
model.py
oracle/README.md
oracle/__init__.py
oracle/__main__.py
oracle/benchmark.py
oracle/config.py
oracle/config.toml
oracle/ipopt.opt
oracle/demonstrations.py
oracle/report.py
oracle/runner.py
oracle/solver.py
predictive_features.py
pyproject.toml
reference/README.md
reference/em_strategy.py.txt
reference/environment.py.txt
reference/manifest.json
reference/train.py.txt
requirements-assistant.txt
requirements-tabfm.txt
requirements-tabicl.txt
requirements.txt
scripts/check_site.py
scripts/build_site.py
scripts/carbon_estimate.py
scripts/assistant_demo/record.py
scripts/make_release.py
scripts/plot_foundation_comparison.py
site/index.html
site/foundation-comparison.svg
site/foundation-comparison.png
site/overview.css
site/performance.css
site/performance.html
site/performance.js
site/performance.json
site/carbon.json
site/performance.pdf
site/performance.png
site/performance.svg
site/process.svg
tests/test_batched_learning.py
tests/test_constraint_audit.py
tests/test_control_benchmark.py
tests/test_control_guidance.py
tests/test_data_period.py
tests/test_deployment.py
tests/test_efficient_pipeline.py
tests/test_energy_assistant.py
tests/test_energy_oracle.py
tests/test_energy_scheduling.py
tests/test_expert_learning.py
tests/test_feature_cache.py
tests/test_final_run.py
tests/test_finalize_study.py
tests/test_foundation_audit.py
tests/test_foundation_report.py
tests/test_foundation_study.py
tests/test_full_report.py
tests/test_live_monitoring.py
tests/test_make_release.py
tests/test_matched_study.py
tests/test_onpolicy.py
tests/test_oracle_period.py
tests/test_package_layout.py
tests/test_predictive_features.py
tests/test_release_evidence.py
tests/test_seasonal_refit.py
tests/test_seasonal_scheduler.py
tests/test_seasonal_study.py
tests/test_shared_scaling.py
tests/test_stronger_pipeline.py
tests/test_submission.py
tests/test_synthetic_days.py
tests/test_tabfm_backend.py
tests/test_tabicl_backend.py
tests/test_tabpfn_models.py
tests/test_temperature_actor.py
tests/test_training_pipeline.py
train.py
training_metrics.py
utils/__init__.py
""".split()
)

PRIVATE_PARTS = frozenset({"wiki", "results", ".git", ".venv", "venv", "__pycache__"})
DATA_SUFFIXES = frozenset(
    {
        ".csv",
        ".tsv",
        ".parquet",
        ".feather",
        ".npz",
        ".npy",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".safetensors",
        ".ckpt",
        ".bin",
        ".h5",
        ".hdf5",
        ".onnx",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".zip",
        ".tar",
        ".gz",
        ".log",
        ".jsonl",
    }
)


def reject_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Symlinks cannot enter release paths: {path}")


def reviewed_payload(root, names):
    """Read every tracked file once; reject unexpected paths, never silently omit."""
    payload = {}
    for name in names:
        path = Path(name)
        if (
            not name
            or path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or "\\" in name
            or name in payload
        ):
            raise ValueError(f"Invalid or duplicate release path: {name}")
        parts = [part.casefold() for part in path.parts]
        if any(part in PRIVATE_PARTS or part.startswith((".env", ".venv-")) for part in parts):
            raise ValueError(f"Private path cannot enter release: {name}")
        if path.suffix.casefold() in DATA_SUFFIXES:
            raise ValueError(f"Raw data, key or model cannot enter release: {name}")
        if name not in RELEASE_PATHS:
            raise ValueError(f"Unexpected tracked path requires release review: {name}")
        source = Path(root) / path
        reject_symlinks(source)
        if not source.is_file():
            raise ValueError(f"Missing regular release file: {name}")
        value = source.read_bytes()
        if re.search(rb"tabpfn_sk_[A-Za-z0-9_-]{20,}", value):
            raise ValueError(f"Potential known credential in {name}")
        payload[name] = value
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/release/gridpfn.zip")
    args = parser.parse_args()
    load_evidence(ROOT / "site/performance.json")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("Commit the reviewed release before packaging")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    payload = reviewed_payload(ROOT, [name for name in paths if name])
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
    reject_symlinks(args.output)
    if args.output.resolve() in {(ROOT / name).resolve() for name in payload}:
        raise ValueError("Release output cannot overwrite a source file")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    blockers = ["Access to the original inputs for the recorded scores remains unresolved."]
    license_path = ROOT / "LICENSE"
    if not license_path.exists() or "Apache License" not in license_path.read_text():
        blockers.append("Owner authorization and Apache-2.0 code license remain pending.")
    report = {
        "commit": commit,
        "files": hashes,
        "public_release_blockers": blockers,
        "scope": "Source-only export. Synthetic demo is reproducible; real-data scores require authorized source inputs.",
        "recorded_score_inputs_included": False,
        "content_review": "Explicit release allowlist; known-key scan is supplementary, not a guarantee of detecting arbitrary credentials.",
        "data_release_status": "Real-input access and redistribution rights remain unresolved. The synthetic demo does not reproduce the recorded real-data scores.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payload.items():
            archive.writestr("gridpfn/" + name, value)
        archive.writestr("gridpfn/RELEASE_MANIFEST.json", json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "archive": str(args.output),
                "files": len(hashes),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "public_release_blockers": blockers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
