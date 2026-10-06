"""Run matched forecasting and control with isolated optional model environments."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
from pathlib import Path

from gridpfn.experiments.foundation_study import prepare
from gridpfn.experiments.foundation_worker import save_json
from gridpfn.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/foundation_prototype_normalized.json"
    )
    parser.add_argument("--tabfm-python", type=Path, required=True)
    parser.add_argument("--tabicl-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forecasts-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    prepare(output, args.protocol)
    gpu_lock = threading.Lock()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TABPFN_NO_BROWSER": "1",
    }

    def execute(command, log):
        with log.open("a") as handle:
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
            )

    def method(kind):
        try:
            if kind not in {"history", "persistence"}:
                python = {
                    "tabfm": str(args.tabfm_python.absolute()),
                    "tabicl": str(args.tabicl_python.absolute()),
                }.get(kind, sys.executable)
                device = args.device if kind in {"tabpfn", "tabfm"} else "cpu"
                command = [
                    python,
                    "-m", "gridpfn.experiments.foundation_worker",
                    str(output),
                    "--kind",
                    kind,
                    "--device",
                    device,
                ]
                if device.startswith("cuda"):
                    with gpu_lock:
                        execute(command, output / f"{kind}.log")
                else:
                    execute(command, output / f"{kind}.log")
            for action in ["features"] if args.forecasts_only else ["features", "train"]:
                execute(
                    [
                        sys.executable,
                        "-m", "gridpfn.experiments.foundation_study",
                        action,
                        "--output",
                        str(output),
                        "--kind",
                        kind,
                    ],
                    output / f"pipeline_{kind}.log",
                )
            save_json(output / f"pipeline_{kind}.json", {"state": "completed"})
            print(f"{kind}: forecasts and policy completed", flush=True)
        except Exception as exc:
            save_json(output / f"pipeline_{kind}.json", {"state": "failed", "error": str(exc)})
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(method, ["history", "persistence", "trees", "tabpfn", "tabfm", "tabicl"]))
    save_json(
        output / "pipeline_status.json",
        {"state": "forecasts_completed" if args.forecasts_only else "completed", "methods": 6},
    )


if __name__ == "__main__":
    main()
