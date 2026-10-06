"""View the recorded submission, or run its controller on generated household data."""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def generate_data(destination, seed=17):
    """Entirely artificial 15-minute traces; no private observations or fitted statistics."""
    destination = Path(destination)
    homes = destination / "split_homes_clean"
    homes.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = {home: [] for home in [101, 102, 103]}
    weather = []
    start = datetime(2019, 6, 1)
    for day in range(92):
        daily_cloud = rng.uniform(0.35, 1)
        warmth = rng.gauss(0, 1.5)
        for quarter in range(96):
            hour = quarter / 4
            stamp = (start + timedelta(days=day, minutes=15 * quarter)).isoformat(sep=" ")
            sun = max(0, math.sin(math.pi * (hour - 6) / 14))
            temperature = 24 + 4 * math.sin(2 * math.pi * (hour - 9) / 24) + warmth
            price = 0.07 + 0.13 * (16 <= hour < 21) + 0.01 * math.sin(day)
            weather.append([stamp, price, temperature])
            for number, home in enumerate(rows):
                fixed = 0.3 + 0.25 * number + 0.9 * (18 <= hour < 22) + rng.random() * 0.2
                pv = sun * daily_cloud * (2 + number)
                cooling = max(0, (temperature - 21) * (0.09 + 0.01 * number))
                ev = (1.5 + 0.4 * number) * (hour < 4)
                washing = 0.45 * (12 <= hour < 15)
                rows[home].append([stamp, cooling / 4, 0, ev / 4, washing / 4, pv / 4, fixed / 4])
    columns = [
        "datetime",
        "ac (kWh)",
        "heater (kWh)",
        "ev (kWh)",
        "wm (kWh)",
        "pv (kWh)",
        "fixed_load (kWh)",
    ]
    for home, records in rows.items():
        with (homes / f"home_{home}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(records)
    with (destination / "temp_price_newyork.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "price ($/kWh)", "temp (C)"])
        writer.writerows(weather)
    (destination / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "kind": "synthetic",
                "seed": seed,
                "homes": list(rows),
                "days": 92,
                "purpose": "Runnable demonstration only; not evidence of real-world savings.",
                "uses_private_data": False,
            },
            indent=2,
        )
    )


def train(args):
    from gridpfn.core.utils.run_io import atomic_json, file_sha256
    from gridpfn.experiments.run_experiment import snapshot

    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Demo output must be inside results/")
    output.mkdir(parents=True, exist_ok=False)
    source = snapshot(ROOT, output)
    generate_data(source / "dataset")
    atomic_json(
        output / "source_hashes.json",
        {
            str(path.relative_to(source)): file_sha256(path)
            for path in source.rglob("*")
            if path.is_file()
        },
    )
    # Training creates its own immutable source snapshot. Its input files are all synthetic.
    run = source / "results/demo"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
    command = [
        sys.executable,
        "-m", "gridpfn.experiments.run_experiment",
        "--preset",
        "ppo",
        "--path_train",
        str(run),
        "--home_ids",
        "101",
        "102",
        "103",
        "--episode",
        str(args.episodes),
        "--eval_step",
        "25",
        "--gpu",
        "0",
        "--bc_rounds",
        "5",
        "--bc_steps",
        "10",
        "--cpu_threads",
        "1",
        "--no-strict_convergence",
        "--patience",
        "0",
        "--validation_only",
    ]
    if args.raw:
        command.extend(["--feature_mode", "raw", "--embedding_weight", "1"])
    subprocess.run(command, cwd=source, env=env, check=True)
    print(
        f"Synthetic run saved: {run}\nThis short smoke run does not establish convergence or savings."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Run TabPFN/PPO on artificial data")
    parser.add_argument("--raw", action="store_true", help="Use the raw MLP instead of TabPFN")
    parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cpu")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "results/synthetic_demo")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if args.train:
        if args.episodes < 25 or args.episodes % 25:
            parser.error("episodes must be a positive multiple of 25")
        train(args)
        return
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "site"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"GridPFN recorded demo: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
