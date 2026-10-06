"""Monitor a launched experiment using only the Python standard library."""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from gridpfn.core.utils.run_io import newest_run
from gridpfn.paths import ROOT


def show_losses(events):
    if not events or events[-1]["wmodel"]:
        reason = "not logged during warmup" if events else "waiting for training logs"
        print(f"Actor loss: N/A | Critic loss: N/A ({reason})")
        return

    current_model = events[-1]["model"]
    latest = {}
    for event in events:
        if event["wmodel"]:
            latest.clear()
        elif event["model"] == current_model:
            losses = re.search(
                r"actor_loss:\s*([^,\s]+),\s*critic_loss:\s*([^,\s]+)", event["details"]
            )
            if losses:
                latest[int(event["home"])] = (int(event["episode"]) + 1, *losses.groups())
    if not latest:
        print("Actor loss: N/A | Critic loss: N/A (waiting for loss records)")
        return

    print(f"\n{current_model} losses — latest logged episode per client:")
    print(f"{'Client':>6} {'Episode':>8} {'Actor loss':>16} {'Critic loss':>16}")
    for home, (episode, actor_loss, critic_loss) in sorted(latest.items()):
        print(f"{home:>6} {episode:>8} {actor_loss:>16} {critic_loss:>16}")


def show_progress(run_dir, lines=6):
    study = None
    if (run_dir / "current.json").exists():
        study_path = run_dir / "study.json"
        study = json.loads(study_path.read_text()) if study_path.exists() else None
        selected = (run_dir / json.loads((run_dir / "current.json").read_text())["run"]).resolve()
        if not selected.is_relative_to(run_dir.resolve()):
            raise ValueError("Study run is outside its root")
        if study is not None:
            print(
                f"Study: {study['state']} | {len(study['runs'])}/{study.get('planned_runs', len(study['seeds']) * (6 if study.get('study') == 'control' else 3))} runs complete"
            )
        run_dir = selected
    manifest = json.loads((run_dir / "run.json").read_text())
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    state = status.get("state", "starting")
    pid = status.get("pid", manifest.get("pid"))
    if state == "running" and pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            state = "stopped without a completion record; inspect train.log"

    log_path = run_dir / "gridpfn.experiments.train.log"
    log_text = ""
    if log_path.exists():
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 65536))
            log_text = handle.read().decode("utf-8", errors="replace")

    if manifest.get("mode") == "scheduling":
        print(f"Run: {run_dir}\nStatus: {state} | PID: {pid} | Device: {manifest['gpu']}")
        print(f"Stage: {status.get('stage', 'finished')} | Split: {manifest['eval_split']}")
        metrics = run_dir / "metrics.jsonl"
        if metrics.exists():
            complete = metrics.read_bytes().rsplit(b"\n", 1)[0]
            if complete:
                event = json.loads(complete.splitlines()[-1])
                print(
                    f"Evaluation pass {event['episode']}/{manifest['episodes']}: {event['policy']}"
                )
                print(
                    f"Objective ${event['elec_cost']:.4f}/day | Bill without DR "
                    f"${event['energy_bill_without_dr']:.4f}/day | Comfort {event['comfort_pct']:.2f}%"
                )
        print("\nRecent log lines:")
        print("\n".join(log_text.splitlines()[-lines:]))
        return state in {"completed", "failed"} or state.startswith("stopped")

    completed = [
        name
        for name in manifest["models"]
        if (run_dir / "logs" / name / "server" / "actor.pt").exists()
        and (
            (run_dir / "logs" / name / "server" / "critic.pt").exists()
            or manifest.get("settings", {}).get("local_critics", False)
        )
    ]
    events = list(
        re.finditer(
            r"\[warmup\] (?P<wmodel>\w+) round=(?P<round>\d+)/(?P<total>\d+)"
            r"|\[(?P<model>FedAvg|EdgeHEM|FedFit|PFFDST|FedDMPQ)\][^\n]*?"
            r"episode: (?P<episode>\d+), home: (?P<home>\d+)(?P<details>[^\n]*)",
            log_text,
        )
    )

    print(f"Run: {run_dir}\nStatus: {state} | PID: {pid} | GPU: {manifest['gpu']}")
    print(
        f"Seed: {manifest['seed']} | Clients: {manifest['clients']} | "
        f"Episodes/method: {manifest['episodes']} | Sparsity: {manifest['sparsity']}"
    )
    print(
        f"Methods saved: {len(completed)}/{len(manifest['models'])} "
        f"({', '.join(completed) or 'none yet'})"
    )
    if events:
        event = events[-1]
        if event["wmodel"]:
            print(f"Warmup: {event['wmodel']} | {event['round']}/{event['total']} rounds")
        elif event["model"].lower() in completed:
            print(
                "Run completed; training checkpoints saved."
                if state == "completed"
                else "Training checkpoints saved; see recent logs for the next phase."
            )
        else:
            episode, home = int(event["episode"]), int(event["home"])
            fraction = (episode + home / manifest["clients"]) / manifest["episodes"]
            print(
                f"Training: {event['model']} | episode {episode + 1}/{manifest['episodes']} "
                f"| client {home}/{manifest['clients']} | {fraction:.1%}"
            )
    else:
        print("Initializing models and data; waiting for the first progress record.")
    show_losses(events)
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                f"--id={manifest['gpu']}",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        print(f"GPU total utilization / used VRAM / free VRAM: {gpu.stdout.strip()}")
    except (OSError, subprocess.SubprocessError):
        print("GPU telemetry unavailable")
    print("\nRecent log lines:")
    print("\n".join(log_text.splitlines()[-lines:]))
    return (
        study["state"] in {"completed", "failed"}
        if study
        else state in {"completed", "failed"} or state.startswith("stopped")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", nargs="?", type=Path, help="Defaults to the newest launched run."
    )
    parser.add_argument("--interval", type=float, default=10, help="Refresh interval in seconds.")
    parser.add_argument("--lines", type=int, default=6, help="Recent log lines to show.")
    parser.add_argument(
        "--once", action="store_true", help="Print one snapshot and exit; takes no value."
    )
    args = parser.parse_args()
    if args.interval <= 0 or args.lines <= 0:
        parser.error("--interval and --lines must be positive")
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = newest_run(ROOT / "results")
        if run_dir is None:
            parser.error("No launched runs found; supply a run directory containing run.json")

    run_dir = run_dir.expanduser().resolve()
    if not (run_dir / "run.json").is_file() and not (run_dir / "current.json").is_file():
        parser.error(
            f"No run.json found in {run_dir}. --once takes no value; "
            "use --interval 5 to refresh every 5 seconds, or --once --lines 5 "
            "for one snapshot with 5 log lines."
        )
    try:
        while True:
            if not args.once:
                print("\033[2J\033[H", end="", flush=True)
            finished = show_progress(run_dir, args.lines)
            if args.once or finished:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped. The experiment continues running.")


if __name__ == "__main__":
    main()
