"""CLI: validate a scenario, run it, inspect progress or render results."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check configuration without solving")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute = commands.add_parser("run", help="Solve, certify and independently replay")
    execute.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute.add_argument("--output", type=Path)
    execute.add_argument("--split", choices=("train", "validation", "test"))
    execute.add_argument("--days", type=int)
    execute.add_argument("--workers", type=int)
    execute.add_argument("--time-limit", type=float)
    execute.add_argument("--resume", action="store_true")
    execute.add_argument(
        "--frontier-reference",
        type=Path,
        help="Reuse certified thermal frontiers; exact thermal signatures are checked.",
    )
    status = commands.add_parser("status", help="Show experiment progress")
    status.add_argument("run", type=Path)
    report = commands.add_parser("report", help="Render a compact two-row figure")
    report.add_argument("run", type=Path)
    report.add_argument("--output", type=Path)
    comparisons = report.add_mutually_exclusive_group()
    comparisons.add_argument("--matched", type=Path, help="Optional original/current RL study")
    comparisons.add_argument("--actor-critic", type=Path, help="Completed TabPFN/raw PPO study")
    report.add_argument("--validation", type=Path)
    args = parser.parse_args()
    try:
        if args.command in ("validate", "run"):
            config = load_config(args.config)
            if args.command == "validate":
                print(json.dumps(config.settings(), indent=2))
                return
            overrides = {
                name: getattr(args, name)
                for name in ("output", "split", "days", "workers", "time_limit")
                if getattr(args, name) is not None
            }
            from .runner import run

            summary = run(
                replace(config, **overrides),
                resume=args.resume,
                frontier_reference=args.frontier_reference,
            )
            print(json.dumps({k: v["mean"] for k, v in summary.items() if "mean" in v}, indent=2))
        elif args.command == "status":
            print((args.run / "status.json").read_text(), end="")
        else:
            from .report import render, render_actor_critic, render_scenario

            output = args.output or args.run / "comparison"
            if args.matched:
                render(args.run, args.validation, args.matched, output)
            elif args.actor_critic:
                if args.validation is None:
                    raise ValueError("Actor-critic plots require the matched validation oracle")
                render_actor_critic(args.run, args.actor_critic, args.validation, output)
            else:
                if args.validation:
                    raise ValueError(
                        "--validation accompanies --matched; omit it for scenario plots"
                    )
                render_scenario(args.run, output)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        ImportError,
        KeyError,
        AssertionError,
    ) as error:
        parser.exit(1, f"oracle: {error}\n")


if __name__ == "__main__":
    main()
