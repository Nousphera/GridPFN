"""python -m energy_assistant prepare|serve|mcp"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup-chat", help="Download a pinned CPU-only local chat model and runtime")
    prepare = sub.add_parser("prepare", help="Run real local inference and simulator comparisons")
    prepare.add_argument("--output", type=Path, default=Path("results/energy_assistant"))
    prepare.add_argument("--homes", type=int, nargs="+", default=[101])
    prepare.add_argument("--days", type=int, default=1)
    prepare.add_argument("--context", type=int, default=96)
    prepare.add_argument("--backend", choices=["tabpfn", "seasonal"], default="tabpfn")
    prepare.add_argument(
        "--run",
        type=Path,
        help="Trusted completed training run; uses its original data/cohort/scaling",
    )
    prepare.add_argument(
        "--checkpoint",
        choices=["best_feasible", "best", "latest", "initial"],
        default="best_feasible",
    )
    prepare.add_argument("--split", choices=["validation", "test"], default="validation")
    for name in ("serve", "mcp"):
        command = sub.add_parser(name)
        command.add_argument("--evidence", type=Path, default=Path("results/energy_assistant"))
        command.add_argument("--home", type=str, help="Bind this application to one home")
        command.add_argument(
            "--live-tabpfn", action="store_true", help="Run fresh CPU forecasts during tool calls"
        )
        if name == "serve":
            command.add_argument("--port", type=int, default=8770)
            command.add_argument(
                "--local-chat",
                action="store_true",
                help="Start and manage the small local chat model with the app",
            )
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "setup-chat":
        from .local_llm import setup

        setup()
    elif command == "prepare":
        from .prepare import prepare

        prepare(**args)
    elif command == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(
            create_app(
                args["evidence"],
                live=args["live_tabpfn"],
                local_chat=args["local_chat"],
                home=args["home"],
                chat_port=args["port"] + 1,
            ),
            host="127.0.0.1",
            port=args["port"],
        )
    else:
        from .mcp_server import serve

        serve(args["evidence"], live=args["live_tabpfn"], home=args["home"])


if __name__ == "__main__":
    main()
