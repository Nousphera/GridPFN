"""Train the configured TabPFN/FedAvg model and export one local bundle per home."""

import argparse
from pathlib import Path

from gridpfn.deployment import train_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gridpfn.toml"))
    parser.add_argument(
        "--export-only", action="store_true", help="Verify and export an already completed refit"
    )
    args = parser.parse_args()
    for path in train_from_config(args.config, export_only=args.export_only):
        print(f"Home model: {path}")


if __name__ == "__main__":
    main()
