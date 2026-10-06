"""Read-only, scenario-checked oracle gaps for periodic policy evaluation."""

import json
from pathlib import Path

import numpy as np

from .demonstrations import validate_protocol
from .runner import read_record


class OracleComparison:
    def __init__(self, directory, server, split):
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "oracle.json").read_text())
        if (
            manifest["split"] != split
            or json.loads((directory / "status.json").read_text())["state"] != "completed"
        ):
            raise ValueError("Oracle reference must be complete and match the evaluation split")
        validate_protocol(
            manifest, server.clients, server.em_strategy, server.p2p_config, training=False
        )
        self.home_ids = manifest["home_ids"]
        self.records = {
            date: read_record(directory / "days" / f"{date}.json", manifest)
            for date in manifest["dates"]
        }

    def annotate(self, record):
        if (
            [h["home_id"] for h in record["homes"]] != self.home_ids
            or not record["dates"]
            or len(set(record["dates"])) != len(record["dates"])
            or not set(record["dates"]).issubset(self.records)
        ):
            raise ValueError("Policy cohort, home order or dates differ from the oracle")
        rewards = [h["reward"] for h in record["homes"]]
        if (
            not np.isfinite([record["reward"], *rewards]).all()
            or not np.isfinite([h["squared_violation"] for h in record["homes"]]).all()
        ):
            raise ValueError("Policy reward and discomfort must be finite")
        if not np.isclose(record["reward"], np.mean(rewards), atol=1e-8, rtol=1e-8):
            raise ValueError("Policy reward differs from the reported home mean")
        dates = [self.records[d] for d in record["dates"]]
        for i, home in enumerate(record["homes"]):
            oracle_reward = np.mean(
                [r["oracles"]["paper_reward"]["audit"]["homes"][i]["reward"] for r in dates]
            )
            minimum = np.mean([r["frontiers"][i]["minimum_squared_violation"] for r in dates])
            home["oracle_regret"] = float(oracle_reward - home["reward"])
            home["excess_squared_violation"] = float(home["squared_violation"] - minimum)
        for key in ("oracle_regret", "excess_squared_violation"):
            record[key] = float(np.mean([h[key] for h in record["homes"]]))
        lower = np.mean(
            [r["oracles"]["paper_reward"]["solution"]["lower_bound"] for r in dates]
        ) / len(record["homes"])
        if -record["reward"] < lower - 1e-4 or any(
            h["excess_squared_violation"] < -1e-4 for h in record["homes"]
        ):
            raise AssertionError("Policy beats a certified bound; check scenario or accounting")
        record["oracle_reference"] = (
            "Matched perfect-foresight paper objective; per-home gaps are allocations, not individual lower bounds in coupled markets"
        )
