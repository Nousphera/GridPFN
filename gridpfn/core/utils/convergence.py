"""Validation-only stopping with the same service gate as checkpoint selection."""

from dataclasses import dataclass

import numpy as np


def stability(records, minimum=3000, checks=20, delta=0.01):
    """Require reward plateau and stable adjacent fixed-date validation windows."""
    if checks < 4 or checks % 2:
        raise ValueError("Convergence checks must be even and at least four")
    if len(records) <= checks or records[-1]["episode"] < minimum:
        return {"stopped": False, "reason": "insufficient validation history"}
    recent, earlier = records[-checks:], records[:-checks]
    previous_best = max(r["reward"] for r in earlier)
    plateau = max(r["reward"] for r in recent) <= previous_best + delta
    half = checks // 2
    drift = {
        key: abs(
            np.mean([r[key] for r in recent[:half]]) - np.mean([r[key] for r in recent[half:]])
        )
        for key in ("reward", "comfort_pct", "elec_cost", "import")
    }
    tolerances = {
        "reward": max(0.05, 0.01 * abs(np.mean([r["reward"] for r in recent]))),
        "comfort_pct": 0.5,
        "elec_cost": 0.05,
        "import": 0.5,
    }
    spread = {key: float(np.std([r[key] for r in recent])) for key in drift}
    stable = all(
        drift[key] <= tolerances[key] and spread[key] <= 2 * tolerances[key] for key in drift
    )
    return {
        "stopped": bool(plateau and stable),
        "reason": "stable validation plateau"
        if plateau and stable
        else "validation still changing",
        "window_checks": checks,
        "unconstrained_reward_plateau": bool(plateau),
        "window_mean_drift": {k: float(v) for k, v in drift.items()},
        "window_sd": spread,
        "tolerances": {k: float(v) for k, v in tolerances.items()},
    }


def feasible(record, reference):
    return (
        record.get("comfort_pct") is not None
        and reference.get("comfort_pct") is not None
        and record.get("elec_cost") is not None
        and reference.get("elec_cost") is not None
        and record["comfort_pct"] >= reference["comfort_pct"] - 1.0
        and record["elec_cost"] <= reference["elec_cost"] + 1e-9
    )


@dataclass
class ValidationStopping:
    min_episodes: int = 3000
    patience: int = 20
    min_delta: float = 0.01
    best: float = -float("inf")
    stale_checks: int = 0
    stopped: bool = False

    def __post_init__(self):
        if min(self.min_episodes, self.patience, self.min_delta) < 0:
            raise ValueError("Stopping settings must be nonnegative")

    def update(self, episode, record, reference):
        if feasible(record, reference) and record["reward"] > self.best + self.min_delta:
            self.best = record["reward"]
            self.stale_checks = 0
        elif episode > 0:
            self.stale_checks += 1
        self.stopped = bool(
            self.patience and episode >= self.min_episodes and self.stale_checks >= self.patience
        )
        return self.stopped
