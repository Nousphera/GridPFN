"""Immutable local causal features; the simulator's 17-value state stays unchanged."""

import hashlib
from pathlib import Path

import numpy as np


class PredictiveContext:
    def __init__(self, path, train_data, scaler):
        path = Path(path)
        self.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            if "privileged" in saved and bool(saved["privileged"]):
                raise ValueError(
                    "Privileged future features are diagnostics, not causal policy inputs"
                )
            self.dates = saved["dates"].tolist()
            self.values = saved["values"].copy()
            expected = str(saved["train_sha256"])
            fitted_dates = saved["train_dates"].tolist()
        actual = hashlib.sha256(np.ascontiguousarray(train_data).tobytes()).hexdigest()
        if expected != actual or fitted_dates != scaler["train_dates"]:
            raise ValueError(
                "Predictive context differs from the training data or chronological split"
            )
        if self.values.ndim != 3 or self.values.shape[:2] != (len(self.dates), 25):
            raise ValueError("Predictive context must contain hourly days including terminal rows")
        if not np.isfinite(self.values).all() or len(set(self.dates)) != len(self.dates):
            raise ValueError("Invalid predictive context values or duplicate dates")
        self.indices = {date: i for i, date in enumerate(self.dates)}
        self.width = self.values.shape[-1]

    def augment(self, state, dates, step):
        states = np.asarray(state, dtype=np.float32)
        single = states.ndim == 1
        dates = [dates] if isinstance(dates, str) else dates
        extras = self.values[[self.indices[d] for d in dates], min(step, 24)]
        combined = np.concatenate((states[None] if single else states, extras), -1)
        return combined[0] if single else combined


def observe(client, state, dates, step):
    context = getattr(client, "predictive_context", None)
    if context is None:
        return state
    if isinstance(dates, (int, np.integer)):
        dates = client.scaler["train_dates"][dates]
    return context.augment(state, dates, step)
