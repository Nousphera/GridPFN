"""Reconstruct the study's exact frozen-weight predictor from its verified X/y context."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from forecasting import DirectForecaster


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_context(settings, home, train_dates):
    if not settings or not settings.get("predictive_features"):
        return None
    root = Path(settings["predictive_features"])
    feature = json.loads((root / "manifest.json").read_text())
    if feature["kind"] != "tabpfn":
        return None
    stage = root.parent.parent
    receipt = stage / "cases.json"
    if sha(receipt) != feature["cases_sha256"]:
        raise ValueError("The trained forecast context manifest changed")
    manifest = json.loads(receipt.read_text())
    cases = [
        c for c in manifest["cases"] if c["home"] == home and c["context_dates"] == train_dates
    ]
    if len(cases) != 1:
        raise ValueError("No unique saved TabPFN context matches the policy training dates")
    case = cases[0]
    path = stage / case["path"]
    if sha(path) != case["sha256"]:
        raise ValueError("The trained TabPFN context changed")
    with np.load(path, allow_pickle=False) as saved:
        x, y = saved["X"].copy(), saved["y"].copy()
    cfg = manifest["protocol"]
    spec = {
        "seed": cfg["seed"],
        "estimators": cfg["n_estimators"],
        "horizon": cfg["horizon"],
        "context_rows": len(x),
        "context_sha256": case["sha256"],
        "context_dates": case["context_dates"],
        "source": "Reconstructed from the same immutable X/y context and pinned TabPFN weights as the study predictor",
    }
    return x, y, spec


class TrainedForecaster(DirectForecaster):
    def predict_table(self, days, max_lead=None):
        table = super().predict_table(days, max_lead=self.horizon)
        # The research predictor supports six hours. Day planning beyond that is
        # an explicit persistence extension, never presented as further predictions.
        for origin in range(24):
            last = min(23, origin + self.horizon)
            if last < 23:
                table[:, origin, last + 1 :] = table[:, origin, last : last + 1]
        return table

    def _predict(self, x, current, origin, target):
        leads = target - origin
        bounded_x = x.copy()
        for i in np.flatnonzero(leads > self.horizon):
            bounded_target = origin[i] + self.horizon
            bounded_x[i, 13] = self.horizon / 24
            bounded_x[i, 14] = np.sin(2 * np.pi * bounded_target / 24)
            bounded_x[i, 15] = np.cos(2 * np.pi * bounded_target / 24)
        return super()._predict(
            bounded_x, current, origin, np.minimum(target, origin + self.horizon)
        )


def fit_context(x, y, spec):
    from gridpfn.foundation_backends.tabpfn_backend import SupervisedTabPFN

    predictor = TrainedForecaster(
        "tabpfn", context=len(x), estimators=spec["estimators"], seed=spec["seed"], device="cpu"
    )
    predictor.horizon = spec["horizon"]
    started = time.monotonic()
    for col in range(3):
        model = SupervisedTabPFN(
            context_size=len(x), n_estimators=spec["estimators"], seed=spec["seed"], device="cpu"
        )
        predictor.models.append(model.fit(x, y[:, col]))
    predictor.fitted = True
    predictor.fit_seconds = time.monotonic() - started
    return predictor, x
