"""Local causal direct forecasts, chronological validation and reusable query tables."""

import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

TARGET_COLUMNS = (0, 1, 4, 3)  # fixed energy, PV energy, outdoor C, base price


def physical_series(days, scaler):
    data = np.asarray(days, dtype=float)[..., TARGET_COLUMNS].copy()
    if data.shape[1:] != (24, 4):
        raise ValueError("Forecast scheduling currently requires hourly daily traces")
    for j, column in enumerate(TARGET_COLUMNS):
        i = scaler["col_to_scaler_idx"][column]
        low, high = scaler["min"][i], scaler["max"][i]
        data[..., j] = data[..., j] * ((high - low) or 1) + low
    return data


def causal_queries(days, max_lead=None):
    """Only the prefix through origin enters features; future rows supply labels only."""
    x, y, indices = [], [], []
    for day, values in enumerate(days):
        for origin in range(23):
            current = values[origin]
            previous = values[max(0, origin - 1)]
            average = values[max(0, origin - 3) : origin + 1].mean(0)
            last = 24 if max_lead is None else min(24, origin + max_lead + 1)
            for target in range(origin + 1, last):
                x.append(
                    np.r_[
                        current,
                        previous,
                        average,
                        origin / 24,
                        (target - origin) / 24,
                        np.sin(2 * np.pi * target / 24),
                        np.cos(2 * np.pi * target / 24),
                    ]
                )
                y.append(values[target])
                indices.append((day, origin, target))
    return np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(indices)


class DirectForecaster:
    """Frozen TabPFN or a cheap baseline, fitted on local earlier training days only."""

    def __init__(self, kind="seasonal", context=512, estimators=1, seed=6, device="cuda:1"):
        if kind not in ("persistence", "seasonal", "trees", "tabpfn"):
            raise ValueError(f"Unknown forecast kind: {kind}")
        self.kind, self.context, self.estimators = kind, context, estimators
        self.seed, self.device = seed, device
        self.models = []
        self.fitted = False
        self.fit_seconds = self.predict_seconds = 0.0

    def fit(self, days):
        days = self._validate_days(days)
        if self.context < 1 or self.estimators < 1:
            raise ValueError("Context and estimator budgets must be positive")
        self.models = []
        self.fitted = False
        if len(days) < 2:
            raise ValueError("Forecast fitting needs at least two earlier complete days")
        started = time.monotonic()
        self.profile = np.mean(days, axis=0)
        if self.kind in ("trees", "tabpfn"):
            x, y, _ = causal_queries(days)
            if self.kind == "trees":
                tree = ExtraTreesRegressor(
                    n_estimators=128, min_samples_leaf=5, random_state=self.seed, n_jobs=3
                )
                # Equalize target units so C variance cannot dominate kWh.
                self.target_mean = y.mean(0)
                self.target_scale = y.std(0).clip(1e-6)
                self.models = [tree.fit(x, (y - self.target_mean) / self.target_scale)]
            else:
                from tabpfn import TabPFNRegressor

                indices = np.random.default_rng(self.seed).choice(
                    len(x), min(self.context, len(x)), replace=False
                )
                # The dispatcher persists the common observed base price.
                # Avoid fitting an independent GPU model that it would never use.
                for column in range(3):
                    model = TabPFNRegressor(
                        n_estimators=self.estimators,
                        fit_mode="fit_with_cache",
                        random_state=self.seed,
                        device=self.device,
                    )
                    self.models.append(model.fit(x[indices], y[indices, column]))
        self.fit_seconds = time.monotonic() - started
        self.fitted = True
        return self

    @staticmethod
    def _validate_days(days):
        days = np.asarray(days, dtype=float)
        if days.ndim != 3 or days.shape[1:] != (24, 4) or not len(days):
            raise ValueError("Expected nonempty daily traces with shape (days, 24, 4)")
        if not np.isfinite(days).all():
            raise ValueError("Forecast inputs must be finite")
        return days

    def _predict(self, x, current, origin, target):
        if not self.fitted:
            raise RuntimeError("Fit the forecaster on earlier days before prediction")
        if self.kind == "trees":
            predictions = self.models[0].predict(x) * self.target_scale + self.target_mean
        elif self.kind == "tabpfn":
            predictions = np.column_stack(
                [
                    np.concatenate(
                        [
                            model.predict(chunk)
                            for chunk in np.array_split(x, max(1, (len(x) + 511) // 512))
                        ]
                    )
                    for model in self.models
                ]
            )
            predictions = np.column_stack((predictions, current[:, 3]))
        else:
            predictions = current.copy()
            if self.kind == "seasonal":
                predictions = self.profile[target] + (current - self.profile[origin]) * np.exp(
                    -(target - origin)[:, None] / 4
                )
        if not np.isfinite(predictions).all():
            raise ValueError("Forecaster returned a nonfinite prediction")
        predictions[:, (0, 1, 3)] = np.maximum(predictions[:, (0, 1, 3)], 0)
        return predictions

    def predict_table(self, days, max_lead=None):
        """Batch causal forecasts for benchmark replay; no future values enter inputs."""
        days = self._validate_days(days)
        started = time.monotonic()
        table = np.zeros((len(days), 24, 24, 4), dtype=np.float64)
        x, _, indices = causal_queries(days, max_lead)
        day, origin, target = indices.T
        table[tuple(indices.T)] = self._predict(x, days[day, origin], origin, target)
        for t in range(24):
            table[:, t, t] = days[:, t]
        self.predict_seconds += time.monotonic() - started
        return table

    def predict_prefix(self, history):
        """Online forecasts: current observation followed by remaining-day predictions.

        Only 23-origin future queries are evaluated, rather than all 276 daily
        origin/target pairs. History is physical fixed/PV energy, outdoor C, price.
        """
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[1] != 4 or not 1 <= len(history) <= 24:
            raise ValueError("history must contain 1..24 rows of four physical features")
        if not np.isfinite(history).all():
            raise ValueError("Forecast inputs must be finite")
        if not self.fitted:
            raise RuntimeError("Fit the forecaster on earlier days before prediction")
        origin = len(history) - 1
        if origin == 23:
            return history[-1:].copy()
        started = time.monotonic()
        targets = np.arange(origin + 1, 24)
        origins = np.full(len(targets), origin)
        current = np.repeat(history[-1:], len(targets), axis=0)
        x = np.array(
            [
                np.r_[
                    history[-1],
                    history[max(0, origin - 1)],
                    history[max(0, origin - 3) :].mean(0),
                    origin / 24,
                    (target - origin) / 24,
                    np.sin(2 * np.pi * target / 24),
                    np.cos(2 * np.pi * target / 24),
                ]
                for target in targets
            ],
            dtype=np.float32,
        )
        result = np.vstack([history[-1], self._predict(x, current, origins, targets)])
        self.predict_seconds += time.monotonic() - started
        return result


def forecast_metrics(table, truth):
    leads = {}
    for lead in (1, 3, 6, 12):
        errors = np.stack([table[:, t, t + lead] - truth[:, t + lead] for t in range(24 - lead)])
        leads[str(lead)] = {
            name: float(np.sqrt(np.mean(errors[..., j] ** 2)))
            for j, name in enumerate(("fixed_kwh", "pv_kwh", "outdoor_c", "base_price"))
        }
    return leads
