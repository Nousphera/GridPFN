"""Optional local, live TabPFN inference; household KV context stays in memory."""

import copy
import hashlib
import json
import threading
import time


class LiveForecast:
    def __init__(self, service):
        self.service = service
        self.lock = threading.Lock()
        self.home = None
        self.predictor = None
        self.explanations = {}

    def __call__(self, home, date, *, origin_hour=None, target_hour=None, target_name="demand"):
        import numpy as np
        import torch

        from .explanations import grouped_shapley
        from .prepare import fit_tabpfn

        record, day = self.service.select(home, date)
        if self.service.data["model"]["backend"] != "tabpfn":
            raise ValueError("Live TabPFN requires a TabPFN evidence bundle")
        detailed = origin_hour is not None
        if day is None:
            raise ValueError("No model forecast is available for this date")
        if detailed:
            horizon = record.get("forecast_context", {}).get("horizon", 23)
            if target_name not in {"demand", "solar", "temperature"} or not (
                0 <= origin_hour < target_hour <= min(23, origin_hour + horizon)
            ):
                raise ValueError("Choose a future hour within the trained forecast horizon")
        with self.lock:
            key = (str(home), date, origin_hour, target_hour, target_name)
            if key in self.explanations:
                result = copy.deepcopy(self.explanations[key])
                if "runtime" in result:
                    result["runtime"].update(
                        reused_fitted_context=True, reused_prediction=True, seconds=0.0
                    )
                return result
            torch.set_num_threads(3)
            started = time.monotonic()
            cached = self.home == str(home)
            if not cached:
                path = self.service.directory / f"home_{int(home)}_context.npz"
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["live_context_sha256"]:
                    raise ValueError("Live model context checksum mismatch")
                with np.load(path, allow_pickle=False) as context:
                    self.days = context["days"].copy()
                    self.dates = context["dates"].tolist()
                    train = context["train"].copy()
                    trained = (
                        (
                            context["trained_X"].copy(),
                            context["trained_y"].copy(),
                            json.loads(str(context["trained_spec"])),
                        )
                        if "trained_X" in context
                        else None
                    )
                self.predictor = None  # Keep only one household's model cache in memory.
                if trained is not None:
                    from .trained_forecast import fit_context

                    self.predictor, self.background = fit_context(*trained)
                else:
                    self.predictor, self.background = fit_tabpfn(
                        train, self.service.data["model"]["context_rows"]
                    )
                self.home = str(home)
            origin, target = (
                (origin_hour, target_hour)
                if detailed
                else (day["forecast"]["origin_hour"], day["forecast"]["target_hour"])
            )
            # The model sees only the observed prefix, never the future rows.
            history = self.days[self.dates.index(date), : origin + 1]
            query = np.r_[
                history[-1],
                history[max(0, origin - 1)],
                history[-4:].mean(0),
                origin / 24,
                (target - origin) / 24,
                np.sin(2 * np.pi * target / 24),
                np.cos(2 * np.pi * target / 24),
            ].astype(np.float32)
            if detailed:
                column = {"demand": 0, "solar": 1, "temperature": 2}[target_name]

                def predict(rows):
                    values = self.predictor.models[column].predict(rows)
                    return values if column == 2 else np.maximum(0, values)

                attribution = grouped_shapley(predict, query, self.background[:8])
                result = {
                    "available": True,
                    "home": str(home),
                    "date": date,
                    "origin_hour": origin,
                    "target_hour": target,
                    "target": target_name,
                    "unit": "°C" if column == 2 else "kWh",
                    "model": "TabPFN-3.5",
                    "baseline": attribution["baseline_kwh"],
                    "prediction": attribution["prediction_kwh"],
                    "factors": [
                        {"label": r["feature"], "value": r["kwh"]}
                        for r in attribution["contributions"]
                    ],
                    "method": attribution["method"],
                    "additivity_error": attribution["additivity_error"],
                    "background_rows": attribution["background_rows"],
                    "recent_readings": [
                        {"hour": h, "value": float(history[h, column])}
                        for h in range(max(0, origin - 3), origin + 1)
                    ],
                    "observed_inputs": [
                        {
                            "hour": h,
                            "demand_kwh": float(history[h, 0]),
                            "solar_kwh": float(history[h, 1]),
                            "outdoor_c": float(history[h, 2]),
                            "base_price": float(history[h, 3]),
                        }
                        for h in range(max(0, origin - 3), origin + 1)
                    ],
                    "context_sha256": record.get("forecast_context", {}).get("context_sha256"),
                    "caveat": attribution["caveat"],
                }
                if len(self.explanations) >= 128:
                    self.explanations.pop(next(iter(self.explanations)))
                self.explanations[key] = copy.deepcopy(result)
                return result
            prediction = self.predictor.predict_prefix(history)
            prediction = prediction[: len(day["forecast"]["rows"])]
            result = copy.deepcopy(day["forecast"])
            for row, values in zip(result["rows"], prediction, strict=True):
                row.update(
                    load_kwh=float(values[0]),
                    solar_kwh=float(values[1]),
                    outdoor_c=float(values[2]),
                )
            result["explanation"] = grouped_shapley(
                lambda rows: np.maximum(0, self.predictor.models[0].predict(rows)),
                query,
                self.background[:8],
            )
            result["runtime"] = {
                "execution": "live local TabPFN-3.5 inference",
                "reused_fitted_context": cached,
                "seconds": time.monotonic() - started,
                "device": "cpu",
                "thinking_mode": False,
            }
            result = {"home": home, "date": date, "model": "TabPFN-3.5", **result}
            if len(self.explanations) >= 128:
                self.explanations.pop(next(iter(self.explanations)))
            self.explanations[key] = copy.deepcopy(result)
            return result
