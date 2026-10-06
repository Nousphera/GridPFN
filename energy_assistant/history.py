"""Browse complete recorded days without refitting a model or relabelling them as tests."""

import numpy as np
import pandas as pd

from dataset import _complete_days, _sample_complete_days, merge_temp_price, reshape_to_3d


def recorded_history_bundle(bundle, home_path, weather):
    """Use every complete recorded day with the original training scaler unchanged."""
    train, _, _, scaler = bundle
    data = merge_temp_price(pd.read_csv(home_path), temp_price_path=weather)
    if scaler.get("grid_prices"):
        data["price ($/kWh)"] = np.asarray(scaler["grid_prices"])[data["datetime"].dt.hour]
    data = _sample_complete_days(_complete_days(data))
    dates = data["datetime"].dt.strftime("%Y-%m-%d").drop_duplicates().tolist()
    physical = reshape_to_3d(data)
    if (
        len(dates) != len(physical)
        or physical.shape[1:] != (24, 8)
        or not np.isfinite(physical).all()
    ):
        raise ValueError("Recorded history must contain finite, complete hourly days")
    normalized = physical.copy()
    for column, index in scaler["col_to_scaler_idx"].items():
        if index is not None:
            low, high = scaler["min"][index], scaler["max"][index]
            normalized[..., column] = (physical[..., column] - low) / ((high - low) or 1)
    return train, normalized, dates, scaler


def attach_history(record, bundle, home_path, weather, strategy, fixed_cost):
    from .prepare import recorded_bill

    history = recorded_history_bundle(bundle, home_path, weather)
    _, _, dates, scaler = history
    record["history_bill"] = recorded_bill(history, strategy, fixed_cost)
    record["history_phase"] = {}
    for date in dates:
        if date in scaler["train_dates"]:
            phase = "Training history"
        elif scaler["validation_start_inclusive"] <= date < scaler["validation_end_exclusive"]:
            phase = "Validation period"
        elif scaler["test_start_inclusive"] <= date < scaler["test_end_exclusive"]:
            phase = "Test period"
        else:
            phase = "Other recorded history"
        record["history_phase"][date] = phase
