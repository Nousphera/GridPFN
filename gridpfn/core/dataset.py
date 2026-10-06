import glob
import os
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from gridpfn.paths import ROOT

project_root = ROOT
dataset_dir = project_root / "dataset"
home_data_dir = dataset_dir / "split_homes_clean"
temp_price_path = dataset_dir / "temp_price_newyork.csv"

PRICE_SCALE_FACTOR = 1.0
SOURCE_STEPS_PER_DAY = 96
SOURCE_DELTA_T_HOURS = 0.25
SAMPLE_RATE = int(os.environ.get("FED_HEMS_SAMPLE_RATE", 4))
STEPS_PER_DAY = SOURCE_STEPS_PER_DAY // SAMPLE_RATE
DELTA_T_HOURS = SOURCE_DELTA_T_HOURS * SAMPLE_RATE

datetime_column = "datetime"
fixed_load_column = "fixed_load (kWh)"
pv_column = "pv (kWh)"
price_column = "price ($/kWh)"
temperature_column = "temp (C)"
ac_column = "ac (kWh)"
ev_column = "ev (kWh)"
wm_column = "wm (kWh)"

energy_columns = [fixed_load_column, pv_column, ac_column, ev_column, wm_column]
dataset_columns = [
    fixed_load_column,
    pv_column,
    "t",
    price_column,
    temperature_column,
    ac_column,
    ev_column,
    wm_column,
]
feature_columns = [*energy_columns[:2], price_column, temperature_column, *energy_columns[2:]]
DATA_PERIODS = (
    "legacy",
    "full",
    *(f"month_{month:02d}_{stage}" for month in range(6, 11) for stage in ("select", "refit")),
)


def period_bounds(data_period="legacy", validation_days=0):
    """Exact calendar boundaries; monthly refits include the prior selection week."""
    if data_period not in DATA_PERIODS:
        raise ValueError("Invalid data_period")
    refit = data_period.endswith("_refit")
    if data_period == "legacy":
        if not 0 <= validation_days < 61:
            raise ValueError("Invalid chronological split")
        train_start = date(2019, 6, 1)
        validation_start = date(2019, 8, 1) - timedelta(days=int(validation_days))
        test_start, test_end = date(2019, 8, 1), date(2019, 9, 1)
    elif data_period == "full":
        train_start, validation_start = date(2019, 5, 1), date(2019, 8, 1)
        test_start, test_end = date(2019, 9, 1), date(2019, 11, 1)
    else:
        month = int(data_period.split("_")[1])
        train_start = date(2019, 5, 1)
        test_start, test_end = date(2019, month, 1), date(2019, month + 1, 1)
        validation_start = test_start - timedelta(days=7)
    return {
        "train_start_inclusive": train_start.isoformat(),
        "train_end_exclusive": (test_start if refit else validation_start).isoformat(),
        "validation_start_inclusive": validation_start.isoformat(),
        "validation_end_exclusive": test_start.isoformat(),
        "test_start_inclusive": test_start.isoformat(),
        "test_end_exclusive": test_end.isoformat(),
        "validation_days": (test_start - validation_start).days,
        "refit": refit,
    }


def setup_seed(seed: int = 100):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def share_training_scale(bundles):
    """Use training-only global extrema while preserving each physical home trace.

    All homes receive the same coordinate system. Only local training extrema
    need aggregation; validation/test values never determine the scale. Constant
    columns keep the same unit-span inverse convention as the simulator.
    """
    if not bundles:
        raise ValueError("Shared scaling requires at least one home")
    low = np.min([b[3]["min"] for b in bundles], axis=0)
    high = np.max([b[3]["max"] for b in bundles], axis=0)
    span = np.where(high == low, 1.0, high - low)
    shared = []
    for train, heldout, dates, scaler in bundles:
        if scaler["cols"] != bundles[0][3]["cols"]:
            raise ValueError("Shared scaling requires matching feature columns")
        local_low, local_high = np.asarray(scaler["min"]), np.asarray(scaler["max"])
        local_span = np.where(local_high == local_low, 1.0, local_high - local_low)

        def convert(array):
            result = array.copy()
            for column, index in scaler["col_to_scaler_idx"].items():
                if index is not None:
                    physical = array[..., column] * local_span[index] + local_low[index]
                    result[..., column] = (physical - low[index]) / span[index]
            return result

        shared.append(
            (
                convert(train),
                convert(heldout),
                dates,
                {**scaler, "min": low.tolist(), "max": high.tolist(), "scaler_mode": "shared"},
            )
        )
    return shared


def load_data(
    path,
    postfix,
    choose=None,
    validation_days=0,
    split="test",
    scaler_mode="local",
    grid_prices=None,
    data_period="legacy",
):
    if scaler_mode not in ("local", "shared"):
        raise ValueError("scaler_mode must be local or shared")

    def construct(frame):
        return _construct_dataset(
            frame, validation_days=validation_days, split=split, grid_prices=grid_prices,
            data_period=data_period
        )

    def scale(bundles):
        return share_training_scale(bundles) if scaler_mode == "shared" else bundles

    files = sorted(glob.glob(str(Path(path) / postfix)))
    if isinstance(choose, int):
        return scale([construct(pd.read_csv(files[choose]))])[0]
    if isinstance(choose, str):
        file = glob.glob(str(Path(path) / f"{choose}{postfix}"))[0]
        return scale([construct(pd.read_csv(file))])[0]
    if isinstance(choose, list):
        datasets = []
        for file in choose:
            if isinstance(file, int):
                datasets.append(construct(pd.read_csv(files[file])))
            elif isinstance(file, str):
                selected = glob.glob(str(Path(path) / f"{file}{postfix}"))[0]
                datasets.append(construct(pd.read_csv(selected)))
        return scale(datasets)
    if choose is None:
        random_choose = np.random.randint(0, len(files))
        return scale([construct(pd.read_csv(files[random_choose]))])[0]
    return scale([construct(pd.read_csv(file)) for file in files]), files


def merge_temp_price(data, temp_price_path=temp_price_path):
    home_data = data.copy()
    home_data[datetime_column] = pd.to_datetime(home_data[datetime_column])
    temp_price = pd.read_csv(
        temp_price_path,
        usecols=[datetime_column, price_column, temperature_column],
        parse_dates=[datetime_column],
    )
    temp_price[price_column] = temp_price[price_column].astype(float) * PRICE_SCALE_FACTOR
    merged = home_data.merge(
        temp_price,
        on=datetime_column,
        how="inner",
    ).sort_values(datetime_column, kind="stable")

    timestamps = merged[datetime_column]
    merged["t"] = timestamps.dt.hour * 4 + timestamps.dt.minute // 15
    return merged


def reshape_to_3d(data, steps_per_day=STEPS_PER_DAY):
    values = []
    for _, day in data.groupby(data[datetime_column].dt.normalize(), sort=False):
        if len(day) != steps_per_day or not np.array_equal(
            day["t"].to_numpy(), np.arange(steps_per_day)
        ):
            continue
        values.append(day[dataset_columns].to_numpy(dtype=float))
    return np.asarray(values, dtype=float)


def _complete_days(data, steps_per_day=SOURCE_STEPS_PER_DAY):
    day = data[datetime_column].dt.normalize()
    coverage = data.groupby(day)["t"].agg(["size", "nunique"])
    complete = coverage.index[
        (coverage["size"] == steps_per_day) & (coverage["nunique"] == steps_per_day)
    ]
    return data[day.isin(complete)].copy()


def _sample_complete_days(data):
    if SAMPLE_RATE == 1:
        return data.copy()

    sampled = data.copy()
    sampled["_sample_day"] = sampled[datetime_column].dt.normalize()
    sampled["_sample_step"] = sampled["t"] // SAMPLE_RATE

    # Aggregate every source interval.
    aggregations = {
        column: (
            "sum"
            if column in energy_columns
            else "mean"
            if column in {price_column, temperature_column}
            else "first"
        )
        for column in data.columns
        if column != "t"
    }
    sampled = (
        sampled.groupby(["_sample_day", "_sample_step"], sort=False, as_index=False)
        .agg(aggregations)
        .rename(columns={"_sample_step": "t"})
        .drop(columns="_sample_day")
    )
    return sampled[data.columns]


def _construct_dataset(
    data, temp_price_path=temp_price_path, validation_days=0, split="test", grid_prices=None,
    data_period="legacy",
):
    """Build chronological daily arrays; fit scaling exclusively on training rows.

    ``full`` fixes May–July training, August validation, and September–October
    testing. ``validation_days`` only controls the legacy June–August period.
    """
    if data_period not in DATA_PERIODS or split not in ("test", "validation"):
        raise ValueError("Invalid chronological split or data_period")
    if data_period == "legacy" and split == "validation" and not validation_days:
        raise ValueError("Validation requires validation_days > 0")
    bounds = period_bounds(data_period, validation_days)
    train_start = pd.Timestamp(bounds["train_start_inclusive"])
    cutoff = pd.Timestamp(bounds["train_end_exclusive"])
    eval_start = pd.Timestamp(bounds[f"{split}_start_inclusive"])
    eval_end = pd.Timestamp(bounds[f"{split}_end_exclusive"])
    data = merge_temp_price(data, temp_price_path=temp_price_path)
    if grid_prices is not None:
        prices = np.asarray(grid_prices, dtype=float)
        if prices.shape != (24,) or not np.isfinite(prices).all() or (prices < 0).any():
            raise ValueError("Grid tariff requires 24 finite nonnegative hourly prices")
        data[price_column] = prices[data[datetime_column].dt.hour.to_numpy()]
    train_data = _sample_complete_days(
        _complete_days(
            data[(data[datetime_column] >= train_start) & (data[datetime_column] < cutoff)]
        )
    )
    test_data = _sample_complete_days(
        _complete_days(
            data[(data[datetime_column] >= eval_start) & (data[datetime_column] < eval_end)]
        )
    )
    if train_data.empty:
        raise ValueError("No complete training days in the requested data_period")
    if test_data.empty:
        raise ValueError(f"No complete {split} days in the requested data_period")

    # The simulator reconstructs physical weather/loads from these values. Clipping
    # unseen test extremes would silently change the evaluated physical scenario.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
    scaler.fit(train_data[feature_columns])
    train_data[feature_columns] = scaler.transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    test_dates = test_data[datetime_column].dt.strftime("%Y-%m-%d").drop_duplicates().tolist()
    train_array = reshape_to_3d(train_data)
    test_array = reshape_to_3d(test_data)

    column_to_feature = {
        column_index: feature_columns.index(column)
        for column_index, column in enumerate(dataset_columns)
        if column in feature_columns
    }
    column_to_feature[dataset_columns.index("t")] = None
    scaler_dict = {
        "min": scaler.data_min_.tolist(),
        "max": scaler.data_max_.tolist(),
        "cols": feature_columns,
        "col_to_scaler_idx": column_to_feature,
        "delta_t": DELTA_T_HOURS,
        "steps_per_day": STEPS_PER_DAY,
        **bounds,
        "data_period": data_period,
        "train_start_inclusive": str(train_start.date()),
        "eval_start_inclusive": str(eval_start.date()),
        "eval_end_exclusive": str(eval_end.date()),
        "eval_split": "refit_diagnostic" if bounds["refit"] and split == "validation" else split,
        "evaluation_is_training_diagnostic": bounds["refit"] and split == "validation",
        "train_end_exclusive": str(cutoff.date()),
        "train_dates": train_data[datetime_column]
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist(),
    }
    if grid_prices is not None:
        scaler_dict["grid_prices"] = prices.tolist()
    return train_array, test_array, test_dates, scaler_dict
