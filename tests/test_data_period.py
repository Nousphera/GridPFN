"""Chronological full-period coverage and train-only scaling contracts."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gridpfn.core.dataset import _construct_dataset, feature_columns, share_training_scale
from gridpfn.core.training_config import parse_args


def full_frame():
    times = pd.date_range("2019-04-30", "2019-11-02", freq="15min", inclusive="left")
    frame = pd.DataFrame({"datetime": times, "t": times.hour * 4 + times.minute // 15})
    for column in feature_columns:
        frame[column] = 1.0
    # Held-out extremes must not enlarge the train-fitted coordinate system.
    frame["temp (C)"] = np.where(times < "2019-08-01", 20.0, 100.0)
    frame.loc[frame.datetime.between("2019-07-31", "2019-08-01", inclusive="left"), "temp (C)"] = 30.0
    return frame


def construct(frame, **kwargs):
    with patch("gridpfn.core.dataset.merge_temp_price", side_effect=lambda data, **_: data):
        return _construct_dataset(frame, **kwargs)


def test_full_period_complete_calendar_and_exact_disjoint_boundaries():
    frame = full_frame()
    train, val, dates, scale = construct(frame, data_period="full", split="validation")
    train2, test, test_dates, scale2 = construct(frame, data_period="full", split="test")
    assert train.shape == (92, 24, 8)
    assert val.shape == (31, 24, 8)
    assert test.shape == (61, 24, 8)
    assert dates == pd.date_range("2019-08-01", "2019-08-31").strftime("%Y-%m-%d").tolist()
    assert test_dates == pd.date_range("2019-09-01", "2019-10-31").strftime("%Y-%m-%d").tolist()
    assert scale["train_dates"][0] == "2019-05-01"
    assert scale["train_dates"][-1] == "2019-07-31"
    assert not set(scale["train_dates"]) & set(dates + test_dates)
    assert not set(dates) & set(test_dates)
    assert scale["data_period"] == "full"
    assert scale["validation_days"] == 31
    assert scale["train_start_inclusive"] == "2019-05-01"
    assert scale["train_end_exclusive"] == "2019-08-01"
    assert scale["eval_start_inclusive"] == "2019-08-01"
    assert scale2["eval_start_inclusive"] == "2019-09-01"
    assert scale2["eval_end_exclusive"] == "2019-11-01"
    np.testing.assert_array_equal(train, train2)
    assert scale["min"] == scale2["min"]
    assert scale["max"] == scale2["max"]
    assert scale["max"][feature_columns.index("temp (C)")] == 30.0
    np.testing.assert_array_equal(val[..., 4], 8.0)
    np.testing.assert_array_equal(test[..., 4], 8.0)


def test_heldout_changes_do_not_change_local_or_shared_training_scale():
    original = full_frame()
    changed = original.copy()
    changed.loc[changed.datetime >= "2019-08-01", feature_columns] = 1e6
    for split in ("validation", "test"):
        a = construct(original, data_period="full", split=split)
        b = construct(changed, data_period="full", split=split)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[3]["max"] == b[3]["max"]
        assert a[3]["min"] == b[3]["min"]
        shared = share_training_scale([a, b])
        np.testing.assert_array_equal(shared[0][0], shared[1][0])
        assert shared[0][3]["max"] == a[3]["max"]


def test_legacy_default_and_full_does_not_use_legacy_validation_days():
    frame = full_frame()
    default = construct(frame, validation_days=14, split="validation")
    explicit = construct(frame, validation_days=14, split="validation", data_period="legacy")
    np.testing.assert_array_equal(default[0], explicit[0])
    assert default[2] == explicit[2]
    assert default[3]["train_start_inclusive"] == "2019-06-01"
    assert default[3]["train_end_exclusive"] == "2019-07-18"
    assert default[2][0] == "2019-07-18"
    assert default[2][-1] == "2019-07-31"
    full_zero = construct(frame, split="validation", data_period="full", validation_days=0)
    full_fourteen = construct(frame, split="validation", data_period="full", validation_days=14)
    np.testing.assert_array_equal(full_zero[0], full_fourteen[0])
    assert full_zero[2] == full_fourteen[2]
    assert parse_args([]).data_period == "legacy"
    assert parse_args(["--data_period", "full"]).data_period == "full"


def test_partial_and_duplicate_days_are_excluded_not_reconstructed():
    frame = full_frame()
    frame = frame[frame.datetime != pd.Timestamp("2019-08-02 00:15")]
    duplicate = frame[frame.datetime == pd.Timestamp("2019-09-03 00:15")]
    frame = pd.concat([frame, duplicate], ignore_index=True)
    _, val, dates, _ = construct(frame, data_period="full", split="validation")
    _, test, test_dates, _ = construct(frame, data_period="full", split="test")
    assert len(val) == 30 and "2019-08-02" not in dates
    assert len(test) == 60 and "2019-09-03" not in test_dates


def test_invalid_period_rejected():
    with pytest.raises(ValueError, match="data_period"):
        construct(full_frame(), data_period="unknown")


def test_evaluation_and_schedule_preserve_saved_period():
    from gridpfn.core.evaluate import ModelEvaluator
    from gridpfn.core.schedule import load_home_bundles, parse_train_settings

    bundle = construct(full_frame(), data_period="full", split="test")
    with patch("gridpfn.core.evaluate.load_data", return_value=[bundle]) as load:
        evaluator = ModelEvaluator(home_ids=[27], data_period="full")
        assert load.call_args.kwargs["data_period"] == "full"
        assert evaluator.homes[1].test_dates[-1] == "2019-10-31"
    with patch("gridpfn.core.schedule.load_data", return_value=[bundle]) as load:
        replay = load_home_bundles("unused", actual_home_ids=[27], data_period="full")
        assert load.call_args.kwargs["data_period"] == "full"
        assert replay[1].test_dates[0] == "2019-09-01"
    with patch("gridpfn.core.schedule.read_logged_settings", return_value={"data_period": "full"}):
        assert parse_train_settings("unused").data_period == "full"
    with patch("gridpfn.core.schedule.read_logged_settings", return_value={}):
        assert parse_train_settings("unused").data_period == "legacy"
