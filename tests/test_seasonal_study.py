"""Stage contexts must remain chronological and refits must include validation."""

import json

import numpy as np
import pandas as pd
import pytest

from gridpfn.core import dataset, forecasting
from gridpfn.experiments.seasonal_study import prepare_stage


@pytest.mark.parametrize("refit,train_days,heldout_days", [(False, 24, 7), (True, 31, 30)])
def test_stage_dates_and_context_labels_are_strictly_separated(
    tmp_path, monkeypatch, refit, train_days, heldout_days
):
    from gridpfn import paths

    inputs = tmp_path / "dataset/split_homes_clean"
    inputs.mkdir(parents=True)
    (inputs / "home_27.csv").write_text("fixture")
    (inputs.parent / "temp_price_newyork.csv").write_text("fixture")
    dates = (
        pd.date_range("2019-05-01", periods=train_days + heldout_days).strftime("%Y-%m-%d").tolist()
    )
    train = np.broadcast_to(np.arange(train_days)[:, None, None], (train_days, 24, 4)).copy()
    heldout = np.full((heldout_days, 24, 4), 99999.0)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(dataset, "home_data_dir", inputs)
    monkeypatch.setattr(
        dataset,
        "load_data",
        lambda *a, **kw: [
            (train, heldout, dates[train_days:], {"train_dates": dates[:train_days]})
        ],
    )
    monkeypatch.setattr(forecasting, "physical_series", lambda values, scaler: values)
    cfg = {
        "home_ids": [27],
        "data_period": "month_06_refit" if refit else "month_06_select",
        "refit": refit,
        "seed": 41,
        "context_rows": 128,
        "horizon": 6,
    }
    output = tmp_path / "stage"
    prepare_stage(output, cfg)
    manifest = json.loads((output / "cases.json").read_text())
    home = manifest["homes"][0]
    assert home["training_days"] == train_days
    assert home["validation_days"] == (0 if refit else heldout_days)
    assert home["test_days"] == (heldout_days if refit else 0)
    assert manifest["cases"][-1]["query_dates"] == dates[train_days:]
    covered = []
    for case in manifest["cases"]:
        assert max(case["context_dates"]) < min(case["query_dates"])
        covered.extend(case["query_dates"])
        with np.load(output / case["path"]) as saved:
            assert saved["y"].max() < case["first"]
            assert "truth" not in saved.files
    assert covered == dates[14:]
