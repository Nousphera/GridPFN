"""Monthly expanding windows and fixed-budget refits without evaluation leakage."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from gridpfn.core.dataset import _construct_dataset, feature_columns, period_bounds
from gridpfn.core.model import heads_from_state
from gridpfn.core.training_config import MODEL_ORDER, parse_args
from gridpfn.core.training_metrics import refit_selection_contract
from gridpfn.core.utils.agent_utils import safe_torch_load


def synthetic_frame():
    times = pd.date_range("2019-05-01", "2019-11-01", freq="15min", inclusive="left")
    frame = pd.DataFrame({"datetime": times, "t": times.hour * 4 + times.minute // 15})
    for column in feature_columns:
        frame[column] = 0.1
    frame["temp (C)"] = 20 + times.dayofyear / 100
    return frame


def construct(frame, **kwargs):
    with patch("gridpfn.core.dataset.merge_temp_price", side_effect=lambda data, **_: data):
        return _construct_dataset(frame, **kwargs)


def source_selection(root, episode=0, month=6, homes=(27,)):
    root.mkdir(parents=True)
    (root / "selection.json").write_text(json.dumps({"best": {"episode": episode, "reward": -1.0}}))
    (root / "run.json").write_text(json.dumps({"settings": {
        "data_period": f"month_{month:02d}_select", "home_ids": list(homes)
    }}))
    (root / "status.json").write_text(json.dumps({"state": "completed"}))
    return root / "selection.json"


@pytest.mark.parametrize("month", range(6, 11))
def test_monthly_selection_and_refit_exact_calendar_boundaries(month):
    frame = synthetic_frame()
    start = pd.Timestamp(2019, month, 1)
    end = start + pd.offsets.MonthBegin(1)
    last_week = pd.date_range(start - pd.Timedelta(days=7), start, freq="D", inclusive="left")
    select = construct(frame, data_period=f"month_{month:02d}_select", split="validation")
    refit = construct(frame, data_period=f"month_{month:02d}_refit", split="validation")
    test = construct(frame, data_period=f"month_{month:02d}_refit", split="test")
    assert select[2] == refit[2] == last_week.strftime("%Y-%m-%d").tolist()
    assert len(refit[0]) == len(select[0]) + 7
    assert set(select[3]["train_dates"]).isdisjoint(select[2])
    assert set(refit[2]).issubset(refit[3]["train_dates"])
    assert refit[3]["eval_split"] == "refit_diagnostic"
    assert refit[3]["evaluation_is_training_diagnostic"] is True
    assert test[2] == pd.date_range(start, end, inclusive="left").strftime("%Y-%m-%d").tolist()
    assert set(test[2]).isdisjoint(refit[3]["train_dates"])
    assert test[3]["eval_split"] == "test"
    np.testing.assert_array_equal(refit[0], test[0])
    bounds = period_bounds(f"month_{month:02d}_refit")
    assert bounds["train_start_inclusive"] == "2019-05-01"
    assert bounds["train_end_exclusive"] == start.date().isoformat()
    assert bounds["test_end_exclusive"] == end.date().isoformat()


@pytest.mark.parametrize("month", range(6, 11))
def test_monthly_refit_scale_excludes_all_test_and_future_measurements(month):
    frame = synthetic_frame()
    changed = frame.copy()
    changed.loc[changed.datetime >= pd.Timestamp(2019, month, 1), feature_columns] = 1e9
    a = construct(frame, data_period=f"month_{month:02d}_refit", split="test")
    b = construct(changed, data_period=f"month_{month:02d}_refit", split="test")
    np.testing.assert_array_equal(a[0], b[0])
    assert a[3]["min"] == b[3]["min"] and a[3]["max"] == b[3]["max"]
    assert not np.array_equal(a[1], b[1])


def test_refit_mode_overrides_evaluation_and_early_stopping_requests():
    args = parse_args([
        "--preset", "ppo", "--data_period", "month_06_refit",
        "--eval_step", "50", "--patience", "20", "--strict_convergence",
        "--no-skip_final_evaluation",
    ])
    assert args.refit and args.eval_step == args.patience == 0
    assert args.validation_only and args.skip_final_evaluation
    assert not args.strict_convergence
    assert not parse_args(["--data_period", "month_06_select"]).refit
    assert not parse_args([]).refit


def test_refit_requires_previously_selected_budget_and_matching_fold(tmp_path):
    path = source_selection(tmp_path / "selected", episode=50)
    contract = refit_selection_contract(path, 50, "month_06_refit", [27])
    assert contract["selected_episode"] == 50 and len(contract["sha256"]) == 64
    for episode, period, homes in [(0, "month_06_refit", [27]), (50, "month_07_refit", [27]),
                                   (50, "month_06_refit", [950])]:
        with pytest.raises(ValueError, match="differs"):
            refit_selection_contract(path, episode, period, homes)
    with pytest.raises(ValueError, match="requires"):
        refit_selection_contract(None, 50, "month_06_refit", [27])


def test_bc_only_refit_never_constructs_evaluator_and_saves_loadable_final_heads(tmp_path, monkeypatch):
    import gridpfn.experiments.train as train

    path = source_selection(tmp_path / "selected", episode=0)
    output = tmp_path / "refit"
    bundle = construct(synthetic_frame(), data_period="month_06_refit", split="validation")
    args = parse_args([
        "--preset", "ppo", "--data_period", "month_06_refit", "--refit_selection", str(path),
        "--path_train", str(output), "--home_ids", "27", "--episode", "0",
        "--feature_mode", "raw", "--embedding_weight", "1", "--head_width", "8",
        "--value_width", "8", "--bc_rounds", "1", "--bc_steps", "2",
        "--no-ppo_compile_mapping",
    ])

    def runtime(settings, logs):
        train.server_module.logs_root = str(logs)
        settings.embedding_device = "cpu"
        train.configure_embedding_device("cpu", None, None)
        return [torch.device("cpu")]

    def forbidden(*args, **kwargs):
        raise AssertionError("Refit must never construct or invoke an evaluator")

    def load(*args, **kwargs):
        assert kwargs["split"] == "validation"
        assert kwargs["data_period"] == "month_06_refit"
        return [bundle]

    monkeypatch.setattr(train, "configure_runtime", runtime)
    monkeypatch.setattr(train, "load_data", load)
    monkeypatch.setattr(train, "PeriodicEvaluator", forbidden)
    monkeypatch.setattr(train, "run_evaluation", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train.main(args, MODEL_ORDER)
    payload = safe_torch_load(output / "checkpoints/latest/heads.pt", "cpu")
    assert payload["episode"] == 0 and payload["reward"] is None and payload["refit"]
    assert payload["split"] == "refit" and payload["home_ids"] == [27]
    assert payload["selection_source"]["selected_episode"] == 0
    assert payload["comfort_pct"] is payload["elec_cost"] is payload["feasible"] is None
    actor, critic = heads_from_state(payload["clients"][0]["actor"], payload["clients"][0]["critic"], "cpu")
    assert actor.state_dim == 17 and critic.state_dim == 17
    assert set(json.loads((output / "selection.json").read_text())) == {"latest"}
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert not any(row["kind"].startswith("eval") for row in rows)
    assert not (output / "convergence.json").exists()
    assert not (output / "evaluation").exists()
