"""Explicit period contracts for the full-cohort oracle, without solving."""

import json
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from oracle import runner
from oracle.config import OracleConfig, load_config
from oracle.runner import _inputs, summarize


def test_oracle_period_defaults_preserve_legacy_and_reject_unknown_values(tmp_path):
    assert OracleConfig().data_period == "legacy"
    for value in ["all", "full_test_only", None, True]:
        with pytest.raises(ValueError, match="data_period"):
            OracleConfig(data_period=value)
    path = tmp_path / "full.toml"
    path.write_text('data_period = "full"\nsplit = "test"\nhome_ids = [27,950]\n')
    config = load_config(path)
    assert config.data_period == "full" and config.settings()["data_period"] == "full"
    assert config.split == "test"
    # Period selection must not change any original physical or tariff setting.
    legacy = replace(config, data_period="legacy").settings()
    assert {**legacy, "data_period": "full"} == config.settings()


@pytest.mark.parametrize("period", ["legacy", "full"])
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_oracle_passes_period_and_declared_split_to_data_loader(tmp_path, monkeypatch, period, split):
    (tmp_path / "home_27.csv").write_text("local input")
    train, heldout = np.zeros((1, 24, 8)), np.ones((1, 24, 8))
    calls = []

    def loader(*args, **kwargs):
        calls.append(kwargs)
        return [(train, heldout, ["2019-09-01"], {
            "delta_t": 1, "col_to_scaler_idx": {}, "train_dates": ["2019-05-01"],
        })]

    monkeypatch.setattr("oracle.runner.load_data", loader)
    monkeypatch.setattr("oracle.runner.compose_em_strategy", lambda strategy, clients: strategy)
    clients, _ = _inputs(OracleConfig(
        data_dir=tmp_path, home_ids=(27,), data_period=period, split=split,
    ))
    assert calls[0]["data_period"] == period
    assert calls[0]["split"] == ("validation" if split == "train" else split)
    assert clients[0].test_dates == (["2019-05-01"] if split == "train" else ["2019-09-01"])
    np.testing.assert_array_equal(clients[0].test_data, train if split == "train" else heldout)


def test_daily_metrics_keep_each_home_date_and_label_bound_scope():
    keys = [
        "reward", "elec_cost", "energy_bill_without_dr", "import", "export", "p2p_kwh",
        "pv_local_use_kwh", "pv_curtailed_kwh", "comfort_pct", "strict_comfort_pct",
        "squared_violation", "ev_completion_ratio", "wm_completed", "final_battery_soe",
    ]
    records = []
    for day in range(2):
        homes = [{**dict.fromkeys(keys, 0.0), "reward": -(day + home + 1)} for home in range(2)]
        records.append({
            "date": f"2019-09-0{day+1}",
            "frontiers": [{"maximum_comfort_pct": 100, "minimum_squared_violation": 0}] * 2,
            "oracles": {"paper_reward": {
                "audit": {"homes": homes, "max_trajectory_difference": 0},
                "solution": {"certificates": [{"certified": True, "absolute_gap": 0}],
                             "lower_bound": sum(-h["reward"] for h in homes),
                             "upper_bound": sum(-h["reward"] for h in homes)},
            }},
        })
    result = summarize(records, [27, 950], ["paper_reward"])["paper_reward"]
    assert result["lower_bound_per_home_day"] == 2
    assert result["upper_bound_per_home_day"] == 2
    assert [row["objective"] for row in result["daily_home_metrics"]] == [1, 2, 2, 3]
    assert [row["home_id"] for row in result["daily_home_metrics"]] == [27, 950, 27, 950]
    assert "not independent lower bounds" in result["bound_scope"]


def test_failed_date_does_not_discard_other_dates_certificates(tmp_path, monkeypatch):
    """A failed first future must not prevent later successful receipt persistence."""
    config = OracleConfig(
        data_dir=tmp_path, output=tmp_path / "oracle", home_ids=(27,),
        objectives=("paper_reward",), data_period="full", split="test", workers=1,
    )
    clients = [SimpleNamespace(
        train_data=np.zeros((1, 24, 8)), test_data=np.zeros((2, 24, 8)),
        test_dates=["2019-09-01", "2019-09-02"],
    )]
    monkeypatch.setattr(runner, "_inputs", lambda config: (clients, {}))
    monkeypatch.setattr(runner, "reference_module", lambda name, revision: (None, "reference-sha"))
    monkeypatch.setattr(runner, "file_sha256", lambda path: "input-sha")
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(runner, "as_completed", lambda futures: iter(futures))

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            future = Future()
            if job[0] == "2019-09-01":
                future.set_exception(RuntimeError("uncertified time limit"))
            else:
                future.set_result({"date": job[0], "frontiers": [], "oracles": {}})
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", Executor)
    with pytest.raises(RuntimeError, match="1 oracle dates failed"):
        runner.run(config)
    assert (config.output / "days/2019-09-02.json").is_file()
    errors = list((config.output / "errors").glob("2019-09-01-*.json"))
    assert len(errors) == 1
    assert "uncertified time limit" in json.loads(errors[0].read_text())["error"]
    assert not (config.output / "summary.json").exists()
    status = json.loads((config.output / "status.json").read_text())
    assert status["state"] == "failed" and status["completed_dates"] == 1
    assert status["failed_dates"] == ["2019-09-01"]


def test_oracle_accepts_explicit_monthly_refit_periods():
    for month in range(6, 11):
        config = OracleConfig(data_period=f"month_{month:02d}_refit", split="test")
        assert config.settings()["data_period"] == f"month_{month:02d}_refit"
    with pytest.raises(ValueError, match="data_period"):
        OracleConfig(data_period="month_11_refit")
