"""Release contracts that do not require downloading foundation weights."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from demo import generate_data
from gridpfn.core.dataset import _construct_dataset
from gridpfn.core.model import Actor, Critic
from gridpfn.experiments.audit_constraints import (
    REFERENCE,
    audit_policy,
    original_defaults,
    reference_module,
)
from gridpfn.experiments.submission_report import collect, summarize, training_contract

ROOT = Path(__file__).resolve().parents[1]


def test_original_physics_reference_works_without_git_history():
    with patch(
        "gridpfn.experiments.audit_constraints.subprocess.check_output",
        side_effect=AssertionError("No Git"),
    ):
        original, digest = reference_module("environment", REFERENCE)
        assert callable(original.HOME_ENERGY_MGNT)
        assert digest == "592fbe13b65e94dfe9d74b4fba65b449b7b0fca481e00b86bb02680218c63f3c"
        assert original_defaults(REFERENCE)["p2p_price"] == 0.1


def test_generated_demo_is_deterministic_complete_and_chronological(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    generate_data(first)
    generate_data(second)
    for path in first.rglob("*.csv"):
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    data = pd.read_csv(first / "split_homes_clean/home_101.csv")
    train, test, dates, scale = _construct_dataset(
        data, temp_price_path=first / "temp_price_newyork.csv", validation_days=14
    )
    assert train.shape == (47, 24, 8)
    assert test.shape == (31, 24, 8)
    assert dates[0] == "2019-08-01" and dates[-1] == "2019-08-31"
    assert max(scale["train_dates"]) == "2019-07-17"
    assert np.isfinite(train).all() and np.isfinite(test).all()
    assert json.loads((first / "PROVENANCE.json").read_text())["uses_private_data"] is False


def test_capacity_control_matches_active_actor_and_value_parameters():
    with patch(
        "gridpfn.core.model._tabpfn_backbone", return_value=SimpleNamespace(embedding_dim=1024)
    ):
        actor = Actor(
            17, 3, feature_mode="hybrid", hidden_dim=64, learned_discrete=True, stochastic_std=0.1
        )
        value = Critic(
            17, 3, 2, feature_mode="hybrid", hidden_dim=64, value_head=True, value_hidden_dim=64
        )
    baseline = Actor(
        17, 3, feature_mode="raw", hidden_dim=2913, learned_discrete=True, stochastic_std=0.1
    )
    base_value = Critic(
        17, 3, 2, feature_mode="raw", hidden_dim=2913, value_head=True, value_hidden_dim=3513
    )

    def count(model):
        return sum(p.numel() for p in model.parameters())

    def active_value(model):
        return sum(p.numel() for n, p in model.named_parameters() if n.startswith("value_"))

    assert abs(count(actor) - count(baseline)) / count(actor) < 0.0002
    assert abs(active_value(value) - active_value(base_value)) / active_value(value) < 0.0002


def test_publication_rejects_incomplete_or_changed_study(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "running"}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Finish every"):
        collect(tmp_path)
    with pytest.raises(ValueError, match="nonfinite"):
        summarize([1, float("nan")])
    with pytest.raises(ValueError, match="Unknown policy checkpoint"):
        audit_policy(tmp_path, checkpoint="unreviewed")


def test_training_contract_rejects_unmatched_learning_rates_and_data(tmp_path):
    cfg = {"arms": ["tabpfn", "mlp"], "seeds": [31], "episodes": 8000, "home_ids": [27]}
    for arm in cfg["arms"]:
        run = tmp_path / f"{arm}_seed31"
        run.mkdir()
        settings = {
            "seed": 31,
            "fixed_seed": 3101,
            "episode": 8000,
            "home_ids": [27],
            "validation_only": True,
            "actor_update": "ppo",
            "bc_rounds": 60,
            "feature_mode": "hybrid" if arm == "tabpfn" else "raw",
            "embedding_weight": 0.1 if arm == "tabpfn" else 1.0,
            "head_width": 64,
            "value_width": 64,
            "lr_actor": 0.0003,
        }
        (run / "run.json").write_text(json.dumps({"settings": settings}))
        (run / "data_hashes.json").write_text(json.dumps({"home_27.csv": "same_input"}))
    assert training_contract(tmp_path, cfg)["shared_settings"]["lr_actor"] == 0.0003
    path = tmp_path / "mlp_seed31/run.json"
    changed = json.loads(path.read_text())
    changed["settings"]["lr_actor"] = 0.003
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Unmatched training"):
        training_contract(tmp_path, cfg)
    changed["settings"]["lr_actor"] = 0.0003
    path.write_text(json.dumps(changed))
    (path.parent / "data_hashes.json").write_text(json.dumps({"home_27.csv": "different_input"}))
    with pytest.raises(ValueError, match="Unmatched training"):
        training_contract(tmp_path, cfg)


def test_seasonal_release_artifact_is_required_when_publishing():
    from gridpfn.release_evidence import load_evidence

    path = ROOT / "site/performance.json"
    if os.environ.get("GRIDPFN_RELEASE_VERIFY") == "1" or path.exists():
        load_evidence(path)
    else:
        # Private development has no exported scores yet. Exercise the fail-closed
        # gate; publication (Pages, packaging, public CI) always requires the file.
        with pytest.raises(ValueError, match="required"):
            load_evidence(path)
