"""Chronological information and provenance contracts for the foundation prototype."""

import numpy as np
import pytest

from gridpfn.experiments.foundation_study import features, trajectory_features
from gridpfn.experiments.foundation_worker import digest, save_json
from gridpfn.experiments.run_experiment import snapshot


def test_hourly_features_preserve_timing_masks_and_invalid_temperature_slots():
    days = np.zeros((1, 24, 4))
    days[:, :, 2] = 20
    table = np.zeros((1, 24, 24, 3))
    table[:, :, :, 2] = 20
    table[0, 0, 1, 0], table[0, 0, 2, 0] = 6, 12
    values = trajectory_features(days, ["2019-07-18"], table)
    assert values.shape == (1, 25, 32)
    assert values[0, 0, 8] == 1 and values[0, 0, 11] == 2
    assert not values[0, 23, 8:].any()
    assert not values[0, 24].any()
    assert values[0, 22, 26:].tolist() == [1, 0, 0, 0, 0, 0]
    history = trajectory_features(days, ["2019-07-18"], table, history_only=True)
    np.testing.assert_array_equal(history[:, :, :8], values[:, :, :8])
    np.testing.assert_array_equal(history[:, :, 26:], values[:, :, 26:])
    assert not history[:, :, 8:26].any()


def test_future_observations_do_not_change_earlier_history():
    days = np.ones((1, 24, 4))
    days[:, :, 2] = 20
    table = np.zeros((1, 24, 24, 3))
    before = trajectory_features(days, ["2019-07-18"], table)
    days[:, 10:] += 100
    after = trajectory_features(days, ["2019-07-18"], table)
    np.testing.assert_array_equal(before[:, :10], after[:, :10])
    assert not np.array_equal(before[:, 10, :8], after[:, 10, :8])


def test_feature_builder_rejects_changed_home_inputs(tmp_path):
    path = tmp_path / "home_27.npz"
    path.write_bytes(b"original")
    save_json(
        tmp_path / "cases.json",
        {
            "protocol": {"methods": ["history"]},
            "homes": [{"path": path.name, "sha256": digest(path), "id": 27}],
            "cases": [],
        },
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs changed"):
        features(tmp_path, "history")


def test_training_snapshot_excludes_optional_environments(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_text("pass\n")
    extra = root / ".venv-tabfm/lib"
    extra.mkdir(parents=True)
    (extra / "private.py").write_text("do_not_copy=True\n")
    output = tmp_path / "out"
    output.mkdir()
    snapshot(root, output)
    assert (output / "source/model.py").exists()
    assert not (output / "source/.venv-tabfm").exists()


def test_resume_rejects_mutated_case_indices(tmp_path):
    from gridpfn.experiments.foundation_study import verify_inputs

    case = tmp_path / "case.npz"
    np.savez(case, indices=np.array([[0, 0, 1]]))
    save_json(
        tmp_path / "cases.json",
        {"homes": [], "cases": [{"path": case.name, "sha256": digest(case)}]},
    )
    np.savez(case, indices=np.array([[0, 0, 2]]))
    with pytest.raises(ValueError, match="inputs changed"):
        verify_inputs(tmp_path)


def test_feature_manifest_from_another_method_is_rejected(tmp_path):
    from gridpfn.experiments.foundation_study import verify_features

    cfg = {"methods": ["history", "persistence"]}
    save_json(tmp_path / "cases.json", {"protocol": cfg, "homes": [], "cases": []})
    out = tmp_path / "features/history"
    out.mkdir(parents=True)
    save_json(
        out / "manifest.json",
        {
            "kind": "persistence",
            "protocol": cfg,
            "cases_sha256": digest(tmp_path / "cases.json"),
            "files": {},
        },
    )
    with pytest.raises(ValueError, match="different study or method"):
        verify_features(tmp_path, "history")


def test_normalized_features_keep_masked_and_terminal_slots_zero():
    from gridpfn.experiments.foundation_study import training_normalization

    train = np.arange(2 * 24 * 4, dtype=float).reshape(2, 24, 4)
    stats = training_normalization([train])
    table = np.zeros((1, 24, 24, 3))
    values = trajectory_features(train[:1], ["2019-07-18"], table, normalization=stats)
    assert not values[0, 23, 8:26].any()
    assert not values[0, 24].any()
    expected = (np.zeros(3) - stats["mean"]) / np.array(stats["std"])
    np.testing.assert_allclose(values[0, 0, 8:11], expected, rtol=1e-6)
    assert stats["training_rows"] == 48
