"""Bind independent policy replay to its declared chronological split."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import gridpfn.experiments.audit_constraints as audit


@pytest.fixture
def record(tmp_path):
    checkpoint = tmp_path / "checkpoints/best/heads.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable checkpoint")
    dates = ["2019-07-18", "2019-07-19"]
    homes = [27, 950]
    expected = {
        "split": "validation", "dates": dates,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "homes": [{"home_id": home} for home in homes],
        "action_records": np.zeros((2, 2, 24, 4)).tolist(),
    }
    path = tmp_path / "evaluation/validation_best.json"
    path.parent.mkdir()
    path.write_text(json.dumps(expected))
    return tmp_path, checkpoint, dates, homes, expected, path


def test_validation_replay_reads_only_declared_split(record, monkeypatch):
    root, checkpoint, dates, homes, expected, path = record
    original = Path.read_text
    reads = []

    def guarded_read(path, *args, **kwargs):
        reads.append(path)
        assert "test_" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    metrics, requests = audit.policy_evaluation_records(
        root, "best", "validation", dates, homes, checkpoint
    )
    assert metrics == expected
    assert np.shape(requests) == (2, 2, 24, 4)
    assert reads == [path]


@pytest.mark.parametrize("change,match", [
    ({"split": "test"}, "split or dates"),
    ({"dates": ["2019-08-01", "2019-08-02"]}, "split or dates"),
    ({"checkpoint_sha256": "wrong"}, "checkpoint"),
    ({"homes": [{"home_id": 950}, {"home_id": 27}]}, "home order"),
    ({"action_records": []}, "every home"),
])
def test_rejects_mislabeled_or_incomplete_evaluation(record, change, match):
    root, checkpoint, dates, homes, expected, path = record
    path.write_text(json.dumps({**expected, **change}))
    with pytest.raises(ValueError, match=match):
        audit.policy_evaluation_records(root, "best", "validation", dates, homes, checkpoint)


def test_rejects_separate_trace_from_other_split(record):
    root, checkpoint, dates, homes, expected, path = record
    trace = path.with_name("validation_best_trace.json")
    trace.write_text(json.dumps({**expected, "split": "test"}))
    with pytest.raises(ValueError, match="Action trace split"):
        audit.policy_evaluation_records(root, "best", "validation", dates, homes, checkpoint)


def test_default_remains_test_and_invalid_split_fails_before_file_access(tmp_path):
    assert inspect.signature(audit.audit_policy).parameters["split"].default == "test"
    with pytest.raises(ValueError, match="Unknown policy audit split"):
        audit.audit_policy(tmp_path, split="training")


def test_cli_passes_explicit_validation_to_policy_audit(tmp_path, monkeypatch):
    recorded = {}
    monkeypatch.setattr(audit, "audit", lambda reference: {})

    def replay(run, reference, gpu, **kwargs):
        recorded.update(kwargs)
        return {"split": kwargs["split"]}

    monkeypatch.setattr(audit, "audit_policy", replay)
    output = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", [
        "audit_constraints.py", "--policy_run", str(tmp_path),
        "--checkpoint", "best", "--split", "validation", "--output", str(output),
    ])
    audit.main()
    assert recorded["split"] == "validation"
    assert recorded["checkpoint"] == "best"
    assert json.loads(output.read_text())["policy_replay"]["split"] == "validation"
