"""Contract checks need neither optional TabICL/Torch nor model downloads."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from gridpfn.foundation_backends import tabicl_backend as backend


@pytest.fixture
def fake(monkeypatch, tmp_path):
    checkpoint = tmp_path / backend.CHECKPOINT
    checkpoint.write_bytes(b"test-only-checkpoint")
    monkeypatch.setattr(backend, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    state = SimpleNamespace(downloads=[], models=[], output=None)

    def download(**kwargs):
        state.downloads.append(kwargs)
        return str(checkpoint)

    class Regressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state.models.append(self)

        def fit(self, X, y):
            self.X, self.y = X.copy(), y.copy()
            return self

        def predict(self, X):
            return np.full(len(X), self.y.mean()) if state.output is None else state.output

    monkeypatch.setattr(backend, "_load_backend", lambda: (Regressor, download))
    return state


def create(**kwargs):
    return backend.create_regressor(context_size=8, n_estimators=2, seed=31, device="cpu", **kwargs)


def test_pinned_model_and_context_are_preserved(fake):
    X, y = np.arange(24).reshape(8, 3), np.arange(8)
    model = create().fit(X, y)
    np.testing.assert_array_equal(fake.models[0].X, X)
    np.testing.assert_array_equal(fake.models[0].y, y)
    assert fake.downloads == [{"repo_id": backend.CHECKPOINT_REPO, "filename": backend.CHECKPOINT,
                               "revision": backend.CHECKPOINT_REVISION}]
    kwargs = fake.models[0].kwargs
    assert kwargs["n_estimators"] == 2
    assert kwargs["random_state"] == 31
    assert kwargs["allow_auto_download"] is False
    assert kwargs["checkpoint_version"] == backend.CHECKPOINT
    assert kwargs["device"] == "cpu"
    assert kwargs["use_amp"] is False
    assert model.metadata_["context_rows"] == 8
    assert model.metadata_["checkpoint_sha256"] == hashlib.sha256(b"test-only-checkpoint").hexdigest()
    np.testing.assert_equal(model.predict(X[:2]), [3.5, 3.5])


def test_context_overflow_is_error_not_truncation(fake):
    with pytest.raises(ValueError, match="select rows centrally"):
        create().fit(np.ones((9, 2)), np.ones(9))
    assert fake.models == []
    assert fake.downloads == []


@pytest.mark.parametrize("X,y", [
    (np.ones(4), np.ones(4)), (np.empty((0, 2)), np.empty(0)),
    (np.ones((2, 2)), np.ones((2, 1))), (np.ones((2, 2)), [1]),
    ([[np.nan, 1]], [1]), ([[1, 2]], [np.inf]),
])
def test_invalid_training_input_rejected_before_download(fake, X, y):
    with pytest.raises(ValueError):
        create().fit(X, y)
    assert fake.downloads == []


@pytest.mark.parametrize("output", [np.array([[1.0], [2.0]]), np.array([np.nan, 1.0]), np.ones(3)])
def test_invalid_backend_predictions_do_not_escape(fake, output):
    model = create().fit(np.ones((4, 2)), np.arange(4))
    fake.output = output
    with pytest.raises(RuntimeError, match="predictions"):
        model.predict(np.ones((2, 2)))


def test_unfitted_and_feature_mismatch(fake):
    model = create()
    with pytest.raises(RuntimeError, match="Fit"):
        model.predict(np.ones((2, 2)))
    model.fit(np.ones((4, 2)), np.arange(4))
    with pytest.raises(ValueError, match="feature count"):
        model.predict(np.ones((2, 3)))
    with pytest.raises(ValueError):
        model.fit(np.ones((9, 2)), np.arange(9))
    with pytest.raises(RuntimeError, match="Fit"):
        model.predict(np.ones((2, 2)))


def test_failures_propagate_without_fallback(monkeypatch):
    def fail():
        raise ImportError("missing optional dependency")
    monkeypatch.setattr(backend, "_load_backend", fail)
    with pytest.raises(ImportError, match="missing optional dependency"):
        create().fit(np.ones((4, 2)), np.arange(4))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget(value):
    with pytest.raises(ValueError, match="positive integer"):
        backend.create_regressor(context_size=8, n_estimators=value, seed=31, device="cpu")


def test_wrong_checkpoint_checksum_rejected(fake, monkeypatch):
    monkeypatch.setattr(backend, "CHECKPOINT_SHA256", "wrong")
    with pytest.raises(RuntimeError, match="checksum"):
        create().fit(np.ones((4, 2)), np.arange(4))
    assert fake.models == []
