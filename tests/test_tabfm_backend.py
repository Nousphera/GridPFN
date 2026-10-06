"""Check the optional TabFM boundary without downloading pretrained weights."""

from unittest.mock import Mock

import numpy as np
import pytest

from gridpfn.foundation_backends import tabfm_backend as backend


def make(**overrides):
    return backend.create_regressor(**{
        "context_size": 128, "n_estimators": 2, "seed": 31, "device": "cpu", **overrides,
    })


def test_lazy_constructor_needs_no_optional_imports(monkeypatch):
    monkeypatch.setattr(backend, "_load_model", Mock(side_effect=AssertionError("not lazy")))
    model = make()
    assert model.estimator_ is None
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1.0]])


def test_full_context_configuration_and_batched_prediction(monkeypatch):
    class Estimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.query_sizes = []

        def fit(self, X, y):
            self.X, self.y = X.copy(), y.copy()

        def predict(self, X):
            self.query_sizes.append(len(X))
            return X[:, 0]

    monkeypatch.setattr(backend, "_load_model", lambda device: (device, Estimator))
    X = np.arange(256, dtype=float).reshape(128, 2)
    reg = make().fit(X, np.arange(128.0))
    assert reg.context_rows_ == 128
    assert reg.metadata_["context_rows"] == 128
    assert reg.metadata_["seed"] == 31
    assert reg.metadata_["device"] == "cpu"
    assert reg.metadata_["n_estimators"] == 2
    assert reg.metadata_["checkpoint_revision"] == backend.CHECKPOINT_REVISION
    assert reg.estimator_.X.shape == (128, 2)
    assert reg.estimator_.kwargs == {
        "model": "cpu", "n_estimators": 2, "max_num_rows": 128,
        "max_num_features": None, "random_state": 31, "batch_size": 1,
        "enable_nnls": False, "cache_context": True, "maybe_quantize_kv_cache": False,
    }
    np.testing.assert_array_equal(reg.predict(X[:67]), X[:67, 0])
    assert reg.estimator_.query_sizes == [32, 32, 3]
    assert reg.predict(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError, match="feature count"):
        reg.predict(np.ones((2, 3)))


@pytest.mark.parametrize("X,y,match", [
    (np.ones((129, 2)), np.ones(129), "centrally"),
    ([[1, 2]], [1], "between 2"),
    ([[1, 2], [3, 4]], [[1], [2]], "vector"),
    ([[1, np.nan], [3, 4]], [1, 2], "finite"),
    ([[1, 2], [3, 4]], [1, np.inf], "finite"),
])
def test_rejects_invalid_context_before_loading(monkeypatch, X, y, match):
    monkeypatch.setattr(backend, "_load_model", Mock(side_effect=AssertionError("loaded")))
    with pytest.raises(ValueError, match=match):
        make().fit(X, y)


@pytest.mark.parametrize("bad_prediction", [np.array([np.nan, 1]), np.ones((2, 1))])
def test_rejects_nonfinite_or_wrong_shape_outputs(monkeypatch, bad_prediction):
    estimator = Mock()
    estimator.predict.return_value = bad_prediction
    monkeypatch.setattr(backend, "_load_model", lambda device: (None, Mock(return_value=estimator)))
    reg = make().fit([[1], [2]], [1, 2])
    with pytest.raises(RuntimeError, match="invalid"):
        reg.predict([[3], [4]])


def test_provenance_discloses_weight_license_without_loading():
    metadata = backend.backend_metadata()
    assert metadata["weights_license"] == "tabfm-non-commercial-v1.0"
    assert len(metadata["checkpoint_revision"]) == 40
    assert len(metadata["tested_source_revision"]) == 40
    assert "checkpoint_sha256" not in metadata or len(metadata["checkpoint_sha256"]) == 64
