"""Supervised TabPFN-3.5 prediction, rather than artificial-context embeddings."""

import hashlib
from importlib.metadata import version

import numpy as np

CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"
CHECKPOINT_SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"


def backend_metadata():
    return {
        "backend": "tabpfn_3.5",
        "package_version": version("tabpfn"),
        "checkpoint": CHECKPOINT,
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "source_url": "https://github.com/PriorLabs/TabPFN",
        "mode": "supervised in-context regression; frozen pretrained weights",
    }


class SupervisedTabPFN:
    def __init__(self, *, context_size, n_estimators, seed, device):
        self.context_size = context_size
        self.options = dict(n_estimators=n_estimators, random_state=seed, device=device)

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        from tabpfn.model_loading import resolve_model_path

        self.__dict__.pop("estimator_", None)
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or y.shape != (len(X),) or not 0 < len(X) <= self.context_size:
            raise ValueError("Invalid caller-selected context")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite context")
        paths, _, _, _ = resolve_model_path(None, "regressor", version="v3.5")
        path = paths[0]
        if path.name != CHECKPOINT or not path.is_file():
            raise RuntimeError(
                "Obtain the pinned TabPFN-3.5 checkpoint through authorized model access"
            )
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError("TabPFN checkpoint changed")
        estimator = TabPFNRegressor(
            **self.options,
            model_path=path,
            auto_scale_n_estimators=False,
            fit_mode="fit_with_cache",
            n_preprocessing_jobs=1,
            kv_cache_precision=None,
            show_progress_bar=False,
        )
        estimator.fit(X, y)
        self.estimator_ = estimator
        self.metadata_ = {
            **backend_metadata(),
            **self.options,
            "context_rows": len(X),
            "checkpoint_sha256": digest,
            "checkpoint_bytes": path.stat().st_size,
        }
        return self

    def predict(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Fit before predicting")
        result = np.asarray(self.estimator_.predict(X), dtype=np.float64)
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError("Invalid TabPFN predictions")
        return result


def create_regressor(**kwargs):
    return SupervisedTabPFN(**kwargs)
