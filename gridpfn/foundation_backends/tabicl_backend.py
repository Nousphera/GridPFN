"""Optional, pinned TabICLv2 regression backend for chronological forecasts.

The caller owns context selection. This adapter never subsamples context, never
uses test labels, and never substitutes a different model after an error.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_VERSION = "2.2.0"
CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
CHECKPOINT_REPO = "jingang/TabICL"
CHECKPOINT_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"


def backend_metadata() -> dict[str, Any]:
    """Describe the requested model without importing Torch or downloading it."""
    try:
        distribution = importlib.metadata.distribution("tabicl")
        installed_version = distribution.version
        source = str(distribution.locate_file("tabicl"))
    except importlib.metadata.PackageNotFoundError:
        installed_version, source = None, None
    return {
        "backend": "tabicl_v2", "package": "tabicl",
        "required_version": PACKAGE_VERSION, "installed_version": installed_version,
        "installed_source": source, "checkpoint": CHECKPOINT,
        "checkpoint_repo": CHECKPOINT_REPO, "checkpoint_revision": CHECKPOINT_REVISION,
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "code_license": "BSD-3-Clause", "weights_license": "BSD-3-Clause",
        "source_url": "https://github.com/soda-inria/tabicl",
        "model_url": f"https://huggingface.co/{CHECKPOINT_REPO}/tree/{CHECKPOINT_REVISION}",
        "mode": "supervised in-context regression; frozen pretrained weights",
        "context_selection": "caller supplied; no internal row truncation",
    }


def _load_backend():
    try:
        from huggingface_hub import hf_hub_download
        from tabicl import TabICLRegressor
    except ImportError as exc:
        raise ImportError(
            f"TabICLv2 requires optional tabicl=={PACKAGE_VERSION}; install it in "
            "the isolated forecast worker environment. No fallback was used."
        ) from exc
    installed = importlib.metadata.version("tabicl")
    if installed != PACKAGE_VERSION:
        raise RuntimeError(f"Expected tabicl=={PACKAGE_VERSION}, found {installed}.")
    return TabICLRegressor, hf_hub_download


def _matrix(X: Any) -> np.ndarray:
    result = np.asarray(X, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("X must be a nonempty 2-D numeric matrix.")
    if not np.isfinite(result).all():
        raise ValueError("X must contain only finite numeric values.")
    return result


class TabICLV2Regressor:
    """Small sklearn-style adapter; runtime provenance is in ``metadata_``."""

    def __init__(self, *, context_size: int, n_estimators: int, seed: int, device: str):
        for name, value in (("context_size", context_size), ("n_estimators", n_estimators)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be explicit, such as 'cpu' or 'cuda:0'.")
        self.context_size = context_size
        self.n_estimators = n_estimators
        self.seed = seed
        self.device = device

    def fit(self, X: Any, y: Any):
        # A failed refit must not leave a usable estimator from a previous fit.
        for attr in ("estimator_", "metadata_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        X = _matrix(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or y.shape[0] != X.shape[0] or not np.isfinite(y).all():
            raise ValueError("y must be a finite 1-D target aligned with X.")
        if X.shape[0] > self.context_size:
            raise ValueError("Caller context exceeds context_size; select rows centrally.")
        regressor_type, download = _load_backend()
        checkpoint_path = Path(download(
            repo_id=CHECKPOINT_REPO, filename=CHECKPOINT, revision=CHECKPOINT_REVISION,
        ))
        with checkpoint_path.open("rb") as handle:
            checkpoint_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if checkpoint_sha != CHECKPOINT_SHA256:
            raise RuntimeError("TabICLv2 checkpoint checksum differs from the pinned model.")
        estimator = regressor_type(
            n_estimators=self.n_estimators, batch_size=min(self.n_estimators, 8),
            random_state=self.seed, device=self.device, n_jobs=1,
            model_path=str(checkpoint_path), checkpoint_version=CHECKPOINT,
            allow_auto_download=False, kv_cache=True, use_amp=False, use_fa3=False,
            verbose=False,
        )
        estimator.fit(X, y)
        self.estimator_ = estimator
        self.n_features_in_ = X.shape[1]
        self.metadata_ = {
            **backend_metadata(), "context_size_limit": self.context_size,
            "context_rows": len(X), "feature_count": X.shape[1],
            "n_estimators": self.n_estimators, "seed": self.seed,
            "device": self.device, "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "kv_cache": True, "use_amp": False, "use_fa3": False, "n_jobs": 1,
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Fit TabICLv2 before predicting.")
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from fit.")
        result = np.asarray(self.estimator_.predict(X), dtype=np.float64)
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise RuntimeError("TabICLv2 returned non-finite or incorrectly shaped predictions.")
        return result


def create_regressor(*, context_size: int, n_estimators: int, seed: int, device: str):
    return TabICLV2Regressor(
        context_size=context_size, n_estimators=n_estimators, seed=seed, device=device,
    )
