"""Optional Google TabFM regression adapter; weights remain separately licensed.

Install the pinned Google source with its ``pytorch`` extra. Pretrained weights
are downloaded at an immutable Hugging Face revision, never bundled here.
This research comparator uses real labelled context, not hidden-state probes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

SOURCE_REVISION = "fbb665569425fd2f490c6576b3af967876fe11ff"
CHECKPOINT_REPOSITORY = "google/tabfm-1.0.0-pytorch"
CHECKPOINT_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
CHECKPOINT_SHA256 = "bd5a615b0322a8f04a895038de6df6fbd71430eca750e1d792f31048654674a9"
WEIGHTS_LICENSE = "tabfm-non-commercial-v1.0"
_QUERY_BATCH_SIZE = 32
_LOADED_CHECKPOINT: dict = {}


def backend_metadata() -> dict:
    """Return provenance without importing torch or downloading weights."""
    try:
        distribution = importlib.metadata.distribution("tabfm")
        version = distribution.version
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        version, direct_url = None, {}
    source_revision = direct_url.get("vcs_info", {}).get("commit_id")
    source_url = urlparse(direct_url.get("url", ""))
    if source_revision is None and source_url.scheme == "file":
        try:
            source_revision = subprocess.check_output(
                ["git", "-C", unquote(source_url.path), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "backend": "tabfm",
        "package_version": version,
        "installed_source": direct_url,
        "installed_source_revision": source_revision,
        "tested_source_revision": SOURCE_REVISION,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_subfolder": "regression",
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "weights_license": WEIGHTS_LICENSE,
        "source_license": "Apache-2.0",
        "usage": "optional non-commercial, non-production research comparator",
        "compute_dtype": "bfloat16",
        "query_batch_size": _QUERY_BATCH_SIZE,
        **_LOADED_CHECKPOINT,
    }


@lru_cache(maxsize=2)
def _load_model(device: str):
    try:
        from huggingface_hub import snapshot_download
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
    except ImportError as exc:
        raise ImportError(
            "Google TabFM PyTorch dependencies are missing. Install "
            f"'tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@{SOURCE_REVISION}' "
            "and safetensors in an isolated environment. Its pretrained weights use "
            f"the separate {WEIGHTS_LICENSE} license."
        ) from exc
    checkpoint = Path(snapshot_download(
        repo_id=CHECKPOINT_REPOSITORY,
        revision=CHECKPOINT_REVISION,
        allow_patterns=["regression/config.json", "regression/model.safetensors", "LICENSE"],
    ))
    weights = checkpoint / "regression/model.safetensors"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError("TabFM checkpoint digest differs from the pinned research checkpoint")
    _LOADED_CHECKPOINT.update({
        "checkpoint_sha256": digest,
        "checkpoint_bytes": weights.stat().st_size,
    })
    model = tabfm_v1_0_0_pytorch.load(
        model_type="regression", checkpoint_path=str(checkpoint), device=device,
    )
    return model, TabFMRegressor


class _TabFMRegressor:
    def __init__(self, *, context_size: int, n_estimators: int, seed: int, device: str):
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 2:
            raise ValueError("context_size must be an integer of at least 2")
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        self.context_size = context_size
        self.n_estimators = n_estimators
        self.seed = seed
        self.device = device
        self.estimator_ = None

    @staticmethod
    def _features(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite numeric matrix with at least one feature")
        return X

    def fit(self, X, y):
        X = self._features(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("y must be a finite numeric vector matching X rows")
        if not 2 <= len(X) <= self.context_size:
            raise ValueError(
                "Fit rows must be between 2 and context_size; select context centrally. "
                "The TabFM adapter never silently subsamples training rows."
            )
        model, regressor = _load_model(self.device)
        self.estimator_ = regressor(
            model=model, n_estimators=self.n_estimators,
            max_num_rows=self.context_size, max_num_features=None,
            random_state=self.seed, batch_size=1,
            enable_nnls=False, cache_context=True, maybe_quantize_kv_cache=False,
        )
        self.estimator_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.context_rows_ = len(X)
        self.metadata_ = {
            **backend_metadata(),
            "context_rows": self.context_rows_,
            "context_size": self.context_size,
            "n_features": self.n_features_in_,
            "seed": self.seed,
            "device": self.device,
            "n_estimators": self.n_estimators,
            "cache_context": True,
            "quantize_kv_cache": False,
            "max_num_features": None,
            "enable_nnls": False,
        }
        return self

    def predict(self, X):
        if self.estimator_ is None:
            raise RuntimeError("Call fit before predict")
        X = self._features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count does not match fitted context")
        if not len(X):
            return np.empty(0, dtype=np.float64)
        chunks = []
        for start in range(0, len(X), _QUERY_BATCH_SIZE):
            batch = X[start:start + _QUERY_BATCH_SIZE]
            predictions = np.asarray(self.estimator_.predict(batch), dtype=np.float64)
            if predictions.shape != (len(batch),) or not np.isfinite(predictions).all():
                raise RuntimeError("TabFM returned invalid regression predictions")
            chunks.append(predictions)
        return np.concatenate(chunks)


def create_regressor(*, context_size: int, n_estimators: int, seed: int, device: str):
    """Create a lazy regressor; no optional imports or model download until fit."""
    return _TabFMRegressor(
        context_size=context_size, n_estimators=n_estimators, seed=seed, device=device,
    )
