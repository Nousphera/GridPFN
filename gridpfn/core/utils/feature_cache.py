"""Atomic local frozen features, keyed by encoder identity and exact input rows."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


class FrozenFeatureCache:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = json.dumps(identity, sort_keys=True).encode()

    def _path(self, rows):
        digest = hashlib.sha256(self.identity)
        digest.update(str(rows.shape).encode())
        digest.update(np.ascontiguousarray(rows, dtype=np.float32).tobytes())
        return self.directory / f"{digest.hexdigest()}.npz"

    def get(self, rows, width):
        path = self._path(rows)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as saved:
            matrix = saved["features"].copy()
            valid = str(saved["key"]) == path.stem
        if (
            not valid
            or matrix.shape != (len(rows), width)
            or matrix.dtype != np.float32
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(f"Invalid frozen feature cache: {path}")
        return matrix

    def put(self, rows, matrix):
        path = self._path(rows)
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, key=path.stem, features=matrix)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
