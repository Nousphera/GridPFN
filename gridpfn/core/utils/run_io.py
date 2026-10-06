"""Shared run discovery, atomic manifests, and settings readers."""

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_json(path, value):
    # Independent workers may update the same dashboard manifest concurrently.
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def newest_run(results):
    """Prefer active runs/studies, then launch time rather than completion mtime."""
    candidates = []
    for path in [*results.glob("*/run.json"), *results.glob("*/study.json")]:
        try:
            manifest = json.loads(path.read_text())
            status_path = path.parent / "status.json"
            status = (
                manifest
                if path.name == "study.json"
                else (json.loads(status_path.read_text()) if status_path.exists() else {})
            )
            active = status.get("state") == "running"
            if active and status.get("pid"):
                try:
                    os.kill(status["pid"], 0)
                except ProcessLookupError:
                    active = False
                except PermissionError:
                    pass
            created = manifest.get("created_at", path.stat().st_mtime)
            candidates.append((active, float(created), path.parent))
        except (OSError, ValueError, TypeError):
            continue
    return max(candidates, key=lambda item: item[:2])[2] if candidates else None


def read_logged_settings(path):
    settings = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            settings[key] = ast.literal_eval(value)
    return settings


def file_sha256(path):
    """Hash large checkpoints without copying them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backbone_identity():
    from tabpfn.model_loading import ModelSource, get_cache_dir

    name = ModelSource.get_v3_5().default_filename
    path = get_cache_dir() / name
    return {"name": name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
