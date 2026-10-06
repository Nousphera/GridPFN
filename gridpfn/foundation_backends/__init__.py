"""Optional foundation models, loaded only by isolated prediction workers."""

import importlib


def create_regressor(kind, **kwargs):
    if kind not in {"tabpfn", "tabfm", "tabicl"}:
        raise ValueError(f"Unknown foundation backend: {kind}")
    module = importlib.import_module(f"gridpfn.foundation_backends.{kind}_backend")
    return module.create_regressor(**kwargs)
