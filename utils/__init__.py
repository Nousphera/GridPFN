"""Compatibility aliases for pre-package integrations."""

import importlib
import sys

_implementation = importlib.import_module("gridpfn.core.utils")
for _name in ("agent_utils", "convergence", "feature_cache", "original_learning", "plot_utils", "plots", "preprocess", "rollout_metrics", "run_io", "thermal_planning"):
    sys.modules[f"utils.{_name}"] = importlib.import_module(f"gridpfn.core.utils.{_name}")
sys.modules[__name__] = _implementation
