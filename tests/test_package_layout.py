"""Canonical imports and compatibility aliases share one implementation object."""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from gridpfn.paths import ROOT


@pytest.mark.parametrize("name", [
    "dataset", "model", "environment", "forecasting", "predictive_features",
    "training_metrics", "control_guidance", "economic_control", "em_strategy",
])
def test_public_compatibility_module_is_the_canonical_implementation(name):
    legacy = importlib.import_module(name)
    canonical = importlib.import_module(f"gridpfn.core.{name}")
    assert legacy is canonical
    assert Path(canonical.__file__).parent == ROOT / "gridpfn/core"


def test_private_attribute_patch_crosses_compatibility_alias():
    import model
    from gridpfn.core import model as canonical

    with patch("model._tabpfn_backbone", return_value="patched"):
        assert canonical._tabpfn_backbone() == "patched"
        assert model._tabpfn_backbone is canonical._tabpfn_backbone


def test_utility_compatibility_alias_keeps_sidechat_imports_working():
    import utils.rollout_metrics

    from gridpfn.core.utils import rollout_metrics

    assert utils.rollout_metrics is rollout_metrics


def test_paths_and_provenance_follow_the_actual_package():
    from gridpfn.core import dataset
    from gridpfn.experiments.submission_report import CORE_SOURCE

    assert dataset.project_root == ROOT
    assert all((ROOT / name).is_file() for name in CORE_SOURCE)
    assert all(name.startswith("gridpfn/") for name in CORE_SOURCE)


def test_advanced_training_help_renders_literal_percentages():
    from gridpfn.core.training_config import build_parser

    text = build_parser().format_help()
    assert "10% over" in text
