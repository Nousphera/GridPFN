"""Compatibility alias; implementation lives in gridpfn.core.predictive_features."""

import sys

from gridpfn.core import predictive_features as _implementation

sys.modules[__name__] = _implementation
