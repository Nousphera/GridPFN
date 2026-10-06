"""Compatibility alias; implementation lives in gridpfn.core.forecasting."""

import sys

from gridpfn.core import forecasting as _implementation

sys.modules[__name__] = _implementation
