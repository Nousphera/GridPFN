"""Compatibility alias; implementation lives in gridpfn.core.training_metrics."""

import sys

from gridpfn.core import training_metrics as _implementation

sys.modules[__name__] = _implementation
