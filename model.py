"""Compatibility alias; implementation lives in gridpfn.core.model."""

import sys

from gridpfn.core import model as _implementation

sys.modules[__name__] = _implementation
