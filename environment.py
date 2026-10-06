"""Compatibility alias; implementation lives in gridpfn.core.environment."""

import sys

from gridpfn.core import environment as _implementation

sys.modules[__name__] = _implementation
