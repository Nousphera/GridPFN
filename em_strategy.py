"""Compatibility alias; implementation lives in gridpfn.core.em_strategy."""

import sys

from gridpfn.core import em_strategy as _implementation

sys.modules[__name__] = _implementation
