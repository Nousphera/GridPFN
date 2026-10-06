"""Compatibility alias; implementation lives in gridpfn.core.economic_control."""

import sys

from gridpfn.core import economic_control as _implementation

sys.modules[__name__] = _implementation
