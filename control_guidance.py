"""Compatibility alias; implementation lives in gridpfn.core.control_guidance."""

import sys

from gridpfn.core import control_guidance as _implementation

sys.modules[__name__] = _implementation
