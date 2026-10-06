"""Compatibility alias; implementation lives in gridpfn.core.dataset."""

import sys

from gridpfn.core import dataset as _implementation

sys.modules[__name__] = _implementation
