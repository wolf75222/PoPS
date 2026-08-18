"""Manufactured L2 ∝ h² series for the native-order campaign helper.

Does not load the pops package or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

RESOLUTIONS = (16, 32, 64, 128)
ORDER_THRESHOLD = 1.8


def spacings(resolutions=RESOLUTIONS):
    """Return h = 1/n for each cell count on the unit interval."""
    return 1.0 / np.asarray(resolutions, dtype=np.float64)


def manufactured_l2(resolutions=RESOLUTIONS):
    """Return manufactured L2 errors with L2 ∝ h²."""
    return spacings(resolutions) ** 2
