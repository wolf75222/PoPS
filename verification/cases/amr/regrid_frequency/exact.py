"""AM-05 regrid-frequency oracle: rebuild every k steps of Δt.

Prescribed moving pulse is the TR-02 translated Gaussian. The tagged
window is rebuilt on steps 0, k, 2k, … . Leftover cadence is
|k_regrid - k_requested|. The exact field does not depend on k.

Does not import pops or read a PoPS output. Does not require a live runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

K_VALUES = (1, 2, 4, 8, 16)
N_STEPS = 16
DT = 1.0 / float(N_STEPS)
N_CELLS = 64
PERIOD = 1.0

_TR02_EXACT = (
    Path(__file__).resolve().parents[2] / "transport" / "gaussian_pulse" / "exact.py"
)
_tr02 = load_sibling_module(_TR02_EXACT)
WINDOW_HALF_WIDTH = 4.0 * float(_tr02.SIGMA)


def expected_rebuilds(k, n_steps: int = N_STEPS) -> int:
    """Return the documented rebuild count N/k over a closed step interval."""
    interval = int(k)
    steps = int(n_steps)
    if interval < 1:
        raise ValueError(f"regrid interval must be >= 1, got {k!r}")
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if steps % interval != 0:
        raise ValueError(f"n_steps={steps} must be divisible by k={interval}")
    return steps // interval


def should_rebuild(step, k) -> bool:
    """True on the documented cadence: rebuild when step is a multiple of k."""
    interval = int(k)
    if interval < 1:
        raise ValueError(f"regrid interval must be >= 1, got {k!r}")
    return int(step) % interval == 0


def interval_leftover(k_regrid, k_requested) -> float:
    """Return |k_regrid - k_requested| for the leftover cadence observation."""
    return abs(float(k_regrid) - float(k_requested))


def patch_center(t, *, x0=None, a=None, period: float = PERIOD) -> float:
    """Closed-form prescribed pulse center: (x0 + a t) mod period."""
    origin = _tr02.X0 if x0 is None else x0
    speed = _tr02.A if a is None else a
    return float((float(origin) + float(speed) * float(t)) % float(period))


def tagged_window_mask(x, center, *, half_width=None, period: float = PERIOD):
    """Periodic mask of cells inside the prescribed pulse window."""
    width = WINDOW_HALF_WIDTH if half_width is None else float(half_width)
    displacement = np.asarray(x, dtype=np.float64) - float(center)
    return np.abs(_tr02.minimum_image(displacement, period)) <= width


def exact_field(x, t, **kwargs):
    """Return the TR-02 Gaussian. Independent of the regrid interval k."""
    return _tr02.exact_gaussian(x, t, **kwargs)
