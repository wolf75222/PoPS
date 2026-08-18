"""AM-02 prescribed moving patch: closed-form center and manufactured regrid jump.

Does not import pops or read a PoPS output. Does not require a live runtime.
The patch trajectory is the TR-02 exact barycenter (x0 + a t) mod 1.
"""
from __future__ import annotations

PERIOD = 1.0
STRESS_CYCLES = 256
ERROR_BEFORE_COEFF = 1.0
ERROR_JUMP_COEFF = 1.0


def patch_center(t, *, x0: float, a: float, period: float = PERIOD) -> float:
    """Closed-form prescribed patch center: (x0 + a t) mod period."""
    return float((float(x0) + float(a) * float(t)) % float(period))


def manufactured_regrid_errors(h: float) -> tuple[float, float]:
    """Return (error_before, error_after) with a manufactured jump ∝ h²."""
    h2 = float(h) ** 2
    before = float(ERROR_BEFORE_COEFF * h2)
    after = float(before + ERROR_JUMP_COEFF * h2)
    return before, after
