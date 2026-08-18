"""AM-04 subcycling oracle: fine dt = coarse_dt / ratio.

Ratios 1, 2, 4. Manufactured temporal error is proportional to dt_fine^2.
Does not import pops or read a PoPS output. Does not require a live runtime.
"""
from __future__ import annotations

RATIOS = (1, 2, 4)
COARSE_DT = 1.0 / 128.0


def fine_dt(coarse_dt, ratio) -> float:
    """Child clock duration is the enclosing coarse step divided by the ratio."""
    count = int(ratio)
    if count < 1:
        raise ValueError(f"subcycling ratio must be >= 1, got {ratio!r}")
    return float(coarse_dt) / float(count)


def fine_steps_per_coarse(ratio) -> int:
    """Number of fine steps that fit in one coarse step."""
    count = int(ratio)
    if count < 1:
        raise ValueError(f"subcycling ratio must be >= 1, got {ratio!r}")
    return count


def manufactured_temporal_error(dt_fine) -> float:
    """RK2-style manufactured temporal error E ∝ dt_fine^2."""
    step = float(dt_fine)
    if step <= 0.0:
        raise ValueError(f"dt_fine must be positive, got {dt_fine!r}")
    return step * step
