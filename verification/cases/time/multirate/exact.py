"""TM-06 exact two-rate linear decays: y' = -λ_f y, z' = -λ_s z.

Closed form: y(t) = y0 e^{-λ_f t}, z(t) = z0 e^{-λ_s t}.
No pops dependency. Does not read PoPS output.
"""
from __future__ import annotations

import math

LAMBDA_F = 8.0
LAMBDA_S = 1.0
Y0 = 1.0
Z0 = 1.0
DT = 0.25
RATIOS = (1, 2, 4, 8)


def exact_y(t, y0=Y0, *, lambda_f=LAMBDA_F) -> float:
    """Return y0 * exp(-λ_f t)."""
    return float(y0) * math.exp(-float(lambda_f) * float(t))


def exact_z(t, z0=Z0, *, lambda_s=LAMBDA_S) -> float:
    """Return z0 * exp(-λ_s t)."""
    return float(z0) * math.exp(-float(lambda_s) * float(t))


def exact_state(t, y0=Y0, z0=Z0, *, lambda_f=LAMBDA_F, lambda_s=LAMBDA_S):
    """Return (y(t), z(t)) of the uncoupled linear pair."""
    return (
        exact_y(t, y0, lambda_f=lambda_f),
        exact_z(t, z0, lambda_s=lambda_s),
    )
