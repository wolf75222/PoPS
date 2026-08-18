"""RB-05 off-center Sedov blast. 2-d mesh, spherical self-similar radius.

Plan §RB-05:

    R(t) = ξ (E t² / ρ₀)^{1/(d+2)}

Spherical geometry uses d = 3, so R(t) ∝ t^{2/5} on this 2-d mesh
(document). The blast centre is offset from the domain, cell, and block
centres. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
RHO0 = 1.0
BLAST_ENERGY = 1.0
XI = 1.0
# Spherical Sedov index in the plan formula. 1/(d+2) = 1/5 ⇒ R ∝ t^{2/5}.
SEDOV_DIMENSION = 3
MESH_DIMENSION = 2
TIME_EXPONENT = 2.0 / 5.0
DOMAIN_LOWER = (0.0, 0.0)
DOMAIN_UPPER = (1.0, 1.0)
# Off-centre: not the origin, not the domain midpoint, not a cell corner.
X0 = 0.4
Y0 = 0.6


def self_similar_time_exponent(dimension: int = SEDOV_DIMENSION) -> float:
    """Return the documented exponent 2/(d+2). Spherical d=3 gives 2/5."""
    return 2.0 / (float(dimension) + 2.0)


def shock_radius(
    t,
    *,
    energy: float = BLAST_ENERGY,
    rho0: float = RHO0,
    xi: float = XI,
    dimension: int = SEDOV_DIMENSION,
) -> float:
    """Return R(t) = ξ (E t² / ρ₀)^{1/(d+2)}."""
    time = float(t)
    if time < 0.0:
        raise ValueError("time must be non-negative")
    scale = float(energy) * time * time / float(rho0)
    return float(xi) * scale ** (1.0 / (float(dimension) + 2.0))


def blast_center() -> tuple[float, float]:
    """Off-centre blast origin (x0, y0)."""
    return (float(X0), float(Y0))


def domain_center() -> tuple[float, float]:
    """Geometric centre of the documented 2-d box."""
    return (
        0.5 * (float(DOMAIN_LOWER[0]) + float(DOMAIN_UPPER[0])),
        0.5 * (float(DOMAIN_LOWER[1]) + float(DOMAIN_UPPER[1])),
    )


def polar_shock_radius(theta, t, **kwargs):
    """Circular front: R(θ) is independent of θ (self-similar sphere)."""
    radius = shock_radius(t, **kwargs)
    angles = np.asarray(theta, dtype=np.float64)
    return np.full(np.shape(angles), radius, dtype=np.float64)
