"""GE-05 polar cell volumes. Midpoint formula r Δr Δθ is exact.

The annulus [r_in, r_out] × [0, 2π) is sampled on a uniform (r, θ) mesh.
A constant state 1 integrates to the analytic area π(r_out² − r_in²).
The axis cell (r=0) is refused by polar_cell_volume and regularized by
axis_cell_volume as ½ (Δr)² Δθ, without dividing by r=0.

Does not load the pops package or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

R_IN = 0.2
R_OUT = 1.0
N_R = 8
N_THETA = 16


def _positive_spacing(value, *, name: str) -> float:
    spacing = float(value)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return spacing


def polar_cell_volume(r, dr, dtheta) -> float:
    """Return the polar cell area r Δr Δθ. Cell-center r must be strictly positive."""
    radius = float(r)
    width = _positive_spacing(dr, name="Δr")
    angle = _positive_spacing(dtheta, name="Δθ")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("polar_cell_volume refuses r=0 (axis cell); use axis_cell_volume")
    return radius * width * angle


def axis_cell_volume(dr, dtheta) -> float:
    """Documented regular volume of the r=0 axis cell: ½ (Δr)² Δθ.

    This is the exact sector area of [0, Δr] × [θ, θ+Δθ]. It is also the
    midpoint formula with r_center = Δr/2. The helper never divides by r=0.
    """
    width = _positive_spacing(dr, name="Δr")
    angle = _positive_spacing(dtheta, name="Δθ")
    return 0.5 * width * width * angle


def annulus_area(r_in: float = R_IN, r_out: float = R_OUT) -> float:
    """Return the analytic annulus area π(r_out² − r_in²)."""
    inner = float(r_in)
    outer = float(r_out)
    if not (math.isfinite(inner) and math.isfinite(outer)) or not (outer > inner >= 0.0):
        raise ValueError("annulus requires finite radii with 0 <= r_in < r_out")
    return math.pi * (outer * outer - inner * inner)


def _radial_centers(r_in: float, r_out: float, n_r: int):
    count = int(n_r)
    if count < 1:
        raise ValueError("n_r must be at least 1")
    inner = float(r_in)
    outer = float(r_out)
    if not (math.isfinite(inner) and math.isfinite(outer)) or not (outer > inner > 0.0):
        raise ValueError("annulus mesh requires finite radii with 0 < r_in < r_out")
    width = (outer - inner) / float(count)
    return inner + (np.arange(count, dtype=np.float64) + 0.5) * width, width


def polar_cell_volumes(
    r_in: float = R_IN,
    r_out: float = R_OUT,
    n_r: int = N_R,
    n_theta: int = N_THETA,
):
    """Return cell volumes of shape (n_r, n_theta) using V = r Δr Δθ."""
    n_azimuth = int(n_theta)
    if n_azimuth < 1:
        raise ValueError("n_theta must be at least 1")
    radii, width = _radial_centers(r_in, r_out, n_r)
    angle = 2.0 * math.pi / float(n_azimuth)
    radial = np.asarray(
        [polar_cell_volume(float(radius), width, angle) for radius in radii],
        dtype=np.float64,
    )
    return np.broadcast_to(radial[:, None], (int(n_r), n_azimuth)).copy()


def annulus_volume(
    r_in: float = R_IN,
    r_out: float = R_OUT,
    n_r: int = N_R,
    n_theta: int = N_THETA,
) -> float:
    """Return the discrete annulus volume, the sum of polar cell areas."""
    return float(np.sum(polar_cell_volumes(r_in, r_out, n_r, n_theta)))


def constant_state_integral(
    value=1.0,
    r_in: float = R_IN,
    r_out: float = R_OUT,
    n_r: int = N_R,
    n_theta: int = N_THETA,
) -> float:
    """Integrate a constant state over the polar annulus (Σ value · V_ij)."""
    return float(value) * annulus_volume(r_in, r_out, n_r, n_theta)
