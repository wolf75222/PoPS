"""GE-01 polar manufactured Poisson: harmonic φ = r^m cos(mθ) on an annulus.

Oracle on r ∈ [0.2, 1], m = 2. Polar Laplacian Δφ = 0. Cartesian equivalent
is Re((x + i y)^m). The origin is excluded from the annulus.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

M = 2
R_MIN = 0.2
R_MAX = 1.0
N_R = 32
N_THETA = 64


def phi(r, theta, *, m: int = M):
    """Return φ(r, θ) = r^m cos(m θ)."""
    radius = np.asarray(r, dtype=np.float64)
    angle = np.asarray(theta, dtype=np.float64)
    mode = float(m)
    return radius**mode * np.cos(mode * angle)


def polar_laplacian(r, theta, *, m: int = M):
    """Return the polar Laplacian Δφ = φ_rr + (1/r) φ_r + (1/r²) φ_θθ.

    For the harmonic φ = r^m cos(m θ) this is identically zero on r > 0.
    The origin is excluded: the polar operator is not evaluated at r = 0.
    """
    radius = np.asarray(r, dtype=np.float64)
    angle = np.asarray(theta, dtype=np.float64)
    if np.any(radius <= 0.0):
        raise ValueError("polar Laplacian requires r > 0 (annulus excludes the origin)")
    mode = float(m)
    phase = np.cos(mode * angle)
    d2_dr2 = mode * (mode - 1.0) * radius ** (mode - 2.0) * phase
    d_dr = mode * radius ** (mode - 1.0) * phase
    d2_dtheta2 = -(mode**2) * radius**mode * phase
    return d2_dr2 + d_dr / radius + d2_dtheta2 / (radius * radius)


def cartesian_equivalent(x, y, *, m: int = M):
    """Return φ(x, y) = Re((x + i y)^m), equal to r^m cos(m θ)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.real((xx + 1.0j * yy) ** int(m))


def in_annulus(r) -> bool:
    """Return True iff every sample lies in the closed annulus [R_MIN, R_MAX]."""
    samples = np.asarray(r, dtype=np.float64)
    return bool(np.all((samples >= R_MIN) & (samples <= R_MAX)))


def polar_cell_grid(n_r: int = N_R, n_theta: int = N_THETA):
    """Uniform polar cell centers and volumes r Δr Δθ on the annulus."""
    n_radial = int(n_r)
    n_azimuth = int(n_theta)
    dr = (R_MAX - R_MIN) / float(n_radial)
    dtheta = 2.0 * math.pi / float(n_azimuth)
    radii = R_MIN + (np.arange(n_radial, dtype=np.float64) + 0.5) * dr
    thetas = (np.arange(n_azimuth, dtype=np.float64) + 0.5) * dtheta
    radius, angle = np.meshgrid(radii, thetas, indexing="ij")
    volumes = radius * dr * dtheta
    return radius, angle, volumes
