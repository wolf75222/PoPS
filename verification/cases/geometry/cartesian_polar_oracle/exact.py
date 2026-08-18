"""GE-04 shared radial Gaussian ring sampled on Cartesian and polar meshes.

φ = exp(-(r-0.5)²/σ²) with σ = 0.08. Polar samples interpolate onto the
Cartesian cell centres (nearest or bilinear). Does not import pops or read
a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

SIGMA = 0.08
RING_RADIUS = 0.5
DOMAIN_LOWER = (-1.0, -1.0)
DOMAIN_UPPER = (1.0, 1.0)
N_CELLS = 32
N_R = 64
N_THETA = 128
R_OUTER = math.sqrt(2.0)
INTERPOLATION_METHODS = ("nearest", "bilinear")


def radius(x, y):
    """Return r = √(x² + y²) about the origin."""
    return np.hypot(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def phi_of_r(r, *, sigma: float = SIGMA, ring_radius: float = RING_RADIUS):
    """Return φ(r) = exp(-(r-r0)²/σ²)."""
    delta = np.asarray(r, dtype=np.float64) - float(ring_radius)
    width = float(sigma)
    return np.exp(-(delta * delta) / (width * width))


def phi(x, y, *, sigma: float = SIGMA, ring_radius: float = RING_RADIUS):
    """Return φ at Cartesian samples (x, y)."""
    return phi_of_r(radius(x, y), sigma=sigma, ring_radius=ring_radius)


def cartesian_centers(n_cells: int = N_CELLS):
    """Uniform cell-centre mesh on the box [lower, upper]²."""
    count = int(n_cells)
    lower_x, lower_y = (float(value) for value in DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / float(count)
    width_y = (upper_y - lower_y) / float(count)
    x_centers = lower_x + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x, y, width_x, width_y


def polar_centers(n_r: int = N_R, n_theta: int = N_THETA, r_outer: float = R_OUTER):
    """Uniform cell-centre mesh on the disk r ∈ [0, r_outer], θ ∈ [0, 2π)."""
    n_radial = int(n_r)
    n_angular = int(n_theta)
    outer = float(r_outer)
    width_r = outer / float(n_radial)
    width_theta = 2.0 * math.pi / float(n_angular)
    r_centers = (np.arange(n_radial, dtype=np.float64) + 0.5) * width_r
    theta_centers = (np.arange(n_angular, dtype=np.float64) + 0.5) * width_theta
    radial, angular = np.meshgrid(r_centers, theta_centers, indexing="ij")
    return radial, angular, width_r, width_theta, r_centers, theta_centers


def sample_cartesian(n_cells: int = N_CELLS):
    """Sample φ on the Cartesian cell-centre mesh."""
    x, y, width_x, width_y = cartesian_centers(n_cells)
    field = np.asarray(phi(x, y), dtype=np.float64)
    volumes = np.full(x.shape, width_x * width_y, dtype=np.float64)
    return {
        "x": x,
        "y": y,
        "r": radius(x, y),
        "phi": field,
        "volumes": volumes,
        "dx": width_x,
        "dy": width_y,
    }


def sample_polar(n_r: int = N_R, n_theta: int = N_THETA, r_outer: float = R_OUTER):
    """Sample φ on the polar cell-centre mesh."""
    radial, angular, width_r, width_theta, r_centers, theta_centers = polar_centers(
        n_r, n_theta, r_outer
    )
    field = np.asarray(phi_of_r(radial), dtype=np.float64)
    return {
        "r": radial,
        "theta": angular,
        "phi": field,
        "dr": width_r,
        "dtheta": width_theta,
        "r_centers": r_centers,
        "theta_centers": theta_centers,
    }


def peak_radius_cartesian(n_cells: int = N_CELLS) -> float:
    """Return the radius of the Cartesian cell with the largest φ."""
    sampled = sample_cartesian(n_cells)
    index = int(np.argmax(sampled["phi"]))
    return float(np.ravel(sampled["r"])[index])


def peak_radius_polar(
    n_r: int = N_R, n_theta: int = N_THETA, r_outer: float = R_OUTER
) -> float:
    """Return the radius of the polar cell with the largest φ."""
    sampled = sample_polar(n_r, n_theta, r_outer)
    index = int(np.argmax(sampled["phi"]))
    return float(np.ravel(sampled["r"])[index])


def interpolate_polar_to_cartesian(
    polar_phi,
    r_centers,
    theta_centers,
    x,
    y,
    *,
    method: str = "bilinear",
):
    """Interpolate polar samples of φ onto Cartesian (x, y). θ is periodic."""
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"method must be one of {INTERPOLATION_METHODS}")
    field = np.asarray(polar_phi, dtype=np.float64)
    radial = np.asarray(r_centers, dtype=np.float64)
    angular = np.asarray(theta_centers, dtype=np.float64)
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    if field.ndim != 2 or field.shape != (radial.size, angular.size):
        raise ValueError("polar_phi must have shape (n_r, n_theta)")
    query_r = radius(samples_x, samples_y)
    query_theta = np.mod(np.arctan2(samples_y, samples_x), 2.0 * math.pi)
    if method == "nearest":
        return _nearest_polar(field, radial, angular, query_r, query_theta)
    return _bilinear_polar(field, radial, angular, query_r, query_theta)


def _nearest_polar(field, radial, angular, query_r, query_theta):
    width_theta = float(angular[1] - angular[0]) if angular.size > 1 else 2.0 * math.pi
    r_index = np.argmin(np.abs(radial[:, None] - query_r.ravel()[None, :]), axis=0)
    theta_index = np.argmin(
        _periodic_theta_distance(angular[:, None], query_theta.ravel()[None, :], width_theta),
        axis=0,
    )
    return field[r_index, theta_index].reshape(query_r.shape)


def _bilinear_polar(field, radial, angular, query_r, query_theta):
    n_radial = int(radial.size)
    n_angular = int(angular.size)
    width_r = float(radial[1] - radial[0]) if n_radial > 1 else 1.0
    width_theta = float(angular[1] - angular[0]) if n_angular > 1 else 2.0 * math.pi
    r_frac = (query_r - float(radial[0])) / width_r
    r_frac = np.clip(r_frac, 0.0, float(max(n_radial - 1, 1)))
    i0 = np.floor(r_frac).astype(np.int64)
    if n_radial == 1:
        i0 = np.zeros_like(i0)
        i1 = i0
        wr = np.zeros_like(query_r)
    else:
        i0 = np.clip(i0, 0, n_radial - 2)
        i1 = i0 + 1
        wr = r_frac - i0.astype(np.float64)
    theta_frac = (query_theta - float(angular[0])) / width_theta
    j0 = np.floor(theta_frac).astype(np.int64)
    wtheta = theta_frac - j0.astype(np.float64)
    j0 = np.mod(j0, n_angular)
    j1 = np.mod(j0 + 1, n_angular)
    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    return (
        (1.0 - wr) * (1.0 - wtheta) * v00
        + wr * (1.0 - wtheta) * v10
        + (1.0 - wr) * wtheta * v01
        + wr * wtheta * v11
    )


def _periodic_theta_distance(left, right, period):
    delta = np.abs(left - right)
    return np.minimum(delta, float(period) - delta)
