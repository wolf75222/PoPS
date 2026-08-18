"""GE-02 solid-body scalar rotation. Polar velocity evaluated in Cartesian.

v_θ = Ω r, Ω = 2π so T = 1. A Gaussian bump starts at (0.5, 0) on the ring
r = 0.5. Cartesian samples use x = r cos θ, y = r sin θ. Does not import
pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

OMEGA = 2.0 * math.pi
PERIOD = 2.0 * math.pi / OMEGA
RING_RADIUS = 0.5
THETA0 = 0.0
X0 = RING_RADIUS * math.cos(THETA0)
Y0 = RING_RADIUS * math.sin(THETA0)
SIGMA = 0.08
AMP = 1.0
DOMAIN_LOWER = (-1.0, -1.0)
DOMAIN_UPPER = (1.0, 1.0)
N_CELLS = 32


def cartesian_from_polar(r, theta):
    """Return (x, y) = (r cos θ, r sin θ)."""
    radius = np.asarray(r, dtype=np.float64)
    angle = np.asarray(theta, dtype=np.float64)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    if np.ndim(radius) == 0 and np.ndim(angle) == 0:
        return float(np.asarray(x)), float(np.asarray(y))
    return x, y


def velocity_polar(r):
    """Return (v_r, v_θ) with v_r = 0 and v_θ = Ω r."""
    radius = np.asarray(r, dtype=np.float64)
    azimuthal = OMEGA * radius
    radial = np.zeros_like(radius)
    if np.ndim(radius) == 0:
        return float(radial), float(np.asarray(azimuthal))
    return radial, azimuthal


def velocity(x, y):
    """Return Cartesian (u, v) = (−Ω y, Ω x) of the solid-body field."""
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    u = -OMEGA * samples_y
    v = OMEGA * samples_x
    if np.ndim(samples_x) == 0 and np.ndim(samples_y) == 0:
        return float(np.asarray(u)), float(np.asarray(v))
    return u, v


def _backward_trace(x, y, t, *, omega: float = OMEGA):
    """Map (x, y, t) to the foot of the solid-body characteristic."""
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    samples_x, samples_y = np.broadcast_arrays(samples_x, samples_y)
    angle = float(omega) * float(t)
    cycles = angle / (2.0 * math.pi)
    if abs(cycles - round(cycles)) <= 1.0e-12:
        return samples_x.astype(np.float64, copy=False), samples_y.astype(
            np.float64, copy=False
        )
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_init = samples_x * cos_a + samples_y * sin_a
    y_init = -samples_x * sin_a + samples_y * cos_a
    return x_init, y_init


def exact_scalar(
    x,
    y,
    t=0.0,
    *,
    x0: float = X0,
    y0: float = Y0,
    sigma: float = SIGMA,
    amp: float = AMP,
    omega: float = OMEGA,
):
    """Gaussian bump advected by solid-body rotation about the origin."""
    x_init, y_init = _backward_trace(x, y, t, omega=omega)
    dx = x_init - float(x0)
    dy = y_init - float(y0)
    field = float(amp) * np.exp(
        -(dx * dx + dy * dy) / (2.0 * float(sigma) * float(sigma))
    )
    if np.ndim(field) == 0:
        return float(field)
    return field.astype(np.float64, copy=False)


def exact_return(x, y, *, x0: float = X0, y0: float = Y0, sigma: float = SIGMA):
    """Oracle at t=T: the scalar returns exactly to the IC."""
    return exact_scalar(x, y, 0.0, x0=x0, y0=y0, sigma=sigma)


def peak_location(t, *, ring_radius: float = RING_RADIUS, theta0: float = THETA0):
    """Return the Cartesian peak of the bump after rotation by Ω t."""
    return cartesian_from_polar(ring_radius, float(theta0) + OMEGA * float(t))


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the box [lower, upper]²."""
    count = int(n_cells)
    lower_x, lower_y = (float(value) for value in DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / count
    width_y = (upper_y - lower_y) / count
    if width_x != width_y:
        raise ValueError("GE-02 mesh must be square")
    x_centers = lower_x + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x, y, width_x


def cell_volumes(n_cells: int = N_CELLS) -> np.ndarray:
    """Return uniform cell volumes on the square box."""
    count = int(n_cells)
    _, _, width = cell_centers(count)
    return np.full((count, count), width * width, dtype=np.float64)
