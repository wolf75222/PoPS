"""Linear diocotron stand-in: hollow ring plus m=2 toy growth.

Cartesian unit square. Background ring

    n(r) = n0    for r1 < r < r2
         = 0     otherwise

plus the documented azimuthal perturbation

    ε Re(exp(i m θ) exp(γ t)) = ε exp(γ t) cos(m θ)

with integer mode m = 2 and toy growth rate γ = 0.1. This is an
in-memory oracle, not a paper reproduction of a published growth rate.

Does not import the pops package or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

N0 = 1.0
R1 = 0.15
R2 = 0.35
X0 = 0.5
Y0 = 0.5
EPS = 1.0e-4
GROWTH_RATE = 0.1
M = 2
N_CELLS = 32
PERIOD = 1.0


def _as_samples(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def radius(x, y):
    """Radial distance from the ring centre (X0, Y0)."""
    xx, yy = np.broadcast_arrays(_as_samples(x), _as_samples(y))
    return np.hypot(xx - X0, yy - Y0)


def azimuth(x, y):
    """Polar angle θ = atan2(y − Y0, x − X0)."""
    xx, yy = np.broadcast_arrays(_as_samples(x), _as_samples(y))
    return np.arctan2(yy - Y0, xx - X0)


def ring_density(r):
    """Unperturbed hollow ring n(r) = n0 on r1 < r < r2, else 0."""
    radius_samples = _as_samples(r)
    return np.where((radius_samples > R1) & (radius_samples < R2), N0, 0.0)


def amplitude(t) -> float:
    """Mode amplitude ε exp(γ t) of the documented toy growth."""
    return float(EPS) * math.exp(float(GROWTH_RATE) * float(t))


def perturbation(theta, t, *, eps=EPS, growth_rate=GROWTH_RATE, m=M):
    """ε Re(exp(i m θ) exp(γ t)) = ε exp(γ t) cos(m θ)."""
    angle = _as_samples(theta)
    return float(eps) * math.exp(float(growth_rate) * float(t)) * np.cos(float(m) * angle)


def polar_density(r, theta, t, *, eps=EPS, growth_rate=GROWTH_RATE, m=M):
    """Density on polar samples: unperturbed ring plus the m=2 mode."""
    radius_samples, angle = np.broadcast_arrays(_as_samples(r), _as_samples(theta))
    return ring_density(radius_samples) + perturbation(
        angle, t, eps=eps, growth_rate=growth_rate, m=m
    )


def density(x, y, t, *, eps=EPS, growth_rate=GROWTH_RATE, m=M):
    """Cartesian samples of the ring plus the growing m=2 perturbation."""
    xx, yy = np.broadcast_arrays(_as_samples(x), _as_samples(y))
    return polar_density(
        radius(xx, yy), azimuth(xx, yy), t, eps=eps, growth_rate=growth_rate, m=m
    )


def uniform_cell_mesh(n_cells: int = N_CELLS):
    """Uniform cell centers and volumes on the periodic unit square."""
    count = int(n_cells)
    width = float(PERIOD) / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    volumes = np.full((count, count), width * width, dtype=np.float64)
    return x, y, volumes


def angular_spectrum(r, t, n_theta: int = 64, *, eps=EPS):
    """|rFFT| of the perturbation sampled on the circle of radius r."""
    count = int(n_theta)
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    samples = polar_density(r, theta, t, eps=eps)
    perturbation_samples = np.asarray(samples, dtype=np.float64) - float(ring_density(r))
    return np.abs(np.fft.rfft(perturbation_samples))
