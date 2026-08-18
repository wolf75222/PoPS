"""GE-03 small-amplitude radial acoustic wave on a 2-d Cartesian mesh.

Velocity potential ψ = φ = ε J0(k r) cos(ω t) with ω = c k, c = 1, ε = 1e-3.
Primitives follow linear acoustics: u = ∇ψ, p' = -ρ0 ∂ψ/∂t, ρ' = p'/c².
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
C = 1.0
EPS = 1.0e-3
K = 1.0
RHO0 = 1.0
P0 = RHO0 * C * C / GAMMA
DOMAIN_LOWER = (-1.0, -1.0)
DOMAIN_UPPER = (1.0, 1.0)
N_CELLS = 32
N_THETA = 64
RING_RADIUS = 0.5
_BESSEL_TERMS = 40


def omega(k: float = K, c: float = C) -> float:
    """Angular frequency of the standing wave: ω = c k."""
    return float(c) * float(k)


def radius(x, y):
    """Return r = √(x² + y²) about the origin."""
    return np.hypot(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def _bessel_j0(z):
    """J0 via the power series. Accurate for |z| on this domain."""
    samples = np.asarray(z, dtype=np.float64)
    half_sq = 0.25 * samples * samples
    term = np.ones_like(samples)
    acc = np.ones_like(samples)
    for index in range(1, _BESSEL_TERMS):
        term = -term * half_sq / float(index * index)
        acc = acc + term
    return acc


def _bessel_j1(z):
    """J1 via the power series. Accurate for |z| on this domain."""
    samples = np.asarray(z, dtype=np.float64)
    half = 0.5 * samples
    half_sq = half * half
    term = np.array(half, copy=True)
    acc = np.array(half, copy=True)
    for index in range(1, _BESSEL_TERMS):
        term = -term * half_sq / float(index * (index + 1))
        acc = acc + term
    return acc


def phi(x, y, t, *, eps: float = EPS, k: float = K, c: float = C):
    """Return φ = ε J0(k r) cos(ω t)."""
    radial = radius(x, y)
    return (
        float(eps)
        * _bessel_j0(float(k) * radial)
        * math.cos(omega(k, c) * float(t))
    )


def primitives(x, y, t, *, eps: float = EPS, k: float = K, c: float = C):
    """Linear-acoustic primitives (ρ, u, v, p) consistent with u = ∇ψ, ψ = φ."""
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    wave_number = float(k)
    speed = float(c)
    amplitude = float(eps)
    time = float(t)
    ang = omega(wave_number, speed)
    radial = radius(samples_x, samples_y)
    argument = wave_number * radial
    phase = math.cos(ang * time)
    bessel = _bessel_j0(argument)
    potential = amplitude * bessel * phase
    # p' = -ρ0 ∂ψ/∂t and ∂ψ/∂t = -ε ω J0(kr) sin(ω t).
    pressure = P0 + RHO0 * amplitude * ang * bessel * math.sin(ang * time)
    density = RHO0 + (pressure - P0) / (speed * speed)
    # u = ∇ψ = ε k J0'(kr) cos(ω t) ê_r and J0' = -J1.
    radial_speed = -amplitude * wave_number * _bessel_j1(argument) * phase
    safe = np.where(radial > 0.0, radial, 1.0)
    velocity_x = np.where(radial > 0.0, radial_speed * samples_x / safe, 0.0)
    velocity_y = np.where(radial > 0.0, radial_speed * samples_y / safe, 0.0)
    return {
        "rho": density,
        "u": velocity_x,
        "v": velocity_y,
        "p": pressure,
        "phi": potential,
    }


def ring_coordinates(ring_radius: float = RING_RADIUS, n_theta: int = N_THETA):
    """Return (x, y) samples on the circle r = ring_radius."""
    theta = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    scale = float(ring_radius)
    return scale * np.cos(theta), scale * np.sin(theta)


def angular_std(ring_radius: float = RING_RADIUS, t=0.0, n_theta: int = N_THETA) -> float:
    """Return the angular standard deviation of φ on a circle of fixed r."""
    x, y = ring_coordinates(ring_radius, n_theta)
    return float(np.std(phi(x, y, t)))
