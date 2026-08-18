"""GE-06 Cartesian hollow-ring density and AMR tagging envelope.

Duplicates the CP-11 unperturbed ring when
``verification/cases/euler_poisson/diocotron/exact.py`` is absent:
n(r) = n0 for r1 < r < r2 else n_bg. Tag |n - n_bg| > θ. Two-level
envelope = tagged cells plus Chebyshev buffer 2.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

N0 = 1.0
N_BG = 0.0
R1 = 0.20
R2 = 0.35
EPS = 1.0e-3
GAMMA = 0.1
MODE = 2
UNUSED_MODE = 3
THETA = 0.5
BUFFER_CELLS = 2
N_CELLS = 32
N_THETA = 64
DOMAIN_LOWER = (-0.5, -0.5)
DOMAIN_UPPER = (0.5, 0.5)


def radius(x, y):
    """Return r = √(x² + y²) about the origin."""
    return np.hypot(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def theta(x, y):
    """Return θ = atan2(y, x)."""
    return np.arctan2(
        np.asarray(y, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
    )


def ring_mask(x, y, *, r1: float = R1, r2: float = R2) -> np.ndarray:
    """True on the open annulus r1 < r < r2."""
    radial = radius(x, y)
    return (radial > float(r1)) & (radial < float(r2))


def unperturbed_density(x, y, *, n0: float = N0, n_bg: float = N_BG, r1: float = R1, r2: float = R2):
    """Hollow ring: n0 on r1 < r < r2, else n_bg. Independent of θ."""
    return np.where(ring_mask(x, y, r1=r1, r2=r2), float(n0), float(n_bg))


def density(
    x,
    y,
    t=0.0,
    *,
    n0: float = N0,
    n_bg: float = N_BG,
    r1: float = R1,
    r2: float = R2,
    eps: float = 0.0,
    mode: int = MODE,
    gamma: float = GAMMA,
):
    """Ring plus optional CP-11 azimuthal mode ε Re(e^{imθ} e^{γt}) on the ring."""
    base = unperturbed_density(x, y, n0=n0, n_bg=n_bg, r1=r1, r2=r2)
    amplitude = float(eps)
    if amplitude == 0.0:
        return base
    phase = amplitude * np.exp(float(gamma) * float(t)) * np.cos(
        float(mode) * theta(x, y)
    )
    return base + phase * ring_mask(x, y, r1=r1, r2=r2).astype(np.float64)


def raw_tag_mask(field, *, theta: float = THETA, n_bg: float = N_BG) -> np.ndarray:
    """Tag cells where |n - n_bg| > θ."""
    return np.abs(np.asarray(field, dtype=np.float64) - float(n_bg)) > float(theta)


def dilate_mask(mask, buffer_cells: int = BUFFER_CELLS) -> np.ndarray:
    """Periodic 2-d Chebyshev dilation by the given cell count."""
    selected = np.asarray(mask, dtype=bool)
    width = int(buffer_cells)
    if width < 0:
        raise ValueError("buffer_cells must be non-negative")
    dilated = selected.copy()
    for shift_i in range(-width, width + 1):
        for shift_j in range(-width, width + 1):
            if shift_i == 0 and shift_j == 0:
                continue
            dilated |= np.roll(np.roll(selected, shift_i, axis=0), shift_j, axis=1)
    return dilated


def envelope_mask(field, *, theta: float = THETA, n_bg: float = N_BG, buffer_cells: int = BUFFER_CELLS):
    """Two-level envelope: raw tags plus a Chebyshev buffer."""
    return dilate_mask(raw_tag_mask(field, theta=theta, n_bg=n_bg), buffer_cells)


def ring_coordinates(ring_radius=None, n_theta: int = N_THETA):
    """Return (x, y, θ) samples on the circle r = (r1+r2)/2 by default."""
    radius_value = 0.5 * (R1 + R2) if ring_radius is None else float(ring_radius)
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    return radius_value * np.cos(angles), radius_value * np.sin(angles), angles


def angular_density(
    ring_radius=None,
    n_theta: int = N_THETA,
    t=0.0,
    *,
    eps: float = 0.0,
    mode: int = MODE,
):
    """Unperturbed (default) ring samples on a circle of fixed r."""
    x, y, angles = ring_coordinates(ring_radius, n_theta)
    return angles, density(x, y, t, eps=eps, mode=mode)


def angular_fft(samples) -> np.ndarray:
    """Normalized rFFT so bin 0 is the mean and bin m is mode m."""
    field = np.asarray(samples, dtype=np.float64)
    return np.fft.rfft(field) / float(field.size)


def unused_mode_amplitude(
    ring_radius=None,
    n_theta: int = N_THETA,
    *,
    unused_mode: int = UNUSED_MODE,
    t=0.0,
    eps: float = 0.0,
) -> float:
    """Return |FFT[m]| of the (unperturbed) ring samples for unused mode m=3."""
    _, samples = angular_density(ring_radius, n_theta, t, eps=eps)
    spectrum = angular_fft(samples)
    return float(np.abs(spectrum[int(unused_mode)]))
