"""1-d isothermal Euler–Poisson pressure–field equilibrium.

Hydrostatic Boltzmann electrons (dimensionless e = 1):

    φ = -(T / q) ln n + C
    E = -dφ/dx
    u = 0
    p = T n

Profiles (δ < 1 so n stays positive):

    n = n0 (1 + δ cos(2πx))
    n = n0 exp(-x² / (2σ²))

Force balance on the exact fields is ∇p = q n E. Does not import pops or
read a PoPS output.
"""
from __future__ import annotations

import numpy as np

N0 = 1.0
T = 1.0
Q = -1.0
DELTA = 0.1
SIGMA = 0.2
C = 0.0
N_CELLS = 64
TWO_PI = 2.0 * np.pi


def uniform_cell_centers(n_cells: int = N_CELLS, x_lo: float = 0.0, x_hi: float = 1.0):
    """Return cell centers and widths on a uniform partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def n_cosine(x, n0: float = N0, delta: float = DELTA) -> np.ndarray:
    """Periodic n = n0 (1 + δ cos(2πx)), δ < 1."""
    if float(delta) >= 1.0:
        raise ValueError("delta must be < 1 so n stays positive")
    xx = np.asarray(x, dtype=np.float64)
    return float(n0) * (1.0 + float(delta) * np.cos(TWO_PI * xx))


def n_gaussian(x, n0: float = N0, sigma: float = SIGMA) -> np.ndarray:
    """Localized n = n0 exp(-x² / (2σ²))."""
    xx = np.asarray(x, dtype=np.float64)
    width = float(sigma)
    return float(n0) * np.exp(-(xx * xx) / (2.0 * width * width))


def u_exact(x) -> np.ndarray:
    """Rest velocity u = 0."""
    return np.zeros_like(np.asarray(x, dtype=np.float64))


def p_exact(n, temperature: float = T) -> np.ndarray:
    """Isothermal p = T n."""
    return float(temperature) * np.asarray(n, dtype=np.float64)


def phi_exact(n, temperature: float = T, charge: float = Q, offset: float = C) -> np.ndarray:
    """Boltzmann φ = -(T / q) ln n + C."""
    density = np.asarray(n, dtype=np.float64)
    return -(float(temperature) / float(charge)) * np.log(density) + float(offset)


def e_from_phi(x, phi) -> np.ndarray:
    """E = -dφ/dx from a periodic spectral derivative on a uniform grid."""
    xx = np.asarray(x, dtype=np.float64)
    potential = np.asarray(phi, dtype=np.float64)
    count = int(xx.size)
    if count < 2:
        return np.zeros_like(potential)
    spacing = float(xx[1] - xx[0])
    wave = TWO_PI * np.fft.fftfreq(count, d=spacing)
    dphi = np.fft.ifft(1j * wave * np.fft.fft(potential)).real
    return -dphi


def _cosine_dn_dx(x, n0: float, delta: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    return float(n0) * (-float(delta) * TWO_PI * np.sin(TWO_PI * xx))


def _gaussian_dn_dx(x, n, sigma: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    return np.asarray(n, dtype=np.float64) * (-xx / (float(sigma) ** 2))


def exact_fields(
    x,
    profile: str = "cosine",
    *,
    n0: float = N0,
    delta: float = DELTA,
    sigma: float = SIGMA,
    temperature: float = T,
    charge: float = Q,
    offset: float = C,
) -> dict:
    """Return n, u=0, p, φ, E, ∇p for one of the documented 1-d profiles."""
    xx = np.asarray(x, dtype=np.float64)
    if profile == "cosine":
        density = n_cosine(xx, n0=n0, delta=delta)
        dn_dx = _cosine_dn_dx(xx, n0, delta)
    elif profile == "gaussian":
        density = n_gaussian(xx, n0=n0, sigma=sigma)
        dn_dx = _gaussian_dn_dx(xx, density, sigma)
    else:
        raise ValueError(f"unknown profile {profile!r}")
    potential = phi_exact(density, temperature=temperature, charge=charge, offset=offset)
    electric = (float(temperature) / float(charge)) * (dn_dx / density)
    return {
        "n": density,
        "u": u_exact(xx),
        "p": p_exact(density, temperature=temperature),
        "phi": potential,
        "E": electric,
        "grad_p": float(temperature) * dn_dx,
        "q": float(charge),
        "T": float(temperature),
        "C": float(offset),
    }
