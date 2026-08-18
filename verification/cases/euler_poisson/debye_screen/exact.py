"""1-d periodic linearized Debye (Helmholtz) screen oracle.

    (-d^{2}/dx^{2} + λ_D^{-2}) φ = f,   f = cos(2π k x),

so the cosine eigenfunction gives

    φ = f / ((2πk)^{2} + λ_D^{-2}).

Defaults: λ_D = 0.1, k = 1. As λ_D → ∞ the screening vanishes and
φ → f / (2πk)^{2} (periodic Poisson). Does not import pops or read a
PoPS output.
"""
from __future__ import annotations

import numpy as np

LAMBDA_D = 0.1
K = 1
TWO_PI = 2.0 * np.pi
N_CELLS = 32
X_LO = 0.0
X_HI = 1.0


def uniform_cell_grid(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return cell centers and widths for a uniform 1-d partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def wave_number(k: float = K) -> float:
    """Fourier wavenumber 2πk of the manufactured cosine."""
    return TWO_PI * float(k)


def screening_coefficient(lambda_d: float = LAMBDA_D) -> float:
    """Zeroth-order Helmholtz coefficient λ_D^{-2}. Infinity screens to 0."""
    if np.isinf(lambda_d):
        return 0.0
    length = float(lambda_d)
    if length <= 0.0:
        raise ValueError("lambda_d must be positive or +inf")
    return 1.0 / (length * length)


def helmholtz_denominator(lambda_d: float = LAMBDA_D, k: float = K) -> float:
    """(2πk)^{2} + λ_D^{-2}."""
    return wave_number(k) ** 2 + screening_coefficient(lambda_d)


def poisson_gain(k: float = K) -> float:
    """Unscreened Poisson amplitude 1/(2πk)^{2}."""
    return 1.0 / (wave_number(k) ** 2)


def helmholtz_gain(lambda_d: float = LAMBDA_D, k: float = K) -> float:
    """Screened amplitude 1 / ((2πk)^{2} + λ_D^{-2})."""
    return 1.0 / helmholtz_denominator(lambda_d, k)


def f_exact(x, k: float = K) -> np.ndarray:
    """Pointwise Helmholtz right-hand side f = cos(2π k x)."""
    samples = np.asarray(x, dtype=np.float64)
    return np.cos(wave_number(k) * samples)


def phi_exact(x, lambda_d: float = LAMBDA_D, k: float = K) -> np.ndarray:
    """Analytic potential φ = f / ((2πk)^{2} + λ_D^{-2})."""
    return f_exact(x, k=k) * helmholtz_gain(lambda_d, k)


def e_exact(x, lambda_d: float = LAMBDA_D, k: float = K) -> np.ndarray:
    """Pointwise E = -dφ/dx for the cosine potential."""
    samples = np.asarray(x, dtype=np.float64)
    kappa = wave_number(k)
    return kappa * np.sin(kappa * samples) * helmholtz_gain(lambda_d, k)


def apply_helmholtz(phi, x, lambda_d: float = LAMBDA_D) -> np.ndarray:
    """Spectral (−d^{2}/dx^{2} + λ_D^{-2}) φ on a uniform periodic grid."""
    samples = np.asarray(x, dtype=np.float64)
    field = np.asarray(phi, dtype=np.float64)
    count = int(samples.size)
    if count < 2:
        return screening_coefficient(lambda_d) * field
    spacing = float(samples[1] - samples[0])
    wave = TWO_PI * np.fft.fftfreq(count, d=spacing)
    minus_d2 = np.fft.ifft((wave**2) * np.fft.fft(field)).real
    return minus_d2 + screening_coefficient(lambda_d) * field
