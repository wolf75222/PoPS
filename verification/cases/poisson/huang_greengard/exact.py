"""1-d periodic Huang–Greengard multi-blob Poisson oracle.

Does not import pops or read a PoPS output.

Stand-in of the multi-Gaussian Poisson test on a unit period:

    φ(x) = Σ_i A_i exp(-((x-c_i)/σ)²)

with centres (0.25, 0.5, 0.75), A_i = 1, σ = 0.04. Displacement uses the
nearest periodic image. ρ = -φ'' is the analytic second derivative of each
blob (images beyond the nearest are negligible at this σ).
"""
from __future__ import annotations

import numpy as np

N_CELLS = 32
X_LO = 0.0
X_HI = 1.0
PERIOD = X_HI - X_LO
SIGMA = 0.04
CENTRES = (0.25, 0.5, 0.75)
AMPLITUDES = (1.0, 1.0, 1.0)


def uniform_cell_grid(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return cell centers and widths for a uniform 1-d partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def nearest_image_displacement(x, centre, period: float = PERIOD) -> np.ndarray:
    """Periodic displacement from ``centre`` to ``x`` on a circle of length ``period``."""
    displacement = np.asarray(x, dtype=np.float64) - float(centre)
    width = float(period)
    return displacement - width * np.round(displacement / width)


def _blob_phi(x, amplitude, centre, sigma: float = SIGMA, period: float = PERIOD):
    radius = nearest_image_displacement(x, centre, period)
    return float(amplitude) * np.exp(-np.square(radius / float(sigma)))


def phi_exact(x) -> np.ndarray:
    """Pointwise φ(x) = Σ_i A_i exp(-((x-c_i)/σ)²) with nearest-image wrap."""
    samples = np.asarray(x, dtype=np.float64)
    total = np.zeros(samples.shape, dtype=np.float64)
    for amplitude, centre in zip(AMPLITUDES, CENTRES, strict=True):
        total = total + _blob_phi(samples, amplitude, centre)
    return total


def dphi_exact(x) -> np.ndarray:
    """Pointwise φ' of the nearest-image multi-blob potential."""
    samples = np.asarray(x, dtype=np.float64)
    sigma = float(SIGMA)
    total = np.zeros(samples.shape, dtype=np.float64)
    for amplitude, centre in zip(AMPLITUDES, CENTRES, strict=True):
        radius = nearest_image_displacement(samples, centre)
        scaled = radius / sigma
        total = total - (2.0 * float(amplitude) / sigma) * scaled * np.exp(
            -np.square(scaled)
        )
    return total


def d2phi_exact(x) -> np.ndarray:
    """Pointwise φ'' of the nearest-image multi-blob potential."""
    samples = np.asarray(x, dtype=np.float64)
    sigma = float(SIGMA)
    total = np.zeros(samples.shape, dtype=np.float64)
    for amplitude, centre in zip(AMPLITUDES, CENTRES, strict=True):
        radius = nearest_image_displacement(samples, centre)
        scaled = radius / sigma
        total = total - (2.0 * float(amplitude) / sigma**2) * (
            1.0 - 2.0 * np.square(scaled)
        ) * np.exp(-np.square(scaled))
    return total


def rhs_exact(x) -> np.ndarray:
    """Pointwise ρ = -φ'' of the nearest-image multi-blob potential."""
    return -d2phi_exact(x)


def e_exact(x) -> np.ndarray:
    """Pointwise E = -dφ/dx of the nearest-image multi-blob potential."""
    return -dphi_exact(x)
