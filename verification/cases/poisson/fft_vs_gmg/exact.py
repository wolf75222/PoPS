"""PO-05 FFT vs GMG cross-oracle. Reuses the PO-01 trigonometric Poisson.

Does not import pops or read a PoPS output.

Spectral route: 1-d periodic FFT Poisson for -Δφ = ρ, i.e. k² φ̂ = ρ̂ with
the k = 0 mode set to 0 (mean-zero gauge). GMG stub: discrete -Δ residual
of a candidate φ against the manufactured ρ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_PO01_EXACT = (
    Path(__file__).resolve().parents[1] / "periodic_trig" / "exact.py"
)
_po01 = load_sibling_module(_PO01_EXACT)

phi_exact = _po01.phi_exact
rhs_exact = _po01.rhs_exact
e_exact = _po01.e_exact
uniform_cell_grid = _po01.uniform_cell_grid
TWO_PI = _po01.TWO_PI
N_CELLS = _po01.N_CELLS
X_LO = _po01.X_LO
X_HI = _po01.X_HI
PERIOD = X_HI - X_LO


def mean_free(values, volumes=None) -> np.ndarray:
    """Return values minus the (optionally volume-weighted) mean."""
    field = np.asarray(values, dtype=np.float64)
    if volumes is None:
        mean = float(np.mean(field))
    else:
        mean = float(np.average(field, weights=np.asarray(volumes, dtype=np.float64)))
    return field - mean


def spectral_solve(rhs) -> np.ndarray:
    """Invert 1-d periodic Poisson in Fourier space: k² φ̂ = ρ̂, k = 0 → 0.

    ``k`` is the continuous wave number 2π ξ / L. The zero mode is dropped so
    the returned potential is mean-zero. This is the spectral (not discrete-
    stencil) Poisson solve named ``(-k² φ̂ = ρ̂, k=0 → 0)`` in the case brief:
    the Fourier symbol of -Δ is +k², so -Δφ = ρ becomes k² φ̂ = ρ̂.
    """
    rho = np.asarray(rhs, dtype=np.float64)
    count = int(rho.size)
    spacing = PERIOD / float(count)
    wave = TWO_PI * np.fft.fftfreq(count, d=spacing)
    rho_hat = np.fft.fft(rho)
    phi_hat = np.zeros_like(rho_hat)
    wave_sq = np.square(wave)
    np.divide(rho_hat, wave_sq, out=phi_hat, where=wave_sq != 0.0)
    return np.fft.ifft(phi_hat).real


def gmg_stub_residual(phi, rhs) -> np.ndarray:
    """Return the discrete -Δ residual of ``phi`` against ``rhs``.

    Uses the second-order periodic 1-d stencil that GeometricMG inverts.
    The residual of the analytic potential versus manufactured ρ is the
    finite-difference truncation; it is reported, not used as a gate.
    """
    potential = np.asarray(phi, dtype=np.float64)
    density = np.asarray(rhs, dtype=np.float64)
    count = int(potential.size)
    spacing = PERIOD / float(count)
    minus_lap = (
        2.0 * potential - np.roll(potential, 1) - np.roll(potential, -1)
    ) / (spacing * spacing)
    return minus_lap - density
