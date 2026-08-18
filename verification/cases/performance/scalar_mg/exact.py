"""PF-02 1-d numpy scalar-MG stand-in: PO-01 residual and damped Jacobi.

Discrete -Δ residual of φ = sin(2πx) against ρ = (2π)² φ. Each damped
Jacobi residual application is counted as one stand-in V-cycle.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

from verification.pops_verify.reference_errors import reference_errors

TWO_PI = 2.0 * np.pi
N_CELLS = 32
N_VCYCLES = 4
JACOBI_OMEGA = 2.0 / 3.0
X_LO = 0.0
X_HI = 1.0
PERIOD = X_HI - X_LO


def uniform_cell_grid(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return cell centers and widths for a uniform 1-d partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def phi_exact(x) -> np.ndarray:
    """Pointwise φ(x) = sin(2πx)."""
    return np.sin(TWO_PI * np.asarray(x, dtype=np.float64))


def rhs_exact(x) -> np.ndarray:
    """Pointwise -Δφ = (2π)² sin(2πx)."""
    return (TWO_PI**2) * phi_exact(x)


def minus_discrete_laplacian(phi, period: float = PERIOD) -> np.ndarray:
    """Return the periodic second-order 1-d stencil for -Δ."""
    potential = np.asarray(phi, dtype=np.float64)
    spacing = float(period) / float(potential.size)
    return (
        2.0 * potential - np.roll(potential, 1) - np.roll(potential, -1)
    ) / (spacing * spacing)


def residual(phi, rhs, period: float = PERIOD) -> np.ndarray:
    """Return Aφ − ρ for the discrete periodic -Δ operator."""
    return minus_discrete_laplacian(phi, period) - np.asarray(rhs, dtype=np.float64)


def residual_l2(res, volumes) -> float:
    """Return the volume-weighted L2 residual norm."""
    field = np.asarray(res, dtype=np.float64)
    return reference_errors(field, np.zeros_like(field), volumes).l2


def damped_jacobi_step(
    phi, rhs, omega: float = JACOBI_OMEGA, period: float = PERIOD
) -> np.ndarray:
    """One damped Jacobi update: φ ← φ − ω D⁻¹ (Aφ − ρ), D = 2/h²."""
    potential = np.asarray(phi, dtype=np.float64)
    spacing = float(period) / float(potential.size)
    inv_diag = 0.5 * spacing * spacing
    return potential - float(omega) * inv_diag * residual(potential, rhs, period)


def apply_standin(
    phi,
    rhs,
    n_vcycles: int = N_VCYCLES,
    omega: float = JACOBI_OMEGA,
    period: float = PERIOD,
    volumes=None,
):
    """Apply n_vcycles residual applications. Returns (φ, residual L2 history)."""
    potential = np.asarray(phi, dtype=np.float64).copy()
    density = np.asarray(rhs, dtype=np.float64)
    if volumes is None:
        cell_volumes = np.full(potential.size, float(period) / float(potential.size))
    else:
        cell_volumes = np.asarray(volumes, dtype=np.float64)
    norms = [residual_l2(residual(potential, density, period), cell_volumes)]
    for _ in range(int(n_vcycles)):
        potential = damped_jacobi_step(potential, density, omega, period)
        norms.append(residual_l2(residual(potential, density, period), cell_volumes))
    return potential, norms
