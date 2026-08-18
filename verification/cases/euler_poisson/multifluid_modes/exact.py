"""2×2 toy multifluid eigenmode generator.

Linearized two-component acoustic toy on the periodic unit interval:

    ∂t n + c ∂x u = 0
    ∂t u + c ∂x n = 0

Fourier convention exp(ikx) gives ∂t Û = M(k) Û with

    M(k) = [[ 0,    -i c k ],
            [ -i c k,  0   ]]

Known eigenpairs:

    λ+ = +i c k,  r+ = (1, -1)
    λ- = -i c k,  r- = (1, +1)

The closed reference is U(x,t) = Ū + ε Re(r exp(ikx + λ t)).
Defaults: c=1, Ū=(1,0), ε=10^{-4}, k=2π. Does not import pops.
"""
from __future__ import annotations

import math

import numpy as np

WAVE_SPEED = 1.0
EPS = 1.0e-4
BACKGROUND = np.array([1.0, 0.0], dtype=np.float64)
MODES = ("plus", "minus")
CANONICAL_K = 2.0 * math.pi


def system_matrix(k) -> np.ndarray:
    """Return the 2×2 Fourier symbol M(k)."""
    off = -1.0j * WAVE_SPEED * float(k)
    return np.array([[0.0, off], [off, 0.0]], dtype=np.complex128)


def eigenvalue(mode: str, k) -> complex:
    """λ+ = +i c k, λ- = -i c k."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    signed = 1.0 if mode == "plus" else -1.0
    return 1.0j * signed * WAVE_SPEED * float(k)


def right_eigenvector(mode: str, k) -> np.ndarray:
    """Right eigenvector of M(k). Independent of k for this toy."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    del k
    if mode == "plus":
        return np.array([1.0, -1.0], dtype=np.complex128)
    return np.array([1.0, 1.0], dtype=np.complex128)


def uniform_cell_centers(n_cells: int):
    """Uniform cell centers and volumes on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_state(x, t, *, mode, k=CANONICAL_K, eps=EPS) -> np.ndarray:
    """U(x,t) = Ū + ε Re(r exp(ikx + λ t)). Shape (2, n)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    lam = eigenvalue(mode, k)
    vector = right_eigenvector(mode, k)
    phase = np.exp(1.0j * float(k) * samples + lam * float(t))
    return BACKGROUND[:, None] + float(eps) * np.real(vector[:, None] * phase[None, :])


def advance_fourier(uhat, t, *, k) -> np.ndarray:
    """Time-advance a Fourier coefficient: exp(M t) û."""
    hat = np.asarray(uhat, dtype=np.complex128).reshape(2)
    plus = right_eigenvector("plus", k)
    minus = right_eigenvector("minus", k)
    basis = np.column_stack((plus, minus))
    coefficients = np.linalg.solve(basis, hat)
    evolved = (
        coefficients[0] * np.exp(eigenvalue("plus", k) * float(t)) * plus
        + coefficients[1] * np.exp(eigenvalue("minus", k) * float(t)) * minus
    )
    return evolved
