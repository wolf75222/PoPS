"""TM-02 manufactured noncommuting split: A=diag(a1,a2), B collision.

a1 ≠ a2 so AB ≠ BA. Exact combined flow is exp((A+B) t). Subflows of A
and B are exact matrix exponentials.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

A1 = 1.0
A2 = 3.0
NU = 1.0
U0 = (1.0, 0.0)
T_END = 1.0
DT = 0.1
DT_SERIES = tuple(DT / factor for factor in (1, 2, 4, 8))


def operator_A(*, a1: float = A1, a2: float = A2) -> np.ndarray:
    """Diagonal rate matrix diag(a1, a2)."""
    return np.diag([float(a1), float(a2)]).astype(np.float64)


def operator_B(*, nu: float = NU) -> np.ndarray:
    """Two-species collision matrix [[-ν, ν], [ν, -ν]]."""
    rate = float(nu)
    return np.array(((-rate, rate), (rate, -rate)), dtype=np.float64)


def commutator(*, a1: float = A1, a2: float = A2, nu: float = NU) -> np.ndarray:
    """Return AB − BA."""
    left = operator_A(a1=a1, a2=a2)
    right = operator_B(nu=nu)
    return left @ right - right @ left


def operators_commute(*, a1: float = A1, a2: float = A2, nu: float = NU) -> bool:
    """True only when the manufactured pair commutes."""
    return bool(np.allclose(commutator(a1=a1, a2=a2, nu=nu), 0.0))


def _expm2(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.float64))
    reconstructed = vectors @ np.diag(np.exp(values)) @ np.linalg.inv(vectors)
    return np.real_if_close(reconstructed).astype(np.float64)


def flow_A(state, t, *, a1: float = A1, a2: float = A2) -> np.ndarray:
    """Exact flow of du/dt = A u."""
    vector = np.asarray(state, dtype=np.float64)
    return np.array(
        (
            np.exp(float(a1) * float(t)) * vector[0],
            np.exp(float(a2) * float(t)) * vector[1],
        ),
        dtype=np.float64,
    )


def flow_B(state, t, *, nu: float = NU) -> np.ndarray:
    """Exact flow of du/dt = B u (mean preserved, difference decays at 2ν)."""
    vector = np.asarray(state, dtype=np.float64)
    mid = 0.5 * (vector[0] + vector[1])
    amp = 0.5 * (vector[0] - vector[1]) * np.exp(-2.0 * float(nu) * float(t))
    return np.array((mid + amp, mid - amp), dtype=np.float64)


def exact_state(t, u0=U0, *, a1: float = A1, a2: float = A2, nu: float = NU) -> np.ndarray:
    """Exact combined flow exp((A+B) t) u0."""
    generator = operator_A(a1=a1, a2=a2) + operator_B(nu=nu)
    return _expm2(generator * float(t)) @ np.asarray(u0, dtype=np.float64)
