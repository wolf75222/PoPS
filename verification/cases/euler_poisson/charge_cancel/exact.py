"""Two-species charge cancellation oracle.

Species charges +q and -q share the same number density n. Then

    rho_q = q n + (-q) n = 0,

so -ε0 φ'' = 0. On a periodic interval with a mean-zero gauge, φ is the
constant PHI0 and E = -dφ/dx = 0.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

Q = 1.0
N0 = 1.0
EPS0 = 1.0
PHI0 = 0.0
K = 2.0 * np.pi
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


def charges() -> tuple[float, float]:
    """Opposite species charges (+q, -q)."""
    return (float(Q), -float(Q))


def density(x, *, n0: float = N0, delta: float = 0.0, k: float = K) -> np.ndarray:
    """Shared number density. Uniform when delta=0; same cosine for both species otherwise."""
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return float(n0) + float(delta) * np.cos(float(k) * samples)


def net_charge(
    x,
    *,
    n0: float = N0,
    delta: float = 0.0,
    k: float = K,
    order=("plus", "minus"),
) -> np.ndarray:
    """rho_q = q n + (-q) n. ``order`` permutes the species accumulation."""
    number_density = density(x, n0=n0, delta=delta, k=k)
    q_plus, q_minus = charges()
    terms = {"plus": q_plus * number_density, "minus": q_minus * number_density}
    if set(order) != {"plus", "minus"} or len(order) != 2:
        raise ValueError(f"order must be a permutation of ('plus', 'minus'), got {order!r}")
    return terms[order[0]] + terms[order[1]]


def poisson_rhs(x, **kwargs) -> np.ndarray:
    """-ε0 φ'' = rho_q / ε0. Identically zero after charge cancellation."""
    return net_charge(x, **kwargs) / float(EPS0)


def phi_exact(x) -> np.ndarray:
    """Constant potential after the mean-zero gauge."""
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return np.full(samples.shape, float(PHI0), dtype=np.float64)


def e_exact(x) -> np.ndarray:
    """E = -dφ/dx = 0 when φ is constant."""
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return np.zeros(samples.shape, dtype=np.float64)
