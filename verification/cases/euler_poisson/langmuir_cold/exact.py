"""1-d closed cold Langmuir standing-wave oracle.

Units: e = m_e = ε0 = 1, n0 = 1 ⇒ ω_pe = 1.

    n_e = n0 + A cos(kx) cos(ω_pe t)
    u_e = (A ω_pe)/(n0 k) sin(kx) sin(ω_pe t)
    E   = -(e A)/(ε0 k) sin(kx) cos(ω_pe t)
    φ   = -(e A)/(ε0 k²) cos(kx) cos(ω_pe t)

Ions are fixed at n_i = n0. Electron charge q_e = -e.
Poisson / Gauss contract (do not flip in analyze):

    ∂x E = e (n_i - n_e) / ε0
    E = -∂x φ

Does not import the pops package or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.0
Q_E = -E_CHARGE
M_E = 1.0
EPS0 = 1.0
N0 = 1.0
N_I = 1.0
A = 1.0e-4
K = 2.0 * np.pi
N_CELLS = 64
PHASE_CFL = 0.4


def plasma_frequency() -> float:
    """ω_pe = sqrt(n0 e² / (m_e ε0)). Equals 1 in the documented units."""
    return float(np.sqrt(N0 * E_CHARGE * E_CHARGE / (M_E * EPS0)))


def uniform_cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers and volumes on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def _xt(x, t):
    return np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
    )


def n_e(x, t):
    """Electron density n_e(x,t) = n0 + A cos(kx) cos(ω_pe t)."""
    xx, tt = _xt(x, t)
    return N0 + A * np.cos(K * xx) * np.cos(plasma_frequency() * tt)


def u_e(x, t):
    """Electron velocity u_e(x,t) = (A ω_pe)/(n0 k) sin(kx) sin(ω_pe t)."""
    xx, tt = _xt(x, t)
    omega = plasma_frequency()
    return (A * omega) / (N0 * K) * np.sin(K * xx) * np.sin(omega * tt)


def e_field(x, t):
    """Electric field E(x,t) = -(e A)/(ε0 k) sin(kx) cos(ω_pe t)."""
    xx, tt = _xt(x, t)
    amplitude = -(E_CHARGE * A) / (EPS0 * K)
    return amplitude * np.sin(K * xx) * np.cos(plasma_frequency() * tt)


def phi(x, t):
    """Potential φ(x,t) = -(e A)/(ε0 k²) cos(kx) cos(ω_pe t)."""
    xx, tt = _xt(x, t)
    amplitude = -(E_CHARGE * A) / (EPS0 * K * K)
    return amplitude * np.cos(K * xx) * np.cos(plasma_frequency() * tt)


def dE_dx(x, t):
    """Analytic ∂x E = -(e A)/ε0 cos(kx) cos(ω_pe t)."""
    xx, tt = _xt(x, t)
    return -(E_CHARGE * A) / EPS0 * np.cos(K * xx) * np.cos(plasma_frequency() * tt)


def gauss_rhs(x, t):
    """Charge source e (n_i - n_e) / ε0 of the Gauss law."""
    return E_CHARGE * (N_I - n_e(x, t)) / EPS0


def linear_eigenmode_fields(x, t):
    """Closed-form fields reconstructed from the linearized cold-fluid eigenvector.

    Spatial eigenfunction ``r(x)`` from Gauss + linearized momentum, then
    ``Re[r(x) exp(-i ω t)]``. This is the analytic eigenmode of *this* PDE, not a
    PoPS-generated matrix eigenvector.
    """
    xx, tt = _xt(x, t)
    omega = plasma_frequency()
    delta_n = A * np.cos(K * xx)
    electric = -(E_CHARGE * A) / (EPS0 * K) * np.sin(K * xx)
    potential = -(E_CHARGE * A) / (EPS0 * K * K) * np.cos(K * xx)
    # -iω δu = (q_e / m_e) E  with q_e = -e  ⇒  δu = i (ω A)/(n0 k) sin(kx)
    velocity = 1j * (omega * A) / (N0 * K) * np.sin(K * xx)
    phase = np.exp(-1j * omega * tt)
    return {
        "n": N0 + np.real(delta_n * phase),
        "u": np.real(velocity * phase),
        "e": np.real(electric * phase),
        "phi": np.real(potential * phase),
    }
