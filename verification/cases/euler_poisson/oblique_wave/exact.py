"""2-d closed cold Langmuir-style oblique electrostatic eigenmode.

Unit square, integer wavevector k=(kx,ky)=(1,2) (not axis-aligned).

Simplest consistent pack. Potential amplitude is ε; density follows Poisson.

    θ = 2π (kx x + ky y) - ω t
    φ = ε cos(θ)
    E = -∇φ
    Δφ = -(2π)² (kx² + ky²) φ
    n = n̄ + (ε0 / e) Δφ

Poisson / Gauss (do not flip):

    ∇·E = e (n_i - n_e) / ε0
    i K · Ê = ρ̂ / ε0     with K = 2π k

Units: e = m_e = ε0 = 1, n̄ = n_i = 1 ⇒ ω_pe = 1.
ε = 10^{-4}. ω is a free parameter (default ω_pe).

Does not import the pops package or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.0
M_E = 1.0
EPS0 = 1.0
N_BAR = 1.0
N_I = 1.0
EPS = 1.0e-4
K_INTEGER = (1, 2)
N_CELLS = 32
TWO_PI = 2.0 * np.pi


def plasma_frequency() -> float:
    """ω_pe = sqrt(n̄ e² / (m_e ε0)). Equals 1 in the documented units."""
    return float(np.sqrt(N_BAR * E_CHARGE * E_CHARGE / (M_E * EPS0)))


def _integer_k(kx=None, ky=None) -> tuple[int, int]:
    default_x, default_y = K_INTEGER
    return (
        default_x if kx is None else int(kx),
        default_y if ky is None else int(ky),
    )


def physical_wavevector(kx=None, ky=None) -> np.ndarray:
    """Physical wavevector K = 2π (kx, ky) on the unit square."""
    kx_i, ky_i = _integer_k(kx, ky)
    return TWO_PI * np.array([kx_i, ky_i], dtype=np.float64)


def wavevector_squared(kx=None, ky=None) -> float:
    """|K|² = (2π)² (kx² + ky²)."""
    wave = physical_wavevector(kx, ky)
    return float(np.dot(wave, wave))


def _omega(omega) -> float:
    return plasma_frequency() if omega is None else float(omega)


def _xy(x, y):
    return np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def phase(x, y, t, *, kx=None, ky=None, omega=None) -> np.ndarray:
    """θ = 2π (kx x + ky y) - ω t."""
    xx, yy = _xy(x, y)
    kx_i, ky_i = _integer_k(kx, ky)
    return TWO_PI * (kx_i * xx + ky_i * yy) - _omega(omega) * float(t)


def phi(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> np.ndarray:
    """Potential φ = ε cos(θ)."""
    return float(eps) * np.cos(phase(x, y, t, kx=kx, ky=ky, omega=omega))


def laplacian_phi(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> np.ndarray:
    """Analytic Δφ = -|K|² φ."""
    return -wavevector_squared(kx, ky) * phi(
        x, y, t, kx=kx, ky=ky, omega=omega, eps=eps
    )


def e_field(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS):
    """Electric field E = -∇φ. Returns (Ex, Ey)."""
    sine = np.sin(phase(x, y, t, kx=kx, ky=ky, omega=omega))
    kx_i, ky_i = _integer_k(kx, ky)
    amplitude = float(eps) * TWO_PI
    return amplitude * kx_i * sine, amplitude * ky_i * sine


def n_e(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> np.ndarray:
    """Electron density n = n̄ + (ε0 / e) Δφ."""
    return N_BAR + (EPS0 / E_CHARGE) * laplacian_phi(
        x, y, t, kx=kx, ky=ky, omega=omega, eps=eps
    )


def div_E(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> np.ndarray:
    """Analytic ∇·E = -Δφ."""
    return -laplacian_phi(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps)


def gauss_rhs(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> np.ndarray:
    """Charge source e (n_i - n_e) / ε0 of the Gauss law."""
    return E_CHARGE * (N_I - n_e(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps)) / EPS0


def poisson_residual(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> np.ndarray:
    """Pointwise ∇·E - e (n_i - n_e) / ε0. Documented sign, no flip."""
    return div_E(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps) - gauss_rhs(
        x, y, t, kx=kx, ky=ky, omega=omega, eps=eps
    )


def velocity(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS):
    """Longitudinal cold velocity. Continuity holds iff ω = ω_pe."""
    cosine = np.cos(phase(x, y, t, kx=kx, ky=ky, omega=omega))
    kx_i, ky_i = _integer_k(kx, ky)
    coefficient = -(E_CHARGE / M_E) * TWO_PI * float(eps) / _omega(omega)
    return coefficient * kx_i * cosine, coefficient * ky_i * cosine


def complex_mode_amplitudes(kx=None, ky=None, eps=EPS) -> dict:
    """Complex Fourier amplitudes of φ = ε Re(exp(i K·x - i ω t))."""
    wave = physical_wavevector(kx, ky)
    phi_hat = float(eps)
    electric_hat = -1.0j * wave * phi_hat
    source_hat = float(np.dot(wave, wave)) * phi_hat
    return {"phi": phi_hat, "E": electric_hat, "source": source_hat}


def uniform_cell_mesh(n_cells: int = N_CELLS):
    """Uniform cell centers and volumes on the periodic unit square."""
    count = int(n_cells)
    width = 1.0 / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    volumes = np.full((count, count), width * width, dtype=np.float64)
    return x, y, volumes


def exact_fields(x, y, t, *, kx=None, ky=None, omega=None, eps=EPS) -> dict:
    """Closed 2-d oblique pack. Each field broadcasts to (x, y)."""
    potential = phi(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps)
    electric_x, electric_y = e_field(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps)
    velocity_x, velocity_y = velocity(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps)
    density = n_e(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps)
    return {
        "phi": potential,
        "E_x": electric_x,
        "E_y": electric_y,
        "n_e": density,
        "u_e": velocity_x,
        "v_e": velocity_y,
        "div_E": div_E(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps),
        "gauss_rhs": gauss_rhs(x, y, t, kx=kx, ky=ky, omega=omega, eps=eps),
    }
