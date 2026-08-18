"""Linear Jeans self-gravity: ω² = c_s² k² − 4π G ρ0.

Attractive gravity is the minus sign. Units c_s = 1, 4π G ρ0 = 1 ⇒ k_J = 1.
Stable k = 2 (ω real). Unstable k = 0.5 (γ = |ω|).

Closed field, primitives U = (ρ, u):

    U(t) = Ū + ε Re(r exp(ikx − iω t))     (stable)
    U(t) = Ū + ε Re(r exp(ikx + γ t))      (unstable, growing)

Poisson / gravity contract (do not flip in analyze):

    ∂xx φ = 4π G δρ
    g = −∂x φ

Overdensity sources a potential well; g points toward the overdensity.
Does not import the pops package or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

C_S = 1.0
RHO0 = 1.0
U0 = 0.0
FOUR_PI_G = 1.0
FOUR_PI_G_RHO0 = FOUR_PI_G * RHO0
K_JEANS = 1.0
K_STABLE = 2.0
K_UNSTABLE = 0.5
EPS = 1.0e-4
BACKGROUND = np.array([RHO0, U0], dtype=np.float64)
DOMAIN_LENGTH = 4.0 * math.pi
N_CELLS = 32


def jeans_wavenumber(c_s=C_S, four_pi_g_rho0=FOUR_PI_G_RHO0) -> float:
    """k_J = sqrt(4π G ρ0) / c_s. Equals 1 in the documented units."""
    speed = float(c_s)
    if speed <= 0.0:
        raise ValueError("sound speed must be positive")
    return float(math.sqrt(float(four_pi_g_rho0)) / speed)


def omega_squared(k, c_s=C_S, four_pi_g_rho0=FOUR_PI_G_RHO0) -> float:
    """ω² = c_s² k² − 4π G ρ0. Attractive gravity is the minus sign."""
    wavenumber = float(k)
    speed = float(c_s)
    return float(speed * speed * wavenumber * wavenumber - float(four_pi_g_rho0))


def angular_frequency(k, c_s=C_S, four_pi_g_rho0=FOUR_PI_G_RHO0) -> float:
    """Real ω for k > k_J. Raises if the mode is unstable."""
    value = omega_squared(k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    if value <= 0.0:
        raise ValueError(f"no real frequency at k={k}: ω²={value}")
    return float(math.sqrt(value))


def growth_rate(k, c_s=C_S, four_pi_g_rho0=FOUR_PI_G_RHO0) -> float:
    """γ = |ω| = sqrt(−ω²) for k < k_J. Raises if the mode is stable."""
    value = omega_squared(k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    if value >= 0.0:
        raise ValueError(f"no growth rate at k={k}: ω²={value}")
    return float(math.sqrt(-value))


def growth_factor(k, t, c_s=C_S, four_pi_g_rho0=FOUR_PI_G_RHO0) -> float:
    """Amplitude ratio exp(γ t) of the growing Jeans mode."""
    return float(math.exp(growth_rate(k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0) * float(t)))


def right_eigenvector(k, c_s=C_S, four_pi_g_rho0=FOUR_PI_G_RHO0, rho0=RHO0) -> np.ndarray:
    """Right eigenvector r of (δρ, δu).

    Stable: r = (1, ω/(ρ0 k)). Unstable growing: r = (1, i γ/(ρ0 k)).
    """
    wavenumber = float(k)
    density0 = float(rho0)
    if wavenumber == 0.0 or density0 == 0.0:
        raise ValueError("wavenumber and background density must be nonzero")
    value = omega_squared(wavenumber, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    if value > 0.0:
        omega = math.sqrt(value)
        return np.array([1.0, omega / (density0 * wavenumber)], dtype=np.complex128)
    if value < 0.0:
        gamma = math.sqrt(-value)
        return np.array([1.0, 1.0j * gamma / (density0 * wavenumber)], dtype=np.complex128)
    raise ValueError("Jeans wavenumber is marginally stable; no unique r")


def _space_time_phase(x, t, k, c_s, four_pi_g_rho0) -> np.ndarray:
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    wavenumber = float(k)
    value = omega_squared(wavenumber, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    if value > 0.0:
        omega = math.sqrt(value)
        return np.exp(1.0j * wavenumber * samples - 1.0j * omega * float(t))
    if value < 0.0:
        gamma = math.sqrt(-value)
        return np.exp(1.0j * wavenumber * samples + gamma * float(t))
    raise ValueError("Jeans wavenumber is marginally stable; no unique phase")


def uniform_cell_centers(n_cells: int, length: float = DOMAIN_LENGTH):
    """Uniform cell centers and volumes on the periodic interval of length 4π."""
    count = int(n_cells)
    width = float(length) / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_state(
    x,
    t,
    *,
    k,
    eps=EPS,
    c_s=C_S,
    four_pi_g_rho0=FOUR_PI_G_RHO0,
    rho0=RHO0,
) -> np.ndarray:
    """Closed Jeans eigenmode. Shape (2, n)."""
    phase = _space_time_phase(x, t, k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    vector = right_eigenvector(k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0, rho0=rho0)
    background = np.array([float(rho0), U0], dtype=np.float64)
    return background[:, None] + float(eps) * np.real(vector[:, None] * phase[None, :])


def potential(
    x,
    t,
    *,
    k,
    eps=EPS,
    c_s=C_S,
    four_pi_g=FOUR_PI_G,
    four_pi_g_rho0=FOUR_PI_G_RHO0,
) -> np.ndarray:
    """φ = −(4π G / k²) δρ. Attractive: overdensity is a potential well."""
    wavenumber = float(k)
    phase = _space_time_phase(x, t, k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    delta = float(eps) * np.real(phase)
    return -(float(four_pi_g) / (wavenumber * wavenumber)) * delta


def gravity(
    x,
    t,
    *,
    k,
    eps=EPS,
    c_s=C_S,
    four_pi_g=FOUR_PI_G,
    four_pi_g_rho0=FOUR_PI_G_RHO0,
) -> np.ndarray:
    """g = −∂x φ = (4π G / k²) ∂x δρ. Attractive: g points toward overdensity."""
    wavenumber = float(k)
    phase = _space_time_phase(x, t, k, c_s=c_s, four_pi_g_rho0=four_pi_g_rho0)
    d_delta_dx = float(eps) * np.real(1.0j * wavenumber * phase)
    return (float(four_pi_g) / (wavenumber * wavenumber)) * d_delta_dx
