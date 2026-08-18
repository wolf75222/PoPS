"""1-d ion-acoustic eigenmode: Boltzmann electrons, cold ions.

Linearized two-fluid toy on the periodic unit interval. State is
(δn_i, δu_i). Fourier convention exp(ikx) gives ∂t Û = M(k) Û with

    M(k) = [[ 0,                          -i k n0 ],
            [ -i k (c_s²/n0)/(1+k²λ_D²),   0      ]]

Dispersion:

    ω² = k² c_s² / (1 + k² λ_D²)
    c_s² = T_e / m_i
    λ_D² = ε0 T_e / (n0 e²)

Eigenpairs of M:

    λ+ = -iω,  r+ = (1,  ω/(k n0))
    λ- = +iω,  r- = (1, -ω/(k n0))

The closed reference is U(x,t) = Ū + ε Re(r exp(ikx + λ t)).
The plus branch is the brief traveling wave exp(ikx - iωt).
Defaults: e = m_i = ε0 = n0 = T_e = 1, Ū = (n0, 0), ε = 10^{-4},
k = 2π. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

E_CHARGE = 1.0
M_I = 1.0
EPS0 = 1.0
N0 = 1.0
T_E = 1.0
EPS = 1.0e-4
BACKGROUND = np.array([N0, 0.0], dtype=np.float64)
MODES = ("plus", "minus")
CANONICAL_K = 2.0 * math.pi
WAVE_NUMBERS_OVER_2PI = (1, 2, 4, 8)
TWO_PI = 2.0 * math.pi


def sound_speed_squared() -> float:
    """c_s² = T_e / m_i."""
    return float(T_E / M_I)


def debye_length_squared() -> float:
    """λ_D² = ε0 T_e / (n0 e²)."""
    return float(EPS0 * T_E / (N0 * E_CHARGE * E_CHARGE))


def angular_frequency(k) -> float:
    """Positive branch of ω² = k² c_s² / (1 + k² λ_D²)."""
    wavenumber = float(k)
    c_s2 = sound_speed_squared()
    lambda_d2 = debye_length_squared()
    return float(
        math.sqrt((wavenumber * wavenumber * c_s2) / (1.0 + wavenumber * wavenumber * lambda_d2))
    )


def wavenumber(cycles) -> float:
    """k = 2π n for integer cycles on the unit interval."""
    return TWO_PI * float(cycles)


def dispersion_residual(k) -> float:
    """ω² - k² c_s² / (1 + k² λ_D²). Zero for the exact branch."""
    wavenumber_value = float(k)
    omega = angular_frequency(wavenumber_value)
    c_s2 = sound_speed_squared()
    lambda_d2 = debye_length_squared()
    return float(
        omega * omega
        - (wavenumber_value * wavenumber_value * c_s2)
        / (1.0 + wavenumber_value * wavenumber_value * lambda_d2)
    )


def system_matrix(k) -> np.ndarray:
    """Return the 2×2 Fourier symbol M(k) on (δn_i, δu_i)."""
    wavenumber_value = float(k)
    screening = 1.0 + wavenumber_value * wavenumber_value * debye_length_squared()
    m12 = -1.0j * wavenumber_value * N0
    m21 = -1.0j * wavenumber_value * (sound_speed_squared() / N0) / screening
    return np.array([[0.0, m12], [m21, 0.0]], dtype=np.complex128)


def eigenvalue(mode: str, k) -> complex:
    """λ+ = -iω, λ- = +iω."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    omega = angular_frequency(k)
    if mode == "plus":
        return -1.0j * omega
    return 1.0j * omega


def right_eigenvector(mode: str, k) -> np.ndarray:
    """Right eigenvector of M(k). r+ = (1, ω/(k n0)), r- = (1, -ω/(k n0))."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    wavenumber_value = float(k)
    amp_u = angular_frequency(wavenumber_value) / (wavenumber_value * N0)
    if mode == "plus":
        return np.array([1.0, amp_u], dtype=np.complex128)
    return np.array([1.0, -amp_u], dtype=np.complex128)


def uniform_cell_centers(n_cells: int):
    """Uniform cell centers and volumes on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_state(x, t, *, mode="plus", k=CANONICAL_K, eps=EPS) -> np.ndarray:
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
