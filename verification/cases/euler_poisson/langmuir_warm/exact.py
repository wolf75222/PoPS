"""Warm Langmuir linear dispersion ω² = ω_pe² + c_e² k².

Default ω_pe=1, c_e=0.2. Sweep k/(2π) = 1, 2, 4, 8 on the periodic unit
interval. Units e = m_e = ε0 = 1, n0 = 1 ⇒ ω_pe = 1. Does not import pops
or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

OMEGA_PE = 1.0
C_E = 0.2
WAVE_NUMBERS_OVER_2PI = (1, 2, 4, 8)
TWO_PI = 2.0 * math.pi
N0 = 1.0
E_CHARGE = 1.0
EPSILON_0 = 1.0
AMPLITUDE = 1.0e-4


def angular_frequency(k, omega_pe=OMEGA_PE, c_e=C_E) -> float:
    """Positive branch of ω² = ω_pe² + c_e² k²."""
    wavenumber = float(k)
    plasma = float(omega_pe)
    thermal = float(c_e)
    return float(math.sqrt(plasma * plasma + thermal * thermal * wavenumber * wavenumber))


def wavenumber(cycles) -> float:
    """k = 2π n for integer cycles on the unit interval."""
    return TWO_PI * float(cycles)


def dispersion_residual(k, omega_pe=OMEGA_PE, c_e=C_E) -> float:
    """ω² - (ω_pe² + c_e² k²). Zero for the exact branch."""
    omega = angular_frequency(k, omega_pe=omega_pe, c_e=c_e)
    wavenumber_value = float(k)
    plasma = float(omega_pe)
    thermal = float(c_e)
    return float(
        omega * omega
        - (plasma * plasma + thermal * thermal * wavenumber_value * wavenumber_value)
    )


def uniform_cell_centers(n_cells: int):
    """Uniform cell centers and volumes on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_fields(
    x,
    t,
    *,
    k,
    omega_pe=OMEGA_PE,
    c_e=C_E,
    n0=N0,
    amplitude=AMPLITUDE,
) -> dict:
    """Closed 1-d warm Langmuir eigenmode. Each field has shape (n,)."""
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    wavenumber_value = float(k)
    omega = angular_frequency(wavenumber_value, omega_pe=omega_pe, c_e=c_e)
    density0 = float(n0)
    amp = float(amplitude)
    phase = wavenumber_value * samples
    time_phase = omega * float(t)
    cosine_t = np.cos(time_phase)
    sine_t = np.sin(time_phase)
    density = density0 + amp * np.cos(phase) * cosine_t
    velocity = (amp * omega) / (density0 * wavenumber_value) * np.sin(phase) * sine_t
    electric = (
        -(E_CHARGE * amp) / (EPSILON_0 * wavenumber_value) * np.sin(phase) * cosine_t
    )
    potential = (
        -(E_CHARGE * amp)
        / (EPSILON_0 * wavenumber_value * wavenumber_value)
        * np.cos(phase)
        * cosine_t
    )
    return {
        "n_e": density,
        "u_e": velocity,
        "E": electric,
        "phi": potential,
        "omega": omega,
    }
