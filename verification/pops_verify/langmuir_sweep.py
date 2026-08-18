"""Langmuir frequency sweep: kL/(2π) = 1, 2, 4, 8.

Plan §CP-03: at least 16–32 cells per wavelength on gate points.
``E_ω = |ω_num - ω_ref| / |ω_ref|`` from ``phase.frequency_error``.
Does not compile or call pops.run; the machine driver supplies probes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verification.pops_verify.phase import frequency_disagreement, frequency_error, numerical_frequency

WAVE_NUMBERS_OVER_2PI = (1, 2, 4, 8)
CELLS_PER_WAVELENGTH = 32
N_PERIODS = 8
SAMPLES_PER_PERIOD = 16
PROBE_X = 0.25


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One (case, k) probe request."""

    case_id: str
    cycles: int
    n_cells: int
    omega_ref: float
    t_end: float
    times: np.ndarray


def cells_for_cycles(cycles: int, *, cells_per_wavelength: int = CELLS_PER_WAVELENGTH) -> int:
    """Return n such that each wavelength has at least ``cells_per_wavelength`` cells."""
    count = int(cycles) * int(cells_per_wavelength)
    if count < int(cells_per_wavelength):
        raise ValueError("non-positive sweep resolution")
    return count


def sample_times(omega: float, *, n_periods: int = N_PERIODS, samples_per_period: int = SAMPLES_PER_PERIOD):
    """Uniform times covering ``n_periods`` of angular frequency ``omega``."""
    period = 2.0 * np.pi / float(omega)
    n = int(n_periods) * int(samples_per_period)
    return np.arange(n, dtype=np.float64) * (period / float(samples_per_period))


def warm_points(
    *,
    omega_pe: float = 1.0,
    c_e: float = 0.2,
    cells_per_wavelength: int = CELLS_PER_WAVELENGTH,
) -> tuple[SweepPoint, ...]:
    """CP-03 points at the four canonical wavenumbers."""
    points = []
    for cycles in WAVE_NUMBERS_OVER_2PI:
        k = 2.0 * np.pi * float(cycles)
        omega = float(np.sqrt(omega_pe * omega_pe + c_e * c_e * k * k))
        times = sample_times(omega)
        points.append(
            SweepPoint(
                case_id="CP-03",
                cycles=int(cycles),
                n_cells=cells_for_cycles(cycles, cells_per_wavelength=cells_per_wavelength),
                omega_ref=omega,
                t_end=float(times[-1]),
                times=times,
            )
        )
    return tuple(points)


def cold_point(*, omega_pe: float = 1.0, n_cells: int = 64) -> SweepPoint:
    """CP-02 cold Langmuir at k/(2π)=1 (ω = ω_pe, independent of k)."""
    times = sample_times(omega_pe)
    return SweepPoint(
        case_id="CP-02",
        cycles=1,
        n_cells=int(n_cells),
        omega_ref=float(omega_pe),
        t_end=float(times[-1]),
        times=times,
    )


def probe_index(n_cells: int, probe_x: float = PROBE_X) -> int:
    """Nearest cell-center index to ``probe_x`` on the unit interval."""
    centers = (np.arange(int(n_cells), dtype=np.float64) + 0.5) / float(n_cells)
    return int(np.argmin(np.abs(centers - float(probe_x))))


def analyze_probe(times, samples, omega_ref: float) -> dict:
    """Return ω_fft, ω_fit, relative errors, and method disagreement."""
    omega_fft = numerical_frequency(times, samples, method="fft")
    omega_fit = numerical_frequency(times, samples, method="phase_fit")
    return {
        "omega_fft": float(omega_fft),
        "omega_fit": float(omega_fit),
        "omega_ref": float(omega_ref),
        "error_fft": float(frequency_error(omega_fft, omega_ref)),
        "error_fit": float(frequency_error(omega_fit, omega_ref)),
        "disagreement": float(frequency_disagreement(omega_fft, omega_fit)),
    }
