"""Langmuir k-sweep campaign helper (in-memory; no solver required)."""
from __future__ import annotations

import numpy as np

from verification.pops_verify.langmuir_sweep import (
    WAVE_NUMBERS_OVER_2PI,
    analyze_probe,
    cells_for_cycles,
    cold_point,
    probe_index,
    sample_times,
    warm_points,
)
from verification.pops_verify.phase import frequency_error


def test_warm_points_cover_four_canonical_wavenumbers():
    points = warm_points()
    assert tuple(point.cycles for point in points) == WAVE_NUMBERS_OVER_2PI
    for point in points:
        assert point.case_id == "CP-03"
        assert point.n_cells == cells_for_cycles(point.cycles)
        assert point.n_cells >= 32 * point.cycles
        k = 2.0 * np.pi * float(point.cycles)
        np.testing.assert_allclose(
            point.omega_ref**2,
            1.0 + 0.04 * k * k,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_analyze_probe_recovers_reference_frequency():
    omega = 2.5
    times = sample_times(omega)
    samples = np.cos(omega * times)
    row = analyze_probe(times, samples, omega)
    np.testing.assert_allclose(frequency_error(row["omega_fft"], omega), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(row["error_fft"], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(row["error_fit"], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(row["disagreement"], 0.0, atol=1.0e-6)


def test_cold_point_is_omega_pe_at_unit_wavenumber():
    point = cold_point()
    assert point.case_id == "CP-02"
    assert point.cycles == 1
    np.testing.assert_allclose(point.omega_ref, 1.0)
    assert 0 <= probe_index(point.n_cells) < point.n_cells
