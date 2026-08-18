"""Phase / frequency diagnostics from an already-sampled probe (plan §7.7)."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.phase import (
    frequency_disagreement,
    frequency_error,
    numerical_frequency,
    phase_error,
)

OMEGA = 2.5
N_PERIODS = 8
SAMPLES_PER_PERIOD = 32


def _cosine_probe(omega=OMEGA, phase=0.0, offset=0.0):
    period = 2.0 * np.pi / omega
    n = N_PERIODS * SAMPLES_PER_PERIOD
    times = np.arange(n, dtype=np.float64) * (period / SAMPLES_PER_PERIOD)
    samples = offset + np.cos(omega * times + phase)
    return times, samples


def test_fft_returns_angular_frequency_not_cyclic():
    times, samples = _cosine_probe()
    omega_num = numerical_frequency(times, samples, method="fft")
    np.testing.assert_allclose(omega_num, OMEGA)
    assert omega_num != OMEGA / (2.0 * np.pi)
    assert omega_num != 1.0


def test_phase_fit_recovers_angular_frequency():
    times, samples = _cosine_probe()
    omega_num = numerical_frequency(times, samples, method="phase_fit")
    np.testing.assert_allclose(omega_num, OMEGA, rtol=1e-6)


def test_zero_crossing_recovers_angular_frequency():
    times, samples = _cosine_probe()
    omega_num = numerical_frequency(times, samples, method="zero_crossing")
    np.testing.assert_allclose(omega_num, OMEGA, rtol=1e-6)


def test_two_methods_agree_on_clean_cosine():
    times, samples = _cosine_probe()
    omega_fft = numerical_frequency(times, samples, method="fft")
    omega_fit = numerical_frequency(times, samples, method="phase_fit")
    disagreement = frequency_disagreement(omega_fft, omega_fit)
    np.testing.assert_allclose(disagreement, 0.0, atol=1e-6)
    assert disagreement >= 0.0


def test_frequency_error_is_zero_when_estimates_match_reference():
    np.testing.assert_allclose(frequency_error(OMEGA, OMEGA), 0.0)


def test_frequency_error_matches_langmuir_relative_form():
    np.testing.assert_allclose(frequency_error(2.75, 2.5), 0.1)


def test_phase_error_of_identical_series_is_zero():
    _, samples = _cosine_probe()
    np.testing.assert_allclose(phase_error(samples, samples), 0.0, atol=1e-12)


def test_phase_error_recovers_known_offsets():
    times, reference = _cosine_probe()
    _, plus_half_pi = _cosine_probe(phase=np.pi / 2.0)
    _, minus_third_pi = _cosine_probe(phase=-np.pi / 3.0)
    np.testing.assert_allclose(phase_error(plus_half_pi, reference), np.pi / 2.0, atol=1e-6)
    np.testing.assert_allclose(phase_error(minus_third_pi, reference), -np.pi / 3.0, atol=1e-6)
    assert times.size == reference.size


def test_offset_cosine_has_frequency_but_no_zero_crossings():
    times, samples = _cosine_probe(offset=2.0)
    np.testing.assert_allclose(
        numerical_frequency(times, samples, method="fft"), OMEGA
    )
    np.testing.assert_allclose(
        numerical_frequency(times, samples, method="phase_fit"), OMEGA, rtol=1e-6
    )
    with pytest.raises(ValueError):
        numerical_frequency(times, samples, method="zero_crossing")


def test_open_frequency_error_is_nonzero_observation():
    error = frequency_error(3.0, 2.5)
    np.testing.assert_allclose(error, 0.2)
    assert error != 0.0


@pytest.mark.parametrize(
    ("times", "samples", "method"),
    [
        (np.array([]), np.array([]), "fft"),
        ([0.0], [1.0], "fft"),
        ([0.0, 1.0], [1.0, 0.0, -1.0], "fft"),
        (0.0, 1.0, "fft"),
        ([[0.0, 1.0]], [[1.0, 0.0]], "fft"),
        (["a", "b"], [1.0, 0.0], "fft"),
        ([0.0, np.nan], [1.0, 0.0], "fft"),
        ([0.0, 1.0, 0.5], [1.0, 0.0, -1.0], "fft"),
        ([0.0, 1.0, 3.0], [1.0, 0.0, -1.0], "fft"),
        (*_cosine_probe(), "unknown"),
        (np.arange(32, dtype=np.float64), np.ones(32), "fft"),
        (*_cosine_probe(offset=2.0), "zero_crossing"),
    ],
    ids=[
        "empty",
        "length_1",
        "length_mismatch",
        "not_1d_scalars",
        "not_1d_matrix",
        "non_numeric",
        "non_finite",
        "non_increasing_times",
        "fft_non_uniform_times",
        "unknown_method",
        "fft_constant_signal",
        "zero_crossing_no_crossings",
    ],
)
def test_fail_closed_on_invalid_frequency_series(times, samples, method):
    with pytest.raises(ValueError):
        numerical_frequency(times, samples, method=method)


@pytest.mark.parametrize(
    ("omega_num", "omega_ref"),
    [
        ("a", 1.0),
        ([1.0], 1.0),
        (1.0, np.nan),
        (1.0, 0.0),
        (np.inf, 1.0),
    ],
    ids=[
        "non_numeric",
        "non_scalar",
        "non_finite_ref",
        "zero_ref",
        "non_finite_num",
    ],
)
def test_fail_closed_on_invalid_frequency_error(omega_num, omega_ref):
    with pytest.raises(ValueError):
        frequency_error(omega_num, omega_ref)


@pytest.mark.parametrize(
    ("omega_a", "omega_b"),
    [
        ("a", 1.0),
        ([1.0], 1.0),
        (1.0, np.nan),
        (np.inf, 1.0),
    ],
    ids=["non_numeric", "non_scalar", "non_finite_b", "non_finite_a"],
)
def test_fail_closed_on_invalid_frequency_disagreement(omega_a, omega_b):
    with pytest.raises(ValueError):
        frequency_disagreement(omega_a, omega_b)


@pytest.mark.parametrize(
    ("samples", "reference"),
    [
        (np.array([]), np.array([])),
        ([1.0, 0.0], [1.0]),
        (1.0, 1.0),
        ([[1.0, 0.0]], [[1.0, 0.0]]),
        (["a", "b"], [1.0, 0.0]),
        ([1.0, np.nan], [1.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
    ids=[
        "empty",
        "length_mismatch",
        "not_1d_scalars",
        "not_1d_matrix",
        "non_numeric",
        "non_finite",
        "vanishing_reference",
    ],
)
def test_fail_closed_on_invalid_phase_error(samples, reference):
    with pytest.raises(ValueError):
        phase_error(samples, reference)


def test_langmuir_probe_composition_recovers_plasma_frequency():
    omega_pe = 2.5
    n0 = 1.0
    amplitude = 1.0e-4
    times, oscillation = _cosine_probe(omega=omega_pe)
    density = n0 + amplitude * oscillation
    omega_fft = numerical_frequency(times, density, method="fft")
    omega_fit = numerical_frequency(times, density, method="phase_fit")
    np.testing.assert_allclose(frequency_error(omega_fft, omega_pe), 0.0, atol=1e-9)
    np.testing.assert_allclose(frequency_error(omega_fit, omega_pe), 0.0, atol=1e-6)
    np.testing.assert_allclose(frequency_disagreement(omega_fft, omega_fit), 0.0, atol=1e-6)
