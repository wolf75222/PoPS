"""Phase and frequency diagnostics from an already-sampled probe.

Plan §7.7:

    E_ω = |ω_num - ω_ref| / |ω_ref|

Frequency is estimated by one named method among temporal FFT, phase
fitting, and zero crossings. Method disagreement is |ω_a - ω_b|.
"""
from __future__ import annotations

import numpy as np

_FREQUENCY_METHODS = frozenset({"fft", "phase_fit", "zero_crossing"})


def _as_float64(value) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric values") from exc
    if array.size == 0:
        raise ValueError("empty input")
    return array


def _as_series(value) -> np.ndarray:
    array = _as_float64(value)
    if array.ndim != 1:
        raise ValueError("shape mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite values")
    return array


def _as_scalar(value) -> float:
    array = _as_float64(value)
    if array.ndim != 0:
        raise ValueError("shape mismatch")
    if not np.isfinite(array):
        raise ValueError("non-finite values")
    return float(array)


def _analytic_signal(samples: np.ndarray) -> np.ndarray:
    n = samples.size
    spectrum = np.fft.fft(samples)
    weights = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        weights[0] = 1.0
        weights[1 : n // 2] = 2.0
        weights[n // 2] = 1.0
    else:
        weights[0] = 1.0
        weights[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * weights)


def _require_times_and_samples(times, samples) -> tuple[np.ndarray, np.ndarray]:
    time_series = _as_series(times)
    sample_series = _as_series(samples)
    if time_series.shape != sample_series.shape:
        raise ValueError("shape mismatch")
    if time_series.size < 2:
        raise ValueError("length < 2")
    steps = np.diff(time_series)
    if np.any(steps <= 0.0):
        raise ValueError("non-increasing times")
    return time_series, sample_series


def _frequency_fft(times: np.ndarray, samples: np.ndarray) -> float:
    steps = np.diff(times)
    if not np.allclose(steps, steps[0]):
        raise ValueError("non-uniform time")
    spectrum = np.fft.rfft(samples)
    spectrum[0] = 0.0
    if not np.any(np.abs(spectrum) > 0.0):
        raise ValueError("no oscillatory mode")
    peak = int(np.argmax(np.abs(spectrum)))
    if peak == 0:
        raise ValueError("no oscillatory mode")
    frequency_hz = float(np.fft.rfftfreq(samples.size, float(steps[0]))[peak])
    omega = 2.0 * np.pi * frequency_hz
    if not np.isfinite(omega):
        raise ValueError("non-finite values")
    return float(omega)


def _frequency_phase_fit(times: np.ndarray, samples: np.ndarray) -> float:
    analytic = _analytic_signal(samples - np.mean(samples))
    phase = np.unwrap(np.angle(analytic))
    slope = float(np.polyfit(times, phase, 1)[0])
    omega = abs(slope)
    if not np.isfinite(omega):
        raise ValueError("non-finite values")
    return omega


def _frequency_zero_crossing(times: np.ndarray, samples: np.ndarray) -> float:
    crossings = []
    for index in range(samples.size - 1):
        left = samples[index]
        right = samples[index + 1]
        if left == 0.0:
            if index == 0 or samples[index - 1] != 0.0:
                crossings.append(float(times[index]))
            continue
        if left * right < 0.0:
            weight = left / (left - right)
            crossings.append(
                float(times[index] + weight * (times[index + 1] - times[index]))
            )
    if len(crossings) < 2:
        raise ValueError("too few zero crossings")
    half_periods = np.diff(np.asarray(crossings, dtype=np.float64))
    omega = np.pi / float(np.mean(half_periods))
    if not np.isfinite(omega):
        raise ValueError("non-finite values")
    return float(omega)


def numerical_frequency(times, samples, *, method) -> float:
    """Return angular frequency ω_num of an already-sampled real probe."""
    if method not in _FREQUENCY_METHODS:
        raise ValueError("unknown method")
    time_series, sample_series = _require_times_and_samples(times, samples)
    if method == "fft":
        return _frequency_fft(time_series, sample_series)
    if method == "phase_fit":
        return _frequency_phase_fit(time_series, sample_series)
    return _frequency_zero_crossing(time_series, sample_series)


def frequency_error(omega_num, omega_ref) -> float:
    """Return the §7.7 relative frequency error E_ω."""
    omega_num_value = _as_scalar(omega_num)
    omega_ref_value = _as_scalar(omega_ref)
    if omega_ref_value == 0.0:
        raise ValueError("zero reference frequency")
    error = abs(omega_num_value - omega_ref_value) / abs(omega_ref_value)
    if not np.isfinite(error):
        raise ValueError("non-finite values")
    return float(error)


def phase_error(samples, reference) -> float:
    """Return the wrapped phase offset of samples versus a same-grid reference."""
    sample_series = _as_series(samples)
    reference_series = _as_series(reference)
    if sample_series.shape != reference_series.shape:
        raise ValueError("shape mismatch")
    if sample_series.size < 2:
        raise ValueError("length < 2")
    analytic_samples = _analytic_signal(sample_series)
    analytic_reference = _analytic_signal(reference_series)
    inner = np.vdot(analytic_reference, analytic_samples)
    if abs(inner) == 0.0:
        raise ValueError("vanishing analytic signal")
    offset = float(np.angle(inner))
    if not np.isfinite(offset):
        raise ValueError("non-finite values")
    return offset


def frequency_disagreement(omega_a, omega_b) -> float:
    """Return |ω_a - ω_b| as the §7.7 method-disagreement signal."""
    disagreement = abs(_as_scalar(omega_a) - _as_scalar(omega_b))
    if not np.isfinite(disagreement):
        raise ValueError("non-finite values")
    return float(disagreement)
