from __future__ import annotations

import numpy as np
import pytest

from helpers.verification import (
    convergence_orders,
    direction_velocity,
    sine_diagnostics,
    sine_wave_cell_averages,
    weighted_error_norms,
)


def test_cell_average_uses_the_finite_volume_sinc_factor_not_a_center_sample():
    values, coordinates = sine_wave_cell_averages((8,), (1,), epsilon=0.2)
    expected = 1.0 + 0.2 * np.sinc(1.0 / 8.0) * np.sin(2.0 * np.pi * coordinates[0])
    assert np.allclose(values, expected)
    assert not np.allclose(values, 1.0 + 0.2 * np.sin(2.0 * np.pi * coordinates[0]))


def test_weighted_norms_and_convergence_are_volume_normalised():
    norms = weighted_error_norms(np.array([1.0, 3.0]), np.array([0.0, 0.0]), np.array([3.0, 1.0]))
    assert norms.l1 == pytest.approx(1.5)
    assert norms.l2 == pytest.approx(np.sqrt(3.0))
    assert norms.linf == pytest.approx(3.0)
    assert convergence_orders((16, 32, 64), (0.04, 0.01, 0.0025)) == [
        None,
        pytest.approx(2.0),
        pytest.approx(2.0),
    ]


def test_direction_rejects_transverse_requests_but_never_changes_wave_numbers():
    assert direction_velocity("diagonal", 3) == (1.0, 1.0, 1.0)
    assert direction_velocity("diagonal", 2) == (1.0, 1.0)
    assert direction_velocity("diagonal", 1) == (1.0,)
    assert direction_velocity("x", 1) == (1.0,)
    with pytest.raises(ValueError, match="unavailable"):
        direction_velocity("y", 1)


def test_phase_is_undefined_when_the_numerical_wave_has_zero_amplitude():
    diagnostics = sine_diagnostics(
        np.ones(8),
        1.0 + 0.1 * np.sin(2.0 * np.pi * (np.arange(8) + 0.5) / 8.0),
        1.0 / 8.0,
    )
    assert diagnostics["phase_cosine"] is None
    assert diagnostics["phase_defined"] is False


def test_integer_inputs_are_exact_and_never_silently_truncated():
    with pytest.raises(ValueError, match="positive extents"):
        sine_wave_cell_averages((8.9,), (1,), epsilon=0.1)
    with pytest.raises(ValueError, match="exact integers"):
        sine_wave_cell_averages((8,), (1.7,), epsilon=0.1)
    with pytest.raises(ValueError, match="positive extents"):
        convergence_orders((16, 32.5), (0.1, 0.025))
    with pytest.raises(ValueError, match="dimension"):
        direction_velocity("x", True)
