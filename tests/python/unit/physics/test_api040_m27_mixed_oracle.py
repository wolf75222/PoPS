"""Independent mathematics for the closed periodic mixed Cahn--Hilliard subcase."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "migration" / "scientific"
sys.path.insert(0, str(EXAMPLES))
from api040_m27_mixed_oracle import (  # noqa: E402
    EPSILON, STEP, STEPS, fourier_mixed_step, initial_means,
    integrated_initial_means, original_residuals, quadratic_energy, trajectory,
)


@pytest.mark.parametrize("cells", (16, 32, 64))
def test_fourier_oracle_keeps_both_original_equations_and_mass(cells):
    initial, before, after, mu = trajectory(cells)
    np.testing.assert_allclose(initial, integrated_initial_means(cells), atol=2e-15, rtol=0)
    first, second = original_residuals(before, after, mu)
    assert first < 2e-13 and second < 2e-13
    assert abs(np.mean(after) - np.mean(initial)) < 2e-15
    assert quadratic_energy(after, EPSILON) < quadratic_energy(before, EPSILON)
    assert quadratic_energy(before, EPSILON) < quadratic_energy(initial, EPSILON)
    assert STEPS == 10 and STEP == .01


def test_permuted_two_equation_oracle_agrees_without_eliminating_mu():
    c = initial_means(16)
    expected_c, expected_mu = fourier_mixed_step(c)
    transformed = np.fft.fft(c)
    obtained_c = np.empty(16, dtype=complex)
    obtained_mu = np.empty(16, dtype=complex)
    for index, load in enumerate(transformed):
        symbol = -4 * 16**2 * np.sin(np.pi * index / 16)**2
        # Reverse both equation rows and unknown columns. This is a distinct
        # algebraic solve, not a relabelled scalar fourth-order formula.
        matrix = np.array(((1., -1. + EPSILON**2 * symbol),
                           (-STEP * symbol, 1.)))
        obtained_mu[index], obtained_c[index] = np.linalg.solve(
            matrix, np.array((0j, load)))
    np.testing.assert_allclose(np.fft.ifft(obtained_c).real, expected_c, atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.fft.ifft(obtained_mu).real, expected_mu, atol=2e-15, rtol=0)
