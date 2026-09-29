"""M04/W02 analytic cell means and combined-bound counterexample."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = (Path(__file__).resolve().parents[4] / "examples" / "migration"
        / "scientific" / "api040_m04_oracle.py")
SPEC = importlib.util.spec_from_file_location("api040_m04_oracle", PATH)
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


@pytest.mark.parametrize("n", (32, 64, 128))
def test_exact_cosine_cell_mean_is_integral_not_center_value(n):
    edges = np.arange(n + 1, dtype=float) / n
    exact_initial = 1 + .2 * n * np.diff(np.sin(2 * np.pi * edges)) / (2 * np.pi)
    cell_mean = oracle.exact_cell_means(n, 0.)
    np.testing.assert_allclose(cell_mean, exact_initial, rtol=0., atol=5.e-15)
    point_value = 1 + .2 * np.cos(2 * np.pi * (np.arange(n) + .5) / n)
    assert np.max(np.abs(cell_mean - point_value)) > 1.e-6
    assert np.mean(cell_mean) == pytest.approx(1., abs=1.e-15)
    final = oracle.exact_cell_means(n, .1)
    assert np.mean(final) == pytest.approx(1., abs=1.e-15)
    assert final.min() > .8


@pytest.mark.parametrize("n", (32, 64, 128))
def test_actual_2d_bound_sums_both_diffusion_axes(n):
    parts = oracle.frequencies(n)
    assert parts["physical_1d"] == pytest.approx(n + .02 * n * n)
    assert parts["installed_2d"] == pytest.approx(n + .04 * n * n)
    dt = .9 / parts["installed_2d"]
    assert dt * parts["installed_2d"] == pytest.approx(.9)
    assert dt <= .9 / parts["physical_1d"]


def test_individually_acceptable_bounds_have_negative_combined_impulse_center():
    n = 32
    dt = .75 / n  # a=1, c=.75, D=.01 gives r=.24
    c = dt * n
    r = dt * .01 * n * n
    assert c < 1 and 2 * r < 1 and 4 * r < 1
    assert c + 2 * r > 1 and c + 4 * r > 1
    one_dimensional = oracle.impulse_weights_1d(c, r)
    two_dimensional = oracle.impulse_weights_2d(c, r)
    assert one_dimensional["center"] == pytest.approx(-.23)
    assert two_dimensional["center"] == pytest.approx(-.71)
    assert sum(one_dimensional.values()) == pytest.approx(1.)
    assert sum(two_dimensional.values()) == pytest.approx(1.)
