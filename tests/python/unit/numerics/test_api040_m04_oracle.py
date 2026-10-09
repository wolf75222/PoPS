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


@pytest.mark.parametrize("n", (32, 64, 128))
def test_exact_zero_transverse_coefficient_has_no_transverse_restriction(n):
    parts = oracle.frequencies(n, transverse_diffusivity=0.)
    assert parts["diffusion_y"] == 0.
    assert parts["installed_2d"] == parts["physical_1d"]
    assert parts["installed_2d"] == pytest.approx(n + .02 * n * n)


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


@pytest.mark.parametrize("method", ("forward_euler", "ssprk2"))
@pytest.mark.parametrize("velocity", (-1.3, 0., 1.))
def test_discrete_fourier_oracle_matches_independent_periodic_cell_updates(method, velocity):
    n, dt, end, diffusion = 13, .007, .051, .02
    value = oracle.exact_cell_means(n, 0.)

    def rhs(q):
        derivative = ((q - np.roll(q, 1)) if velocity >= 0
                      else (np.roll(q, -1) - q)) * n
        return (-velocity * derivative + diffusion * n*n *
                (np.roll(q, 1) - 2*q + np.roll(q, -1)))

    time = 0.
    while time < end:
        step = min(dt, end - time)
        first = value + step * rhs(value)
        value = first if method == "forward_euler" else .5 * value + .5 * (first + step * rhs(first))
        time += step
    expected = oracle.discrete_cell_means(n, end, dt, method=method,
                                          velocity=velocity, diffusivity=diffusion)
    np.testing.assert_allclose(value, expected, rtol=0., atol=8.e-16)


def test_forward_euler_finite_grid_order_failure_is_a_property_of_the_selected_method():
    errors = {method: [] for method in ("forward_euler", "ssprk2")}
    for n in (32, 64, 128):
        dt = .9 / oracle.frequencies(n, transverse_diffusivity=0.)["physical_1d"]
        exact = oracle.exact_cell_means(n, .1)
        for method in errors:
            computed = oracle.discrete_cell_means(n, .1, dt, method=method)
            errors[method].append(np.mean(np.abs(computed - exact)))
    # No acceptance threshold is relaxed: retain the failed FE trajectory and
    # test a distinct authored temporal method against the same PDE criteria.
    assert np.log2(errors["forward_euler"][0] / errors["forward_euler"][1]) < .7
    assert np.all(np.log2(np.array(errors["ssprk2"][:-1]) / errors["ssprk2"][1:]) > .7)
