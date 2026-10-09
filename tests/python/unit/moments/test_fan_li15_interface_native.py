"""Isolated common-covector interface formulas; no native spatial/AMR runtime."""

from fractions import Fraction
from math import cos, sin

import numpy as np
import pytest

from pops.moments import fan_li15_expressions
from tests.python.support.fan_li15_oracle import TOP, derivative_gaussian_flux, quadrature_path
from tests.python.unit.moments import test_fan_li15_path_native as path_witnesses

path_bridge = path_witnesses.path_bridge
states = path_witnesses.states


@pytest.mark.compiler
@pytest.mark.parametrize(
    "direction",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.6, -0.8),
        (cos(0.731), sin(0.731)),
        (-sin(0.731) / 7.5, cos(0.731) / 7.5),
    ],
)
def test_native_grad_endpoint_flux_matches_independent_gaussian_identity(path_bridge, direction):
    for state in states():
        result = path_bridge.flux(state, direction)
        assert result["status"] == 0
        rational = tuple(map(Fraction.from_float, state))
        g = tuple(map(Fraction.from_float, direction))
        oracle = np.asarray(list(map(float, derivative_gaussian_flux(rational, g))))
        symbolic_builder = fan_li15_expressions(state).directional_flux(direction)
        np.testing.assert_allclose(result["flux"], oracle, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(result["flux"], symbolic_builder, rtol=3e-13, atol=3e-13)


@pytest.mark.compiler
def test_rusanov_interface_matches_path_bound_and_separate_two_cell_balances(path_bridge):
    left, right = states()
    right = tuple(3 * x for x in right)
    direction = 0.6, -0.8
    result = path_bridge.interface(left, right, direction)
    assert result["status"] == 0
    fluxes = [fan_li15_expressions(raw).directional_flux(direction) for raw in (left, right)]
    bound = max(
        fan_li15_expressions(raw).fixed_direction_path_speed_majorant(direction)
        for raw in (left, right)
    )
    assert result["bound"] == pytest.approx(bound, rel=3e-15)
    flux = (np.asarray(fluxes[0]) + fluxes[1] - bound * (np.asarray(right) - left)) / 2
    path = np.asarray(list(map(float, quadrature_path(left, right, direction))))
    np.testing.assert_allclose(result["flux"], flux, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result["left_ncp"], -path / 2, rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(result["left_ncp"], result["right_ncp"])
    for k in range(15):
        if k not in TOP:
            assert result["left_ncp"][k] == result["right_ncp"][k] == 0
    area, volume_left, volume_right = 0.7, 2.0, 3.0
    residual_left = area / volume_left * (-result["flux"] + result["left_ncp"])
    residual_right = area / volume_right * (result["flux"] + result["right_ncp"])
    total = volume_left * residual_left + volume_right * residual_right
    np.testing.assert_allclose(total, -area * path, rtol=5e-13, atol=5e-13)
    assert np.max(abs(total[list(TOP)])) > 1e-3  # Top rows are not spuriously conserved.
    # Explicit physical invariant rows: mass, both momenta and kinetic energy.
    np.testing.assert_allclose(total[[0, 1, 5]], 0, atol=5e-14)
    assert (total[2] + total[9]) / 2 == pytest.approx(0, abs=5e-14)


@pytest.mark.compiler
def test_reversed_face_and_metric_density_scaling_keep_distinct_side_signs(path_bridge):
    left, right = states()
    direction = 0.375, -0.625
    base = path_bridge.interface(left, right, direction)
    reverse = path_bridge.interface(right, left, tuple(-x for x in direction))
    assert base["status"] == reverse["status"] == 0
    np.testing.assert_array_equal(reverse["flux"], -base["flux"])
    np.testing.assert_array_equal(reverse["left_ncp"], base["right_ncp"])
    np.testing.assert_array_equal(reverse["right_ncp"], base["left_ncp"])
    assert reverse["bound"] == base["bound"]
    for scale in (2.0**-80, 8.0, 2.0**80):
        scaled = path_bridge.interface(
            tuple(scale * x for x in left), tuple(scale * x for x in right), direction
        )
        assert scaled["status"] == 0
        assert scaled["bound"] == base["bound"]
        for key in ("flux", "left_ncp", "right_ncp"):
            np.testing.assert_array_equal(scaled[key], scale * base[key])


@pytest.mark.compiler
def test_interface_consistency_zero_direction_and_fail_closed_outputs(path_bridge):
    left, right = states()
    same = path_bridge.interface(left, left, (0.6, -0.8))
    physical = path_bridge.flux(left, (0.6, -0.8))
    assert same["status"] == physical["status"] == 0
    np.testing.assert_array_equal(same["flux"], physical["flux"])
    np.testing.assert_array_equal(same["left_ncp"], np.zeros(15))
    zero = path_bridge.interface(left, right, (0, 0))
    assert zero["status"] == 0 and zero["bound"] == 0
    for key in ("flux", "left_ncp", "right_ncp"):
        np.testing.assert_array_equal(zero[key], np.zeros(15))
    invalid = list(left)
    invalid[2] = 0
    for a, b, direction in (
        (invalid, right, (1, 0)),
        (right, invalid, (1, 0)),
        (invalid, right, (0, 0)),
        (left, right, (float("nan"), 0)),
    ):
        refused = path_bridge.interface(a, b, direction)
        assert refused["status"] != 0
        for key in ("flux", "left_ncp", "right_ncp"):
            np.testing.assert_array_equal(refused[key], np.zeros(15))
    overflowing = list(left)
    overflowing[4] = 1e308
    refused = path_bridge.flux(overflowing, (10, 0))
    assert refused["status"] == 7
    np.testing.assert_array_equal(refused["flux"], np.zeros(15))
