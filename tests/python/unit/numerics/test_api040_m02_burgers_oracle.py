"""Pure M02 mathematics, including the actual authored Python face function."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCIENTIFIC = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
SPEC = importlib.util.spec_from_file_location(
    "m02_burgers_oracle", SCIENTIFIC / "api040_m02_burgers_oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def authored_face():
    # Execute exactly the shipped body's arithmetic without importing PoPS,
    # compiling a kernel, or executing the example's native run loop.
    path = SCIENTIFIC / "api040_m02_burgers.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "godunov_face")
    namespace = {"where": lambda condition, yes, no: yes() if condition else no()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["godunov_face"]


@pytest.mark.parametrize("left,right", [(1., 0.), (0., 1.), (-1., 1.),
                                       (1., -1.), (-2., -.3), (-.3, -2.),
                                       (2., -1.), (1., -2.)])
def test_authored_face_matches_convex_entropy_extremum_in_every_sign_region(left, right):
    result = authored_face()((left,), (right,), (.5 * left**2,), (.5 * right**2,),
                             max(abs(left), abs(right)))
    assert isinstance(result, tuple) and len(result) == 1
    assert result[0] == pytest.approx(oracle.godunov_flux(left, right), abs=1.e-15)
    # Canonical face orientation matters: reverse states without changing the
    # normal and a transsonic rarefaction becomes an entropy shock.
    if left == -1 and right == 1:
        assert result[0] == 0
        assert oracle.godunov_flux(right, left) == .5


@pytest.mark.parametrize("value", [-3., -.4, 0., .7, 2.])
def test_consistency_reflection_and_axis_with_zero_physical_flux(value):
    body = authored_face()
    assert body((value,), (value,), (.5 * value**2,), (.5 * value**2,), abs(value))[0] \
        == pytest.approx(.5 * value**2)
    # The x reflection maps u(x) to -u(-x), with an even physical flux.
    left, right = value, value + .6
    assert oracle.godunov_flux(left, right) == oracle.godunov_flux(-right, -left)
    assert body((left,), (right,), (0.,), (0.,), 0.) == (0.,)


@pytest.mark.parametrize("left,right", [(1., 0.), (0., 1.), (-1., 1.), (1., -1.)])
@pytest.mark.parametrize("n", [5, 100, 200, 400])
def test_exact_cell_mass_balances_far_field_flux(left, right, n):
    initial = oracle.entropy_cell_averages(n, 0., left, right)
    final = oracle.entropy_cell_averages(n, .2, left, right)
    dx = 2. / n
    np.testing.assert_allclose(dx * np.sum(final - initial),
                               .2 * (.5 * left**2 - .5 * right**2),
                               rtol=0., atol=5.e-15)
    assert np.all(final >= min(left, right))
    assert np.all(final <= max(left, right))


def test_shock_and_fan_cuts_are_integrated_not_sampled_at_centers():
    shock = oracle.entropy_cell_averages(5, .2, 1., 0.)
    fan = oracle.entropy_cell_averages(5, .2, 0., 1.)
    # Central cell [-.2,.2]: shock at .1 gives mean .75; fan [0,.2]
    # gives an integral .1, hence mean .25. Point samples give 1 and 0.
    assert shock[2] == pytest.approx(.75, abs=1.e-15)
    assert fan[2] == pytest.approx(.25, abs=1.e-15)
    assert oracle.weak_shock_speed(1., 0.) == .5
    np.testing.assert_allclose(oracle.entropy_cell_averages(5, 0., 1., 0.),
                               [1., 1., .5, 0., 0.], rtol=0., atol=1.e-15)


@pytest.mark.parametrize("left,right", [(1., 0.), (0., 1.)])
def test_first_godunov_step_equals_entropy_averages_and_rejects_rusanov(left, right):
    n = 100
    dx = 2. / n
    dt = .01 * dx
    initial = oracle.entropy_cell_averages(n, 0., left, right)
    face_left = np.r_[initial[0], initial]
    face_right = np.r_[initial, initial[-1]]
    flux = oracle.godunov_flux(face_left, face_right)
    updated = initial - dt / dx * np.diff(flux)
    exact = oracle.entropy_cell_averages(n, dt, left, right)
    np.testing.assert_allclose(updated, exact, rtol=0., atol=2.e-14)
    rusanov = .25 * (face_left**2 + face_right**2) - .5 * np.maximum(
        abs(face_left), abs(face_right)) * (face_right - face_left)
    wrong = initial - dt / dx * np.diff(rusanov)
    assert np.max(abs(wrong - exact)) > .002
    # One shared face value telescopes to the boundary flux, exactly the
    # conservative update that the production finite-volume action must use.
    assert dx * sum(updated - initial) == pytest.approx(dt * (flux[0] - flux[-1]), abs=1.e-15)


def test_nonlinear_coordinate_change_does_not_preserve_the_weak_law_automatically():
    # w=u^2/2 on u>=0. The correct weak law is (sqrt(2w))_t+w_x=0.
    # Its smooth rewrite w_t+((2w)^(3/2)/3)_x=0 conserves w instead of u
    # and selects a different shock. Smooth equivalence is insufficient.
    ul, ur = 1., 0.
    wl, wr = .5 * ul**2, .5 * ur**2
    correct_speed = (wl - wr) / (ul - ur)
    wrong_speed = (ul**3 / 3 - ur**3 / 3) / (wl - wr)
    assert correct_speed == .5
    assert wrong_speed == pytest.approx(2. / 3)
    assert wrong_speed * (ul - ur) - (wl - wr) == pytest.approx(1. / 6)
    mean_u = oracle.entropy_cell_averages(1, .2, ul, ur)[0]
    mean_w = .55 * wl + .45 * wr
    assert mean_w != pytest.approx(.5 * mean_u**2)
