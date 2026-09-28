"""Source contract for ordinary two-point diagonal semidefinite diffusion."""
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import CoeffGradient, grad
from pops.numerics import Diffusion, ScharfetterGummel
from pops.physics.diffusion import DiffusiveBoundary


def _law(diagonal):
    frame = Rectangle("semidefinite", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("semidefinite", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.diffusive_flux("D", state=state, value=CoeffGradient(
        state[0], ((diagonal[0], 0.), (0., diagonal[1]))))
    return model, state, flux


@pytest.mark.parametrize("diagonal", ((.01, 0.), (0., 0.), (0., .01)))
def test_exact_zero_axes_are_authorized_without_positive_surrogate(diagonal):
    _, _, flux = _law(diagonal)
    method = Diffusion(flux=flux)
    assert method.validate() is True
    assert tuple(value.value for axis, row in enumerate(method.law.coefficients)
                 for column, value in enumerate(row) if axis == column) == diagonal


@pytest.mark.parametrize("diagonal", ((-.01, 0.), (0., -.01)))
def test_negative_diagonal_is_still_rejected(diagonal):
    _, _, flux = _law(diagonal)
    with pytest.raises(ValueError, match="nonnegative"):
        Diffusion(flux=flux)


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_constant_diagonal_is_rejected_before_codegen(invalid):
    with pytest.raises(ValueError, match="symbolic floating metadata must be finite"):
        _law((.01, invalid))


def test_dynamic_coefficient_may_touch_zero_but_needs_native_finite_guard():
    frame = Rectangle("dynamic", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("dynamic", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.diffusive_flux("D", state=state, value=CoeffGradient(
        state[0], ((.01 * state[0], 0.), (0., 0.))))
    assert Diffusion(flux=flux).validate() is True


def test_robin_is_not_a_public_diffusive_boundary_kind():
    with pytest.raises(ValueError, match="value, conormal or periodic"):
        DiffusiveBoundary(1, "lower", "robin", 0.)


def test_fitted_scharfetter_gummel_keeps_strictly_positive_diffusivity():
    frame = Rectangle("fitted", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("fitted", frame=frame)
    state = model.state("U", components=("u",))
    potential = model.aux("potential")
    drift = model.drift_flux("drift", state=state, mobility=.1, potential=potential)
    diffusion = model.diffusive_flux("D", state=state, value=0. * grad(state))
    with pytest.raises(ValueError, match="finite D>0"):
        ScharfetterGummel(drift=drift, flux=diffusion)
