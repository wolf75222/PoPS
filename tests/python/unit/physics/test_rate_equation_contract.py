"""ADC-652: the explicit physical equation remains an inspectable symbolic graph."""
from __future__ import annotations

from fractions import Fraction

import pytest

from pops.ir import Divergence, Equation, RateExpr, TimeDerivative
from pops.math import ddt, div
from pops.model import OperatorHandle
from pops.physics import Model


def _scalar_advection_model():
    model = Model("scalar_advection")
    state = model.state("U", components=["u"], roles={"u": "density"})
    (u,) = state
    flux = model.flux("F", on=state, x=[u], y=[u], waves={"x": [1], "y": [1]})
    return model, state, flux


def test_ddt_equals_minus_div_builds_the_physical_equation_before_registration():
    model, state, flux = _scalar_advection_model()

    equation = ddt(state) == -div(flux)

    assert isinstance(equation, Equation)
    assert isinstance(equation.lhs, TimeDerivative)
    assert equation.lhs.state is state
    assert isinstance(equation.rhs, RateExpr)
    [(kind, referenced_flux, sign)] = equation.rhs._rate_terms()
    assert kind == "flux"
    assert referenced_flux is flux
    assert sign == -1
    assert model.inspect()["operators"] == []


def test_rate_registration_consumes_the_same_flux_handle_and_returns_an_owned_handle():
    model, state, flux = _scalar_advection_model()
    equation = ddt(state) == -div(flux)

    rate = model.rate("A", equation)

    assert isinstance(rate, OperatorHandle)
    assert rate.owner_path == model.owner_path
    assert rate.local_id == "A"
    assert "A" in model.module.list_operators()


def test_rate_rejects_a_positive_flux_divergence():
    model, state, flux = _scalar_advection_model()

    with pytest.raises(ValueError, match=r"must be -div\(F\)"):
        model.rate("wrong_sign", ddt(state) == div(flux))


@pytest.mark.parametrize("scale", [-2, Fraction(-1, 3), -0.5])
def test_rate_rejects_flux_coefficients_the_current_lowering_cannot_represent(scale):
    model, state, flux = _scalar_advection_model()

    with pytest.raises(ValueError, match=r"exact unit coefficient.*discard a scale"):
        model.rate("scaled_flux", ddt(state) == Divergence(flux, scale=scale))


def test_rate_rejects_source_coefficients_instead_of_silently_dropping_them():
    model, state, _ = _scalar_advection_model()
    (u,) = state
    source = model.source("forcing", on=state, value=[0 * u])
    scaled_source = RateExpr([("source", source, Fraction(2, 1))])

    with pytest.raises(ValueError, match=r"exact unit coefficient.*discard scale"):
        model.rate("scaled_source", ddt(state) == scaled_source)


def test_rate_rejects_multiple_divergences_instead_of_collapsing_them_to_one_bool():
    model, state, flux = _scalar_advection_model()

    with pytest.raises(ValueError, match="one -div"):
        model.rate("duplicate_flux", ddt(state) == -div(flux) - div(flux))


def test_physics_model_owner_anchor_is_read_only():
    model, _, _ = _scalar_advection_model()

    with pytest.raises(AttributeError):
        model.owner_path = model.owner_path.child("other")


@pytest.mark.parametrize("name", ["", 3, object()])
def test_physics_model_rejects_invalid_names_before_allocating_an_owner(name):
    with pytest.raises(TypeError, match="non-empty string"):
        Model(name)
