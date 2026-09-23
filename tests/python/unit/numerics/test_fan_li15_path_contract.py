"""Authenticate both physical Fan–Li terms before selecting the analytic path."""
import pytest

from pops.math import ddt, div
from pops.moments.fan_li import fan_li15_expressions, FAN_LI15_REGULARIZED_COMPONENTS
from pops.moments.model_builder import moment_names
from pops.numerics import (
    FanLi15RawMomentPath, PathConservativeFiniteVolume, reconstruction, riemann, variables,
)
from pops.physics import Model
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS


def _declarations(*, flux_scale=1, product_scale=1):
    model = Model("Fan Li path contract", frame=FRAME)
    state = model.state("U", components=tuple(moment_names(4)))
    physical = fan_li15_expressions(state)
    covectors = {X_AXIS: (1, 0), Y_AXIS: (0, 1)}
    flux = model.flux("Grad flux", frame=FRAME, state=state,
        components={axis: tuple(value if flux_scale == 1 else flux_scale * value
                                for value in physical.directional_flux(g))
                    for axis, g in covectors.items()})
    product = model.nonconservative_product("Fan Li regularization", state=state,
        matrices={axis: tuple(tuple(value if product_scale == 1 else product_scale * value
                                    for value in row)
                              for row in physical.directional_nonconservative_matrix(g))
                  for axis, g in covectors.items()},
        conservative_components=tuple(name for slot, name in enumerate(moment_names(4))
                                       if slot not in FAN_LI15_REGULARIZED_COMPONENTS))
    return model, state, flux, product, covectors


def _method(state, flux, path, *, reconstruct=None, variable=None):
    return PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state) if variable is None else variable,
        reconstruction=reconstruction.FirstOrder() if reconstruct is None else reconstruct,
        riemann=riemann.Rusanov())


def test_exact_grad_and_product_define_one_path_method():
    model, state, flux, product, covectors = _declarations()
    rate = model.rate("transport", equation=ddt(state) == -div(flux) - product)
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    method = _method(state, flux, path)
    assert method.validate_rate_contract(model.rate_contract(rate))
    assert method.validate_balance_view(rate.view)
    assert method.formal_order == 1
    assert path.to_data()["face_geometry"] == "arithmetic_trace_covector"


def test_modified_product_cannot_reuse_the_analytic_integral():
    model, _, _, product, covectors = _declarations(product_scale=2)
    with pytest.raises(ValueError, match="does not match the declared physical product"):
        FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)


def test_modified_flux_cannot_reuse_the_full_system_speed_proof():
    model, state, flux, product, covectors = _declarations(flux_scale=2)
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    with pytest.raises(ValueError, match="complete declared Grad flux plus B"):
        _method(state, flux, path)


def test_higher_order_reconstruction_needs_its_missing_cell_path_integral():
    model, state, flux, product, covectors = _declarations()
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    with pytest.raises(ValueError, match="FirstOrder and Rusanov"):
        _method(state, flux, path, reconstruct=reconstruction.MUSCL())


def test_product_cannot_be_selected_as_an_unrelated_source_or_with_wrong_sign():
    model, state, flux, product, covectors = _declarations()
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    method = _method(state, flux, path)
    wrong = model.rate("wrong sign", equation=ddt(state) == -div(flux) + product)
    with pytest.raises(ValueError, match="exactly -div"):
        method.validate_balance_view(wrong.view)


def test_resolved_numerical_identity_retains_the_separate_product_and_path():
    import pops
    from pops.numerics import DiscretizationPlan
    model, state, flux, product, covectors = _declarations()
    rate = model.rate("transport", equation=ddt(state) == -div(flux) - product)
    path = FanLi15RawMomentPath(product, frame=FRAME, covectors=covectors)
    method = _method(state, flux, path)
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("Fan Li path identity")
    block = case.block("plasma", model)
    resolved = plan.resolve_for(case, block)
    data = resolved.rates[0].method.to_data()
    assert data["method"] == "path_conservative_finite_volume"
    assert data["path"]["kind"] == "fan_li15_straight_raw_moment_path"
    assert data["path"]["product"]["kind"] == "nonconservative_product"
    assert data["nonconservative_interfaces"] == "canonical_fine_subface_side_contributions"
