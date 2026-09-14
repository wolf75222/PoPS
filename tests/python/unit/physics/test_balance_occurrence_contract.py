"""Signed physical equations remain authoritative across numerical partitioning."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from pops.math import Accumulation, accumulation, ddt, div
from tests.python.unit.physics.test_rate_equation_contract import _scalar_advection_model
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS


def test_partition_preserves_sign_scale_target_identity_and_repeated_source_occurrences():
    model, state, flux = _scalar_advection_model()
    source = model.source("forcing", on=state, value=[state[0]])
    rate = model.rate("balance", equation=ddt(state) == (
        -div(flux) + Fraction(2, 3) * source - source + source))

    selected = rate.select(source)

    assert selected.balance is rate.balance
    assert selected.view.ordinals == (1, 2, 3)
    assert [item.coefficient for item in selected.occurrences] == [Fraction(2, 3), -1, 1]
    assert all(item.target is state for item in selected.occurrences)
    assert all(left is right for left, right in zip(
        selected.occurrences, rate.occurrences[1:], strict=True))
    assert selected.select(source) is selected
    assert rate.select(source, source) is selected
    assert selected.select(selected.occurrences[1]).occurrences == (rate.occurrences[2],)
    with pytest.raises(ValueError, match="absent"):
        selected.select(flux)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        selected.occurrences[0].coefficient = 0


def test_selecting_repeated_flux_preserves_each_occurrence():
    model, state, flux = _scalar_advection_model()
    source = model.source("forcing", on=state, value=[state[0]])
    rate = model.rate("balance", equation=ddt(state) == -div(flux) + div(flux) + source)
    selected = rate.select(flux)
    assert selected.view.ordinals == (0, 1)
    assert [item.coefficient for item in selected.occurrences] == [-1, 1]
    assert selected.occurrences[0].identity != selected.occurrences[1].identity
    assert model.selected_rate_contracts() == {rate: model.rate_contract(rate)}


def test_alternative_balances_require_explicit_selection_but_partitions_do_not():
    model, state, flux = _scalar_advection_model()
    source = model.source("forcing", on=state, value=[state[0]])
    first = model.rate("advection", equation=ddt(state) == -div(flux) + source)
    first.select(flux)
    assert model.validate_balance_selection()
    second = model.rate("alternative", equation=ddt(state) == source)
    with pytest.raises(ValueError, match="select_balance"):
        model.validate_balance_selection()
    assert model.select_balance(second) is second
    assert model.selected_rate_contracts() == {second: model.rate_contract(second)}
    with pytest.raises(ValueError, match="complete physical balance"):
        model.select_balance(first.select(flux))


def test_default_accumulation_is_identity_and_nonlinear_law_retains_discrete_contract():
    model, state, flux = _scalar_advection_model()
    identity_rate = model.rate("identity", equation=ddt(state) == -div(flux))
    assert identity_rate.balance.accumulation.is_identity
    q = state[0]
    law = q * q
    mapping = accumulation(state, coordinates=q, law=law,
                           representation={"input": "point_value", "output": "cell_average"},
                           quadrature={"rule": "gauss", "order": 2})
    rate = model.rate("nonlinear", equation=ddt(mapping) == -div(flux))
    assert rate.balance.accumulation is mapping
    assert mapping.law is law
    residual = mapping.temporal_residual(q + 1, state[0], interval=Fraction(1, 10), rate=rate)
    assert residual.accumulation is mapping
    assert residual.previous is state[0]
    assert residual.interval == Fraction(1, 10)
    assert "discrete temporal solve" in rate.view.legacy_incompatibility()
    with pytest.raises(TypeError):
        mapping.representation["input"] = "cell_average"
    with pytest.raises(ValueError, match="representation and quadrature"):
        Accumulation(state, coordinates=q, law=law)


def test_unsupported_balance_is_registered_as_scientific_ir_without_dummy_computation():
    model, state, flux = _scalar_advection_model()
    rate = model.rate("scaled", equation=ddt(state) == Fraction(3, 2) * div(flux))
    module = model.module
    model._install_retained_rates(module)
    lowering = module.operator_registry().get(rate.local_id).lowering
    assert lowering["physical_balance"] is model.balance_contract(rate)
    assert lowering["native_unsupported"]["code"] == "unsupported_balance_realization"
    assert lowering["native_unsupported"]["phase"] == "resolve"
    assert "flux" not in lowering
    assert "sources" not in lowering


def test_foreign_flux_and_source_do_not_gain_balance_ownership_from_names():
    model, state, _flux = _scalar_advection_model()
    foreign, other_state, other_flux = _scalar_advection_model()
    other_source = foreign.source("forcing", on=other_state, value=[other_state[0]])
    with pytest.raises(ValueError, match="declared by this physics model"):
        model.rate("foreign_flux", equation=ddt(state) == div(other_flux))
    with pytest.raises(ValueError, match="declared by this physics model"):
        model.rate("foreign_source", equation=ddt(state) == other_source)


def test_two_physical_fluxes_preserve_default_and_signed_balance_with_named_route():
    model, state, transport = _scalar_advection_model()
    old_default = model._dsl._m._flux
    diffusion = model.flux("diffusive flux", frame=FRAME, state=state,
                           components={X_AXIS: [2 * state[0]], Y_AXIS: [3 * state[0]]})
    source = model.source("source", on=state, value=[state[0]])
    balance = model.rate("balance", equation=ddt(state) == -div(transport) + div(diffusion) + source)
    assert transport.is_default
    assert not diffusion.is_default
    assert model._dsl._m._flux is old_default
    assert [item.coefficient for item in balance.occurrences] == [-1, 1, 1]
    assert balance.select(diffusion).occurrences == (balance.occurrences[1],)
    module = model.module
    assert module.operator_binding(transport).registered_operator_name == "flux_default"
    assert module.operator_binding(diffusion).registered_operator_name == diffusion.reg_name
    named_advection = model.rate("named_advection", equation=ddt(state) == -div(diffusion))
    assert model._dsl._m._rate_operators[named_advection.local_id]["fluxes"] == [diffusion.reg_name]
    assert module.operator_registry().get(balance.local_id).lowering["physical_balance"] is balance.view


def test_additional_flux_cannot_silently_replace_default_wave_speed_law():
    model, state, _transport = _scalar_advection_model()
    before = dict(model._dsl._m._flux_terms)
    with pytest.raises(ValueError, match="cannot replace the default wave-speed law"):
        model.flux("faster", frame=FRAME, state=state,
                   components={X_AXIS: [2 * state[0]], Y_AXIS: [2 * state[0]]},
                   waves={X_AXIS: [2], Y_AXIS: [2]})
    assert "faster" not in model.fluxes
    assert model._dsl._m._flux_terms == before


def test_qualified_first_species_keeps_its_complete_type_through_promotion():
    from pops._ir.quantity import QuantityRef
    from pops.physics import Model
    model = Model("promotion", frame=FRAME)
    first = model.species("first", state=["density"])
    old_leaf = first[0]
    model.species("second", state=["density"])
    assert isinstance(old_leaf, QuantityRef)
    promoted_space = model._multi_module.state_spaces()[first.name]
    assert old_leaf.handle == model._multi_module.state_handle(promoted_space)
    assert old_leaf.space == promoted_space
    model.local_transform("repair", [old_leaf + 1], on=first)


def test_named_single_state_balance_and_primitive_coordinates_use_exact_alias():
    from pops._ir.quantity import QuantityRef
    from pops.physics import Model
    model = Model("named", frame=FRAME)
    state = model.state("tracer", components=["rho", "mx"])
    density, momentum = state
    assert isinstance(density, QuantityRef)
    velocity = model.primitive("velocity", momentum / density)
    model.primitive_state(density, velocity, conservative=[density, density * velocity])
    source = model.source("forcing", on=state, value=[density, momentum])
    rate = model.rate("balance", equation=ddt(state) == source)
    assert rate.balance.target is state
    assert model._dsl._m.cons_from[0] is density
    assert model.check()
