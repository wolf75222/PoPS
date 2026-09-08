"""Constitutive diffusion remains physical IR before a method is selected."""
import pytest

import pops
from pops import math
from pops.frames import Cartesian2D
from pops._ir.expr import Partial
from pops._ir.visitors import _children, _dag_key_data


def declaration(name="heat", coefficient=0.1):
    model = pops.Model(name, frame=Cartesian2D())
    state = model.state("U", components=("u",))
    flux = model.diffusive_flux("thermal", state=state,
                                value=coefficient * math.grad(state))
    rate = model.rate("heat_rate", equation=math.ddt(state) == math.div(flux))
    return model, state, flux, rate


def test_diffusion_has_no_hyperbolic_flux_and_keeps_exact_occurrences():
    model, state, flux, rate = declaration()
    view = model.balance_contract(rate)
    assert view.target == state
    assert view.occurrences[0].kind == "diffusion"
    assert view.occurrences[0].payload is flux
    assert view.occurrences[0].coefficient == 1
    assert not model._dsl._m._flux
    module = model.module
    assert module.operator_binding(flux).kind == "expression"
    assert module.operator_registry().get("thermal").lowering["diffusive_law"] is flux.law
    assert module.module_hash() == module.module_hash()
    changed, *_ = declaration(coefficient=0.2)
    assert changed.module.module_hash() != module.module_hash()


def test_diffusion_does_not_apply_a_discrete_chain_rule():
    model = pops.Model("nonlinear", frame=Cartesian2D())
    state = model.state("U", components=("u",))
    (u,) = state
    variable = u ** 3
    flux = model.diffusive_flux("thermal", state=state, value=0.1 * math.grad(variable))
    assert flux.law.variable is variable
    roots = flux.law.flux_expressions()
    seen = []
    def visit(node):
        if isinstance(node, Partial):
            seen.append(node)
        for child in _children(node):
            visit(child)
    for root in roots:
        visit(root)
    assert seen and all(node.field is variable for node in seen)
    _dag_key_data(roots)


def test_diffusion_signed_sum_retains_repeated_uses():
    model, state, flux, _ = declaration()
    repeated = model.rate("repeated", equation=math.ddt(state) == math.div(flux) - 2 * math.div(flux))
    occurrences = model.balance_contract(repeated).occurrences
    assert [row.coefficient for row in occurrences] == [1, -2]
    assert occurrences[0].identity != occurrences[1].identity
    assert all(row.payload is flux for row in occurrences)


def test_foreign_gradient_is_refused_before_registry_changes():
    model, state, _, _ = declaration()
    other, foreign, _, _ = declaration("foreign")
    before = model.module.module_hash()
    with pytest.raises(ValueError, match="foreign"):
        model.diffusive_flux("bad", state=state, value=math.grad(tuple(foreign)[0]))
    assert model.module.module_hash() == before


def test_diffusion_requires_explicit_frame_and_scalar_target():
    model = pops.Model("unranked")
    state = model.state("U", components=("u",))
    with pytest.raises(ValueError, match="explicit physical Cartesian frame"):
        model.diffusive_flux("bad", state=state, value=math.grad(state))
    model = pops.Model("vector", frame=Cartesian2D())
    state = model.state("U", components=("u", "v"))
    with pytest.raises(ValueError, match="scalar evolved state"):
        model.diffusive_flux("bad", state=state, value=math.grad(state))


def test_program_state_brick_has_conversion_without_fabricated_transport():
    from pops.physics._facade import Model
    from pops.codegen.component_provider_packs import resolve_component_provider_packs
    from pops.codegen.module_codegen import _emit_bricks
    model = Model("storage")
    (u,) = model.conservative_vars("u")
    model.primitive_vars(u)
    model.conservative_from([u])
    source = model.module
    model.__pops_bind_component_provider_packs__(resolve_component_provider_packs(source))
    object.__setattr__(model._m, "_program_only_storage_axes", ("x", "y"))
    _, body, _ = _emit_bricks(model._m)
    assert "program_only_storage = true" in body
    assert "State flux(" not in body
    assert "max_wave_speed(" not in body
    assert "Prim to_primitive(" in body
    assert "State to_conservative(" in body
    assert not model.module.operator_registry().names()
