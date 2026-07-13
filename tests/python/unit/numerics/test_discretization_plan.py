"""Final family-organized numerical-plan contract."""
from __future__ import annotations

import sys

import pytest

import pops
from pops.math import ddt, div
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative


def _declarations():
    model = pops.Model("transport")
    state = model.state("U", ("u",))
    (u,) = state
    velocity = model.vector_field("a", 1, 0)
    flux = model.flux("F", on=state, x=(u,), y=(0 * u,))
    rate = model.rate("A", ddt(state) == -div(flux))
    method = FiniteVolume(
        flux=flux,
        variables=Conservative(state),
        reconstruction=MUSCL(VanLeer()),
        riemann=ScalarUpwind(velocity=velocity),
    )
    return model, state, flux, rate, method


def test_method_infers_order_ghosts_and_validates_physical_flux() -> None:
    model, _, _, rate, method = _declarations()
    assert method.formal_order == 2
    assert method.ghost_depth == 2
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    assert plan.validate_for(model)


def test_case_resolves_plan_per_owner_qualified_instance_without_native_import() -> None:
    model, state, _, rate, method = _declarations()
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("advection")
    block = case.block("tracer", model)
    case.numerics(plan, block=block)

    program = pops.Program("rk2")
    program.bind_operators(model.module)
    from pops.lib.time import SSPRK2_TABLEAU, explicit_rk

    explicit_rk(program, block, state, rhs_operator=rate, tableau=SSPRK2_TABLEAU)
    case.program(program)
    pops.validate(case)
    resolved = pops.resolve(
        case, layout=Uniform(CartesianMesh(n=8, periodic=True)))

    numerical = resolved.blocks[0].numerics
    assert numerical.block.local_id == "tracer"
    assert numerical.rates[0].rate.is_resolved
    assert resolved.blocks[0].spatial.formal_order == 2
    assert "pops._pops" not in sys.modules


def test_double_or_inconsistent_authorities_are_rejected() -> None:
    model, state, flux, rate, method = _declarations()
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    with pytest.raises(ValueError, match="already has"):
        plan.rates.add(rate, method)

    other_flux = pops.Model("other")
    other_state = other_flux.state("U", ("v",))
    (v,) = other_state
    foreign = other_flux.flux("F", on=other_state, x=(v,), y=(v,))
    with pytest.raises(ValueError, match="different Models"):
        FiniteVolume(
            flux=foreign,
            variables=Conservative(state),
            reconstruction=MUSCL(VanLeer()),
            riemann=method.riemann,
        )

    case = pops.Case("conflict")
    case.block("tracer", model, spatial=method)
    with pytest.raises(ValueError, match="already carries a spatial method"):
        case.numerics(plan)

