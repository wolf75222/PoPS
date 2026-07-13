"""Final family-organized numerical-plan contract."""
from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

import pops
from pops.math import ddt, div
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative


def _declarations():
    frame = Rectangle(
        "unit-square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    velocity = model.vector(
        "a", frame=frame, components={frame.x: 1, frame.y: 0})
    flux = model.flux(
        "F", frame=frame, state=state,
        components={frame.x: (u,), frame.y: (0 * u,)},
        waves={frame.x: (1,), frame.y: (0,)},
    )
    rate = model.rate("A", equation=ddt(state) == -div(flux))
    method = FiniteVolume(
        flux=flux,
        variables=Conservative(state),
        reconstruction=MUSCL(VanLeer()),
        riemann=ScalarUpwind(velocity=velocity),
    )
    return frame, model, state, flux, rate, method


def test_method_infers_order_ghosts_and_validates_physical_flux() -> None:
    _, model, _, _, rate, method = _declarations()
    assert method.formal_order == 2
    assert method.ghost_depth == 2
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    assert plan.validate_for(model)


def test_case_resolves_plan_per_owner_qualified_instance_without_native_import() -> None:
    _, model, state, _, rate, method = _declarations()
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("advection")
    block = case.block("tracer", model)
    case.numerics(plan, block=block)

    from pops.lib.time import SSPRK2

    program = SSPRK2(block[state], rate=rate)
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
    frame, model, state, _, rate, method = _declarations()
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    with pytest.raises(ValueError, match="already has"):
        plan.rates.add(rate, method)

    other_flux = pops.Model("other", frame=frame)
    other_state = other_flux.state("U", components=("v",))
    (v,) = other_state
    foreign = other_flux.flux(
        "F", frame=frame, state=other_state,
        components={frame.x: (v,), frame.y: (v,)},
        waves={frame.x: (1,), frame.y: (1,)},
    )
    with pytest.raises(ValueError, match="different Models"):
        FiniteVolume(
            flux=foreign,
            variables=Conservative(state),
            reconstruction=MUSCL(VanLeer()),
            riemann=method.riemann,
        )

    case = pops.Case("conflict")
    case.block("tracer", model)
    case.numerics(plan)
    with pytest.raises(ValueError, match="already has a DiscretizationPlan"):
        case.numerics(plan)


@dataclass(frozen=True)
class _ReferenceAuthority:
    role: str
    reference: object

    def resolve_references(self, resolver):
        return type(self)(self.role, resolver(self.reference))

    def to_data(self):
        return {"role": self.role, "reference": self.reference.canonical_identity()}


def test_every_nonempty_family_resolves_handles_and_has_canonical_data() -> None:
    _, model, state, flux, rate, method = _declarations()
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    plan.fields.add(
        _ReferenceAuthority("field-subject", state),
        _ReferenceAuthority("field-method", state),
    )
    plan.sources.add(
        _ReferenceAuthority("source-subject", flux),
        _ReferenceAuthority("source-method", flux),
    )
    plan.boundaries.add(_ReferenceAuthority("boundary", state))
    plan.interfaces.add(_ReferenceAuthority("interface", flux))

    case = pops.Case("all-families")
    block = case.block("tracer", model)
    case.numerics(plan, block=block)
    from pops.lib.time import SSPRK2

    case.program(SSPRK2(block[state], rate=rate))
    resolved = pops.resolve(
        pops.validate(case), layout=Uniform(CartesianMesh(n=8, periodic=True)))
    data = resolved.blocks[0].numerics.to_data()

    assert [len(data[name]) for name in ("fields", "boundaries", "sources", "interfaces")] \
        == [1, 1, 1, 1]
    assert data["fields"][0]["subject"]["reference"]["kind"] == "state"
    assert data["sources"][0]["subject"]["reference"]["kind"] == "flux"
