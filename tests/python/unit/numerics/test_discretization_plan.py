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
from pops.layouts import Uniform
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative
from pops.model.ownership import MissingOwnershipError


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


def _multistate_declarations():
    frame = Rectangle(
        "unit-square-multi", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = pops.Model("multi-transport", frame=frame)
    first = model.species("first", state=("u",))
    second = model.species("second", state=("v",))
    result = []
    for state, flux_name, rate_name in (
            (first, "F", "A"), (second, "G", "B")):
        (component,) = state
        flux = model.flux(
            flux_name, frame=frame, state=state,
            components={frame.x: (component,), frame.y: (0 * component,)},
            waves={frame.x: (1,), frame.y: (0,)},
        )
        rate = model.rate(rate_name, equation=ddt(state) == -div(flux))
        velocity = model.vector(
            "velocity_" + rate_name, frame=frame,
            components={frame.x: 1, frame.y: 0})
        method = FiniteVolume(
            flux=flux, variables=Conservative(state),
            reconstruction=MUSCL(VanLeer()),
            riemann=ScalarUpwind(velocity=velocity),
        )
        result.append((state, rate, method))
    return model, result[0], result[1]


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


def test_multistate_blocks_select_typed_states_and_require_exact_rate_coverage() -> None:
    model, (first, first_rate, first_method), (
        second, second_rate, second_method) = _multistate_declarations()

    case = pops.Case("selected-state")
    with pytest.raises(ValueError, match="multi-state Model"):
        case.block("ambiguous", model)
    block = case.block("first", model, states=(first,))
    assert block[first].declaration_ref == first
    with pytest.raises(MissingOwnershipError, match="not selected"):
        block[second]
    assert case.resolve(first).canonical_identity() == case.resolve(
        block[first]).canonical_identity()
    with pytest.raises(MissingOwnershipError):
        case.resolve(second)

    exact = DiscretizationPlan()
    exact.rates.add(first_rate, first_method)
    case.numerics(exact, block=block)
    assert exact.validate_for(model, states=(first,))

    overbroad = DiscretizationPlan()
    overbroad.rates.add(first_rate, first_method)
    overbroad.rates.add(second_rate, second_method)
    with pytest.raises(ValueError, match="extra=.*B"):
        overbroad.validate_for(model, states=(first,))


def test_selected_state_without_rates_needs_no_spatial_fallback() -> None:
    model = pops.Model("mixed-rate")
    evolved = model.species("evolved", state=("u",))
    passive = model.species("diagnostic", state=("marker",))
    (u,) = evolved
    zero = model.source("zero", on=evolved, value=(0 * u,))
    model.rate("source", equation=ddt(evolved) == zero)
    case = pops.Case("diagnostic-only")
    block = case.block("diagnostic", model, states=(passive,))

    assert case.resolve(passive).canonical_identity() == case.resolve(
        block[passive]).canonical_identity()
    assert pops.validate(case) is case
    with pytest.raises(MissingOwnershipError):
        case.resolve(evolved)


@dataclass(frozen=True)
class _ReferenceAuthority:
    role: str
    reference: object

    def resolve_references(self, resolver):
        return type(self)(self.role, resolver(self.reference))

    def to_data(self):
        return {"role": self.role, "reference": self.reference.canonical_identity()}


@dataclass(frozen=True)
class _BoundaryAuthority:
    reference: object

    def resolve_for_numerics(self, context):
        return _ReferenceAuthority("boundary", context.resolve(self.reference))


@dataclass(frozen=True)
class _ExternalRateMethod:
    inner: object

    def validate(self, context=None):
        return self.inner.validate(context)

    def validate_rate_contract(self, contract):
        return self.inner.validate_rate_contract(contract)

    def resolve_references(self, resolver):
        return type(self)(self.inner.resolve_references(resolver))

    def to_data(self):
        return {"extension": "external-rate-method", "inner": self.inner.to_data()}

    def freeze(self):
        self.inner.freeze()
        return self


def test_rate_family_accepts_a_small_external_method_protocol() -> None:
    _, model, state, _, rate, method = _declarations()
    plan = DiscretizationPlan()
    plan.rates.add(rate, _ExternalRateMethod(method))
    case = pops.Case("external-method")
    block = case.block("tracer", model)
    case.numerics(plan, block=block)
    from pops.lib.time import SSPRK2

    case.program(SSPRK2(block[state], rate=rate))
    resolved = pops.resolve(
        pops.validate(case), layout=Uniform(CartesianMesh(n=8, periodic=True)))

    assert resolved.blocks[0].numerics.rates[0].method.to_data()["extension"] \
        == "external-rate-method"


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
    plan.boundaries.add(_BoundaryAuthority(state))
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
