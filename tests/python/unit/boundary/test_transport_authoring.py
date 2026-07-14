from __future__ import annotations

import pytest

import pops
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import ResolvedTransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.representations import Conservative
from pops.spaces import CellState


def _authoring():
    domain = Rectangle("unit", (0.0, 0.0), (1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    model = pops.Model("transport_boundary_model", frame=frame)
    state = model.state(
        "U", components=("u",), representation=Conservative(), space=CellState(frame=frame)
    )
    (u,) = state
    speed = model.param(RuntimeParam("speed", default=1.0))
    inlet = model.param(RuntimeParam("inlet", default=0.25))
    speed_value = model.value(speed)
    inlet_value = model.value(inlet)
    velocity = model.vector(
        "velocity", frame=frame, components={frame.x: speed_value, frame.y: speed_value}
    )
    flux = model.flux(
        "flux",
        frame=frame,
        state=state,
        components={frame.x: (speed_value * u,), frame.y: (speed_value * u,)},
        waves={frame.x: (speed_value,), frame.y: (speed_value,)},
    )
    rate = model.rate("rate", equation=ddt(state) == -div(flux))
    method = FiniteVolume(
        flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    )
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("transport_boundary_case")
    block = case.block("tracer", model=model)
    block_state = block[state]
    return frame, state, inlet, inlet_value, numerics, case, block, block_state


def _complete_set(frame, state, inlet_value):
    return TransportBoundarySet({
        frame.boundaries.x_min: Inflow(state=state, value=inlet_value),
        frame.boundaries.x_max: Outflow(state=state),
        frame.boundaries.y_min: Inflow(state=state, value=inlet_value),
        frame.boundaries.y_max: Outflow(state=state),
    })


def test_transport_set_resolves_exact_ports_values_and_derived_stencil_requirements():
    frame, _, inlet, inlet_value, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(_complete_set(frame, block_state, inlet_value))
    case.numerics(numerics, block=block)

    resolved = case._resolved_numerics_for("tracer")
    assert len(resolved.boundaries) == 1
    authority = resolved.boundaries[0]
    assert isinstance(authority, ResolvedTransportBoundarySet)
    assert len(authority.conditions) == 4
    assert len(authority.plan.needs) == 4
    assert len(authority.plan.bindings) == 4
    assert {row.requirement.ghost_depth for row in authority.conditions} == {2}
    assert {row.requirement.formal_orders for row in authority.conditions} == {(2,)}

    inflows = [row for row in authority.conditions if row.condition_type == "inflow"]
    outflows = [row for row in authority.conditions if row.condition_type == "outflow"]
    assert len(inflows) == len(outflows) == 2
    canonical_inlet = case.resolve(inlet, block=block)
    for condition in inflows:
        assert condition.values[0].declaration_references() == (canonical_inlet,)
        assert condition.provider.dependencies.runtime_params == (canonical_inlet,)
        assert condition.provider.dependencies.states == ()
    for condition in outflows:
        assert condition.provider.dependencies.states == (condition.state,)
        assert condition.values == ()

    data = authority.canonical_identity()
    assert data["authority_type"] == "transport_boundary_set"
    assert {row["condition_type"] for row in data["conditions"]} == {"inflow", "outflow"}
    assert data["plan"]["plan_type"] == "boundary_providers"


def test_transport_set_rejects_incomplete_geometry_at_resolution():
    frame, _, _, inlet_value, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Inflow(state=block_state, value=inlet_value),
        frame.boundaries.x_max: Outflow(state=block_state),
        frame.boundaries.y_min: Inflow(state=block_state, value=inlet_value),
    }))
    case.numerics(numerics, block=block)

    with pytest.raises(ValueError, match="geometry coverage mismatch.*y_max"):
        case._resolved_numerics_for("tracer")


def test_transport_conditions_require_instance_handles_and_exact_component_coverage():
    frame, model_state, _, inlet_value, numerics, case, block, block_state = _authoring()
    with pytest.raises(TypeError, match="block-qualified state"):
        Inflow(state=model_state, value=inlet_value)

    numerics.boundaries.add(TransportBoundarySet({
        boundary: Inflow(state=block_state, value=(inlet_value, inlet_value))
        for boundary in frame.boundaries.all
    }))
    case.numerics(numerics, block=block)
    with pytest.raises(ValueError, match="prescribe exactly 1 components, got 2"):
        case._resolved_numerics_for("tracer")
