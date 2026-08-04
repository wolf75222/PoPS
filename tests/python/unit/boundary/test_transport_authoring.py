from __future__ import annotations

from dataclasses import replace

import pytest

import pops
from pops.boundary import TransportBoundarySet, model_primitive_to_conservative
from pops.boundary.transport import ResolvedTransportBoundarySet
from pops.boundary.transport import Inflow, NoFlux, Outflow, SlipWall
from pops.domain import Rectangle
from pops.frames import Cartesian2D, Z_AXIS
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Axial, Density, Momentum
from pops.representations import Conservative, Primitive
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
    compiled = authority.compile_boundary_data()
    inflow_values = [
        face["values"][0] for face in compiled["faces"] if face["type"] == "dirichlet"
    ]
    assert inflow_values == [
        ["handle_value", canonical_inlet.qualified_id],
        ["handle_value", canonical_inlet.qualified_id],
    ]
    from pops.model._bind_expression import expression_reference_keys

    assert expression_reference_keys(
        inflow_values[0], where="compiled inflow") == {
            ("qid", canonical_inlet.qualified_id)
        }


def test_no_flux_lowers_to_one_prepared_ghost_and_post_riemann_face_law():
    from pops.mesh.boundaries import BoundaryProviderKind, NumericalFlux
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan

    frame, _, _, _, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(TransportBoundarySet({
        boundary: NoFlux(state=block_state) for boundary in frame.boundaries.all
    }))
    case.numerics(numerics, block=block)

    authority = case._resolved_numerics_for("tracer").boundaries[0]
    assert {row.condition_type for row in authority.conditions} == {"no_flux"}
    for condition in authority.conditions:
        assert condition.values == ()
        assert condition.provider.kind is BoundaryProviderKind.NO_FLUX
        assert isinstance(condition.provider.outputs[0], NumericalFlux)
        assert condition.provider.dependencies.states == (condition.state,)

    compiled = authority.compile_boundary_data()
    runtime = authority.runtime_boundary_data({})
    assert [row["type"] for row in compiled["faces"]] == ["no_flux"] * 4
    assert [row["type"] for row in runtime["faces"]] == ["no_flux"] * 4
    assert all(row["values"] == [0.0] for row in runtime["faces"])

    detached = dict(compiled)
    detached.update({
        "ghost_plan_identity": authority.plan.canonical_id,
        "producer_order": [],
        "component_region_templates": [],
    })
    assert [
        row["type"]
        for row in CompiledBoundaryPlan(detached).runtime_boundary_data({})["faces"]
    ] == ["no_flux"] * 4

    # The immutable provider contract rejects a NumericalFlux law forged into a ghost-state family.
    foreign_provider = next(
        row.provider for row in authority.conditions
        if row.provider.kind is BoundaryProviderKind.NO_FLUX
    )
    with pytest.raises((TypeError, ValueError)):
        replace(foreign_provider, kind=BoundaryProviderKind.OUTFLOW)


def test_primitive_fixed_state_lowers_only_through_the_exact_block_model_converter():
    frame, _, _, _, numerics, case, block, block_state = _authoring()
    converter = model_primitive_to_conservative(block_state)
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Inflow(
            state=block_state,
            value=0.25,
            representation=Primitive(),
            converter=converter,
        ),
        frame.boundaries.x_max: Outflow(state=block_state),
        frame.boundaries.y_min: Inflow(state=block_state, value=0.25),
        frame.boundaries.y_max: Outflow(state=block_state),
    }))
    case.numerics(numerics, block=block)
    case.validate_report().raise_if_error()

    authority = case._resolved_numerics_for("tracer").boundaries[0]
    compiled = authority.compile_boundary_data()
    runtime = authority.runtime_boundary_data({})
    compiled_xmin = next(face for face in compiled["faces"] if face["ordinal"] == 0)
    runtime_xmin = next(face for face in runtime["faces"] if face["ordinal"] == 0)
    expected_state = authority.conditions[0].state
    expected = model_primitive_to_conservative(expected_state).qualified_id
    assert compiled_xmin["representation"] == "primitive"
    assert compiled_xmin["converter"] == expected
    assert runtime_xmin["representation"] == "primitive"
    assert runtime_xmin["converter"] == expected

    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan

    detached_compile = dict(compiled)
    detached_compile.update({
        "ghost_plan_identity": authority.plan.canonical_id,
        "producer_order": [],
        "component_region_templates": [],
    })
    detached_xmin = next(
        face for face in CompiledBoundaryPlan(detached_compile).runtime_boundary_data({})["faces"]
        if face["ordinal"] == 0
    )
    assert detached_xmin["representation"] == "primitive"
    assert detached_xmin["converter"] == expected

    from pops.model import Handle

    converted_condition = next(
        row for row in authority.conditions if row.geometry.axis.index == 0
        and row.geometry.side.value == "lower"
    )
    forged_flow = replace(
        converted_condition.provider.dependencies.representation,
        converter=Handle(
            "forged-converter",
            kind="representation_conversion",
            owner=converted_condition.state.owner_path,
        ),
    )
    forged_dependencies = replace(
        converted_condition.provider.dependencies,
        representation=forged_flow,
    )
    forged_condition = replace(
        converted_condition,
        provider=replace(converted_condition.provider, dependencies=forged_dependencies),
    )
    with pytest.raises(NotImplementedError, match="exact model_primitive_to_conservative"):
        replace(
            authority,
            conditions=tuple(
                forged_condition if row is converted_condition else row
                for row in authority.conditions
            ),
        )


def test_analytic_inflow_lowers_typed_x_time_and_bound_parameters_without_callback():
    from pops.analytic import param, time, x
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
    from pops.model import BindSchema

    frame, _, inlet, _, numerics, case, block, block_state = _authoring()
    program = pops.Program("analytic-boundary-clock")
    analytic_value = 1.0 + x(frame) + time(program.clock) + param(inlet)
    numerics.boundaries.add(
        TransportBoundarySet(
            {
                frame.boundaries.x_min: Inflow(state=block_state, value=analytic_value),
                frame.boundaries.x_max: Outflow(state=block_state),
                frame.boundaries.y_min: Inflow(state=block_state, value=0.25),
                frame.boundaries.y_max: Outflow(state=block_state),
            }
        )
    )
    case.numerics(numerics, block=block)
    authority = case._resolved_numerics_for("tracer").boundaries[0]

    analytic_condition = next(
        row
        for row in authority.conditions
        if row.geometry.axis.index == 0 and row.geometry.side.value == "lower"
    )
    assert analytic_condition.provider.dependencies.states == ()
    assert len(analytic_condition.provider.dependencies.time) == 1
    assert len(analytic_condition.provider.dependencies.runtime_params) == 1
    assert analytic_condition.values[0].frame_id == frame.canonical_id

    schema = BindSchema.from_problem(case)
    bindings = schema.resolve_bind({}, compile_values=schema.resolve_compile())
    runtime = authority.runtime_boundary_data(bindings)
    xlo = next(face for face in runtime["faces"] if face["ordinal"] == 0)
    assert xlo["values"] == [0.0]
    assert xlo["analytic_clock"] == program.clock.qualified_id
    assert xlo["analytic_programs"][0]["opcodes"] == [
        "constant",
        "x",
        "add",
        "input",
        "add",
        "constant",
        "add",
    ]
    assert xlo["analytic_programs"][0]["literals"][3] == 0.0
    assert xlo["analytic_programs"][0]["literals"][5] == 0.25

    compiled = authority.compile_boundary_data()
    compiled.update(
        {
            "ghost_plan_identity": authority.plan.canonical_id,
            "producer_order": [],
            "component_region_templates": [],
        }
    )
    detached = CompiledBoundaryPlan(compiled).runtime_boundary_data(bindings)
    assert detached["faces"] == runtime["faces"]


def test_analytic_inflow_fails_closed_for_primitive_per_point_conversion():
    from pops.analytic import x

    frame, _, _, _, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(
        TransportBoundarySet(
            {
                frame.boundaries.x_min: Inflow(
                    state=block_state,
                    value=x(frame),
                    representation=Primitive(),
                    converter=model_primitive_to_conservative(block_state),
                ),
                frame.boundaries.x_max: Outflow(state=block_state),
                frame.boundaries.y_min: Inflow(state=block_state, value=0.25),
                frame.boundaries.y_max: Outflow(state=block_state),
            }
        )
    )
    case.numerics(numerics, block=block)

    with pytest.raises(NotImplementedError, match="analytic primitive inflow"):
        case._resolved_numerics_for("tracer")


def test_analytic_inflow_fails_closed_for_discrete_setup_inputs():
    from pops.analytic import input

    frame, _, _, _, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(
        TransportBoundarySet(
            {
                frame.boundaries.x_min: Inflow(
                    state=block_state, value=input(0, "n")),
                frame.boundaries.x_max: Outflow(state=block_state),
                frame.boundaries.y_min: Inflow(state=block_state, value=0.25),
                frame.boundaries.y_max: Outflow(state=block_state),
            }
        )
    )
    case.numerics(numerics, block=block)

    with pytest.raises(NotImplementedError, match="setup-program discrete inputs"):
        case._resolved_numerics_for("tracer")


def test_analytic_inflow_fails_closed_when_one_plan_mixes_logical_clocks():
    from pops.analytic import time

    frame, _, _, _, numerics, case, block, block_state = _authoring()
    first = pops.Program("analytic-boundary-first-clock")
    second = pops.Program("analytic-boundary-second-clock")
    numerics.boundaries.add(
        TransportBoundarySet(
            {
                frame.boundaries.x_min: Inflow(
                    state=block_state, value=time(first.clock)),
                frame.boundaries.x_max: Outflow(state=block_state),
                frame.boundaries.y_min: Inflow(
                    state=block_state, value=time(second.clock)),
                frame.boundaries.y_max: Outflow(state=block_state),
            }
        )
    )
    case.numerics(numerics, block=block)

    with pytest.raises(ValueError, match="plan cannot mix several logical Clocks"):
        case._resolved_numerics_for("tracer")


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


def test_directional_characteristic_provider_cannot_fall_back_to_native_inflow():
    from pops.mesh.boundaries import (
        CharacteristicClosure,
        ClosureMode,
        DirectionalTransport,
        IncomingMultiplicity,
        SignDependence,
        SonicPolicy,
    )

    class CharacteristicInflow:
        def __init__(self, base):
            self.base = base
            self.state = base.state

        def inspect(self):
            return {**self.base.inspect(), "characteristic": "directional"}

        def resolve_references(self, resolver):
            return type(self)(self.base.resolve_references(resolver))

        def resolve_condition(self, **kwargs):
            resolved = self.base.resolve_condition(**kwargs)
            dependencies = replace(
                resolved.provider.dependencies,
                characteristic=CharacteristicClosure(
                    mode=ClosureMode.DIRECTIONAL,
                    sign_dependence=SignDependence.FIXED,
                    sonic=SonicPolicy.NEUTRAL,
                    incoming=IncomingMultiplicity.SINGLE,
                    characteristics=(resolved.state,),
                ),
            )
            return replace(
                resolved,
                provider=DirectionalTransport(
                    handle=resolved.provider.handle,
                    outputs=resolved.provider.outputs,
                    dependencies=dependencies,
                ),
            )

    frame, _, _, inlet_value, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: CharacteristicInflow(
            Inflow(state=block_state, value=inlet_value)
        ),
        frame.boundaries.x_max: Outflow(state=block_state),
        frame.boundaries.y_min: Inflow(state=block_state, value=inlet_value),
        frame.boundaries.y_max: Outflow(state=block_state),
    }))
    case.numerics(numerics, block=block)

    with pytest.raises(
        NotImplementedError,
        match="prepared model eigenstructure.*cannot fall back",
    ):
        case._resolved_numerics_for("tracer")


def test_model_characteristic_no_inflow_lowers_one_exact_prepared_face():
    from pops.boundary import model_characteristic_no_inflow

    frame, _, inlet, inlet_value, numerics, case, block, block_state = _authoring()
    provider = model_characteristic_no_inflow(block_state)
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Inflow(
            state=block_state,
            value=inlet_value,
            characteristic=provider,
        ),
        frame.boundaries.x_max: Outflow(state=block_state),
        frame.boundaries.y_min: Inflow(state=block_state, value=inlet_value),
        frame.boundaries.y_max: Outflow(state=block_state),
    }))
    case.numerics(numerics, block=block)

    authority = case._resolved_numerics_for("tracer").boundaries[0]
    compiled = authority.compile_boundary_data()
    canonical_inlet = case.resolve(inlet, block=block)
    runtime = authority.runtime_boundary_data({canonical_inlet: 0.25})
    assert compiled["faces"][0]["type"] == "characteristic_no_inflow"
    assert runtime["faces"][0]["type"] == "characteristic_no_inflow"
    assert runtime["faces"][0]["values"] == [0.25]
    assert runtime["faces"][1]["type"] == "foextrap"


def test_characteristic_no_inflow_rejects_forged_or_primitive_provider():
    from pops.boundary import model_characteristic_no_inflow
    from pops.model import Handle
    from pops.representations import Primitive

    _, _, _, inlet_value, _, _, _, block_state = _authoring()
    forged = Handle(
        "forged-characteristics",
        kind="boundary_eigenstructure",
        owner=block_state.owner_path,
    )
    with pytest.raises(ValueError, match="exact model_characteristic_no_inflow"):
        Inflow(state=block_state, value=inlet_value, characteristic=forged)
    with pytest.raises(NotImplementedError, match="conservative reference"):
        Inflow(
            state=block_state,
            value=inlet_value,
            representation=Primitive(),
            characteristic=model_characteristic_no_inflow(block_state),
        )


def test_resolved_transport_condition_rejects_a_forged_provider_law():
    from pops.mesh.boundaries import BoundaryProviderKind

    frame, _, _, inlet_value, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Inflow(state=block_state, value=inlet_value),
        frame.boundaries.x_max: Outflow(state=block_state),
        frame.boundaries.y_min: Inflow(state=block_state, value=inlet_value),
        frame.boundaries.y_max: Outflow(state=block_state),
    }))
    case.numerics(numerics, block=block)
    authority = case._resolved_numerics_for("tracer").boundaries[0]
    condition = next(
        row for row in authority.conditions if row.condition_type == "inflow")
    forged = replace(condition.provider, kind=BoundaryProviderKind.OUTFLOW)

    with pytest.raises(ValueError, match="condition 'inflow'.*provider law 'outflow'"):
        replace(condition, provider=forged)


def test_slip_wall_requires_roles_and_lowers_one_model_aware_face_law():
    frame, _, _, _, numerics, case, block, block_state = _authoring()
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Outflow(state=block_state),
        frame.boundaries.x_max: Outflow(state=block_state),
        frame.boundaries.y_min: SlipWall(state=block_state),
        frame.boundaries.y_max: Outflow(state=block_state),
    }))
    case.numerics(numerics, block=block)
    with pytest.raises(ValueError, match="declared normal polar-vector component"):
        case._resolved_numerics_for("tracer")

    domain = Rectangle("fluid_unit", (0.0, 0.0), (1.0, 1.0))
    fluid_frame = domain.frame(Cartesian2D())
    x_axis, y_axis = fluid_frame.axes
    model = pops.Model("wall_model", frame=fluid_frame)
    state = model.state(
        "U",
        components=("rho", "mx", "my", "bz"),
        representation=Conservative(),
        space=CellState(frame=fluid_frame),
        roles={
            "rho": Density(),
            "mx": Momentum(axis=x_axis),
            "my": Momentum(axis=y_axis),
            "bz": Axial(axis=Z_AXIS),
        },
    )
    rho, mx, my, bz = state
    flux = model.flux(
        "flux",
        frame=fluid_frame,
        state=state,
        components={
            x_axis: (rho, mx, my, bz),
            y_axis: (rho, mx, my, bz),
        },
        waves={
            x_axis: (1.0, 1.0, 1.0, 1.0),
            y_axis: (1.0, 1.0, 1.0, 1.0),
        },
    )
    rate = model.rate("rate", equation=ddt(state) == -div(flux))
    method = FiniteVolume(
        flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov(),
    )
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    wall_case = pops.Case("wall_case")
    wall_block = wall_case.block("fluid", model=model)
    wall_state = wall_block[state]
    plan.boundaries.add(TransportBoundarySet({
        boundary: SlipWall(state=wall_state)
        for boundary in fluid_frame.boundaries.all
    }))
    wall_case.numerics(plan, block=wall_block)

    authority = wall_case._resolved_numerics_for("fluid").boundaries[0]
    assert {row.condition_type for row in authority.conditions} == {"slip_wall"}
    runtime = authority.runtime_boundary_data({})
    assert [row["type"] for row in runtime["faces"]] == ["slip_wall"] * 4
    assert all(row["values"] == [0.0] * 4 for row in runtime["faces"])

    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan

    detached_compile_data = authority.compile_boundary_data()
    detached_compile_data.update(
        {
            "ghost_plan_identity": authority.plan.canonical_id,
            "producer_order": [],
            "component_region_templates": [],
        }
    )
    detached_runtime = CompiledBoundaryPlan(detached_compile_data).runtime_boundary_data({})
    assert detached_runtime["faces"] == runtime["faces"]
    assert detached_runtime["required_depth"] == runtime["required_depth"]
