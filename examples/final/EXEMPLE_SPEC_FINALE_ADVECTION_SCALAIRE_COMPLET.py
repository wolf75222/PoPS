"""Final PoPS target: conservative scalar advection with explicit RK2 and AMR.

This is the executable acceptance target for the operator-first Python interface.  Physics,
numerics, time, mesh adaptation, consumers and execution controls each have one authority.  The
module deliberately has no compatibility aliases and no lower-level substitute for a missing public
hook: an unavailable join must fail where it is authored, not silently change the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pops
from pops.domain import Rectangle, RectangleBoundaryNames
from pops.frames import Cartesian2D
from pops.ir import ValueExpr, ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import Interval, Positive, RuntimeParam
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, StagePoint, TimePoint


OUTPUT_ROOT = Path("outputs/scalar_advection")


@dataclass(frozen=True, slots=True)
class ScalarAdvectionAuthoring:
    """The inert declarations shared by validation, resolution, bind and run."""

    domain: Any
    frame: Any
    grid: Any
    model: Any
    state: Any
    scalar: Any
    velocity: Any
    flux: Any
    rate: Any
    finite_volume: Any
    numerics: Any
    case: Any
    tracer: Any
    tracer_state: Any
    program: Any
    velocity_x_param: Any
    velocity_y_param: Any
    inlet_x_param: Any
    inlet_y_param: Any
    inlet_x_value: Any
    inlet_y_value: Any
    refine_threshold: Any
    coarsen_threshold: Any
    run_controls: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FinalScalarAdvectionCase:
    """The complete Case and adaptive layout consumed by validation/resolution."""

    authoring: ScalarAdvectionAuthoring
    layout: Any


def build_authoring() -> ScalarAdvectionAuthoring:
    """Build the pure operator-first declarations without importing native code."""

    domain = Rectangle(
        "unit_square",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        boundaries=RectangleBoundaryNames(
            x_min="inlet_x",
            x_max="outlet_x",
            y_min="inlet_y",
            y_max="outlet_y",
        ),
    ).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    grid = CartesianGrid(frame=frame, cells=(128, 128))

    # Physics: U is stored conservatively and F is the explicit physical flux F(U) = a U.
    model = pops.Model("scalar_advection", frame=frame)
    state = model.state(
        "U",
        components=("u",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (u,) = state

    velocity_x_param = model.param(
        RuntimeParam("a_x", default=1.0, domain=Positive())
    )
    velocity_y_param = model.param(
        RuntimeParam("a_y", default=0.25, domain=Positive())
    )
    inlet_x_param = model.param(
        RuntimeParam("u_in_x", default=0.0, domain=Interval(-10.0, 10.0))
    )
    inlet_y_param = model.param(
        RuntimeParam("u_in_y", default=0.0, domain=Interval(-10.0, 10.0))
    )

    # Handles remain stable identities.  Only explicit value reads enter symbolic algebra.
    a_x = model.value(velocity_x_param)
    a_y = model.value(velocity_y_param)
    u_in_x = model.value(inlet_x_param)
    u_in_y = model.value(inlet_y_param)

    velocity = model.vector(
        "a",
        frame=frame,
        components={x_axis: a_x, y_axis: a_y},
    )
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (a_x * u,), y_axis: (a_y * u,)},
        waves={x_axis: (a_x,), y_axis: (a_y,)},
    )
    rate = model.rate(
        "advection_rate",
        equation=ddt(state) == -div(flux),
    )

    # Numerics: formal order, halo depth and the CFL provider are properties of these bricks.
    finite_volume = FiniteVolume(
        flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    )
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, finite_volume)

    case = pops.Case("tutorial_scalar_advection_rk2_amr")
    tracer = case.block("tracer", model=model)
    tracer_state = tracer[state]

    # AMR thresholds belong to the Case because they configure this model instance, not its physics.
    refine_threshold = case.param(
        RuntimeParam("refine_u_gradient", default=0.10, domain=Positive())
    )
    coarsen_threshold = case.param(
        RuntimeParam("coarsen_u_gradient", default=0.04, domain=Positive())
    )

    # Explicit SSPRK2/Heun program.  The qualified state is the sole block/model authority.  A
    # pops.lib.time.SSPRK2 preset may expand to this graph; it never replaces the generic spelling.
    program = pops.Program("rk2_heun_advection")
    q = program.state(tracer_state)
    predictor = StagePoint(
        "predictor",
        {"explicit": TimePoint(program.clock, 1)},
    )
    k0 = rate(q.n)
    q_stage = program.value(
        "q_stage",
        q.n + program.dt * k0,
        at=predictor,
    )
    k1 = rate(q_stage)
    q_next = program.value(
        "q_next",
        Fraction(1, 2) * q.n
        + Fraction(1, 2) * q_stage
        + Fraction(1, 2) * program.dt * k1,
        at=q.next.point,
    )
    program.commit(q.next, q_next)
    program.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))

    # Run controls do not select physics, a spatial method, a time method or a CFL strategy.
    run_controls = {
        "t_end": 1.0,
        "max_steps": 100_000,
        "output_dir": OUTPUT_ROOT,
    }

    return ScalarAdvectionAuthoring(
        domain=domain,
        frame=frame,
        grid=grid,
        model=model,
        state=state,
        scalar=u,
        velocity=velocity,
        flux=flux,
        rate=rate,
        finite_volume=finite_volume,
        numerics=numerics,
        case=case,
        tracer=tracer,
        tracer_state=tracer_state,
        program=program,
        velocity_x_param=velocity_x_param,
        velocity_y_param=velocity_y_param,
        inlet_x_param=inlet_x_param,
        inlet_y_param=inlet_y_param,
        inlet_x_value=u_in_x,
        inlet_y_value=u_in_y,
        refine_threshold=refine_threshold,
        coarsen_threshold=coarsen_threshold,
        run_controls=run_controls,
    )


def build_transport_boundaries(core: ScalarAdvectionAuthoring) -> Any:
    """Build the single boundary authority consumed by ``DiscretizationPlan``."""

    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow

    boundaries = core.frame.boundaries
    return TransportBoundarySet(
        {
            boundaries.x_min: Inflow(
                state=core.tracer_state,
                value=core.inlet_x_value,
            ),
            boundaries.x_max: Outflow(state=core.tracer_state),
            boundaries.y_min: Inflow(
                state=core.tracer_state,
                value=core.inlet_y_value,
            ),
            boundaries.y_max: Outflow(state=core.tracer_state),
        }
    )


def build_amr_layout(core: ScalarAdvectionAuthoring) -> Any:
    """Build one AMR layout that owns hierarchy, tagging, transfer and execution semantics."""

    from pops.amr import (
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Coarsen,
        PriorityOrder,
        Tag,
    )
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import grad, norm
    from pops.time import every

    # The explicit Handle -> Expr conversion preserves Python identity semantics.  AMRTagging binds
    # this continuous-looking predicate to the resolved discrete gradient/stencil context of U.
    tracer_value = ValueExpr(core.tracer_state)
    gradient_magnitude = norm(grad(tracer_value))
    tagging = AMRTagging(
        rules=(
            Tag(gradient_magnitude > core.case.value(core.refine_threshold)),
            Coarsen(gradient_magnitude < core.case.value(core.coarsen_threshold)),
            Buffer(cells=2),
        ),
        combine=PriorityOrder(),
    )

    # Transfer accuracy/halos come from StateTransfer's installed policies; order is never repeated.
    transfer = AMRTransfer()
    transfer.state(core.tracer_state, StateTransfer())

    return AMR(
        grid=core.grid,
        hierarchy=AMRHierarchy(max_levels=3, ratios=(2, 2)),
        tagging=tagging,
        regrid=AMRRegrid(schedule=every(5, clock=core.program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled(),
    )


def build_initial_condition(core: ScalarAdvectionAuthoring) -> Any:
    """Build analytic data and its explicit conservative projection."""

    from pops.initial import InitialCondition
    from pops.lib.initial import Gaussian
    from pops.projection import ConservativeCellAverage

    gaussian = Gaussian(
        frame=core.frame,
        center={core.frame.x: 0.30, core.frame.y: 0.35},
        background=0.05,
        amplitude=0.95,
        inverse_width=120.0,
    )
    return InitialCondition(
        state=core.tracer_state,
        value=gaussian,
        projection=ConservativeCellAverage(),
    )


def build_consumer_graph(core: ScalarAdvectionAuthoring) -> Any:
    """Build the sole accepted-side-effect graph for diagnostics, output and checkpointing."""

    from pops.diagnostics import Integral
    from pops.output import Checkpoint, HDF5, ParaView, ScientificOutput
    from pops.runtime import ConsumerGraph
    from pops.time import every

    tracer_mass = Integral(block=core.tracer, cadence=every(10, clock=core.program.clock))
    consumers = (
        ScientificOutput(
            format=ParaView(),
            schedule=every(10, clock=core.program.clock),
            fields=(core.tracer_state,),
            diagnostics=(tracer_mass,),
            target="solution/tracer",
        ),
        ScientificOutput(
            format=HDF5(parallel=True),
            schedule=every(50, clock=core.program.clock),
            fields=(core.tracer_state,),
            target="state/tracer",
        ),
        Checkpoint(
            schedule=every(100, clock=core.program.clock),
            bit_identical=True,
            target="checkpoints/restart",
        ),
    )
    return ConsumerGraph.from_consumers(consumers)


def build_final_case() -> FinalScalarAdvectionCase:
    """Assemble every public authority exactly once."""

    core = build_authoring()
    core.numerics.boundaries.add(build_transport_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.initials.add(build_initial_condition(core))
    core.case.program(core.program)
    core.case.consumers(build_consumer_graph(core))

    layout = build_amr_layout(core)
    return FinalScalarAdvectionCase(core, layout)


def build_bind_inputs(core: ScalarAdvectionAuthoring) -> Any:
    """Build bind values only after validation has made every Handle canonical."""

    from pops.codegen import BindInputs

    resolve = core.case.resolve
    return BindInputs(
        params={
            resolve(core.velocity_x_param): 1.0,
            resolve(core.velocity_y_param): 0.25,
            resolve(core.inlet_x_param): 0.0,
            resolve(core.inlet_y_param): 0.0,
            resolve(core.refine_threshold): 0.10,
            resolve(core.coarsen_threshold): 0.04,
        }
    )


def main() -> None:
    """Run the one final lifecycle: Case -> validate -> resolve -> compile -> bind -> run."""

    from pops.codegen import Production

    target = build_final_case()
    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout, backend=Production())
    artifact = pops.compile(resolved)
    simulation = pops.bind(artifact, inputs=build_bind_inputs(target.authoring))
    simulation.run(**target.authoring.run_controls)


if __name__ == "__main__":
    main()
