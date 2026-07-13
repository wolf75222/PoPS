#!/usr/bin/env python3
"""Normative final PoPS target: scalar advection-relaxation with explicit IMEX and AMR.

The example keeps every authority inspectable: the physical operators live on ``Model``; spatial
methods and boundaries live on ``DiscretizationPlan``; the additive Runge--Kutta coefficients and
stage points live in an ordinary ``Program``; the adaptive hierarchy owns tagging, transfers,
subcycling and reflux requirements; accepted-step consumers own publication and restart.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import Rectangle, RectangleBoundaryNames
from pops.fields import (
    CellCenteredSecondOrder,
    CompositeHierarchySolve,
    FieldDiscretization,
    FieldOutput,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet
from pops.frames import Cartesian2D
from pops.ir import ValueExpr, ddt, div
from pops.math import laplacian
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import Interval, Positive, RuntimeParam
from pops.representations import Conservative
from pops.solvers import DenseLU
from pops.solvers.elliptic import GeometricMG
from pops.solvers.tolerances import Relative
from pops.spaces import CellState
from pops.time import (
    AdaptiveCFL,
    AdditiveRungeKuttaTableau,
    FailRun,
    LocalLinear,
    Program,
    RejectAttempt,
    RungeKuttaTableau,
    StagePoint,
    TimePoint,
)


OUTPUT_ROOT = Path("outputs/advection_imex_amr")

# Rational CN/Heun IMEX: both partitions and every abscissa retain their exact authoring domain.
IMEX_CN_HEUN = AdditiveRungeKuttaTableau(
    RungeKuttaTableau(
        A=((), (Fraction(1),)),
        b=(Fraction(1, 2), Fraction(1, 2)),
        c=(Fraction(0), Fraction(1)),
        name="heun-explicit",
    ),
    implicit_A=((Fraction(0),), (Fraction(1, 2), Fraction(1, 2))),
    implicit_b=(Fraction(1, 2), Fraction(1, 2)),
    implicit_c=(Fraction(0), Fraction(1)),
    name="cn-heun-imex",
)


@dataclass(frozen=True, slots=True)
class IMEXAMRAuthoring:
    domain: Any
    frame: Any
    grid: Any
    model: Any
    state: Any
    scalar: Any
    velocity: Any
    flux: Any
    explicit_rate: Any
    implicit_operator: Any
    field_operator: Any
    field_provider: Any
    finite_volume: Any
    numerics: Any
    case: Any
    tracer: Any
    tracer_state: Any
    diagnostic_field: Any
    program: Any
    velocity_x: Any
    velocity_y: Any
    relaxation_rate: Any
    inlet_value: Any
    refine_value: Any
    coarsen_value: Any
    refine_gradient: Any
    run_controls: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FinalIMEXAMRCase:
    authoring: IMEXAMRAuthoring
    layout: Any


def _stage_point(program: Program, name: str, explicit: Any, implicit: Any) -> StagePoint:
    return StagePoint(
        name,
        {
            "explicit": TimePoint(program.clock, explicit),
            "implicit": TimePoint(program.clock, implicit),
        },
    )


def _manual_imex_program(core: IMEXAMRAuthoring, *, solve_action: Any) -> Program:
    """Spell the generic additive method explicitly, including both stage coordinates."""

    tableau = IMEX_CN_HEUN
    program = Program("IMEX")
    temporal = program.state(core.tracer_state)
    u0 = temporal.n
    explicit_rates: list[Any] = []
    implicit_rates: list[Any] = []
    tag = "%s_" % tableau.name

    for index in range(tableau.stages):
        point = _stage_point(
            program,
            "%sstage_%d" % (tag, index),
            tableau.explicit.c[index],
            tableau.implicit_c[index],
        )
        predictor = 1 * u0
        for previous in range(index):
            a_explicit = tableau.explicit.A[index][previous]
            a_implicit = tableau.implicit_A[index][previous]
            if a_explicit != 0:
                predictor = predictor + program.dt * a_explicit * explicit_rates[previous]
            if a_implicit != 0:
                predictor = predictor + program.dt * a_implicit * implicit_rates[previous]
        predictor = program.value(
            "%spredictor_%d" % (tag, index), predictor, at=point)

        linear = program.value(
            "%sL_%d" % (tag, index),
            core.implicit_operator(program=program),
            at=point,
        )
        stage = predictor
        diagonal = tableau.implicit_A[index][index]
        if diagonal != 0:
            stage = program.solve(
                LocalLinear(
                    operator=program.I - program.dt * diagonal * linear,
                    rhs=predictor,
                ),
                solver=DenseLU(),
                name="%sstage_solve_%d" % (tag, index),
            ).consume(action=FailRun())
            stage = program.value("%sstage_%d" % (tag, index), stage, at=point)

        # The field is solved from the actual implicit stage, never the predictor. Its exact
        # FieldContext can therefore feed the explicit rate at this same StagePoint.
        outcome = core.field_provider(stage)
        fields = outcome.consume(action=solve_action)
        fields = program.value("%sfields_%d" % (tag, index), fields, at=point)

        explicit_rates.append(program.value(
            "%sk_exp_%d" % (tag, index),
            core.explicit_rate(stage, fields),
            at=point,
        ))
        implicit_rates.append(program.value(
            "%sk_imp_%d" % (tag, index),
            program.apply(linear, stage),
            at=point,
        ))

    final = u0
    for weight, rate in zip(tableau.explicit.b, explicit_rates, strict=True):
        if weight != 0:
            final = final + program.dt * weight * rate
    for weight, rate in zip(tableau.implicit_b, implicit_rates, strict=True):
        if weight != 0:
            final = final + program.dt * weight * rate
    next_state = program.value("%sstep" % tag, final, at=temporal.next.point)
    program.commit(temporal.next, next_state)
    program.step_strategy(AdaptiveCFL(0.40))
    return program


def _preset_imex_program(core: IMEXAMRAuthoring, *, solve_action: Any) -> Program:
    """Build the library spelling of the exact same ordinary Program graph."""

    from pops.lib.time import IMEX

    program = IMEX(
        core.tracer_state,
        explicit_operator=core.explicit_rate,
        implicit_operator=core.implicit_operator,
        fields_operator=core.field_provider,
        tableau=IMEX_CN_HEUN,
        solve_action=solve_action,
    )
    program.step_strategy(AdaptiveCFL(0.40))
    return program


def build_authoring(
    *, use_preset: bool = False, field_solver: Any | None = None,
) -> IMEXAMRAuthoring:
    domain = Rectangle(
        "unit_square",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        boundaries=RectangleBoundaryNames(
            x_min="inlet_x", x_max="outlet_x", y_min="inlet_y", y_max="outlet_y"),
    ).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    grid = CartesianGrid(frame=frame, cells=(32, 32))

    model = pops.Model("scalar_advection_relaxation", frame=frame)
    state = model.state(
        "U", components=("u",), representation=Conservative(),
        space=CellState(frame=frame))
    (u,) = state
    # Positive velocity domains make the min/max inflow/outflow classification a validated physical
    # contract. Signed runtime velocities require a characteristic boundary operator that can switch
    # the incoming subspace; a static boundary table must not silently pretend to support them.
    velocity_x = model.param(RuntimeParam("a_x", default=1.0, domain=Positive()))
    velocity_y = model.param(RuntimeParam("a_y", default=0.25, domain=Positive()))
    relaxation_rate = model.param(RuntimeParam("lambda", default=50.0, domain=Positive()))
    inlet_value = model.param(RuntimeParam("u_in", default=0.0, domain=Interval(-10.0, 10.0)))
    a_x = model.value(velocity_x)
    a_y = model.value(velocity_y)

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
    relaxation_potential = model.aux("relaxation_potential")
    field_forcing = model.source(
        "field_forcing", on=state, value=(relaxation_potential,))
    explicit_rate = model.rate(
        "explicit_advection_and_field_forcing",
        equation=ddt(state) == -div(flux) + field_forcing,
    )
    implicit_operator = model.operator(
        "implicit_relaxation",
        returns=model.local_linear_operator(
            "implicit_relaxation", on=state, matrix=((-model.value(relaxation_rate),),)),
    )

    # A stage-dependent diagnostic field makes FieldContext visible without smuggling a field read
    # into the local relaxation operator. It is recomputed after regrid, never interpolated as state.
    diagnostic_unknown = model.field("relaxation_potential")
    field_operator = model.field_operator(
        "fields",
        unknown=diagnostic_unknown,
        equation=(-laplacian(diagnostic_unknown) == u),
        outputs=(FieldOutput("relaxation_potential", diagnostic_unknown),),
    )
    field_provider = next(iter(field_operator.providers)).provider

    finite_volume = FiniteVolume(
        flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    )
    numerics = DiscretizationPlan()
    numerics.rates.add(explicit_rate, finite_volume)

    case = pops.Case("scalar_advection_imex_amr")
    tracer = case.block("tracer", model=model)
    tracer_state = tracer[state]
    diagnostic_field = case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.0)),),
            solver=(field_solver if field_solver is not None else GeometricMG(
                tolerance=Relative(1.0e-6), max_cycles=100)),
            hierarchy_policy=CompositeHierarchySolve(),
        ),
    )

    refine_value = case.param(RuntimeParam("refine_u", default=0.70, domain=Positive()))
    coarsen_value = case.param(RuntimeParam("coarsen_u", default=0.25, domain=Positive()))
    refine_gradient = case.param(RuntimeParam("refine_grad_u", default=0.10, domain=Positive()))

    provisional = IMEXAMRAuthoring(
        domain=domain,
        frame=frame,
        grid=grid,
        model=model,
        state=state,
        scalar=u,
        velocity=velocity,
        flux=flux,
        explicit_rate=explicit_rate,
        implicit_operator=implicit_operator,
        field_operator=field_operator,
        field_provider=field_provider,
        finite_volume=finite_volume,
        numerics=numerics,
        case=case,
        tracer=tracer,
        tracer_state=tracer_state,
        diagnostic_field=diagnostic_field,
        program=None,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        relaxation_rate=relaxation_rate,
        inlet_value=inlet_value,
        refine_value=refine_value,
        coarsen_value=coarsen_value,
        refine_gradient=refine_gradient,
        run_controls={
            "t_end": 1.0e-4,
            "max_steps": 1,
            "output_dir": OUTPUT_ROOT,
        },
    )
    action = RejectAttempt()
    program = (
        _preset_imex_program(provisional, solve_action=action)
        if use_preset else _manual_imex_program(provisional, solve_action=action)
    )
    return replace(provisional, program=program)


def build_boundaries(core: IMEXAMRAuthoring) -> Any:
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow

    boundary = core.frame.boundaries
    inlet = core.model.value(core.inlet_value)
    return TransportBoundarySet({
        boundary.x_min: Inflow(state=core.tracer_state, value=inlet),
        boundary.x_max: Outflow(state=core.tracer_state),
        boundary.y_min: Inflow(state=core.tracer_state, value=inlet),
        boundary.y_max: Outflow(state=core.tracer_state),
    })


def build_initial(
    core: IMEXAMRAuthoring, *, background: float = 0.05, amplitude: float = 0.95,
) -> Any:
    from pops.initial import InitialCondition
    from pops.lib.initial import Gaussian
    from pops.projection import ConservativeCellAverage

    profile = Gaussian(
        frame=core.frame,
        center={core.frame.x: 0.30, core.frame.y: 0.50},
        background=background,
        amplitude=amplitude,
        inverse_width=100.0,
    )
    return InitialCondition(
        state=core.tracer_state, value=profile, projection=ConservativeCellAverage())


def build_layout(core: IMEXAMRAuthoring) -> Any:
    from pops.amr import (
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Coarsen,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        Tag,
    )
    from pops.layouts import AMR
    from pops.lib.amr import EllipticRecompute, StateTransfer
    from pops.math import grad, norm
    from pops.time import every

    value = ValueExpr(core.tracer_state)
    gradient = norm(grad(value))
    tagging = AMRTagging(
        rules=(
            Tag(value > core.case.value(core.refine_value)),
            Tag(gradient > core.case.value(core.refine_gradient)),
            Coarsen(value < core.case.value(core.coarsen_value)),
            Buffer(cells=2),
        ),
        # Equality is explicit. A non-zero temporal dwell requires a checkpointed per-cell tagging
        # state provider; this example does not pretend that an in-memory counter is restart-safe.
        hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )
    transfer = AMRTransfer()
    transfer.state(core.tracer_state, StateTransfer())
    transfer.field(core.diagnostic_field, EllipticRecompute())
    return AMR(
        grid=core.grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=tagging,
        regrid=AMRRegrid(schedule=every(4, clock=core.program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled(),
    )


def build_consumers(core: IMEXAMRAuthoring) -> Any:
    from pops.diagnostics import Integral
    from pops.output import Checkpoint, HDF5, NPZ, ParaView, ScientificOutput
    from pops.runtime import ConsumerGraph
    from pops.time import every

    fields = (core.tracer_state, core.diagnostic_field)
    diagnostic = Integral(block=core.tracer, cadence=every(1, clock=core.program.clock))
    return ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(), schedule=every(1, clock=core.program.clock), fields=fields,
            diagnostics=(diagnostic,), target="hdf5/state"),
        ScientificOutput(
            format=NPZ(), schedule=every(1, clock=core.program.clock), fields=fields,
            target="npz/state"),
        ScientificOutput(
            format=ParaView(), schedule=every(1, clock=core.program.clock), fields=fields,
            target="paraview/state"),
        Checkpoint(
            schedule=every(1, clock=core.program.clock),
            target="checkpoints/restart", bit_identical=True),
    ))


def build_final_case(
    *, use_preset: bool = False, field_solver: Any | None = None,
    initial_background: float = 0.05, initial_amplitude: float = 0.95,
) -> FinalIMEXAMRCase:
    core = build_authoring(use_preset=use_preset, field_solver=field_solver)
    core.numerics.boundaries.add(build_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.initials.add(build_initial(
        core, background=initial_background, amplitude=initial_amplitude))
    core.case.program(core.program)
    core.case.consumers(build_consumers(core))
    return FinalIMEXAMRCase(core, build_layout(core))


def build_bind_params(core: IMEXAMRAuthoring, *, inlet_value: float = 0.0) -> dict[Any, float]:
    resolve = core.case.resolve
    return {
        resolve(core.velocity_x): 1.0,
        resolve(core.velocity_y): 0.25,
        resolve(core.relaxation_rate): 50.0,
        resolve(core.inlet_value): inlet_value,
        resolve(core.refine_value): 0.70,
        resolve(core.coarsen_value): 0.25,
        resolve(core.refine_gradient): 0.10,
    }


def _existing_artifact(root: Path, suffix: str) -> Path:
    paths = tuple(sorted(root.rglob("*%s" % suffix)))
    if not paths:
        raise RuntimeError("accepted run did not publish a %s artifact under %s" % (suffix, root))
    return paths[-1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT,
        help="directory receiving accepted scientific output and restart artifacts",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()

    target = build_final_case()
    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout)
    artifact = pops.compile(resolved)
    params = build_bind_params(target.authoring)
    simulation = pops.bind(artifact, params=params)
    controls = dict(target.authoring.run_controls)
    controls["output_dir"] = output_dir
    pops.run(simulation, **controls)

    from pops.output import read_hdf5, read_paraview

    hdf5_path = _existing_artifact(output_dir, ".h5")
    paraview_path = _existing_artifact(output_dir, ".vtu")
    hdf5 = read_hdf5(hdf5_path)
    if not hdf5.arrays or not read_paraview(paraview_path).arrays:
        raise RuntimeError("published scientific artifacts reopened without arrays")
    finite = all(np.isfinite(value).all() for value in hdf5.arrays.values())

    checkpoint = Path(simulation.checkpoint(str(output_dir / "checkpoints/manual_restart")))
    if not checkpoint.is_file():
        raise RuntimeError("checkpoint API returned a missing artifact: %s" % checkpoint)
    with np.load(checkpoint, allow_pickle=False) as stored:
        restart_token = str(stored["pops_restart_identity"])
    restarted = pops.bind(artifact, params=params)
    restarted.restart(checkpoint)
    restart_equal = restarted.last_restart_identity.token == restart_token
    if not restart_equal:
        raise RuntimeError("accepted-state checkpoint did not restart bit-identically")

    print("HDF5: %s" % hdf5_path)
    print("ParaView: %s" % paraview_path)
    print("checkpoint: %s" % checkpoint)
    print("bit-identical restart: %s" % restart_equal)
    print("report: " + json.dumps({
        "artifact": artifact.artifact_identity.token,
        "checkpoint_restart_bit_identical": restart_equal,
        "finite": finite,
        "levels": resolved.resolved_hierarchy.plan.level_count,
        "program_hash": target.authoring.program.to_graph().graph_hash,
        "runtime_steps": simulation.macro_step(),
        "runtime_time": simulation.time(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
