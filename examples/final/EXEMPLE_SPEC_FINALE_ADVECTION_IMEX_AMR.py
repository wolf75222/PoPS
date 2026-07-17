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
from pops.math import ValueExpr, ddt, div, laplacian
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
    LocalLinear,
    Program,
    RejectAttempt,
    RungeKuttaTableau,
    StagePoint,
    TimePoint,
)


OUTPUT_ROOT = Path("outputs/advection_imex_amr")


def _native_output_mode() -> Any:
    """Return the shared-file topology proved by the loaded native backend."""

    from pops.output import ParallelMode
    from pops.runtime_environment import runtime_environment_report

    communicator = runtime_environment_report().get("communicator")
    if communicator == "serial":
        return ParallelMode.SERIAL
    if communicator == "MPI_COMM_WORLD":
        return ParallelMode.ROOT
    raise RuntimeError(
        "the final IMEX example requires a proved serial or MPI_COMM_WORLD backend"
    )


def _bind_artifact(artifact: Any, **inputs: Any) -> Any:
    """Bind serial artifacts directly and MPI artifacts with their native world authority."""

    communicator = artifact.platform_manifest.communicator.require(
        "IMEX artifact communicator"
    )
    if communicator == "serial":
        return pops.bind(artifact, **inputs)
    if communicator == "MPI_COMM_WORLD":
        return pops.bind(
            artifact,
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
            **inputs,
        )
    raise RuntimeError("unsupported IMEX artifact communicator %r" % communicator)

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


@dataclass(frozen=True, slots=True)
class IMEXRuntimeSnapshot:
    """Every accepted runtime value needed for parity and strict restart proofs."""

    time: float
    macro_step: int
    states: dict[str, tuple[np.ndarray, ...]]
    fields: dict[str, tuple[np.ndarray, ...]]
    patch_boxes: tuple[tuple[int, ...], ...]
    regrid_count: int
    topology_epoch: int
    program_hash: str
    consumer_graph_identity: str
    consumer_cursors: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IMEXAMRProgramEvidence:
    """Accepted native Program evidence for conservative multi-level coupling."""

    flux_ledger_levels: tuple[int, ...]
    synchronization_phases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IMEXExecutionEvidence:
    """Artifacts plus exact pre/post-restart and continuation snapshots."""

    hdf5_path: Path
    paraview_path: Path
    checkpoint_path: Path
    hdf5_identity: str
    paraview_identity: str
    artifact_identity: str
    level_count: int
    program_evidence: IMEXAMRProgramEvidence
    accepted: IMEXRuntimeSnapshot
    restored: IMEXRuntimeSnapshot
    continuous: IMEXRuntimeSnapshot
    restarted: IMEXRuntimeSnapshot


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
            ).consume(action=solve_action)
            stage = program.value("%sstage_%d" % (tag, index), stage, at=point)

        # Match the library factory exactly: solve the Case-owned field from the implicit coordinate,
        # then lift its authenticated FieldContext onto the joint additive StagePoint.
        field_state = program.value(
            "%sfield_state_%d" % (tag, index), stage,
            at=point.time_for("implicit"),
        )
        outcome = core.diagnostic_field(field_state)
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
        fields_operator=core.diagnostic_field,
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
        AMRClockRelation,
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
        # The acceptance proof spans the first step plus one continuation. Cadence one makes a
        # regrid due in that window; the counter/epoch delta below proves that it actually completed.
        regrid=AMRRegrid(schedule=every(1, clock=core.program.clock)),
        transfer=transfer,
        # Temporal subcycling is declared independently from spatial refinement.
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )


def build_consumers(core: IMEXAMRAuthoring, *, output_mode: Any = None) -> Any:
    from pops.diagnostics import Integral
    from pops.output import (
        Checkpoint, ConsumerGraph, HDF5, NPZ, ParallelMode, ParaView, ScientificOutput,
    )
    from pops.time import every

    if output_mode is None:
        output_mode = ParallelMode.SERIAL

    fields = (core.tracer_state, core.diagnostic_field)
    diagnostic = Integral(block=core.tracer, cadence=every(1, clock=core.program.clock))
    return ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(mode=output_mode),
            schedule=every(1, clock=core.program.clock), fields=fields,
            diagnostics=(diagnostic,), target="hdf5/state"),
        ScientificOutput(
            format=NPZ(mode=output_mode),
            schedule=every(1, clock=core.program.clock), fields=fields,
            target="npz/state"),
        ScientificOutput(
            format=ParaView(mode=output_mode),
            schedule=every(1, clock=core.program.clock), fields=fields,
            target="paraview/state"),
        Checkpoint(
            schedule=every(1, clock=core.program.clock),
            target="checkpoints/restart", bit_identical=True),
    ))


def build_final_case(
    *, use_preset: bool = False, field_solver: Any | None = None,
    initial_background: float = 0.05, initial_amplitude: float = 0.95,
    output_mode: Any = None,
) -> FinalIMEXAMRCase:
    core = build_authoring(use_preset=use_preset, field_solver=field_solver)
    core.numerics.boundaries.add(build_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.initials.add(build_initial(
        core, background=initial_background, amplitude=initial_amplitude))
    core.case.program(core.program)
    core.case.consumers(build_consumers(core, output_mode=output_mode))
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


def compile_final_case(
    *, use_preset: bool = False,
) -> tuple[FinalIMEXAMRCase, Any, Any]:
    """Compile one exact manual or preset-authored target through the public lifecycle."""

    target = build_final_case(
        use_preset=use_preset, output_mode=_native_output_mode()
    )
    resolved = pops.resolve(pops.validate(target.authoring.case), layout=target.layout)
    return target, resolved, pops.compile(resolved)


def _snapshot(simulation: Any) -> IMEXRuntimeSnapshot:
    """Capture state, solved fields, hierarchy, clocks, identities and consumer cursors."""

    blocks = tuple(simulation.block_names())
    if blocks != ("tracer",):
        raise RuntimeError("IMEX acceptance expected exactly the qualified tracer block")
    level_count = int(simulation.n_levels())
    if level_count <= 0:
        raise RuntimeError("IMEX acceptance installed no AMR hierarchy levels")
    slots = tuple(simulation.field_provider_slots())
    if not slots:
        raise RuntimeError("IMEX acceptance installed no resolved diagnostic-field provider")
    field_level_counts = {
        slot: int(simulation.field_provider_levels(slot))
        for slot in slots
    }
    if any(count <= 0 for count in field_level_counts.values()):
        raise RuntimeError("IMEX acceptance installed an empty diagnostic-field hierarchy")
    regrid = simulation.amr.explain_regrid()
    return IMEXRuntimeSnapshot(
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
        states={
            block: tuple(
                np.asarray(
                    simulation.block_level_state_global(block, level), dtype=np.float64,
                ).copy()
                for level in range(level_count)
            )
            for block in blocks
        },
        fields={
            slot: tuple(
                np.asarray(
                    simulation.field_potential_level_global(slot, level), dtype=np.float64,
                ).copy()
                for level in range(field_level_counts[slot])
            )
            for slot in slots
        },
        patch_boxes=tuple(
            tuple(int(value) for value in row)
            for row in simulation.patch_boxes()
        ),
        regrid_count=int(regrid.regrid_count),
        topology_epoch=int(regrid.topology_epoch),
        program_hash=str(simulation.installed_program_hash()),
        consumer_graph_identity=simulation.consumer_graph.identity.token,
        consumer_cursors=simulation.consumer_cursors.to_data(),
    )


def _require_regrid_progress(
    before: IMEXRuntimeSnapshot,
    after: IMEXRuntimeSnapshot,
    *,
    where: str,
) -> None:
    """Require an actually completed regrid, not only a due schedule."""

    if after.regrid_count <= before.regrid_count:
        raise RuntimeError(
            "%s did not increase the completed AMR regrid count (%d -> %d)"
            % (where, before.regrid_count, after.regrid_count)
        )
    if after.topology_epoch <= before.topology_epoch:
        raise RuntimeError(
            "%s did not install a new AMR topology epoch (%d -> %d)"
            % (where, before.topology_epoch, after.topology_epoch)
        )


def _require_multilevel_program_evidence(report: Any) -> IMEXAMRProgramEvidence:
    """Authenticate one accepted AMR ledger/synchronization from program_report()."""

    if not report.installed:
        raise RuntimeError("IMEX acceptance has no installed native Program report")
    levels = tuple(sorted({int(row["level"]) for row in report.flux_ledger}))
    if levels != (0, 1):
        raise RuntimeError(
            "IMEX acceptance requires flux-ledger contributions on levels 0 and 1, got %r"
            % (levels,)
        )

    phase_groups: dict[tuple[int, ...], list[str]] = {}
    for row in report.synchronization:
        clock_phase = row["clock_phase"]
        key = (
            int(row["parent_level"]),
            int(row["child_level"]),
            int(row["block"]),
            int(row["macro_step"]),
            int(clock_phase["numerator"]),
            int(clock_phase["denominator"]),
        )
        phase_groups.setdefault(key, []).append(str(row["phase"]))
    expected_phases = ("reflux", "average_down")
    if not phase_groups:
        raise RuntimeError("IMEX acceptance published no AMR synchronization phases")
    for key, phases in phase_groups.items():
        if tuple(phases) != expected_phases:
            raise RuntimeError(
                "IMEX AMR synchronization %r must be reflux then average_down, got %r"
                % (key, tuple(phases))
            )
    return IMEXAMRProgramEvidence(
        flux_ledger_levels=levels,
        synchronization_phases=expected_phases,
    )


def _require_same_snapshot(
    left: IMEXRuntimeSnapshot,
    right: IMEXRuntimeSnapshot,
    *,
    where: str,
) -> None:
    """Reject any hidden state, field, topology, identity, clock or schedule drift."""

    if np.asarray(left.time, dtype=np.float64).tobytes() != np.asarray(
        right.time, dtype=np.float64,
    ).tobytes():
        raise RuntimeError("%s changed time" % where)
    exact = {
        "macro_step": (left.macro_step, right.macro_step),
        "patch_boxes": (left.patch_boxes, right.patch_boxes),
        "regrid_count": (left.regrid_count, right.regrid_count),
        "topology_epoch": (left.topology_epoch, right.topology_epoch),
        "program_hash": (left.program_hash, right.program_hash),
        "consumer_graph_identity": (
            left.consumer_graph_identity,
            right.consumer_graph_identity,
        ),
        "consumer_cursors": (left.consumer_cursors, right.consumer_cursors),
        "state routes": (tuple(left.states), tuple(right.states)),
        "field routes": (tuple(left.fields), tuple(right.fields)),
    }
    for name, (expected, actual) in exact.items():
        if expected != actual:
            raise RuntimeError("%s changed %s" % (where, name))
    for category, expected, actual in (
        ("conservative state", left.states, right.states),
        ("solved field", left.fields, right.fields),
    ):
        for route in expected:
            if len(expected[route]) != len(actual[route]):
                raise RuntimeError("%s changed %s %r level count" % (where, category, route))
            for level, (expected_level, actual_level) in enumerate(
                zip(expected[route], actual[route], strict=True),
            ):
                if not _array_bits_equal(expected_level, actual_level):
                    raise RuntimeError(
                        "%s changed %s %r level %d" % (where, category, route, level)
                    )


def _array_bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare dtype, shape and the complete contiguous byte representation."""

    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _snapshots_bit_identical(
    left: IMEXRuntimeSnapshot,
    right: IMEXRuntimeSnapshot,
) -> bool:
    """Return the same strict result enforced by :func:`_require_same_snapshot`."""

    try:
        _require_same_snapshot(left, right, where="runtime proof")
    except RuntimeError:
        return False
    return True


def _existing_artifact(root: Path, suffix: str) -> Path:
    paths = tuple(sorted(root.rglob("*%s" % suffix)))
    if not paths:
        raise RuntimeError("accepted run did not publish a %s artifact under %s" % (suffix, root))
    return paths[-1]


def _reopen_scientific_outputs(root: Path) -> tuple[Path, Path, str, str]:
    """Reopen independently persisted HDF5 and ParaView artifacts."""

    from pops.output import read_hdf5, read_paraview

    hdf5_path = _existing_artifact(root, ".h5")
    paraview_path = _existing_artifact(root, ".vtu")
    hdf5 = read_hdf5(hdf5_path)
    paraview = read_paraview(paraview_path)
    if not hdf5.arrays or not paraview.arrays:
        raise RuntimeError("published scientific artifacts reopened without arrays")
    if not all(
        np.isfinite(value).all()
        for artifact in (hdf5, paraview)
        for value in artifact.arrays.values()
    ):
        raise RuntimeError("published scientific output contains a non-finite value")
    return (
        hdf5_path,
        paraview_path,
        hdf5.output_identity.token,
        paraview.output_identity.token,
    )


def run_manual_and_restart(output_dir: Any) -> IMEXExecutionEvidence:
    """Run the manual Program, reopen output, restart fresh, then continue bit-identically."""

    root = Path(output_dir)
    accepted_root = root / "accepted"
    target, resolved, artifact = compile_final_case(use_preset=False)
    params = build_bind_params(target.authoring)
    simulation = _bind_artifact(artifact, params=params)
    controls = dict(target.authoring.run_controls)
    controls["output_dir"] = accepted_root
    if pops.run(simulation, **controls).accepted_steps <= 0:
        raise RuntimeError("the manual IMEX Program executed no accepted macro-step")

    hdf5_path, paraview_path, hdf5_identity, paraview_identity = \
        _reopen_scientific_outputs(accepted_root)
    # Keep the explicit API checkpoint in its own deterministic namespace.  The accepted-step
    # consumer publishes below ``accepted/checkpoints``; this independent restart proof must never
    # reuse that consumer-owned target or depend on its cadence.
    checkpoint_path = Path(simulation.checkpoint(root / "manual_restart"))
    if not checkpoint_path.is_file():
        raise RuntimeError("checkpoint API returned a missing artifact: %s" % checkpoint_path)
    accepted = _snapshot(simulation)

    resumed = _bind_artifact(artifact, params=params)
    resumed.restart(checkpoint_path)
    restored = _snapshot(resumed)
    _require_same_snapshot(accepted, restored, where="independent strict restart")
    if simulation.bind_identity != resumed.bind_identity:
        raise RuntimeError("fresh bind changed the authenticated IMEX install identity")
    if resumed.last_restart_identity is None:
        raise RuntimeError("restart did not publish an authenticated checkpoint identity")

    final_time = 2.0 * float(controls["t_end"])
    continuation = {
        "t_end": final_time,
        "max_steps": int(controls["max_steps"]),
    }
    if pops.run(
        simulation, output_dir=root / "continuous", **continuation,
    ).accepted_steps <= 0:
        raise RuntimeError("the uninterrupted IMEX continuation executed no macro-step")
    if pops.run(
        resumed, output_dir=root / "restarted", **continuation,
    ).accepted_steps <= 0:
        raise RuntimeError("the restarted IMEX continuation executed no macro-step")
    continuous, restarted = _snapshot(simulation), _snapshot(resumed)
    _require_same_snapshot(continuous, restarted, where="bit-identical continuation")
    _require_regrid_progress(accepted, continuous, where="uninterrupted continuation")
    _require_regrid_progress(restored, restarted, where="restarted continuation")
    continuous_report = simulation.program_report()
    restarted_report = resumed.program_report()
    continuous_program = _require_multilevel_program_evidence(continuous_report)
    restarted_program = _require_multilevel_program_evidence(restarted_report)
    if continuous_program != restarted_program:
        raise RuntimeError("restart changed the accepted AMR Program ledger/synchronization report")
    if (
        continuous_report.flux_ledger != restarted_report.flux_ledger
        or continuous_report.synchronization != restarted_report.synchronization
    ):
        raise RuntimeError("restart changed AMR Program ledger/synchronization entries")
    return IMEXExecutionEvidence(
        hdf5_path=hdf5_path,
        paraview_path=paraview_path,
        checkpoint_path=checkpoint_path,
        hdf5_identity=hdf5_identity,
        paraview_identity=paraview_identity,
        artifact_identity=artifact.artifact_identity.token,
        level_count=resolved.resolved_hierarchy.plan.level_count,
        program_evidence=continuous_program,
        accepted=accepted,
        restored=restored,
        continuous=continuous,
        restarted=restarted,
    )


def run_preset_parity(
    output_dir: Any,
    expected: IMEXRuntimeSnapshot,
) -> IMEXRuntimeSnapshot:
    """Prove manual/factory graph semantics and one-step runtime parity."""

    from pops.identity.semantic import program_semantic_data, semantic_identity_of

    manual = build_final_case(use_preset=False)
    preset, _resolved, artifact = compile_final_case(use_preset=True)
    manual_program = manual.authoring.program
    preset_program = preset.authoring.program
    if manual_program.to_graph().to_data() != preset_program.to_graph().to_data():
        raise RuntimeError("pops.lib.time.IMEX changed the manual Program graph")
    if program_semantic_data(manual_program) != program_semantic_data(preset_program):
        raise RuntimeError("pops.lib.time.IMEX changed normalized Program semantics")
    if semantic_identity_of(program=manual_program) != semantic_identity_of(
        program=preset_program,
    ):
        raise RuntimeError("pops.lib.time.IMEX changed the semantic Program identity")

    simulation = _bind_artifact(
        artifact,
        params=build_bind_params(preset.authoring),
    )
    if pops.run(
        simulation,
        t_end=expected.time,
        max_steps=int(preset.authoring.run_controls["max_steps"]),
        output_dir=Path(output_dir),
    ).accepted_steps <= 0:
        raise RuntimeError("the preset IMEX Program executed no accepted macro-step")
    actual = _snapshot(simulation)
    _require_same_snapshot(expected, actual, where="manual/pops.lib.time.IMEX parity")
    return actual


def main(argv: list[str] | None = None) -> None:
    """Run the final lifecycle, strict continuation and manual/factory parity proofs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT,
        help="directory receiving accepted scientific output and restart artifacts",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()

    evidence = run_manual_and_restart(output_dir / "manual")
    preset = run_preset_parity(output_dir / "preset", evidence.accepted)
    restart_equal = _snapshots_bit_identical(evidence.accepted, evidence.restored)
    continuation_equal = _snapshots_bit_identical(evidence.continuous, evidence.restarted)
    preset_equal = _snapshots_bit_identical(evidence.accepted, preset)
    finite = bool(
        all(
            np.isfinite(value).all()
            for levels in evidence.restarted.states.values()
            for value in levels
        )
        and all(
            np.isfinite(value).all()
            for levels in evidence.restarted.fields.values()
            for value in levels
        )
    )

    print("HDF5: %s" % evidence.hdf5_path)
    print("ParaView: %s" % evidence.paraview_path)
    print("checkpoint: %s" % evidence.checkpoint_path)
    print("bit-identical restart: %s" % restart_equal)
    print("bit-identical continuation: %s" % continuation_equal)
    print("manual/pops.lib.time.IMEX parity: %s" % preset_equal)
    print(
        "regrid count: %d -> %d (topology epoch %d -> %d)"
        % (
            evidence.accepted.regrid_count,
            evidence.restarted.regrid_count,
            evidence.accepted.topology_epoch,
            evidence.restarted.topology_epoch,
        )
    )
    print("report: " + json.dumps({
        "artifact": evidence.artifact_identity,
        "checkpoint_restart_bit_identical": restart_equal,
        "continuation_bit_identical": continuation_equal,
        "finite": finite,
        "flux_ledger_levels": list(evidence.program_evidence.flux_ledger_levels),
        "levels": evidence.level_count,
        "manual_preset_bit_identical": preset_equal,
        "program_hash": preset.program_hash,
        "regrid_count": evidence.accepted.regrid_count,
        "regrid_count_after_continuation": evidence.restarted.regrid_count,
        "runtime_steps": evidence.accepted.macro_step,
        "runtime_time": evidence.accepted.time,
        "runtime_steps_after_continuation": evidence.restarted.macro_step,
        "runtime_time_after_continuation": evidence.restarted.time,
        "synchronization_phases": list(evidence.program_evidence.synchronization_phases),
        "topology_epoch": evidence.accepted.topology_epoch,
        "topology_epoch_after_continuation": evidence.restarted.topology_epoch,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
