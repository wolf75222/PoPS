"""Final PoPS target: conservative scalar advection with explicit RK2 and AMR.

This is the executable acceptance target for the operator-first Python interface.  Physics,
numerics, time, mesh adaptation, consumers and execution controls each have one authority.  The
module deliberately has no compatibility aliases and no lower-level substitute for a missing public
hook: an unavailable join must fail where it is authored, not silently change the simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import Rectangle, RectangleBoundaryNames
from pops.frames import Cartesian2D
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import Interval, Positive, RuntimeParam
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, StagePoint, TimePoint


OUTPUT_ROOT = Path("outputs/scalar_advection")
VELOCITY_X = 1.0
VELOCITY_Y = 0.25
INFLOW_X = 0.0
INFLOW_Y = 0.0
GAUSSIAN_BACKGROUND = 0.05
GAUSSIAN_AMPLITUDE = 0.95
GAUSSIAN_INVERSE_WIDTH = 120.0
GAUSSIAN_CENTER_X = 0.30
GAUSSIAN_CENTER_Y = 0.35
RELATIVE_L2_TOLERANCE = 0.10
ProgramBuilder = Callable[[Any, Any], pops.Program]


def _native_output_mode() -> Any:
    """Select the portable publication topology proved by the loaded native backend."""

    from pops.output import ParallelMode
    from pops.runtime_environment import runtime_environment_report

    communicator = runtime_environment_report().get("communicator")
    if communicator == "serial":
        return ParallelMode.SERIAL
    if communicator == "MPI_COMM_WORLD":
        # ROOT keeps the example independent of parallel-HDF5 availability while retaining one
        # shared, complete scientific artifact under an explicit native MPI context.
        return ParallelMode.ROOT
    raise RuntimeError(
        "the final scalar example requires a proved serial or MPI_COMM_WORLD backend"
    )


def _bind_artifact(artifact: Any, **inputs: Any) -> Any:
    """Bind with the exact context required by the artifact's communicator contract."""

    communicator = artifact.platform_manifest.communicator.require(
        "scalar artifact communicator"
    )
    if communicator == "serial":
        return pops.bind(artifact, **inputs)
    if communicator == "MPI_COMM_WORLD":
        return pops.bind(
            artifact,
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
            **inputs,
        )
    raise RuntimeError("unsupported scalar artifact communicator %r" % communicator)


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


@dataclass(frozen=True, slots=True)
class ScalarRuntimeSnapshot:
    """Restart-sensitive evidence retained without exposing native implementation objects."""

    time: float
    macro_step: int
    states: tuple[np.ndarray, ...]
    patch_boxes: tuple[tuple[int, ...], ...]
    program_hash: str
    program_transaction_state: str
    consumer_graph_identity: str
    consumer_cursors: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScalarErrorNorms:
    """Cell-volume-weighted error against the exact characteristic solution."""

    time: float
    active_cells: int
    l1: float
    l2: float
    linf: float
    relative_l2: float


@dataclass(frozen=True, slots=True)
class ScalarAMRProgramEvidence:
    """Accepted conservative coupling recorded by the native Program report."""

    flux_ledger_levels: tuple[int, ...]
    synchronization_relations: tuple[tuple[int, int], ...]
    synchronization_phases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScalarExecutionEvidence:
    """Artifacts and snapshots proving manual execution, strict restart and continuation."""

    hdf5_path: Path
    paraview_path: Path
    checkpoint_path: Path
    hdf5_identity: str
    paraview_identity: str
    error_norms: ScalarErrorNorms
    program_evidence: ScalarAMRProgramEvidence
    accepted: ScalarRuntimeSnapshot
    restored: ScalarRuntimeSnapshot
    continuous: ScalarRuntimeSnapshot
    restarted: ScalarRuntimeSnapshot


def explicit_ssprk2(state: Any, rate: Any) -> pops.Program:
    """Spell SSPRK2 entirely with generic Program operations.

    Node names and algebra intentionally match the canonical factory expansion so presentation-only
    provenance is the only difference between this function and ``pops.lib.time.SSPRK2``.
    """

    program = pops.Program("SSPRK2")
    q = program.state(state)
    stage_0 = StagePoint(
        "ssprk2_stage_0",
        {"main": TimePoint(program.clock, 0)},
    )
    k0 = program.value("ssprk2_k_0", rate(q.n), at=stage_0)
    stage_1 = StagePoint(
        "ssprk2_stage_1",
        {"main": TimePoint(program.clock, 1)},
    )
    q_stage = program.value(
        "ssprk2_U1",
        q.n + program.dt * 1 * k0,
        at=stage_1,
    )
    k1 = program.value("ssprk2_k_1", rate(q_stage), at=stage_1)
    half = Fraction(1, 2)
    q_next = program.value(
        "ssprk2_step",
        q.n + program.dt * half * k0 + program.dt * half * k1,
        at=q.next.point,
    )
    program.commit(q.next, q_next)
    return program


def preset_ssprk2(state: Any, rate: Any) -> pops.Program:
    """Return the library spelling of exactly the same canonical Program graph."""

    from pops.lib.time import SSPRK2

    return SSPRK2(state, rate=rate)


def build_authoring(
    *,
    program_builder: ProgramBuilder = explicit_ssprk2,
    output_root: Any = OUTPUT_ROOT,
) -> ScalarAdvectionAuthoring:
    """Build the pure operator-first declarations without importing native code."""

    if not callable(program_builder):
        raise TypeError("program_builder must construct one ordinary pops.Program")

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
        RuntimeParam("a_x", default=VELOCITY_X, domain=Positive())
    )
    velocity_y_param = model.param(
        RuntimeParam("a_y", default=VELOCITY_Y, domain=Positive())
    )
    inlet_x_param = model.param(
        RuntimeParam("u_in_x", default=INFLOW_X, domain=Interval(-10.0, 10.0))
    )
    inlet_y_param = model.param(
        RuntimeParam("u_in_y", default=INFLOW_Y, domain=Interval(-10.0, 10.0))
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

    # The explicit builder remains the normative spelling. pops.lib.time.SSPRK2 is only a factory
    # for the same graph and is executed independently by the acceptance proof below.
    program = program_builder(tracer_state, rate)
    if type(program) is not pops.Program:
        raise TypeError("program_builder must return an exact pops.Program")
    program.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))

    # Run controls do not select physics, a spatial method, a time method or a CFL strategy.
    run_controls = {
        # At t=0.2 the translated Gaussian is still inside the domain, so the accepted scientific
        # artifact can be checked against the non-trivial characteristic solution.
        "t_end": 0.20,
        "max_steps": 100_000,
        "output_dir": Path(output_root),
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
        PatchLayout,
        Tag,
    )
    from pops.layouts import AMR
    from pops.lib.amr import BergerRigoutsos, StateTransfer, SymbolicTagger
    from pops.math import grad, norm
    from pops.time import every

    # The explicit Handle -> Expr conversion preserves Python identity semantics.  AMRTagging binds
    # this continuous-looking predicate to U's resolved spatial method.  That method owns the exact
    # typed, serializable gradient coefficients/offsets/order/halos transported to both builtin and
    # external Taggers; there is no runtime reconstruction-name switch or centered fallback.
    tracer_value = ValueExpr(core.tracer_state)
    gradient_magnitude = norm(grad(tracer_value))
    tagging = AMRTagging(
        rules=(
            Tag(gradient_magnitude > core.case.value(core.refine_threshold)),
            Coarsen(gradient_magnitude < core.case.value(core.coarsen_threshold)),
            Buffer(cells=2),
        ),
        hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )

    # Transfer accuracy/halos come from StateTransfer's installed policies; order is never repeated.
    transfer = AMRTransfer()
    transfer.state(core.tracer_state, StateTransfer())

    return AMR(
        grid=core.grid,
        hierarchy=AMRHierarchy(max_levels=3, ratios=(2, 2)),
        tagging=tagging,
        # Providers are explicit typed objects. A Tagger evaluates this exact resolved graph (all
        # state inputs come from its leaves); it is never a second, independent tagging policy.
        # pops.lib contains builtins; pops.amr exposes exact external native provider bindings.
        tagger=SymbolicTagger(),
        clustering=BergerRigoutsos(),
        regrid=AMRRegrid(schedule=every(5, clock=core.program.clock)),
        transfer=transfer,
        # Temporal subcycling is an independent authority; it is never inferred from spatial ratios.
        execution=AMRExecution.subcycled((
            AMRClockRelation(0, 1, 2),
            AMRClockRelation(1, 2, 2),
        )),
        # The distribution choice is explicit. The native provider derives the coarse patch size;
        # no public integer sentinel or duplicated grid-size policy is authored here.
        patch_layout=PatchLayout(distribute_coarse=True),
    )


def build_initial_condition(core: ScalarAdvectionAuthoring) -> Any:
    """Build analytic data and its explicit conservative projection."""

    from pops.initial import InitialCondition
    from pops.lib.initial import Gaussian
    from pops.projection import ConservativeCellAverage

    gaussian = Gaussian(
        frame=core.frame,
        center={
            core.frame.x: GAUSSIAN_CENTER_X,
            core.frame.y: GAUSSIAN_CENTER_Y,
        },
        background=GAUSSIAN_BACKGROUND,
        amplitude=GAUSSIAN_AMPLITUDE,
        inverse_width=GAUSSIAN_INVERSE_WIDTH,
    )
    return InitialCondition(
        state=core.tracer_state,
        value=gaussian,
        projection=ConservativeCellAverage(),
    )


def build_consumer_graph(
    core: ScalarAdvectionAuthoring, *, output_mode: Any = None,
) -> Any:
    """Build the sole accepted-side-effect graph for diagnostics, output and checkpointing."""

    from pops.diagnostics import Integral, MinMax, Norm
    from pops.linalg.norms import L1, L2, LInf
    from pops.output import (
        Checkpoint, ConsumerGraph, HDF5, ParallelMode, ParaView, ScientificOutput,
    )
    from pops.time import every, on_end

    if output_mode is None:
        output_mode = ParallelMode.SERIAL

    diagnostic_schedule = every(10, clock=core.program.clock)
    diagnostics = (
        Integral(block=core.tracer, cadence=diagnostic_schedule),
        Norm(L1(), block=core.tracer, cadence=diagnostic_schedule),
        Norm(L2(), block=core.tracer, cadence=diagnostic_schedule),
        Norm(LInf(), block=core.tracer, cadence=diagnostic_schedule),
        MinMax(block=core.tracer, cadence=diagnostic_schedule),
    )
    consumers = (
        ScientificOutput(
            format=ParaView(mode=output_mode),
            schedule=diagnostic_schedule,
            fields=(core.tracer_state,),
            diagnostics=diagnostics,
            target="solution/tracer",
        ),
        ScientificOutput(
            # ROOT is the portable shared-file route for an MPI artifact; the dedicated MPI
            # conformance matrix separately exercises true collective HDF5.
            format=HDF5(mode=output_mode),
            schedule=every(50, clock=core.program.clock),
            fields=(core.tracer_state,),
            target="state/tracer",
        ),
        Checkpoint(
            # A fixed target is a single immutable publication, so publish it exactly once at the
            # accepted end of each run.  A repeating cadence would correctly fail on its second
            # attempt rather than silently overwrite the first restart state.
            schedule=on_end(clock=core.program.clock),
            bit_identical=True,
            target="checkpoints/restart",
        ),
    )
    return ConsumerGraph.from_consumers(consumers)


def build_final_case(
    *,
    program_builder: ProgramBuilder = explicit_ssprk2,
    output_root: Any = OUTPUT_ROOT,
    output_mode: Any = None,
) -> FinalScalarAdvectionCase:
    """Assemble every public authority exactly once."""

    core = build_authoring(program_builder=program_builder, output_root=output_root)
    core.numerics.boundaries.add(build_transport_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.initials.add(build_initial_condition(core))
    core.case.program(core.program)
    core.case.consumers(build_consumer_graph(core, output_mode=output_mode))

    layout = build_amr_layout(core)
    return FinalScalarAdvectionCase(core, layout)


def build_bind_params(core: ScalarAdvectionAuthoring) -> dict[Any, float]:
    """Build parameter values only after validation has made every Handle canonical."""

    resolve = core.case.resolve
    return {
        resolve(core.velocity_x_param): VELOCITY_X,
        resolve(core.velocity_y_param): VELOCITY_Y,
        resolve(core.inlet_x_param): INFLOW_X,
        resolve(core.inlet_y_param): INFLOW_Y,
        resolve(core.refine_threshold): 0.10,
        resolve(core.coarsen_threshold): 0.04,
    }


def compile_final_case(
    *,
    program_builder: ProgramBuilder = explicit_ssprk2,
    output_root: Any = OUTPUT_ROOT,
) -> tuple[FinalScalarAdvectionCase, Any]:
    """Resolve and compile one exact manual or factory-authored target."""

    target = build_final_case(
        program_builder=program_builder,
        output_root=output_root,
        output_mode=_native_output_mode(),
    )
    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout)
    return target, pops.compile(resolved)


def _program_transaction_state(simulation: Any) -> str:
    """Canonicalize every restart-sensitive Program registry without native objects."""

    report = simulation.program_report().to_dict()
    return json.dumps({
        "cache": report["cache"],
        "clocks": report["clocks"],
        "diagnostics": report["diagnostics"],
        "flux_ledger": report["flux_ledger"],
        "histories": report["histories"],
        "level_relations": report["level_relations"],
        "synchronization": report["synchronization"],
        "temporal": report["temporal"],
    }, sort_keys=True, separators=(",", ":"))


def _snapshot(simulation: Any) -> ScalarRuntimeSnapshot:
    """Capture every state item required for strict AMR continuation parity."""

    blocks = tuple(simulation.block_names())
    if blocks != ("tracer",):
        raise RuntimeError("scalar acceptance expected exactly the qualified tracer block")
    level_count = int(simulation.n_levels())
    if level_count <= 0:
        raise RuntimeError("scalar acceptance installed no AMR hierarchy levels")
    return ScalarRuntimeSnapshot(
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
        states=tuple(
            np.asarray(
                simulation.block_level_state_global("tracer", level),
                dtype=np.float64,
            ).copy()
            for level in range(level_count)
        ),
        patch_boxes=tuple(
            tuple(int(value) for value in row)
            for row in simulation.patch_boxes()
        ),
        program_hash=str(simulation.installed_program_hash()),
        program_transaction_state=_program_transaction_state(simulation),
        consumer_graph_identity=simulation.consumer_graph.identity.token,
        consumer_cursors=simulation.consumer_cursors.to_data(),
    )


def _require_same_snapshot(
    left: ScalarRuntimeSnapshot,
    right: ScalarRuntimeSnapshot,
    *,
    where: str,
) -> None:
    """Reject any hidden state, topology, identity, clock or schedule drift."""

    exact = {
        "time": (left.time, right.time),
        "macro_step": (left.macro_step, right.macro_step),
        "patch_boxes": (left.patch_boxes, right.patch_boxes),
        "program_hash": (left.program_hash, right.program_hash),
        "program_transaction_state": (
            left.program_transaction_state,
            right.program_transaction_state,
        ),
        "consumer_graph_identity": (
            left.consumer_graph_identity,
            right.consumer_graph_identity,
        ),
        "consumer_cursors": (left.consumer_cursors, right.consumer_cursors),
    }
    for name, (expected, actual) in exact.items():
        if expected != actual:
            raise RuntimeError("%s changed %s" % (where, name))
    if len(left.states) != len(right.states) or any(
        not np.array_equal(expected, actual)
        for expected, actual in zip(left.states, right.states, strict=True)
    ):
        raise RuntimeError("%s changed the conservative tracer AMR hierarchy" % where)


def _require_refined_hierarchy(snapshot: ScalarRuntimeSnapshot, *, where: str) -> None:
    """Require the AMR acceptance target to execute at least one genuinely refined level."""

    # ``patch_boxes`` reports adaptive patches only; the level-zero base box is represented by the
    # first public level state.  Require every installed refined state level to have real patch
    # geometry instead of incorrectly expecting a synthetic level-zero patch row.
    expected_levels = tuple(range(1, len(snapshot.states)))
    actual_levels = tuple(sorted({row[0] for row in snapshot.patch_boxes}))
    if not expected_levels or actual_levels != expected_levels:
        raise RuntimeError(
            "%s did not execute the requested refined AMR hierarchy: expected=%r, actual=%r"
            % (where, expected_levels, actual_levels)
        )


def _analytic_solution(
    x: np.ndarray,
    y: np.ndarray,
    *,
    time: float,
) -> np.ndarray:
    """Backtrace positive characteristics through the two zero-inflow boundaries."""

    departure_x = x - VELOCITY_X * time
    departure_y = y - VELOCITY_Y * time
    inside = (
        (departure_x >= 0.0)
        & (departure_x <= 1.0)
        & (departure_y >= 0.0)
        & (departure_y <= 1.0)
    )
    exact = np.zeros_like(x, dtype=np.float64)
    exact[inside] = (
        GAUSSIAN_BACKGROUND
        + GAUSSIAN_AMPLITUDE
        * np.exp(
            -GAUSSIAN_INVERSE_WIDTH
            * (
                (departure_x[inside] - GAUSSIAN_CENTER_X) ** 2
                + (departure_y[inside] - GAUSSIAN_CENTER_Y) ** 2
            )
        )
    )
    return exact


def _scalar_error_norms(paraview: Any) -> ScalarErrorNorms:
    """Measure the accepted leaf-cell solution stored in one reopened VTU artifact."""

    field_records = tuple(paraview.manifest["datasets"]["fields"].values())
    field_names = {
        str(record["name"])
        for record in field_records
        if record["association"] == "cell"
    }
    if len(field_names) != 1:
        raise RuntimeError(
            "scalar acceptance expected one cell-field family, got %r"
            % (tuple(sorted(field_names)),)
        )
    (field_name,) = tuple(field_names)
    values = np.asarray(paraview.arrays[field_name], dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise RuntimeError("scalar VTU field must contain one component per cell")

    points = np.asarray(paraview.arrays["Points"], dtype=np.float64)
    offsets = np.asarray(paraview.arrays["offsets"], dtype=np.int64)
    connectivity = np.asarray(paraview.arrays["connectivity"], dtype=np.int64)
    cell_sizes = np.diff(np.concatenate((np.asarray((0,), dtype=np.int64), offsets)))
    if (
        offsets.size != values.size
        or cell_sizes.size == 0
        or not np.all(cell_sizes == cell_sizes[0])
    ):
        raise RuntimeError("scalar VTU topology is not one fixed-size cell family")
    cell_points = connectivity.reshape((offsets.size, int(cell_sizes[0])))
    centers = np.mean(points[cell_points, :2], axis=1)

    coverage = np.asarray(paraview.arrays["pops_coverage"], dtype=np.uint8)
    ghost_types = np.asarray(paraview.arrays["vtkGhostType"], dtype=np.uint8)
    volumes = np.asarray(paraview.arrays["pops_cell_volume"], dtype=np.float64)
    if not (
        coverage.shape == ghost_types.shape == volumes.shape == values.shape
    ):
        raise RuntimeError("scalar VTU geometry and field arrays have inconsistent extents")
    # Ignore covered coarse cells and replicated MPI cells. Bit 0 is VTK_DUPLICATECELL.
    active = (coverage == 0) & ((ghost_types & np.uint8(1)) == 0)
    if not np.any(active) or np.any(volumes[active] <= 0.0):
        raise RuntimeError("scalar VTU contains no positive-volume active leaf cells")

    time_values = np.asarray(paraview.arrays["TimeValue"], dtype=np.float64)
    if time_values.shape != (1,) or not np.isfinite(time_values[0]):
        raise RuntimeError("scalar VTU must contain one finite physical TimeValue")
    time = float(time_values[0])
    exact = _analytic_solution(centers[:, 0], centers[:, 1], time=time)
    error = values - exact
    weights = volumes[active]
    active_error = error[active]
    exact_l2 = float(np.sqrt(np.sum(exact[active] ** 2 * weights)))
    if not np.isfinite(active_error).all() or exact_l2 <= 0.0:
        raise RuntimeError("scalar analytic comparison is non-finite or has zero reference norm")
    l1 = float(np.sum(np.abs(active_error) * weights))
    l2 = float(np.sqrt(np.sum(active_error**2 * weights)))
    linf = float(np.max(np.abs(active_error)))
    result = ScalarErrorNorms(
        time=time,
        active_cells=int(np.count_nonzero(active)),
        l1=l1,
        l2=l2,
        linf=linf,
        relative_l2=l2 / exact_l2,
    )
    if result.relative_l2 > RELATIVE_L2_TOLERANCE:
        raise RuntimeError(
            "scalar relative L2 error %.6e exceeds documented tolerance %.6e at t=%.6e"
            % (result.relative_l2, RELATIVE_L2_TOLERANCE, result.time)
        )
    return result


def _reopen_scientific_outputs(
    root: Path,
) -> tuple[Path, Path, str, str, ScalarErrorNorms]:
    """Reopen one independently persisted HDF5 and ParaView artifact."""

    from pops.output import read_hdf5, read_paraview

    hdf5_paths = tuple(sorted(root.rglob("*.h5")))
    paraview_paths = tuple(sorted(root.rglob("*.vtu")))
    if not hdf5_paths or not paraview_paths:
        raise RuntimeError("accepted scalar run did not publish both HDF5 and ParaView artifacts")
    hdf5_path, paraview_path = hdf5_paths[-1], paraview_paths[-1]
    hdf5 = read_hdf5(hdf5_path)
    paraview = read_paraview(paraview_path)
    if not hdf5.arrays or not paraview.arrays:
        raise RuntimeError("published scalar artifacts reopened without arrays")
    if not all(
        np.isfinite(value).all()
        for artifact in (hdf5, paraview)
        for value in artifact.arrays.values()
    ):
        raise RuntimeError("published scalar output contains a non-finite value")
    return (
        hdf5_path,
        paraview_path,
        hdf5.output_identity.token,
        paraview.output_identity.token,
        _scalar_error_norms(paraview),
    )


def _require_multilevel_program_evidence(
    report: Any,
    *,
    expected_levels: tuple[int, ...],
) -> ScalarAMRProgramEvidence:
    """Authenticate flux contributions and reflux-before-average-down coupling."""

    if not report.installed:
        raise RuntimeError("scalar acceptance has no installed native Program report")
    levels = tuple(sorted({int(row["level"]) for row in report.flux_ledger}))
    if levels != expected_levels:
        raise RuntimeError(
            "scalar flux ledger levels differ from the installed hierarchy: %r != %r"
            % (levels, expected_levels)
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
        raise RuntimeError("scalar acceptance published no AMR synchronization phases")
    for key, phases in phase_groups.items():
        if tuple(phases) != expected_phases:
            raise RuntimeError(
                "scalar AMR synchronization %r must be reflux then average_down, got %r"
                % (key, tuple(phases))
            )
    relations = tuple(sorted({(key[0], key[1]) for key in phase_groups}))
    expected_relations = tuple(
        (parent, parent + 1) for parent in range(len(expected_levels) - 1)
    )
    if relations != expected_relations:
        raise RuntimeError(
            "scalar synchronization relations differ from the installed hierarchy: %r != %r"
            % (relations, expected_relations)
        )
    return ScalarAMRProgramEvidence(
        flux_ledger_levels=levels,
        synchronization_relations=relations,
        synchronization_phases=expected_phases,
    )


def run_manual_and_restart(output_dir: Any) -> ScalarExecutionEvidence:
    """Execute the explicit Program, reopen outputs, restart, then continue bit-identically."""

    root = Path(output_dir)
    accepted_root = root / "accepted"
    target, artifact = compile_final_case(
        program_builder=explicit_ssprk2,
        output_root=accepted_root,
    )
    params = build_bind_params(target.authoring)
    simulation = _bind_artifact(artifact, params=params)
    controls = dict(target.authoring.run_controls)
    run_report = pops.run(simulation, **controls)
    if run_report.accepted_steps <= 0:
        raise RuntimeError("the explicit scalar Program executed no accepted macro-step")

    hdf5_path, paraview_path, hdf5_identity, paraview_identity, error_norms = \
        _reopen_scientific_outputs(accepted_root)
    checkpoint_path = Path(simulation.checkpoint(root / "accepted_restart"))
    accepted = _snapshot(simulation)
    _require_refined_hierarchy(accepted, where="accepted scalar run")

    resumed = _bind_artifact(artifact, params=params)
    resumed.restart(checkpoint_path)
    restored = _snapshot(resumed)
    _require_same_snapshot(accepted, restored, where="independent strict restart")
    _require_refined_hierarchy(restored, where="restored scalar run")
    if simulation.bind_identity != resumed.bind_identity:
        raise RuntimeError("fresh bind changed the authenticated scalar install identity")
    if resumed.last_restart_identity is None:
        raise RuntimeError("restart did not publish an authenticated checkpoint identity")

    final_time = 2.0 * float(controls["t_end"])
    pops.run(
        simulation,
        t_end=final_time,
        max_steps=int(controls["max_steps"]),
        output_dir=root / "continuous",
    )
    pops.run(
        resumed,
        t_end=final_time,
        max_steps=int(controls["max_steps"]),
        output_dir=root / "restarted",
    )
    continuous, restarted = _snapshot(simulation), _snapshot(resumed)
    _require_same_snapshot(continuous, restarted, where="bit-identical continuation")
    expected_levels = tuple(range(len(continuous.states)))
    continuous_report = simulation.program_report()
    restarted_report = resumed.program_report()
    continuous_program = _require_multilevel_program_evidence(
        continuous_report,
        expected_levels=expected_levels,
    )
    restarted_program = _require_multilevel_program_evidence(
        restarted_report,
        expected_levels=expected_levels,
    )
    if continuous_program != restarted_program:
        raise RuntimeError("restart changed scalar AMR ledger/synchronization evidence")
    if (
        continuous_report.flux_ledger != restarted_report.flux_ledger
        or continuous_report.synchronization != restarted_report.synchronization
    ):
        raise RuntimeError("restart changed scalar AMR ledger/synchronization entries")
    return ScalarExecutionEvidence(
        hdf5_path=hdf5_path,
        paraview_path=paraview_path,
        checkpoint_path=checkpoint_path,
        hdf5_identity=hdf5_identity,
        paraview_identity=paraview_identity,
        error_norms=error_norms,
        program_evidence=continuous_program,
        accepted=accepted,
        restored=restored,
        continuous=continuous,
        restarted=restarted,
    )


def run_preset_parity(output_dir: Any, expected: ScalarRuntimeSnapshot) -> ScalarRuntimeSnapshot:
    """Prove factory graph/hash parity and execute it to the same accepted state."""

    from pops.identity.semantic import program_semantic_data, semantic_identity_of

    manual = build_final_case(program_builder=explicit_ssprk2, output_root=output_dir)
    preset, artifact = compile_final_case(
        program_builder=preset_ssprk2,
        output_root=output_dir,
    )
    manual_program = manual.authoring.program
    preset_program = preset.authoring.program
    if manual_program.to_graph().to_data() != preset_program.to_graph().to_data():
        raise RuntimeError("pops.lib.time.SSPRK2 changed the explicit Program graph")
    if program_semantic_data(manual_program) != program_semantic_data(preset_program):
        raise RuntimeError("pops.lib.time.SSPRK2 changed normalized Program semantics")
    if semantic_identity_of(program=manual_program) != semantic_identity_of(program=preset_program):
        raise RuntimeError("pops.lib.time.SSPRK2 changed the semantic Program identity")

    simulation = _bind_artifact(
        artifact, params=build_bind_params(preset.authoring)
    )
    # Mirror the explicit proof's two run boundaries.  ``on_end`` checkpoint cursors are part of
    # restart-sensitive state, so manual/factory parity includes the same two accepted end events
    # while each immutable publication receives its own output root.
    split_time = float(preset.authoring.run_controls["t_end"])
    if not 0.0 < split_time < expected.time:
        raise RuntimeError("preset parity requires an intermediate accepted run boundary")
    first_report = pops.run(
        simulation,
        t_end=split_time,
        max_steps=int(preset.authoring.run_controls["max_steps"]),
        output_dir=Path(output_dir) / "accepted",
    )
    second_report = pops.run(
        simulation,
        t_end=expected.time,
        max_steps=int(preset.authoring.run_controls["max_steps"]),
        output_dir=Path(output_dir) / "continued",
    )
    if first_report.accepted_steps <= 0 or second_report.accepted_steps <= 0:
        raise RuntimeError("preset SSPRK2 parity did not execute both accepted run segments")
    actual = _snapshot(simulation)
    _require_same_snapshot(expected, actual, where="manual/pops.lib.time.SSPRK2 parity")
    return actual


def main() -> None:
    """Run the final lifecycle, strict restart and manual/factory parity proof."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    evidence = run_manual_and_restart(args.output_dir / "manual")
    preset = run_preset_parity(args.output_dir / "preset", evidence.continuous)
    print("PoPS final scalar-advection acceptance:")
    print("  HDF5: %s" % evidence.hdf5_identity)
    print("  ParaView: %s" % evidence.paraview_identity)
    print(
        "  analytic error at t=%.6f: L1=%.6e L2=%.6e Linf=%.6e relative-L2=%.6e"
        % (
            evidence.error_norms.time,
            evidence.error_norms.l1,
            evidence.error_norms.l2,
            evidence.error_norms.linf,
            evidence.error_norms.relative_l2,
        )
    )
    print(
        "  AMR synchronization: levels=%r relations=%r phases=%r"
        % (
            evidence.program_evidence.flux_ledger_levels,
            evidence.program_evidence.synchronization_relations,
            evidence.program_evidence.synchronization_phases,
        )
    )
    print("  checkpoint: %s" % evidence.checkpoint_path)
    print("  bit-identical restart: step %d" % evidence.restarted.macro_step)
    print("  explicit/pops.lib.time.SSPRK2 parity: %s" % preset.program_hash)


if __name__ == "__main__":
    main()
