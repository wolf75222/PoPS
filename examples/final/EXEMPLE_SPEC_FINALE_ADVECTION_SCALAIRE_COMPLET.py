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
from pathlib import Path
from typing import Any

import numpy as np
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
ProgramBuilder = Callable[[Any, Any], pops.Program]


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
    state: np.ndarray
    patch_boxes: tuple[tuple[int, ...], ...]
    program_hash: str
    consumer_graph_identity: str
    consumer_cursors: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScalarExecutionEvidence:
    """Artifacts and snapshots proving manual execution, strict restart and continuation."""

    hdf5_path: Path
    paraview_path: Path
    checkpoint_path: Path
    hdf5_identity: str
    paraview_identity: str
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

    # The explicit builder remains the normative spelling. pops.lib.time.SSPRK2 is only a factory
    # for the same graph and is executed independently by the acceptance proof below.
    program = program_builder(tracer_state, rate)
    if type(program) is not pops.Program:
        raise TypeError("program_builder must return an exact pops.Program")
    program.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))

    # Run controls do not select physics, a spatial method, a time method or a CFL strategy.
    run_controls = {
        "t_end": 1.0,
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


def build_final_case(
    *,
    program_builder: ProgramBuilder = explicit_ssprk2,
    output_root: Any = OUTPUT_ROOT,
) -> FinalScalarAdvectionCase:
    """Assemble every public authority exactly once."""

    core = build_authoring(program_builder=program_builder, output_root=output_root)
    core.numerics.boundaries.add(build_transport_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.initials.add(build_initial_condition(core))
    core.case.program(core.program)
    core.case.consumers(build_consumer_graph(core))

    layout = build_amr_layout(core)
    return FinalScalarAdvectionCase(core, layout)


def build_bind_params(core: ScalarAdvectionAuthoring) -> dict[Any, float]:
    """Build parameter values only after validation has made every Handle canonical."""

    resolve = core.case.resolve
    return {
        resolve(core.velocity_x_param): 1.0,
        resolve(core.velocity_y_param): 0.25,
        resolve(core.inlet_x_param): 0.0,
        resolve(core.inlet_y_param): 0.0,
        resolve(core.refine_threshold): 0.10,
        resolve(core.coarsen_threshold): 0.04,
    }


def compile_final_case(
    *,
    program_builder: ProgramBuilder = explicit_ssprk2,
    output_root: Any = OUTPUT_ROOT,
) -> tuple[FinalScalarAdvectionCase, Any]:
    """Resolve and compile one exact manual or factory-authored target."""

    target = build_final_case(program_builder=program_builder, output_root=output_root)
    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout)
    return target, pops.compile(resolved)


def _snapshot(simulation: Any) -> ScalarRuntimeSnapshot:
    """Capture every state item required for strict AMR continuation parity."""

    blocks = tuple(simulation.block_names())
    if blocks != ("tracer",):
        raise RuntimeError("scalar acceptance expected exactly the qualified tracer block")
    return ScalarRuntimeSnapshot(
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
        state=np.asarray(simulation.state_global("tracer"), dtype=np.float64).copy(),
        patch_boxes=tuple(
            tuple(int(value) for value in row)
            for row in simulation.patch_boxes()
        ),
        program_hash=str(simulation.installed_program_hash()),
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
        "consumer_graph_identity": (
            left.consumer_graph_identity,
            right.consumer_graph_identity,
        ),
        "consumer_cursors": (left.consumer_cursors, right.consumer_cursors),
    }
    for name, (expected, actual) in exact.items():
        if expected != actual:
            raise RuntimeError("%s changed %s" % (where, name))
    if not np.array_equal(left.state, right.state):
        raise RuntimeError("%s changed the conservative tracer state" % where)


def _reopen_scientific_outputs(root: Path) -> tuple[Path, Path, str, str]:
    """Reopen one independently persisted HDF5 and ParaView artifact."""

    from pops.output.writers import read_hdf5, read_paraview

    hdf5_paths = tuple(sorted(root.rglob("*.h5")))
    paraview_paths = tuple(sorted(root.rglob("*.vtu")))
    if not hdf5_paths or not paraview_paths:
        raise RuntimeError("accepted scalar run did not publish both HDF5 and ParaView artifacts")
    hdf5_path, paraview_path = hdf5_paths[-1], paraview_paths[-1]
    hdf5 = read_hdf5(hdf5_path)
    paraview = read_paraview(paraview_path)
    return (
        hdf5_path,
        paraview_path,
        hdf5.output_identity.token,
        paraview.output_identity.token,
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
    simulation = pops.bind(artifact, params=params)
    controls = dict(target.authoring.run_controls)
    if pops.run(simulation, **controls) <= 0:
        raise RuntimeError("the explicit scalar Program executed no accepted macro-step")

    hdf5_path, paraview_path, hdf5_identity, paraview_identity = \
        _reopen_scientific_outputs(accepted_root)
    checkpoint_path = Path(simulation.checkpoint(root / "accepted_restart"))
    accepted = _snapshot(simulation)

    resumed = pops.bind(artifact, params=params)
    resumed.restart(checkpoint_path)
    restored = _snapshot(resumed)
    _require_same_snapshot(accepted, restored, where="independent strict restart")
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
    return ScalarExecutionEvidence(
        hdf5_path=hdf5_path,
        paraview_path=paraview_path,
        checkpoint_path=checkpoint_path,
        hdf5_identity=hdf5_identity,
        paraview_identity=paraview_identity,
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

    simulation = pops.bind(artifact, params=build_bind_params(preset.authoring))
    pops.run(
        simulation,
        t_end=expected.time,
        max_steps=int(preset.authoring.run_controls["max_steps"]),
        output_dir=output_dir,
    )
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
    print("  checkpoint: %s" % evidence.checkpoint_path)
    print("  bit-identical restart: step %d" % evidence.restarted.macro_step)
    print("  explicit/pops.lib.time.SSPRK2 parity: %s" % preset.program_hash)


if __name__ == "__main__":
    main()
