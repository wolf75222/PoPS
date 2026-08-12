#!/usr/bin/env python3
"""MPI parity of the final public AMR lifecycle, including generated cell-local FE.

Both legs execute the exact public route

``Case -> validate -> resolve -> compile -> ExecutionContext.mpi_world -> bind -> run``.

One leg authors SSPRK2 with generic :class:`pops.Program` operations; the other uses the
``pops.lib.time.SSPRK2`` preset.  Under ``mpiexec -n 2`` their collectively gathered conservative
states must be bit-identical and each route must conserve global mass across real AMR regrids.
No native carrier, codegen driver, deleted test helper, or private runtime state is used.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
import hashlib
import inspect
from pathlib import Path
import sys
from typing import Any

from _compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import allgather_value
    from pops.runtime._component_execution_context import component_execution_data
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        PatchLayout,
        Tag,
    )
    from pops.codegen import Production
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib import time as libtime
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import Gaussian
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.params import RuntimeParam
    from pops.physics import Model
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt, StagePoint, TimePoint, every
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("Python MPI runtime import failed: %s" % exc)


ROOT = Path(__file__).resolve().parents[4]
N = 16
NSTEPS = 4
DT = 1.0e-3
_fails = 0
_COMM = _pops.mpi_world()


def _phase(label: str) -> None:
    """Emit an unbuffered per-rank marker around potentially collective phases."""
    print("[rank %d] %s" % (int(_COMM.rank), label), flush=True)


def chk(cond: Any, label: str) -> None:
    global _fails
    if _COMM.rank == 0:
        print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _require_world() -> None:
    size = int(_COMM.size)
    if size >= 2:
        return
    require_mpi_or_skip("needs mpiexec -n 2; MPI_COMM_WORLD size=%d" % size)


def _world_identical(array: np.ndarray) -> bool:
    contiguous = np.ascontiguousarray(array)
    witness = (
        tuple(contiguous.shape),
        contiguous.dtype.str,
        hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    )
    return len(set(allgather_value(_COMM, witness))) == 1


def _explicit_ssprk2(state: Any, rate: Any) -> pops.Program:
    """Normative two-stage SSPRK2 built only from generic Program operations."""
    program = pops.Program("SSPRK2")
    q = program.state(state)
    stage_0 = StagePoint("ssprk2_stage_0", {"main": TimePoint(program.clock, 0)})
    k0 = program.value("ssprk2_k_0", rate(q.n), at=stage_0)
    stage_1 = StagePoint("ssprk2_stage_1", {"main": TimePoint(program.clock, 1)})
    q_stage = program.value("ssprk2_U1", q.n + program.dt * k0, at=stage_1)
    k1 = program.value("ssprk2_k_1", rate(q_stage), at=stage_1)
    half = Fraction(1, 2)
    q_next = program.value(
        "ssprk2_step",
        q.n + program.dt * half * k0 + program.dt * half * k1,
        at=q.next.point,
    )
    program.commit(q.next, q_next)
    return program


def _preset_ssprk2(state: Any, rate: Any) -> pops.Program:
    from pops.lib.time import SSPRK2

    return SSPRK2(state, rate=rate)


def _forward_euler_pack(states: Any, rate: Any) -> pops.Program:
    routes = tuple(
        libtime.RungeKuttaRoute(state, rate)
        for state in (states if isinstance(states, tuple) else (states,))
    )
    return libtime.RungeKutta(routes=routes, tableau=libtime.FORWARD_EULER_TABLEAU)


def _cell_local_forward_euler_pack(states: Any, rate: Any) -> pops.Program:
    program = _forward_euler_pack(states, rate)
    # Rung zero is the finest-level authored rung. The native route derives rung one on L0 from the
    # exact 2:1 clock relation, so both levels execute one identical Forward-Euler batch per window.
    program.cell_local_time(tick_denominator=2000, rung=0)
    return program


def _resolved(
    program_builder: Callable[[Any, Any], pops.Program],
    *,
    distribute_coarse: bool,
    multi_block: bool = False,
) -> Any:
    frame = Rectangle("mpi-public-amr-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = Model("mpi-public-amr-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (rho,), y_axis: (Fraction(1, 4) * rho,)},
        waves={x_axis: (1.0 + 0.0 * rho,), y_axis: (0.25 + 0.0 * rho,)},
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))

    case = pops.Case("mpi-public-amr-case")
    blocks = [case.block("tracer", model)]
    if multi_block:
        blocks.append(case.block("tracer_peer", model))
    state_instances = tuple(block[state] for block in blocks)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    for block in blocks:
        case.numerics(numerics, block=block)
    program = program_builder(state_instances if multi_block else state_instances[0], rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    for state_instance in state_instances:
        case.initials.add(
            InitialCondition(
                state=state_instance,
                value=Gaussian(
                    frame=frame,
                    center={x_axis: 0.35, y_axis: 0.55},
                    background=1.0,
                    amplitude=0.3,
                    inverse_width=90.0,
                ),
                projection=ConservativeCellAverage(),
            )
        )
    threshold = case.param(RuntimeParam("refine_threshold", default=1.05))
    transfer = AMRTransfer()
    for state_instance in state_instances:
        transfer.state(state_instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(state_instances[0]) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
        patch_layout=PatchLayout(
            distribute_coarse=distribute_coarse,
            coarse_max_grid=max(4, N // 2),
        ),
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )


def _execute(
    program_builder: Callable[[Any, Any], pops.Program],
    *,
    distribute_coarse: bool,
    multi_block: bool = False,
) -> tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    builder_name = getattr(program_builder, "__name__", type(program_builder).__name__)
    route = "%s/coarse=%s/blocks=%d" % (
        builder_name,
        distribute_coarse,
        2 if multi_block else 1,
    )
    _phase(route + ": resolve start")
    resolved = _resolved(
        program_builder,
        distribute_coarse=distribute_coarse,
        multi_block=multi_block,
    )
    _phase(route + ": resolve done")
    artifact = compile_resolved_plan_once(
        _COMM,
        resolved,
        route=route,
        compile_artifact=pops.compile,
    )

    _phase(route + ": execution-context bind start")
    context = pops.ExecutionContext.mpi_world(artifact)
    projected = component_execution_data(context)
    chk(
        type(context.communicator.handle) is _pops._NativeWorldCommunicator
        and context.communicator.handle is _COMM,
        "ExecutionContext carries the exact native MPI_COMM_WORLD handle",
    )
    chk(
        type(context.datatype.handle) is _pops._NativeMpiDatatype
        and _COMM.is_float64_datatype(context.datatype.handle),
        "ExecutionContext carries the exact native MPI_DOUBLE datatype",
    )
    chk(
        projected["communicator_f_handle"] == int(_COMM.fortran_handle)
        and projected["communicator_datatype_f_handle"]
        == int(context.datatype.handle.fortran_handle),
        "component execution data exposes native C++ MPI Fortran handles",
    )
    chk(
        tuple(inspect.signature(pops.ExecutionContext.mpi_world).parameters) == ("artifact",),
        "MPI context construction accepts only the compiled artifact",
    )
    runtime = pops.bind(artifact, resources={"execution_context": context})
    _phase(route + ": execution-context bind done")
    # AMR global state is level-qualified in the final contract; the old unqualified accessor is
    # deliberately rejected because it cannot identify a level on a refined hierarchy.
    block_names = ("tracer", "tracer_peer") if multi_block else ("tracer",)
    initial = np.ascontiguousarray(
        np.concatenate(
            [
                np.asarray(runtime.block_level_state_global(block, 0), dtype=np.float64)
                for block in block_names
            ]
        )
    ).copy()
    _phase(route + ": run start")
    report = pops.run(runtime, t_end=NSTEPS * DT, max_steps=NSTEPS)
    _phase(route + ": run done")
    result = np.ascontiguousarray(
        np.concatenate(
            [
                np.asarray(runtime.block_level_state_global(block, 0), dtype=np.float64)
                for block in block_names
            ]
        )
    ).copy()
    return (
        initial,
        result,
        report,
        runtime.amr.explain_regrid(),
        runtime.amr.patch_table(),
    )


def test_public_mpi_explicit_and_preset_ssprk2_are_bit_identical() -> None:
    _require_world()
    if _COMM.rank == 0:
        print("== final public AMR lifecycle under MPI: explicit SSPRK2 == preset SSPRK2 ==")

    manual_initial, manual, manual_report, manual_regrid, manual_patches = _execute(
        _explicit_ssprk2, distribute_coarse=True
    )
    preset_initial, preset, preset_report, preset_regrid, preset_patches = _execute(
        _preset_ssprk2, distribute_coarse=True
    )
    chk(
        np.array_equal(manual_initial, preset_initial),
        "both public routes bootstrap the identical global conservative state",
    )
    chk(
        np.array_equal(manual, preset),
        "np=%d: explicit and preset SSPRK2 global states are BIT-IDENTICAL (max|d|=%.3e)"
        % (int(_COMM.size), float(np.max(np.abs(manual - preset)))),
    )
    chk(
        _world_identical(manual) and _world_identical(preset),
        "both collective state_global results are byte-identical on every rank",
    )
    chk(
        manual_report.accepted_steps == preset_report.accepted_steps == NSTEPS,
        "both public runs accept exactly %d macro steps" % NSTEPS,
    )
    chk(
        manual_regrid.regrid_count > 0
        and preset_regrid.regrid_count > 0
        and manual_patches.n_levels == preset_patches.n_levels == 2
        and manual_patches.n_patches > 0
        and preset_patches.n_patches > 0,
        "both public routes complete a real regrid on a populated two-level hierarchy",
    )
    initial_mass = float(np.mean(manual_initial))
    manual_mass = float(np.mean(manual))
    preset_mass = float(np.mean(preset))
    chk(
        abs(manual_mass - initial_mass) < 1.0e-12 and abs(preset_mass - initial_mass) < 1.0e-12,
        "both distributed public routes conserve global mass to round-off",
    )


def test_public_mpi_distributed_equals_replicated_coarse() -> None:
    """The public PatchLayout authority must preserve the historical MPI layout oracle."""
    _require_world()
    replicated_initial, replicated, _, replicated_regrid, replicated_patches = _execute(
        _explicit_ssprk2, distribute_coarse=False
    )
    distributed_initial, distributed, _, distributed_regrid, distributed_patches = _execute(
        _explicit_ssprk2, distribute_coarse=True
    )
    chk(
        np.array_equal(replicated_initial, distributed_initial),
        "replicated and distributed coarse layouts start bit-identically",
    )
    chk(
        np.array_equal(replicated, distributed),
        "np=%d: distributed coarse == replicated coarse BIT-IDENTICALLY (max|d|=%.3e)"
        % (int(_COMM.size), float(np.max(np.abs(replicated - distributed)))),
    )
    chk(
        _world_identical(replicated) and _world_identical(distributed),
        "both coarse policies gather one byte-identical global array on every rank",
    )
    chk(
        replicated_regrid.regrid_count == distributed_regrid.regrid_count > 0,
        "both coarse-layout policies complete the same nonzero number of regrids",
    )
    chk(
        not any(allgather_value(_COMM, bool(replicated_patches.coarse_is_distributed)))
        and all(allgather_value(_COMM, bool(distributed_patches.coarse_is_distributed))),
        "public PatchLayout selects replicated versus genuinely distributed coarse ownership",
    )


def test_public_mpi_generated_cell_local_multiblock_regrids_and_matches_forward_euler() -> None:
    """The gate proof crosses the compiled DSO facade, never the direct prepared test runtime."""
    _require_world()
    ordinary_initial, ordinary, ordinary_report, ordinary_regrid, ordinary_patches = _execute(
        _forward_euler_pack, distribute_coarse=True, multi_block=True
    )
    local_initial, local, local_report, local_regrid, local_patches = _execute(
        _cell_local_forward_euler_pack, distribute_coarse=True, multi_block=True
    )
    chk(
        np.array_equal(ordinary_initial, local_initial),
        "ordinary and cell-local generated DSOs bootstrap the complete two-block pack identically",
    )
    chk(
        np.array_equal(ordinary, local),
        "generated cell-local FE matches ordinary FE bit-for-bit across the full two-block pack",
    )
    chk(
        _world_identical(local),
        "the distributed cell-local two-block result is byte-identical on every rank",
    )
    chk(
        ordinary_report.accepted_steps == local_report.accepted_steps == NSTEPS,
        "both generated DSO routes publish every accepted macro barrier",
    )
    chk(
        ordinary_regrid.regrid_count == local_regrid.regrid_count > 0
        and ordinary_patches.n_levels == local_patches.n_levels == 2,
        "cell-local configuration requalifies after a real distributed two-level regrid",
    )
    chk(
        abs(float(np.mean(local)) - float(np.mean(local_initial))) < 1.0e-12,
        "the generated cell-local multi-block route conserves mass to round-off",
    )


def _run_all() -> int:
    functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for function in functions:
        function()
    if _COMM.rank == 0:
        print(
            "\n%s test_amr_clean_route_program_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails)
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
