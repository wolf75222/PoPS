#!/usr/bin/env python3
"""Two-rank public bind-only regression for initial distributed coarse ownership.

The authored 64x64 AMR layout covers both public ``coarse_max_grid`` contracts:
an explicit 16x16 cap, and ``None`` whose native sentinel selects a 32x32 cap.
Both use the default public Morton space-filling-curve authority.  No call to
:func:`pops.run` is made: the complete contract is the hierarchy materialized
by the public ``compile -> ExecutionContext.mpi_world -> bind -> local_boxes``
route.  This detects a lowering/runtime regression where the cap is ignored
before the first accepted step and an MPI rank receives no base box.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import pops
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    from pops import _pops
    from pops._native_collectives import allgather_value
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
    require_mpi_or_skip("initial coarse ownership MPI runtime import failed: %s" % exc)


ROOT = Path(__file__).resolve().parents[4]
N = 64
COARSE_MAX_GRID = 16
DEFAULT_COARSE_MAX_GRID = N // 2
DT = 1.0e-3
COMM = _pops.mpi_world()
RANK = int(COMM.rank)
SIZE = int(COMM.size)


if getattr(_pops, "__has_mpi__", False) is not True:
    require_mpi_or_skip("initial coarse ownership requires an MPI-enabled native build")
if SIZE != 2:
    require_mpi_or_skip(
        "initial coarse ownership requires exactly mpiexec -n 2 (observed %d)" % SIZE
    )


def _resolved(coarse_max_grid: int | None) -> Any:
    """Author one ordinary public AMR case; execution deliberately stops at bind."""
    frame = Rectangle(
        "initial-coarse-ownership-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("initial-coarse-ownership-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.25 * rho,), y_axis: (-0.125 * rho,)},
        waves={x_axis: (0.25,), y_axis: (-0.125,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))

    case = pops.Case("initial-coarse-ownership-case")
    block = case.block("tracer", model)
    tracer = block[state]
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
    case.numerics(numerics, block=block)

    program = pops.Program("initial-coarse-ownership-forward-euler")
    temporal = program.state(tracer)
    stage = StagePoint("main_stage", {"main": TimePoint(program.clock, 0)})
    rhs = program.value("rhs", rate(temporal.n), at=stage)
    accepted = program.value(
        "accepted", temporal.n + program.dt * rhs, at=temporal.next.point
    )
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=tracer,
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
    transfer.state(tracer, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(tracer) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
        patch_layout=PatchLayout(
            distribute_coarse=True,
            coarse_max_grid=coarse_max_grid,
        ),
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )


def _expected_cover(coarse_max_grid: int) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    return {
        ((x, y), (x + coarse_max_grid, y + coarse_max_grid))
        for y in range(0, N, coarse_max_grid)
        for x in range(0, N, coarse_max_grid)
    }


def _expected_y_half(
    rank: int, coarse_max_grid: int
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    y_start = rank * (N // 2)
    return {
        ((x, y), (x + coarse_max_grid, y + coarse_max_grid))
        for y in range(y_start, y_start + N // 2, coarse_max_grid)
        for x in range(0, N, coarse_max_grid)
    }


def _bind(*, coarse_max_grid: int | None, route: str) -> Any:
    resolved = _resolved(coarse_max_grid)
    artifact = compile_resolved_plan_once(
        COMM,
        resolved,
        route=route,
        compile_artifact=pops.compile,
    )
    return pops.bind(
        artifact,
        resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
    )


def _check(condition: bool, label: str) -> None:
    passed = all(bool(value) for value in allgather_value(COMM, condition))
    if RANK == 0:
        print("  [%s] %s" % ("OK " if passed else "XX ", label), flush=True)
    if not passed:
        raise AssertionError(label)


def _check_case(
    *,
    label: str,
    authored_coarse_max_grid: int | None,
    native_coarse_max_grid: int,
) -> None:
    runtime = _bind(
        coarse_max_grid=authored_coarse_max_grid,
        route="mpi-amr-initial-coarse-ownership-" + label,
    )
    expected = _expected_cover(native_coarse_max_grid)
    expected_local_count = len(expected) // SIZE
    _check(runtime.macro_step() == 0, "bind leaves the accepted macro-step at zero")

    local = tuple(runtime.local_boxes("tracer"))
    _check(
        runtime.macro_step() == 0,
        "local-box inspection does not advance the accepted macro-step",
    )
    _check(
        len(local) == expected_local_count,
        "%s gives each rank %d initial coarse tiles" % (label, expected_local_count),
    )
    gathered = tuple(
        tuple((tuple(lower), tuple(upper)) for lower, upper in rank_boxes)
        for rank_boxes in allgather_value(COMM, local)
    )
    global_boxes = tuple(box for rank_boxes in gathered for box in rank_boxes)
    _check(
        len(global_boxes) == len(expected) and set(global_boxes) == expected,
        "%s gives the exact public 64x64 cover without holes or overlap" % label,
    )

    # The public Morton SFC gives each rank its contiguous half of this equal-weight lattice.
    _check(
        set(local) == _expected_y_half(RANK, native_coarse_max_grid),
        "%s uses the public Morton-SFC contiguous y-half ownership" % label,
    )


def main() -> None:
    _check_case(
        label="explicit-16",
        authored_coarse_max_grid=COARSE_MAX_GRID,
        native_coarse_max_grid=COARSE_MAX_GRID,
    )
    _check_case(
        label="default-half-domain",
        authored_coarse_max_grid=None,
        native_coarse_max_grid=DEFAULT_COARSE_MAX_GRID,
    )
    if RANK == 0:
        print("PASS test_amr_initial_coarse_ownership_mpi", flush=True)


if __name__ == "__main__":
    main()
