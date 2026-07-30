#!/usr/bin/env python3
"""Real MPI qualification of sparse Balance cadence and detached async snapshots.

Both Uniform and two-level AMR execute the public
``Case -> Program.cadence -> compile -> mpi_world -> bind -> run`` route. The Program closes one
stride-3 window every third accepted macro-step, while async Balance consumers fire every two and
three accepted steps. Held windows must therefore publish exact zero ledgers and due windows must
publish native nonzero ledgers. A separate every-step async field series proves that each worker
receives the accepted field image captured on its own tick, never the latest native state.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any

from _compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import (
        allgather_value,
        barrier,
        broadcast_value,
        rank as world_rank,
        size as world_size,
    )
    from pops.amr import (
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
    from pops.diagnostics import Balance, BalanceLedger
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.identity import make_identity
    from pops.initial import InitialCondition
    from pops.layouts import AMR, Uniform
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import Gaussian
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import (
        DiscretizationPlan,
        FiniteVolume,
        reconstruction,
        riemann,
        variables,
    )
    from pops.output import (
        AsyncScientificOutput,
        ConsumerGraph,
        NPZ,
        ParallelMode,
        read_npz,
    )
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt, every
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("async Balance MPI runtime import failed: %s" % exc)


ROOT = Path(__file__).resolve().parents[4]
N = 8
DT = 1.0e-2
NSTEPS = 6
COMM = _pops.mpi_world()
RANK = world_rank(COMM)
SIZE = world_size(COMM)


if getattr(_pops, "__has_mpi__", False) is not True:
    require_mpi_or_skip("async Balance cadence requires a native MPI build")
if SIZE != 2:
    require_mpi_or_skip("async Balance cadence requires exactly mpiexec -n 2")


def _collective_local(label: str, operation: Any) -> Any:
    result = None
    error = None
    try:
        result = operation()
    except BaseException as exc:  # noqa: BLE001 -- publish every local cause before proceeding
        error = "%s: %s" % (type(exc).__name__, exc)
    errors = allgather_value(COMM, error)
    failures = [
        "rank %d: %s" % (rank, value)
        for rank, value in enumerate(errors)
        if value is not None
    ]
    if failures:
        raise RuntimeError("%s failed: %s" % (label, "; ".join(failures)))
    return result


@contextmanager
def _shared_directory() -> Iterator[Path]:
    local = tempfile.mkdtemp(prefix="pops-async-balance-mpi-") if RANK == 0 else None
    root = Path(broadcast_value(COMM, local, root=0))
    barrier(COMM)
    try:
        yield root
    finally:
        barrier(COMM)
        if RANK == 0:
            shutil.rmtree(root, ignore_errors=True)
        barrier(COMM)


def _authored_case(*, adaptive: bool) -> tuple[pops.Case, Any]:
    label = "amr" if adaptive else "uniform"
    frame = Rectangle(
        "async-balance-%s-domain" % label,
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("async-balance-%s-model" % label, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "zero_flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    rate = model.rate("zero_rate", equation=ddt(state) == -div(flux))
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

    case = pops.Case("async-balance-%s-case" % label)
    block = case.block("tracer", model=model)
    evolved = block[state]
    case.numerics(numerics, block=block)
    program = pops.Program("async-balance-%s-program" % label)
    temporal = program.state(evolved)
    total = program.sum(temporal.n)
    zero = total * 0.0
    ledger = BalanceLedger("accepted-mass")
    program.record_balance(
        ledger,
        storage_change=total,
        outward_boundary_flux=zero,
        sources=zero,
        reflux=zero,
        projection=zero,
    )
    accepted = program.value(
        "accepted_growth",
        temporal.n + program.dt * Fraction(1, 2) * temporal.n,
        at=temporal.next.point,
    )
    program.commit(temporal.next, accepted)
    program.cadence(stride=3)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    every_step = every(1, clock=program.clock)
    every_two = every(2, clock=program.clock)
    every_three = every(3, clock=program.clock)
    root_npz = NPZ(mode=ParallelMode.ROOT)
    case.consumers(ConsumerGraph.from_consumers((
        AsyncScientificOutput(
            format=root_npz,
            schedule=every_step,
            fields=(evolved,),
            target="state_every_1",
            queue_capacity=1,
        ),
        AsyncScientificOutput(
            format=root_npz,
            schedule=every_two,
            diagnostics=(Balance(ledger, block=block, cadence=every_two),),
            target="balance_every_2",
            queue_capacity=1,
        ),
        AsyncScientificOutput(
            format=root_npz,
            schedule=every_three,
            diagnostics=(Balance(ledger, block=block, cadence=every_three),),
            target="balance_every_3",
            queue_capacity=1,
        ),
    )))
    case.initials.add(InitialCondition(
        state=evolved,
        value=Gaussian(
            frame=frame,
            center={x_axis: 0.5, y_axis: 0.5},
            background=1.0,
            amplitude=0.5,
            inverse_width=40.0,
        ),
        projection=ConservativeCellAverage(),
    ))
    grid = CartesianGrid(
        frame=frame,
        cells=(N, N),
        periodic=PeriodicAxes(frame.axes),
    )
    if not adaptive:
        return case, Uniform(grid)

    threshold = case.param(RuntimeParam(
        "async_balance_refine_threshold",
        default=1.1,
    ))
    transfer = AMRTransfer()
    transfer.state(evolved, StateTransfer())
    return case, AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(evolved) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
        patch_layout=PatchLayout(
            distribute_coarse=True,
            coarse_max_grid=4,
        ),
    )


def _artifact(*, adaptive: bool) -> Any:
    label = "amr" if adaptive else "uniform"
    case, layout = _collective_local(
        label + " authoring",
        lambda: _authored_case(adaptive=adaptive),
    )
    resolved = _collective_local(
        label + " resolution",
        lambda: pops.resolve(
            pops.validate(case),
            layout=layout,
            backend=Production(),
            compile_options={"include": str(ROOT / "include")},
        ),
    )
    return compile_resolved_plan_once(
        COMM,
        resolved,
        route="async-balance-" + label,
        compile_artifact=pops.compile,
    )


def _snapshots(path: Path) -> dict[int, Any]:
    result = {}
    for artifact in path.rglob("*.npz"):
        reopened = read_npz(artifact)
        step = int(reopened.manifest["snapshot"]["clock"]["macro_step"])
        if step in result:
            raise AssertionError("duplicate output at accepted step %d under %s" % (step, path))
        result[step] = reopened
    return result


def _coarse_values(reopened: Any) -> np.ndarray:
    snapshot = reopened.manifest["snapshot"]
    field = next(row for row in snapshot["fields"] if row["key"]["level"] == 0)
    token = make_identity("output-field", field["key"]).token
    pieces = reopened.manifest["datasets"]["fields"][token]["pieces"]
    return np.concatenate([
        np.asarray(reopened.arrays[piece["name"]]).ravel()
        for piece in sorted(pieces, key=lambda row: (row["lower"], row["upper"]))
    ])


def _balance(reopened: Any) -> tuple[float, dict[str, float]]:
    (payload,) = reopened.manifest["snapshot"]["diagnostics"]
    return (
        float.fromhex(payload["value"]),
        {name: float.fromhex(value) for name, value in payload["terms"].items()},
    )


def _verify(root: Path, *, adaptive: bool) -> None:
    if RANK != 0:
        return
    label = "amr" if adaptive else "uniform"
    case_root = root / label
    states = _snapshots(case_root / "state_every_1")
    every_two = _snapshots(case_root / "balance_every_2")
    every_three = _snapshots(case_root / "balance_every_3")
    if set(states) != set(range(1, NSTEPS + 1)):
        raise AssertionError("%s every-step async series is incomplete: %r" % (label, states))
    if set(every_two) != {2, 4, 6}:
        raise AssertionError("%s every(2) Balance cadence differs: %r" % (label, every_two))
    if set(every_three) != {3, 6}:
        raise AssertionError("%s every(3) Balance cadence differs: %r" % (label, every_three))

    images = {step: _coarse_values(reopened) for step, reopened in states.items()}
    if not np.array_equal(images[1], images[2]):
        raise AssertionError("%s stride held state changed before step 3" % label)
    if np.array_equal(images[2], images[3]):
        raise AssertionError("%s due stride window did not advance at step 3" % label)
    if not np.array_equal(images[3], images[4]) \
            or not np.array_equal(images[4], images[5]):
        raise AssertionError("%s stride held state changed between steps 3 and 6" % label)
    if np.array_equal(images[5], images[6]):
        raise AssertionError("%s due stride window did not advance at step 6" % label)

    expected_terms = {
        "storage_change",
        "outward_boundary_flux",
        "sources",
        "reflux",
        "projection",
    }
    for step in (2, 4):
        value, terms = _balance(every_two[step])
        if value != 0.0 or set(terms) != expected_terms \
                or any(term != 0.0 for term in terms.values()):
            raise AssertionError(
                "%s held step %d did not publish the exact zero Balance ledger"
                % (label, step)
            )
    for series, steps in ((every_two, (6,)), (every_three, (3, 6))):
        for step in steps:
            value, terms = _balance(series[step])
            if set(terms) != expected_terms \
                    or value <= 0.0 \
                    or terms["storage_change"] <= 0.0 \
                    or any(
                        terms[name] != 0.0
                        for name in expected_terms - {"storage_change"}
                    ):
                raise AssertionError(
                    "%s due step %d did not publish its native nonzero Balance ledger"
                    % (label, step)
                )


def _run_case(root: Path, *, adaptive: bool) -> None:
    label = "amr" if adaptive else "uniform"
    artifact = _artifact(adaptive=adaptive)
    runtime = pops.bind(
        artifact,
        resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
    )
    levels = allgather_value(COMM, int(runtime.n_levels()))
    expected_levels = 2 if adaptive else 1
    if levels != (expected_levels,) * SIZE:
        raise AssertionError("%s hierarchy differs across ranks: %r" % (label, levels))
    native_cadence = allgather_value(
        COMM,
        (
            int(runtime._executor._s.program_substeps()),
            int(runtime._executor._s.program_stride()),
        ),
    )
    if native_cadence != ((1, 3),) * SIZE:
        raise AssertionError(
            "%s did not bind the public Program cadence: %r" % (label, native_cadence)
        )
    report = pops.run(
        runtime,
        t_end=NSTEPS * DT,
        max_steps=NSTEPS,
        output_dir=root / label,
    )
    reports = allgather_value(
        COMM,
        (
            report.accepted_steps,
            report.run_identity.token,
            report.bind_identity.token,
        ),
    )
    if any(row != reports[0] for row in reports[1:]) or reports[0][0] != NSTEPS:
        raise AssertionError("%s run report differs across ranks: %r" % (label, reports))
    barrier(COMM)
    _collective_local(label + " output verification", lambda: _verify(root, adaptive=adaptive))


def main() -> None:
    with _shared_directory() as root:
        os.environ["POPS_CACHE_DIR"] = str(root / "cache")
        _run_case(root, adaptive=False)
        _run_case(root, adaptive=True)
        if RANK == 0:
            print("PASS test_async_balance_cadence_mpi")


if __name__ == "__main__":
    main()
