#!/usr/bin/env python3
"""Public MPI AMR history, restart, regrid and coarse-distribution contracts.

Every runtime in this file is produced by the final lifecycle

``Case -> validate -> resolve -> compile -> ExecutionContext.mpi_world -> bind -> run``.

The dense AB2 rate ring and a selectively persisted depth-three state ring are checkpointed on a
real two-level hierarchy, restored into a fresh public ``RuntimeInstance``, and continued.  Rings
and global conservative state must match an uninterrupted run bit-for-bit.  The complete scenarios
are then repeated with replicated and distributed coarse layouts and must remain bit-identical.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


_REQUIRE_MPI = os.environ.get("POPS_REQUIRE_MPI_TESTS") == "1"

try:
    import numpy as np
    from mpi4py import MPI

    import pops
    import pops.lib.time as libtime
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
    from pops.time import FixedDt, Interval, every
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    if _REQUIRE_MPI:
        raise RuntimeError("required AMR history MPI contract could not import its runtime") from exc
    print("skip test_amr_history_mpi (MPI runtime unavailable: %s)" % exc)
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 2.0e-3
_C = 0.6
_COMM = MPI.COMM_WORLD
_fails = 0


ProgramFactory = Callable[[Any, Any], pops.Program]


def chk(cond: Any, label: str) -> None:
    global _fails
    if _COMM.Get_rank() == 0:
        print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _require_world() -> bool:
    size = int(_COMM.Get_size())
    if size >= 2:
        return True
    if _REQUIRE_MPI:
        raise RuntimeError("required AMR history contract did not enter MPI_COMM_WORLD size >= 2")
    if _COMM.Get_rank() == 0:
        print("skip test_amr_history_mpi (needs mpiexec -n 2; size=%d)" % size)
    return False


@contextmanager
def _shared_temporary_directory() -> Iterator[Path]:
    """Create one rank-0 path and make every collective checkpoint use exactly that path."""
    root = tempfile.mkdtemp(prefix="pops-amr-history-mpi-") if _COMM.Get_rank() == 0 else None
    shared = Path(_COMM.bcast(root, root=0))
    _COMM.Barrier()
    try:
        yield shared
    finally:
        _COMM.Barrier()
        if _COMM.Get_rank() == 0:
            shutil.rmtree(shared, ignore_errors=True)
        _COMM.Barrier()


def _model(name: str) -> tuple[Model, Any, Any, Any]:
    """One conservative scalar with zero transport and a load-bearing linear source."""
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("growth", on=state, value=(_C * rho,))
    rate = model.rate("source_rate", equation=ddt(state) == -div(flux) + source)
    return model, state, flux, rate


def _ab2_program(state: Any, rate: Any) -> pops.Program:
    program = libtime.AdamsBashforth(state, rate=rate, order=2)
    program.step_strategy(FixedDt(DT))
    return program


def _state_ring_program(state: Any, rate: Any) -> pops.Program:
    del rate
    program = pops.Program("mpi-public-state-ring")
    temporal = program.state(state)
    program.keep_history(temporal, depth=3, checkpoint_policy=Interval(2))
    next_value = program.value(
        "state_ring_next",
        temporal.n + program.dt * (_C * temporal.n) + 0.0 * temporal.prev(2),
        at=temporal.next.point,
    )
    program.commit(temporal.next, next_value)
    program.step_strategy(FixedDt(DT))
    return program


def _resolved(
    program_factory: ProgramFactory,
    *,
    distribute_coarse: bool,
    regrid_every: int,
) -> Any:
    model, state, flux, rate = _model(
        "mpi-public-history-model-%s" % program_factory.__name__.lstrip("_")
    )
    frame = model.frame
    x_axis, y_axis = frame.axes

    case = pops.Case("mpi-public-history-case-%s" % program_factory.__name__.lstrip("_"))
    evolved = case.block("blk", model)
    evolved_state = evolved[state]
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
    case.numerics(numerics, block=evolved)
    program = program_factory(evolved_state, rate)
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=evolved_state,
            value=Gaussian(
                frame=frame,
                center={x_axis: 0.5, y_axis: 0.5},
                background=1.0,
                amplitude=0.5,
                inverse_width=1.0 / (0.15 * 0.15),
            ),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("history_refine_threshold", default=1.2))
    transfer = AMRTransfer()
    transfer.state(evolved_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(evolved_state) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(regrid_every, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
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


def _compile(
    program_factory: ProgramFactory,
    *,
    distribute_coarse: bool,
    regrid_every: int,
) -> Any:
    return pops.compile(
        _resolved(
            program_factory,
            distribute_coarse=distribute_coarse,
            regrid_every=regrid_every,
        )
    )


def _bind(artifact: Any) -> Any:
    context = pops.ExecutionContext.mpi_world(artifact, _COMM)
    return pops.bind(artifact, resources={"execution_context": context})


def _advance(runtime: Any, steps: int) -> Any:
    return pops.run(
        runtime,
        t_end=float(runtime.time()) + steps * DT,
        max_steps=steps,
    )


def _rings(runtime: Any) -> dict[str, tuple[np.ndarray, ...]]:
    return {
        name: tuple(
            np.asarray(runtime.history_global(name, slot), dtype=np.float64).copy()
            for slot in range(runtime.history_depth(name))
        )
        for name in runtime.history_names()
    }


def _rings_equal(
    first: dict[str, tuple[np.ndarray, ...]],
    second: dict[str, tuple[np.ndarray, ...]],
) -> bool:
    return first.keys() == second.keys() and all(
        len(first[name]) == len(second[name])
        and all(
            np.array_equal(left, right)
            for left, right in zip(first[name], second[name], strict=True)
        )
        for name in first
    )


def _world_identical(array: np.ndarray) -> bool:
    contiguous = np.ascontiguousarray(array)
    witness = (
        tuple(contiguous.shape),
        contiguous.dtype.str,
        hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    )
    return len(set(_COMM.allgather(witness))) == 1


def _run_restart_case(
    program_factory: ProgramFactory,
    *,
    distribute_coarse: bool,
    regrid_every: int,
    half: int,
    nsteps: int,
    label: str,
) -> dict[str, Any]:
    artifact = _compile(
        program_factory,
        distribute_coarse=distribute_coarse,
        regrid_every=regrid_every,
    )

    continuous = _bind(artifact)
    _advance(continuous, half)
    continuous_rings_at_half = _rings(continuous)
    continuous_regrid_at_half = continuous.amr.explain_regrid()
    _advance(continuous, nsteps - half)
    # AMR exposes level-qualified global state; the unqualified state_global route is intentionally
    # rejected because it is ambiguous once several levels exist.  Level 0 is the deterministic
    # coarse state used for the restart parity witness.
    reference = np.asarray(
        continuous.block_level_state_global("blk", 0), dtype=np.float64
    ).copy()
    continuous_regrid = continuous.amr.explain_regrid()

    interrupted = _bind(artifact)
    _advance(interrupted, half)
    interrupted_rings_at_half = _rings(interrupted)
    interrupted_regrid_at_half = interrupted.amr.explain_regrid()
    patch_report = interrupted.amr.patch_table()
    with _shared_temporary_directory() as root:
        checkpoint = interrupted.checkpoint(root / label)
        restored = _bind(artifact)
        restored.restart(checkpoint)
        restored_rings = _rings(restored)
        restored_regrid_at_half = restored.amr.explain_regrid()
        _advance(restored, nsteps - half)
        result = np.asarray(
            restored.block_level_state_global("blk", 0), dtype=np.float64
        ).copy()
        restored_regrid = restored.amr.explain_regrid()

    return {
        "reference": reference,
        "result": result,
        "continuous_rings_at_half": continuous_rings_at_half,
        "interrupted_rings_at_half": interrupted_rings_at_half,
        "restored_rings": restored_rings,
        "continuous_regrid_at_half": continuous_regrid_at_half,
        "interrupted_regrid_at_half": interrupted_regrid_at_half,
        "restored_regrid_at_half": restored_regrid_at_half,
        "continuous_regrid": continuous_regrid,
        "restored_regrid": restored_regrid,
        "coarse_is_distributed": patch_report.coarse_is_distributed,
        "n_levels": patch_report.n_levels,
    }


def _assert_public_restart(out: dict[str, Any], *, label: str) -> None:
    chk(
        bool(out["continuous_rings_at_half"]),
        "%s exposes at least one public temporal history ring" % label,
    )
    chk(
        _rings_equal(out["continuous_rings_at_half"], out["interrupted_rings_at_half"]),
        "%s independently reaches the same pre-checkpoint rings bit-for-bit" % label,
    )
    chk(
        _rings_equal(out["continuous_rings_at_half"], out["restored_rings"]),
        "%s fresh public restart restores every history slot bit-for-bit" % label,
    )
    chk(
        np.array_equal(out["reference"], out["result"]),
        "%s uninterrupted == checkpoint/restart continuation BIT-IDENTICALLY (max|d|=%.3e)"
        % (label, float(np.max(np.abs(out["reference"] - out["result"])))),
    )
    chk(
        _world_identical(out["reference"]) and _world_identical(out["result"]),
        "%s collective global state is byte-identical on every rank" % label,
    )
    chk(
        out["n_levels"] == 2
        and out["continuous_regrid_at_half"].regrid_count > 0
        and out["restored_regrid_at_half"].regrid_count
        == out["interrupted_regrid_at_half"].regrid_count,
        "%s runs on two levels and restart preserves the exact nonzero regrid counter" % label,
    )
    chk(
        out["continuous_regrid"].regrid_count == out["restored_regrid"].regrid_count
        and out["continuous_regrid"].topology_epoch == out["restored_regrid"].topology_epoch,
        "%s continuation preserves the final regrid count and topology epoch" % label,
    )


def _assert_distributed_equals_replicated(
    replicated: dict[str, Any], distributed: dict[str, Any], *, label: str
) -> None:
    replicated_flag = bool(
        _COMM.allreduce(not bool(replicated["coarse_is_distributed"]), op=MPI.LAND)
    )
    distributed_flag = bool(
        _COMM.allreduce(bool(distributed["coarse_is_distributed"]), op=MPI.LAND)
    )
    chk(replicated_flag, "%s public replicated coarse layout is replicated on every rank" % label)
    chk(distributed_flag, "%s public distributed coarse layout owns a strict subset per rank" % label)
    chk(
        np.array_equal(replicated["result"], distributed["result"]),
        "%s distributed coarse == replicated coarse BIT-IDENTICALLY (max|d|=%.3e)"
        % (
            label,
            float(np.max(np.abs(replicated["result"] - distributed["result"]))),
        ),
    )
    chk(
        _rings_equal(replicated["restored_rings"], distributed["restored_rings"]),
        "%s restored distributed and replicated history rings are bit-identical" % label,
    )


def test_amr_history_mpi_ab2_public_restart_and_distribution_parity() -> None:
    if not _require_world():
        return
    if _COMM.Get_rank() == 0:
        print("== public MPI AB2 dense history: restart + distributed/replicated parity ==")
    replicated = _run_restart_case(
        _ab2_program,
        distribute_coarse=False,
        regrid_every=2,
        half=3,
        nsteps=6,
        label="ab2-replicated",
    )
    distributed = _run_restart_case(
        _ab2_program,
        distribute_coarse=True,
        regrid_every=2,
        half=3,
        nsteps=6,
        label="ab2-distributed",
    )
    _assert_public_restart(replicated, label="AB2 replicated")
    _assert_public_restart(distributed, label="AB2 distributed")
    _assert_distributed_equals_replicated(replicated, distributed, label="AB2")


def test_amr_history_mpi_in_window_regrid_public_restart_and_distribution_parity() -> None:
    if not _require_world():
        return
    if _COMM.Get_rank() == 0:
        print("== public MPI selective state ring: in-window regrid replay + layout parity ==")
    replicated = _run_restart_case(
        _state_ring_program,
        distribute_coarse=False,
        regrid_every=4,
        half=6,
        nsteps=10,
        label="state-ring-replicated",
    )
    distributed = _run_restart_case(
        _state_ring_program,
        distribute_coarse=True,
        regrid_every=4,
        half=6,
        nsteps=10,
        label="state-ring-distributed",
    )
    _assert_public_restart(replicated, label="state-ring replicated")
    _assert_public_restart(distributed, label="state-ring distributed")
    _assert_distributed_equals_replicated(replicated, distributed, label="state-ring")


def _run_all() -> int:
    functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for function in functions:
        function()
    if _COMM.Get_rank() == 0:
        print(
            "\n%s test_amr_history_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails)
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
