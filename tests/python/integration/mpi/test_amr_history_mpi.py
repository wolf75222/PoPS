#!/usr/bin/env python3
"""Public MPI AMR history, restart, regrid and coarse-distribution contracts.

Every runtime in this file is produced by the final lifecycle

``Case -> validate -> resolve -> compile -> ExecutionContext.mpi_world -> bind -> run``.

The dense AB2 rate ring and a selectively authored state ring are checkpointed on a real two-level
hierarchy, restored into a fresh public ``RuntimeInstance``, and continued. A selective window that
straddles a regrid must expose ``dense_regrid_safety`` effective storage. Rings and global conservative
state must match an uninterrupted run bit-for-bit under replicated and distributed coarse layouts.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from _compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import allgather_value, barrier, broadcast_value
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
    require_mpi_or_skip("AMR history MPI runtime import failed: %s" % exc)


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 2.0e-3
_C = 0.6
_COMM = _pops.mpi_world()
_fails = 0


ProgramFactory = Callable[[Any, Any], pops.Program]


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
    require_mpi_or_skip("AMR history needs mpiexec -n 2; size=%d" % size)


@contextmanager
def _shared_temporary_directory() -> Iterator[Path]:
    """Create one rank-0 path and make every collective checkpoint use exactly that path."""
    root = tempfile.mkdtemp(prefix="pops-amr-history-mpi-") if _COMM.rank == 0 else None
    shared = Path(broadcast_value(_COMM, root, root=0))
    barrier(_COMM)
    try:
        yield shared
    finally:
        barrier(_COMM)
        if _COMM.rank == 0:
            shutil.rmtree(shared, ignore_errors=True)
        barrier(_COMM)


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


def _state_ring_program(state: Any, _rate: Any) -> pops.Program:
    program = pops.Program("mpi-public-state-ring")
    temporal = program.state(state)
    # The resulting native ring contains current + three lagged slots.  Interval(3) persists both
    # replay anchors (slots 0 and 3); Interval(2) would omit the oldest slot and is correctly refused.
    program.keep_history(temporal, depth=3, checkpoint_policy=Interval(3))
    next_value = program.value(
        "state_ring_next",
        temporal.n + program.dt * _C * temporal.n + 0.0 * temporal.prev(2),
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


def _bind(artifact: Any) -> Any:
    context = pops.ExecutionContext.mpi_world(artifact)
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


def _ring_diff_summary(
    first: dict[str, tuple[np.ndarray, ...]],
    second: dict[str, tuple[np.ndarray, ...]],
) -> str:
    rows = []
    for name in sorted(set(first) | set(second)):
        if name not in first or name not in second:
            rows.append("%s:missing" % name)
            continue
        if len(first[name]) != len(second[name]):
            rows.append("%s:slots=%d/%d" % (name, len(first[name]), len(second[name])))
            continue
        for slot, (left, right) in enumerate(
            zip(first[name], second[name], strict=True)
        ):
            if not np.array_equal(left, right):
                rows.append(
                    "%s[%d]:max|d|=%.17g,left=[%.17g,%.17g],right=[%.17g,%.17g]" % (
                        name,
                        slot,
                        float(np.max(np.abs(left - right))),
                        float(np.min(left)),
                        float(np.max(left)),
                        float(np.min(right)),
                        float(np.max(right)),
                    )
                )
    return ", ".join(rows) if rows else "none"


def _world_identical(array: np.ndarray) -> bool:
    contiguous = np.ascontiguousarray(array)
    witness = (
        tuple(contiguous.shape),
        contiguous.dtype.str,
        hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    )
    return len(set(allgather_value(_COMM, witness))) == 1


def _run_restart_case(
    program_factory: ProgramFactory,
    *,
    distribute_coarse: bool,
    regrid_every: int,
    half: int,
    nsteps: int,
    label: str,
) -> dict[str, Any]:
    factory_name = getattr(program_factory, "__name__", type(program_factory).__name__)
    route = "%s/coarse=%s/regrid=%d" % (
        factory_name,
        distribute_coarse,
        regrid_every,
    )
    resolved = _resolved(
        program_factory,
        distribute_coarse=distribute_coarse,
        regrid_every=regrid_every,
    )
    artifact = compile_resolved_plan_once(
        _COMM,
        resolved,
        route=route,
        compile_artifact=pops.compile,
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
        history_storage: dict[str, dict[str, Any]] = {}
        with np.load(checkpoint, allow_pickle=False) as payload:
            rank_count = int(payload["n_ranks"])
            level_count = int(payload["n_levels"])
            assert rank_count == int(_COMM.size)
            assert "program_accepted_state" not in payload.files
            assert not any(name.startswith("dmap_") and not name.startswith("dmap_rank_")
                           for name in payload.files)
            for rank in range(rank_count):
                state_key = "program_accepted_state_rank_%d" % rank
                assert state_key in payload.files
                assert payload[state_key].dtype == np.dtype("uint8")
                assert payload[state_key].ndim == 1
                for level in range(level_count):
                    dmap_key = "dmap_rank_%d_level_%d" % (rank, level)
                    assert dmap_key in payload.files
                    assert payload[dmap_key].dtype.kind in "iu"
                    assert payload[dmap_key].ndim == 1
            for raw_name in payload["history_names"]:
                name = str(raw_name)
                fp_key = "history_regrid_steps_" + name
                history_storage[name] = {
                    "requested": tuple(int(v) for v in payload[
                        "history_requested_stored_slots_" + name]),
                    "stored": tuple(int(v) for v in payload[
                        "history_stored_slots_" + name]),
                    "mode": str(payload["history_storage_mode_" + name]),
                    "regrid_steps": (
                        tuple(int(v) for v in payload[fp_key])
                        if fp_key in payload.files else None
                    ),
                }
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
        "history_storage": history_storage,
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
        "%s fresh public restart restores every history slot bit-for-bit (%s)"
        % (
            label,
            _ring_diff_summary(
                out["continuous_rings_at_half"], out["restored_rings"]
            ),
        ),
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


def _assert_dense_regrid_safety(out: dict[str, Any], *, label: str) -> None:
    rows = out["history_storage"]
    chk(bool(rows) and all(
        row["mode"] == "dense_regrid_safety"
        and len(row["requested"]) < len(row["stored"])
        and row["regrid_steps"]
        for row in rows.values()),
        "%s exposes selective intent, dense safety storage and its regrid schedule: %r"
        % (label, rows))


def _assert_distributed_equals_replicated(
    replicated: dict[str, Any], distributed: dict[str, Any], *, label: str
) -> None:
    replicated_flag = all(allgather_value(
        _COMM, not bool(replicated["coarse_is_distributed"])
    ))
    distributed_flag = all(allgather_value(
        _COMM, bool(distributed["coarse_is_distributed"])
    ))
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
    _require_world()
    if _COMM.rank == 0:
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
    _require_world()
    if _COMM.rank == 0:
        print("== public MPI selective state ring: dense regrid safety + layout parity ==")
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
    _assert_dense_regrid_safety(replicated, label="state-ring replicated")
    _assert_dense_regrid_safety(distributed, label="state-ring distributed")
    _assert_distributed_equals_replicated(replicated, distributed, label="state-ring")


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
            "\n%s test_amr_history_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails)
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
