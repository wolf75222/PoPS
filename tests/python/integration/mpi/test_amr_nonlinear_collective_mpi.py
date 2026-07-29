#!/usr/bin/env python3
"""Two-rank regression for collective nonlinear AMR failure and rollback.

The full nonlinear oracle is owned by
``integration/amr/test_amr_newton_full.py``.  This MPI entrypoint reuses that
exact authored case and adds only the distributed bug contract: a finite
no-root value stored by one fine-patch owner must fail the local nonlinear solve
on every rank, then restore both accepted AMR levels, the clock, and topology.

This is intentionally one artifact, one bind, and one rejected step.  Broader
Strang/MPI, GPU, and performance qualification belongs to explicit non-routine
campaigns rather than the normal MPI CI plan.
"""
from __future__ import annotations

from typing import Any
import sys

from _compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import allgather_value
    from tests.python.integration.amr import test_amr_newton_full as nonlinear_case
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("Python MPI runtime import failed: %s" % exc)


_COMM = _pops.mpi_world()
_fails = 0
NO_ROOT_VALUE = -2.0 / (
    4.0 * nonlinear_case.DT * nonlinear_case.REACTION_RATE
)


def chk(condition: bool, label: str) -> None:
    """Record one all-rank check so every process exits with the same status."""
    global _fails
    flags = tuple(bool(row) for row in allgather_value(_COMM, bool(condition)))
    passed = all(flags)
    if int(_COMM.rank) == 0:
        print("  [%s] %s" % ("OK " if passed else "XX ", label), flush=True)
    if not passed:
        _fails += 1


def _require_world() -> None:
    size = int(_COMM.size)
    if size == 2:
        return
    require_mpi_or_skip(
        "needs exactly mpiexec -n 2; MPI_COMM_WORLD size=%d" % size
    )


def _state(runtime: Any, level: int) -> np.ndarray:
    side = nonlinear_case.N * (2**level)
    return np.asarray(
        runtime.block_level_state_global("reactant", level),
        dtype=np.float64,
    ).reshape(side, side)


def _compile_bind() -> Any:
    resolved, program = nonlinear_case._resolved(None)
    nonlinear_case._program_solve_contract(program)
    artifact = compile_resolved_plan_once(
        _COMM,
        resolved,
        route="mpi-amr-nonlinear-collective",
        compile_artifact=pops.compile,
    )
    return pops.bind(
        artifact,
        resources={
            "execution_context": pops.ExecutionContext.mpi_world(artifact)
        },
    )


def _rank_owned_fine_target(runtime: Any) -> tuple[int, int, int]:
    local = tuple(runtime._executor.output_state_local_pieces("reactant", 1))
    local_indices = tuple(int(row["global_box_index"]) for row in local)
    inventories = tuple(
        tuple(int(index) for index in row)
        for row in allgather_value(_COMM, local_indices)
    )
    targets = sorted({index for row in inventories for index in row})
    if not targets:
        raise AssertionError("the nonlinear MPI case has no owned fine patch")
    target = targets[0]
    owners = [
        rank for rank, indices in enumerate(inventories) if target in indices
    ]
    if len(owners) != 1:
        raise AssertionError(
            "fine patch %d must have one owner, got %r" % (target, owners)
        )
    owner = owners[0]
    if int(_COMM.rank) == owner:
        piece = next(
            row for row in local if int(row["global_box_index"]) == target
        )
        jlo, ilo = (int(value) for value in piece["lower"])
        jhi, ihi = (int(value) for value in piece["upper"])
        target_j = (jlo + jhi - 1) // 2
        target_i = (ilo + ihi - 1) // 2
    else:
        target_j = target_i = -1
    locations = allgather_value(_COMM, (target_j, target_i))
    target_j, target_i = locations[owner]
    if target_j < 0 or target_i < 0:
        raise AssertionError("fine patch owner did not publish an injection cell")
    return owner, target_j, target_i


def _local_has_fault(runtime: Any) -> bool:
    return any(
        np.any(
            np.asarray(piece["values"], dtype=np.float64) == NO_ROOT_VALUE
        )
        for piece in runtime._executor.output_state_local_pieces("reactant", 1)
    )


def test_rank_local_nonlinear_failure_rolls_back_collectively() -> None:
    _require_world()
    if int(_COMM.rank) == 0:
        print("== nonlinear AMR rank-local failure under two-rank MPI ==")
    runtime = _compile_bind()
    chk(
        runtime.n_levels() == 2 and bool(runtime.patch_boxes()),
        "the shared nonlinear case materializes a populated two-level hierarchy",
    )

    failure_rank, target_j, target_i = _rank_owned_fine_target(runtime)
    local_state = _state(runtime, 1).copy()
    if int(_COMM.rank) == failure_rank:
        local_state[target_j, target_i] = NO_ROOT_VALUE
    runtime._executor.set_block_level_state(
        "reactant",
        1,
        np.ascontiguousarray(local_state),
    )
    fault_locations = tuple(
        bool(row) for row in allgather_value(_COMM, _local_has_fault(runtime))
    )
    chk(
        sum(fault_locations) == 1 and fault_locations[failure_rank],
        "the no-root value exists in native storage on exactly one owner rank",
    )

    before = tuple(_state(runtime, level).copy() for level in (0, 1))
    time_before = runtime.time()
    step_before = runtime.macro_step()
    topology_before = tuple(runtime.patch_boxes())
    owners_before = tuple(
        tuple(runtime._executor.level_owner_ranks(level))
        for level in (0, 1)
    )

    error = None
    try:
        pops.run(
            runtime,
            t_end=nonlinear_case.DT,
            max_steps=1,
            console=False,
        )
    except RuntimeError as exc:
        error = str(exc)
    errors = tuple(allgather_value(_COMM, error))
    chk(
        all(
            item is not None
            and any(
                "local_nonlinear failed: %s" % status in item
                for status in (
                    "iteration_limit",
                    "singular",
                    "invalid_evaluation",
                )
            )
            and "action=fail_run" in item
            for item in errors
        ),
        "one rank-local no-root value fails the nonlinear step on every rank",
    )
    chk(
        len(set(errors)) == 1,
        "all ranks report one identical collective nonlinear failure",
    )

    after = tuple(_state(runtime, level) for level in (0, 1))
    chk(
        runtime.time() == time_before == 0.0
        and runtime.macro_step() == step_before == 0,
        "collective FailRun leaves the accepted clock unchanged",
    )
    chk(
        all(
            np.array_equal(actual, expected)
            for actual, expected in zip(after, before, strict=True)
        ),
        "collective FailRun restores both accepted AMR levels bit-for-bit",
    )
    chk(
        tuple(runtime.patch_boxes()) == topology_before
        and tuple(
            tuple(runtime._executor.level_owner_ranks(level))
            for level in (0, 1)
        )
        == owners_before,
        "collective FailRun preserves patch topology and ownership",
    )


def _run_all() -> int:
    functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for function in functions:
        function()
    if int(_COMM.rank) == 0:
        print(
            "\n%s test_amr_nonlinear_collective_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails),
            flush=True,
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
