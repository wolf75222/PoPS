#!/usr/bin/env python3
"""Two-rank fail-closed proof for the operational RegridOnRestart transaction.

Rank one first injects a local accepted-state validation failure immediately before the restart
route's first composite reduction. Every rank must receive one coherent failure and roll back
without entering a mismatched collective. The same fresh runtime then retries successfully,
rebinds the AB2 history/lagged-flux topology, publishes one common transformed-hierarchy receipt,
and carries that continuation identity into the next run manifest.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
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
    from tests.python.integration.amr.test_amr_regrid_on_restart import (
        DT,
        NSTEPS,
        _resolved,
    )
    from tests.python.integration.runtime.test_shared_interface_runtime import (
        _assert_same_shared_interface_image,
        _resolve_shared_interface_amr,
        _shared_interface_accepted_image,
        _shared_interface_amr_authoring,
    )
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("RegridOnRestart MPI runtime import failed: %s" % exc)


_COMM = _pops.mpi_world()
_fails = 0


def chk(condition: Any, label: str) -> None:
    global _fails
    if int(_COMM.rank) == 0:
        print("  [%s] %s" % ("OK " if condition else "XX ", label), flush=True)
    if not condition:
        _fails += 1


def _require_world() -> None:
    if int(_COMM.size) < 2:
        require_mpi_or_skip(
            "RegridOnRestart collective proof needs mpiexec -n 2; size=%d" % int(_COMM.size)
        )


@contextmanager
def _shared_temporary_directory() -> Iterator[Path]:
    root = tempfile.mkdtemp(prefix="pops-regrid-restart-mpi-") if int(_COMM.rank) == 0 else None
    shared = Path(broadcast_value(_COMM, root, root=0))
    barrier(_COMM)
    try:
        yield shared
    finally:
        barrier(_COMM)
        if int(_COMM.rank) == 0:
            shutil.rmtree(shared, ignore_errors=True)
        barrier(_COMM)


def _bind(artifact):
    context = pops.ExecutionContext.mpi_world(artifact)
    return pops.bind(artifact, resources={"execution_context": context})


def _accepted_image(runtime):
    native = runtime._executor._s
    levels = int(runtime.n_levels())
    histories = tuple(
        (
            str(name),
            tuple(
                np.asarray(runtime.history_global(name, slot), dtype=np.float64).copy()
                for slot in range(int(runtime.history_depth(name)))
            ),
        )
        for name in runtime.history_names()
    )
    return {
        "time": float(runtime.time()),
        "step": int(runtime.macro_step()),
        "boxes": tuple(tuple(int(value) for value in box) for box in runtime.patch_boxes()),
        "states": tuple(
            np.asarray(
                runtime.block_level_state_global("tracer", level),
                dtype=np.float64,
            ).copy()
            for level in range(levels)
        ),
        "histories": histories,
        "program_state": bytes(native.program_accepted_state()),
        "regrid_count": int(native.checkpoint_regrid_count()),
        "topology_epoch": int(native.checkpoint_topology_epoch()),
        "run_identity": runtime.last_run_identity,
        "consumer_cursors": runtime.consumer_cursors.to_data(),
    }


def _same_image(left, right) -> bool:
    arrays = {"states", "histories"}
    histories_equal = (
        [name for name, _slots in left["histories"]]
        == [name for name, _slots in right["histories"]]
        and all(
            len(current_slots) == len(recorded_slots)
            and all(
                np.array_equal(current, recorded)
                for current, recorded in zip(current_slots, recorded_slots, strict=True)
            )
            for (_name, current_slots), (_expected_name, recorded_slots) in zip(
                left["histories"], right["histories"], strict=True
            )
        )
    )
    return (
        {key: value for key, value in left.items() if key not in arrays}
        == {key: value for key, value in right.items() if key not in arrays}
        and len(left["states"]) == len(right["states"])
        and all(
            np.array_equal(current, recorded)
            for current, recorded in zip(left["states"], right["states"], strict=True)
        )
        and histories_equal
    )


def test_regrid_on_restart_mpi_collective_rollback_and_lineage() -> None:
    _require_world()
    if int(_COMM.rank) == 0:
        print("== RegridOnRestart two-rank collective rollback + lineage ==", flush=True)

    resolved = _resolved()
    artifact = compile_resolved_plan_once(
        _COMM,
        resolved,
        route="regrid-on-restart-mpi",
        compile_artifact=pops.compile,
    )
    source = _bind(artifact)
    report = pops.run(
        source,
        t_end=NSTEPS * DT,
        max_steps=NSTEPS,
        console=False,
    )
    chk(report.accepted_steps == NSTEPS, "source reaches the exact accepted checkpoint boundary")
    chk(bool(tuple(source.history_names())), "source checkpoint carries a real multistep history")

    with _shared_temporary_directory() as root:
        checkpoint = source.checkpoint(root / "moving-profile")
        restarted = _bind(artifact)
        rollback_image = _accepted_image(restarted)

        from pops.runtime import _amr_checkpoint_contract as contract

        original_validation = contract.validate_restored_contract

        def fail_on_rank_one(sim, payload):
            if int(_COMM.rank) == 1:
                raise RuntimeError("injected rank-local pre-collective validation failure")
            return original_validation(sim, payload)

        contract.validate_restored_contract = fail_on_rank_one
        caught = False
        try:
            restarted.restart(checkpoint)
        except RuntimeError:
            caught = True
        finally:
            contract.validate_restored_contract = original_validation

        chk(
            all(allgather_value(_COMM, caught)),
            "one rank-local validation fault fails every rank coherently",
        )
        chk(
            _same_image(_accepted_image(restarted), rollback_image),
            "failed collective restart restores each rank's complete accepted image",
        )

        restart_identity = restarted.restart(checkpoint)
        continuation_identity = restarted.last_run_identity
        receipt = restarted._executor.last_restart_regrid_receipt()
        chk(
            restart_identity == restarted.last_restart_identity,
            "successful retry publishes the authenticated restart identity",
        )
        chk(
            receipt["changed"] is True
            and receipt["before"]["topology_identity"] != receipt["after"]["topology_identity"],
            "successful retry performs one real structural hierarchy transform",
        )
        chk(
            np.allclose(
                [row["value"] for row in receipt["composite_integrals_after"]],
                [row["value"] for row in receipt["composite_integrals_before"]],
                rtol=2.0e-12,
                atol=2.0e-13,
            ),
            "the collective hierarchy transform conserves every block component",
        )
        chk(
            len(
                set(
                    allgather_value(
                        _COMM,
                        (
                            restart_identity.token,
                            continuation_identity.token,
                            json.dumps(
                                receipt,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                        ),
                    )
                )
            )
            == 1,
            "restart identity, continuation identity and receipt agree on every rank",
        )

        continued = pops.run(
            restarted,
            t_end=float(restarted.time()) + DT,
            max_steps=1,
            console=False,
        )
        manifest = restarted._executor.last_run_manifest
        chk(
            manifest.continuation_identity == continuation_identity,
            "the next run manifest authenticates the transformed continuation lineage",
        )
        chk(
            continued.accepted_steps == 1
            and continued.run_identity != source.last_run_identity
            and all(
                np.all(np.isfinite(np.asarray(restarted.history_global(name, slot))))
                for name in restarted.history_names()
                for slot in range(int(restarted.history_depth(name)))
            ),
            "the transformed multistep continuation advances without colliding with its source",
        )


def test_regrid_on_restart_mpi_shared_interface_collective_rollback_and_retry() -> None:
    """Cross the exact RegridOnRestart + refined shared-interface MPI seam."""

    _require_world()
    if int(_COMM.rank) == 0:
        print("== RegridOnRestart two-rank refined shared-interface transaction ==", flush=True)

    with _shared_temporary_directory() as root:
        component_root = root / ("component-rank-%d" % int(_COMM.rank))
        authoring = _shared_interface_amr_authoring(
            root / "authoring",
            component_root=component_root,
        )
        resolved = _resolve_shared_interface_amr(authoring, max_levels=2)
        artifact = compile_resolved_plan_once(
            _COMM,
            resolved,
            route="regrid-on-restart-shared-interface-mpi",
            compile_artifact=pops.compile,
        )
        interface = resolved.blocks[0].numerics.boundaries[0].interfaces[0]
        initial_values = {
            authoring.core.tracer_state: authoring.left_initial,
            authoring.right_state: authoring.right_initial,
        }

        source = authoring.example._bind_artifact(
            artifact,
            initial_values=initial_values,
            params=authoring.params,
        )
        initial_integral = source.integral("tracer") + source.integral("right")
        source_report = pops.run(
            source,
            t_end=1.0e-3,
            max_steps=1,
            console=False,
            output_dir=root / "source-output",
        )
        source_counts = tuple(
            source._executor._s._interface_evaluation_count(interface.qualified_id, level)
            for level in range(2)
        )
        chk(
            source_report.accepted_steps == 1
            and source.n_levels() == 2
            and source_counts == (2, 4),
            "the source executes the two-level shared interface before checkpoint",
        )
        checkpoint_integral = source.integral("tracer") + source.integral("right")
        chk(
            np.isclose(checkpoint_integral, initial_integral, rtol=0.0, atol=2.0e-13),
            "the source shared-interface step is conservative",
        )
        checkpoint = source.checkpoint(root / "accepted-shared-interface")

        restarted = authoring.example._bind_artifact(
            artifact,
            initial_values=initial_values,
            params=authoring.params,
        )
        priming_report = pops.run(
            restarted,
            t_end=1.0e-3,
            max_steps=1,
            console=False,
            output_dir=root / "candidate-output",
        )
        chk(priming_report.accepted_steps == 1, "the restart candidate owns an accepted image")
        rollback_image = _shared_interface_accepted_image(restarted)

        from pops.runtime import _amr_checkpoint_v3 as checkpoint_codec

        original_conservation_check = checkpoint_codec._require_restart_conservation
        transformed_boxes = None

        def fail_after_native_regrid(before, after):
            nonlocal transformed_boxes
            del before, after
            transformed_boxes = tuple(
                tuple(int(value) for value in row) for row in restarted.patch_boxes()
            )
            if int(_COMM.rank) == 1:
                raise RuntimeError(
                    "injected rank-local shared-interface restart validation failure"
                )

        checkpoint_codec._require_restart_conservation = fail_after_native_regrid
        caught = False
        caught_message = ""
        try:
            restarted.restart(checkpoint)
        except RuntimeError as error:
            caught = True
            caught_message = str(error)
        finally:
            checkpoint_codec._require_restart_conservation = original_conservation_check

        chk(
            all(allgather_value(_COMM, caught))
            and all(
                "injected rank-local shared-interface restart validation failure" in message
                for message in allgather_value(_COMM, caught_message)
            ),
            "one post-regrid rank-local fault fails every rank coherently",
        )
        transformed_rows = allgather_value(_COMM, transformed_boxes)
        chk(
            transformed_boxes is not None
            and len(set(transformed_rows)) == 1
            and transformed_boxes != rollback_image["boxes"],
            "the rejected attempt crossed one rank-consensus structural transform",
        )
        rollback_ok = True
        try:
            _assert_same_shared_interface_image(restarted, rollback_image)
        except AssertionError:
            rollback_ok = False
        chk(
            all(allgather_value(_COMM, rollback_ok)),
            "the failed shared-interface restart restores every accepted rank image exactly",
        )

        restart_identity = restarted.restart(checkpoint)
        continuation_identity = restarted.last_run_identity
        receipt = restarted._executor.last_restart_regrid_receipt()
        chk(
            receipt is not None
            and receipt["changed"] is True
            and tuple(restarted.patch_boxes()) == transformed_boxes,
            "retry commits the same transformed hierarchy",
        )
        collective_identity = (
            restart_identity.token,
            continuation_identity.token,
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        chk(
            len(set(allgather_value(_COMM, collective_identity))) == 1,
            "restart identity, continuation identity and topology receipt agree on every rank",
        )
        restarted_integral = restarted.integral("tracer") + restarted.integral("right")
        chk(
            np.allclose(
                [row["value"] for row in receipt["composite_integrals_after"]],
                [row["value"] for row in receipt["composite_integrals_before"]],
                rtol=2.0e-12,
                atol=2.0e-13,
            )
            and np.isclose(
                restarted_integral,
                checkpoint_integral,
                rtol=2.0e-12,
                atol=2.0e-13,
            ),
            "the committed shared-interface transform is conservative",
        )

        counts_before = tuple(
            restarted._executor._s._interface_evaluation_count(interface.qualified_id, level)
            for level in range(2)
        )
        continued = pops.run(
            restarted,
            t_end=float(restarted.time()) + 1.0e-3,
            max_steps=1,
            console=False,
        )
        counts_after = tuple(
            restarted._executor._s._interface_evaluation_count(interface.qualified_id, level)
            for level in range(2)
        )
        count_delta = tuple(
            after - before
            for before, after in zip(counts_before, counts_after, strict=True)
        )
        collective_count_delta = allgather_value(_COMM, count_delta)
        chk(
            continued.accepted_steps == 1
            and count_delta == (2, 4)
            and len(set(collective_count_delta)) == 1,
            "the retried hierarchy resumes the rematerialized shared interface on every rank",
        )
        continued_integral = restarted.integral("tracer") + restarted.integral("right")
        chk(
            np.isclose(
                continued_integral,
                checkpoint_integral,
                rtol=0.0,
                atol=2.0e-13,
            ),
            "the post-restart shared-interface continuation remains conservative",
        )


def _run_all() -> int:
    test_regrid_on_restart_mpi_collective_rollback_and_lineage()
    test_regrid_on_restart_mpi_shared_interface_collective_rollback_and_retry()
    if int(_COMM.rank) == 0:
        print(
            "\n%s test_amr_regrid_on_restart_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails),
            flush=True,
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
