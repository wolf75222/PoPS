"""Public W11 composition: nontrivial volume/history/current, parent refusal and retry."""
import json
from pathlib import Path
import numpy as np
import pops
import pytest
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory, save_public_snapshot
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.support.w11_child_trace_case import build_case, oracle, Q0, CELLS

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
TOL = 3e-13


def _history(world, runtime, directory, phase):
    report = collective_call(world, runtime.program_report)
    images = {}
    with collective_check(world):
        rows = report.histories
        assert rows, "the witness must configure an actual child history"
    for row in rows:
        if row["initialized"]:
            for slot in range(row["depth"]):
                value = collective_call(world, lambda: runtime.history_global(row["name"], slot))
                with collective_check(world):
                    if world is None or world.rank == 0:
                        images[row["name"]+"/"+str(slot)] = np.asarray(value).copy()
    with collective_check(world):
        if world is None or world.rank == 0:
            np.savez(directory/(phase+"-history.npz"), **images)
            (directory/(phase+"-history.json")).write_text(json.dumps(rows, sort_keys=True, indent=2)+"\n")
    return rows, images


@pytest.mark.parametrize("ssprk2", (False, True))
def test_public_child_current_is_provisional_until_parent_accepts(
        tmp_path, ssprk2, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

    native = select_native_dimension(2)
    world = native.mpi_world() if native_mpi_communicator(native) == "MPI_COMM_WORLD" else None
    rank = 0 if world is None else int(world.rank)
    directory = collective_directory(world, tmp_path/"w11-current")
    case, layout, initial, quantity = collective_call(world, lambda: build_case(
        ssprk2=ssprk2, guard_parent=True))
    validated = collective_call(world, lambda: pops.validate(case))
    resolved = collective_call(world, lambda: pops.resolve(validated, layout=layout))
    artifact = (pops.compile(resolved) if world is None else
                compile_resolved_plan_once(world, resolved, route="W11 child current",
                                           compile_artifact=pops.compile))
    collective_call(world, lambda: retain_v_provenance(artifact, native, directory, rank))
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values={subject: initial},
        resources={"execution_context": artifact_execution_context(artifact)}))
    _, before = save_public_snapshot(world, runtime, artifact, quantity, directory, "before")
    before_history, before_images = _history(world, runtime, directory, "before")
    _, failures = collective_attempt(world, lambda: pops.run(
        runtime, t_end=.1, max_steps=1, console=False))
    with collective_check(world):
        assert all(failure and "parent_reject_after_child_current" in failure[1]
                   for failure in failures), failures
        # The earlier compiled q>Q0 guard passed: current was consumed before this parent refusal.
        assert runtime.integral_state(quantity) == Q0
        assert runtime.time() == 0 and runtime.macro_step() == 0
    _, rejected = save_public_snapshot(world, runtime, artifact, quantity, directory, "rejected")
    rejected_history, rejected_images = _history(world, runtime, directory, "rejected")
    with collective_check(world):
        assert rejected_history == before_history
        if world is None or world.rank == 0:
            np.testing.assert_array_equal(rejected[0], before[0])
            assert rejected[1] == before[1]  # Whole accepted exchange image, including consumption keys.
            assert set(rejected_images) == set(before_images)
            for key in before_images:
                np.testing.assert_array_equal(rejected_images[key], before_images[key])

    expected, charge = oracle(initial, .08, ssprk2=ssprk2)
    result = collective_call(world, lambda: pops.run(runtime, t_end=.08, max_steps=1, console=False))
    with collective_check(world):
        assert result.accepted_steps == 1
        assert abs(runtime.integral_state(quantity)-charge) < TOL
        assert abs(runtime.time()-.08) < TOL and runtime.macro_step() == 1
    _, accepted = save_public_snapshot(world, runtime, artifact, quantity, directory, "accepted")
    accepted_history, accepted_images = _history(world, runtime, directory, "accepted")
    with collective_check(world):
        assert all(row["initialized"] and row["fill_count"] == 2 for row in accepted_history)
        if world is None or world.rank == 0:
            np.testing.assert_allclose(accepted[0].reshape(initial.shape), expected, rtol=0, atol=TOL)
            assert np.max(np.abs(expected-initial)) > .001
            assert accepted_images
            (directory/"acceptance.json").write_text(json.dumps({
                "scope": "W11 finite FV/capacity composition; not M14 kinetic sheath",
                "scheme": "SSPRK2" if ssprk2 else "Euler", "cells": CELLS,
                "parent_dt_rejected": .1, "parent_dt_accepted": .08,
                "q0": Q0, "q_after_retry": charge, "tolerance": TOL,
                "parent_refusal": failures, "history_fill_count": 2,
            }, sort_keys=True, indent=2)+"\n")
