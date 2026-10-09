"""Installed Dim1 reception of both original M27 mixed equations."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_check


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _witness(monkeypatch):
    examples = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    monkeypatch.syspath_prepend(str(examples))
    return importlib.import_module("api040_m27_mixed_linear")


def test_mixed_linear_three_grids_and_permutation_from_saved_c_and_mu(tmp_path, monkeypatch):
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(1)
    witness = _witness(monkeypatch)
    records = witness.run_and_archive(tmp_path)
    world = witness._native_world(native)
    with collective_check(world):
        assert [(row["cells"], row["permuted"]) for row in records] == [
            (16, False), (32, False), (64, False), (16, True)]
        assert all(row["original_mass_residual"] < witness.CRITERIA[
            "original_mass_residual"] for row in records)
        assert all(row["original_chemical_residual"] < witness.CRITERIA[
            "original_chemical_residual"] for row in records)
        assert all(row["final_energy"] < row["initial_energy"] for row in records)
    failure = b""
    if witness._rank(world) == 0:
        try:
            assert (tmp_path / "receipt.json").is_file()
            assert len(tuple(tmp_path.glob("state_*.npz"))) == 4
        except Exception as exception:
            failure = (type(exception).__name__ + ": " + str(exception)).encode()
    failure = witness._broadcast_bytes(world, failure)
    if failure:
        raise AssertionError(failure.decode())


def test_one_iteration_gmres_collectively_refuses_without_accepted_publication(
        monkeypatch):
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(1)
    witness = _witness(monkeypatch)
    world = witness._native_world(native)
    initial, subject, resolved = witness._collective_call(
        world, "one-iteration case", lambda: witness._prepared_case(
            16, False, solver_iterations=1))
    if world is not None:
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

        artifact = compile_resolved_plan_once(
            world, resolved, route="M27 one-iteration original residual refusal",
            compile_artifact=pops.compile)
    else:
        artifact = pops.compile(resolved)
    resources = witness._collective_call(
        world, "execution resources", lambda: witness._execution_resources(artifact))
    runtime = witness._collective_call(
        world, "bind", lambda: pops.bind(
            artifact, initial_values={subject: initial},
            resources=resources))
    before = witness._collective_call(world, "before state", lambda: runtime.state_global(
        "concentration"))
    before_report = witness._collective_call(
        world, "before report", lambda: runtime.program_report())
    rejected = b""
    try:
        pops.run(runtime, t_end=witness.STEP, max_steps=1, console=False)
    except Exception as exception:
        rejected = (type(exception).__name__ + ": " + str(exception)).encode()
    failures = witness._allgather_bytes(world, rejected)
    after = witness._collective_call(world, "after state", lambda: runtime.state_global(
        "concentration"))
    status = witness._collective_call(world, "after status", lambda: (
        runtime.time(), runtime.macro_step(), runtime.program_report()))
    with collective_check(world):
        assert all(failures), "a rank accepted an unconverged original mixed residual"
        assert all(b"solve_linear failed: iteration_limit" in failure
                   for failure in failures), failures
        assert status[0] == 0. and status[1] == 0
        assert status[2].histories == before_report.histories
        assert status[2].diagnostics == before_report.diagnostics
        if witness._rank(world) == 0:
            np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
            np.testing.assert_array_equal(np.asarray(after).reshape(initial.shape), initial)
