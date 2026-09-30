"""Native public constant-gradient receipts; runs unchanged under installed MPI.

Source-only review can resolve/emit the fixture. These tests require a rebuilt
native package and compile its real conservative face operator, without injected fluxes.
"""
import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_diffusion_case import (
    DIFFUSIVITY, DT, LEFT0, Q0, build_diffusion_integral_case,
)
from tests.python.support.integral_state_receipts import (
    collective_directory, save_public_snapshot, selected_external_records,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
TOL = 2e-12


def _world():
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator
    return _pops.mpi_world() if native_mpi_communicator(_pops) == "MPI_COMM_WORLD" else None


def _compile(world, case, layout):
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    if world is None:
        return pops.compile(resolved)
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
    return compile_resolved_plan_once(world, resolved, route="integral-diffusion",
                                      compile_artifact=pops.compile)


def _bind(world, artifact, initial):
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    return collective_call(world, lambda: pops.bind(artifact,
        initial_values={subject: initial},
        resources={"execution_context": artifact_execution_context(artifact)}))


def _check_ledger(ledgers, right, left, kind):
    groups = {}
    for ledger in ledgers:
        assert abs(ledger["quantities"][right.identity][1] - (Q0 + DIFFUSIVITY * DT)) < TOL
        assert abs(ledger["quantities"][left.identity][1] - (LEFT0 - DIFFUSIVITY * DT)) < TOL
        exterior = selected_external_records(ledger, side=0) + selected_external_records(ledger)
        assert set(ledger["consumed"]) == {
            (r["operation"], r["occurrence"], r["context"], r["quadrature"]) for r in exterior}
        for row in exterior:
            assert row["axis"] == 0 and row["component"] == 0
            assert row["orientation"] == (-1 if row["side"] == 0 else 1)
            assert row["multiplicity"] == 1
            assert abs(row["flux"] - DIFFUSIVITY) < TOL
            # Read the runtime frame using its clock length, not a slash assumption.
            frame = row["context"].removeprefix("pops.exchange.frame.v1/")
            length, suffix = frame.split(":", 1)
            fields = suffix[int(length) + 1:].split("/")
            level, substep = int(fields[1]), int(fields[2])
            groups.setdefault((row["side"], level, substep), []).append(row)
    expected = {(0, 0, 0): (32, DT), (1, 1, 0): (64, DT/2), (1, 1, 1): (64, DT/2)} \
        if kind == "amr2" else {(0, 0, 0): (8, DT), (1, 0, 0): (8, DT)}
    assert set(groups) == set(expected), {key: len(rows) for key, rows in groups.items()}
    for key, (count, weight) in expected.items():
        rows = groups[key]
        assert len(rows) == count
        assert all(abs(row["weight"] - weight) < 1e-17 for row in rows)
        assert all(abs(row["measure"] - 1/count) < 1e-17 for row in rows)
        amount = sum(row["orientation"] * row["measure"] * row["flux"] * row["weight"]
                     for row in rows)
        assert abs(amount - (-1 if key[0] == 0 else 1) * DIFFUSIVITY * weight) < TOL


@pytest.mark.parametrize("kind", ("uniform", "amr2"))
def test_public_diffusive_integral_constant_gradient_balance_and_restart(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property, kind):
    world = _world()
    case, layout, right, left, initial = collective_call(world, lambda: build_diffusion_integral_case(kind))
    artifact = _compile(world, case, layout)
    runtime = _bind(world, artifact, initial)
    directory = collective_directory(world, tmp_path)
    with collective_check(world):
        assert runtime.integral_state(right) == Q0 and runtime.integral_state(left) == LEFT0
    if kind == "amr2":
        from tests.python.support.amr_snapshots import composite_active_mask
        mask = collective_call(world, lambda: composite_active_mask(runtime, 0, refinement_ratio=2))
        with collective_check(world):
            assert runtime.n_levels() == 2
            assert mask.any() and (~mask).any() and not mask[:, -1].any()
            assert mask[:, 0].all(), "left face must remain coarse-owned for the balance oracle"
    _, before = save_public_snapshot(world, runtime, artifact, right, directory, "initial",
                                     amr=kind == "amr2")
    if kind == "amr2":
        from tests.python.support.amr_snapshots import level_valid_mask
        valid = collective_call(world, lambda: level_valid_mask(runtime, 1, refinement_ratio=2))
        fine = collective_call(world, lambda: runtime.block_level_state_global("fluid", 1))
        with collective_check(world):
            if world is None or int(world.rank) == 0:
                np.testing.assert_allclose(before[0], initial.reshape(-1), rtol=0, atol=2e-14)
                expected = np.broadcast_to(1. + (np.arange(64) + .5) / 64, (64, 64))
                np.testing.assert_allclose(np.asarray(fine).reshape(64, 64)[valid],
                                           expected[valid], rtol=0, atol=2e-14)
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    checkpoint, accepted = save_public_snapshot(world, runtime, artifact, right, directory,
                                                "accepted", amr=kind == "amr2")
    with collective_check(world):
        assert abs(runtime.integral_state(right) - (Q0 + DIFFUSIVITY * DT)) < TOL
        assert abs(runtime.integral_state(left) - (LEFT0 - DIFFUSIVITY * DT)) < TOL
        assert abs(runtime.integral_state(right) + runtime.integral_state(left) - (Q0 + LEFT0)) < TOL
        if world is None or int(world.rank) == 0:
            np.testing.assert_allclose(accepted[0], before[0], rtol=0, atol=TOL)
            _check_ledger(accepted[2], right, left, kind)
    restarted = _bind(world, artifact, initial)
    collective_call(world, lambda: restarted.restart(checkpoint))
    _, restored = save_public_snapshot(world, restarted, artifact, right, directory,
                                       "restored", amr=kind == "amr2")
    with collective_check(world):
        assert restarted.time() == DT and restarted.macro_step() == 1
        assert restarted.integral_state(right) == runtime.integral_state(right)
        assert restarted.integral_state(left) == runtime.integral_state(left)
        if world is None or int(world.rank) == 0:
            np.testing.assert_array_equal(restored[0], accepted[0])
            assert restored[1] == accepted[1]
    record_property("integral_diffusion_receipts", str(directory))
    record_property("artifact_identity", artifact.artifact_identity.token)


@pytest.mark.parametrize("failure", ("periodic_trace", "unsafe_dt"))
def test_public_diffusive_integral_refusal_restores_state_clock_and_mailbox(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property, failure):
    world = _world()
    periodic = failure == "periodic_trace"
    case, layout, right, left, initial = collective_call(world, lambda: build_diffusion_integral_case(
        proposed_dt=DT if periodic else .5, selected_axis=1 if periodic else 0))
    artifact = _compile(world, case, layout)
    runtime = _bind(world, artifact, initial)
    directory = collective_directory(world, tmp_path)
    _, before = save_public_snapshot(world, runtime, artifact, right, directory, "before")
    _, errors = collective_attempt(world, lambda: pops.run(runtime,
        t_end=DT if periodic else .5, max_steps=1, console=False))
    with collective_check(world):
        diagnostic = "no unconsumed face contribution" if periodic else "combined_transport_diffusion_stability"
        assert all(error and diagnostic in error[1] for error in errors), errors
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert runtime.integral_state(right) == Q0 and runtime.integral_state(left) == LEFT0
    _, rejected = save_public_snapshot(world, runtime, artifact, right, directory, "rejected")
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            np.testing.assert_array_equal(rejected[0], before[0])
            assert rejected[1] == before[1]
    if not periodic:
        collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
        _, retry = save_public_snapshot(world, runtime, artifact, right, directory, "retry")
        with collective_check(world):
            assert abs(runtime.integral_state(right) - (Q0 + DIFFUSIVITY * DT)) < TOL
            assert abs(runtime.integral_state(left) - (LEFT0 - DIFFUSIVITY * DT)) < TOL
            if world is None or int(world.rank) == 0:
                _check_ledger(retry[2], right, left, "uniform")
    record_property("integral_diffusion_receipts", str(directory))
