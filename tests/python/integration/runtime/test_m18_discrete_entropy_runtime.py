"""Installed public M18 witness; source collection does not qualify native execution."""

import copy
from pathlib import Path

import numpy as np
import pops
import pytest

from examples.migration.scientific.api040_m18_entropy import (
    make_case, moderate_multipliers, target_moments,
)
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.discrete_entropy_oracle import (
    BASIS, WEIGHTS, cone_position, entropy,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.m18_entropy_receipts import (
    save_entropy_provenance, save_entropy_snapshot, save_entropy_execution_owner,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _failures(world, action):
    failure = None
    try:
        action()
    except Exception as error:
        failure = (type(error).__name__, str(error))
    if world is None:
        return (failure,)
    from pops._native_collectives import allgather_value
    return allgather_value(world, failure)


def _all_check(world, operation):
    """Converge rank-local assertions before any following collective check."""
    errors = _failures(world, operation)
    assert not any(errors), errors


def test_twenty_interior_targets_and_outside_cone_refusal(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, subjects = make_case()
    artifact, world = _compile(case, layout, "m18-five-node-entropy")
    directory = collective_directory(world, tmp_path / "m18-entropy")
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    from examples.migration.scientific import api040_m18_entropy as example
    identity = save_entropy_provenance(world, artifact, directory, fixture=__file__, example=example.__file__)
    owner_metadata = save_entropy_execution_owner(world, artifact, directory, fixture=__file__)
    record_property("execution_owner_metadata", str(owner_metadata))
    targets = target_moments(moderate_multipliers())
    zero_seed = np.zeros_like(targets)

    def bound(values):
        return collective_call(world, lambda: pops.bind(artifact,
                         initial_values={subjects[0]: zero_seed.copy(),
                                         subjects[1]: values.copy()},
                         resources={"execution_context": context}))

    accepted = bound(targets)
    save_entropy_snapshot(world, accepted, artifact, directory, "initial")
    errors = _failures(world, lambda: pops.run(
        accepted, t_end=.01, max_steps=1, console=False))
    assert not any(errors), errors
    after_dual, after_target = state_snapshots(accepted, world, ("dual", "target"))
    checkpoint, _ = save_entropy_snapshot(world, accepted, artifact, directory, "accepted")
    def verify_accepted_clock():
        assert accepted.time() == .01 and accepted.macro_step() == 1
    _all_check(world, verify_accepted_clock)

    def verify_accepted():
        multipliers = after_dual.reshape(3, 4, 5)
        moments = after_target.reshape(3, 4, 5)
        np.testing.assert_array_equal(moments, targets)
        assert moments.tobytes() == targets.tobytes()
        for x, y in np.ndindex(4, 5):
            population = WEIGHTS * np.exp(BASIS.T @ multipliers[:, x, y])
            assert np.all(np.isfinite(population)) and np.all(population > 0)
            assert cone_position(targets[:, x, y])[0] == "interior"
            np.testing.assert_allclose(BASIS@population, targets[:, x, y],
                                       rtol=0, atol=2.e-11)
            # A nonzero nullspace perturbation preserves the moments and has
            # strictly larger primal entropy at the interior minimizer.
            _, _, right = np.linalg.svd(BASIS, full_matrices=True)
            varied = population + right[3:].T @ np.asarray((.01, -.005))
            assert np.min(varied) > 0
            assert entropy(varied) > entropy(population) + 1.e-7
        np.testing.assert_allclose(multipliers, moderate_multipliers(),
                                   rtol=0, atol=1.e-8)
    _root_check(world, verify_accepted)

    collective_call(world, lambda: pops.run(accepted, t_end=.02, max_steps=1, console=False))
    save_entropy_snapshot(world, accepted, artifact, directory, "continuous")
    continuous = state_snapshots(accepted, world, ("dual", "target"))
    restored = bound(targets)
    collective_call(world, lambda: restored.restart(checkpoint))
    save_entropy_snapshot(world, restored, artifact, directory, "restored")
    restored_states = state_snapshots(restored, world, ("dual", "target"))
    _root_check(world, lambda: [np.testing.assert_array_equal(a, b) for a, b in
                               zip(restored_states, (after_dual, after_target), strict=True)])
    with collective_check(world):
        assert restored.time() == .01 and restored.macro_step() == 1
    collective_call(world, lambda: pops.run(restored, t_end=.02, max_steps=1, console=False))
    save_entropy_snapshot(world, restored, artifact, directory, "replayed")
    replayed = state_snapshots(restored, world, ("dual", "target"))
    _root_check(world, lambda: [np.testing.assert_array_equal(a, b) for a, b in
                               zip(replayed, continuous, strict=True)])

    outside = targets.copy()
    outside[:, 1, 2] = (1., 0., 1.1)
    assert cone_position(outside[:, 1, 2])[0] == "outside"
    failed = bound(outside)
    before = state_snapshots(failed, world, ("dual", "target"))
    save_entropy_snapshot(world, failed, artifact, directory, "outside_before")
    with collective_check(world):
        temporal_before = copy.deepcopy(failed._executor._temporal_restart_state.to_data())
    for attempt in range(2):
        errors = _failures(world, lambda: pops.run(
            failed, t_end=.01, max_steps=1, console=False))
        assert all(row is not None and row[0] == "RuntimeError" for row in errors), errors
        assert all("coupled_implicit failed:" in row[1] for row in errors), errors
        record_property("outside_cone_failure_%d" % attempt, repr(errors))
        after = state_snapshots(failed, world, ("dual", "target"))
        def verify_local_rollback():
            assert failed.time() == 0. and failed.macro_step() == 0
            assert failed._executor._temporal_restart_state.to_data() == temporal_before
        _all_check(world, verify_local_rollback)
        _root_check(world, lambda after=after: [np.testing.assert_array_equal(a, b)
                                                for a, b in zip(after, before, strict=True)])
        _root_check(world, lambda after=after: [pytest.fail("rollback changed exact buffer bytes")
            for a, b in zip(after, before, strict=True) if a.tobytes() != b.tobytes()])
        save_entropy_snapshot(world, failed, artifact, directory, "outside_rejected_%d" % attempt, failures=errors)

    # An impossible immutable target cannot be cured by reducing dt. This is an
    # explicit fresh bind, not an accepted retry of the outside-cone instance.
    safe = bound(targets)
    save_entropy_snapshot(world, safe, artifact, directory, "safe_rebind_initial")
    collective_call(world, lambda: pops.run(safe, t_end=.01, max_steps=1, console=False))
    save_entropy_snapshot(world, safe, artifact, directory, "safe_rebind_accepted")
    safe_states = state_snapshots(safe, world, ("dual", "target"))
    _root_check(world, lambda: [np.testing.assert_array_equal(a, b) for a, b in
                               zip(safe_states, (after_dual, after_target), strict=True)])
    with collective_check(world):
        assert safe.time() == .01 and safe.macro_step() == 1
        record_property("dim", 2)
        record_property("rank", 0 if world is None else int(world.rank))
        record_property("artifact_identity", artifact.artifact_identity.token)
        record_property("native_sha256", identity["native_sha256"])
        record_property("native_abi_key", identity["abi_key"])
        record_property("native_capability_abi", identity["native_capabilities"]["abi_version"])
        record_property("saved_receipts", str(Path(directory)))
    record_property("mpi_ranks", 1 if world is None else int(world.size))
