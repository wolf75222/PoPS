"""Installed public M18 witness; source collection does not qualify native execution."""

import copy

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


def _snapshot(runtime):
    # Both state_global reads are collective even though only root compares arrays.
    return (np.asarray(runtime.state_global("dual")).copy(),
            np.asarray(runtime.state_global("target")).copy())


def test_twenty_interior_targets_and_outside_cone_refusal(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, subjects = make_case()
    artifact, world = _compile(case, layout, "m18-five-node-entropy")
    context = artifact_execution_context(artifact)
    targets = target_moments(moderate_multipliers())
    zero_seed = np.zeros_like(targets)

    def bound(values):
        return pops.bind(artifact,
                         initial_values={subjects[0]: zero_seed.copy(),
                                         subjects[1]: values.copy()},
                         resources={"execution_context": context})

    accepted = bound(targets)
    errors = _failures(world, lambda: pops.run(
        accepted, t_end=.01, max_steps=1, console=False))
    assert not any(errors), errors
    after_dual, after_target = _snapshot(accepted)
    assert accepted.time() == .01 and accepted.macro_step() == 1

    def verify_accepted():
        multipliers = after_dual.reshape(3, 4, 5)
        moments = after_target.reshape(3, 4, 5)
        np.testing.assert_array_equal(moments, targets)
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

    outside = targets.copy()
    outside[:, 1, 2] = (1., 0., 1.1)
    assert cone_position(outside[:, 1, 2])[0] == "outside"
    failed = bound(outside)
    before = _snapshot(failed)
    temporal_before = copy.deepcopy(failed._executor._temporal_restart_state.to_data())
    for attempt in range(2):
        errors = _failures(world, lambda: pops.run(
            failed, t_end=.01, max_steps=1, console=False))
        assert all(row is not None and row[0] == "RuntimeError" for row in errors), errors
        assert all("coupled_implicit failed:" in row[1] for row in errors), errors
        record_property("outside_cone_failure_%d" % attempt, repr(errors))
        after = _snapshot(failed)
        assert failed.time() == 0. and failed.macro_step() == 0
        assert failed._executor._temporal_restart_state.to_data() == temporal_before
        _root_check(world, lambda after=after: [np.testing.assert_array_equal(a, b)
                                                for a, b in zip(after, before, strict=True)])
    record_property("mpi_ranks", 1 if world is None else int(world.size))
