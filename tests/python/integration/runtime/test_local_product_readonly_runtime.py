"""Native publication changes the unknown while preserving a captured State."""

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_local_product_operators_runtime import _failure_rows
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.local_product_readonly_case import make_case
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("reverse", (False, True))
def test_readonly_capture_binds_without_becoming_an_unknown_or_commit(
        isolated_native_cache, native_cxx, kokkos_root, reverse):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, subjects = make_case(reverse=reverse)
    artifact, world = _compile(case, layout, "readonly-local-product-%s" % reverse)
    assert set(artifact.arguments().instances) == {"dual", "target"}
    exact = 1. + np.arange(48, dtype=float).reshape(3, 4, 4) / 64.
    target = exact**2
    resources = {"execution_context": artifact_execution_context(artifact)}

    missing = _failure_rows(world, lambda: pops.bind(
        artifact, initial_values={subjects["dual"]: .9 * exact}, resources=resources))
    assert all(error is not None for error in missing), missing

    runtime = pops.bind(artifact, initial_values={subjects["dual"]: .9 * exact,
                                               subjects["target"]: target.copy()},
                        resources=resources)
    errors = _failure_rows(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    assert not any(errors), errors
    dual_after = np.asarray(runtime.state_global("dual"))
    target_after = np.asarray(runtime.state_global("target"))

    def check():
        actual = dual_after.reshape(exact.shape)
        np.testing.assert_allclose(actual, exact, rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual**2-target, 0., rtol=0., atol=1e-12)
        np.testing.assert_array_equal(target_after.reshape(target.shape), target)
    _root_check(world, check)
    assert runtime.time() == .01 and runtime.macro_step() == 1
