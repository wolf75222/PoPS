"""Deferred installed-PoPS reception of mandatory relation checks and nested methods.

Authoring tests do not qualify these results. Run only after the v09 Python package,
SDK and both native variants are built and authenticated by the parent workflow.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.relations import SolveFailure
from examples.api_v02.relation_newton_krylov import build_nested_relation
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import (
    _compile, _bind, _require_owned_array,
)
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _run(**kwargs):
    spec, artifact, world = _compile("nested_relation", builder=build_nested_relation, **kwargs)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=.1, max_steps=1, console=False))
    actual = collective_call(world, lambda: runtime.state_global("relation"))
    _require_owned_array(world, actual)
    if actual is not None:
        actual = np.asarray(actual).reshape(6, 4, 4)
    return runtime, world, actual


@pytest.mark.parametrize("target", ((1., 2.), (.6, 1.4)))
def test_newton_cg_armijo_selected_root_with_independent_oracle(
        target, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, world, actual = _run(target=target)
    with collective_check(world):
        assert runtime.time() == .1 and runtime.macro_step() == 1
        if actual is not None:
            expected = np.broadcast_to(np.array(target)[:, None, None], (2, 4, 4))
            np.testing.assert_allclose(actual[:2], expected, rtol=0, atol=2.e-10)
            np.testing.assert_array_equal(actual[2:4], 0.)
            assert np.all(actual[4] > 0) and np.all(actual[5] > 0)
            # This residual is an independent post-run evaluation of the original
            # equation, not the algorithm's recursive CG/Newton estimate.
            x, y = actual[:2]; tx, ty = target
            residual = np.array([x*x*x+x+.25*y-(tx**3+tx+.25*ty),
                                 y*y*y+2*y+.25*x-(ty**3+2*ty+.25*tx)])
            assert np.max(abs(residual)) <= 1.01e-10
            record_property("selected_root_error", float(np.max(abs(actual[:2]-expected))))
            record_property("original_residual_norm", float(np.max(abs(residual))))
            record_property("line_search_reductions", float(actual[5,0,0]))


@pytest.mark.parametrize("options,code,origin", (
    ({"mode": "fabricated_residual"}, SolveFailure.ORIGINAL_RESIDUAL, 1),
    ({"mode": "negative_root"}, SolveFailure.UNSELECTED, 1),
    ({"mode": "inadmissible"}, SolveFailure.INADMISSIBLE, 1),
    ({"mode": "bad_request"}, SolveFailure.REQUEST_REJECTED, 1),
    ({"mode": "bad_evidence"}, SolveFailure.EVIDENCE_OUTSIDE_DOMAIN, 1),
    ({"mode": "bad_accuracy"}, SolveFailure.INVALID_ACCURACY, 1),
    ({"mode": "bad_boundary"}, SolveFailure.BOUNDARY, 1),
    ({"mode": "bad_norm"}, SolveFailure.INVALID_NORM, 1),
    ({"inner_budget": 0}, 1, 2),
    ({"line_budget": 0}, 1, 2),
))
def test_no_algorithm_success_bypasses_original_problem_checks(
        options, code, origin, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, world, actual = _run(**options)
    with collective_check(world):
        # The caller explicitly handles Failure and publishes the unchanged guess
        # plus its error. It does not turn Failure into a successful solve result.
        assert runtime.time() == .1 and runtime.macro_step() == 1
        if actual is not None:
            expected = np.broadcast_to(np.array([.2,.2,int(code),origin,0.,0.])[:,None,None], actual.shape)
            np.testing.assert_array_equal(actual, expected)


def test_request_failure_preserves_observation_payload_through_solve(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, world, actual = _run(mode="request_payload_failure")
    with collective_check(world):
        assert runtime.time() == .1 and runtime.macro_step() == 1
        if actual is not None:
            # Both values originate in request_test's native error payload.
            expected = np.broadcast_to(np.array([3.,2.5,2201.,1.,0.,0.])[:,None,None], actual.shape)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2.e-10)


def test_unhandled_inner_solve_failure_cannot_publish_an_outer_candidate(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("nested_relation", builder=build_nested_relation,
                                      inner_budget=0, recover=False)
    runtime = _bind(spec, artifact, world)
    before = collective_call(world, lambda: runtime.state_global("relation"))
    _require_owned_array(world, before)
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=.1, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global("relation"))
    with collective_check(world):
        assert failures and all(failures), failures
        assert runtime.time() == 0. and runtime.macro_step() == 0
        if before is not None:
            np.testing.assert_array_equal(before, after)
