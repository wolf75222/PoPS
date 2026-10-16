"""Prepared native reception; execute only after integration and a new installed identity."""
import numpy as np
import pops
import pytest

from examples.api_v02.implicit_relation_checks import build_implicit_check
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize('mode', ('jvp', 'vjp', 'bad_base', 'bad_gauge', 'bad_tangent', 'omit_constraints',
                                 'singular', 'nondifferentiable', 'budget'))
def test_native_implicit_relation_derivatives(mode, isolated_native_cache, native_cxx,
                                            kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='implicit derivative '+mode)
    runtime, artifact, metrics = execute_case(lambda: build_implicit_check(mode), world,
                                             'implicit-derivative-'+mode)
    with collective_check(world):
        assert runtime.macro_step() == 1
    record_property('artifact_identity', artifact.artifact_identity.hexdigest)
    record_property('native_metrics', str(metrics))


def test_native_wrong_tangent_prevents_commit(isolated_native_cache, native_cxx,
                                            kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='unhandled tangent defect')
    spec = collective_call(world, lambda: build_implicit_check('bad_tangent', recover=False))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route='implicit-wrong-tangent', compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values), resources={'execution_context': artifact_execution_context(artifact)}))
    before = collective_call(world, lambda: runtime.state_global('method'))
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global('method'))
    with collective_check(world):
        assert errors and all(error and 'typed Result failure' in error[1] for error in errors), errors
        assert runtime.time() == 0 and runtime.macro_step() == 0
        if before is not None: np.testing.assert_array_equal(before, after)
    record_property('artifact_identity', artifact.artifact_identity.hexdigest)
    record_property('collective_failure', str(errors))
