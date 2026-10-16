"""Receive only after the integration owner builds and freezes the native package."""
import numpy as np
import pops
import pytest

from examples.api_v02.array_derivative_v14 import build_array_derivative_v14
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile
from tests.python.support.public_native_profile import compile_profile_plan_once
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize('count', (0, 1, 3, 7))
def test_dynamic_array_weighted_gather_transpose(count, isolated_native_cache, native_cxx,
                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='dynamic Array AD v14 '+str(count))
    _, artifact, metrics = execute_case(lambda: build_array_derivative_v14(count), world,
                                       'array-derivative-v14-'+str(count))
    record_property('artifact_identity', artifact.artifact_identity.hexdigest)
    record_property('native_metrics', str(metrics))


@pytest.mark.parametrize('count', (0, 3))
def test_equal_cardinality_foreign_domain_rejected_even_when_empty(count, isolated_native_cache,
                                                                  native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='Array AD exact domain '+str(count))
    spec = collective_call(world, lambda: build_array_derivative_v14(count, wrong_domain=True))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route='array-ad-domain-'+str(count),
                                        compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=dict(spec.initial_values),
        resources={'execution_context': artifact_execution_context(artifact)}))
    before = collective_call(world, lambda: runtime.state_global('method'))
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global('method'))
    with collective_check(world):
        assert errors and all(error is not None for error in errors), errors
        assert runtime.macro_step() == 0 and runtime.time() == 0
        if before is not None: np.testing.assert_array_equal(before, after)
