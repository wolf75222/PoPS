"""Real PoPS execution of equations reloaded from closed GraphFunction data."""
import numpy as np
import pops
import pytest

from examples.api_v02.graph_function_checks import build_graph_function_check
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize('kind,coefficient', [('serialized', 2.), ('serialized', 3.),
                                             ('field', 2.), ('capture', 2.), ('jvp', 2.), ('vjp', 2.)])
def test_native_closed_graph_function(kind, coefficient, isolated_native_cache, native_cxx,
                                      kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='GraphFunction '+kind)
    runtime, artifact, metrics = execute_case(
        lambda: build_graph_function_check(kind, coefficient=coefficient), world,
        'api-v02-graph-'+kind+'-'+str(coefficient))
    with collective_check(world):
        assert runtime.macro_step() == 1
    record_property('artifact_identity', artifact.artifact_identity.hexdigest)
    record_property('native_metrics', str(metrics))


def test_native_closed_graph_function_failure_prevents_commit(isolated_native_cache, native_cxx,
                                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='GraphFunction propagated failure')
    spec = collective_call(world, lambda: build_graph_function_check('propagate'))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route='graph-propagated-failure',
                                         compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values),
        resources={'execution_context': artifact_execution_context(artifact)}))
    before = collective_call(world, lambda: runtime.state_global('method'))
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global('method'))
    with collective_check(world):
        assert errors and all(error and 'origin=2' in error[1] for error in errors), errors
        assert runtime.time() == 0 and runtime.macro_step() == 0
        if before is not None:
            np.testing.assert_array_equal(before, after)
    record_property('artifact_identity', artifact.artifact_identity.hexdigest)
    record_property('collective_failure', str(errors))
