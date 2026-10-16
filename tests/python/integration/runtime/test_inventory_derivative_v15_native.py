"""Native receiver for JVP of the complete checked inventory conversion.

This test records execution separately from source emission; no result is implied
by defining it. The local derivative oracle is independent of PoPS AD.
"""
import math
import pops
import pytest
from examples.api_v02.inventory_publication_v15 import DT
from tests.python.support.api_v02_native import _compile, _bind
from tests.python.support.api_v02_native import continuation_snapshot, recorded_attempt, assert_typed_failure
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _close(actual, expected):
    assert type(actual) is float and math.isfinite(actual), actual
    assert abs(actual-float(expected)) <= 4.e-14, (actual, expected)


def _check_derivative(spec, image, step):
    root = _decode(spec, image)['root']
    initial_q = .5
    initial_inventory = initial_q+initial_q*initial_q
    discriminant = 1+4*initial_inventory
    q = (math.sqrt(discriminant+8*step)-1)/2
    old_q = (math.sqrt(discriminant+8*(step-1))-1)/2
    # Independent implicit differentiation of Q(q_new)=Q(q_old)+2.
    expected_derivative = (1+2*old_q)/(1+2*q)
    old_derivative = 0. if step == 1 else math.sqrt(discriminant+8*(step-2))/math.sqrt(discriminant+8*(step-1))
    assert set(root) == {'q','inventory','derivative','time','step','history','record'}
    assert root['time'] == DT*step and root['step'] == step
    _close(root['q'], q); _close(root['inventory'], initial_inventory+2*step)
    _close(root['derivative'], expected_derivative)
    assert set(root['history']) == {'q','derivative'}
    _close(root['history']['q'], old_q); _close(root['history']['derivative'], old_derivative)
    assert set(root['record']) == {'defect','target'}
    _close(root['record']['defect'], 0.); _close(root['record']['target'], initial_inventory+2*step)
    # A tangent substituted for the primal inventory cannot satisfy both checks.
    assert abs(root['derivative']-root['q']) > .2
    return root


def test_inventory_derivative_preserves_primal_proof_and_restart(tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    from examples.api_v02.derivative_inventory_v15 import build_derivative_inventory_v15
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('C12_derivative', builder=build_derivative_inventory_v15)
    original = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(original, t_end=DT, max_steps=1, console=False))
    first = collective_call(world, lambda: _image(original))
    with collective_check(world):
        _check_derivative(spec, first, 1)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: original.checkpoint(str(directory/'inventory_derivative')))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = collective_call(world, lambda: _image(resumed))
    with collective_check(world):
        assert restored == first
        _check_derivative(spec, restored, 1)
    for runtime in (original, resumed):
        collective_call(world, lambda runtime=runtime: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    images = [collective_call(world, lambda runtime=runtime: _image(runtime)) for runtime in (original, resumed)]
    with collective_check(world):
        roots = [_check_derivative(spec, image, 2) for image in images]
        assert roots[0] == roots[1]


def test_inventory_derivative_refuses_selection_boundary_without_publication(isolated_native_cache, native_cxx, kokkos_root):
    from examples.api_v02.derivative_inventory_v15 import build_derivative_inventory_v15
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('C12_derivative_boundary', builder=build_derivative_inventory_v15,
                                     initial_q=0.)
    runtime = _bind(spec, artifact, world)
    before = collective_call(world, lambda: continuation_snapshot(runtime, world))
    error, failures = recorded_attempt(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    after = collective_call(world, lambda: continuation_snapshot(runtime, world))
    with collective_check(world):
        assert before.status == 'uninitialized'
        assert failures and all(failures)
        assert_typed_failure(error, world, tag='differential', code=91001)
        assert runtime.time() == 0 and runtime.macro_step() == 0 and before == after
