"""Prepared true-PoPS C12 receivers. No acceptance is implied by this file.

Each parameter compiles once. Checkpoint/restart and uninterrupted progression
reuse that exact artifact. Expected values are independent rational/stdlib oracles,
never supplied to the Program or its decoder.
"""
from fractions import Fraction
import math

import pops
import pytest

from examples.api_v02.inventory_publication_v15 import build_inventory_publication_v15, DT
from tests.python.support.api_v02_native import _compile, _bind, recorded_attempt, typed_failure_values
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _expected(mode, step):
    if step == 0:
        return dict(q=0., inventory=Fraction(0), delta=Fraction(0), defect=Fraction(0), target=Fraction(0))
    if mode == 'dynamic_replacement':
        inventory = Fraction(2*step-1)
        return dict(q=float(inventory), inventory=inventory,
                    delta=Fraction(1 if step == 1 else 2), defect=Fraction(0), target=inventory)
    if mode == 'nonlinear_tolerated':
        inventory = Fraction(171, 100)+2*(step-1)
        q = .9 if step == 1 else (math.sqrt(1+4*float(inventory))-1)/2
        defect = Fraction(-29, 100) if step == 1 else Fraction(0)
        return dict(q=q, inventory=inventory, delta=Fraction(2), defect=defect,
                    target=inventory-defect)
    inventory = Fraction(2*step)
    return dict(q=(math.sqrt(1+4*float(inventory))-1)/2, inventory=inventory,
                delta=Fraction(2), defect=Fraction(0), target=inventory)


def _close(actual, expected):
    assert type(actual) is float and math.isfinite(actual), actual
    assert abs(actual-float(expected)) <= 4.e-14, (actual, expected)


def _check(spec, image, mode, step):
    decoded = _decode(spec, image)
    root = decoded['root']; expected = _expected(mode, step); prior = _expected(mode, step-1)
    assert set(root) == {'q', 'inventory', 'time', 'step', 'history', 'record', 'controller'}
    assert root['step'] == step and root['time'] == DT*step
    assert root['controller'] == {'accepted': step}
    _close(root['q'], expected['q']); _close(root['inventory'], expected['inventory'])
    assert set(root['record']) == {'delta', 'defect', 'target'}
    for key in root['record']:
        _close(root['record'][key], expected[key])
    assert set(root['history']) == {'q', 'inventory', 'defect'}
    for key in root['history']:
        _close(root['history'][key], prior[key])
    # This rules out both a stale q and applying the measured defect a second time.
    Q = root['q'] if mode == 'dynamic_replacement' else root['q']+root['q']**2
    assert abs(Q-root['inventory']) <= 4.e-14
    assert abs(root['inventory']-float(prior['inventory'])-root['record']['delta']-root['record']['defect']) <= 4.e-14
    assert root['q'] > prior['q']
    if mode == 'nonlinear_tolerated' and step == 1:
        assert abs(root['record']['defect']) > .28
        assert abs(root['q']-1.) > .09
        assert abs(root['q']-(.9-.29)) > .28
    return root


@pytest.mark.parametrize('mode', ('dynamic_replacement', 'nonlinear_exact', 'nonlinear_tolerated'))
def test_inventory_publication_exact_paths_defect_and_restart(mode, tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('C12_'+mode, builder=build_inventory_publication_v15, mode=mode)
    assert not spec.case._time._block_indices() and not spec.initial_values
    original = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(original, t_end=DT, max_steps=1, console=False))
    first = collective_call(world, lambda: _image(original))
    with collective_check(world):
        _check(spec, first, mode, 1)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: original.checkpoint(str(directory/('inventory_'+mode))))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = collective_call(world, lambda: _image(resumed))
    with collective_check(world):
        assert restored == first
        _check(spec, restored, mode, 1)
    for runtime in (original, resumed):
        collective_call(world, lambda runtime=runtime: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    images = [collective_call(world, lambda runtime=runtime: _image(runtime)) for runtime in (original, resumed)]
    with collective_check(world):
        roots = [_check(spec, image, mode, 2) for image in images]
        assert roots[0] == roots[1]
        if mode == 'dynamic_replacement':
            # First path retains (q1,J1)=(1,1), second path retains (q2,J2)=(3,2).
            # No stale (q2,J2)=(2,2) or unmodified state can satisfy this oracle.
            assert roots[0]['history']['q'] == 1. and roots[0]['q'] == 3.
            assert roots[0]['record']['delta'] == 2.


def _assert_cause(mode, signature, payloads):
    assert payloads and signature['vtype'] == 'variant'
    tag = 'inventory_residual' if mode == 'bad_inverse' else 'inventory_decode'
    child = dict(signature['members'])[tag]
    assert child['vtype'] == 'product'
    fields = dict(child['members'])
    if mode == 'bad_inverse':
        assert {key: value['vtype'] for key, value in fields.items()} == {'defect': 'scalar', 'bound': 'scalar'}
        # On the second attempt Q1=2, target=4, bad decoder q=0: R=-4.
        expected = {'tag': tag, 'value': {'defect': -4., 'bound': 1.e-12}}
    else:
        assert {key: value['vtype'] for key, value in fields.items()} == {'code': 'int64'}
        # The negative root solves Q(q)=4 but violates the declared selection q>=0.
        expected = {'tag': tag, 'value': {'code': 15105}}
    assert all(payload == expected for payload in payloads), payloads


@pytest.mark.parametrize('mode', ('bad_inverse', 'negative_branch'))
def test_inventory_conversion_typed_failure_restores_full_continuation(mode, tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('C12_'+mode, builder=build_inventory_publication_v15, mode=mode)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    before = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        _check(spec, before, mode, 1)
    error, failures = recorded_attempt(world, lambda: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    after = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        assert failures and all(failures)
        assert all(row[0] == 'StepAttemptRejected' for row in failures), failures
        signature, payloads = typed_failure_values(error, world)
        _assert_cause(mode, signature, payloads)
        assert after == before and runtime.time() == DT and runtime.macro_step() == 1
        _check(spec, after, mode, 1)
    # Strict checkpoint after the failure must still restore the complete last
    # accepted history/controller, not a partially prepared failed candidate.
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/('rejected_'+mode))))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = collective_call(world, lambda: _image(resumed))
    with collective_check(world):
        assert restored == before
        _check(spec, restored, mode, 1)


def _check_current(spec, image, step):
    root = _decode(spec, image)['root']
    q = 2*step-1
    history = 0 if step == 1 else 2*step-3
    assert root == {'q': float(q), 'z': float(2*q+history), 'history': float(history),
                    'time': DT*step, 'step': step}
    assert root['history'] != root['q']
    assert root['z'] != 2*root['history']+root['history']
    return root


def test_inventory_current_equation_preserves_declared_old_history_and_restart(tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    from examples.api_v02.current_inventory_v15 import build_current_inventory_v15
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('C12_current', builder=build_current_inventory_v15)
    original = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(original, t_end=DT, max_steps=1, console=False))
    first = collective_call(world, lambda: _image(original))
    with collective_check(world):
        _check_current(spec, first, 1)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: original.checkpoint(str(directory/'current_dependency')))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = collective_call(world, lambda: _image(resumed))
    with collective_check(world):
        assert restored == first
        _check_current(spec, restored, 1)
    for runtime in (original, resumed):
        collective_call(world, lambda runtime=runtime: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    images = [collective_call(world, lambda runtime=runtime: _image(runtime)) for runtime in (original, resumed)]
    with collective_check(world):
        roots = [_check_current(spec, image, 2) for image in images]
        assert roots[0] == roots[1]



def _check_bundle_inventory(spec,image,step):
    root=_decode(spec,image)['root'];bundle=root['bundle']
    assert root['time']==DT*step and root['step']==step
    inventory=bundle['maps']['inventory'];current=bundle['instance'];source=bundle['maps']['source']
    assert current is not source and current is not inventory
    assert all(d['size']==3 and d['capacity']==3 for d in (inventory,current,source))
    assert bundle['fields']['q']['domain'] is inventory
    assert bundle['histories']['q']['domain'] is inventory
    assert bundle['geometry']['center']['domain'] is current
    assert bundle['maps']['parent']['domain'] is current
    # Independent integer formula for q0=(1,2,3), delta_n=n*(1,2,3).
    factor=1+step*(step+1)//2;previous=1+(step-1)*step//2
    expected=[float(factor*i) for i in (1,2,3)]
    history=[float(previous*i) for i in (1,2,3)]
    assert bundle['fields']['q']['values']==expected
    assert bundle['histories']['q']['values']==history and expected!=history
    assert bundle['geometry']['center']['values']==[float((i+step)%3) for i in range(3)]
    for i,index in enumerate(bundle['maps']['parent']['values']):
        assert index['domain'] is source and index['offset']==(i+1)%3
    return (expected,history,bundle['geometry']['center']['values'])


def test_inventory_bundle_exact_path_remap_and_restart(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    from examples.api_v02.bundle_inventory_v15 import build_bundle_inventory_v15
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('C12_bundle',builder=build_bundle_inventory_v15)
    assert not spec.case._time._block_indices() and not spec.initial_values
    original=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(original,t_end=DT,max_steps=1,console=False))
    first=collective_call(world,lambda:_image(original))
    with collective_check(world):_check_bundle_inventory(spec,first,1)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:original.checkpoint(str(directory/'bundle_inventory')))
    resumed=_bind(spec,artifact,world)
    collective_call(world,lambda:resumed.restart(checkpoint))
    restored=collective_call(world,lambda:_image(resumed))
    with collective_check(world):
        assert restored==first
        _check_bundle_inventory(spec,restored,1)
    for runtime in (original,resumed):
        collective_call(world,lambda runtime=runtime:pops.run(runtime,t_end=2*DT,max_steps=1,console=False))
    images=[collective_call(world,lambda runtime=runtime:_image(runtime)) for runtime in (original,resumed)]
    with collective_check(world):
        # Restart legitimately reissues domain capabilities; compare every numeric
        # value plus each image's exact internal aliases, not capability bytes.
        assert _check_bundle_inventory(spec,images[0],2)==_check_bundle_inventory(spec,images[1],2)
