"""Real zero-block PoPS receiver. This file is not a claim that it has been run."""
from __future__ import annotations
import struct
import numpy as np
import pops
import pytest
from tests.python.support.api_v02_receipts import _decode, _image
from examples.api_v02.bundle_remap_v14 import build_bundle_remap_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile, _bind
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.api_v02_native import recorded_attempt, typed_failure_values

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]




def _check(spec, image, step):
    decoded = _decode(spec, image); root = decoded['root']; bundle = root['bundle']
    n = 2**(step+1); current = bundle['instance']; source = bundle['maps']['source']
    assert root['step'] == step and abs(root['time']-.1*step) < 1.e-14
    assert current['size'] == n and source['size'] == n//2 and current is not source
    for array in (*bundle['fields'].values(), *bundle['geometry'].values(), *bundle['histories'].values(),
                  bundle['maps']['parent']):
        assert array['domain'] is current
    q = np.asarray(bundle['fields']['q']['values'])
    previous = np.asarray(bundle['histories']['previous']['values'])
    coarse = np.repeat([1.,2.], n//2)
    np.testing.assert_allclose(q, coarse*.9**step, rtol=0, atol=2.e-14)
    np.testing.assert_allclose(previous, coarse*.9**(step-1), rtol=0, atol=2.e-14)
    np.testing.assert_array_equal(bundle['geometry']['volume']['values'], np.full(n,1/n))
    np.testing.assert_array_equal(bundle['geometry']['center']['values'], (np.arange(n)+.5)/n)
    for j, index in enumerate(bundle['maps']['parent']['values']):
        assert index['domain'] is source and index['offset'] == j//2
    # An unchanged state that conserves mass cannot satisfy the entrywise decay/history oracle.
    assert np.max(np.abs(q-coarse)) > .05
    return decoded


def test_bundle_remap_complete_native_state_and_restart(tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('Bundle_v14', builder=build_bundle_remap_v14, steps=3)
    assert not spec.case._time._block_indices() and not spec.initial_values
    first = _bind(spec, artifact, world)
    collective_call(world, lambda:pops.run(first,t_end=.2,max_steps=2,console=False))
    image = collective_call(world, lambda:_image(first))
    with collective_check(world): _check(spec,image,2)
    directory = collective_directory(world,tmp_path)
    checkpoint = collective_call(world,lambda:first.checkpoint(str(directory/'finite_bundle')))
    restored = _bind(spec,artifact,world)
    collective_call(world,lambda:restored.restart(checkpoint))
    restored_image = collective_call(world,lambda:_image(restored))
    with collective_check(world):
        _check(spec,restored_image,2)
        assert restored_image == image
    for runtime in (first,restored):
        collective_call(world,lambda:pops.run(runtime,t_end=.3,max_steps=1,console=False))
    images = [collective_call(world,lambda runtime=runtime:_image(runtime)) for runtime in (first,restored)]
    with collective_check(world):
        decoded = [_check(spec,image,3)['root']['bundle'] for image in images]
        # Fresh live issuance after restart may burn different ordinals. Scientific values
        # and internal alias relations must agree; old capability identities must not persist.
        for role in ('fields','geometry','histories'):
            for key in decoded[0][role]:
                np.testing.assert_array_equal(decoded[0][role][key]['values'], decoded[1][role][key]['values'])


def _assert_failure_payloads(mode, signature, payloads):
    """Every shard must carry the complete, exactly typed graph-selected cause."""
    assert payloads
    tag = 'parent' if mode == 'parent_reject' else 'runtime'
    fields = {'code': 'int64'} if tag == 'parent' else {'code': 'int64', 'origin': 'int64'}
    assert signature['vtype'] == 'variant'
    declared = dict(signature['members'])[tag]
    assert declared['vtype'] == 'product'
    assert [(name, child['vtype']) for name, child in declared['members']] == sorted(fields.items())
    expected = {'tag': tag, 'value': {'code': 77} if tag == 'parent' else {'code': 3, 'origin': 2}}
    assert all(value == expected for value in payloads), payloads


@pytest.mark.parametrize('mode',('stale_history','capacity','parent_reject'))
def test_failed_remap_or_parent_preserves_whole_native_bundle(mode, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec,artifact,world = _compile('Bundle_v14_'+mode,builder=build_bundle_remap_v14,mode=mode,steps=2)
    runtime = _bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    before = collective_call(world,lambda:_image(runtime))
    with collective_check(world): _check(spec,before,1)
    error, failures = recorded_attempt(world,lambda:pops.run(runtime,t_end=.2,max_steps=1,console=False))
    after = collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        assert failures and all(failures)
        assert all(row[0] == 'StepAttemptRejected' for row in failures), failures
        signature, payloads = typed_failure_values(error, world)
        _assert_failure_payloads(mode, signature, payloads)
        assert runtime.time() == .1 and runtime.macro_step() == 1
        assert after == before
        _check(spec,after,1)
