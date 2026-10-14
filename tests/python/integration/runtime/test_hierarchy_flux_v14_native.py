"""Native N07 receiver: finite hierarchy, two flux barriers, ALE and rollback.

The NumPy oracle evolves the three active cells directly; it has no PoPS graph,
ledger replacement or hierarchy helper in common with the implementation.
"""
from __future__ import annotations
import numpy as np
import pops
import pytest
from examples.api_v02.hierarchy_flux_v14 import build_hierarchy_flux_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile, _bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _reference(steps, constant):
    volume=np.array([.5,.25,.25]); q=np.full(3,2.) if constant else np.arange(1.,4.)
    amount=volume*q; w=np.array([0.,.02,-.01]);h=.05
    first=retained=np.zeros(3)
    for _ in range(steps):
        f0=(.4-w)*np.roll(amount/volume,1)
        moved=volume+h*(np.roll(w,-1)-w)
        predictor=amount+h*(f0-np.roll(f0,-1))
        f1=(.4-w)*np.roll(predictor/moved,1)
        first=h*f0;retained=h*(f0+f1)/2
        amount=amount+retained-np.roll(retained,-1);volume=moved
    return amount,volume,first,retained


def _check(spec,image,steps,constant):
    root=_decode(spec,image)['root'];b=root['bundle'];d={'active':b['instance'],**b['maps']['domains']}
    fields=b['fields'];geometry=b['geometry']
    expected,V,first,J=_reference(steps,constant)
    assert root['step']==steps and root['barriers']==2*steps
    assert abs(root['time']-.05*steps)<1.e-14
    assert abs(root['begin']-.05*(steps-1))<1.e-14 and abs(root['end']-.05*steps)<1.e-14
    assert [d[x]['size'] for x in ('active','coarse','fine','faces')]==[3,2,2,3]
    for key in ('active','coarse','fine'):
        assert fields[key]['domain'] is d[key] and geometry[key]['domain'] is d[key]
    np.testing.assert_allclose(fields['active']['values'],expected,rtol=0,atol=2.e-14)
    np.testing.assert_allclose(fields['fine']['values'],expected[1:],rtol=0,atol=2.e-14)
    np.testing.assert_allclose(fields['coarse']['values'],[expected[0],expected[1]+expected[2]],rtol=0,atol=2.e-14)
    np.testing.assert_allclose(geometry['active']['values'],V,rtol=0,atol=2.e-14)
    np.testing.assert_allclose(geometry['fine']['values'],V[1:],rtol=0,atol=2.e-14)
    np.testing.assert_allclose(geometry['coarse']['values'],[V[0],V[1]+V[2]],rtol=0,atol=2.e-14)
    np.testing.assert_allclose(geometry['position']['values'],[0,V[0],V[0]+V[1]],rtol=0,atol=2.e-14)
    np.testing.assert_allclose(b['histories']['first_integral']['values'],first,rtol=0,atol=2.e-14)
    np.testing.assert_allclose(b['histories']['retained_integral']['values'],J,rtol=0,atol=2.e-14)
    for index in b['maps']['parent']['values']:
        assert index['domain'] is d['coarse'] and index['offset']==1
    for ordinal,index in enumerate(b['maps']['fine_active']['values']):
        assert index['domain'] is d['active'] and index['offset']==ordinal+1
    np.testing.assert_allclose(np.sum(expected),2 if constant else 1.75,rtol=0,atol=2.e-14)
    if constant:np.testing.assert_allclose(expected/V,2,rtol=0,atol=2.e-14)
    else:assert np.max(np.abs(J-first))>1.e-6


@pytest.mark.parametrize('mode',('success','constant'))
def test_two_barrier_hierarchy_full_solution_restart_and_gcl(mode,tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('N07_'+mode,builder=build_hierarchy_flux_v14,mode=mode)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.05,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(runtime))
    with collective_check(world):_check(spec,image,1,mode=='constant')
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/'hierarchy')))
    restored=_bind(spec,artifact,world);collective_call(world,lambda:restored.restart(checkpoint))
    restored_image=collective_call(world,lambda:_image(restored))
    with collective_check(world):assert restored_image==image
    for target in (runtime,restored):
        collective_call(world,lambda:pops.run(target,t_end=.1,max_steps=1,console=False))
        image=collective_call(world,lambda:_image(target))
        with collective_check(world):_check(spec,image,2,mode=='constant')


@pytest.mark.parametrize('mode',('stale_stage','last_flux','double_h','old_volume'))
def test_incorrect_flux_or_geometry_is_refused_before_publication(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('N07_bad_'+mode,builder=build_hierarchy_flux_v14,mode=mode)
    runtime=_bind(spec,artifact,world)
    _unused,error=collective_attempt(world,lambda:pops.run(runtime,t_end=.05,max_steps=1,console=False))
    with collective_check(world):assert error is not None and runtime.macro_step()==0 and runtime.time()==0


def test_parent_refusal_restores_all_hierarchy_domains_geometry_and_ledger(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('N07_parent',builder=build_hierarchy_flux_v14,mode='parent_reject')
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.05,max_steps=1,console=False))
    before=collective_call(world,lambda:_image(runtime))
    _unused,error=collective_attempt(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    after=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        assert error is not None and runtime.macro_step()==1 and runtime.time()==.05
        assert after==before
        _check(spec,after,1,False)
