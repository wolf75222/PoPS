"""Installed finite distributed AD, independent dense oracle, full native checkpoint."""
import numpy as np
import pops
import pytest
from examples.api_v02.distributed_derivative_v14 import build_distributed_derivative_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile, _bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import native_profile, profile_gather_bytes

from tests.python.support.api_v02_native import (continuation_snapshot, recorded_attempt,
    assert_legacy_failure, assert_requests_surface_failure)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _check(spec, images, mode, empty, step):
    packets=[_decode(spec,image,rank=rank)['root'] for rank,image in enumerate(images)]
    n=0 if empty else 3
    arrays={role:[] for role in ('source','value','tangent','pullback')}
    for rank,packet in enumerate(packets):
        D=packet['domain'];assert D['size']==n and packet['step']==step
        assert abs(packet['time']-.01*step)<1e-14
        expected_offsets=([0]*(len(images)+1) if empty else [0,3] if len(images)==1
            else [0,1,3] if len(images)==2 else [0,1,1,3])
        assert D['offsets']==expected_offsets
        for role in arrays:
            array=packet[role];assert array['domain'] is D
            assert len(array['values'])==expected_offsets[rank+1]-expected_offsets[rank]
            arrays[role].extend(array['values'])
    x=np.arange(1,n+1,dtype=float)
    matrix=(np.zeros((0,0)) if empty else
        np.array([[1.,0,0],[0,1,0],[1,0,0]]) if mode=='gather' else
        np.array([[1.,0,1],[0,1,0],[0,0,0]]) if mode=='scatter' else np.tril(np.ones((3,3))))
    transformed=matrix@x
    jacobian=2*transformed[:,None]*matrix
    direction=1/(np.arange(n)+1)
    output_weights=np.arange(n)+1
    input_weights=np.arange(n)+2
    covector=np.full(n,1/3)
    expected={'source':x,'value':transformed**2,'tangent':jacobian@direction,
        'pullback':(jacobian.T@(output_weights*covector))/input_weights}
    for role in arrays:
        np.testing.assert_allclose(arrays[role],expected[role],rtol=0,atol=3e-12)
    left=np.dot(output_weights*covector,arrays['tangent'])
    right=np.dot(input_weights*direction,arrays['pullback'])
    assert abs(left-right)<3e-12
    if not empty:
        assert np.linalg.norm(jacobian-jacobian.T)>0 or mode=='gather'
        # Removing duplicate requests/contributions is a genuinely different map.
        if mode=='gather':assert abs(expected['pullback'][0]-1/3)>.5
        if mode=='scatter':assert expected['pullback'][2]>0
    return packets


@pytest.mark.parametrize('mode,empty',(('gather',False),('scatter',False),('scan',False),('gather',True)))
def test_distributed_jvp_vjp_complete_native_continuation(mode,empty,tmp_path,
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='distributed AD v14')
    ranks=1 if world is None else int(world.size)
    spec,artifact,world=_compile('DistributedAD_v14_'+mode+str(empty),
        builder=build_distributed_derivative_v14,mode=mode,ranks=ranks,empty=empty)
    assert not spec.case._time._block_indices() and not spec.initial_values
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(runtime));images=profile_gather_bytes(world,image)
    with collective_check(world):_check(spec,images,mode,empty,1)
    directory=collective_directory(world,tmp_path)
    path=collective_call(world,lambda:runtime.checkpoint(str(directory/'distributed_ad')))
    resumed=_bind(spec,artifact,world)
    collective_call(world,lambda:resumed.restart(path))
    restored=collective_call(world,lambda:_image(resumed))
    with collective_check(world):assert restored==image
    collective_call(world,lambda:pops.run(resumed,t_end=.02,max_steps=1,console=False))
    after=collective_call(world,lambda:_image(resumed));all_after=profile_gather_bytes(world,after)
    with collective_check(world):
        _check(spec,all_after,mode,empty,2)
        record_property('mpi_ranks',ranks)
        record_property('weighted_duality_and_all_coordinates',True)
        record_property('rank_one_empty',ranks==3)
        record_property('zero_blocks_and_authoritative_restart',True)


@pytest.mark.parametrize('empty',(False,True))
def test_distributed_ad_rejects_foreign_tangent_instance_even_empty(empty,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='distributed AD identity v14')
    ranks=1 if world is None else int(world.size)
    spec,artifact,world=_compile('DistributedAD_v14_foreign'+str(empty),
        builder=build_distributed_derivative_v14,ranks=ranks,empty=empty,wrong_domain=True)
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:continuation_snapshot(runtime,world))
    error,failures=recorded_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    after=collective_call(world,lambda:continuation_snapshot(runtime,world))
    with collective_check(world):
        assert before.status=="uninitialized"
        assert failures and all(failures)
        assert_legacy_failure(error,world,origin=1,code=91002)
        assert runtime.macro_step()==0 and runtime.time()==0 and before==after


@pytest.mark.parametrize('wrapped',(False,True))
@pytest.mark.parametrize('surface',(False,True))
def test_request_topology_surface_and_open_branch_native(wrapped,surface,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from examples.api_v02.distributed_derivative_v14 import build_request_topology_derivative_v14
    _,world=native_profile(dimension=2,scope='Requests topology derivative v14')
    ranks=1 if world is None else int(world.size)
    spec,artifact,world=_compile('RequestsTopology_v14_'+str(wrapped),
        builder=build_request_topology_derivative_v14,ranks=ranks,wrapped=wrapped)
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:continuation_snapshot(runtime,world))
    with collective_check(world):assert before.status=="uninitialized"
    if surface:
        error,failures=recorded_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
        after=collective_call(world,lambda:continuation_snapshot(runtime,world))
        with collective_check(world):
            assert failures and all(failures)
            assert_requests_surface_failure(error,world)
            assert before==after and runtime.time()==0 and runtime.macro_step()==0
    else:
        collective_call(world,lambda:pops.run(runtime,t_end=.015,max_steps=1,console=False))
        image=collective_call(world,lambda:_image(runtime))
        with collective_check(world):
            data=_decode(spec,image,rank=0 if world is None else int(world.rank))['root']
            x=100*(.015-.01)
            # Off the switching surface every request reads source[0]=1.
            expected_value=3*x if wrapped else x*x+3
            expected_tangent=3 if wrapped else 2*x
            assert abs(data['input']-x)<1e-14
            assert abs(data['value']-expected_value)<1e-12
            assert abs(data['tangent']-expected_tangent)<1e-12
            assert data['step']==1 and abs(data['time']-.015)<1e-14
