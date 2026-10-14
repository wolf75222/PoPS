"""MPI2/MPI3 fixed-communicator repartition and structural restart receipts."""
import json
import numpy as np
import pops
import pytest
from examples.api_v02.bundle_repartition_v14 import build_bundle_repartition_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.runtime.test_api_v02_terminal_failure_native import _shards
from tests.python.support.decision_assess_v14 import _error_value
from tests.python.support.public_native_profile import native_profile

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _check(spec,image,step,rank,ranks):
    decoded=_decode(spec,image,rank=rank);assert decoded['wire']=='POPSCONT2'
    root=decoded['root'];bundle=root['bundle'];domain=bundle['instance'];source=bundle['maps']['source']
    split=4 if step%2 else 0
    offsets=[0,split,4] if ranks==2 else [0,split,split,4]
    oldsplit=4 if (step-1)%2 else 0
    previous_offsets=([0,1,4] if ranks==2 else [0,1,1,4]) if step==1 else (
        [0,oldsplit,4] if ranks==2 else [0,oldsplit,oldsplit,4])
    assert domain['offsets']==offsets and source['offsets']==previous_offsets
    assert domain is not source and domain['family']==source['family']
    assert root['step']==step and abs(root['time']-.1*step)<1.e-14
    q=np.arange(1.,5.);previous=q.copy();centers=(np.arange(4)+.5)/4;permutation=np.array([1,0,3,2])
    for _ in range(step):
        moved=q[permutation];old_history=previous[permutation]
        q=.8*moved+.1*old_history;previous=moved;centers=centers[permutation]
    local=slice(offsets[rank],offsets[rank+1])
    for a in (*bundle['fields'].values(),*bundle['histories'].values(),
              *bundle['geometry'].values(),bundle['maps']['parent']):
        assert a['domain'] is domain
    np.testing.assert_allclose(bundle['fields']['q']['values'],q[local],rtol=0,atol=2.e-14)
    np.testing.assert_allclose(bundle['histories']['previous']['values'],previous[local],rtol=0,atol=2.e-14)
    np.testing.assert_array_equal(bundle['geometry']['volume']['values'],np.full(4,.25)[local])
    np.testing.assert_array_equal(bundle['geometry']['center']['values'],centers[local])
    requests=bundle['maps']['parent'];assert requests['index_domain'] is source
    for gid,index in zip(range(offsets[rank],offsets[rank+1]),requests['values']):
        assert index['domain'] is source and index['offset']==int(permutation[gid])
    assert len(bundle['fields']['q']['values'])==offsets[rank+1]-offsets[rank]
    return decoded


def test_bundle_repartition_history_geometry_aliases_and_checkpoint(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='Bundle fixed-communicator repartition v14')
    assert world is not None and int(world.size) in (2,3)
    ranks,rank=int(world.size),int(world.rank)
    spec,artifact,world=_compile('Bundle_repartition',builder=build_bundle_repartition_v14,ranks=ranks)
    assert not spec.case._time._block_indices() and not spec.initial_values
    first=_bind(spec,artifact,world)
    for step in (1,2):
        collective_call(world,lambda step=step:pops.run(first,t_end=.1*step,max_steps=1,console=False))
        image=collective_call(world,lambda:_image(first))
        with collective_check(world):_check(spec,image,step,rank,ranks)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:first.checkpoint(str(directory/'distributed_bundle')))
    restored=_bind(spec,artifact,world)
    collective_call(world,lambda:restored.restart(checkpoint))
    restored_image=collective_call(world,lambda:_image(restored))
    with collective_check(world):
        assert restored_image==image
        _check(spec,restored_image,2,rank,ranks)
    for runtime in (first,restored):
        collective_call(world,lambda:pops.run(runtime,t_end=.3,max_steps=1,console=False))
        image=collective_call(world,lambda runtime=runtime:_image(runtime))
        with collective_check(world):_check(spec,image,3,rank,ranks)


@pytest.mark.parametrize('mode',('stale_history','parent_reject'))
def test_failed_repartition_keeps_every_shard_and_history(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='Bundle rejected repartition v14')
    assert world is not None and int(world.size) in (2,3)
    ranks,rank=int(world.size),int(world.rank)
    spec,artifact,world=_compile('Bundle_repartition_'+mode,builder=build_bundle_repartition_v14,
                                ranks=ranks,mode=mode,steps=2)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    before=collective_call(world,lambda:_image(runtime))
    with collective_check(world):_check(spec,before,1,rank,ranks)
    error=None
    try:pops.run(runtime,t_end=.2,max_steps=1,console=False)
    except Exception as caught:error=caught
    after=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error,StepAttemptRejected),repr(error)
        signature=json.loads(error.typed_failure_schema)['type']
        shards=_shards(error.typed_failure_payload)
        assert len(shards)==ranks
        expected=({'tag':'parent','value':{'code':14}} if mode=='parent_reject' else
                  {'tag':'runtime','value':{'code':3,'origin':2}})
        for shard in shards:
            assert _error_value(shard,signature)==expected
        assert runtime.time()==.1 and runtime.macro_step()==1
        assert after==before
        _check(spec,after,1,rank,ranks)
