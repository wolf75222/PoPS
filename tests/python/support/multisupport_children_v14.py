"""Shared N02 receivers; definitions do not qualify the native scenarios."""
import json
import numpy as np
import pops
import pytest
from examples.api_v02.multisupport_children_v14 import build_multisupport_children_v14
from tests.python.support.api_v02_native import _compile,_bind
from tests.python.support.api_v02_receipts import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory

from tests.python.support.api_v02_receipts import _shards
from tests.python.support.decision_assess_v14 import _error_value

def _check(spec,image,step):
    root=_decode(spec,image)['root']
    expected={'volume':np.full(2,2-.05*step),'surface':np.full(3,1+step/120),
              'circuit':np.full(1,1+.075*step)}
    previous={'volume':np.full(2,2-.05*(step-1)),'surface':np.full(3,1+(step-1)/120),
              'circuit':np.full(1,1+.075*(step-1))}
    for key in expected:
        np.testing.assert_allclose(root[key]['values'],expected[key],rtol=0,atol=2.e-14)
        np.testing.assert_allclose(root['histories'][key]['values'],previous[key],rtol=0,atol=2.e-14)
        assert root[key]['domain'] is root['histories'][key]['domain']
    assert len({id(root[key]['domain']) for key in expected})==3
    assert abs(sum(sum(root[key]['values']) for key in expected)-8)<2.e-14
    assert abs(root['volume']['values'][0]-2)>.04
    assert root['step']==step and abs(root['time']-.1*step)<1.e-14
    assert root['attempts']==2*step and root['rejections']==step and root['transforms']==step
    packet=root['ledger']['shared'];assert abs(packet['value']-.1)<1.e-15
    assert abs(packet['interval']['begin']-.1*(step-1))<1.e-14
    assert abs(packet['interval']['end']-.1*step)<1.e-14
    return root

def run_success(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('N02_multisupport',builder=build_multisupport_children_v14)
    assert not spec.case._time._block_indices() and not spec.initial_values
    first=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(first,t_end=.2,max_steps=2,console=False))
    image=collective_call(world,lambda:_image(first))
    with collective_check(world):_check(spec,image,2)
    directory=collective_directory(world,tmp_path)
    path=collective_call(world,lambda:first.checkpoint(str(directory/'multisupport')))
    restored=_bind(spec,artifact,world)
    collective_call(world,lambda:restored.restart(path))
    with collective_check(world):assert _image(restored)==image
    for runtime in (first,restored):
        collective_call(world,lambda:pops.run(runtime,t_end=.3,max_steps=1,console=False))
    images=[collective_call(world,lambda r=r:_image(r)) for r in (first,restored)]
    with collective_check(world):
        roots=[_check(spec,raw,3) for raw in images]
        for key in ('volume','surface','circuit'):
            np.testing.assert_array_equal(roots[0][key]['values'],roots[1][key]['values'])

def run_failure(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('N02_'+mode,builder=build_multisupport_children_v14,mode=mode,steps=2)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    before=collective_call(world,lambda:_image(runtime))
    with collective_check(world):_check(spec,before,1)
    error=None
    try:pops.run(runtime,t_end=.2,max_steps=1,console=False)
    except Exception as caught:error=caught
    after=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error,StepAttemptRejected),repr(error)
        signature=json.loads(error.typed_failure_schema)['type']
        shards=_shards(error.typed_failure_payload)
        assert len(shards)==(1 if world is None else int(world.size))
        for shard in shards:
            payload=_error_value(shard,signature)
            if mode=='parent_reject':
                assert payload['tag']=='parent_trial'
                assert payload['value']['step']==1
                assert abs(payload['value']['total']-8)<2.e-14
            else:
                # Both negative authority operations are captured native
                # StepAttemptRejected errors: exact runtime code/origin.
                assert payload=={'tag':'runtime','value':{'code':3,'origin':2}}
        assert runtime.time()==.1 and runtime.macro_step()==1
        assert after==before
        _check(spec,after,1)
