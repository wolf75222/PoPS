"""Actual Program receivers, unqualified until run against the matching native build."""
import json
import struct
import numpy as np
import pops
from examples.api_v02.decision_assess_v14 import build_decision_assess_v14
from tests.python.support.api_v02_native import _compile,_bind
from tests.python.support.api_v02_receipts import _decode,_image
from tests.python.support.api_v02_receipts import _shards, _error_value
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory



def _check(spec,image,step,*,nested=False):
    root=_decode(spec,image)['root']
    np.testing.assert_allclose(root['q']['values'],np.array([1.,2.])+.05*step,rtol=0,atol=2.e-14)
    np.testing.assert_allclose(root['previous']['values'],np.array([1.,2.])+.05*(step-1),rtol=0,atol=2.e-14)
    assert root['q']['domain'] is root['previous']['domain']
    assert root['attempt']==1 and root['corrections']==step*(2 if nested else 1)
    assert root['step']==step and abs(root['time']-.1*step)<1.e-14
    return root



def run_success(mode,tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('decision_'+mode,builder=build_decision_assess_v14,mode=mode)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.2,max_steps=2,console=False))
    image=collective_call(world,lambda:_image(runtime))
    with collective_check(world):_check(spec,image,2,nested=mode.startswith('nested_'))
    directory=collective_directory(world,tmp_path)
    path=collective_call(world,lambda:runtime.checkpoint(str(directory/'decisions')))
    restored=_bind(spec,artifact,world)
    collective_call(world,lambda:restored.restart(path))
    with collective_check(world):assert _image(restored)==image
    for system in (runtime,restored):
        collective_call(world,lambda:pops.run(system,t_end=.3,max_steps=1,console=False))
        after=collective_call(world,lambda:_image(system))
        with collective_check(world):_check(spec,after,3,nested=mode.startswith('nested_'))


def run_failure(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('decision_'+mode,builder=build_decision_assess_v14,mode=mode,steps=2)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    before=collective_call(world,lambda:_image(runtime))
    error=None
    try:pops.run(runtime,t_end=.2,max_steps=1,console=False)
    except Exception as caught:error=caught
    after=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error,StepAttemptRejected),repr(error)
        assert runtime.time()==.1 and runtime.macro_step()==1 and after==before
        _check(spec,after,1,nested=mode.startswith('nested_'))
        signature=json.loads(error.typed_failure_schema)['type']
        shards=_shards(error.typed_failure_payload)
        assert len(shards)==(1 if world is None else world.size)
        for shard in shards:
            payload=_error_value(shard,signature)
            if mode=='budget':
                assert payload=={'tag':'decision_budget','value':{'limit':0,'used':0}}
            elif mode=='nested_reject':
                assert payload['tag']=='trial_failure'
                child=payload['value'];assert child['tag']=='decision_reject'
                assert child['value']['controller_update']=={'attempt':1}
                reason=child['value']['reason'];assert reason['code']==41
                assert abs(reason['time']-.2)<1.e-14 and abs(reason['observed']-3.2)<2.e-14
            else:
                assert payload['tag']=='decision_stop'
                reason=payload['value'];assert reason['code']==91
                assert abs(reason['time']-.2)<1.e-14 and abs(reason['observed']-3.2)<2.e-14
