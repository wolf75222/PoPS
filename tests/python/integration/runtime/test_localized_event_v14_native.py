"""Installed PoPS event reception; no Python calculation advances its trajectory."""
import numpy as np
import pops
import pytest
from examples.api_v02.localized_event_v14 import build_localized_event_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _check(spec,image,step):
    state=_decode(spec,image)['root']
    expected=(
        dict(q=.6,z=.36,integral=.18,last_integral=.18,previous_q=0.,previous_z=0.,
             before_reset=.6,duration=.6,branch=0,time=.6,step=1),
        dict(q=0.,z=0.,integral=.5,last_integral=.32,previous_q=.6,previous_z=.36,
             before_reset=1.,duration=.4,branch=1,time=1.,step=2),
        dict(q=.6,z=.36,integral=.68,last_integral=.18,previous_q=0.,previous_z=0.,
             before_reset=.6,duration=.6,branch=1,time=1.6,step=3),
    )[step-1]
    assert set(state)==set(expected)
    for name,value in expected.items():
        if name in ('step','branch'):assert state[name]==value
        else:np.testing.assert_allclose(state[name],value,rtol=0,atol=3.e-12,err_msg=name)


def test_interior_event_reconstructs_full_state_then_restarts_and_continues(
        tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('localized_event',builder=build_localized_event_v14)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.6,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(runtime))
    with collective_check(world):_check(spec,image,1)
    _,failures=collective_attempt(
        world,lambda:pops.run(runtime,t_end=1.2,max_steps=1,console=False))
    with collective_check(world):
        reached=runtime.time()
        expected=("max_steps exhausted before t_end: accepted 1 step(s), "
                  f"reached t={reached!r}, requested t_end=1.2")
        assert failures and all(row==('RuntimeError',expected,True) for row in failures),failures
        assert runtime.macro_step()==2 and abs(reached-1.)<3.e-12 and reached<1.2
    image=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        _check(spec,image,2)
        assert abs(runtime.time()-1.)<3.e-12 and runtime.time()<1.2
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/'localized_event')))
    restored=_bind(spec,artifact,world)
    collective_call(world,lambda:restored.restart(checkpoint))
    copied=collective_call(world,lambda:_image(restored))
    with collective_check(world):assert copied==image
    for system in (runtime,restored):
        target=system.time()+.6
        collective_call(world,lambda:pops.run(system,t_end=target,max_steps=1,console=False))
        final=collective_call(world,lambda:_image(system))
        with collective_check(world):_check(spec,final,3)


def test_clamped_q_with_stale_constraint_and_flux_is_rejected_atomically(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('localized_event_stale',builder=build_localized_event_v14,mode='stale_dense')
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.6,max_steps=1,console=False))
    before=collective_call(world,lambda:_image(runtime))
    _,failure=collective_attempt(world,lambda:pops.run(runtime,t_end=1.2,max_steps=1,console=False))
    after=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        assert failure and all(failure)
        assert runtime.macro_step()==1 and abs(runtime.time()-.6)<3.e-12
        assert after==before
        _check(spec,after,1)
