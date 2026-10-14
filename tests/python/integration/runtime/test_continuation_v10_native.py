"""Run only after integration/build. No result in this file is a native acceptance claim."""
import numpy as np
import pops
import pytest
from examples.api_v02.continuation_checks import build_continuation
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind,_require_owned_array
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


@pytest.mark.parametrize('mode',('success','retry'))
def test_complete_continuation_recurs_without_mutating_retry_base(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('Continuation_'+mode,builder=build_continuation,mode=mode,steps=3)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.3,max_steps=3,console=False))
    actual=collective_call(world,lambda:runtime.state_global('method'));_require_owned_array(world,actual)
    with collective_check(world):
        assert runtime.macro_step()==3 and abs(runtime.time()-.3)<1.e-14
        if actual is not None:np.testing.assert_allclose(actual,10.,atol=1.e-13,rtol=0)


@pytest.mark.parametrize('mode',('assessment_reject','stale_transform','wrong_candidate','parent_reject','duplicate_increment','loop_identity'))
def test_invalid_acceptance_never_publishes_any_outer_leaf(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('Continuation_'+mode,builder=build_continuation,mode=mode,steps=1)
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:runtime.state_global('method'));_require_owned_array(world,before)
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    after=collective_call(world,lambda:runtime.state_global('method'))
    with collective_check(world):
        assert failures and all(failures),failures
        assert runtime.time()==0. and runtime.macro_step()==0
        if before is not None:np.testing.assert_array_equal(before,after)


def test_full_array_variant_and_domain_aliases_survive_strict_restart(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('Continuation_restart',builder=build_continuation,steps=4)
    uninterrupted=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(uninterrupted,t_end=.2,max_steps=2,console=False))
    before=collective_call(world,lambda:uninterrupted.state_global('method'));_require_owned_array(world,before)
    with collective_check(world):
        if before is not None:np.testing.assert_array_equal(before,4.)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:uninterrupted.checkpoint(str(directory/'Continuation')))
    restored=_bind(spec,artifact,world);collective_call(world,lambda:restored.restart(checkpoint))
    for runtime in (uninterrupted,restored):
        collective_call(world,lambda:pops.run(runtime,t_end=.4,max_steps=2,console=False))
    expected=collective_call(world,lambda:uninterrupted.state_global('method'))
    actual=collective_call(world,lambda:restored.state_global('method'))
    with collective_check(world):
        assert restored.macro_step()==4 and abs(restored.time()-.4)<1.e-14
        if actual is not None:
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual,20.)
