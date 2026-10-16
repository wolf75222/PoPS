"""Real zero-block PoPS execution of a singular action and adversarial contracts."""
import pops
import pytest
from examples.api_v02.action_checks_v14 import build_action_checks_v14
from pops.actions import ActionFailure
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory

from tests.python.support.api_v02_native import continuation_snapshot, recorded_attempt, assert_typed_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


@pytest.mark.parametrize('coefficient',(1.,2.))
def test_singular_phi1_original_action_native(coefficient,tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('SingularAction_v14_'+str(coefficient),
        builder=build_action_checks_v14,coefficient=coefficient)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        assert not spec.case._time._block_indices() and not spec.initial_values
        result=_decode(spec,image)['root']
        # A^2=0, so the independently evaluated finite polynomial is exact.
        assert abs(result['value']['x']-(1+.3*coefficient))<1e-13
        assert result['value']['y']==3 and result['error']<1e-13 and result['step']==1
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/'action')))
    resumed=_bind(spec,artifact,world);collective_call(world,lambda:resumed.restart(checkpoint))
    restored=collective_call(world,lambda:_image(resumed))
    with collective_check(world):assert restored==image


@pytest.mark.parametrize('mode',('fabricated','budget','domain'))
def test_action_false_ok_budget_and_domain_are_rejected_natively(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('ActionReject_v14_'+mode,builder=build_action_checks_v14,mode=mode)
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:continuation_snapshot(runtime,world))
    error,failures=recorded_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    after=collective_call(world,lambda:continuation_snapshot(runtime,world))
    with collective_check(world):
        assert before.status=="uninitialized"
        assert failures and all(failures)
        expected=ActionFailure.EVIDENCE_OUTSIDE_DOMAIN if mode=='domain' else ActionFailure.ERROR_BOUND_EXCEEDED
        assert_typed_failure(error,world,tag="action",code=int(expected))
        assert before==after and runtime.time()==0 and runtime.macro_step()==0
