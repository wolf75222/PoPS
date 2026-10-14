"""The full original overdetermined equations control accepted implicit derivatives."""
import pops
import pytest
from examples.api_v02.implicit_domain_v14 import build_implicit_domain_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check

from tests.python.support.api_v02_native import continuation_snapshot, recorded_attempt, assert_typed_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

@pytest.mark.parametrize('mode',('compatible','incompatible','transpose'))
def test_exact_data_domain_of_overdetermined_implicit_relation(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('ImplicitDomain_v14_'+mode,builder=build_implicit_domain_v14,mode=mode)
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:continuation_snapshot(runtime,world))
    error,failures=recorded_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    after=collective_call(world,lambda:_image(runtime) if mode=="compatible" else continuation_snapshot(runtime,world))
    with collective_check(world):
        assert before.status=="uninitialized"
        if mode=='compatible':
            assert error is None and not any(failures)
            root=_decode(spec,after)['root']
            assert root['value']==1 and root['answer']==1 and root['step']==1
            assert runtime.time()==.01
        else:
            # The Jacobian [1,1]^T is regular, but (1,0) is not a compatible RHS.
            # A column-transpose VJP must reject that ambient basis, not invent a gradient.
            assert failures and all(failures)
            assert_typed_failure(error,world,tag="relation",code=2107)
            assert before==after and runtime.time()==0 and runtime.macro_step()==0
