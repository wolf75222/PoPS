"""The finite-algorithm derivative is distinct from the implicit solution derivative."""
import pops
import pytest
from examples.api_v02.finite_iteration_ad_v14 import build_finite_iteration_ad_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check

from tests.python.support.api_v02_native import continuation_snapshot, recorded_attempt, assert_typed_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

@pytest.mark.parametrize('mode,value,tangent',(('iteration',.04,.2),('budget',1.2,3.),
    ('typed_error',-1.2,3.),('second',.48,2.4)))
def test_finite_algorithm_derivative_native(mode,value,tangent,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('FiniteAD_'+mode,builder=build_finite_iteration_ad_v14,mode=mode)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        assert not spec.case._time._block_indices()
        data=_decode(spec,image)['root']
        assert abs(data['value']-value)<1e-13 and abs(data['tangent']-tangent)<1e-13
        assert data['step']==1 and abs(data['time']-.01)<1e-14


def test_switching_surface_cannot_be_hidden_by_primal_budget_fallback(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('FiniteAD_surface',builder=build_finite_iteration_ad_v14,mode='surface')
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:continuation_snapshot(runtime,world))
    error,failures=recorded_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    after=collective_call(world,lambda:continuation_snapshot(runtime,world))
    with collective_check(world):
        assert before.status=="uninitialized"
        assert failures and all(failures)
        assert_typed_failure(error,world,tag="differential",code=91001)
        assert runtime.time()==0 and runtime.macro_step()==0 and before==after
