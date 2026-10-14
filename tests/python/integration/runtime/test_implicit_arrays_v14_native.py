"""Zero-block PoPS Array implicit AD; the independent oracle reads the accepted image."""
import numpy as np
import pops
import pytest
from examples.api_v02.implicit_array_v14 import build_implicit_array_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile, _bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

from tests.python.support.api_v02_native import continuation_snapshot, recorded_attempt, assert_typed_failure

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _check(spec, image, count, transpose, step):
    packet = _decode(spec, image)['root']
    domain = packet['domain']
    assert domain['size'] == count and packet['step'] == step
    assert abs(packet['time']-.01*step)<1e-14
    for key in ('roots','data','answer'):
        assert packet[key]['domain'] is domain
    u = 1+(np.arange(count)+1)/4
    np.testing.assert_array_equal(packet['roots']['values'], u)
    np.testing.assert_array_equal(packet['data']['values'], u*u)
    matrix = np.diag(2*u)
    expected = ((np.linalg.solve(matrix.T, (np.arange(count)+2)/3)/(np.arange(count)+3)
                 if transpose else np.linalg.solve(matrix, 1/(np.arange(count)+1))) if count else np.empty(0))
    np.testing.assert_allclose(packet['answer']['values'], expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize('count,transpose,redundant', ((0,False,False),(3,False,False),(0,True,False),(3,True,False),(3,False,True),(3,True,True)))
def test_complete_array_implicit_native_image(count, transpose, redundant, tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('ArrayImplicit_v14_'+str((count,transpose,redundant)),
        builder=build_implicit_array_v14, count=count, transpose=transpose,redundant=redundant)
    assert not spec.case._time._block_indices() and not spec.initial_values
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    image = collective_call(world,lambda:_image(runtime))
    with collective_check(world): _check(spec,image,count,transpose,1)
    directory = collective_directory(world,tmp_path)
    checkpoint = collective_call(world,lambda:runtime.checkpoint(str(directory/'array_implicit')))
    resumed = _bind(spec,artifact,world)
    collective_call(world,lambda:resumed.restart(checkpoint))
    restored = collective_call(world,lambda:_image(resumed))
    with collective_check(world): assert restored == image
    collective_call(world,lambda:pops.run(resumed,t_end=.02,max_steps=1,console=False))
    after = collective_call(world,lambda:_image(resumed))
    with collective_check(world): _check(spec,after,count,transpose,2)


def test_dishonest_array_tangent_rejected_by_original_equations(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world = _compile('ArrayImplicit_v14_dishonest',builder=build_implicit_array_v14,
                                   count=3,dishonest=True)
    runtime = _bind(spec,artifact,world)
    before = collective_call(world,lambda:continuation_snapshot(runtime,world))
    error,failures = recorded_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    after = collective_call(world,lambda:continuation_snapshot(runtime,world))
    with collective_check(world):
        assert before.status=="uninitialized"
        assert failures and all(failures)
        assert_typed_failure(error,world,tag="relation",code=2107)
        assert runtime.time()==0 and runtime.macro_step()==0 and before==after


@pytest.mark.parametrize('transpose',(False,True))
def test_distributed_implicit_array_complete_native_image(transpose,tmp_path,
        isolated_native_cache,native_cxx,kokkos_root):
    from tests.python.support.public_native_profile import native_profile,profile_gather_bytes
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='distributed implicit Array v14')
    ranks=1 if world is None else int(world.size)
    spec,artifact,world=_compile('DistributedImplicit_v14_'+str(transpose),
        builder=build_implicit_array_v14,count=3,transpose=transpose,distributed_ranks=ranks)
    assert not spec.case._time._block_indices() and not spec.initial_values
    runtime=_bind(spec,artifact,world)
    def check(images,step):
        packets=[_decode(spec,image,rank=rank)['root'] for rank,image in enumerate(images)]
        arrays={name:[] for name in ('roots','data','answer')}
        for packet in packets:
            D=packet['domain'];assert D['size']==3 and packet['step']==step
            assert abs(packet['time']-.01*step)<1e-14
            for name in arrays:
                assert packet[name]['domain'] is D
                arrays[name].extend(packet[name]['values'])
        u=1+(np.arange(3)+1)/4;matrix=np.diag(2*u)
        expected=(np.linalg.solve(matrix.T,(np.arange(3)+2)/3)/(np.arange(3)+3)
                  if transpose else np.linalg.solve(matrix,1/(np.arange(3)+1)))
        np.testing.assert_array_equal(arrays['roots'],u)
        np.testing.assert_array_equal(arrays['data'],u*u)
        np.testing.assert_allclose(arrays['answer'],expected,rtol=0,atol=2e-12)
    collective_call(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(runtime));images=profile_gather_bytes(world,image)
    with collective_check(world):check(images,1)
    directory=collective_directory(world,tmp_path)
    path=collective_call(world,lambda:runtime.checkpoint(str(directory/'distributed_implicit')))
    resumed=_bind(spec,artifact,world);collective_call(world,lambda:resumed.restart(path))
    restored=collective_call(world,lambda:_image(resumed))
    with collective_check(world):assert restored==image
    collective_call(world,lambda:pops.run(resumed,t_end=.02,max_steps=1,console=False))
    image=collective_call(world,lambda:_image(resumed));images=profile_gather_bytes(world,image)
    with collective_check(world):check(images,2)
