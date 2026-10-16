"""Independent Newton oracle and true Bundle checkpoint/restart for a new model."""
import json
import numpy as np
import pops
import pytest
from examples.api_v02.extension_closure_v14 import build_extension_closure_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def oracle(steps,*,h=.1,kappa=.4):
    """Dense NumPy Newton, independent of the generated contractive iteration."""
    q=np.array([.1,.2,.3,.4],dtype=float)
    lap=-2*np.eye(4)+np.roll(np.eye(4),1,axis=1)+np.roll(np.eye(4),-1,axis=1)
    result=[q.copy()]
    for _ in range(steps):
        rhs=q+q**3+.5*h*kappa*(lap@q);z=q.copy()
        for iteration in range(20):
            residual=z+z**3-.5*h*kappa*(lap@z)-rhs
            if np.max(np.abs(residual))<5e-16:break
            z-=np.linalg.solve(np.diag(1+3*z*z)-.5*h*kappa*lap,residual)
        assert iteration<19 and np.max(np.abs(z+z**3-.5*h*kappa*(lap@z)-rhs))<1e-15
        q=z;result.append(q.copy())
    return result,lap


def check(spec,image,step,permutation,names):
    data=_decode(spec,image);root=data['root'];bundle=root['bundle'];domain=bundle['instance']
    expected,lap=oracle(step);physical_order=np.argsort(permutation)
    assert root['step']==step and abs(root['time']-.1*step)<2e-14 and domain['size']==4
    for array in (*bundle['fields'].values(),*bundle['geometry'].values(),
                  *bundle['histories'].values(),*bundle['maps'].values()):
        assert array['domain'] is domain
    raw=np.asarray(bundle['fields'][names[0]]['values']);raw_H=np.asarray(bundle['fields'][names[1]]['values'])
    q=raw[physical_order];H=raw_H[physical_order]
    old=np.asarray(bundle['histories']['previous']['values'])[physical_order]
    np.testing.assert_allclose(q,expected[-1],rtol=0,atol=2e-13)
    np.testing.assert_allclose(old,expected[-2],rtol=0,atol=2e-13)
    np.testing.assert_allclose(H,q+q**3,rtol=0,atol=2e-16)
    initial_H=expected[0]+expected[0]**3
    assert abs(np.mean(H)-np.mean(initial_H))<3e-13
    residual=q+q**3-old-old**3-.02*(lap@q+lap@old)
    assert float(residual@residual)<1.1e-26
    assert root['residual_squared']<1e-26 and root['balance']**2<1e-26
    assert 1<=root['iterations']<=80
    assert np.max(np.abs(q-expected[0]))>1e-3 # an unchanged carrier cannot pass
    np.testing.assert_array_equal(bundle['geometry']['volume']['values'],np.full(4,.25))
    np.testing.assert_array_equal(bundle['geometry']['coordinate']['values'],(np.array(permutation)+.5)/4)
    for side,delta in [('left',-1),('right',1)]:
        for slot,index in enumerate(bundle['maps'][side]['values']):
            assert index['domain'] is domain
            assert permutation[index['offset']]==(permutation[slot]+delta)%4
    return {'step':step,'max_q_error':float(np.max(np.abs(q-expected[-1]))),
            'inventory_error':float(abs(np.mean(H)-np.mean(initial_H))),
            'residual_squared':float(residual@residual),'iterations':root['iterations']}


@pytest.mark.parametrize('permutation,names',[
    ((0,1,2,3),('q','H')),
    ((2,0,3,1),('temperature','inventory')),
])
def test_new_python_equation_method_bundle_restart(permutation,names,tmp_path,
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('extension_closure_'+names[0],builder=build_extension_closure_v14,
        permutation=permutation,names=names,steps=3)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    first=collective_call(world,lambda:_image(runtime))
    with collective_check(world):first_metrics=check(spec,first,1,permutation,names)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/('extension_'+names[0]))))
    resumed=_bind(spec,artifact,world)
    collective_call(world,lambda:resumed.restart(checkpoint))
    restored=collective_call(world,lambda:_image(resumed))
    with collective_check(world):assert restored==first
    collective_call(world,lambda:pops.run(runtime,t_end=.3,max_steps=2,console=False))
    collective_call(world,lambda:pops.run(resumed,t_end=.3,max_steps=2,console=False))
    uninterrupted=collective_call(world,lambda:_image(runtime))
    final=collective_call(world,lambda:_image(resumed))
    with collective_check(world):
        assert final==uninterrupted
        final_metrics=check(spec,final,3,permutation,names)
        record_property('artifact_identity',artifact.artifact_identity.hexdigest)
        record_property('fields',json.dumps(names))
        record_property('permutation',json.dumps(permutation))
        record_property('metrics',json.dumps([first_metrics,final_metrics],sort_keys=True))
