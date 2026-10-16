"""Independent scientific/rollback receiver for the complete finite native ledger."""
from fractions import Fraction
import numpy as np
import pops
import pytest
from examples.api_v02.finite_bdf2_v14 import build_finite_bdf2_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode,_image
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.api_v02_native import recorded_attempt, typed_failure_values

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _check(spec,image,step):
    root=_decode(spec,image)['root']; b=root['bundle']
    delta=Fraction(-2,5); H=Fraction(3); previous_H=H
    for n in range(step):
        r=Fraction(1,2) if n==0 else Fraction(1)
        alpha=(1+2*r)/(1+r); beta=r*r/(1+r)
        delta=(Fraction(-1,5)+beta*delta+Fraction(1,100))/alpha
        previous_H,H=H,H+delta
    assert step!=1 or delta==Fraction(-77,400)
    assert root['step']==step and abs(root['time']-.1*step)<1e-14
    assert b['instance']['size']==1
    for value in (*b['fields'].values(),*b['geometry'].values(),*b['histories'].values()):
        assert value['domain'] is b['instance']
    np.testing.assert_allclose(b['fields']['H']['values'],[float(H)],rtol=0,atol=3e-14)
    np.testing.assert_allclose(b['fields']['q']['values'],[(np.sqrt(1+4*float(H))-1)/2],rtol=0,atol=3e-14)
    np.testing.assert_allclose(b['histories']['delta']['values'],[float(delta)],rtol=0,atol=3e-14)
    np.testing.assert_array_equal(b['geometry']['volume']['values'],[1.])
    assert abs(root['mean_H']-float(previous_H))<3e-14
    assert root['transferred_F']==-4
    if step==1:
        assert root['mean_q']==1 and root['reevaluated_F']==-1 and root['mean_H']==3
        assert root['mean_H']!=root['mean_q']+root['mean_q']**2
    return root


def test_native_bdf2_complete_inventory_history_restart(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('finite_bdf2_v14',builder=build_finite_bdf2_v14,steps=2)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    first=collective_call(world,lambda:_image(runtime))
    with collective_check(world): _check(spec,first,1)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/'bdf2_finite')))
    resumed=_bind(spec,artifact,world)
    collective_call(world,lambda:resumed.restart(checkpoint))
    restored=collective_call(world,lambda:_image(resumed))
    with collective_check(world):
        assert restored==first
        _check(spec,restored,1)
    collective_call(world,lambda:pops.run(resumed,t_end=.2,max_steps=1,console=False))
    second=collective_call(world,lambda:_image(resumed))
    with collective_check(world): _check(spec,second,2)


def _assert_failure_payloads(mode, signature, payloads):
    """Check every typed cause, including both original assessment residuals."""
    assert payloads
    tag = {'wrong_domain': 'runtime', 'negative_history': 'inventory_domain',
           'parent_reject': 'assessment_rejected'}[mode]
    assert signature['vtype'] == 'variant'
    declared = dict(signature['members'])[tag]
    assert declared['vtype'] == 'product'
    members = dict(declared['members'])
    assert len(members) == len(declared['members'])
    if mode == 'parent_reject':
        assert set(members) == {'checks'} and members['checks']['vtype'] == 'product'
        assert [(name, child['vtype']) for name, child in members['checks']['members']] == [
            ('original_defect', 'scalar'), ('inventory', 'scalar')]
        for value in payloads:
            assert set(value) == {'tag', 'value'} and value['tag'] == tag
            assert set(value['value']) == {'checks'}
            checks = value['value']['checks']
            assert set(checks) == {'original_defect', 'inventory'}
            # Exact rational BDF2 gives H1=1123/400 and H2=157/60, hence
            # original_defect=1/100 and H-q-q^2=0. Allow only binary64
            # evaluation error at the existing independent state-oracle bound;
            # 3e-14 is stricter than the graph's 1e-12 acceptance threshold.
            assert abs(checks['original_defect'] - .01) <= 3e-14, value
            assert abs(checks['inventory']) <= 3e-14, value
    else:
        expected_fields = ({'code': 'int64', 'origin': 'int64'} if mode == 'wrong_domain'
                           else {'begin': 'scalar', 'end': 'scalar'})
        assert {name: child['vtype'] for name, child in members.items()} == expected_fields
        expected = {'tag': tag, 'value': ({'code': 91002, 'origin': 1} if mode == 'wrong_domain'
                                        else {'begin': .2, 'end': .1})}
        assert all(value == expected for value in payloads), payloads


@pytest.mark.parametrize('mode',('wrong_domain','negative_history','parent_reject'))
def test_native_bdf2_failure_preserves_complete_accepted_inventory(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('finite_bdf2_'+mode,builder=build_finite_bdf2_v14,mode=mode,steps=2)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    before=collective_call(world,lambda:_image(runtime))
    error,failure=recorded_attempt(world,lambda:pops.run(runtime,t_end=.2,max_steps=1,console=False))
    after=collective_call(world,lambda:_image(runtime))
    with collective_check(world):
        assert failure and all(failure)
        assert all(row[0] == 'StepAttemptRejected' for row in failure), failure
        signature,payloads=typed_failure_values(error,world)
        _assert_failure_payloads(mode,signature,payloads)
        assert before==after and runtime.macro_step()==1 and abs(runtime.time()-.1)<1e-14
        _check(spec,after,1)
