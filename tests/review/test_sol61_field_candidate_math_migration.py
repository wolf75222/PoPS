"""Synthetic offline witnesses only; never a Native receipt or authority claim."""
import copy
import numpy as np
import pytest
from tests.review.test_sol61_initial_field_ghost_native_preparation import wire,fields
from tests.review.sol61_amr_full_carrier_offline import decode
from tests.python.support.initial_field_ghost_native_oracle import check_observed,DT


def candidate(steps,delta=0.):
    image=decode(np.frombuffer(wire(steps=steps),dtype=np.uint8))
    b=bytearray(b'POPSCAR1')
    def word(v,signed=False): b.extend(int(v).to_bytes(8,'little',signed=signed))
    for v in (2,64,1):word(v)
    word(0,True)
    for v in (2,1,4):word(v)
    b.extend(b'slot');word(len(image['patches']))
    for p in image['patches']:
        for v in (*p['key'],1):word(v)
        word(p['owner'],True)
        for axis in p['axes']:
            for v in axis:word(v,True)
        size=np.prod([a[3]-a[2]+1 for a in p['axes']]);word(size)
        b.extend(np.full(int(size),2*(1+DT)**steps+delta,dtype='<f8').tobytes())
    rows=[]
    for level in (0,1):
        rows.append(dict(schema='pops.amr.field-candidate-observation@1',status='producer-completed-consumer-preparation-completed',accepted_publication=False,provider_slot='slot',consumer_block='marker',consumer_level=level,owner_macro_step=steps,owner_time=float(steps*DT),configuration_identity='config',provider_identity='provider',plan_identity='plan',output_owner_identity='output',output_block='marker',output_key='phi',topology_epoch=0,materialization_generation=0,point=dict(clock='primary',tick=0,level=level,substep=0,stage=0,fraction_numerator=steps,fraction_denominator=1,dt=float(DT if steps else 0),physical_time=float(steps*DT),graph_identity='',rate_identity='',application_identity=''),carrier_bytes=bytes(b)))
    return [rows]

@pytest.mark.parametrize('steps',(0,1))
def test_synthetic_consumed_image_not_cache_endpoint(steps):
    rows=candidate(steps)
    result=check_observed(wire(steps=steps),[m for a,m in fields(steps)],rows,steps)
    assert len(result['invocations'])==2

@pytest.mark.parametrize('mutation',('stale','publication','duplicate','wrong_level','wrong_owner','bool_tick','wrong_point'))
def test_synthetic_observation_mutations_fail_closed(mutation):
    rows=candidate(1)
    if mutation=='stale':rows=candidate(0)
    if mutation=='publication':rows[0][0]['accepted_publication']=True
    if mutation=='duplicate':rows[0][1]=copy.deepcopy(rows[0][0])
    if mutation=='wrong_level':rows[0][0]['point']['level']=1
    if mutation=='wrong_owner':rows[0][0]['owner_macro_step']=0
    if mutation=='bool_tick':rows[0][0]['point']['tick']=True
    if mutation=='wrong_point':rows[0][0]['point']['physical_time']=0.
    with pytest.raises((AssertionError,ValueError)):check_observed(wire(steps=1),[m for a,m in fields(1)],rows,1)


def test_independent_full_original_F_refuses_small_uniform_phi_error():
    rows=candidate(1,delta=9e-11)
    with pytest.raises(AssertionError):
        check_observed(wire(steps=1),[m for a,m in fields(1)],rows,1)


def test_missing_rank_and_truncated_candidate_refused():
    with pytest.raises(AssertionError):check_observed(wire(),[m for a,m in fields(0)],[],0)
    rows=candidate(0);rows[0][0]['carrier_bytes']=rows[0][0]['carrier_bytes'][:-1]
    with pytest.raises(ValueError):check_observed(wire(),[m for a,m in fields(0)],rows,0)


def test_external_primary_clock_and_empty_builder_context_are_required():
    rows=candidate(1)
    with pytest.raises(AssertionError):
        check_observed(wire(steps=1),[m for a,m in fields(1)],rows,1,expected_clock='foreign.primary')
    for key in ('graph_identity','rate_identity','application_identity'):
        rows=candidate(1)
        rows[0][0]['point'][key]='forged.context'
        with pytest.raises(AssertionError):check_observed(wire(steps=1),[m for a,m in fields(1)],rows,1,expected_clock='primary')
