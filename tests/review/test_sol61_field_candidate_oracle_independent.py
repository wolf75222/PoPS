"""Independent synthetic reader adversaries, never Native evidence."""
import numpy as np
import pytest
from tests.review.test_sol61_field_candidate_math_migration import candidate
from tests.review.test_sol61_initial_field_ghost_native_preparation import wire,fields
from tests.review.sol61_amr_full_carrier_offline import decode
from tests.python.support.initial_field_ghost_native_oracle import check_observed

def encode(image):
    out=bytearray(b"POPSCAR1")
    def word(value,signed=False):out.extend(int(value).to_bytes(8,"little",signed=signed))
    for value in (image['dim'],image['real'],image['ranks']):word(value)
    word(image['shard'],True)
    word(image['levels']);word(len(image['blocks']))
    for name in image['blocks']:
        value=name.encode();word(len(value));out.extend(value)
    word(len(image['patches']))
    for patch in image['patches']:
        for value in (*patch['key'],patch['components']):word(value)
        word(patch['owner'],True)
        for axis in patch['axes']:
            for value in axis:word(value,True)
        word(len(patch['bits']))
        for value in patch['bits']:word(value)
    return bytes(out)

@pytest.mark.parametrize('mutation',('tick','stage','fraction'))
def test_accepted_point_must_match_genuine_scoped_fe_endpoint(mutation):
    rows=candidate(1)
    for row in rows[0]:
        # Real one-macro preparation precedes owner cursor increment: tick0, phase1.
        row['point'].update(tick=0,fraction_numerator=1,fraction_denominator=1)
        if mutation=='tick':row['point']['tick']=999
        if mutation=='stage':row['point']['stage']=17
        if mutation=='fraction':row['point']['fraction_numerator']=0
    with pytest.raises((AssertionError,ValueError)):
        check_observed(wire(steps=1),[m for a,m in fields(1)],rows,1)

def test_consumed_xmin_field_strip_must_match_real_ghost_expression():
    rows=candidate(1)
    for row in rows[0]:
        image=decode(np.frombuffer(row['carrier_bytes'],dtype=np.uint8))
        for patch in image['patches']:
            (xl,xh,gxl,gxh),(yl,yh,gyl,gyh)=patch['axes']
            if xl==0:
                bits=list(patch['bits'])
                for y in range(yl,yh+1):
                    bits[(y-gyl)*(gxh-gxl+1)+(-1-gxl)]=int(np.asarray(12345.,dtype=np.float64).view(np.uint64))
                patch['bits']=tuple(bits)
        row['carrier_bytes']=encode(image)
    with pytest.raises((AssertionError,ValueError)):
        check_observed(wire(steps=1),[m for a,m in fields(1)],rows,1)
