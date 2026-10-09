"""Independent synthetic POPSCAR1 wire probes; no Native qualification."""
import struct
import numpy as np
import pytest
from tests.python.support.amr_gather_bitwise_oracle import decode,reconstruct_level


def wire(*,shard=-1,owner=1):
    u=lambda x:struct.pack('<Q',x)
    i=lambda x:struct.pack('<q',x)
    # Two components, x:[1,2], y:[2,3], grown x:[0,3], y:[1,4].
    vals=np.arange(32,dtype=np.float64).reshape(2,4,4)+0.125
    vals[0,1,1]=-0.0;vals[1,2,2]=-3.75
    row=u(0)+u(1)+u(7)+u(2)+i(owner)
    row+=b''.join(i(x) for x in (1,2,0,3,2,3,1,4))
    data=b'POPSCAR1'+u(2)+u(64)+u(2)+i(shard)+u(2)+u(1)+u(3)+b'abc'+u(1)
    return data+row+u(32)+vals.astype('<f8').tobytes(),vals


def test_independent_axis_component_and_hole_bits():
    raw,vals=wire();archive=decode(np.frombuffer(raw,dtype=np.uint8))
    values,coverage=reconstruct_level(archive,0,1,4)
    assert values.dtype==np.float64 and values.shape==(2,4,4)
    assert np.array_equal(values[:,2:4,1:3].view(np.uint64),vals[:,1:3,1:3].view(np.uint64))
    assert values[0,2,1].view(np.uint64)==1<<63
    assert values[1,3,2]==-3.75 and coverage.sum()==4
    assert np.all(values.view(np.uint64)[:,coverage==0]==0)
    assert archive['patches'][0]['bits']==tuple(vals.view(np.uint64).ravel())


@pytest.mark.parametrize('change',('wrong-owner','wrong-shard','trailing','truncate','dtype','rank'))
def test_forged_wire_authority_refused(change):
    raw,_=wire();data=np.frombuffer(raw,dtype=np.uint8)
    if change=='wrong-owner':raw,_=wire(owner=2);data=np.frombuffer(raw,dtype=np.uint8)
    elif change=='wrong-shard':raw,_=wire(shard=0);data=np.frombuffer(raw,dtype=np.uint8)
    elif change=='trailing':data=np.frombuffer(raw+b'x',dtype=np.uint8)
    elif change=='truncate':data=data[:-1]
    elif change=='dtype':data=data.astype(np.int8)
    elif change=='rank':data=data.reshape(1,-1)
    with pytest.raises(ValueError):decode(data)


def test_grown_only_changes_do_not_invent_valid_or_ghost_qualification():
    raw,vals=wire();a=decode(np.frombuffer(raw,dtype=np.uint8));before,coverage=reconstruct_level(a,0,1,4)
    mutated=bytearray(raw);mutated[-8:]=struct.pack('<d',-912.0)
    b=decode(np.frombuffer(bytes(mutated),dtype=np.uint8));after,other=reconstruct_level(b,0,1,4)
    assert a['patches'][0]['bits']!=b['patches'][0]['bits']
    assert before.tobytes()==after.tobytes() and np.array_equal(coverage,other)
