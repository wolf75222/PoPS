"""Synthetic byte-codec adversaries only, not Native carrier evidence."""
import struct
import pytest
from tests.python.support.initial_ghost_failure_selection import select_xmin_owner,require_selection_agreement

def wire(*,dim=2,ranks=2,shard=-1,owners=(0,1),xmin=True):
    def u(v):return struct.pack('<Q',v)
    def i(v):return struct.pack('<q',v)
    blob=b'POPSCAR1'+u(dim)+u(64)+u(ranks)+i(shard)+u(1)+u(1)+u(1)+b'Q'+u(len(owners))
    for index,owner in enumerate(owners):
        blob+=u(0)+u(0)+u(index)+u(1)+i(owner)
        for axis in range(dim):
            lo=0 if xmin else 1
            blob+=i(lo)+i(lo)+i(lo)+i(lo)
        blob+=u(1)+u(0)
    return blob

def test_real_decoder_selection_and_exact_agreement():
    row=select_xmin_owner(wire(),2);assert row['target']==1 and row['eligible']==[0,1]
    assert require_selection_agreement([row,row],row)['agreed']==row
    zero=select_xmin_owner(wire(owners=(0,)),2);assert zero['target']==0
    forged={**zero,'target':False}
    with pytest.raises(ValueError):require_selection_agreement([zero,forged],zero)
    with pytest.raises(ValueError):require_selection_agreement([row],row)
    with pytest.raises(ValueError):require_selection_agreement([row,{**row,'carrier_sha256':'forged'}],row)

@pytest.mark.parametrize('blob,size',[(wire(dim=1),2),(wire(ranks=1,owners=(0,)),2),(wire(shard=0),2),(wire(owners=()),2),(wire(xmin=False),2),(wire(owners=(2,)),2),(b'',2),(wire(),True)])
def test_selection_refuses_non_global_or_empty_authority(blob,size):
    with pytest.raises((ValueError,TypeError)):select_xmin_owner(blob,size)
