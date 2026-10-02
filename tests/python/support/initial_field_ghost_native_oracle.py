"""Offline independent uniform Helmholtz/FE/Ghost reference, no PoPS import."""
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

CONTRACT='accepted-initial-field-ghost-public@1'
DT=1/64
FIELD_BOUND=1e-10  # fixed pre-execution uniform Helmholtz absolute error guard

def check(blob,fields,steps):
    image=decode(np.frombuffer(blob,dtype=np.uint8))
    assert image['dim']==2 and image['real']==64 and image['levels']==2
    expected=2*(1+DT)**steps
    physical=[]
    for patch in image['patches']:
        assert patch['components']==2
        (xlo,xhi,gxlo,gxhi),(ylo,yhi,gylo,gyhi)=patch['axes']
        values=np.asarray(patch['bits'],dtype=np.uint64).view(np.float64).reshape(2,gyhi-gylo+1,gxhi-gxlo+1)
        valid=values[1,ylo-gylo:yhi-gylo+1,xlo-gxlo:xhi-gxlo+1]
        # FE multiplication/addition and independent factor construction: 4ops per step.
        eps=np.finfo(np.float64).eps;k=4*steps+2;bound=k*eps/(1-k*eps)*max(1,expected)
        assert np.max(np.abs(valid-expected))<=bound
        if xlo==0:
            ghost=values[1,ylo-gylo:yhi-gylo+1,-1-gxlo]
            target=expected+1+steps*DT
            error=float(np.max(np.abs(ghost-target)))
            assert error<=FIELD_BOUND+bound
            # Explicitly reject unchanged state / stale-initial Field / omitted time.
            assert np.min(np.abs(ghost-expected))>.5
            if steps:
                assert np.min(np.abs(ghost-(3+steps*DT)))>DT
                assert np.min(np.abs(ghost-(expected+1)))>DT/2
            physical.append({'key':patch['key'],'count':ghost.size,'max_error':error})
    assert physical
    assert len(fields)==2
    for field,mask in fields:
        a=np.asarray(field);mask=np.asarray(mask,dtype=bool)
        assert a.shape==mask.shape and mask.any()
        assert np.isfinite(a[mask]).all() and np.max(np.abs(a[mask]-expected))<=FIELD_BOUND
    return {'contract':CONTRACT,'steps':steps,'expected_m_phi':expected,
            'expected_xmin_m_ghost':expected+1+steps*DT,'field_absolute_guard':FIELD_BOUND,
            'physical_faces':physical}
