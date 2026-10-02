"""Source preparation and independent oracle attacks; no Native execution."""
import numpy as np
import pops
import pytest
from tests.python.support.atomic_cubature_path_case import make_case
from tests.python.support.atomic_cubature_fv_oracle import (DT,NX,NY,initial_averages,
                                                           directional_flux,forward_euler)


@pytest.mark.parametrize('nonconservative',[False,True])
def test_native_case_public_validate_resolve(nonconservative):
    case,layout=make_case(nonconservative=nonconservative,amr=True,fixed_dt=DT)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert len(resolved.blocks)==1
    assert initial_averages().shape==(6,NY,NX)


def test_true_cell_averages_not_points_and_conservative_periodic_mass():
    q=initial_averages()
    # Exact sinus integral attenuates each atom's fluctuation by sinc(pi/n).
    # This finite-cell distinction is substantial compared with numerical guards.
    from tests.python.support.atomic_cubature_fv_oracle import ATOMS,INDICES
    x=(np.arange(NX)+.5)/NX;y=(np.arange(NY)+.5)/NY
    points=np.stack([sum((.6+.05*k+.03*np.sin(2*np.pi*x+k/7)[None,:]
                          +.02*np.cos(2*np.pi*y-k/5)[:,None])*ATOMS[k,0]**p*ATOMS[k,1]**r
                         for k in range(6)) for p,r in INDICES])
    assert np.max(np.abs(points-q))>1e-3
    conservative=forward_euler(q,nonconservative=False)
    np.testing.assert_allclose(conservative.sum(axis=(1,2)),q.sum(axis=(1,2)),rtol=0,atol=1e-13)
    active=forward_euler(q,nonconservative=True)
    assert np.max(np.abs(active-conservative))>1e-6
    assert np.max(np.abs(active-q))>1e-6
    assert not np.array_equal(q.transpose(0,2,1),q)


def test_independent_flux_particles_and_bad_geometry_refused():
    q=initial_averages()
    cross,rho,yy,x,xx,y=q
    # Reconstruct independent node weights from the six elementary moment equations.
    weights=(rho-xx-yy+cross,(xx+x-2*cross)/2,(xx-x)/2,
             (yy+y-2*cross)/2,(yy-y)/2,cross)
    from tests.python.support.atomic_cubature_fv_oracle import ATOMS,INDICES
    assert min(np.min(w) for w in weights)>0
    for axis in (0,1):
        reference=np.stack([sum(weights[k]*ATOMS[k,axis]*ATOMS[k,0]**p*ATOMS[k,1]**r
                                 for k in range(6)) for p,r in INDICES])
        np.testing.assert_allclose(directional_flux(q,axis),reference,rtol=0,atol=5e-16)
    with pytest.raises(ValueError):forward_euler(q.transpose(0,2,1),nonconservative=True)


def test_synthetic_carrier_valid_bits_crosscheck_and_corruption_refusal():
    import struct
    from tests.python.support.atomic_cubature_fv_oracle import authenticate_carrier_values
    q=initial_averages()
    # Synthetic codec host fixture only, never labeled a native capture/receipt.
    word=lambda x:struct.pack('<q',x)
    name=b'population'
    header=b'POPSCAR1'+b''.join(word(x) for x in (2,64,1,-1,1,1))+word(len(name))+name+word(1)
    grown=np.pad(q,((0,0),(1,1),(1,1)),mode='wrap')
    row=b''.join(word(x) for x in (0,0,0,6,0,0,NX-1,-1,NX,0,NY-1,-1,NY,grown.size))
    blob=header+row+grown.astype('<f8').tobytes()
    authenticate_carrier_values(blob,q)
    poisoned=grown.copy();poisoned[1,1,1]+=1
    with pytest.raises(ValueError,match='valid bits'):
        authenticate_carrier_values(header+row+poisoned.astype('<f8').tobytes(),q)
    with pytest.raises(ValueError):authenticate_carrier_values(blob[:-1],q)
