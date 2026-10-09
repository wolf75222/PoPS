"""acceptedHaloTemporalDiscriminator@1: independent Euler operation-count oracle."""
import numpy as np

DT=1/64
CONTRACT="acceptedHaloTemporalDiscriminator@1"

def factor(level):
    return 1+DT if level==0 else (1+2*DT/5)**2*(1+DT/5)

def bound(initial,steps,level):
    # Per child substep: dt ratio division/multiplication (2), RHS product
    # and Euler addition (2). Three substeps/macro =>12 operations. Independent
    # closed factor has divisions/additions/powers/products <=12 per macro.
    # Comparison subtraction adds1. Coarse is included in this upper bound.
    k=24*steps+1
    u=2.**-53
    gamma=k*u/(1-k*u)
    return gamma*np.abs(initial)*factor(level)**steps

def check(initial,current,mask,steps,level):
    a=np.asarray(initial).reshape(2,*mask.shape)
    b=np.asarray(current).reshape(2,*mask.shape)
    np.testing.assert_array_equal(b[0][mask],a[0][mask])
    expected=a[1][mask]*factor(level)**steps
    assert np.all(np.abs(b[1][mask]-expected)<=bound(a[1][mask],steps,level))
    return float(np.max(np.abs(b[1][mask]-expected)))

def discriminate(initial,mask):
    values=np.asarray(initial).reshape(2,*mask.shape)[1][mask]
    # Each wrong schedule has a closed-form nonzero gap at at least one
    # actually saved active cell, above the operation-count rounding envelope.
    wrong={'one_coarse_step':1+DT,'grid_ratio_two':(1+DT/2)**2,
           'missing_remainder':(1+2*DT/5)**2}
    gaps={key:float(np.max(np.abs(values)*(abs(factor(1)-value)))) for key,value in wrong.items()}
    tolerance=float(np.max(bound(values,1,1)))
    assert all(gap>2*tolerance for gap in gaps.values())
    return {'wrong_schedule_gaps':gaps,'roundoff_envelope':tolerance}
