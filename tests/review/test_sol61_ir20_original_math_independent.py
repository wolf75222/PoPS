"""SOURCE_ONLY original-F/JVP obligations; no PoPS/runtime/solver is executed."""
from fractions import Fraction

import numpy as np
import pytest


def witness(n=5, width=3):
    x=(np.arange(n)+.5)/n
    # Positive nonuniform measure; signed/nonsymmetric kernel has no SPD assumption.
    measure=(np.arange(n)+1).astype(float)
    measure/=measure.sum()
    kernel=.25+x[:,None]*(1-x[None,:])-2*x[None,:]+.5*x[:,None]
    q=np.stack([(-1)**c*(.12+.03*c+.02*np.arange(n)) for c in range(width)])
    v=np.stack([.05*(c+1)-.01*np.arange(n) for c in range(width)])
    return kernel,measure,q,v


def integral(kernel,measure,values):
    return np.array([float(sum(Fraction(float(k))*Fraction(float(w))*Fraction(float(q))
                              for k,w,q in zip(row,measure,values,strict=True))) for row in kernel])


def original(q,kernel,measure,*,frozen_source=None):
    source=q if frozen_source is None else frozen_source
    result=[]
    for component in range(len(q)):
        # A distinct source component, candidate nonlinear prefactor and original reaction.
        coupled=integral(kernel,measure,source[(component+1)%len(q)])
        result.append(q[component]+q[component]**3+(1+.2*q[component])*coupled)
    return np.stack(result)


def derivative(q,v,kernel,measure):
    rows=[]
    for c in range(len(q)):
        coupled=integral(kernel,measure,q[(c+1)%len(q)])
        variation=integral(kernel,measure,v[(c+1)%len(q)])
        rows.append((1+3*q[c]**2)*v[c]+.2*v[c]*coupled+(1+.2*q[c])*variation)
    return np.stack(rows)


@pytest.mark.parametrize("n,width",((3,2),(5,3),(7,5)))
def test_full_central_F_contains_nonlocal_variation_and_never_freezes_seed(n,width):
    kernel,measure,q,v=witness(n,width)
    h=1e-6
    observed=(original(q+h*v,kernel,measure)-original(q-h*v,kernel,measure))/(2*h)
    expected=derivative(q,v,kernel,measure)
    np.testing.assert_allclose(observed,expected,rtol=0,atol=8e-11)
    frozen=(original(q+h*v,kernel,measure,frozen_source=q)-original(q-h*v,kernel,measure,frozen_source=q))/(2*h)
    assert np.max(np.abs(expected-frozen))>.001
    seed=q*.8
    assert np.max(np.abs(original(q,kernel,measure,frozen_source=seed)-original(q,kernel,measure)))>.001
    # Same-layout/current-sized foreign source or wrong tuple component has different F.
    assert np.max(np.abs(original(q,kernel,measure,frozen_source=np.roll(q,1,axis=0))-original(q,kernel,measure)))>.001


def test_same_address_two_evaluations_require_new_invocation_not_pointer_cache():
    kernel,measure,q,v=witness()
    trial=q.copy()
    pointer=id(trial)
    h=1e-5
    np.copyto(trial,q+h*v)
    first=original(trial,kernel,measure)
    np.copyto(trial,q-h*v)
    assert id(trial)==pointer
    second=original(trial,kernel,measure)
    assert np.max(np.abs(first-second))>1e-7
    expected=derivative(q,v,kernel,measure)
    np.testing.assert_allclose((first-second)/(2*h),expected,rtol=0,atol=1e-10)
    stale_jvp=(first-first)/(2*h)
    assert np.max(np.abs(stale_jvp-expected))>.01


def test_original_equations_preserve_component_and_spatial_permutations():
    kernel,measure,q,v=witness()
    spatial=np.array((4,0,3,1,2))
    inverse=np.argsort(spatial)
    observed=original(q[:,spatial],kernel[np.ix_(spatial,spatial)],measure[spatial])[:,inverse]
    np.testing.assert_allclose(observed,original(q,kernel,measure),rtol=0,atol=2e-16)
    # This witness has cyclic coupling; cyclic component reordering preserves its declared route.
    perm=np.array((2,0,1))
    inv=np.argsort(perm)
    observed=original(q[perm],kernel,measure)[inv]
    np.testing.assert_allclose(observed,original(q,kernel,measure),rtol=0,atol=2e-16)


def test_rank_local_omission_and_double_owner_are_not_valid_original_F():
    kernel,measure,q,_=witness()
    expected=original(q,kernel,measure)
    half=measure.copy()
    half[2:]=0
    omitted=original(q,kernel,half)
    doubled=original(q,kernel,measure*2)
    assert np.max(np.abs(omitted-expected))>.001
    assert np.max(np.abs(doubled-expected))>.001
