"""Independent axis-permutation composite-stencil counterexample, no Native."""
import numpy as np
from tests.review.sol61_initial_fac_operator_audit import residuals


def test_original_operator_commutes_with_axis_swap_and_ignores_covered_parent_poison():
    mask=np.ones((8,8),dtype=bool);mask[1:5,2:6]=False
    fine=np.repeat(np.repeat(~mask,2,0),2,1)
    y,x=np.indices((8,8));fy,fx=np.indices((16,16))
    phi=2+1e-6*(x+3*y);fphi=2+1e-6*(fx/2+3*fy/2)
    state=[np.full((8,8),2.),np.full((16,16),2.)]
    original=residuals([(phi,mask),(fphi,fine)],state)
    swapped=residuals([(phi.T,mask.T),(fphi.T,fine.T)],[a.T for a in state])
    for a,b in zip(original,swapped,strict=True):
        np.testing.assert_allclose(a,b.T,rtol=0,atol=1e-12,equal_nan=True)
    poison=phi.copy();poison[~mask]=12345.
    poisoned=residuals([(poison,mask),(fphi,fine)],state)
    for a,b in zip(original,poisoned,strict=True):
        np.testing.assert_array_equal(a,b)
    assert max(np.nanmax(np.abs(a)) for a in original)>1e-10
