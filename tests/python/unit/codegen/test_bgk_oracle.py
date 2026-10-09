"""Independent numerical reference checks; never an engine execution claim."""
import numpy as np
import pytest
from tests.python.support.m19_bgk_oracle import initial,equilibrium,moments,step

@pytest.mark.parametrize('nx,nv',((32,32),(64,64)))
def test_isolated_collision_has_positive_relaxation_and_finite_quadrature_defect(nx,nv):
    f=initial(nx,nv);m=equilibrium(f);accepted,stages=step(f)
    assert np.isfinite(accepted).all() and accepted.min()>0
    assert np.max(abs(accepted-f))>1e-4
    assert np.max(abs(accepted-m))<np.max(abs(f-m))
    for old,new in zip(moments(f),moments(accepted),strict=True):
        np.testing.assert_allclose(old,new,rtol=0,atol=1e-12)
    for old,new in zip(moments(f),moments(m),strict=True):
        np.testing.assert_allclose(old,new,rtol=0,atol=1e-12)
    # SSPRK2 recomputes the predictor moments; no exact exponential time step substitute.
    h=1/128;nu=1.5
    direct=(1-h*nu+(h*nu)**2/2)*f+(h*nu-(h*nu)**2/2)*m
    np.testing.assert_allclose(accepted,direct,rtol=0,atol=1e-14)
    assert np.max(abs(accepted-(m+(f-m)*np.exp(-nu*h))))>1e-9

@pytest.mark.parametrize('kind',('nan','zero','negative'))
def test_equilibrium_refuses_nonphysical_reference_inputs(kind):
    f=initial();f[:]=np.nan if kind=='nan' else 0 if kind=='zero' else -1
    with pytest.raises(ValueError,match='finite|positive'):equilibrium(f)
