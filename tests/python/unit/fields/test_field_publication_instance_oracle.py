"""Source NumPy controls for the prospective installed witness, not Native results."""
import numpy as np
import pytest
from tests.python.support.field_publication_instance_oracle import data,stage_potential,step


def negative_laplacian(value,cells):
    return sum(n*n*(2*value-np.roll(value,1,axis)-np.roll(value,-1,axis))
               for n,axis in zip(cells,(1,0),strict=True))


@pytest.mark.parametrize('cells', ((7,3),(11,2),(3,11)))
def test_discrete_equations_stage_time_and_readonly_catalyst(cells):
    states=data(cells);forcing=states[3]
    nx,_=cells
    eigenvalue=4*nx*nx*np.sin(np.pi/nx)**2
    for fraction in (1/4,3/4):
        ramp=forcing[0]*(1+fraction/64)
        phi=stage_potential(forcing,cells,fraction,1/64)
        np.testing.assert_allclose(negative_laplacian(phi,cells),ramp,atol=2e-14,rtol=0)
        beta=stage_potential(forcing,cells,fraction,1/64,different=True)
        alpha=(eigenvalue+6)/(eigenvalue*(eigenvalue+4))*ramp
        np.testing.assert_allclose(negative_laplacian(alpha,cells)+2*(alpha-beta),ramp,atol=3e-14,rtol=0)
        np.testing.assert_allclose(negative_laplacian(beta,cells)+2*(beta-alpha),2*ramp,atol=5e-14,rtol=0)
    assert not np.array_equal(stage_potential(forcing,cells,1/4,1/64),
                              stage_potential(forcing,cells,3/4,1/64))
    current=states
    for index in (1,2):
        current=step(current,cells,(index-1)/64,different=True)
        assert current[2].tobytes()==states[2].tobytes()
        np.testing.assert_allclose(current[0][0]+current[1][0],states[0][0]+states[1][0],atol=8e-16,rtol=0)
    wrong_sibling=step(states,cells,0,different=False)
    right=step(states,cells,0,different=True)
    assert np.max(np.abs(wrong_sibling[0]-right[0]))>1e-8
