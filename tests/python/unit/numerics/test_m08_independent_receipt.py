"""Independent analytic stencil and discriminating reception checks for M08."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest

EXAMPLES=Path(__file__).resolve().parents[4]/"examples/migration/scientific"
spec=importlib.util.spec_from_file_location("m08_review_oracle",EXAMPLES/"api040_m08_oracle.py")
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def criteria():
    tree=ast.parse((EXAMPLES/"api040_m08_guiding_center.py").read_text())
    value=next(node.value for node in tree.body if isinstance(node,ast.Assign)
               and any(isinstance(target,ast.Name) and target.id=="CRITERIA" for target in node.targets))
    return ast.literal_eval(value)


@pytest.mark.parametrize("n",(32,64))
def test_native_tolerance_rejects_deliberately_stale_transport(n):
    q,c=oracle.cell_means(n)
    fresh=oracle.ssprk2_reference(q,c,dt=.02,steps=2)
    stale=oracle.ssprk2_reference(q,c,dt=.02,steps=2,stale_second_stage=True)
    gap=max(float(np.max(abs(a-b))) for a,b in zip(fresh,stale))
    assert gap < 3.e-7  # The superseded threshold would accept this wrong trajectory.
    assert gap > 20*criteria()["reference_fv_max_error"]


@pytest.mark.parametrize("n",(16,32))
def test_fourier_mode_fixes_poisson_sign_and_both_gradient_orientations(n):
    h=2*np.pi/n
    x,y=np.meshgrid((np.arange(n)+.5)*h,(np.arange(n)+.5)*h)
    charge=np.cos(2*x)*np.sin(3*y)
    eigenvalue=4/h**2*(np.sin(h)**2+np.sin(1.5*h)**2)
    expected=charge/eigenvalue
    phi=oracle.poisson_discrete(charge)
    np.testing.assert_allclose(phi,expected,rtol=2e-14,atol=3e-16)
    ux,uy=oracle.guiding_velocity(phi)
    np.testing.assert_allclose(ux,np.cos(2*x)*np.cos(3*y)*np.sin(3*h)/(h*eigenvalue),atol=8e-16)
    np.testing.assert_allclose(uy,np.sin(2*x)*np.sin(3*y)*np.sin(2*h)/(h*eigenvalue),atol=8e-16)
    np.testing.assert_allclose(oracle.centered_divergence(ux,uy),0,atol=2e-15)
