"""Independent synthetic MMS algebra only: no PoPS, solver or native bytes."""
from fractions import Fraction
import numpy as np
import pytest


def conservative_divergence(q, matrix, spacing):
    """One positive physical face, opposing cell budgets; arithmetic face D."""
    out = np.zeros_like(q)
    for axis, h in enumerate(spacing):
        for cell in np.ndindex(q.shape[1:]):
            other = list(cell)
            other[axis] = (other[axis]+1) % q.shape[axis+1]
            other = tuple(other)
            jump = q[(slice(None), *other)]-q[(slice(None), *cell)]
            face = (matrix[(slice(None), slice(None), *cell)] +
                    matrix[(slice(None), slice(None), *other)])/2
            flux = face@jump/h**2
            out[(slice(None), *cell)] += flux
            out[(slice(None), *other)] -= flux
    return out


def stage_data(width, cells=(5, 4)):
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0],
                       1.5*(np.arange(cells[1])+.5)/cells[1], indexing="ij")
    target = np.array([.2+.03*np.cos(2*np.pi*(i+1)*x)+.02*np.sin(2*np.pi*y/1.5)
                       for i in range(width)])
    prior = target-.04
    d = np.eye(width)*.012
    if width == 3:
        d = np.array([[.012,.002,0],[-.001,.014,0],[.001,0,0]])  # signed, singular
    alpha = .25*np.sin(2*np.pi*x)+.15*np.cos(2*np.pi*y/1.5)
    # Matrix column follows the differentiated candidate; no row-only fiction.
    material = d[:, :, None, None]*(1+alpha)[None, None]*(1+3*target**2)[None]
    spatial = conservative_divergence(target, material, (1/cells[0],1.5/cells[1]))
    tau = float(Fraction(1, 200))
    conserved = target+target**3
    previous = prior+prior**3
    forcing = (conserved-previous)/tau-spatial
    return target, previous, conserved, spatial, forcing, tau


@pytest.mark.parametrize("width", (1,3))
def test_independent_forcing_closes_exact_additive_original_stage(width):
    _, previous, q, spatial, forcing, tau = stage_data(width)
    residual = q-tau*(spatial+forcing)-previous
    assert np.max(np.abs(residual)) < 2e-16
    assert np.max(np.abs(np.sum(spatial, axis=(1,2)))) < 2e-14
    # Corruptions must be visible in original residual, not just seed proximity.
    for bad in (q-2*tau*(spatial+forcing)-previous,
                q-tau*spatial-previous, q-tau*(spatial+forcing)-q):
        assert np.max(np.abs(bad)) > 1e-3


@pytest.mark.parametrize("width", (1,3))
def test_autonomous_source_is_finite_at_zero_target_and_previous(width):
    source = np.arange(1,width+1,dtype=float)[:,None,None]*np.ones((width,3,2))
    tau = float(Fraction(1,100))
    target = np.zeros_like(source)
    previous = -tau*source
    assert np.array_equal(target-tau*source-previous, np.zeros_like(source))
    with np.errstate(divide="ignore",invalid="ignore"):
        fictional_rate = (source/target)*target
    assert np.any(~np.isfinite(fictional_rate))


def test_permutation_is_simultaneous_rows_columns_and_Q_components():
    target, previous, q, spatial, forcing, tau = stage_data(3)
    permutation = np.array([2,0,1])
    assert np.max(np.abs(q[permutation]-tau*(spatial[permutation]+forcing[permutation])-
                         previous[permutation])) < 2e-16
    wrong_forcing = forcing[[1,2,0]]
    assert np.max(np.abs(q[permutation]-tau*(spatial[permutation]+wrong_forcing)-
                         previous[permutation])) > 1e-3
    assert target.shape == (3,5,4)


@pytest.mark.parametrize("cells",(8,16))
@pytest.mark.parametrize("dt",(.01,.02))
@pytest.mark.parametrize("width,candidate",((1,False),(2,True)))
def test_frozen_native_fixture_forcing_against_independent_face_budgets(cells,dt,width,candidate):
    # The extracted author functions are the comparison target, never the oracle.
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    path = root/"tests/python/support/evolved_stage_mms.py"
    tree = ast.parse(path.read_text())
    defs = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
            ("accumulation","diffusion_action","manufactured_data")]
    namespace = {"np":np,"DIFFUSION":np.array([[.012,.002],[-.001,.014]]),"CANDIDATE_BETA":.4}
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(path),"exec"),namespace)
    actual,initial_t,target,_,volumes = namespace["manufactured_data"](cells,width,dt,candidate_diffusion=candidate)
    c = (np.arange(cells)+.5)/cells
    y,x = np.meshgrid(c,c,indexing="ij")
    before = np.array([.2+.03*np.sin(2*np.pi*x)+.02*np.cos(2*np.pi*y)+.07*i for i in range(width)])
    after = np.array([.21+.025*np.cos(2*np.pi*(i+1)*x)+.018*np.sin(2*np.pi*y)+.07*i for i in range(width)])
    def independent_Q(t):
        if width == 1:
            return t+t**2
        a,b = t
        return np.array([a+a**2+.1*b**2,b+b**2+.2*a*b])
    d = np.array([[.012,.002],[-.001,.014]])[:width,:width]
    matrix = np.broadcast_to(d[:,:,None,None],(width,width,cells,cells)).copy()
    if candidate:
        matrix *= (1+.4*after**2)[None]
    spatial = conservative_divergence(after,matrix,(1/cells,1/cells))
    previous,q = independent_Q(before),independent_Q(after)
    expected = (q-previous)/dt-spatial
    np.testing.assert_array_equal(initial_t,before)
    np.testing.assert_array_equal(target,after)
    for i in range(width):
        np.testing.assert_allclose(actual["Q%d"%i],previous[i:i+1],rtol=0,atol=1e-16)
    np.testing.assert_allclose(actual["forcing"],expected,rtol=0,atol=8e-15)
    np.testing.assert_array_equal(volumes,np.full((cells,cells),1/cells**2))
    assert np.max(np.abs(q-dt*(spatial+actual["forcing"])-previous)) < 3e-16
    assert np.max(np.abs(np.sum(q-previous-dt*actual["forcing"],axis=(1,2))/cells**2)) < 3e-16
    assert np.max(np.abs(q-2*dt*(spatial+actual["forcing"])-previous)) > 1e-3
    assert np.max(np.abs(q-dt*spatial-previous)) > 1e-3
