"""Dense independently assembled original equations; no PoPS or author oracle."""
import numpy as np
import pytest
import importlib.util
from pathlib import Path


@pytest.mark.parametrize("cells", (5, 8, 13))
@pytest.mark.parametrize("permuted", (False, True))
def test_mixed_block_equations_permutation_and_unconverged_candidate(cells, permuted):
    dt, epsilon = .01, .08
    laplacian = -2*np.eye(cells)
    for index in range(cells):
        laplacian[index, (index-1) % cells] += 1
        laplacian[index, (index+1) % cells] += 1
    laplacian *= cells**2
    rng = np.random.default_rng(77271+cells)
    old = rng.uniform(.2, .6, cells)
    identity = np.eye(cells)
    original = np.block([[identity, -dt*laplacian],
                         [-identity+epsilon**2*laplacian, identity]])
    rhs = np.r_[old, np.zeros(cells)]
    indices = np.r_[np.arange(cells, 2*cells), np.arange(cells)] if permuted else np.arange(2*cells)
    candidate = np.linalg.solve(original[np.ix_(indices, indices)], rhs[indices])[np.argsort(indices)]
    c, mu = candidate[:cells], candidate[cells:]
    np.testing.assert_allclose(c-old-dt*(laplacian@mu), 0, atol=8.e-15)
    np.testing.assert_allclose(mu-c+epsilon**2*(laplacian@c), 0, atol=8.e-15)
    assert abs(c.mean()-old.mean()) < 5.e-16
    def energy(value):
        return .5/cells*(value@value-epsilon**2*value@laplacian@value)
    assert energy(c) < energy(old)
    # One unpreconditioned GMRES iteration from zero spans rhs only. Its
    # optimal least-squares candidate still violates the original block system.
    image = original@rhs
    one_iteration = rhs*(image@rhs)/(image@image)
    assert np.linalg.norm(original@one_iteration-rhs) > .1
    # Satisfying only the mass equation is not acceptance of the mixed pair.
    assert np.max(abs(old-old-dt*(laplacian@np.zeros(cells)))) == 0
    assert np.max(abs(-old+epsilon**2*(laplacian@old))) > .1
    # Cross-check the author's FFT oracle against this unrelated dense assembly
    # on non-corpus sizes and a non-Fourier-manufactured random load.
    path = Path(__file__).resolve().parents[4] / "examples/migration/scientific/api040_m27_mixed_oracle.py"
    spec = importlib.util.spec_from_file_location("m27_author_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    fourier_c, fourier_mu = oracle.fourier_mixed_step(old)
    np.testing.assert_allclose(fourier_c, c, rtol=2.e-14, atol=3.e-15)
    np.testing.assert_allclose(fourier_mu, mu, rtol=2.e-14, atol=3.e-15)
