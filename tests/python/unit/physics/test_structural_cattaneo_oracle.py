"""Pure matrix-exponential and cell-average checks, no PoPS runtime."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples/migration/scientific"))
import structural_cattaneo_oracle as oracle


@pytest.mark.parametrize("tau,kappa", [(0., .1), (-.1, .1), (.1, 0.), (.1, -.2),
                                        (float("nan"), .1), (.1, float("inf"))])
def test_constitutive_parameters_require_positive_finite_values(tau, kappa):
    with pytest.raises(ValueError, match="finite tau>0"):
        oracle.coefficients(tau, kappa)


def test_full_principal_symbol_has_longitudinal_waves_and_transverse_zero_mode():
    tau, kappa = .1, .025
    x, y, _ = oracle.coefficients(tau, kappa)
    for normal in ((1., 0.), (0., 1.), (.6, -.8)):
        symbol = normal[0] * x + normal[1] * y
        speed = np.sqrt(kappa / tau) * np.linalg.norm(normal)
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(symbol)), (-speed, 0., speed), atol=1e-15)
    assert x[0, 1] != 0 and x[1, 0] != 0 and y[0, 2] != 0 and y[2, 0] != 0


def test_true_two_dimensional_cell_averages_match_independent_quadrature():
    n = 16
    values = oracle.field(oracle.MEAN, oracle.AMPLITUDE, n)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    for iy, ix in ((0, 0), (3, 9), (15, 15)):
        average = np.zeros(3)
        for a, wa in zip(nodes, weights, strict=True):
            for b, wb in zip(nodes, weights, strict=True):
                x, y = (ix + (1+a)/2)/n, (iy + (1+b)/2)/n
                average += wa * wb / 4 * (oracle.MEAN + np.real(
                    oracle.AMPLITUDE * np.exp(2j*np.pi*(x+2*y))))
        np.testing.assert_allclose(values[:, iy, ix], average, atol=8e-16, rtol=0.)
    np.testing.assert_allclose(values.mean(axis=(1, 2)), oracle.MEAN, atol=1e-15)


def test_fourier_step_matches_separate_array_face_assembly_and_implicit_equation():
    n, dt, tau, kappa = 16, .004, .1, .025
    initial = oracle.exact(n, 0., tau, kappa)
    matrices = oracle.coefficients(tau, kappa)[:2]
    rhs = np.zeros_like(initial)
    for matrix, axis in zip(matrices, (2, 1), strict=True):
        right = np.roll(initial, -1, axis=axis)
        face = .5 * (np.einsum("ij,jyx->iyx", matrix, initial+right)
                     - np.sqrt(kappa/tau) * (right-initial))
        rhs += n * (np.roll(face, 1, axis=axis)-face)
    star = initial + dt*rhs
    final = star.copy()
    final[1:] /= 1 + dt/tau
    np.testing.assert_allclose(final, oracle.imex_euler(n, dt, 1, tau, kappa), atol=8e-16, rtol=0.)
    np.testing.assert_allclose(final[1:] - star[1:] + dt/tau*final[1:], 0., atol=2e-17)
    assert abs(final[0].mean() - initial[0].mean()) < 1e-15
    # Explicit relaxation is distinguishable from the declared implicit method.
    assert np.max(np.abs((star[1:] - dt/tau*star[1:]) - final[1:])) > 1e-5


def test_continuum_fourier_equation_and_dissipative_energy():
    tau, kappa = .1, .025
    principal, relaxation = oracle.symbols(tau, kappa)
    full = principal + relaxation
    h = np.diag((1., tau/kappa, tau/kappa))
    # A*H+HA has only the specified physical relaxation loss.
    np.testing.assert_allclose(full.conj().T@h + h@full,
                               np.diag((0., -2/kappa, -2/kappa)), atol=2e-14)
    wave0 = oracle.AMPLITUDE
    wave1 = expm(.04*full)@wave0
    assert np.real(wave1.conj()@h@wave1) < np.real(wave0.conj()@h@wave0)
    # Omitting the y coupling is observable: this is not an x extrusion.
    xonly, _ = oracle.symbols(tau, kappa, mode=(1, 0))
    assert np.max(np.abs(wave1-expm(.04*(xonly+relaxation))@wave0)) > 1e-3


@pytest.mark.parametrize("tau,kappa", [(.1, .025), (.12, .048)])
def test_predeclared_first_order_temporal_convergence_excludes_spatial_plateau(tau, kappa):
    final_time, n = .04, 16
    reference = oracle.exact(n, final_time, tau, kappa, discrete_space=True)
    errors = [np.max(np.abs(oracle.imex_euler(n, dt, round(final_time/dt), tau, kappa)-reference))
              for dt in (.004, .002, .001)]
    orders = np.log2(np.asarray(errors[:-1])/errors[1:])
    assert np.all(orders > .8) and np.all(orders < 1.2)
    spatial = np.max(np.abs(reference-oracle.exact(n, final_time, tau, kappa)))
    assert spatial > 10*errors[0]  # A continuum-only test would hide time convergence.


def test_rebind_changes_physics_and_not_only_parameter_metadata():
    first = oracle.exact(16, .04, .1, .025)
    second = oracle.exact(16, .04, .12, .048)
    assert np.max(np.abs(first-second)) > 1e-3
    np.testing.assert_allclose(first[0].mean(), 1., atol=1e-15)
    np.testing.assert_allclose(first[1:].mean(axis=(1,2)), oracle.MEAN[1:]*np.exp(-.4), atol=1e-16)
