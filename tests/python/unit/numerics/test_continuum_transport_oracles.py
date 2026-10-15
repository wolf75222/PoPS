"""Discriminating controls for independent discrete transport receivers."""
import numpy as np

from examples.api_v02.continuum_transport_oracles import (
    anisotropic_ssprk2, guiding_center_ssprk2, periodic_poisson,
)


def test_full_tensor_fourier_symbol_matches_an_independent_real_stencil():
    ny, nx = 8, 16
    hx, hy = 1 / nx, 1 / ny
    y, x = np.meshgrid((np.arange(ny)+.5)/ny, (np.arange(nx)+.5)/nx, indexing="ij")
    initial = 1 + .2 * np.cos(2*np.pi*(x+y))
    velocity, tensor = (.4, -.2), ((.02, .006), (.006, .015))
    def rhs(u):
        out = np.zeros_like(u)
        for axis, h, a, k in ((1, hx, .4, .02), (0, hy, -.2, .015)):
            left, right = np.roll(u, 1, axis), np.roll(u, -1, axis)
            out += -a * (right-left)/(2*h) + abs(a)*(right-2*u+left)/(2*h)
            out += k * (right-2*u+left)/h**2
        out += .012 * (np.roll(np.roll(u, -1, 0), -1, 1)
                       - np.roll(np.roll(u, -1, 0), 1, 1)
                       - np.roll(np.roll(u, 1, 0), -1, 1)
                       + np.roll(np.roll(u, 1, 0), 1, 1))/(4*hx*hy)
        return out
    dt = .03 / nx**2
    expected = initial.copy()
    for _ in range(2):
        predictor = expected + dt*rhs(expected)
        expected = .5*expected + .5*(predictor + dt*rhs(predictor))
    options = dict(velocity=velocity, lengths=(1., 1.), dt=dt, steps=2)
    actual = anisotropic_ssprk2(initial, tensor=tensor, **options)
    np.testing.assert_allclose(actual, expected, atol=7e-16, rtol=0)
    diagonal = anisotropic_ssprk2(initial, tensor=((.02, 0.), (0., .015)), **options)
    assert np.max(abs(actual-diagonal)) > 1e-6


def test_guiding_transport_distinguishes_velocity_sign_despite_equal_poisson_residual():
    n = 16
    y, x = np.meshgrid(2*np.pi*(np.arange(n)+.5)/n, 2*np.pi*(np.arange(n)+.5)/n, indexing="ij")
    initial = np.sin(x) + .3*np.cos(2*y)
    options = dict(lengths=(2*np.pi, 2*np.pi), dt=.002, steps=2)
    actual, _ = guiding_center_ssprk2(initial, **options)
    wrong, _ = guiding_center_ssprk2(initial, velocity_sign=-1, **options)
    assert np.max(abs(actual-wrong)) > 1e-5
    assert abs(actual.mean()-initial.mean()) < 1e-15
    assert abs(wrong.mean()-initial.mean()) < 1e-15
    for values in (actual, wrong):
        phi = periodic_poisson(values, options["lengths"])
        recovered = sum(2*phi-np.roll(phi, -1, a)-np.roll(phi, 1, a) for a in (0, 1))/(2*np.pi/n)**2
        np.testing.assert_allclose(recovered, values, atol=2e-14, rtol=0)
