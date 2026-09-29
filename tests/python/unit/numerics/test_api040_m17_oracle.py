"""Pure independent M17 mathematics; no PoPS compiler or native dependency."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples/migration/scientific"))
from api040_m17_oracle import (CONSERVATIVE, INDICES, flux, gaussian_mixture,
                               independent_path, primary_product, solve_reference)


def _mixture(weight):
    return np.asarray(gaussian_mixture((
        (weight, (.15, -.1), (.8, 0., 1.1)),
        (1.-weight, (-.2, .15), (1.2, 0., .9)))))


def test_gaussian_mixture_has_exact_mass_momentum_and_second_moments():
    weight = .6
    raw = _mixture(weight)
    assert INDICES[0] == (0, 0)
    np.testing.assert_allclose(raw[[0, 1, 5]],
                               (1., weight*.15+(1-weight)*(-.2),
                                weight*(-.1)+(1-weight)*.15), atol=2.e-15)
    np.testing.assert_allclose(raw[[2, 6, 9]],
                               (weight*(.8+.15**2)+(1-weight)*(1.2+(-.2)**2),
                                weight*(.15*(-.1))+(1-weight)*(-.2*.15),
                                weight*(1.1+(-.1)**2)+(1-weight)*(.9+.15**2)), atol=2.e-15)


def test_independent_primary_product_conserves_exactly_ten_rows_and_is_active():
    left, right = _mixture(.52), _mixture(.58)
    action = primary_product(.35*left+.65*right, right-left, (1., 0.))
    assert np.array_equal(action[list(CONSERVATIVE)], np.zeros(10))
    assert np.max(np.abs(action)) > 1.e-8


def test_density_ratio_path_quadrature_24_48_and_full_small_oracle():
    left = _mixture(.55)
    for density_ratio in (.2, 1., 5.):
        right = density_ratio*_mixture(.58)
        a = independent_path(left[:, None], right[:, None], (1., 0.), points=24)
        b = independent_path(left[:, None], right[:, None], (1., 0.), points=48)
        assert np.max(np.abs(a-b)) < 3.e-8
        if density_ratio != 1.:
            # The authored Gauss4 is an approximation, not a certified exact
            # integral for all endpoint density contrasts.
            coarse = independent_path(left[:, None], right[:, None], (1., 0.), points=4)
            assert np.max(np.abs(coarse-b)) > 1.e-8
    n = 4
    centers = (np.arange(n)+.5)/n
    weights = .55+.03*np.sinc(1./n)*np.cos(2*np.pi*centers)
    initial = np.stack([_mixture(float(w)) for w in weights], axis=1)
    final = solve_reference(initial, 1.e-4)
    assert np.all(np.isfinite(final))
    assert np.max(np.abs(final[list(CONSERVATIVE)].mean(axis=1)
                         - initial[list(CONSERVATIVE)].mean(axis=1))) < 3.e-13


def test_endpoint_raw_second_majorant_covers_complete_numerical_characteristics():
    # Independent finite-difference check of DF+B, not a proof on all states.
    left, right = .2*_mixture(.51), 5.*_mixture(.59)
    for axis, second in (((1., 0.), 2), ((0., 1.), 9)):
        endpoints = [np.sqrt(6+np.sqrt(10))*np.sqrt(state[second]/state[0])
                     for state in (left, right)]
        bound = max(endpoints)
        for s in (0., .2, .5, .8, 1.):
            state = (1-s)*left+s*right
            jacobian = np.empty((15, 15))
            for j in range(15):
                step = 1.e-6*max(1., abs(state[j]))
                perturb = np.zeros(15)
                perturb[j] = step
                jacobian[:, j] = (flux(state+perturb, axis)-flux(state-perturb, axis)
                                  + primary_product(state, 2*perturb, axis))/(2*step)
            radius = np.max(np.abs(np.linalg.eigvals(jacobian)))
            assert radius <= bound*(1+2.e-5)
