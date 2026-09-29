"""Pre-native M15 criteria on the independently specified non-Gaussian data."""

from __future__ import annotations

from math import pi

import numpy as np

from tests.python.support import api040_m15_independent_oracle as oracle


def _quadrature_means(cells):
    nodes,weights=np.polynomial.legendre.leggauss(40)
    left=np.array([oracle.normal_raw(k,-.4,.3) for k in range(5)])
    right=np.array([oracle.normal_raw(k,.7,.2) for k in range(5)])
    output=np.empty((5,cells))
    for index in range(cells):
        x=(index+.5+.5*nodes)/cells
        density=(.6+.08*np.cos(2*pi*x))[None,:]*left[:,None]
        density+=(.4+.06*np.sin(2*pi*x))[None,:]*right[:,None]
        output[:,index]=.5*density@weights
    return output


def test_cell_means_are_true_integrals_and_realizable_for_all_planned_grids():
    for cells in (32,64,128):
        initial=oracle.exact_cell_means(cells)
        np.testing.assert_allclose(initial,_quadrature_means(cells),rtol=0,
                                   atol=oracle.INITIAL_ATOL)
        assert np.min(initial[0])>.9
        assert oracle.hankel_minimum(initial)>.25
        assert oracle.planned_steps(cells)==2*cells
        assert abs(oracle.planned_steps(cells)/(100*cells)-oracle.FINAL_TIME)<oracle.TIME_ATOL


def test_b1_fifth_is_discriminating_and_characteristic_bound_precedes_native():
    for cells in (32,64,128):
        initial=oracle.exact_cell_means(cells)
        gap=np.max(np.abs(oracle.b1_flux(initial)[4]-oracle.gaussian_fifth(initial)))
        assert gap>oracle.NON_GAUSSIAN_GAP
        speed=max(max(abs(value) for value in oracle.signed_speeds(initial[:,i]))
                  for i in range(cells))
        assert speed/(100)<.1  # exact dt/h; stricter than authored CFL=.1
    left=oracle.exact_cell_means(32)[:,0]
    assert tuple(oracle.b1_flux(left)[:4])==tuple(left[1:])


def test_finite_volume_oracle_is_conservative_and_remains_interior():
    initial=oracle.exact_cell_means(32)
    final=oracle.forward_euler(initial,oracle.planned_steps(32))
    assert np.max(np.abs(final-initial))>1.e-5
    assert oracle.hankel_minimum(final)>0
    np.testing.assert_allclose(final.sum(axis=1),initial.sum(axis=1),rtol=0,
                               atol=oracle.INTEGRAL_ATOL*32)
