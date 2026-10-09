"""Independent multidirectional Fourier/energy/face oracles, not native execution."""
import numpy as np
import pytest

from tests.python.support.coupled_gradient_nd_oracle import (
    energy_rate, face_records, fourier_cell_means, laplacian_symbol, ledger_increment, rate,
)

D = np.outer((1., 2., -1.), (1., 2., -1.))
R = np.array(((0., -.3, .2), (.3, 0., -.1), (-.2, .1, 0.)))
CASES = [((8, 10), (1.3, 2.7), (1, 2)),
         ((6, 8, 10), (1.1, 2.3, 3.7), (1, 2, 1))]


@pytest.mark.parametrize("cells,lengths,modes", CASES)
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_oblique_cell_average_symbol_permutation_and_energy(cells, lengths, modes, order):
    spacings = np.array(lengths)/cells
    initial = fourier_cell_means(cells, lengths, modes, (1+.2j, -.3+.5j, .7-.1j))
    matrix = D+R
    actual = rate(initial, matrix, spacings)
    expected = laplacian_symbol(cells, lengths, modes)*np.einsum("ij,j...->i...", matrix, initial)
    np.testing.assert_allclose(actual, expected, rtol=2.e-14, atol=2.e-12)
    np.testing.assert_allclose(rate(initial[list(order)], matrix[np.ix_(order, order)], spacings),
                               actual[list(order)], rtol=2.e-14, atol=2.e-12)
    energy = np.prod(spacings)*np.sum(initial*actual)
    assert energy < 0
    np.testing.assert_allclose(energy, energy_rate(initial, D, spacings), rtol=2.e-14)
    assert abs(np.prod(spacings)*np.sum(initial*rate(initial, R, spacings))) < 2.e-12
    # Omitting an axis or reusing h_x cannot pass this oblique witness.
    wrong_symbol = -4*sum(np.sin(np.pi*np.array(modes)/cells)**2)/spacings[0]**2
    assert abs(wrong_symbol-laplacian_symbol(cells, lengths, modes)) > 1


@pytest.mark.parametrize("cells,lengths,modes", CASES)
def test_face_measures_orientations_and_ssprk2_ledger(cells, lengths, modes):
    spacings = np.array(lengths)/cells
    initial = fourier_cell_means(cells, lengths, modes, (1+.2j, -.3+.5j, .7-.1j))
    dt = 1.e-5
    predictor = initial+dt*rate(initial, D+R, spacings)
    records = list(face_records(initial, D+R, spacings, .5*dt))
    records += list(face_records(predictor, D+R, spacings, .5*dt))
    assert len(records) == 2*3*np.prod(cells)*2*len(cells)
    expected = .5*dt*(rate(initial, D+R, spacings)+rate(predictor, D+R, spacings))
    actual = ledger_increment(records, initial.shape, spacings)
    np.testing.assert_allclose(actual, expected, rtol=2.e-13, atol=2.e-16)
    np.testing.assert_allclose(actual.sum(axis=tuple(range(1, actual.ndim))), 0, atol=2.e-15)
    wrong = [dict(row, face_measure=1.) for row in records]
    assert np.max(abs(ledger_increment(wrong, initial.shape, spacings)-expected)) > 1.e-4


@pytest.mark.parametrize("cells,lengths,modes", CASES)
def test_null_direction_constant_mode_and_skew_temporal_limit(cells, lengths, modes):
    spacings = np.array(lengths)/cells
    constant = fourier_cell_means(cells, lengths, (0,)*len(cells), (1., -2., .5))
    assert laplacian_symbol(cells, lengths, (0,)*len(cells)) == 0
    np.testing.assert_array_equal(rate(constant, D+R, spacings), 0)
    directional = (0,)*(len(cells)-1)+(1,)
    value = fourier_cell_means(cells, lengths, directional, (1., 0., 0.))
    np.testing.assert_allclose(rate(value, D+R, spacings),
        laplacian_symbol(cells, lengths, directional)*np.einsum("ij,j...->i...", D+R, value),
        rtol=2.e-14, atol=2.e-12)
    # SSPRK2 amplification on a nonzero imaginary eigenvalue exceeds unity.
    z = .2j
    assert abs(1+z+z*z/2)**2 == pytest.approx(1+.2**4/4)
    assert abs(1+z+z*z/2) > 1


@pytest.mark.parametrize("cells,lengths,modes", CASES)
def test_rank_one_kernel_and_checkerboard_are_distinct(cells, lengths, modes):
    spacings = np.array(lengths)/cells
    # This component vector is in ker(D), while R still evolves it.
    initial = fourier_cell_means(cells, lengths, modes, (2., -1., 0.))
    np.testing.assert_allclose(rate(initial, D, spacings), 0, atol=1.e-13)
    assert np.max(abs(rate(initial, R, spacings))) > 1
    assert abs(energy_rate(initial, D, spacings)) < 1.e-12
    # Imaginary amplitude avoids zero cosine averages at checkerboard centers.
    checkerboard_mode = (cells[0]//2,)+(0,)*(len(cells)-1)
    checkerboard = fourier_cell_means(cells, lengths, checkerboard_mode, (1j, 0j, 0j))
    assert np.max(abs(checkerboard)) > .5
    symbol = laplacian_symbol(cells, lengths, checkerboard_mode)
    assert symbol == pytest.approx(-4/spacings[0]**2)
    np.testing.assert_allclose(rate(checkerboard, D+R, spacings),
        symbol*np.einsum("ij,j...->i...", D+R, checkerboard), rtol=2.e-14, atol=2.e-12)


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_old_frequency_factor_does_not_bound_discrete_checkerboard(dimension):
    cells = (4,)*dimension
    spacings = (np.array((.2, .3, .7)))[:dimension]
    checkerboard = (-1.)**np.indices(cells).sum(axis=0)
    initial = np.stack((checkerboard, np.zeros(cells)))
    skew = np.array(((0., -.3), (.3, 0.)))
    observed = np.max(abs(rate(initial, skew, spacings)))/np.max(abs(initial))
    row_norm = np.max(np.sum(abs(skew), axis=1))
    old_bound = 2.5*row_norm*np.sum(spacings**-2)
    new_bound = 4*row_norm*np.sum(spacings**-2)
    assert observed > old_bound  # Already false as a magnitude bound in Dim1.
    assert observed == pytest.approx(new_bound, rel=2.e-15)
