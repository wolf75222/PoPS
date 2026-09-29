"""Independent exact relations and source-only authoring for M06/M13."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples/migration/scientific"
sys.path.insert(0, str(EXAMPLES))
from api040_m06_m13_oracles import (chain_exact, enthalpy_exact,
                                     enthalpy_thermometer)


def test_enthalpy_piecewise_and_reverse_latent_crossing():
    h = np.asarray([0., 1.5, 2., 3.5, 5., 5.5])
    t, f = enthalpy_thermometer(h)
    np.testing.assert_array_equal(t, [0., .75, 1., 1., 1., 1.25])
    np.testing.assert_allclose(f, [0., 0., 0., .5, 1., 1.], rtol=0., atol=0.)
    forward = enthalpy_exact(np.asarray([1.5]), 4., .5)
    reverse = enthalpy_exact(np.asarray([5.5]), -4., .5)
    for value in (forward, reverse):
        np.testing.assert_allclose(value[0], [3.5], atol=0.)
        np.testing.assert_allclose(value[1], [1.], atol=0.)
        np.testing.assert_allclose(value[2], [.5], atol=0.)


@pytest.mark.parametrize("law", [{"capacity": 0.}, {"latent": -1.},
                                   {"melting": float("nan")}])
def test_invalid_enthalpy_law_rejected(law):
    with pytest.raises(ValueError):
        enthalpy_thermometer([2.], **law)


def test_chain_matches_independent_matrix_exponential():
    scipy = pytest.importorskip("scipy.linalg")
    initial = np.asarray([[1., 1.2], [.2, .25], [.1, .1]])
    for kab, kbc in ((1.3, .4), (.4, 1.3), (1., 1.), (0., 0.)):
        matrix = np.asarray([[-kab, 0., 0.],
                             [kab, -kbc, 0.], [0., kbc, 0.]])
        expected = scipy.expm(matrix) @ initial
        actual = chain_exact(initial, kab, kbc, 1.)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=6.e-16)
        np.testing.assert_allclose(actual.sum(axis=0), initial.sum(axis=0),
                                   rtol=0., atol=5.e-16)
        assert actual.min() >= 0.


@pytest.mark.parametrize("rates", [(-.1, 1.), (1., float("inf"))])
def test_invalid_reaction_rates_rejected_by_oracle(rates):
    with pytest.raises(ValueError):
        chain_exact(np.asarray([1., 0., 0.]), *rates, 1.)


def test_chain_true_cell_means_and_permutation():
    from api040_m13_reaction_chain import cell_mean_initial
    cells = (6, 8)
    canonical = cell_mean_initial(cells, ("A", "B", "C"))
    reverse = cell_mean_initial(cells, ("C", "B", "A"))
    np.testing.assert_array_equal(reverse, canonical[::-1])
    edges = np.arange(cells[0]+1)/cells[0]
    integral = 1. + .1*np.diff(np.sin(2*np.pi*edges))*cells[0]/(2*np.pi)
    np.testing.assert_allclose(canonical[0, 0], integral, rtol=0., atol=3.e-16)
    assert np.max(np.abs(canonical[0, 0] -
                         (1.+.1*np.cos(2*np.pi*(edges[:-1]+.5/cells[0]))))) > 1.e-5


def test_predeclared_ssprk2_error_is_below_receipt_threshold():
    # Numerical method check, not the oracle itself: the exact solution above
    # remains a separate closed form and SciPy matrix exponential cross-check.
    from api040_m13_reaction_chain import DT, RATES, CRITERIA
    initial = np.asarray([1., .2, .1])
    for _, kab, kbc in RATES:
        matrix = np.asarray([[-kab, 0., 0.],
                             [kab, -kbc, 0.], [0., kbc, 0.]])
        value = initial.copy()
        for _ in range(round(1./DT)):
            first = value + DT*matrix@value
            value = value + .5*DT*(matrix@value + matrix@first)
        assert np.max(np.abs(value-chain_exact(initial, kab, kbc, 1.))) < CRITERIA["state_max_error"]


def test_invalid_law_and_order_rejected_before_publication():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from api040_m06_enthalpy import build_case as enthalpy_case
    from api040_m13_reaction_chain import build_case as reaction_case
    frame = Rectangle("M06_M13_invalid_domain", lower=(0.,0.),
                      upper=(1.,1.)).frame(Cartesian2D())
    with pytest.raises(ValueError):
        enthalpy_case(frame, latent=0.)
    with pytest.raises(ValueError):
        reaction_case(frame, order=("A","A","C"))


def test_programs_validate_and_resolve_source_only():
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from api040_m06_enthalpy import build_case as enthalpy_case
    from api040_m13_reaction_chain import build_case as reaction_case

    for label, builder, cells, upper, kwargs in (
        ("M06", enthalpy_case, (4,4), (1.,2.), {}),
        ("M06_alt", enthalpy_case, (6,8), (2.,1.),
         {"capacity": 3., "latent": 2., "melting": .5}),
        ("M13", reaction_case, (4,4), (1.,2.), {}),
        ("M13_reverse", reaction_case, (6,8), (2.,1.),
         {"order": ("C","B","A")}),
    ):
        frame = Rectangle(label+"_source_domain", lower=(0., 0.),
                          upper=upper).frame(Cartesian2D())
        case, declarations = builder(frame, name=label+"_source_case", **kwargs)
        validated = pops.validate(case)
        for parameter in (declarations if isinstance(declarations, tuple) else (declarations,)):
            assert validated.resolve(parameter) is not None
        layout = Uniform(CartesianGrid(frame=frame, cells=cells,
                                       periodic=PeriodicAxes(frame.axes)))
        assert pops.resolve(validated, layout=layout) is not None
