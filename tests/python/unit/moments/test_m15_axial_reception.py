"""Frozen numerical reception criteria and true moment averages before native runs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load(name):
    path = Path(__file__).resolve().parents[4] / "examples/migration/scientific" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


oracle = _load("api040_m15_axial_oracle.py")


@pytest.mark.parametrize("cells", oracle.RESOLUTIONS)
def test_cell_means_integrate_the_positive_distribution_and_preflight(cells):
    points, weights = np.polynomial.legendre.leggauss(12)
    x = (np.arange(cells)[:, None]+.5+.5*points)/cells
    mixture = (oracle.normal_moments(-.4, .3)[:, None, None]*(.6+.08*np.cos(2*np.pi*x))
               + oracle.normal_moments(.7, .2)[:, None, None]*(.4+.06*np.sin(2*np.pi*x)))
    integrated = .5*np.sum(mixture*weights, axis=-1)
    initial = oracle.initial_averages(cells)
    np.testing.assert_allclose(initial, integrated, rtol=0, atol=6.e-16)
    center = (oracle.normal_moments(-.4, .3)[:, None]*(.6+.08*np.cos(2*np.pi*(np.arange(cells)+.5)/cells))
              + oracle.normal_moments(.7, .2)[:, None]*(.4+.06*np.sin(2*np.pi*(np.arange(cells)+.5)/cells)))
    assert np.max(np.abs(initial-center)) > 1.e-6
    final, courant = oracle.trajectory(cells)
    oracle.require_admissible(final)
    assert courant < oracle.CRITERIA["preflight_face_courant_max"]
    np.testing.assert_allclose(final.mean(axis=1), initial.mean(axis=1), rtol=0, atol=2.e-14)
    assert np.max(np.abs(final-initial)) > 1.e-3


def test_invalid_domain_is_not_repaired_and_b1_is_not_gaussian():
    for label, raw in oracle.invalid_states(8).items():
        with pytest.raises(ValueError, match="moment domain"):
            oracle.speeds(raw)
        rho, h2, h3 = (value[0] for value in oracle.domain_values(raw))
        assert (rho <= 0, h2 <= 0, h3 < 0) == {
            "negative_density": (True, False, False),
            "zero_variance": (False, True, False),
            "negative_hankel": (False, False, True),
        }[label]
    raw = oracle.initial_averages(32)
    u, variance = raw[1]/raw[0], raw[2]/raw[0]-(raw[1]/raw[0])**2
    gaussian = raw[0]*(u**5+10*u**3*variance+15*u*variance**2)
    assert np.max(np.abs(oracle.flux(raw)[4]-gaussian)) > .01
    example = _load("api040_m15_hyqmom_axial_b1.py")
    assert example.is_native_admission_refusal(
        "prepared ND hyperbolic face evaluation refused publication status=1")
    assert not example.is_native_admission_refusal("C++ compilation failed")
    assert not example.is_native_admission_refusal(
        "prepared ND hyperbolic face evaluation refused publication status=10")


def test_predeclared_oracle_tolerance_distinguishes_hll_from_rusanov():
    cells = 32
    raw = oracle.initial_averages(cells)
    right = np.roll(raw, -1, axis=1)
    lower, upper = oracle.speeds(raw)
    speed = np.maximum(np.abs(lower), np.abs(upper))
    bound = np.maximum(speed, np.roll(speed, -1))
    face = .5*(oracle.flux(raw)+oracle.flux(right))-.5*bound*(right-raw)
    wrong_rhs = -cells*(face-np.roll(face, 1, axis=1))
    actual_rhs, _ = oracle.rhs(raw)
    assert np.max(np.abs(wrong_rhs-actual_rhs))/(100*cells) > 20*oracle.CRITERIA["state_max_error"]


@pytest.mark.parametrize("order", oracle.ORDERS)
def test_true_one_dimensional_permutation_authors_and_emits(order):
    from tests.python.unit.moments.test_api040_m15_m16 import _emit
    example = _load("api040_m15_hyqmom_axial_b1.py")
    case, layout, state = example.build_case(8, order=order, dt=1/800)
    assert tuple(state.components) == tuple("M%d" % k for k in order)
    program, physical = _emit(case, layout)
    assert "sqrt" in physical and physical.count("F[") == 5
    assert "pops_program" in program
    # Existing default authoring must remain available without selecting FixedDt.
    example.build_case(8)
