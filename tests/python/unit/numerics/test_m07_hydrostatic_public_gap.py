"""Historical route refusals and frozen reference; no native acceptance claim."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def _load(name):
    path = Path(__file__).resolve().parents[4] / "examples/migration/scientific" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load("api040_m07_hydrostatic_oracle")
example = _load("api040_m07_saint_venant")


@pytest.mark.parametrize("n", oracle.RESOLUTIONS)
def test_true_means_and_frozen_full_time_lake_reference(n):
    edges = np.linspace(-1., 1., n+1)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    centers, dx = .5*(edges[1:]+edges[:-1]), 2./n
    x = centers[:, None] + .5*dx*nodes
    integrated = .5*np.sum(.2*np.exp(-50*x*x)*weights, axis=1)
    initial = oracle.initial_averages(n)
    np.testing.assert_allclose(initial[2], integrated, rtol=0., atol=2.e-15)
    assert np.max(np.abs(initial[2] - .2*np.exp(-50*centers**2))) > 1.e-5
    final, receipt = oracle.trajectory(n)
    assert receipt["time"] == 1.
    assert receipt["steps"] == 5*n
    assert receipt["max_courant"] <= .1 + 2.e-16
    assert receipt["error"] <= oracle.EQUILIBRIUM_TOLERANCE
    np.testing.assert_array_equal(final[2], initial[2])
    # Both omission and double counting are discriminated before native execution.
    correct, _ = oracle.rhs(initial)
    omitted, _ = oracle.rhs(initial, source_factor=0.)
    doubled, _ = oracle.rhs(initial, source_factor=2.)
    dt = 1./(5*n)
    assert np.max(np.abs(dt*omitted[1])) > 1.e-3
    assert np.max(np.abs(dt*doubled[1])) > 1.e-3
    np.testing.assert_allclose(omitted[1] + doubled[1], 2*correct[1], atol=2.e-14)


def test_two_distinct_face_contributions_and_nontrivial_off_equilibrium_response():
    left, right = np.array([.9, 0., .1]), np.array([.8, 0., .2])
    lower, upper, _ = oracle.face_contributions(left, right)
    np.testing.assert_allclose(lower, [0., .405, 0.], rtol=0., atol=1.e-16)
    np.testing.assert_allclose(upper, [0., .32, 0.], rtol=0., atol=1.e-16)
    assert abs(lower[1] - upper[1]) > .08
    state = oracle.initial_averages(40)
    state[0, 19] += .001
    derivative, _ = oracle.rhs(state)
    assert np.max(np.abs(derivative[:2])) > .01
    np.testing.assert_array_equal(derivative[2], 0.)
    assert abs(np.sum(derivative[0])) < 1.e-14


@pytest.mark.parametrize("order", oracle.ORDERS)
def test_legacy_user_does_not_silently_accept_a_coordinated_face_output(order):
    # The new CoordinatedFace contract does not reinterpret old User output.
    with pytest.raises(TypeError, match="one Expr per state component"):
        example.unavailable_hydrostatic_method(order)


@pytest.mark.parametrize("n", oracle.RESOLUTIONS)
@pytest.mark.parametrize("order", oracle.ORDERS)
def test_each_native_mirror_boundary_reproduces_its_exact_equilibrium_ghost(n, order):
    dx = 2./n
    permutation = [oracle.ORDERS[0].index(name) for name in order]
    initial = oracle.initial_averages(n)[permutation]
    ghosts = oracle.equilibrium_averages([-1.-dx, -1., 1., 1.+dx])[permutation]
    values = example.equilibrium_boundary_values(n, order=order)
    for side, cell, ghost in ((0, 0, 0), (1, -1, -1)):
        np.testing.assert_allclose(2*np.array(values[side])-initial[:, cell],
                                   ghosts[:, ghost], rtol=0., atol=2.e-16)


@pytest.mark.parametrize("order", oracle.ORDERS)
def test_shared_user_cannot_bypass_the_retained_topographic_product(order):
    from pops.numerics import FiniteVolume, PathConservativeFiniteVolume
    from pops.numerics import reconstruction, riemann, variables
    model, state, flux, product, rate, path, face = example.declarations(order)
    assert len(rate.occurrences) == 2
    assert rate.occurrences[1].payload is product
    assert rate.occurrences[1].coefficient == -1
    # Returning one side is authorable, but would replace the other cell's correction.
    shared = riemann.User(body=lambda *args: face(*args)[0], state=state,
                          stability=lambda left, right, fl, fr, speed: speed)
    method = FiniteVolume(flux=flux, variables=variables.Conservative(state),
                          reconstruction=reconstruction.FirstOrder(), riemann=shared)
    with pytest.raises(ValueError, match="cannot omit a physical nonconservative product"):
        method.validate_rate_contract(model.rate_contract(rate))
    with pytest.raises(ValueError, match="FirstOrder and Rusanov"):
        PathConservativeFiniteVolume(flux=flux, path=path,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=shared)


@pytest.mark.parametrize("order", oracle.ORDERS)
def test_symbolic_required_faces_match_independent_numpy_under_permutation(order):
    face = example.declarations(order)[-1]
    left, right = np.array([.9, .07, .1]), np.array([.8, -.03, .2])
    indices = [oracle.ORDERS[0].index(name) for name in order]
    lower, upper = face(left[indices], right[indices], None, None, None)
    actual = [[float(value.eval({})) for value in side] for side in (lower, upper)]
    expected_lower, expected_upper, _ = oracle.face_contributions(left, right)
    np.testing.assert_allclose(actual, [expected_lower[indices], expected_upper[indices]],
                               rtol=0., atol=2.e-16)
