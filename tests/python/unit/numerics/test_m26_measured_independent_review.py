"""Nonuniform-measure counterexamples; IR inspection/math, not native execution."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pops._ir.expr import Add, Const, Mul, Sub
from pops._ir.finite_linear import FiniteProjection
from pops.linalg import FiniteMeasure, FiniteSupport, FiniteSymmetricInteraction


M = np.array((.125, .5, 2., 1.25))
W = np.array(((2., -1., .5, .25), (-1., 3., 1., -.75),
              (.5, 1., -2., .5), (.25, -.75, .5, 1.)))
X = np.array((.4, 1.2, .8, 1.7))
Y = np.array((1.4, .2, 1.1, .9))


def literal_expression_value(node):
    """Read constant authoring IR only; this is not a Program execution backend."""
    if isinstance(node, Const):
        return node.eval({})
    if isinstance(node, FiniteProjection):
        call = node.application
        assert call.operation == "apply"
        values = np.array([literal_expression_value(x) for x in call.inputs])
        return (np.array(call.coefficients) @ values)[node.index]
    if isinstance(node, (Add, Sub, Mul)):
        a, b = literal_expression_value(node.a), literal_expression_value(node.b)
        return a+b if isinstance(node, Add) else a-b if isinstance(node, Sub) else a*b
    raise AssertionError(type(node))


@pytest.mark.parametrize("order", ((0, 1, 2, 3), (2, 0, 3, 1)))
@pytest.mark.parametrize("variation", (False, True))
def test_nonuniform_action_adjoint_energy_variation_and_joint_permutation(order, variation):
    x, y = (X+.15*Y, Y+.2*X) if variation else (X, Y)
    support = FiniteSupport("measured", tuple(f"q{i}" for i in order))
    measure = FiniteMeasure(support, tuple(float(M[i]) for i in order))
    interaction = FiniteSymmetricInteraction(measure,
        tuple(tuple(float(W[i, j]) for j in order) for i in order))
    left, right = support.bind(x[list(order)]), support.bind(y[list(order)])
    matrix = np.array(interaction.map.coefficients)
    expected = W @ np.diag(M)
    np.testing.assert_array_equal(matrix, expected[np.ix_(order, order)])
    action = np.array([literal_expression_value(v) for v in interaction.apply(left)])
    adjoint = np.array([literal_expression_value(v) for v in interaction.adjoint(right)])
    np.testing.assert_allclose(action, (expected@x)[list(order)], atol=2.e-15)
    np.testing.assert_allclose(adjoint, (expected@y)[list(order)], atol=2.e-15)
    np.testing.assert_allclose(action[np.argsort(order)], expected@x, rtol=0, atol=2.e-15)
    np.testing.assert_allclose(adjoint[np.argsort(order)], expected@y, rtol=0, atol=2.e-15)
    # The Euclidean adjoint and row-weighted map are wrong for this measure.
    assert np.max(abs(expected.T@y-expected@y)) > .5
    assert abs(x@np.diag(M)@expected.T@y-(expected@x)@np.diag(M)@y) > .1
    pairing = literal_expression_value(measure.pair(left, interaction.adjoint(right)))
    assert pairing == pytest.approx((expected@x)@np.diag(M)@y, abs=3.e-15)
    energy = .5*x@np.diag(M)@W@np.diag(M)@x
    assert literal_expression_value(interaction.energy(left)) == pytest.approx(energy, abs=3.e-15)
    derivative = y@np.diag(M)@W@np.diag(M)@x
    assert literal_expression_value(interaction.directional_derivative(left, right)) == pytest.approx(derivative, abs=3.e-15)
    def independent_energy(value):
        return .5*value@np.diag(M)@W@np.diag(M)@value
    h = .125
    assert (independent_energy(x+h*y)-independent_energy(x-h*y))/(2*h) == pytest.approx(derivative, abs=2.e-14)
    assert independent_energy(x+y)-energy == pytest.approx(derivative+independent_energy(y), abs=2.e-14)


@pytest.mark.parametrize("bad", (0., -1., float("nan"), float("inf"),
                                -float("inf"), True))
def test_invalid_measure_is_not_silently_normalized(bad):
    support = FiniteSupport("measured", ("a", "b"))
    with pytest.raises(ValueError, match="positive finite weight"):
        FiniteMeasure(support, (1., bad))


@pytest.mark.parametrize("kernel,message", (
    (((1.,),), "support"),
    (((1., 2.), (2.,)), "support"),
    (((1., float("nan")), (float("nan"), 1.)), "finite numeric"),
    (((1., float("inf")), (float("inf"), 1.)), "finite numeric"),
    (((1., -float("inf")), (-float("inf"), 1.)), "finite numeric"),
    (((1., True), (True, 1.)), "finite numeric"),
    (((1., "2"), ("2", 1.)), "finite numeric"),
    (((1., 2.), (3., 1.)), "exactly symmetric"),
))
def test_invalid_kernel_rejected_before_action(kernel, message, monkeypatch):
    from pops.linalg import FiniteLinearMap
    def forbidden_action(*args, **kwargs):
        pytest.fail("invalid kernel reached finite action")
    monkeypatch.setattr(FiniteLinearMap, "apply", forbidden_action)
    measure = FiniteMeasure(FiniteSupport("measured", ("a", "b")), (1., 3.))
    with pytest.raises(ValueError, match=message):
        FiniteSymmetricInteraction(measure, kernel).apply(measure.support.bind((1., 2.)))


def test_zero_kernel_and_nonuniform_measure_are_admissible():
    support = FiniteSupport("zero_kernel", ("a", "b"))
    interaction = FiniteSymmetricInteraction(FiniteMeasure(support, (2., .125)),
                                            ((0., 0.), (0., 0.)))
    density = support.bind((2., -3.))
    np.testing.assert_array_equal([literal_expression_value(x)
                                  for x in interaction.apply(density)], (0., 0.))
    assert literal_expression_value(interaction.energy(density)) == 0.
    assert literal_expression_value(interaction.directional_derivative(
        density, support.bind((7., 4.)))) == 0.


def test_overflowed_measured_matrix_is_rejected_before_action(monkeypatch):
    from pops.linalg import FiniteLinearMap
    def forbidden_action(*args, **kwargs):
        pytest.fail("overflowed measured matrix reached finite action")
    monkeypatch.setattr(FiniteLinearMap, "apply", forbidden_action)
    support = FiniteSupport("overflow", ("a", "b"))
    interaction = FiniteSymmetricInteraction(FiniteMeasure(support, (2., 3.)),
                                            ((1e308, 0.), (0., 1.)))
    # W and m are individually finite; their product is not admissible for
    # FiniteLinearMap, so no application IR or Program mutation is produced.
    with pytest.raises(ValueError, match="finite numeric constants"):
        interaction.apply(support.bind((1., 2.)))


def test_kernel_symmetry_is_exact_and_inputs_are_immutable():
    support = FiniteSupport("measured", ("a", "b"))
    weights, kernel = [1., 3.], [[1., .5], [.5, -2.]]
    measure = FiniteMeasure(support, weights)
    interaction = FiniteSymmetricInteraction(measure, kernel)
    weights[0], kernel[0][1] = 9., 9.
    assert measure.weights == (1., 3.)
    assert interaction.kernel[0][1] == .5
    with pytest.raises(FrozenInstanceError):
        interaction.kernel = ((0., 0.), (0., 0.))
    with pytest.raises(ValueError, match="exactly symmetric"):
        FiniteSymmetricInteraction(measure, ((1., .5), (float(np.nextafter(.5, 1.)), 1.)))
    with pytest.raises(ValueError, match="ordered support"):
        interaction.apply(FiniteSupport("measured", ("b", "a")).bind((1., 2.)))


@pytest.mark.parametrize("permuted", (False, True))
@pytest.mark.parametrize("variation", (False, True))
def test_initial_arrays_keep_physical_label_provenance(permuted, variation):
    from tests.python.unit.numerics.test_api040_m26_finite_interaction import _modules
    witness, _oracle = _modules()
    _case, _layout, subjects, order = witness.build_case(permuted=permuted)
    values = witness._initial_values(subjects, order, variation=variation)
    theta = 2*np.pi*(np.arange(12)+.5)/12
    if variation:
        expected_first = .9 + .15*np.cos(theta) - .03*np.sin(3*theta)
        expected_second = 1.2 - .11*np.sin(theta) + .04*np.cos(2*theta)
    else:
        expected_first = 1 + .2*np.cos(theta) + .1*np.sin(2*theta)
        expected_second = .85 + .12*np.sin(theta) + .07*np.cos(3*theta)
    for name, expected in (("density_first", expected_first),
                           ("density_second", expected_second)):
        authored = values[subjects[name]]
        assert authored.shape == (12, 1)
        assert authored.flags.c_contiguous
        np.testing.assert_allclose(authored[:, 0][np.argsort(order)], expected,
                                   rtol=0, atol=3e-16)
    assert values[subjects["measured_pairing"]].shape == (7, 1)
    assert not np.shares_memory(values[subjects["density_first"]],
                               values[subjects["density_second"]])


@pytest.mark.parametrize("bad", (0., float("nan"), float("inf")))
def test_invalid_witness_density_refused_before_binding(monkeypatch, bad):
    from tests.python.unit.numerics.test_api040_m26_finite_interaction import _modules
    witness, _oracle = _modules()
    _case, _layout, subjects, order = witness.build_case()
    first, second = np.ones(12), np.ones(12)
    first[3] = bad
    monkeypatch.setattr(witness, "density_pair", lambda **unused: (first, second))
    with pytest.raises(ValueError, match="finite positive densities"):
        witness._initial_values(subjects, order, variation=False)
