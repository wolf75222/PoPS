"""Explicit finite DOFs are not grid cells or unnamed vector widths."""
import pytest


def test_rectangular_supports_and_order_are_exact():
    from pops.linalg import FiniteLinearMap, FiniteSupport
    source = FiniteSupport("potential", ("p0", "p1", "p2", "p3"))
    target = FiniteSupport("velocity", tuple("v%d" % i for i in range(8)))
    mapping = FiniteLinearMap(source, target, tuple(tuple(i+j for j in range(4)) for i in range(8)))
    assert len(mapping.apply(source.bind((1, 2, 3, 4)))) == 8
    with pytest.raises(ValueError, match="support"):
        mapping.apply(FiniteSupport("potential", tuple(reversed(source.dofs))).bind((1, 2, 3, 4)))
    with pytest.raises(ValueError, match="square"):
        mapping.solve(target.bind(tuple(range(8))))


def test_coefficients_and_labels_are_frozen_and_invalid_inputs_refused():
    from pops.linalg import FiniteLinearMap, FiniteSupport
    from dataclasses import FrozenInstanceError
    s = FiniteSupport("x", ("x", "y"))
    data = [[1., 0.], [0., 2.]]
    m = FiniteLinearMap(s, s, data)
    data[0][0] = 77.
    assert m.coefficients[0][0] == 1.
    with pytest.raises(FrozenInstanceError):
        m.source = FiniteSupport("other", s.dofs)
    with pytest.raises(ValueError, match="unique"):
        FiniteSupport("bad", ("x", "x"))
    with pytest.raises(ValueError, match="finite"):
        FiniteLinearMap(s, s, ((float("nan"), 0), (0, 1)))
    with pytest.raises(ValueError, match="shape"):
        FiniteLinearMap(s, s, ((1, 0),))
    with pytest.raises(ValueError, match="support"):
        m.apply(FiniteSupport("foreign", s.dofs).bind((1, 2)))
