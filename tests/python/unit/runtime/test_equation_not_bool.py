#!/usr/bin/env python3
"""ADC-529: a board Equation refuses bool() / if equation:.

``lhs == rhs`` on a board node builds an inspectable :class:`pops._ir.expr.Equation`, NOT a truth
value. Using it as a Python condition (``if ddt(U) == R:`` / ``bool(ddt(U) == R)``) is almost always
a mistaken comparison, so both the Equation and the bare board node refuse ``__bool__`` with a
sourced symbolic-truth diagnostic. ``==`` itself still builds an Equation.

Pure Python (pops._ir only, no numerics / no _pops); skips if pops is not importable.
"""
from tests.python.support.requirements import require_native_or_skip

try:
    from pops import math as bm
    from pops.math import Equation
except Exception as exc:  # pops not importable here -> skip, never fake
    require_native_or_skip('test_equation_not_bool (pops unavailable: %s)' % exc)


def test_equal_on_board_node_builds_an_equation():
    eq = bm.ddt("U") == bm.unknown("R")
    assert isinstance(eq, Equation), type(eq)
    assert eq.lhs is not None and eq.rhs is not None
    print("OK  '==' on a board node still builds an inspectable Equation")


def test_bool_of_equation_raises():
    eq = bm.ddt("U") == bm.unknown("R")
    try:
        bool(eq)
        raise AssertionError("bool(Equation) must raise")
    except TypeError as exc:
        msg = str(exc)
        assert "[symbolic_truth_value]" in msg, msg
        assert "Equation has no Python truth value" in msg, msg
    print("OK  bool(Equation) raises a sourced symbolic-truth diagnostic")


def test_if_equation_raises():
    eq = -bm.laplacian(bm.unknown("phi")) == bm.unknown("rhs")
    try:
        if eq:  # noqa: SIM102 -- deliberately exercising the refusal
            pass
        raise AssertionError("if equation: must raise")
    except TypeError as exc:
        assert "[symbolic_truth_value]" in str(exc), str(exc)
    print("OK  'if equation:' raises")


def test_bare_board_node_bool_raises():
    # A board node itself (before ==) must also refuse truthiness, catching e.g. a stray 'if ddt(U):'.
    node = bm.ddt("U")
    try:
        bool(node)
        raise AssertionError("bool(board node) must raise")
    except TypeError as exc:
        assert "[symbolic_truth_value]" in str(exc), str(exc)
    print("OK  a bare board node refuses bool()")


def main():
    test_equal_on_board_node_builds_an_equation()
    test_bool_of_equation_raises()
    test_if_equation_raises()
    test_bare_board_node_bool_raises()
    print("OK  test_equation_not_bool")


if __name__ == "__main__":
    main()
