#!/usr/bin/env python3
"""ADC-529: a board Equation refuses bool() / if equation:.

``lhs == rhs`` on a board node builds an inspectable :class:`pops.ir.expr.Equation`, NOT a truth
value. Using it as a Python condition (``if ddt(U) == R:`` / ``bool(ddt(U) == R)``) is almost always
a mistaken comparison, so both the Equation and the bare board node refuse ``__bool__`` with a clear
error naming the lowering APIs (m.rate / m.solve_field / T.value / T.solve). ``==`` itself still
builds an Equation (the authoring surface is unchanged).

Pure Python (pops.ir only, no numerics / no _pops); skips if pops is not importable.
"""
import sys

try:
    from pops import math as bm
    from pops.ir.expr import Equation
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_equation_not_bool (pops unavailable: %s)" % exc)
    sys.exit(0)


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
