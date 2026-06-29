#!/usr/bin/env python3
"""pops.lib.time IMEX / Lie / Adams-Bashforth macros.

These ready-made schemes must build canonical ``pops.time.Program`` IR from typed
operator handles. They are unit tests for the Program that the macros produce.
Runtime parity belongs in the clean ``compile_problem -> System.install`` integration
tests.
"""

from pops.ir.expr import Const
from pops import model
from pops import time as t
import pops.lib.time as lt


def _rate_module(name, ncomp=1):
    """Small module with one explicit rate operator and one local-linear operator."""
    m = model.Module(name + "_module")
    U = m.state_space("U", tuple("q%d" % i for i in range(ncomp)))
    rhs = m.operator(
        "rhs", signature=(U,) >> model.Rate(U), kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(0.0) for _ in range(ncomp)])
    source = m.operator(
        "source", signature=(U,) >> model.Rate(U), kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(0.0) for _ in range(ncomp)])
    linear = m.operator(
        "linear", signature=() >> model.LocalLinearOperator(U, U),
        kind="local_linear_operator",
        capabilities={"linear": True, "solve_i_minus_a": True},
        expr=[[Const(0.0) for _ in range(ncomp)] for _ in range(ncomp)])
    return m, rhs, source, linear


def _program(name, module):
    return t.Program(name).bind_operators(module)


def _work_ops(P):
    return [v.op for v in P._values if v.op not in ("state",)]


def test_imex_local_builds_and_lowers():
    m, rhs, _, linear = _rate_module("imex", ncomp=3)
    P = _program("imex", m)
    out = lt.imex_local(P, "plasma", explicit_operator=rhs, implicit_operator=linear)
    assert P.validate() is True and P.commits()["plasma"] is out
    assert all(v.op != "rhs" for v in P._values), "imex_local must use call nodes, not Program rhs"
    src = P.emit_cpp_program(model=m)
    assert "GeneratedModule::Operators::" in src
    assert "pops::detail::mat_inverse<3>(" in src


def test_imex_local_theta_guard_and_string_reject():
    m, rhs, _, linear = _rate_module("imex_guard", ncomp=3)
    for bad in (0.0, -0.5, 1.5):
        try:
            lt.imex_local(_program("bad", m), "plasma",
                          explicit_operator=rhs, implicit_operator=linear, theta=bad)
        except ValueError as exc:
            assert "theta" in str(exc)
        else:
            raise AssertionError("imex_local theta=%r must raise" % (bad,))
    try:
        lt.imex_local(_program("string", m), "plasma",
                      explicit_operator="rhs", implicit_operator=linear)
    except TypeError as exc:
        assert "typed operator handles" in str(exc)
    else:
        raise AssertionError("imex_local must reject string operator selectors")


def test_lie_chains_two_operator_first_subflows():
    m, rhs, source, _ = _rate_module("lie")
    P = _program("lie", m)

    def flow(prog, U, frac):
        return lt.explicit_flow(prog, U, frac, rhs_operator=rhs)

    def source_flow(prog, U, frac):
        return lt.explicit_flow(prog, U, frac, rhs_operator=source)

    out = lt.lie(P, "plasma", flow, source_flow)
    P.validate()
    assert P.commits()["plasma"] is out
    assert _work_ops(P).count("call") == 2
    assert _work_ops(P).count("linear_combine") == 2

    strang = _program("lie", m)
    lt.strang(strang, "plasma", flow, source_flow)
    assert P._ir_hash() != strang._ir_hash(), "Lie has two stages; Strang has three"


def test_adams_bashforth_orders_build_from_typed_rate():
    m, rhs, _, _ = _rate_module("ab")
    for order in (1, 2, 3):
        P = _program("ab%d" % order, m)
        lt.adams_bashforth(P, "plasma", order, rhs_operator=rhs)
        assert P.validate() is True, "AB%d must validate" % order
        assert all(v.op != "rhs" for v in P._values), "AB%d must use call nodes" % order

    ab1 = _program("p", m)
    lt.adams_bashforth(ab1, "plasma", 1, rhs_operator=rhs)
    fe = _program("p", m)
    lt.forward_euler(fe, "plasma", rhs_operator=rhs)
    assert ab1._ir_hash() == fe._ir_hash(), "AB1 is Forward Euler"

    a2 = _program("p", m)
    lt.adams_bashforth2(a2, "plasma", rhs_operator=rhs)
    g2 = _program("p", m)
    lt.adams_bashforth(g2, "plasma", 2, rhs_operator=rhs)
    assert a2._ir_hash() == g2._ir_hash(), "adams_bashforth2 aliases order 2"


def test_adams_bashforth_bad_order():
    m, rhs, _, _ = _rate_module("ab_bad")
    for bad in (0, 4, 2.0):
        try:
            lt.adams_bashforth(_program("x", m), "plasma", bad, rhs_operator=rhs)
        except ValueError as exc:
            assert "order" in str(exc)
        else:
            raise AssertionError("AB order=%r must raise" % (bad,))


def test_ab3_lowers_with_two_history_reads():
    m, rhs, _, _ = _rate_module("ab3")
    P = _program("ab3", m)
    lt.adams_bashforth(P, "plasma", 3, rhs_operator=rhs)
    src = P.emit_cpp_program(model=m)
    assert 'ctx.history("plasma.R", 1)' in src
    assert 'ctx.history("plasma.R", 2)' in src
    assert "ctx.store_history" in src
    assert "ctx.rotate_histories();" in src
    for w in ("1.9166", "-1.3333", "0.41666"):
        assert w in src.replace("e+00", ""), "AB3 weight %s missing" % w


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("PASS test_time_std_imex_lie_ab")


if __name__ == "__main__":
    main()
