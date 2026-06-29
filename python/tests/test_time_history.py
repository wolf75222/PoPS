#!/usr/bin/env python3
"""pops.time history and Adams-Bashforth macros.

The history ring is part of the canonical ``Program`` IR: a Program may store a
state/rate value under a stable history name, read previous lags, and emit the
runtime ``ctx.history`` / ``ctx.store_history`` / ``ctx.rotate_histories`` calls.

This file intentionally stays at the IR/codegen layer. Runtime parity belongs in
the clean combined-problem installation tests, not in a time-macro unit test.
"""

from pops.ir.expr import Const
from pops import model
from pops import time as t
import pops.lib.time as lt


_C = 0.75


def _rate_module(name, ncomp=1):
    m = model.Module(name + "_module")
    U = m.state_space("U", tuple("q%d" % i for i in range(ncomp)))
    rhs = m.operator(
        "rhs",
        signature=(U,) >> model.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(_C) for _ in range(ncomp)],
    )
    return m, rhs


def _program(name, module):
    return t.Program(name).bind_operators(module)


def _state(P, block="blk"):
    return P.state("U", block=block).n


def _rate(P, rhs, U, name="R"):
    return P.call(rhs, U, name=name)


def test_history_builds_state_value():
    m, rhs = _rate_module("history")
    P = _program("p", m)
    U = _state(P)
    R = _rate(P, rhs, U)
    P.store_history("blk.R", R)
    Rp = P.history("blk.R", lag=1)
    assert Rp.vtype == "state"
    assert Rp.is_field()
    P.commit("blk", P.linear_combine("step", U + P.dt * (R - Rp)))
    assert P.validate() is True


def test_store_history_requires_a_field():
    P = t.Program("p")
    for bad in (5, "x", None):
        try:
            P.store_history("blk.R", bad)
        except ValueError as exc:
            assert "field" in str(exc)
        else:
            raise AssertionError("store_history must reject %r" % (bad,))


def test_history_lag_must_be_positive_int():
    P = t.Program("p")
    for bad in (0, -1, 1.0, True):
        try:
            P.history("blk.R", lag=bad)
        except ValueError as exc:
            assert "lag" in str(exc)
        else:
            raise AssertionError("history lag=%r must raise" % (bad,))


def test_ab2_macro_lowers_from_typed_rate_operator():
    m, rhs = _rate_module("ab2")
    P = _program("ab2", m)
    lt.adams_bashforth2(P, "plasma", rhs_operator=rhs)
    assert P.validate() is True
    assert all(v.op != "rhs" for v in P._values), "AB2 must use call nodes, not Program rhs"

    src = P.emit_cpp_program(model=m)
    for frag in (
        'ctx.history("plasma.R", 1)',
        'ctx.store_history("plasma.R"',
        "ctx.rotate_histories();",
        "GeneratedModule::Operators::",
    ):
        assert frag in src, "AB2 codegen must contain %r\n%s" % (frag, src)
    assert "1.5 * dt" in src and "-0.5 * dt" in src


def test_store_before_read_in_body():
    """The store is emitted before the lag-1 read; the first runtime store cold-starts the ring."""
    m, rhs = _rate_module("ab2_order")
    P = _program("ab2", m)
    lt.adams_bashforth2(P, "plasma", rhs_operator=rhs)
    src = P.emit_cpp_program(model=m)
    body = src[src.index("ctx.install"):]
    read = body.index('= ctx.history("plasma.R", 1);')
    assert body.index("ctx.store_history") < read
    assert read < body.index("ctx.rotate_histories")


def test_non_history_schemes_emit_no_rotate():
    for sched in ("forward_euler", "ssprk2", "ssprk3", "rk4"):
        m, rhs = _rate_module(sched)
        P = _program(sched, m)
        getattr(lt, sched)(P, "blk", rhs_operator=rhs)
        src = P.emit_cpp_program(model=m)
        assert "ctx.rotate_histories" not in src, "%s must not rotate" % sched
        assert "ctx.history(" not in src, "%s must not read a history" % sched
        assert "GeneratedModule::Operators::" in src


def _hist_program(name, lag):
    m, rhs = _rate_module("hist")
    P = _program("h", m)
    U = _state(P)
    R = _rate(P, rhs, U)
    P.store_history(name, R)
    Rp = P.history(name, lag=lag)
    P.commit("blk", P.linear_combine("step", U + P.dt * (R - Rp)))
    return P


def test_ir_hash_distinguishes_name_and_lag():
    h_a1 = _hist_program("a.R", 1)._ir_hash()
    h_b1 = _hist_program("b.R", 1)._ir_hash()
    h_a2 = _hist_program("a.R", 2)._ir_hash()
    assert h_a1 != h_b1
    assert h_a1 != h_a2


def test_absent_history_program_lowers():
    """A read of a never-stored history lowers; the runtime owns the uninitialized-history rejection."""
    m, rhs = _rate_module("missing")
    P = _program("miss", m)
    U = _state(P)
    Rp = P.history("missing.R", lag=1)
    R = _rate(P, rhs, U)
    P.commit("blk", P.linear_combine("step", U + P.dt * (R - Rp)))
    assert P.validate() is True
    src = P.emit_cpp_program(model=m)
    assert 'ctx.history("missing.R", 1)' in src
    assert "ctx.store_history" not in src
    assert "GeneratedModule::Operators::" in src


def test_ab3_lowers_with_two_history_lags():
    m, rhs = _rate_module("ab3")
    P = _program("ab3", m)
    lt.adams_bashforth(P, "plasma", 3, rhs_operator=rhs)
    src = P.emit_cpp_program(model=m)
    assert 'ctx.history("plasma.R", 1)' in src
    assert 'ctx.history("plasma.R", 2)' in src
    assert "ctx.store_history" in src
    assert "ctx.rotate_histories();" in src
    for w in ("1.9166", "-1.3333", "0.41666"):
        assert w in src.replace("e+00", ""), "AB3 weight %s missing" % w


def test_adams_bashforth_rejects_strings():
    m, _ = _rate_module("bad_selector")
    P = _program("bad_selector", m)
    try:
        lt.adams_bashforth2(P, "plasma", rhs_operator="rhs")
    except TypeError as exc:
        assert "typed operator handles" in str(exc)
    else:
        raise AssertionError("AB2 must reject string operator selectors")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("PASS test_time_history")


if __name__ == "__main__":
    main()
