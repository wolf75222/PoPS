#!/usr/bin/env python3
"""pops.time.Program.step decorator mode (epic ADC-399 / ADC-423).

``@P.step`` records a Program's IR by calling the decorated function ONCE at build time. It is sugar for
an inline builder body: it must produce the same IR (same ``_ir_hash``) as writing the body
directly, and it must NEVER run the function numerically during a step (it runs exactly once, here, to
populate the SSA value list -- the compiled ``.so`` owns the runtime step).

Pure Python IR construction only. If pops is not importable, this test fails.
"""
import pops.model as pm
import pops.time as t
import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)


def _program(name):
    m = pm.Module(name + "_module")
    U = m.state_space("U", ("rho",))
    rhs = m.operator(
        "rhs", signature=(U,) >> pm.Rate(U), kind="local_rate",
        capabilities={"produces_rate": True}, lowering={"flux": False, "sources": []},
        expr=0.0)
    return t.Program(name).bind_operators(m), rhs


def test_decorator_matches_inline_ir():
    """A decorated forward_euler builds the SAME IR as the builder forward_euler (equal _ir_hash)."""
    inline, rhs = _program("fe")
    lt.forward_euler(inline, "plasma", rhs_operator=rhs)

    deco, rhs2 = _program("fe")  # same name: _ir_hash includes it

    @deco.step
    def _build(P):
        lt.forward_euler(P, "plasma", rhs_operator=rhs2)

    assert deco._ir_hash() == inline._ir_hash(), \
        "the @P.step decorator must build IR identical to the inline builder body"


def test_decorator_calls_fn_exactly_once_at_build():
    """fn runs exactly ONCE -- at decoration (build) time -- and never again (no per-step execution)."""
    calls = []
    P, rhs = _program("fe")

    @P.step
    def _build(prog):
        calls.append(prog)
        lt.forward_euler(prog, "plasma", rhs_operator=rhs)

    assert calls == [P], "the build fn must be called exactly once, with the Program, at decoration time"
    # Building the IR again (a second Program) must not re-run the first Program's fn.
    other, rhs2 = _program("fe")
    lt.forward_euler(other, "plasma", rhs_operator=rhs2)
    assert calls == [P], "no further calls happen after the IR is recorded"


def test_decorator_returns_program():
    """Program.step returns the Program so a one-liner P = Program(name).step(build) reads cleanly."""
    P, rhs = _program("rk4")

    def build(P):
        lt.rk4(P, "plasma", rhs_operator=rhs)

    P = P.step(build)
    assert isinstance(P, t.Program) and P.validate() is True
    inline, rhs2 = _program("rk4")
    lt.rk4(inline, "plasma", rhs_operator=rhs2)
    assert P._ir_hash() == inline._ir_hash()


def test_decorator_rejects_non_callable():
    P = t.Program("bad")
    try:
        P.step(42)
    except TypeError as exc:
        assert "callable" in str(exc)
    else:
        raise AssertionError("Program.step must reject a non-callable")


def test_decorator_works_for_a_multistage_body():
    """A non-trivial body (an explicit inline scheme) records identically through the decorator."""
    prog, rhs = _program("custom")

    def build(P):
        U = P._state_value("plasma")
        out = lt.explicit_flow(P, U, 1.0, rhs_operator=rhs, name="step")
        P.commit("plasma", out)

    deco = prog.step(build)
    inline, rhs2 = _program("custom")

    def build_inline(P):
        U = P._state_value("plasma")
        out = lt.explicit_flow(P, U, 1.0, rhs_operator=rhs2, name="step")
        P.commit("plasma", out)

    build_inline(inline)
    assert deco._ir_hash() == inline._ir_hash()


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("PASS test_time_std_decorator (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
