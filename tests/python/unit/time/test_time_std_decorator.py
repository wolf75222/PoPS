#!/usr/bin/env python3
"""pops.time.Program.step decorator mode (epic ADC-399 / ADC-423).

``@P.step`` records a Program's IR by calling the decorated function ONCE at build time. It is sugar for
an inline builder body: it must produce byte-identical IR (same ``_ir_hash``) to writing the body
directly, and it must NEVER run the function numerically during a step (it runs exactly once, here, to
populate the SSA value list -- the compiled ``.so`` owns the runtime step).

Pure Python (IR construction only); skips cleanly if pops.time is unavailable, never fakes.
"""
from typed_program_support import typed_state

import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_std_decorator (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def test_decorator_matches_inline_ir(t):
    """A decorated builder builds the same IR as the identical inline body."""
    def build(program):
        state = typed_state(program, "plasma", state_name="U")
        program.commit(
            state.next,
            program.value("copy", 1 * state.n, at=state.next.point),
        )

    inline = t.Program("fe")
    build(inline)

    deco = t.Program("fe")  # same name: _ir_hash includes it

    @deco.step
    def _build(P):
        build(P)

    assert deco._ir_hash() == inline._ir_hash(), \
        "the @P.step decorator must build IR identical to the inline builder body"


def test_decorator_calls_fn_exactly_once_at_build(t):
    """fn runs exactly ONCE -- at decoration (build) time -- and never again (no per-step execution)."""
    calls = []
    P = t.Program("fe")

    @P.step
    def _build(prog):
        calls.append(prog)
        state = typed_state(prog, "plasma", state_name="U")
        prog.commit(state.next, prog.value("copy", state.n, at=state.next.point))

    assert calls == [P], "the build fn must be called exactly once, with the Program, at decoration time"
    # Building the IR again (a second Program) must not re-run the first Program's fn.
    other = t.Program("fe")
    state = typed_state(other, "plasma", state_name="U")
    other.commit(state.next, other.value("copy", state.n, at=state.next.point))
    assert calls == [P], "no further calls happen after the IR is recorded"


def test_decorator_returns_program(t):
    """Program.step returns the Program so a one-liner P = Program(name).step(build) reads cleanly."""
    def build(P):
        state = typed_state(P, "plasma", state_name="U")
        P.commit(state.next, P.value("copy", state.n, at=state.next.point))
    P = t.Program("rk4").step(build)
    assert isinstance(P, t.Program) and P.validate() is True
    inline = t.Program("rk4")
    build(inline)
    assert P._ir_hash() == inline._ir_hash()


def test_decorator_rejects_non_callable(t):
    P = t.Program("bad")
    try:
        P.step(42)
    except TypeError as exc:
        assert "callable" in str(exc)
    else:
        raise AssertionError("Program.step must reject a non-callable")


def test_decorator_works_for_a_multistage_body(t):
    """A non-trivial body (an explicit inline scheme) records identically through the decorator."""
    def build(P):
        state = typed_state(P, "plasma", state_name="U")
        stage = P.value("stage", 0.5 * state.n)
        P.commit(
            state.next,
            P.value("step", state.n + stage, at=state.next.point),
        )
    deco = t.Program("custom").step(build)
    inline = t.Program("custom")
    build(inline)
    assert deco._ir_hash() == inline._ir_hash()


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_std_decorator (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
