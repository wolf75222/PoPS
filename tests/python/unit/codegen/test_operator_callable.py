"""ADC-560: the callable OperatorHandle facade over P.call (tableau-style Programs).

A canonical :class:`pops.model.OperatorHandle` is now callable inside a Program: ``R(U, f)`` finds
the Program from its ProgramValue arguments and delegates to the INTERNAL ``P._call(name, ...)`` -- the same
lowering the public ``P.call(handle, ...)`` uses. So the callable facade builds the BYTE-IDENTICAL IR
(same ``_ir_hash``) as ``P.call`` / ``T.call``, runs the SAME signature type-checks, raises the SAME
errors, and does ZERO numerics (it only builds IR).

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation); skips cleanly if pops is
unavailable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops.ir.expr import Const
    from pops.model import OperatorHandle
    from pops.physics.facade import Model
    from pops.problem import Case
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_callable (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    m = Model("euler_poisson_lorentz")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    h_src = m.source_term("electric", [Const(0.0), rho * (-gx), rho * (-gy)])
    h_lin = m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, bz],
                                        [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    h_rate = m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m, {"electric": h_src, "lorentz": h_lin, "explicit_rhs": h_rate}


def _operator_handle(model, name):
    """Return the registry-issued handle, preserving its declaring owner."""
    return model.module.operator_handle(name)


def _program_state(model, name):
    """Create one typed block instance and bind its declared state to ``Program``."""
    module = model.module
    case = Case(name="%s-case" % name)
    block = case.block("plasma", model)
    state = module.state_handle(module.state_spaces()["U"])
    program = adctime.Program(name)._bind_operators(module)
    return program, program.state(block, state)


def test_callable_handle_ir_byte_identical_to_pcall():
    """A predictor written with the callable facade produces the byte-identical IR to T.call."""
    m, h = build_model()

    def via_pcall():
        P, state = _program_state(m, "prog")
        U = state.n
        f = P.call(_operator_handle(m, "fields_from_state"), U)
        R = P.call(h["explicit_rhs"], U, f)
        P.commit(state.next, P.value("u1", U + P.dt * R, at=state.next.point))
        return P

    def via_callable():
        P, state = _program_state(m, "prog")
        U = state.n
        f = _operator_handle(m, "fields_from_state")(U)  # callable facade
        R = h["explicit_rhs"](U, f)          # callable facade
        P.commit(state.next, P.value("u1", U + P.dt * R, at=state.next.point))
        return P

    a, b = via_pcall(), via_callable()
    a.validate()
    b.validate()
    assert a._ir_hash() == b._ir_hash(), (
        "the callable handle facade must lower to the byte-identical IR as P.call\n"
        "  P.call   : %s\n  callable : %s" % (a._ir_hash(), b._ir_hash()))
    print("OK  R(U, f) IR hash == P.call(R, U, f) IR hash: %s" % a._ir_hash())


def test_callable_facade_across_all_kinds():
    """rate / source / linear all match the P.call IR through the callable facade."""
    m, h = build_model()

    def src(via_callable):
        P, state = _program_state(m, "p")
        U = state.n
        fields = _operator_handle(m, "fields_from_state")
        f = fields(U) if via_callable else P.call(fields, U)
        s = h["electric"](U, f) if via_callable else P.call(h["electric"], U, f)
        P.commit(state.next, P.value("u1", U + P.dt * s, at=state.next.point))
        return P

    def lin(via_callable):
        P, state = _program_state(m, "p")
        U = state.n
        fields = _operator_handle(m, "fields_from_state")
        rhs = P.value("rhs", U, at=state.next.point)
        f = fields(rhs) if via_callable else P.call(fields, rhs)
        L = h["lorentz"](f) if via_callable else P.call(h["lorentz"], f)
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=rhs, fields=f)
        P.commit(state.next, U1)
        return P

    assert src(True)._ir_hash() == src(False)._ir_hash()
    assert lin(True)._ir_hash() == lin(False)._ir_hash()
    print("OK  callable facade matches P.call IR for source and linear operators")


def test_callable_facade_same_signature_errors():
    """Signature errors are identical between the callable facade and P.call (same _call path)."""
    m, h = build_model()
    P, state = _program_state(m, "p")
    U = state.n
    f = _operator_handle(m, "fields_from_state")(U)

    # arity: electric needs (state, fields)
    with pytest.raises(ValueError, match="expects 2 argument"):
        h["electric"](U)
    with pytest.raises(ValueError, match="expects 2 argument"):
        P.call(h["electric"], U)
    # vtype: a fields where a state is expected
    with pytest.raises(ValueError, match="expects a state value"):
        h["electric"](f, f)
    with pytest.raises(ValueError, match="expects a state value"):
        P.call(h["electric"], f, f)
    print("OK  callable facade raises the same signature errors as P.call")


def test_call_outside_program_refused():
    """A callable handle with no Program value cannot find a Program to build into -> clear error."""
    from pops.model import OwnerPath
    h_rate = OperatorHandle(
        "explicit_rhs", kind="local_rate", owner=OwnerPath.descriptor("outside-program"))
    with pytest.raises(ValueError, match="must be called with time-Program values"):
        h_rate("not a value", 3)
    print("OK  calling a handle outside a Program is refused with a clear message")


def test_callable_does_no_numerics():
    """__call__ only builds IR: the result is an IR ProgramValue, never a numpy array."""
    m, h = build_model()
    _, state = _program_state(m, "p")
    U = state.n
    f = _operator_handle(m, "fields_from_state")(U)
    R = h["explicit_rhs"](U, f)
    from pops.time.values import ProgramValue
    assert isinstance(R, ProgramValue) and R.vtype == "rhs"
    assert type(R).__module__.startswith("pops."), (
        "the result must be an IR ProgramValue, not numeric data")
    assert not (hasattr(R, "shape") and hasattr(R, "dtype")), "no ndarray may leak from __call__"
    print("OK  the callable facade builds an IR ProgramValue and does no numerics")


def main():
    test_callable_handle_ir_byte_identical_to_pcall()
    test_callable_facade_across_all_kinds()
    test_callable_facade_same_signature_errors()
    test_call_outside_program_refused()
    test_callable_does_no_numerics()
    print("OK  test_operator_callable")


if __name__ == "__main__":
    main()
