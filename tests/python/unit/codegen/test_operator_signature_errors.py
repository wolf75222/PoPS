#!/usr/bin/env python3
"""ADC-528: precise per-type signature-mismatch diagnostics of the operator-first core.

The operator-first type system rejects a mismatched composition with a CLEAR error that names the
operator, the argument and the expected vs received space -- one case per typed input flavour:

  - State: an argument over the wrong StateSpace fed to a P.call input typed 'U';
  - Rate: a Rate(U) combined with a State(V) (a rate must share its state's space);
  - Field: a 'state' value where a FieldSpace input is expected (wrong value flavour) and a
    wrong-arity call;
  - Bundle: a coupled RateBundle whose block rate does not live over the required StateSpace;
  - LocalLinearOperator: an L: U -> U applied to a State(V) in a local solve.

Pure Python (pops.model + pops.time, no numerics / no _pops); skips if pops is not importable.
"""
import sys

try:
    from pops import model
    from pops.ir.expr import Const
    from pops.physics.facade import Model
    from pops import time as adctime
    from typed_program_support import typed_state
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_signature_errors (pops unavailable: %s)" % exc)
    sys.exit(0)


_OTHER = model.StateSpace("V", ("a", "b", "c"))


def _typed(P, block, space, physical=None):
    if physical is not None:
        module = physical.module
        declared = module.state_spaces().get(space.name)
        if declared is None:
            declared = module.state_space(space.name, space.components)
        return typed_state(
            P, block, space=declared, model=module, state=module.state_handle(declared))
    module = model.Module("%s_%s_model" % (P.name, block))
    declared = module.state_space(space.name, space.components)
    return typed_state(
        P, block, space=declared, model=module, state=module.state_handle(declared))


def _model():
    m = Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def test_state_space_mismatch_message():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p")._bind_operators(m)
    fields = P._call("fields_from_state", _typed(P, "plasma", u, m))
    wrong = _typed(P, "other", _OTHER, m)
    try:
        P._call("explicit_rhs", wrong, fields)
        raise AssertionError("expected a State-space mismatch error")
    except ValueError as exc:
        msg = str(exc)
        assert "explicit_rhs" in msg and "expects state 'U'" in msg and "over 'V'" in msg, msg
    print("OK  State mismatch names the operator, the input space and the received space")


def test_rate_combined_with_wrong_state_message():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p")._bind_operators(m)
    u_n = _typed(P, "plasma", u, m)
    rate = P._call("explicit_rhs", u_n, P._call("fields_from_state", u_n))  # Rate(U)
    wrong = _typed(P, "other", _OTHER, m)
    try:
        P.value(
            "bad", u_n + P.dt * rate + wrong,
            at=adctime.TimePoint(P.clock, step=1),
        )
        raise AssertionError("expected a Rate/State space combination error")
    except ValueError as exc:
        assert "incompatible state spaces" in str(exc), str(exc)
    print("OK  Rate combined with a State over another space is rejected")


def test_field_input_wrong_value_flavour_message():
    # explicit_rhs's second input is a FieldSpace: passing a 'state' value there is a clear error.
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p")._bind_operators(m)
    u_n = _typed(P, "plasma", u, m)
    try:
        P._call("explicit_rhs", u_n, u_n)  # second arg should be a fields value, not a state
        raise AssertionError("expected a Field-input value-flavour error")
    except ValueError as exc:
        msg = str(exc)
        assert "explicit_rhs" in msg and "expects a fields value" in msg, msg
    print("OK  Field input rejects a state value with a precise message")


def test_field_wrong_arity_message():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p")._bind_operators(m)
    u_n = _typed(P, "plasma", u, m)
    try:
        P._call("explicit_rhs", u_n)  # explicit_rhs expects (state, fields); only state given
        raise AssertionError("expected an arity error")
    except ValueError as exc:
        msg = str(exc)
        assert "explicit_rhs" in msg and "argument" in msg, msg
    print("OK  wrong call arity names the operator and its expected arguments")


def test_rate_bundle_block_space_mismatch_message():
    # A coupled RateBundle: require() rejects a block rate that lives over the wrong StateSpace.
    u = model.StateSpace("U", ("rho", "mx", "my"))
    v = model.StateSpace("V", ("a", "b"))
    bundle = model.RateBundle({"electrons": model.Rate(u)})
    # correct: require the electrons rate over U
    assert bundle.require("electrons", u) == model.Rate(u)
    try:
        bundle.require("electrons", v)  # electrons rate is Rate(U), not Rate(V)
        raise AssertionError("expected a RateBundle block-space mismatch")
    except TypeError as exc:
        msg = str(exc)
        assert "electrons" in msg and "live over its block's StateSpace" in msg, msg
    print("OK  RateBundle.require rejects a block rate over the wrong StateSpace")


def test_local_linear_operator_domain_mismatch_message():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p")._bind_operators(m)
    fields = P._call("fields_from_state", _typed(P, "plasma", u, m))
    lin = P._call("lorentz", fields)  # LocalLinearOperator(U, U)
    rhs_v = _typed(P, "other", _OTHER, m)
    try:
        P.solve_local_linear("bad", operator=P.I - P.dt * lin, rhs=rhs_v)
        raise AssertionError("expected an operator/state domain error")
    except ValueError as exc:
        msg = str(exc)
        assert "operator maps StateSpace('U'" in msg and "State over StateSpace('V'" in msg, msg
    print("OK  LocalLinearOperator L: U->U rejects a State over another space")


def main():
    test_state_space_mismatch_message()
    test_rate_combined_with_wrong_state_message()
    test_field_input_wrong_value_flavour_message()
    test_field_wrong_arity_message()
    test_rate_bundle_block_space_mismatch_message()
    test_local_linear_operator_domain_mismatch_message()
    print("OK  test_operator_signature_errors")


if __name__ == "__main__":
    main()
