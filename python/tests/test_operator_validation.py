"""Spec 2 (S2-12 / ADC-448): operator-first Program type diagnostics.

When states are tagged with their pops.model.StateSpace (P.state(block, space=U)) and rates/operators
flow from P.call, the Program type-checks the composition: a value over one StateSpace cannot feed an
operator typed for another, a Rate(U) cannot be combined with a State(V), and an L: U -> U cannot
drive a solve over State(V). Untagged (legacy) programs skip the checks, so Spec 1 is unaffected.
Pure Python; skips if pops is not importable.
"""
import sys

try:
    from pops import model
    from pops.ir.expr import Const, Var
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_validation (pops unavailable: %s)" % exc)
    sys.exit(0)


def _model():
    mod = model.Module("ep")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y", "B_z"))
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy, bz = Var("grad_x", "aux"), Var("grad_y", "aux"), Var("B_z", "aux")
    mod.operator(
        name="fields_from_state", signature=(u,) >> fields, kind="field_operator",
        capabilities={"default": True}, expr=rho - 1.0)
    mod.operator(
        name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
        expr={"x": [mx, mx * mx / rho, mx * my / rho],
              "y": [my, mx * my / rho, my * my / rho]})
    electric = mod.operator(
        name="electric", signature=(u, fields) >> model.Rate(u), kind="local_source",
        expr=[Const(0.0), -rho * gx, -rho * gy])
    mod.operator(
        name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
        kind="local_linear_operator",
        expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    mod.rate_operator("explicit_rhs", flux=True, sources=[electric])
    return mod


_OTHER = model.StateSpace("V", ("a", "b", "c"))


def test_well_typed_program_passes():
    m = _model()
    u = m.state_spaces()["U"]
    P = adctime.Program("ok").bind_operators(m)
    u_n = P.state("plasma", space=u)
    fields = P._call("fields_from_state", u_n)
    rate = P._call("explicit_rhs", u_n, fields)
    lin = P._call("lorentz", fields)
    rhs = P.linear_combine("rhs", u_n + P.dt * rate)
    P.solve_local_linear("ustar", operator=P.I - P.dt * lin, rhs=rhs, fields=fields)
    print("OK  a well-typed operator-first program passes")


def test_call_input_space_mismatch():
    m = _model()
    u = m.state_spaces()["U"]
    P = adctime.Program("p").bind_operators(m)
    fields = P._call("fields_from_state", P.state("plasma", space=u))
    wrong = P.state("other", space=_OTHER)
    try:
        P._call("explicit_rhs", wrong, fields)  # explicit_rhs expects state 'U', got 'V'
        raise AssertionError("expected a state-space mismatch error")
    except ValueError as exc:
        assert "expects state 'U'" in str(exc) and "over 'V'" in str(exc), str(exc)
    print("OK  P.call rejects an argument over the wrong StateSpace")


def test_combine_space_mismatch():
    m = _model()
    u = m.state_spaces()["U"]
    P = adctime.Program("p").bind_operators(m)
    u_n = P.state("plasma", space=u)
    rate = P._call("explicit_rhs", u_n, P._call("fields_from_state", u_n))  # Rate(U)
    wrong = P.state("other", space=_OTHER)
    try:
        P.linear_combine("bad", u_n + P.dt * rate + wrong)  # mixes U and V
        raise AssertionError("expected a state-space combination error")
    except ValueError as exc:
        assert "different state spaces" in str(exc), str(exc)
    print("OK  linear_combine rejects mixing two StateSpaces")


def test_solve_local_linear_domain_mismatch():
    m = _model()
    u = m.state_spaces()["U"]
    P = adctime.Program("p").bind_operators(m)
    fields = P._call("fields_from_state", P.state("plasma", space=u))
    lin = P._call("lorentz", fields)  # LocalLinearOperator(U, U)
    rhs_v = P.state("other", space=_OTHER)
    try:
        P.solve_local_linear("bad", operator=P.I - P.dt * lin, rhs=rhs_v, fields=fields)
        raise AssertionError("expected an operator/state domain error")
    except ValueError as exc:
        assert "maps U -> U" in str(exc) and "State over 'V'" in str(exc), str(exc)
    print("OK  solve_local_linear rejects L: U->U on a State(V)")


def test_legacy_untagged_unaffected():
    # No space= tags -> the checks are skipped; a plain Spec-1-style program still builds.
    m = _model()
    P = adctime.Program("legacy").bind_operators(m)
    u = P.state("plasma")
    fields = P._legacy_solve_fields(u)
    r = P._legacy_rhs(state=u, fields=fields, sources=["electric"])
    P.commit("plasma", P.linear_combine("u1", u + P.dt * r))
    P.validate()
    print("OK  untagged (legacy) programs skip the space checks")


def main():
    test_well_typed_program_passes()
    test_call_input_space_mismatch()
    test_combine_space_mismatch()
    test_solve_local_linear_domain_mismatch()
    test_legacy_untagged_unaffected()
    print("OK  test_operator_validation")


if __name__ == "__main__":
    main()
