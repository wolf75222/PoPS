"""Spec 2 (S2-12 / ADC-448): operator-first Program type diagnostics.

When states carry their declared ``StateSpace`` through typed block/state handles and rates/operators
flow from P.call, the Program type-checks the composition: a value over one StateSpace cannot feed an
operator typed for another, a Rate(U) cannot be combined with a State(V), and an L: U -> U cannot
drive a solve over State(V). Every state enters through typed block and declaration handles.
Pure Python; skips if pops is not importable.
"""
import sys

try:
    from pops import model
    from pops.ir.expr import Const
    from pops.physics.facade import Model
    from pops import time as adctime
    from typed_program_support import typed_state
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_validation (pops unavailable: %s)" % exc)
    sys.exit(0)


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


_OTHER = model.StateSpace("V", ("a", "b", "c"))


def _typed(P, block, space, physical=None):
    if physical is not None:
        # All values participating in one operator graph must be declared by the same model owner.
        # Add the deliberately different test space to the facade's canonical Module rather than
        # manufacturing a second owner (which ownership validation correctly rejects first).
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


def test_well_typed_program_passes():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("ok").bind_operators(m)
    u_n = _typed(P, "plasma", u, m)
    fields = P._call("fields_from_state", u_n)
    rate = P._call("explicit_rhs", u_n, fields)
    lin = P._call("lorentz", fields)
    rhs = P.linear_combine(
        "rhs", u_n + P.dt * rate,
        at=adctime.TimePoint(P.clock, step=1),
    )
    P.solve_local_linear("ustar", operator=P.I - P.dt * lin, rhs=rhs, fields=fields)
    print("OK  a well-typed operator-first program passes")


def test_call_input_space_mismatch():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p").bind_operators(m)
    fields = P._call("fields_from_state", _typed(P, "plasma", u, m))
    wrong = _typed(P, "other", _OTHER, m)
    try:
        P._call("explicit_rhs", wrong, fields)  # explicit_rhs expects state 'U', got 'V'
        raise AssertionError("expected a state-space mismatch error")
    except ValueError as exc:
        assert "expects state 'U'" in str(exc) and "over 'V'" in str(exc), str(exc)
    print("OK  P.call rejects an argument over the wrong StateSpace")


def test_call_rejects_same_space_name_with_different_component_order():
    m = _model()
    module = m.module
    declared = module.state_spaces()["U"]
    permuted = model.StateSpace("U", tuple(reversed(declared.components)))
    try:
        module.state_space(permuted.name, permuted.components)
        raise AssertionError("expected a register-once structural state-space mismatch")
    except ValueError as exc:
        assert "components" in str(exc) and "register-once" in str(exc), str(exc)


def test_space_structure_participates_in_program_ir_identity():
    first = adctime.Program("space_identity")
    _typed(first, "plasma", model.StateSpace("U", ("rho", "mx", "my")))
    second = adctime.Program("space_identity")
    _typed(second, "plasma", model.StateSpace("U", ("rho", "my", "mx")))

    assert first._ir_hash() != second._ir_hash()


def test_combine_space_mismatch():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p").bind_operators(m)
    u_n = _typed(P, "plasma", u, m)
    rate = P._call("explicit_rhs", u_n, P._call("fields_from_state", u_n))  # Rate(U)
    wrong = _typed(P, "other", _OTHER, m)
    try:
        P.linear_combine(
            "bad", u_n + P.dt * rate + wrong,
            at=adctime.TimePoint(P.clock, step=1),
        )  # mixes U and V
        raise AssertionError("expected a state-space combination error")
    except ValueError as exc:
        assert "incompatible state spaces" in str(exc), str(exc)
    print("OK  linear_combine rejects mixing two StateSpaces")


def test_solve_local_linear_domain_mismatch():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p").bind_operators(m)
    fields = P._call("fields_from_state", _typed(P, "plasma", u, m))
    lin = P._call("lorentz", fields)  # LocalLinearOperator(U, U)
    rhs_v = _typed(P, "other", _OTHER, m)
    try:
        P.solve_local_linear("bad", operator=P.I - P.dt * lin, rhs=rhs_v)
        raise AssertionError("expected an operator/state domain error")
    except ValueError as exc:
        assert "operator maps StateSpace('U'" in str(exc) \
            and "State over StateSpace('V'" in str(exc), str(exc)
    print("OK  solve_local_linear rejects L: U->U on a State(V)")


def test_declared_state_builds_with_typed_handles():
    m = _model()
    P = adctime.Program("typed").bind_operators(m)
    u = _typed(P, "plasma", m.state_space("U"), m)
    module = m.module
    declared = module.state_spaces()["U"]
    fields = P.solve_fields(u)
    r = P._rhs_legacy(state=u, fields=fields, sources=["electric"])
    endpoint = typed_state(
        P, "plasma", state_name="U", space=declared,
        model=module, state=module.state_handle(declared),
    ).next
    P.commit(endpoint, P.linear_combine("u1", u + P.dt * r, at=endpoint.point))
    P.validate()
    print("OK  typed block/state handles preserve the declared StateSpace")


def main():
    test_well_typed_program_passes()
    test_call_input_space_mismatch()
    test_call_rejects_same_space_name_with_different_component_order()
    test_space_structure_participates_in_program_ir_identity()
    test_combine_space_mismatch()
    test_solve_local_linear_domain_mismatch()
    test_declared_state_builds_with_typed_handles()
    print("OK  test_operator_validation")


if __name__ == "__main__":
    main()
