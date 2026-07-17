"""pops.time.Program -- compiled time-program IR, builder mode (ADC-401, Phase 2a).

Pure Python; no compilation. The Program is a typed intermediate representation built by Python and
later lowered to C++ (other ADC-399 phases). This test exercises ONLY the IR construction layer:

  - states / solve_fields / rhs / linear_combine / commit produce typed IR values;
  - the affine algebra of linear_combine (U + dt*R, RK coefficients) records the right
    per-input coefficient polynomials in dt;
  - structural validation: each block committed at most once, at least one block committed;
  - IR values cannot be used as a Python bool (runtime values are not host booleans);
  - the IR hash is deterministic and sensitive to a coefficient change.

It does NOT compile or run anything (no ProgramContext, no .so) -- that is Phase 2b/2c.
Run with python3 (PYTHONPATH = built pops package).
"""
from typed_program_support import commits_by_block, solve_field, typed_state

from pops import time as adctime
from pops.numerics.terms import DefaultSource, Flux


def _coeff(node, value):
    """Coefficient polynomial (dict power->float) attached to `value` in a linear_combine node."""
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


def test_forward_euler_ir():
    P = adctime.Program("forward_euler")
    dt = P.dt
    U = typed_state(P, "plasma")
    fields = solve_field(P, U)
    R = P.rhs(state=U, fields=fields, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U").next
    U1 = P.value("U1", U + dt * R, at=endpoint.point)
    P.commit(endpoint, U1)
    P.validate()
    assert U.vtype == "state" and R.vtype == "rhs" and fields.vtype == "fields"
    assert U1.vtype == "state" and U1.op == "linear_combine"
    assert {value.id for value in U1.inputs} == {U.id, R.id}
    assert _coeff(U1, U) == {0: 1.0}
    assert _coeff(U1, R) == {1: 1.0}
    assert commits_by_block(P)["plasma"] is U1
    print("OK  1. forward euler IR (U + dt*R) -> commit, coeffs {0:1},{1:1}")


def test_ssprk2_ir():
    P = adctime.Program("ssprk2")
    dt = P.dt
    U0 = typed_state(P, "plasma")
    f0 = solve_field(P, U0, name="f0")
    k0 = P.rhs("k0", state=U0, fields=f0, terms=[Flux(), DefaultSource()])
    U1 = P.value("U1", U0 + dt * k0, at=adctime.TimePoint(P.clock, 1))
    f1 = solve_field(P, U1, name="f1")
    k1 = P.rhs("k1", state=U1, fields=f1, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U").next
    U2 = P.value(
        "U2", 0.5 * U0 + 0.5 * (U1 + dt * k1), at=endpoint.point)
    P.commit(endpoint, U2)
    P.validate()
    assert _coeff(U2, U0) == {0: 0.5}
    assert _coeff(U2, U1) == {0: 0.5}
    assert _coeff(U2, k1) == {1: 0.5}
    print("OK  2. ssprk2 IR coefficients (0.5 U0 + 0.5 U1 + 0.5 dt k1)")


def test_rk4_ir():
    P = adctime.Program("rk4")
    dt = P.dt
    U0 = typed_state(P, "plasma")
    k1 = P.rhs("k1", state=U0, fields=solve_field(P, U0), terms=[Flux(), DefaultSource()])
    U1 = P.value("U1", U0 + 0.5 * dt * k1, at=adctime.TimePoint(P.clock, 0.5))
    k2 = P.rhs("k2", state=U1, fields=solve_field(P, U1), terms=[Flux(), DefaultSource()])
    U2 = P.value("U2", U0 + 0.5 * dt * k2, at=adctime.TimePoint(P.clock, 0.5))
    k3 = P.rhs("k3", state=U2, fields=solve_field(P, U2), terms=[Flux(), DefaultSource()])
    U3 = P.value("U3", U0 + dt * k3, at=adctime.TimePoint(P.clock, 1))
    k4 = P.rhs("k4", state=U3, fields=solve_field(P, U3), terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U").next
    Unp1 = P.value(
        "Unp1", U0 + dt / 6.0 * k1 + dt / 3.0 * k2 + dt / 3.0 * k3
        + dt / 6.0 * k4, at=endpoint.point)
    P.commit(endpoint, Unp1)
    P.validate()
    assert _coeff(Unp1, U0) == {0: 1.0}
    assert abs(_coeff(Unp1, k1)[1] - 1.0 / 6.0) < 1e-15
    assert abs(_coeff(Unp1, k2)[1] - 1.0 / 3.0) < 1e-15
    assert abs(_coeff(Unp1, k4)[1] - 1.0 / 6.0) < 1e-15
    print("OK  3. rk4 IR coefficients (no special RK4 class)")


def test_commit_once():
    P = adctime.Program("p")
    U = typed_state(P, "plasma")
    endpoint = typed_state(P, "plasma", state_name="U").next
    U1 = P.value(
        "U1", U + P.dt * P.rhs(
            state=U, fields=solve_field(P, U), terms=[Flux(), DefaultSource()]),
        at=endpoint.point)
    P.commit(endpoint, U1)
    try:
        P.commit(endpoint, U1)
    except ValueError as e:
        assert "committed more than once" in str(e), str(e)
        print("OK  4. double commit rejected")
        return
    raise SystemExit("expected ValueError on double commit")


def test_no_commit_rejected():
    P = adctime.Program("p")
    U = typed_state(P, "plasma")
    P.rhs(state=U, fields=solve_field(P, U), terms=[Flux(), DefaultSource()])
    try:
        P.validate()
    except ValueError as e:
        assert "commit" in str(e), str(e)
        print("OK  5. program with no commit rejected")
        return
    raise SystemExit("expected ValueError on missing commit")


def test_value_not_python_bool():
    P = adctime.Program("p")
    U = typed_state(P, "plasma")
    try:
        bool(U)
    except TypeError as e:
        assert "Program control flow" in str(e) or "Python bool" in str(e), str(e)
        print("OK  6. IR value cannot be used as a Python bool")
        return
    raise SystemExit("expected TypeError on bool(IR value)")


def _build_euler(scale=1.0):
    P = adctime.Program("forward_euler")
    dt = P.dt
    U = typed_state(P, "plasma")
    R = P.rhs(
        state=U, fields=solve_field(P, U), terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "plasma", state_name="U").next
    P.commit(endpoint, P.value(
        "U1", U + (scale * dt) * R, at=endpoint.point))
    return P


def test_ir_hash_deterministic_and_sensitive():
    assert _build_euler()._ir_hash() == _build_euler()._ir_hash()
    assert _build_euler(1.0)._ir_hash() != _build_euler(2.0)._ir_hash()
    print("OK  7. IR hash deterministic and coefficient-sensitive")


def test_solve_fields_distinct():
    P = adctime.Program("p")
    U = typed_state(P, "plasma")
    f0 = solve_field(P, U)
    f1 = solve_field(P, U)
    assert f0 is not f1 and f0.id != f1.id and f0.vtype == "fields"
    print("OK  8. each solve_fields is a distinct FieldContext value")


def test_rhs_records_sources_and_flux():
    from pops.physics._facade import Model

    model = Model("named-rhs")
    (u,) = model.conservative_vars("u")
    model.elliptic_rhs(u)
    electric = model.source_term("electric", [-u])
    chemistry = model.source_term("chemistry", [2 * u])
    P = adctime.Program("p")
    U = typed_state(P, "plasma", model=model)
    R = P.rhs(
        state=U, fields=solve_field(P, U), terms=[Flux(), electric, chemistry])
    assert R.attrs["flux"] is True
    assert R.attrs["sources"] == ("electric", "chemistry")
    print("OK  9. rhs records its flux flag and named sources")


def main():
    test_forward_euler_ir()
    test_ssprk2_ir()
    test_rk4_ir()
    test_commit_once()
    test_no_commit_rejected()
    test_value_not_python_bool()
    test_ir_hash_deterministic_and_sensitive()
    test_solve_fields_distinct()
    test_rhs_records_sources_and_flux()
    print("test_time_program : tout est vert")


if __name__ == "__main__":
    main()
