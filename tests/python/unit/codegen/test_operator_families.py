"""ADC-559: the mathematical operator declarers and inspectable OperatorHandle.

``m.rate`` / ``m.field_solve`` / ``m.local_linear_map`` are thin MATHEMATICAL aliases funneling
into the ONE typed registry (rate_operator / elliptic_field / linear_source); each returns the
canonical :class:`pops.model.OperatorHandle` stamped with the derived ``Signature`` and the readable
``category``. The handle is inspectable (``.signature`` / ``.category`` / ``.inspect()``); a bare,
unregistered math object (a board ``local_linear_operator``) still raises when called directly.

Pure Python (no compilation); skips cleanly if pops is not importable. Never fakes the engine.
"""
import sys

try:
    from pops._ir.expr import Const, Var
    from pops.model import (
        OPERATOR_FAMILIES, OPERATOR_KINDS, OperatorHandle, operator_family,
    )
    from pops.physics._facade import Model
    from pops.problem import Case
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_families (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A model with the three surfaces the mathematical declarers cover."""
    m = Model("euler_poisson_lorentz")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), rho * (-gx), rho * (-gy)])
    m.elliptic_rhs(rho - 1.0)
    return m, rho, bz


def _program_state(model, name, provider):
    """Create a concrete state plus the Case-owned field solve backed by ``provider``."""
    from pops.descriptors import Descriptor
    from pops.fields import FieldDiscretization, FieldOperator
    from pops.math import ValueExpr
    from pops.math import laplacian
    from pops.model import Module

    module = model.module
    case = Case(name="%s-case" % name)
    block = case.block("plasma", model)
    state = module.state_handle(module.state_spaces()["U"])
    provider_space = provider.signature.output
    field_module = Module("%s-field" % name)
    field_space = field_module.field_space("fields", provider_space.components)
    field_block = case.block("field-storage", field_module)
    unknown = field_block[field_module.field_handle(field_space)]

    class _Method(Descriptor):
        category = "field_method"

        def to_data(self):
            return {"type": "unit-second-order"}

    class _Solver(Descriptor):
        category = "elliptic_solver"

        def to_data(self):
            return {"type": "unit-krylov"}

    field = case.field(
        FieldOperator(
            "fields",
            unknown=unknown,
            equation=-laplacian(ValueExpr(unknown)) == ValueExpr(block[state]),
            providers=provider,
        ),
        FieldDiscretization(method=_Method(), boundaries=(), solver=_Solver()),
    )
    program = adctime.Program(name)
    return program, program.state(block[state]), field


def test_operator_family_is_total_over_kinds():
    """Every declared operator kind maps to a mathematical family (no KeyError, no gap)."""
    for kind in OPERATOR_KINDS:
        fam = operator_family(kind)
        assert isinstance(fam, str) and fam, "kind %r has no family" % (kind,)
        assert kind in OPERATOR_FAMILIES, "kind %r missing from the family table" % (kind,)
    # An unknown kind maps to "other" rather than raising.
    assert operator_family("not_a_kind") == "other"
    print("OK  operator_family is total over OPERATOR_KINDS; unknown -> 'other'")


def test_rate_declarer_returns_typed_inspectable_handle():
    """m.rate(...) returns a canonical OperatorHandle with the (U, Fields) -> Rate(U) signature."""
    m, _, _ = build_model()
    R = m.rate("explicit_rhs", flux=True, sources=["electric"])
    assert isinstance(R, OperatorHandle)
    assert R.name == "explicit_rhs" and R.kind == "local_rate" and R.category == "rate"
    # The declared signature is the mathematically-explicit (U, Fields) -> Rate(U).
    ins = R.inspect()
    assert ins["name"] == "explicit_rhs" and ins["kind"] == "local_rate"
    assert ins["category"] == "rate"
    assert "Rate(U)" in ins["signature"] and "StateSpace('U'" in ins["signature"]
    assert "FieldSpace" in ins["signature"]  # reads 'electric' -> depends on fields
    print("OK  m.rate -> OperatorHandle category 'rate', signature (U, Fields) -> Rate(U)")


def test_local_linear_map_signature():
    """m.local_linear_map(...) is Fields -> LocalLinearOperator(U, U), category local_linear_map."""
    m, _, bz = build_model()
    L = m.local_linear_map("lorentz", [[0.0, 0.0, 0.0],
                                       [0.0, 0.0, bz],
                                       [0.0, -bz, 0.0]])
    assert isinstance(L, OperatorHandle)
    assert L.kind == "local_linear_operator" and L.category == "local_linear_map"
    assert "LocalLinearOperator(StateSpace('U'" in repr(L.signature)
    print("OK  m.local_linear_map -> Fields -> LocalLinearOperator(U, U)")


def test_field_solve_signature():
    """m.field_solve(...) is U -> Fields, category field_solve."""
    m, _, _ = build_model()
    F = m.field_solve("psi", rhs=Var("rho", "cons"), aux=["psi", "psi_x", "psi_y"])
    assert isinstance(F, OperatorHandle)
    assert F.kind == "field_operator" and F.category == "field_solve"
    sig = repr(F.signature)
    assert "StateSpace('U'" in sig and "FieldSpace('psi'" in sig
    print("OK  m.field_solve -> U -> Fields")


def test_declarers_funnel_into_one_registry():
    """The mathematical declarers register into the SAME typed registry as the classic ones (no
    parallel facade registry) and return the exact registry-issued typed identity.
    """
    m, _, bz = build_model()
    r_math = m.rate("rate_math", flux=True, sources=["electric"])
    # The classic declarer produces the byte-identical operator under a distinct name.
    r_classic = m.rate_operator("rate_classic", flux=True, sources=["electric"])
    reg = m.operator_registry()
    assert "rate_math" in reg and "rate_classic" in reg
    # Same kind + same structural signature (only the name differs).
    assert reg.get("rate_math").kind == reg.get("rate_classic").kind == "local_rate"
    assert reg.get("rate_math").signature == reg.get("rate_classic").signature

    fields_h = m.module.operator_handle("fields_from_state")
    registered_math = m.module.operator_handle("rate_math")
    registered_classic = m.module.operator_handle("rate_classic")
    assert r_math == registered_math
    assert r_classic == registered_classic
    assert r_math.qualified_id == registered_math.qualified_id
    assert r_classic.qualified_id == registered_classic.qualified_id

    def prog(handle):
        from pops.time import FailRun

        P, state, field = _program_state(m, "p", fields_h)
        U = state.n
        f = field(U).consume(action=FailRun())
        R = handle(U, f)
        P.commit(state.next, P.value("u1", U + P.dt * R, at=state.next.point))
        return P

    assert prog(r_math)._ir_hash() == prog(registered_math)._ir_hash()
    assert prog(r_classic)._ir_hash() == prog(registered_classic)._ir_hash()
    # Two distinct declarations keep distinct qualified identities even when their
    # implementations and signatures are structurally equal.
    assert r_math != r_classic
    assert prog(r_math)._ir_hash() != prog(r_classic)._ir_hash()
    print("OK  both declarers funnel into one registry with distinct qualified identities")


def test_bare_math_object_raises_when_called():
    """A bare, unregistered local-linear math object raises when called directly (ADC-559)."""
    from pops.physics.board_handles import LocalLinearOperatorExpr
    expr = LocalLinearOperatorExpr("C_B", [[1.0]])
    try:
        expr(object())
        raise AssertionError("expected a TypeError calling an unregistered math object")
    except TypeError as exc:
        assert "not a callable operator" in str(exc)
        assert "Register it" in str(exc)
    print("OK  a bare unregistered math object raises when called")


def test_fields_handle_still_field_solve_category():
    """A FieldsHandle (ADC-556 subtype) keeps its own __call__ AND reports category field_solve."""
    from pops.model import OwnerPath
    from pops.physics.board_handles import FieldsHandle
    fh = FieldsHandle(
        "E", outputs={"E": object()}, solver=None,
        owner=OwnerPath.descriptor("fields-handle"))
    assert isinstance(fh, OperatorHandle)
    assert fh.kind == "field_operator" and fh.category == "field_solve"
    # inspect() works on the subtype (its signature is None: FieldsHandle carries no Signature).
    assert fh.inspect()["category"] == "field_solve"
    print("OK  FieldsHandle reports category 'field_solve' and inspect() works")


def main():
    test_operator_family_is_total_over_kinds()
    test_rate_declarer_returns_typed_inspectable_handle()
    test_local_linear_map_signature()
    test_field_solve_signature()
    test_declarers_funnel_into_one_registry()
    test_bare_math_object_raises_when_called()
    test_fields_handle_still_field_solve_category()
    print("OK  test_operator_families")


if __name__ == "__main__":
    main()
