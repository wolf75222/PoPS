"""Spec 5 sec.14.2.3: typed operator handles (epic ADC-479 / ADC-464).

A user-facing operator declarer (``m.rate_operator`` / ``m.source_term`` /
``m.linear_source``) returns an inert :class:`pops.model.OperatorHandle` carrying the
operator ``name`` (and ``kind``). The PUBLIC ``P.call`` requires the owner-qualified handle (a bare
string is refused). A declarer-returned handle and the authoritative registry-issued handle are the
same identity and therefore build BYTE-IDENTICAL IR (same ``_ir_hash``).

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation); skips cleanly if pops is
not importable. Never fakes the engine.
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
    print("skip test_operator_handles (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A model declaring a named source, a named linear operator and a rate operator.

    Returns the model plus the handles the declarers returned (so the test can pass a
    handle straight into ``P.call``)."""
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


def test_declarers_return_operator_handles():
    """Each user-facing declarer returns a typed OperatorHandle naming the declared operator."""
    m, h = build_model()
    assert isinstance(h["electric"], OperatorHandle)
    assert isinstance(h["lorentz"], OperatorHandle)
    assert isinstance(h["explicit_rhs"], OperatorHandle)
    assert h["electric"].name == "electric" and h["electric"].kind == "local_source"
    assert h["lorentz"].name == "lorentz" and h["lorentz"].kind == "local_linear_operator"
    assert h["explicit_rhs"].name == "explicit_rhs" and h["explicit_rhs"].kind == "local_rate"
    # The default source_term alias also returns a handle (name 'default').
    m2 = Model("ds")
    rho2, mx2, my2 = m2.conservative_vars("rho", "mx", "my")
    h_def = m2.source_term("default", [Const(0.0), -mx2, -my2])
    assert isinstance(h_def, OperatorHandle) and h_def.name == "default"
    print("OK  declarers return typed OperatorHandle(name, kind)")


def _operator_handle(model, name):
    """Return the owner-qualified handle for an operator declared by ``model``."""
    return model.module.operator_handle(name)


def _program_state(model, name):
    """Create one typed block instance and bind its declared state to a Program."""
    module = model.module
    block = Case(name="%s-case" % name).block("plasma", model)
    state = module.state_handle(module.state_spaces()["U"])
    program = adctime.Program(name).bind_operators(module)
    return program, program.state(block, state)


def _rate_program(m, selector):
    """Build a one-step predictor through the public typed operator path."""
    P, state = _program_state(m, "prog")
    U = state.n
    f = P.call(_operator_handle(m, "fields_from_state"), U)
    R = P.call(selector, U, f)
    P.commit(state.next, P.value("u1", U + P.dt * R, at=state.next.point))
    return P


def test_declarer_handle_byte_identical_to_registry_handle():
    """Declarer and registry handles are one identity and lower to byte-identical IR."""
    m, h = build_model()
    registry_handle = _operator_handle(m, "explicit_rhs")
    assert h["explicit_rhs"] == registry_handle
    assert h["explicit_rhs"].qualified_id == registry_handle.qualified_id
    declared = _rate_program(m, h["explicit_rhs"])
    registered = _rate_program(m, registry_handle)
    assert declared._ir_hash() == registered._ir_hash()
    print("OK  declarer handle IR == authoritative registry handle IR: %s" % declared._ir_hash())


def test_typed_path_is_deterministic_and_complete_across_models():
    """Qualified handles distinguish live declarations while structural IR stays deterministic."""
    m_a, handles_a = build_model()
    m_b, handles_b = build_model()
    assert handles_a["explicit_rhs"] != handles_b["explicit_rhs"]
    h_a = _rate_program(m_a, handles_a["explicit_rhs"])._ir_hash()
    h_b = _rate_program(m_b, handles_b["explicit_rhs"])._ir_hash()
    assert h_a == h_b
    m, h = build_model()

    def src_prog(selector):
        P, state = _program_state(m, "p")
        U = state.n
        f = P.call(_operator_handle(m, "fields_from_state"), U)
        s = P.call(selector, U, f)
        P.commit(state.next, P.value("u1", U + P.dt * s, at=state.next.point))
        return P

    assert src_prog(_operator_handle(m, "electric"))._ir_hash() == src_prog(
        h["electric"])._ir_hash()

    def lin_prog(selector):
        P, state = _program_state(m, "p")
        U = state.n
        rhs = P.value("rhs", U, at=state.next.point)
        f = P.call(_operator_handle(m, "fields_from_state"), rhs)
        L = P.call(selector, f)
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=rhs, fields=f)
        P.commit(state.next, U1)
        return P

    assert lin_prog(_operator_handle(m, "lorentz"))._ir_hash() == lin_prog(
        h["lorentz"])._ir_hash()
    print("OK  typed rate / source / linear paths preserve qualified identity and deterministic IR")


def test_public_call_rejects_a_string():
    """The PUBLIC P.call refuses a bare string operator NAME with a clear TypeError naming the
    handle path (the one public path is the typed handle)."""
    m, _ = build_model()
    P, state = _program_state(m, "p")
    U = state.n
    with pytest.raises(TypeError, match="typed operator handle"):
        P.call("explicit_rhs", U)
    print("OK  public P.call('explicit_rhs') -> TypeError naming the handle path")


def test_bad_type_rejected():
    """A non-handle selector is a clear TypeError on the public surface (typed handle required)."""
    m, _ = build_model()
    P, state = _program_state(m, "p")
    U = state.n
    with pytest.raises(TypeError, match="OperatorHandle"):
        P.call(123, U)
    with pytest.raises(TypeError, match="OperatorHandle"):
        P.call(None, U)
    print("OK  P.call(non-handle) -> clear TypeError")


def test_foreign_and_unknown_handles_rejected():
    """Neither a foreign owner nor an unknown local declaration can fall back by name."""
    m, _ = build_model()
    foreign_model, foreign_handles = build_model()
    P, state = _program_state(m, "p")
    U = state.n
    with pytest.raises(ValueError, match="no operator registry is bound for owner"):
        P.call(foreign_handles["explicit_rhs"], U)

    unknown = OperatorHandle(
        "not_declared_here", kind="local_rate",
        owner=m.operator_registry().owner_path)
    with pytest.raises(KeyError, match="unknown operator"):
        P.call(unknown, U)
    assert foreign_model.operator_registry().owner_path != m.operator_registry().owner_path
    print("OK  foreign-owner and unknown-local handles are rejected without name fallback")


def test_handle_equality_and_repr():
    """OperatorHandle equality and hashing use the complete qualified identity."""
    from pops.model import OwnerPath
    owner = OwnerPath.descriptor("operator-handles")
    a = OperatorHandle("r", kind="local_rate", owner=owner)
    b = OperatorHandle("r", kind="local_rate", owner=owner)
    c = OperatorHandle("r", kind="local_source", owner=owner)
    assert a == b and hash(a) == hash(b)
    assert a != c and a != "r"
    assert repr(a) == (
        "OperatorHandle('r', kind='local_rate', owner='descriptor:operator-handles')")
    assert repr(OperatorHandle("s", kind="local_source", owner=owner)) == (
        "OperatorHandle('s', kind='local_source', owner='descriptor:operator-handles')")
    assert a.inspect()["owner_path"] == owner.to_data()
    print("OK  OperatorHandle equality / hash / repr")


def main():
    test_declarers_return_operator_handles()
    test_declarer_handle_byte_identical_to_registry_handle()
    test_typed_path_is_deterministic_and_complete_across_models()
    test_public_call_rejects_a_string()
    test_bad_type_rejected()
    test_foreign_and_unknown_handles_rejected()
    test_handle_equality_and_repr()
    print("OK  test_operator_handles")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
