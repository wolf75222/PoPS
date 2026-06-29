"""Spec 5 sec.14.2.3: typed operator handles (epic ADC-479 / ADC-464).

A user-facing operator declarer (``m.rate_operator`` / ``m.source_term`` /
``m.linear_source``) returns an inert :class:`pops.model.OperatorHandle` carrying the
operator ``name`` (and ``kind``). The PUBLIC ``P.call`` requires the handle (a bare string is
refused); the INTERNAL ``P._call`` resolves a name token. Both follow the IDENTICAL registry
lookup + lowering, so ``P.call(handle, ...)`` builds the BYTE-IDENTICAL IR (same ``_ir_hash``) as
``P._call(name, ...)``.

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation); skips cleanly if pops is
not importable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops import model
    from pops.ir.expr import Const, Var
    from pops.model import OperatorHandle
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_handles (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A model declaring a named source, a named linear operator and a rate operator.

    Returns the model plus the handles the declarers returned (so the test can pass a
    handle straight into ``P.call``)."""
    mod = model.Module("euler_poisson_lorentz")
    u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
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
    h_src = mod.operator(
        name="electric", signature=(u, fields) >> model.Rate(u), kind="local_source",
        expr=[Const(0.0), rho * (-gx), rho * (-gy)])
    h_lin = mod.operator(
        name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
        kind="local_linear_operator",
        expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    h_rate = mod.rate_operator("explicit_rhs", flux=True, sources=[h_src])
    return mod, {"electric": OperatorHandle(h_src.name, kind=h_src.kind),
                 "lorentz": OperatorHandle(h_lin.name, kind=h_lin.kind),
                 "explicit_rhs": OperatorHandle(h_rate.name, kind=h_rate.kind)}


def test_handles_name_declared_operators():
    """Each typed OperatorHandle names an operator declared on the Module."""
    m, h = build_model()
    assert isinstance(h["electric"], OperatorHandle)
    assert isinstance(h["lorentz"], OperatorHandle)
    assert isinstance(h["explicit_rhs"], OperatorHandle)
    assert h["electric"].name == "electric" and h["electric"].kind == "local_source"
    assert h["lorentz"].name == "lorentz" and h["lorentz"].kind == "local_linear_operator"
    assert h["explicit_rhs"].name == "explicit_rhs" and h["explicit_rhs"].kind == "local_rate"
    # A handle for a default source name remains a typed selector.
    h_def = OperatorHandle("default", kind="local_source")
    assert isinstance(h_def, OperatorHandle) and h_def.name == "default"
    print("OK  typed OperatorHandle(name, kind) names Module operators")


def test_rate_operator_rejects_string_source_selectors():
    m, _ = build_model()
    with pytest.raises(TypeError, match="typed source operators/handles"):
        m.rate_operator("bad_rhs", flux=True, sources=["electric"])
    # The implicit built-in source sentinel remains allowed.
    m.rate_operator("default_rhs", flux=True, sources=["default"])
    print("OK  rate_operator rejects named source strings and accepts the default sentinel")


# A handle for the built-in default-Poisson field operator; the public P.call needs it (a bare
# string field name is refused). Internally _call resolves the name token identically.
_FIELDS = OperatorHandle("fields_from_state", kind="field_operator")


def _state(P, m):
    return P.state("U", block="plasma", space=m.state_spaces()["U"]).n


def _select(P, selector, *args, name=None):
    """Dispatch a selector: a handle goes through the PUBLIC P.call, a bare name through the
    INTERNAL P._call (the private name-token path the macros / lowering use)."""
    if isinstance(selector, OperatorHandle):
        return P.call(selector, *args, name=name)
    return P._call(selector, *args, name=name)


def _rate_program(m, selector):
    """Build a one-step predictor Program calling the rate operator via ``selector`` (a name or
    a handle). Returns the Program (its ``_ir_hash`` is the IR fingerprint)."""
    P = adctime.Program("prog").bind_operators(m)
    U = _state(P, m)
    f = P.call(_FIELDS, U)
    R = _select(P, selector, U, f)
    P.commit("plasma", P.linear_combine("u1", U + P.dt * R))
    return P


def test_call_handle_byte_identical_to_name():
    """P.call(handle) lowers to the byte-identical IR as the internal P._call(name) -- same _ir_hash."""
    m, h = build_model()
    prog_name = _rate_program(m, "explicit_rhs")          # internal _call(name)
    prog_handle = _rate_program(m, h["explicit_rhs"])     # public call(handle)
    assert prog_name._ir_hash() == prog_handle._ir_hash(), (
        "P.call(handle) must lower to the byte-identical IR as the internal P._call(name)")
    print("OK  P.call(handle) IR hash == P._call(name) IR hash: %s" % prog_name._ir_hash())


def test_name_path_byte_identical_across_models():
    """The internal name path P._call('name') is deterministic and equals the handle path for each
    declarer kind (rate / source / linear)."""
    m_a, _ = build_model()
    m_b, _ = build_model()
    h_a = _rate_program(m_a, "explicit_rhs")._ir_hash()
    h_b = _rate_program(m_b, "explicit_rhs")._ir_hash()
    assert h_a == h_b, "the internal P._call name path must be deterministic / unperturbed"
    m, h = build_model()

    def src_prog(selector):
        P = adctime.Program("p").bind_operators(m)
        U = _state(P, m)
        f = P.call(_FIELDS, U)
        s = _select(P, selector, U, f)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * s))
        return P

    assert src_prog("electric")._ir_hash() == src_prog(h["electric"])._ir_hash()

    def lin_prog(selector):
        P = adctime.Program("p").bind_operators(m)
        U = _state(P, m)
        f = P.call(_FIELDS, U)
        L = _select(P, selector, f)
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=U, fields=f)
        P.commit("plasma", U1)
        return P

    assert lin_prog("lorentz")._ir_hash() == lin_prog(h["lorentz"])._ir_hash()
    print("OK  internal name path byte-identical (rate / source / linear all match handle path)")


def test_public_call_rejects_a_string():
    """The PUBLIC P.call refuses a bare string operator NAME with a clear TypeError naming the
    handle path (the one public path is the typed handle)."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = _state(P, m)
    with pytest.raises(TypeError, match="typed operator handle"):
        P.call("explicit_rhs", U)
    print("OK  public P.call('explicit_rhs') -> TypeError naming the handle path")


def test_bad_type_rejected():
    """A non-handle selector is a clear TypeError on the public surface (typed handle required)."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = _state(P, m)
    with pytest.raises(TypeError, match="OperatorHandle"):
        P.call(123, U)
    with pytest.raises(TypeError, match="OperatorHandle"):
        P.call(None, U)
    print("OK  P.call(non-handle) -> clear TypeError")


def test_foreign_handle_rejected():
    """A handle whose name is not in the bound registry is rejected like an unknown string name."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = _state(P, m)
    foreign = OperatorHandle("not_declared_here", kind="local_rate")
    with pytest.raises(KeyError, match="unknown operator"):
        P.call(foreign, U)
    print("OK  foreign handle (name not in registry) rejected like an unknown name")


def test_handle_equality_and_repr():
    """OperatorHandle is value-like: equal by (name, kind), hashable, with a clear repr."""
    a = OperatorHandle("r", kind="local_rate")
    b = OperatorHandle("r", kind="local_rate")
    c = OperatorHandle("r", kind="local_source")
    assert a == b and hash(a) == hash(b)
    assert a != c and a != "r"
    assert repr(a) == "OperatorHandle('r', kind='local_rate')"
    assert repr(OperatorHandle("s")) == "OperatorHandle('s')"
    print("OK  OperatorHandle equality / hash / repr")


def main():
    test_handles_name_declared_operators()
    test_call_handle_byte_identical_to_name()
    test_name_path_byte_identical_across_models()
    test_public_call_rejects_a_string()
    test_bad_type_rejected()
    test_foreign_handle_rejected()
    test_handle_equality_and_repr()
    print("OK  test_operator_handles")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
