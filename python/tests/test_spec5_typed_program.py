"""Spec 5 sec.15: the time Program requires TYPED operators + terms (epic ADC-479, #23/#27).

De-stringing the time Program is the ONE public path -- not an opt-in. By DEFAULT:

  - criterion 23: ``P.call`` requires an operator HANDLE (from ``m.rate`` / ``m.field_operator`` /
    ``m.source_term`` / ``m.rate_operator`` / ``m.linear_source``); a bare string operator NAME is
    REFUSED with a clear ``TypeError`` naming the handle path;
  - criterion 27: ``P.rhs`` requires the typed ``terms=[Flux(), source]`` list; the stringly
    ``flux=``/``sources=``/``fluxes=`` boolean/name form (and a bare ``P.rhs``) is REFUSED with a
    clear ``TypeError`` naming ``terms=``.

The string operator name + ``flux=``/``sources=`` builders survive ONLY as INTERNAL
``P._call`` / ``P._rate_from_transport`` (prefixed ``_``, undocumented). The public typed ``P.call`` and
the internal ``P._call`` both record first-class ``call`` nodes; only the package macros
that intentionally build primitive schemes still use ``_rate_from_transport``. There is no second public
path and no enable flag.

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation, no ``_pops``); skips cleanly if
pops is not importable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops import model
    from pops.ir.expr import Const, Var
    from pops.model import OperatorHandle
    from pops.numerics.terms import Flux
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_spec5_typed_program (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A model declaring a named source and a rate operator (mirrors test_operator_handles).

    Returns the model plus the handles the declarers returned, so the test can pass a typed handle
    straight into ``P.call``."""
    mod = model.Module("euler_poisson_lorentz")
    u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy = Var("grad_x", "aux"), Var("grad_y", "aux")
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
    h_rate = mod.rate_operator("explicit_rhs", flux=True, sources=[h_src])
    return mod, {"electric": OperatorHandle(h_src.name, kind=h_src.kind),
                 "explicit_rhs": OperatorHandle(h_rate.name, kind=h_rate.kind)}


# The built-in default-Poisson field operator, as a handle (the public P.call needs a handle; the
# internal P._call resolves the same name token byte-identically).
_FIELDS = OperatorHandle("fields_from_state", kind="field_operator")


def _state(P, m=None):
    space = m.state_spaces()["U"] if m is not None else None
    return P.state("U", block="plasma", space=space).n


# --- criterion 23: P.call requires a typed handle ------------------------------------------------

def test_call_rejects_a_string_operator():
    """P.call('name', ...) with a STRING operator is a clear TypeError naming the typed handle path."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = _state(P, m)
    with pytest.raises(TypeError, match="P.call requires a typed operator handle"):
        P.call("fields_from_state", U)
    with pytest.raises(TypeError, match="P.call requires a typed operator handle"):
        P.call("explicit_rhs", U)
    print("OK  P.call(str) -> TypeError naming the typed handle path")


def test_call_accepts_an_operator_handle():
    """P.call(handle) is the allowed spelling and builds a valid program."""
    m, h = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = _state(P, m)
    f = P.call(_FIELDS, U)
    R = P.call(h["explicit_rhs"], U, f)
    P.commit("plasma", P.linear_combine("u1", U + P.dt * R))
    P.validate()
    print("OK  P.call(handle) accepted + validates")


def _rate_program(m, *, selector, fields_selector):
    """A one-step predictor Program. ``selector`` / ``fields_selector`` are EITHER a handle (->
    public P.call) OR a name str (-> internal P._call), so the test can build the same IR both ways."""
    P = adctime.Program("prog").bind_operators(m)
    U = _state(P, m)
    f = (P.call(fields_selector, U) if isinstance(fields_selector, OperatorHandle)
         else P._call(fields_selector, U))
    R = (P.call(selector, U, f) if isinstance(selector, OperatorHandle)
         else P._call(selector, U, f))
    P.commit("plasma", P.linear_combine("u1", U + P.dt * R))
    return P


def test_call_handle_byte_identical_to_private_name():
    """The typed P.call(handle) path builds the BYTE-IDENTICAL IR as the internal P._call(name) path:
    the public reject changes only the spelling, never the lowering (criterion 23)."""
    m, h = build_model()
    public = _rate_program(m, selector=h["explicit_rhs"], fields_selector=_FIELDS)._ir_hash()
    private = _rate_program(m, selector="explicit_rhs",
                            fields_selector="fields_from_state")._ir_hash()
    assert public == private, (public, private)
    print("OK  P.call(handle) IR == P._call(name) IR: %s" % public)


# --- criterion 27: no public P.rhs; internal _rate_from_terms requires terms= ---------------------

def test_rhs_rejects_stringly_flux_sources():
    """There is no public P.rhs. The internal typed-term helper refuses stringly flux/sources kwargs,
    while _rate_from_transport remains the primitive internal lowering target."""
    m, _ = build_model()
    P = adctime.Program("p")
    U = _state(P)
    f = P._fields_from_state(U)
    msg = "_rate_from_terms requires terms="
    assert not hasattr(P, "rhs")
    with pytest.raises(TypeError, match=msg):
        P._rate_from_terms("R", state=U, fields=f, flux=True)
    with pytest.raises(TypeError, match=msg):
        P._rate_from_terms("R", state=U, fields=f, sources=["electric"])
    with pytest.raises(TypeError, match=msg):
        P._rate_from_terms("R", state=U, fields=f, fluxes=["default"])
    with pytest.raises(TypeError, match=msg):
        P._rate_from_terms("R", state=U, fields=f)
    print("OK  no P.rhs; _rate_from_terms(flux=/sources=/fluxes=/bare) -> TypeError naming terms=")


def _rhs_program(*, terms=None, transport=None):
    """A one-block Euler Program whose single rhs is built either via the public terms= (a list) or
    the INTERNAL _rate_from_transport (a (flux, sources) pair)."""
    P = adctime.Program("rhs")
    U = _state(P)
    f = P._fields_from_state(U)
    if terms is not None:
        R = P._rate_from_terms("R", state=U, fields=f, terms=terms)
    else:
        flux, sources = transport
        R = P._rate_from_transport(name="R", state=U, fields=f, flux=flux, sources=sources)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


def test_rhs_accepts_terms():
    """P._rate_from_terms(terms=[Flux(), source]) is the internal typed-term spelling."""
    P = _rhs_program(terms=[Flux(), "electric"])
    P.validate()
    print("OK  P._rate_from_terms(terms=[Flux(), 'electric']) accepted + validates")


def test_rhs_terms_byte_identical_to_private_transport():
    """The typed terms= path builds the BYTE-IDENTICAL IR as the internal _rate_from_transport(flux=,sources=)
    path: the public reject changes only the spelling, never the lowering (criterion 27)."""
    public = _rhs_program(terms=[Flux(), "electric"])._ir_hash()
    private = _rhs_program(transport=(True, ["electric"]))._ir_hash()
    assert public == private, (public, private)
    print("OK  P._rate_from_terms(terms=) IR == P._rate_from_transport(flux=,sources=) IR: %s" % public)


# --- the typed-call internal lowering is one public path (no leak through P.rhs) -----------------

def test_typed_call_records_call_not_rhs():
    """A typed P.call rate operator records call IR, not a private rhs node."""
    m, h = build_model()
    typed = _rate_program(m, selector=h["explicit_rhs"], fields_selector=_FIELDS)
    typed.validate()
    private = _rate_program(m, selector="explicit_rhs", fields_selector="fields_from_state")
    assert typed._ir_hash() == private._ir_hash()
    assert any(v.op == "call" for v in typed._values)
    assert not any(v.op == "rhs" for v in typed._values)
    print("OK  P.call records call IR and validates")


# --- a lib.time macro authors through the private path and stays green ---------------------------

def test_lib_time_macro_uses_the_private_path():
    """A pops.lib.time scheme macro (ssprk2) builds and validates: it authors the RHS through the
    private P._rate_from_transport, so the public terms=-only reject does not break the ready schemes."""
    from pops.lib.time import ssprk2
    P = adctime.Program("m")
    ssprk2(P, "plasma")
    P.validate()
    print("OK  lib.time.ssprk2 builds + validates via the private path")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
