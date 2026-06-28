"""Spec 5 sec.15: the time Program requires TYPED operators + terms (epic ADC-479, #23/#27).

De-stringing the time Program is the ONE public path -- not an opt-in. By DEFAULT:

  - criterion 23: ``P.call`` requires an operator HANDLE (from ``m.rate`` / ``m.field_operator`` /
    ``m.source_term`` / ``m.rate_operator`` / ``m.linear_source``); a bare string operator NAME is
    REFUSED with a clear ``TypeError`` naming the handle path;
  - criterion 27: ``P.rhs`` requires the typed ``terms=[Flux(), source]`` list; the legacy
    ``flux=``/``sources=``/``fluxes=`` boolean/name form (and a bare ``P.rhs``) is REFUSED with a
    clear ``TypeError`` naming ``terms=``.

The legacy string operator name + ``flux=``/``sources=`` builders survive ONLY as INTERNAL
``P._call`` / ``P._rhs_legacy`` (prefixed ``_``, undocumented). The public typed ``P.call`` and
the internal ``P._call`` both record first-class ``operator_call`` nodes; only the package macros
that intentionally build primitive schemes still use ``_rhs_legacy``. There is no second public
path and no enable flag.

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation, no ``_pops``); skips cleanly if
pops is not importable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops.ir.expr import Const
    from pops.model import OperatorHandle
    from pops.numerics.terms import Flux
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_spec5_typed_program (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A model declaring a named source and a rate operator (mirrors test_operator_handles).

    Returns the model plus the handles the declarers returned, so the test can pass a typed handle
    straight into ``P.call``."""
    m = Model("euler_poisson_lorentz")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    h_src = m.source_term("electric", [Const(0.0), rho * (-gx), rho * (-gy)])
    m.elliptic_rhs(rho - 1.0)
    h_rate = m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m, {"electric": h_src, "explicit_rhs": h_rate}


# The built-in default-Poisson field operator, as a handle (the public P.call needs a handle; the
# internal P._call resolves the same name token byte-identically).
_FIELDS = OperatorHandle("fields_from_state", kind="field_operator")


# --- criterion 23: P.call requires a typed handle ------------------------------------------------

def test_call_rejects_a_string_operator():
    """P.call('name', ...) with a STRING operator is a clear TypeError naming the typed handle path."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = P.state("plasma")
    with pytest.raises(TypeError, match="P.call requires a typed operator handle"):
        P.call("fields_from_state", U)
    with pytest.raises(TypeError, match="P.call requires a typed operator handle"):
        P.call("explicit_rhs", U)
    print("OK  P.call(str) -> TypeError naming the typed handle path")


def test_call_accepts_an_operator_handle():
    """P.call(handle) is the allowed spelling and builds a valid program."""
    m, h = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = P.state("plasma")
    f = P.call(_FIELDS, U)
    R = P.call(h["explicit_rhs"], U, f)
    P.commit("plasma", P.linear_combine("u1", U + P.dt * R))
    P.validate()
    print("OK  P.call(handle) accepted + validates")


def _rate_program(m, *, selector, fields_selector):
    """A one-step predictor Program. ``selector`` / ``fields_selector`` are EITHER a handle (->
    public P.call) OR a name str (-> internal P._call), so the test can build the same IR both ways."""
    P = adctime.Program("prog").bind_operators(m)
    U = P.state("plasma")
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


# --- criterion 27: no public P.rhs; internal _rhs_terms requires terms= --------------------------

def test_rhs_rejects_legacy_flux_sources():
    """There is no public P.rhs. The internal typed-term helper refuses legacy flux/sources kwargs,
    while _rhs_legacy remains the primitive internal lowering target."""
    m, _ = build_model()
    P = adctime.Program("p")
    U = P.state("plasma")
    f = P._solve_fields(U)
    msg = "_rhs_terms requires terms="
    assert not hasattr(P, "rhs")
    with pytest.raises(TypeError, match=msg):
        P._rhs_terms("R", state=U, fields=f, flux=True)
    with pytest.raises(TypeError, match=msg):
        P._rhs_terms("R", state=U, fields=f, sources=["electric"])
    with pytest.raises(TypeError, match=msg):
        P._rhs_terms("R", state=U, fields=f, fluxes=["default"])
    with pytest.raises(TypeError, match=msg):
        P._rhs_terms("R", state=U, fields=f)
    print("OK  no P.rhs; _rhs_terms(flux=/sources=/fluxes=/bare) -> TypeError naming terms=")


def _rhs_program(*, terms=None, legacy=None):
    """A one-block Euler Program whose single rhs is built either via the public terms= (a list) or
    the INTERNAL _rhs_legacy (a (flux, sources) pair)."""
    P = adctime.Program("rhs")
    U = P.state("plasma")
    f = P._solve_fields(U)
    if terms is not None:
        R = P._rhs_terms("R", state=U, fields=f, terms=terms)
    else:
        flux, sources = legacy
        R = P._rhs_legacy(name="R", state=U, fields=f, flux=flux, sources=sources)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


def test_rhs_accepts_terms():
    """P._rhs_terms(terms=[Flux(), source]) is the internal typed-term spelling."""
    P = _rhs_program(terms=[Flux(), "electric"])
    P.validate()
    print("OK  P._rhs_terms(terms=[Flux(), 'electric']) accepted + validates")


def test_rhs_terms_byte_identical_to_private_legacy():
    """The typed terms= path builds the BYTE-IDENTICAL IR as the internal _rhs_legacy(flux=,sources=)
    path: the public reject changes only the spelling, never the lowering (criterion 27)."""
    public = _rhs_program(terms=[Flux(), "electric"])._ir_hash()
    private = _rhs_program(legacy=(True, ["electric"]))._ir_hash()
    assert public == private, (public, private)
    print("OK  P._rhs_terms(terms=) IR == P._rhs_legacy(flux=,sources=) IR: %s" % public)


# --- the typed-call internal lowering is one public path (no leak through P.rhs) -----------------

def test_typed_call_records_operator_call_not_rhs():
    """A typed P.call rate operator records operator_call IR, not a private rhs node."""
    m, h = build_model()
    typed = _rate_program(m, selector=h["explicit_rhs"], fields_selector=_FIELDS)
    typed.validate()
    private = _rate_program(m, selector="explicit_rhs", fields_selector="fields_from_state")
    assert typed._ir_hash() == private._ir_hash()
    assert any(v.op == "operator_call" for v in typed._values)
    assert not any(v.op == "rhs" for v in typed._values)
    print("OK  P.call records operator_call IR and validates")


# --- a lib.time macro authors through the private path and stays green ---------------------------

def test_lib_time_macro_uses_the_private_path():
    """A pops.lib.time scheme macro (ssprk2) builds and validates: it authors the RHS through the
    private P._rhs_legacy, so the public terms=-only reject does not break the ready schemes."""
    from pops.lib.time import ssprk2
    P = adctime.Program("m")
    ssprk2(P, "plasma")
    P.validate()
    print("OK  lib.time.ssprk2 builds + validates via the private path")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
