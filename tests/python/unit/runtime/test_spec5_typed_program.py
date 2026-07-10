"""Spec 5 sec.15: the time Program requires TYPED operators + terms (epic ADC-479, #23/#27).

De-stringing the time Program is the ONE public path -- not an opt-in. By DEFAULT:

  - criterion 23: ``P.call`` requires an operator HANDLE (from ``m.rate`` / ``m.field_operator`` /
    ``m.source_term`` / ``m.rate_operator`` / ``m.linear_source``); a bare string operator NAME is
    REFUSED with a clear ``TypeError`` naming the handle path;
  - criterion 27: ``P.rhs`` requires the typed ``terms=[Flux(), source]`` list; the legacy
    ``flux=``/``sources=``/``fluxes=`` boolean/name form (and a bare ``P.rhs``) is REFUSED with a
    clear ``TypeError`` naming ``terms=``.

The legacy string operator name + ``flux=``/``sources=`` builders survive ONLY as the INTERNAL
``P._call`` / ``P._rhs_legacy`` (prefixed ``_``, undocumented): the typed front doors and the
``pops.lib.time`` macros lower through them, and the typed path builds the BYTE-IDENTICAL IR (same
``Program._ir_hash``) as the private path. There is no second public path and no enable flag.

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation, no ``_pops``); skips cleanly if
pops is not importable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops.ir.expr import Const
    from pops.model import OperatorHandle
    from pops.numerics.terms import Flux, SourceTerm
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


def _operator_handle(model, name):
    """Return the owner-qualified handle for an operator declared by ``model``."""
    registry = model.operator_registry()
    operator = registry.get(name)
    return OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)


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
    f = P.call(_operator_handle(m, "fields_from_state"), U)
    R = P.call(h["explicit_rhs"], U, f)
    P.commit(P.state("U", block="plasma").next, P.linear_combine("u1", U + P.dt * R))
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
    P.commit(P.state("U", block="plasma").next, P.linear_combine("u1", U + P.dt * R))
    return P


def test_call_handle_byte_identical_to_private_name():
    """The typed P.call(handle) path builds the BYTE-IDENTICAL IR as the internal P._call(name) path:
    the public reject changes only the spelling, never the lowering (criterion 23)."""
    m, h = build_model()
    public = _rate_program(
        m, selector=h["explicit_rhs"],
        fields_selector=_operator_handle(m, "fields_from_state"))._ir_hash()
    private = _rate_program(m, selector="explicit_rhs",
                            fields_selector="fields_from_state")._ir_hash()
    assert public == private, (public, private)
    print("OK  P.call(handle) IR == P._call(name) IR: %s" % public)


# --- criterion 27: P.rhs requires terms= ---------------------------------------------------------

def test_rhs_rejects_legacy_flux_sources():
    """P.rhs(flux=True) / P.rhs(sources=[...]) / P.rhs(fluxes=[...]) / a bare P.rhs(state=U) are all
    refused with a clear TypeError naming P.rhs(terms=[...]) (criterion 27)."""
    m, _ = build_model()
    P = adctime.Program("p")
    U = P.state("plasma")
    f = P.solve_fields(U)
    msg = "P.rhs requires the typed terms="
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f, flux=True)
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f, sources=["electric"])
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f, fluxes=["default"])
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f)  # the bare legacy default is the bool/name form too
    print("OK  P.rhs(flux=/sources=/fluxes=/bare) -> TypeError naming terms=")


def _rhs_program(*, terms=None, legacy=None):
    """A one-block Euler Program whose single rhs is built either via the public terms= (a list) or
    the INTERNAL _rhs_legacy (a (flux, sources) pair)."""
    m, _ = build_model()
    P = adctime.Program("rhs").bind_operators(m)
    U = P.state("plasma")
    f = P.solve_fields(U)
    if terms is not None:
        R = P.rhs("R", state=U, fields=f, terms=terms)
    else:
        flux, sources = legacy
        R = P._rhs_legacy(name="R", state=U, fields=f, flux=flux, sources=sources)
    P.commit(P.state("U", block="plasma").next, P.linear_combine("U1", U + P.dt * R))
    return P


def test_rhs_accepts_terms():
    """P.rhs(terms=[Flux(), source]) is the allowed typed spelling."""
    P = _rhs_program(terms=[Flux(), SourceTerm("electric")])
    P.validate()
    print("OK  P.rhs(terms=[Flux(), SourceTerm('electric')]) accepted + validates")


def test_rhs_rejects_free_source_name():
    """A public RHS source selector retains a typed descriptor/handle; strings stay private."""
    with pytest.raises(TypeError, match="free source name"):
        _rhs_program(terms=[Flux(), "electric"])


def test_rhs_terms_byte_identical_to_private_legacy():
    """The typed terms= path builds the BYTE-IDENTICAL IR as the internal _rhs_legacy(flux=,sources=)
    path: the public reject changes only the spelling, never the lowering (criterion 27)."""
    public = _rhs_program(terms=[Flux(), SourceTerm("electric")])._ir_hash()
    private = _rhs_program(legacy=(True, ["electric"]))._ir_hash()
    assert public == private, (public, private)
    print("OK  P.rhs(terms=) IR == P._rhs_legacy(flux=,sources=) IR: %s" % public)


# --- the typed-call internal lowering is one public path (no leak through P.rhs) -----------------

def test_typed_call_lowers_through_private_rhs():
    """A typed P.call rate operator lowers INTERNALLY through P._rhs_legacy (not the public P.rhs),
    so the public reject never sees the internal lowering: the rate program (whose rhs is built by
    P.call) compiles cleanly and is byte-identical to the private name path."""
    m, h = build_model()
    typed = _rate_program(
        m, selector=h["explicit_rhs"],
        fields_selector=_operator_handle(m, "fields_from_state"))
    typed.validate()
    private = _rate_program(m, selector="explicit_rhs", fields_selector="fields_from_state")
    assert typed._ir_hash() == private._ir_hash()
    print("OK  P.call lowering through P._rhs_legacy is byte-identical + validates")


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
