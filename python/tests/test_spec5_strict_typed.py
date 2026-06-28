"""Spec 5 sec.15 strict de-stringing enforcement (epic ADC-479, criteria 23 + 27).

De-stringing the Program (typed operator handles in ``P.call``, typed ``P.rhs(terms=[...])``) is
ADDITIVE but, by default, NOT ENFORCED: the legacy string operator name and the
``flux=``/``sources=`` boolean/name spelling are still accepted. ``Program(strict_typed=True)`` (and
the ``POPS_STRICT_TYPED=1`` env override) opt into the ENFORCEMENT path the criteria ask for:

  - criterion 23: ``P.call("name", ...)`` with a STRING operator is REFUSED with a clear ``TypeError``
    naming the typed handle alternative; an ``OperatorHandle`` still resolves identically;
  - criterion 27: ``P.rhs(flux=True / sources=[...])`` (the legacy bool/name form, including a bare
    ``P.rhs(state=U)``) is REFUSED; ``P.rhs(terms=[Flux(), source])`` still works.

Strict mode only REJECTS a spelling -- it changes no lowering, so the typed path under strict ON
builds the BYTE-IDENTICAL IR (same ``Program._ir_hash``) as the same typed path under strict OFF.
Default OFF leaves every legacy form working (the existing suite stays green).

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
    from pops.time.program import _strict_typed_from_env
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_spec5_strict_typed (pops unavailable: %s)" % exc)
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


# The built-in default-Poisson field operator. A handle naming it resolves by .name (the kind on the
# handle is not consulted), so this is the strict-mode spelling of the historical P.call("fields_from
# _state", ...). Under strict OFF a plain string is equally accepted.
_FIELDS_OP = OperatorHandle("fields_from_state", kind="field_operator")


def _rate_program(m, selector, *, strict_typed=None):
    """A one-step predictor Program calling the rate operator via ``selector`` (a name or handle).

    @p strict_typed is forwarded to the Program constructor so the test can flip the mode. The field
    operator is named by a handle so the whole program is valid under strict ON; under strict OFF a
    string would work identically (proven by the byte-identity tests)."""
    P = adctime.Program("prog", strict_typed=strict_typed).bind_operators(m)
    U = P.state("plasma")
    f = P.call(_FIELDS_OP, U)
    R = P.call(selector, U, f)
    P.commit("plasma", P.linear_combine("u1", U + P.dt * R))
    return P


# --- the opt-in surface -------------------------------------------------------------------------

def test_default_is_permissive():
    """A plain Program() defaults to strict_typed OFF (the permissive, byte-identical default)."""
    P = adctime.Program("p")
    assert P._strict_typed is False
    assert "strict_typed" not in str(P)
    print("OK  default Program() is strict_typed=False (permissive)")


def test_explicit_strict_flag_surfaced_in_str():
    """Program(strict_typed=True) records the flag and surfaces it in __str__ (only when ON)."""
    on = adctime.Program("p", strict_typed=True)
    off = adctime.Program("p", strict_typed=False)
    assert on._strict_typed is True and off._strict_typed is False
    assert "strict_typed=True" in str(on)
    assert "strict_typed" not in str(off)
    print("OK  Program(strict_typed=True) surfaced in __str__; OFF stays one-line")


def test_env_override_enables_strict(monkeypatch):
    """POPS_STRICT_TYPED=1 enables strict mode when the explicit argument is left at its None default;
    an explicit argument always wins over the env."""
    monkeypatch.setenv("POPS_STRICT_TYPED", "1")
    assert _strict_typed_from_env() is True
    assert adctime.Program("p")._strict_typed is True            # env supplies the default
    assert adctime.Program("p", strict_typed=False)._strict_typed is False  # explicit wins
    monkeypatch.setenv("POPS_STRICT_TYPED", "0")
    assert _strict_typed_from_env() is False
    assert adctime.Program("p")._strict_typed is False
    monkeypatch.delenv("POPS_STRICT_TYPED", raising=False)
    assert adctime.Program("p")._strict_typed is False           # unset -> permissive
    print("OK  POPS_STRICT_TYPED env enables strict (truthy) / explicit arg wins")


# --- criterion 23: P.call requires a typed handle under strict -----------------------------------

def test_strict_rejects_string_operator():
    """Strict ON: P.call('name', ...) with a STRING operator is a clear TypeError naming the typed
    handle alternative (criterion 23). A field-operator NAME on the same Program is also refused."""
    m, _ = build_model()
    P = adctime.Program("p", strict_typed=True).bind_operators(m)
    U = P.state("plasma")
    with pytest.raises(TypeError, match="strict_typed: P.call requires a typed operator handle"):
        P.call("fields_from_state", U)
    print("OK  strict ON: P.call(str) -> TypeError naming the typed handle")


def test_strict_accepts_operator_handle():
    """Strict ON: an OperatorHandle still resolves (the typed path is the allowed spelling)."""
    m, h = build_model()
    prog = _rate_program(m, h["explicit_rhs"], strict_typed=True)
    assert prog is not None
    print("OK  strict ON: P.call(handle) accepted")


def test_call_handle_byte_identical_strict_on_vs_off():
    """The typed P.call(handle) path builds the BYTE-IDENTICAL IR under strict ON and OFF: strict
    mode rejects only the legacy spelling, it does not change lowering (criterion 23)."""
    m, h = build_model()
    off = _rate_program(m, h["explicit_rhs"], strict_typed=False)._ir_hash()
    on = _rate_program(m, h["explicit_rhs"], strict_typed=True)._ir_hash()
    # And the permissive STRING path lowers to the same IR (proving strict only refuses the spelling).
    legacy = _rate_program(m, "explicit_rhs", strict_typed=False)._ir_hash()
    assert on == off == legacy, (on, off, legacy)
    print("OK  P.call(handle) IR hash identical strict ON / OFF / legacy string: %s" % on)


# --- criterion 27: P.rhs requires terms= under strict --------------------------------------------

def _rhs_program(*, strict_typed, use_terms):
    """A one-block Euler Program whose single rhs is built either via terms= or the legacy form."""
    P = adctime.Program("rhs", strict_typed=strict_typed)
    U = P.state("plasma")
    f = P.solve_fields(U)
    if use_terms:
        R = P.rhs("R", state=U, fields=f, terms=[Flux(), "electric"])
    else:
        R = P.rhs("R", state=U, fields=f, flux=True, sources=["electric"])
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


def test_strict_rejects_legacy_flux_sources():
    """Strict ON: P.rhs(flux=True) / P.rhs(sources=[...]) / a bare P.rhs(state=U) are all refused
    with a clear TypeError naming P.rhs(terms=[...]) (criterion 27)."""
    m, _ = build_model()
    P = adctime.Program("p", strict_typed=True)
    U = P.state("plasma")
    f = P.solve_fields(U)
    msg = "strict_typed: P.rhs requires the typed terms="
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f, flux=True)
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f, sources=["electric"])
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f, fluxes=["default"])
    with pytest.raises(TypeError, match=msg):
        P.rhs("R", state=U, fields=f)  # the bare legacy default is the bool/name form too
    print("OK  strict ON: legacy P.rhs(flux=/sources=/fluxes=/bare) -> TypeError naming terms=")


def test_strict_accepts_terms():
    """Strict ON: P.rhs(terms=[Flux(), source]) is the allowed typed spelling."""
    P = _rhs_program(strict_typed=True, use_terms=True)
    assert P is not None
    print("OK  strict ON: P.rhs(terms=[Flux(), 'electric']) accepted")


def test_rhs_terms_byte_identical_strict_on_vs_off():
    """The typed terms= path builds the BYTE-IDENTICAL IR under strict ON and OFF, and equals the
    permissive legacy flux=/sources= form: strict only refuses the spelling (criterion 27)."""
    on = _rhs_program(strict_typed=True, use_terms=True)._ir_hash()
    off = _rhs_program(strict_typed=False, use_terms=True)._ir_hash()
    legacy = _rhs_program(strict_typed=False, use_terms=False)._ir_hash()
    assert on == off == legacy, (on, off, legacy)
    print("OK  P.rhs(terms=) IR hash identical strict ON / OFF / legacy flux=: %s" % on)


# --- the typed-call internal lowering is exempt from the rhs strict guard ------------------------

def test_strict_call_lowers_through_rhs():
    """A typed P.call rate operator lowers INTERNALLY through self.rhs(flux=...): strict mode must
    NOT mistake that internal lowering for a user's legacy P.rhs spelling. The rate program (which
    builds its rhs via P.call) must compile cleanly under strict ON and stay byte-identical."""
    m, h = build_model()
    on = _rate_program(m, h["explicit_rhs"], strict_typed=True)
    on.validate()
    off = _rate_program(m, h["explicit_rhs"], strict_typed=False)
    assert on._ir_hash() == off._ir_hash()
    print("OK  strict ON: P.call lowering through self.rhs(flux=...) is exempt + byte-identical")


# --- default OFF leaves every legacy form working ------------------------------------------------

def test_default_off_legacy_forms_still_work():
    """Default (strict OFF): the legacy string P.call and the flux=/sources= P.rhs both work, exactly
    as before this feature -- the no-false-positive guarantee for the existing suite."""
    m, _ = build_model()
    _rate_program(m, "explicit_rhs", strict_typed=None).validate()  # string P.call, no strict_typed
    _rhs_program(strict_typed=None, use_terms=False)                # legacy flux=/sources= P.rhs
    print("OK  default OFF: legacy string P.call + flux=/sources= P.rhs still work")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
