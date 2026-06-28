"""P._rhs_legacy(terms=[...]) typed RHS composition (Spec 5 sec.14.2.4, ADC-479 criterion 27).

The typed ``terms=`` front door -- the ONE public RHS path -- lowers onto the INTERNAL
``P._rhs_legacy`` ``flux=``/``sources=`` builder: each :class:`pops.numerics.terms.Flux`/source term
maps onto the existing booleans/name-list, so the built IR is BYTE-IDENTICAL to the private path.
These tests pin that equivalence on the ``Program._ir_hash``:

  - ``terms=[Flux(), <source>]`` builds the byte-identical hash to the private
    ``_rhs_legacy(flux=True, sources=[<name>])``;
  - ``Flux()`` is a typed term, not a bool (a bare bool in terms= is a TypeError);
  - every accepted source form (name str / SourceTerm / OperatorHandle) maps onto the same name;
  - a non-term object in terms= is a clear TypeError.

Pure Python; no compilation, no ``_pops``. Run with python3 (PYTHONPATH = built pops package).
"""
import pytest

from pops import time as adctime
from pops.model import OperatorHandle
from pops.numerics.terms import Flux, LocalTerm, SourceTerm


def _terms_program(terms):
    """A one-block forward-Euler Program whose single rhs is built from ``terms=``."""
    P = adctime.Program("rhs_terms")
    dt = P.dt
    U = P.state("plasma")
    f = P._solve_fields(U)
    R = P._rhs_terms("R", state=U, fields=f, terms=terms)
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    P.validate()
    return P


def _legacy_program(flux, sources):
    """The same Program built through the INTERNAL ``_rhs_legacy`` flux=/sources= builder (the typed
    terms= path lowers onto this private builder; it is the byte-identity target, not a public path)."""
    P = adctime.Program("rhs_terms")
    dt = P.dt
    U = P.state("plasma")
    f = P._solve_fields(U)
    R = P._rhs_legacy(name="R", state=U, fields=f, flux=flux, sources=sources)
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    P.validate()
    return P


def test_terms_flux_plus_source_is_byte_identical():
    """terms=[Flux(), "electric"] == _rhs_legacy(flux=True, sources=["electric"]) (same _ir_hash)."""
    h_terms = _terms_program([Flux(), "electric"])._ir_hash()
    h_legacy = _legacy_program(True, ["electric"])._ir_hash()
    assert h_terms == h_legacy, (h_terms, h_legacy)
    print("OK  1. terms=[Flux(), 'electric'] _ir_hash == _rhs_legacy(flux=True, sources=['electric'])")


def test_terms_flux_only_is_byte_identical():
    """terms=[Flux()] (no source) == _rhs_legacy(flux=True, sources=[]) (flux only)."""
    assert _terms_program([Flux()])._ir_hash() == _legacy_program(True, [])._ir_hash()
    print("OK  2. terms=[Flux()] _ir_hash == _rhs_legacy(flux=True, sources=[])")


def test_terms_source_only_is_byte_identical():
    """terms=["electric"] (no Flux) == _rhs_legacy(flux=False, sources=["electric"]) (source only)."""
    assert _terms_program(["electric"])._ir_hash() == _legacy_program(False, ["electric"])._ir_hash()
    print("OK  3. terms=['electric'] _ir_hash == _rhs_legacy(flux=False, sources=['electric'])")


def test_source_forms_map_to_same_name():
    """Every accepted source form (name str / SourceTerm / OperatorHandle) folds in the SAME
    source name, so all three build the byte-identical IR."""
    h_str = _terms_program([Flux(), "electric"])._ir_hash()
    h_srcterm = _terms_program([Flux(), SourceTerm("electric")])._ir_hash()
    h_handle = _terms_program([Flux(), OperatorHandle("electric", kind="local_source")])._ir_hash()
    h_local = _terms_program([Flux(), LocalTerm("electric")])._ir_hash()
    assert h_str == h_srcterm == h_handle == h_local, (h_str, h_srcterm, h_handle, h_local)
    print("OK  5. source forms (str/SourceTerm/OperatorHandle/LocalTerm) -> same name -> same hash")


def test_flux_is_a_term_not_a_bool():
    """Flux() is a typed term (sets flux=True), and a bare bool in terms= is rejected: the spec
    distinguishes a Flux term from a flux boolean."""
    # Flux() lowers to flux=True (proven by the byte-identical hash above); a bare True does not.
    assert _terms_program([Flux()])._ir_hash() != _terms_program([])._ir_hash()
    P = adctime.Program("rhs_terms")
    U = P.state("plasma")
    f = P._solve_fields(U)
    with pytest.raises(TypeError):
        P._rhs_terms("R", state=U, fields=f, terms=[True])
    print("OK  6. Flux() is a term not a bool; a bare bool in terms= is a TypeError")


def test_legacy_flux_sources_rejected_in_public_surface():
    """The legacy flux=/sources=/fluxes= form (and a bare P.rhs) is NOT a public path: it is refused
    with a clear TypeError naming terms= (Spec 5: terms= is the one public RHS path)."""
    P = adctime.Program("rhs_terms")
    U = P.state("plasma")
    f = P._solve_fields(U)
    for kw in ({"flux": True}, {"sources": ["electric"]}, {"fluxes": ["default"]}, {}):
        with pytest.raises(TypeError, match="_rhs_terms requires terms="):
            P._rhs_terms("R", state=U, fields=f, **kw)
    print("OK  7. legacy P._rhs_legacy(flux=/sources=/fluxes=/bare) -> TypeError naming terms=")


def test_bad_term_raises_typeerror():
    """A non-term object in terms= is a clear TypeError (transparent typed surface)."""
    P = adctime.Program("rhs_terms")
    U = P.state("plasma")
    f = P._solve_fields(U)
    for bad in (123, 4.5, object(), ["nested"]):
        with pytest.raises(TypeError):
            P._rhs_terms("R", state=U, fields=f, terms=[Flux(), bad])
    # An unnamed SourceTerm/LocalTerm has no declared source name to fold in.
    for unnamed in (SourceTerm(), LocalTerm()):
        with pytest.raises(ValueError, match="must be named"):
            P._rhs_terms("R", state=U, fields=f, terms=[Flux(), unnamed])
    print("OK  8. a non-term in terms= -> TypeError; an unnamed source term -> ValueError")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
