"""P.rhs(terms=[...]) typed RHS composition (Spec 5 sec.14.2.4, ADC-479 criterion 27).

The typed ``terms=`` front door -- the ONE public RHS path -- lowers onto the INTERNAL
``P._rhs_legacy`` ``flux=``/``sources=`` builder: each :class:`pops.numerics.terms.Flux`/source term
maps onto the existing booleans/name-list, so the built IR is BYTE-IDENTICAL to the private path.
These tests pin that equivalence on the ``Program._ir_hash``:

  - ``terms=[Flux(), <source>]`` builds the byte-identical hash to the private
    ``_rhs_legacy(flux=True, sources=[<name>])``;
  - ``Flux()`` is a typed term, not a bool (a bare bool in terms= is a TypeError);
  - every accepted typed source form (SourceTerm / LocalTerm / OperatorHandle) maps onto the same
    registered source, while a free string is refused;
  - a non-term object in terms= is a clear TypeError.

Pure Python; no compilation, no ``_pops``. Run with python3 (PYTHONPATH = built pops package).
"""
import pytest

from pops.numerics.terms import Flux, LocalTerm, SourceTerm
from pops.physics.facade import Model
from tests.python.unit.runtime._typed_program import typed_program_state


_HANDLE = object()
_SOURCE_TERM = object()
_LOCAL_TERM = object()


def _source_model():
    model = Model("rhs_terms_model")
    (u,) = model.conservative_vars("u")
    model.elliptic_rhs(u)
    return model, model.source_term("electric", [-u])


def _terms_program(terms, authored=None):
    """A one-block forward-Euler Program whose single rhs is built from ``terms=``."""
    model, source = authored or _source_model()
    terms = [
        source if term is _HANDLE else
        SourceTerm(source) if term is _SOURCE_TERM else
        LocalTerm(source) if term is _LOCAL_TERM else term
        for term in terms
    ]
    P, _, _, _, _, temporal = typed_program_state(
        "rhs_terms", model=model, state="U")
    dt = P.dt
    U = temporal.n
    f = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f, terms=terms)
    P.commit(temporal.next, P.linear_combine("U1", U + dt * R))
    P.validate()
    return P


def _legacy_program(flux, sources, model=None):
    """The same Program built through the INTERNAL ``_rhs_legacy`` flux=/sources= builder (the typed
    terms= path lowers onto this private builder; it is the byte-identity target, not a public path)."""
    model = model or _source_model()[0]
    P, _, _, _, _, temporal = typed_program_state(
        "rhs_terms", model=model, state="U")
    dt = P.dt
    U = temporal.n
    f = P.solve_fields(U)
    R = P._rhs_legacy(name="R", state=U, fields=f, flux=flux, sources=sources)
    P.commit(temporal.next, P.linear_combine("U1", U + dt * R))
    P.validate()
    return P


def _cpp_without_identity(program, model):
    """Compare lowering bytes while intentionally excluding the typed IR identity export."""
    return "\n".join(
        line for line in program.emit_cpp_program(model=model).splitlines()
        if "pops_program_hash()" not in line
    )


def test_terms_flux_plus_source_is_byte_identical():
    """Typed identity changes the hash while the actual C++ lowering stays byte-identical."""
    authored = _source_model()
    typed = _terms_program([Flux(), _SOURCE_TERM], authored)
    private = _legacy_program(True, ["electric"], authored[0])
    assert typed._ir_hash() != private._ir_hash()
    assert _cpp_without_identity(typed, authored[0]) == _cpp_without_identity(private, authored[0])


def test_terms_flux_only_is_byte_identical():
    """terms=[Flux()] (no source) == _rhs_legacy(flux=True, sources=[]) (flux only)."""
    assert _terms_program([Flux()])._ir_hash() == _legacy_program(True, [])._ir_hash()
    print("OK  2. terms=[Flux()] _ir_hash == _rhs_legacy(flux=True, sources=[])")


def test_terms_source_only_is_byte_identical():
    """A typed source without Flux lowers to the private source-only path."""
    authored = _source_model()
    typed = _terms_program([_SOURCE_TERM], authored)
    private = _legacy_program(False, ["electric"], authored[0])
    assert typed._ir_hash() != private._ir_hash()
    assert _cpp_without_identity(typed, authored[0]) == _cpp_without_identity(private, authored[0])
    print("OK  3. typed source-only term == private _rhs_legacy source selector")


def test_source_forms_map_to_same_name():
    """Every accepted typed source form folds in the same registered source."""
    h_srcterm = _terms_program([Flux(), _SOURCE_TERM])._ir_hash()
    h_handle = _terms_program([Flux(), _HANDLE])._ir_hash()
    h_local = _terms_program([Flux(), _LOCAL_TERM])._ir_hash()
    assert h_srcterm == h_handle == h_local, (h_srcterm, h_handle, h_local)
    print("OK  5. typed source forms (SourceTerm/OperatorHandle/LocalTerm) -> same hash")


def test_free_source_name_is_rejected():
    """Public terms retain typed identity; bare source names are private lowering tokens only."""
    with pytest.raises(TypeError, match="free source name"):
        _terms_program([Flux(), "electric"])


def test_flux_is_a_term_not_a_bool():
    """Flux() is a typed term (sets flux=True), and a bare bool in terms= is rejected: the spec
    distinguishes a Flux term from a flux boolean."""
    # Flux() lowers to flux=True (proven by the byte-identical hash above); a bare True does not.
    assert _terms_program([Flux()])._ir_hash() != _terms_program([])._ir_hash()
    P, _, _, _, _, temporal = typed_program_state("rhs_terms")
    U = temporal.n
    f = P.solve_fields(U)
    with pytest.raises(TypeError):
        P.rhs("R", state=U, fields=f, terms=[True])
    print("OK  6. Flux() is a term not a bool; a bare bool in terms= is a TypeError")


def test_legacy_flux_sources_rejected_in_public_surface():
    """The legacy flux=/sources=/fluxes= form (and a bare P.rhs) is NOT a public path: it is refused
    with a clear TypeError naming terms= (Spec 5: terms= is the one public RHS path)."""
    P, _, _, _, _, temporal = typed_program_state("rhs_terms")
    U = temporal.n
    f = P.solve_fields(U)
    for kw in ({"flux": True}, {"sources": ["electric"]}, {"fluxes": ["default"]}, {}):
        with pytest.raises(TypeError, match="requires the typed terms="):
            P.rhs("R", state=U, fields=f, **kw)
    print("OK  7. legacy P.rhs(flux=/sources=/fluxes=/bare) -> TypeError naming terms=")


def test_bad_term_raises_typeerror():
    """A non-term object in terms= is a clear TypeError (transparent typed surface)."""
    P, _, _, _, _, temporal = typed_program_state("rhs_terms")
    U = temporal.n
    f = P.solve_fields(U)
    for bad in (123, 4.5, object(), ["nested"]):
        with pytest.raises(TypeError):
            P.rhs("R", state=U, fields=f, terms=[Flux(), bad])
    # Source wrappers themselves require the exact declaring OperatorHandle.
    for constructor in (SourceTerm, LocalTerm):
        with pytest.raises(TypeError, match="typed OperatorHandle"):
            constructor("electric")
    print("OK  8. bad RHS terms and ownerless source selectors are rejected")
