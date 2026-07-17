"""The sole public RHS route composes exact typed terms."""
import pytest

from pops.numerics.terms import DefaultSource, Flux, LocalTerm, SourceTerm
from pops.physics._facade import Model
from tests.python.unit.runtime._typed_program import solve_field, typed_program_state


_HANDLE = object()
_SOURCE_TERM = object()
_LOCAL_TERM = object()


def _source_model():
    model = Model("rhs_terms_model")
    (u,) = model.conservative_vars("u")
    model.elliptic_rhs(u)
    return model, model.source_term("electric", [-u])


def _terms_program(terms, authored=None):
    model, source = authored or _source_model()
    terms = [
        source if term is _HANDLE else
        SourceTerm(source) if term is _SOURCE_TERM else
        LocalTerm(source) if term is _LOCAL_TERM else term
        for term in terms
    ]
    program, _, _, _, _, temporal = typed_program_state(
        "rhs_terms", model=model, state="U")
    fields = solve_field(program, temporal.n)
    rate = program.rhs("R", state=temporal.n, fields=fields, terms=terms)
    program.commit(
        temporal.next,
        program.value("U1", temporal.n + program.dt * rate, at=temporal.next.point),
    )
    assert program.validate() is True
    return program, rate


def test_flux_and_typed_source_retain_exact_public_choices():
    _, rate = _terms_program([Flux(), _SOURCE_TERM])
    assert rate.attrs["flux"] is True
    assert rate.attrs["sources"] == ("electric",)
    assert rate.attrs["source_handles"][0].local_id == "electric"


def test_flux_default_and_source_only_routes_are_explicit():
    _, flux = _terms_program([Flux()])
    _, default = _terms_program([DefaultSource()])
    _, source = _terms_program([_SOURCE_TERM])
    assert flux.attrs["flux"] is True and flux.attrs["sources"] == ()
    assert default.attrs["flux"] is False and default.attrs["sources"] == ("default",)
    assert source.attrs["flux"] is False and source.attrs["sources"] == ("electric",)


def test_source_wrappers_and_handle_have_one_semantic_identity():
    handle = _terms_program([Flux(), _HANDLE])[0]._ir_hash()
    source = _terms_program([Flux(), _SOURCE_TERM])[0]._ir_hash()
    local = _terms_program([Flux(), _LOCAL_TERM])[0]._ir_hash()
    assert handle == source == local


def test_free_source_name_boolean_and_bad_terms_are_rejected():
    for bad in ("electric", True, 123, 4.5, object(), ["nested"]):
        with pytest.raises(TypeError):
            _terms_program([Flux(), bad])
    for constructor in (SourceTerm, LocalTerm):
        with pytest.raises(TypeError, match="typed OperatorHandle"):
            constructor("electric")


def test_rhs_requires_explicit_terms_keyword():
    program, _, _, _, _, temporal = typed_program_state("rhs_terms_required")
    with pytest.raises(TypeError, match="terms"):
        program.rhs(state=temporal.n)
