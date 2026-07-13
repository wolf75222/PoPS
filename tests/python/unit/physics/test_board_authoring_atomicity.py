"""ADC-652: strict board inputs and all-or-nothing declarations."""
from __future__ import annotations

import pytest

from pops import model as typed_model
from pops.fields import FieldOutput
from pops.math import ddt, div, laplacian
from pops.model import OwnerPath
from pops.physics import Model
from pops.physics.board_handles import (FieldHandle, FieldsHandle, FluxHandle, SourceHandle,
                                        StateHandle, _safe_name)
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS


def _expr_lists(mapping):
    return tuple(sorted((key, tuple(repr(value) for value in values))
                        for key, values in mapping.items()))


def _snapshot(model):
    """JSON-like authoring state; no symbolic ``==`` is invoked by comparisons."""
    hyp = model._dsl._m
    module = model._multi_module
    return {
        "states": tuple(sorted((key, value.qualified_id) for key, value in model._states.items())),
        "species": tuple(sorted((key, value.qualified_id)
                                for key, value in model._species.items())),
        "fields": tuple(sorted((key, value.qualified_id) for key, value in model._fields.items())),
        "fluxes": tuple(sorted((key, value.qualified_id) for key, value in model._fluxes.items())),
        "sources": tuple(sorted((key, value.qualified_id) for key, value in model._sources.items())),
        "operators": tuple(sorted(model._operators)),
        "field_operators": tuple(sorted(model._field_operators)),
        "cons_names": tuple(hyp.cons_names),
        "cons_roles": None if hyp.cons_roles is None else tuple(hyp.cons_roles),
        "n_vars": hyp.n_vars,
        "aux_names": tuple(hyp.aux_names),
        "aux_extra_names": tuple(hyp.aux_extra_names),
        "flux": _expr_lists(hyp._flux),
        "eigenvalues": _expr_lists(hyp._eig),
        "sources_ir": _expr_lists(hyp._source_terms),
        "linear_sources": tuple(sorted((key, repr(value))
                                       for key, value in hyp._linear_sources.items())),
        "elliptic": None if hyp._elliptic is None else repr(hyp._elliptic),
        "elliptic_fields": tuple(sorted((key, repr(value))
                                         for key, value in hyp._elliptic_fields.items())),
        "rates": tuple(sorted((key, repr(value)) for key, value in hyp._rate_operators.items())),
        "riemann": repr(model._riemann),
        "reconstruction": repr(model._reconstruction),
        "multi_module": None if module is None else {
            "states": tuple(sorted((key, repr(value))
                                   for key, value in module.state_spaces().items())),
            "fields": tuple(sorted((key, repr(value))
                                   for key, value in module.field_spaces().items())),
            "operators": tuple(module.list_operators()),
        },
    }


def _scalar(name="scalar"):
    model = Model(name, frame=FRAME)
    state = model.state("U", components=["u"])
    return model, state, state[0]


def _scalar_flux(model, state, value, *, waves=None):
    return model.flux(
        "F",
        frame=FRAME,
        state=state,
        components={X_AXIS: [value], Y_AXIS: [value]},
        waves=(None if waves is None else {X_AXIS: waves["x"], Y_AXIS: waves["y"]}),
    )


def test_handle_constructors_never_coerce_names_or_boolean_flags():
    owner = OwnerPath.descriptor("board-strict")
    with pytest.raises(TypeError, match="non-empty string"):
        _safe_name(object())
    with pytest.raises(ValueError, match="non-empty string"):
        FieldHandle("", owner=owner)
    with pytest.raises(TypeError, match="must be bool"):
        FluxHandle("F", is_default="false", owner=owner)
    with pytest.raises(TypeError, match="non-empty string"):
        SourceHandle("S", object(), owner=owner)
    with pytest.raises(TypeError, match="non-empty string"):
        FieldsHandle(object(), outputs={}, owner=owner)
    with pytest.raises(TypeError, match="component"):
        StateHandle("U", [object()], [1], None, owner=owner)


@pytest.mark.parametrize(
    "name, components, error",
    [("", ["u"], ValueError), ("U", ["u", object()], TypeError),
     ("U", ["u", "u"], ValueError)],
)
def test_invalid_state_is_observationally_atomic(name, components, error):
    model = Model("state_atomic")
    before = _snapshot(model)

    with pytest.raises(error):
        model.state(name, components=components)

    assert _snapshot(model) == before
    assert model._dsl._m.cons_names == [] and model._dsl._m.n_vars == 0


def test_foreign_flux_state_is_rejected_before_any_flux_mutation():
    model, state, u = _scalar("local_flux")
    foreign, foreign_state, _ = _scalar("foreign_flux")
    before = _snapshot(model)

    with pytest.raises(ValueError, match="declared by this physics model"):
        _scalar_flux(model, foreign_state, u)

    assert _snapshot(model) == before


def test_flux_builder_failure_restores_flux_and_wave_registries(monkeypatch):
    model, state, u = _scalar("flux_builder")
    before = _snapshot(model)

    def fail_after_mutation(x, y):
        model._dsl._m._eig = {"x": list(x), "y": list(y)}
        raise RuntimeError("injected eigenvalue builder failure")

    monkeypatch.setattr(model._dsl, "eigenvalues", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        _scalar_flux(model, state, u, waves={"x": [1], "y": [1]})

    assert _snapshot(model) == before


def test_source_builder_failure_restores_source_registries(monkeypatch):
    model, state, u = _scalar("source_builder")
    before = _snapshot(model)

    def fail_after_mutation(name, expressions):
        model._dsl._m._source_terms[name] = list(expressions)
        raise RuntimeError("injected source builder failure")

    monkeypatch.setattr(model._dsl, "source_term", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        model.source("forcing", on=state, value=[u])

    assert _snapshot(model) == before


def test_field_operator_builder_failure_is_observationally_atomic(monkeypatch):
    model, _state, u = _scalar("elliptic_builder")
    phi = model.field("phi")
    before = _snapshot(model)

    def fail_after_mutation(name, rhs, operator="poisson", aux=None):
        model._dsl._m._elliptic_fields[name] = {
            "rhs": rhs, "operator": operator, "aux": aux,
        }
        raise RuntimeError("injected elliptic builder failure")

    monkeypatch.setattr(model._dsl, "elliptic_field", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        model.field_operator(
            "poisson", unknown=phi, equation=(-laplacian(phi) == u),
            outputs=(FieldOutput("potential", phi),))

    assert _snapshot(model) == before


def test_invalid_field_operator_output_and_foreign_unknown_are_atomic():
    model, _state, u = _scalar("field_local")
    phi = model.field("phi")
    foreign, _foreign_state, foreign_u = _scalar("field_foreign")
    foreign_phi = foreign.field("phi")
    before = _snapshot(model)

    with pytest.raises(ValueError, match="outputs must start with FieldOutput"):
        model.field_operator(
            "bad_outputs", unknown=phi, equation=(-laplacian(phi) == u),
            outputs=(object(),))
    assert _snapshot(model) == before

    with pytest.raises(ValueError, match="not declared by this physics model"):
        model.field_operator(
            "foreign", unknown=foreign_phi,
            equation=(-laplacian(foreign_phi) == foreign_u),
            outputs=(FieldOutput("phi", foreign_phi),))
    assert _snapshot(model) == before


def test_failed_finite_volume_rate_never_publishes_reconstruction(monkeypatch):
    model, state, u = _scalar("rate_builder")
    flux = _scalar_flux(model, state, u)
    marker = object()
    before = _snapshot(model)

    def fail_after_mutation(name, **kwargs):
        model._dsl._m._rate_operators[name] = dict(kwargs)
        raise RuntimeError("injected rate builder failure")

    monkeypatch.setattr(model._dsl, "rate_operator", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        model.finite_volume_rate("A", flux=flux, reconstruction=marker)

    assert _snapshot(model) == before
    assert model._reconstruction is None


def test_failed_second_species_promotion_keeps_single_species_state(monkeypatch):
    model = Model("promotion")
    first = model.species("electrons", state=["ne"])
    before = _snapshot(model)

    def fail_state_space(candidate, *args, **kwargs):
        candidate._state_spaces["leak"] = object()
        raise RuntimeError("injected state-space builder failure")

    monkeypatch.setattr(typed_model.Module, "state_space", fail_state_space)
    with pytest.raises(RuntimeError, match="injected"):
        model.species("ions", state=["ni"])

    assert _snapshot(model) == before
    assert model._multi_module is None and model._species["electrons"] is first


def test_later_species_space_builder_failure_restores_existing_module(monkeypatch):
    model = Model("later_species")
    model.species("electrons", state=["ne"])
    model.species("ions", state=["ni"])
    module = model._multi_module
    before = _snapshot(model)

    def fail_after_space_mutation(*args, **kwargs):
        module._state_spaces["leak"] = object()
        raise RuntimeError("injected later state-space failure")

    monkeypatch.setattr(module, "state_space", fail_after_space_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        model.species("neutrals", state=["nn"])

    assert _snapshot(model) == before


def test_coupled_rate_builder_failure_restores_operator_registry(monkeypatch):
    model = Model("coupled_builder")
    electrons = model.species("electrons", state=["ne"])
    ions = model.species("ions", state=["ni"])
    before = _snapshot(model)
    original = model._multi_module.operator

    def fail_after_registration(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected coupled operator failure")

    monkeypatch.setattr(model._multi_module, "operator", fail_after_registration)
    with pytest.raises(RuntimeError, match="injected"):
        model.coupled_rate(
            "collision", inputs=[electrons, ions],
            outputs={electrons: [ions["ni"]], ions: [electrons["ne"]]})

    assert _snapshot(model) == before


def test_foreign_multispecies_handle_does_not_register_coupled_operator():
    model = Model("multi_local")
    electrons = model.species("electrons", state=["ne"])
    ions = model.species("ions", state=["ni"])
    foreign = Model("multi_foreign")
    foreign_e = foreign.species("electrons", state=["ne"])
    foreign.species("ions", state=["ni"])
    before = _snapshot(model)

    with pytest.raises(ValueError, match="another physics model"):
        model.coupled_rate(
            "collision", inputs=[electrons, foreign_e],
            outputs={electrons: [ions["ni"]]})

    assert _snapshot(model) == before


def test_rate_equation_with_foreign_flux_is_atomic():
    model, state, _ = _scalar("rate_local")
    foreign, foreign_state, foreign_u = _scalar("rate_foreign")
    foreign_flux = _scalar_flux(foreign, foreign_state, foreign_u)
    before = _snapshot(model)

    with pytest.raises(ValueError, match="declared by this physics model"):
        model.rate("A", ddt(state) == -div(foreign_flux))

    assert _snapshot(model) == before
