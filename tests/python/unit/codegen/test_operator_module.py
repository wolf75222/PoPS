"""The public operator-first Module surface is typed and string-selector free."""
from __future__ import annotations

import pops.model as model
from pops.params import RuntimeParam
from pops.time import Program


def test_signature_sugar_preserves_exact_typed_spaces() -> None:
    state = model.StateSpace("U", ("rho", "mx", "my"))
    fields = model.FieldSpace("fields", ("phi", "grad_x", "grad_y"))

    assert ((state, fields) >> model.Rate(state)) == model.Signature(
        (state, fields), model.Rate(state))
    assert (state >> fields) == model.Signature((state,), fields)
    assert ((fields,) >> model.LocalLinearOperator(state, state)) == model.Signature(
        (fields,), model.LocalLinearOperator(state, state))
    assert (() >> model.LocalLinearOperator(state, state)) == model.Signature(
        (), model.LocalLinearOperator(state, state))


def test_module_builders_return_exact_operator_handles() -> None:
    module = model.Module("euler-poisson-lorentz")
    state = module.state_space(
        "U", ("rho", "mx", "my"), roles={"rho": "Density"})
    fields = module.field_space("fields", ("phi", "grad_x", "grad_y"))
    parameters = module.parameters(
        RuntimeParam("alpha", default=1.0),
        RuntimeParam("cs2", default=0.0),
    )
    declarations = {
        handle.name: module.param_declaration(handle) for handle in parameters
    }
    aux = module.aux_fields(B_z="cell_scalar")

    field_provider = module.operator(
        name="fields_from_state",
        signature=(state,) >> fields,
        kind="field_operator",
        expr="<ir>",
    )

    @module.operator(
        name="explicit_rhs",
        signature=(state, fields) >> model.Rate(state),
        kind="local_rate",
    )
    def explicit_rhs(current, solved_fields):
        return current, solved_fields

    assert declarations["alpha"].default == 1.0
    assert aux["B_z"].kind == "cell_scalar"
    assert isinstance(field_provider, model.OperatorHandle)
    assert isinstance(explicit_rhs, model.OperatorHandle)
    assert module.operator_registry().names() == [
        "fields_from_state", "explicit_rhs"]
    assert module.operator_handle("fields_from_state") == field_provider


def test_program_operator_selection_requires_an_exact_handle() -> None:
    selector = "fields_from_state"
    program = Program("string-operator-refusal")
    try:
        program._call(selector)
        raise AssertionError("expected string operator selection to be refused")
    except TypeError as error:
        assert "exact OperatorHandle" in str(error)
        assert repr(selector) in str(error)
