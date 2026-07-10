"""Owner-safe resolution of every public typed time-operator route (ADC-652)."""
from __future__ import annotations

import pytest

from pops import time as adctime
from pops.model import (
    LocalLinearOperator, Module, OperatorHandle, RateSpace, Signature, StateSpace,
)
from pops.numerics.terms import Flux, SourceTerm
from pops.physics.facade import Model


def _model(name: str):
    model = Model(name)
    u, v = model.conservative_vars("u", "v")
    model.flux(x=[u, v], y=[u, v])
    source = model.source_term("shared_source", [-u, -v])
    linear = model.local_linear_map("shared_linear", [[-1, 0], [0, -1]])
    return model, source, linear


def _program(model: Model):
    program = adctime.Program("handles").bind_operators(model)
    return program, program.state("block")


def _rate_model(name: str):
    model = Model(name)
    u, v = model.conservative_vars("u", "v")
    model.flux(x=[u, v], y=[u, v])
    return model, model.rate("shared_rate", flux=True, sources=[])


def test_homonymous_foreign_handles_are_rejected_on_every_route():
    first, first_source, first_linear = _model("first")
    second, second_source, second_linear = _model("second")
    program, state = _program(first)

    # The same local names exist in both registries. Identity must survive until
    # the exact bound owner is checked; name lookup alone would accept all four.
    for build in (
        lambda: program.linear_source(second_linear),
        lambda: program.apply(second_linear, state=state),
        lambda: program.source(second_source, state=state),
        lambda: program.rhs(state=state, terms=[Flux(), second_source]),
        lambda: program.condensed_coeffs(
            state=state, linear_operator=second_linear, subset=(0, 1), c=1, th_dt=1),
    ):
        with pytest.raises(ValueError, match="belongs to owner"):
            build()

    # The handles from the bound model remain valid on the same public routes.
    assert program.linear_source(first_linear).attrs["linear_source"] == "shared_linear"
    assert program.apply(first_linear, state=state).attrs["linear_source"] == "shared_linear"
    assert program.source(first_source, state=state).attrs["source"] == "shared_source"
    assert program.rhs(state=state, terms=[first_source]).attrs["sources"] == ("shared_source",)
    assert program.condensed_coeffs(
        state=state, linear_operator=first_linear, subset=(0, 1), c=1,
        th_dt=1).attrs["linear_operator"] == "shared_linear"


def test_forged_kind_and_signature_are_rejected_before_lowering():
    model, source, linear = _model("forgery")
    program, state = _program(model)
    registry = model.operator_registry()
    declared = registry.get(linear.name)

    wrong_kind = OperatorHandle(
        linear.name, kind="local_source", owner=registry.owner_path,
        signature=declared.signature)
    with pytest.raises(ValueError, match="declares kind"):
        program.linear_source(wrong_kind)

    wrong_signature = OperatorHandle(
        linear.name, kind=declared.kind, owner=registry.owner_path,
        signature=registry.get(source.name).signature)
    for build in (
        lambda: program.call(wrong_signature),
        lambda: program.apply(wrong_signature, state=state),
        lambda: program.condensed_coeffs(
            state=state, linear_operator=wrong_signature, subset=(0, 1), c=1, th_dt=1),
    ):
        with pytest.raises(ValueError, match="carries signature"):
            build()


def test_forged_alias_target_cannot_route_to_another_compatible_operator():
    module = Module("alias-forgery")
    state_space = module.state_space("U", ("u",))
    first = module.operator(
        "first", kind="local_source",
        signature=(state_space,) >> RateSpace(state_space), expr="first")
    module.operator(
        "second", kind="local_source",
        signature=(state_space,) >> RateSpace(state_space), expr="second")
    registry = module.operator_registry()
    registry.register_alias("readable", "first")
    legitimate = module.operator_handle("readable")
    forged = OperatorHandle(
        "readable", kind=first.kind, owner=module.owner_path,
        signature=first.signature, registered_operator_name="second")

    assert legitimate != forged
    assert hash(legitimate) != hash(forged)
    assert legitimate.qualified_id != forged.qualified_id

    program = adctime.Program("alias-forgery").bind_operators(module)
    state = program.state("block", space=state_space)
    assert program.call(legitimate, state).attrs["source"] == "first"
    with pytest.raises(ValueError, match="authenticates target.*first"):
        program.call(forged, state)


def test_registry_aliases_are_collision_safe_and_cannot_be_retargeted():
    module = Module("alias-table")
    state = module.state_space("U", ("u",))
    module.operator(
        "first", kind="local_source",
        signature=(state,) >> RateSpace(state), expr="first")
    module.operator(
        "second", kind="local_source",
        signature=(state,) >> RateSpace(state), expr="second")
    registry = module.operator_registry()

    assert registry.register_alias("readable", "first") == "readable"
    assert registry.register_alias("readable", "first") == "readable"
    with pytest.raises(ValueError, match="cannot retarget"):
        registry.register_alias("readable", "second")
    with pytest.raises(ValueError, match="collides with a registered operator"):
        registry.register_alias("first", "second")
    with pytest.raises(ValueError, match="collides with registered alias"):
        module.operator(
            "readable", kind="local_source",
            signature=(state,) >> RateSpace(state), expr="collision")


def test_typed_rhs_descriptors_resolve_presence_and_kind_and_strings_are_private():
    model, source, linear = _model("rhs")
    program, state = _program(model)

    assert program.rhs(
        state=state, terms=[SourceTerm(source.name)]).attrs["sources"] == (source.name,)
    with pytest.raises(KeyError, match="unknown operator"):
        program.rhs(state=state, terms=[SourceTerm("missing")])
    with pytest.raises(ValueError, match="expected one of"):
        program.rhs(state=state, terms=[SourceTerm(linear.name)])
    with pytest.raises(TypeError, match="free source name"):
        program.rhs(state=state, terms=[source.name])


def test_readable_default_source_alias_has_an_explicit_registered_target():
    model = Model("default-source")
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[u])
    source = model.source_term("default", [-u])
    program = adctime.Program("default-source").bind_operators(model)
    state = program.state("block")

    assert source.name == "default"
    assert source.registered_operator_name == "source_default"
    standalone = program.source(source, state=state)
    assert standalone.op == "rhs" and standalone.attrs["sources"] == ("default",)
    assert program.rhs(state=state, terms=[source]).attrs["sources"] == ("default",)


def test_public_handle_routes_require_a_bound_registry():
    model, source, linear = _model("unbound")
    program = adctime.Program("unbound")
    state = program.state("block")

    for build in (
        lambda: program.linear_source(linear),
        lambda: program.apply(linear, state=state),
        lambda: program.source(source, state=state),
        lambda: program.rhs(state=state, terms=[source]),
        lambda: program.condensed_coeffs(
            state=state, linear_operator=linear, subset=(0, 1), c=1, th_dt=1),
    ):
        with pytest.raises(ValueError, match="no operators are bound"):
            build()


def test_lib_time_helpers_preserve_handle_identity_until_resolution():
    from pops.lib import time as libtime

    first, first_rate = _rate_model("macro-first")
    _, foreign_rate = _rate_model("macro-second")
    program = adctime.Program("macro").bind_operators(first)
    with pytest.raises(ValueError, match="belongs to owner"):
        libtime.explicit_rk(
            program, "block", rhs_operator=foreign_rate,
            tableau=libtime.SSPRK2_TABLEAU)

    valid = adctime.Program("macro").bind_operators(first)
    libtime.explicit_rk(
        valid, "block", rhs_operator=first_rate,
        tableau=libtime.SSPRK2_TABLEAU)
    assert valid.validate() is True


def test_public_operator_routes_reject_free_string_selectors():
    model, source, linear = _model("strings")
    program, state = _program(model)

    for build in (
        lambda: program.linear_source(linear.name),
        lambda: program.apply(linear.name, state=state),
        lambda: program.source(source.name, state=state),
        lambda: program.rhs(state=state, terms=[source.name]),
        lambda: program.condensed_coeffs(
            state=state, linear_operator=linear.name, subset=(0, 1), c=1, th_dt=1),
    ):
        with pytest.raises(TypeError, match="free|string|OperatorHandle"):
            build()


def test_local_linear_state_compatibility_is_structural_not_name_only():
    module = Module("spaces")
    operator_space = module.state_space("U", ("a", "b"))
    operator = module.operator(
        name="L", kind="local_linear_operator",
        signature=Signature((), LocalLinearOperator(operator_space, operator_space)),
        expr=object())
    # One rate declaration makes the registry expose the state input too; the
    # explicit state below still uses a deliberately different same-named space.
    module.operator(
        name="R", kind="local_rate",
        signature=Signature((operator_space,), RateSpace(operator_space)), expr=object())
    handle = OperatorHandle(
        operator.name, kind=operator.kind, owner=module.owner_path,
        signature=operator.signature)
    program = adctime.Program("spaces").bind_operators(module)
    same_name_different_shape = StateSpace("U", ("x", "y"))
    state = program.state("block", space=same_name_different_shape)
    linear = program.call(handle)

    with pytest.raises(ValueError, match="structural, not name-based"):
        program.apply(linear, state=state)


@pytest.mark.parametrize(
    ("name", "block", "message"),
    [
        ("U", True, "block must be a non-empty string"),
        ("U", "", "block must be a non-empty string"),
        (True, "block", "name must be a non-empty string"),
        ("", "block", "name must be a non-empty string"),
    ],
)
def test_invalid_state_identity_does_not_leave_partial_space_declarations(name, block, message):
    program = adctime.Program("invalid-state")
    spaces_before = dict(program._state_spaces)
    values_before = tuple(program._values)
    with pytest.raises(ValueError, match=message):
        program.state(name, block=block)
    assert program._state_spaces == spaces_before
    assert tuple(program._values) == values_before
