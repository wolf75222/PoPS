"""Owner-safe resolution of every public typed time-operator route (ADC-652)."""
from __future__ import annotations

import pytest

from pops import time as adctime
from pops.model import (
    DoubleOwnershipError, Handle, LocalLinearOperator, Module, OperatorHandle,
    RateSpace, Signature,
)
from pops.numerics.terms import Flux, SourceTerm
from pops.physics.facade import Model
from pops.problem import Case


def _model(name: str):
    model = Model(name)
    u, v = model.conservative_vars("u", "v")
    model.flux(x=[u, v], y=[u, v])
    source = model.source_term("shared_source", [-u, -v])
    linear = model.local_linear_map("shared_linear", [[-1, 0], [0, -1]])
    return model, source, linear


def _references(model: Model | Module, *, case_name: str | None = None):
    """Return the exact block/state declarations consumed by ``Program.state``.

    A facade's Module is the authoritative declaration provider: the state handle
    and its StateSpace come from that Module, while the Case qualifies the
    declaration into one concrete block instance.
    """
    module = model.module if isinstance(model, Model) else model
    state_spaces = module.state_spaces()
    assert tuple(state_spaces) == ("U",)
    state = module.state_handle(state_spaces["U"])
    case = Case(name=case_name or "%s-case" % module.name)
    block = case.block("block", module)
    return module, block, state


def _program(model: Model):
    module, block, state = _references(model)
    program = adctime.Program("handles")._bind_operators(module)
    return program, program.state(block, state).n


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
        with pytest.raises(ValueError, match="no operator registry is bound for owner"):
            build()

    # The handles from the bound model remain valid on the same public routes.
    linear_value = program.linear_source(first_linear)
    assert linear_value.attrs["linear_source"] == "shared_linear"
    assert linear_value.attrs["operator_handle"] is first_linear
    applied = program.apply(first_linear, state=state)
    assert applied.attrs["linear_source"] == "shared_linear"
    assert applied.attrs["operator_handle"] is first_linear
    sourced = program.source(first_source, state=state)
    assert sourced.attrs["source"] == "shared_source"
    assert sourced.attrs["operator_handle"] is first_source
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
        lambda: program._call(wrong_signature),
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

    _, block, state_declaration = _references(module)
    program = adctime.Program("alias-forgery")._bind_operators(module)
    state = program.state(block, state_declaration).n
    assert legitimate(state).attrs["source"] == "first"
    with pytest.raises(ValueError, match="authenticates target.*first"):
        forged(state)


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
    with pytest.raises(ValueError, match="register-once"):
        registry.register_alias("readable", "first")
    with pytest.raises(ValueError, match="register-once"):
        registry.register_alias("readable", "second")
    assert registry.target_for_handle("readable") == "first"
    assert registry.target_for_handle("readable") == "first"
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
        state=state, terms=[SourceTerm(source)]).attrs["sources"] == (source.name,)
    missing = OperatorHandle(
        "missing", kind=source.kind, owner=source.owner_path,
        signature=source.signature)
    with pytest.raises(KeyError, match="unknown operator"):
        program.rhs(state=state, terms=[SourceTerm(missing)])
    with pytest.raises(ValueError, match="expected one of"):
        program.rhs(state=state, terms=[SourceTerm(linear)])
    with pytest.raises(TypeError, match="typed OperatorHandle"):
        SourceTerm(source.name)
    with pytest.raises(TypeError, match="free source name"):
        program.rhs(state=state, terms=[source.name])


def test_readable_default_source_alias_has_an_explicit_registered_target():
    model = Model("default-source")
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[u])
    source = model.source_term("default", [-u])
    module, block, state_declaration = _references(model)
    program = adctime.Program("default-source")._bind_operators(module)
    state = program.state(block, state_declaration).n

    assert source.name == "default"
    assert source.registered_operator_name == "source_default"
    standalone = program.source(source, state=state)
    assert standalone.op == "rhs" and standalone.attrs["sources"] == ("default",)
    assert program.rhs(state=state, terms=[source]).attrs["sources"] == ("default",)


def test_public_handle_routes_require_a_bound_registry():
    model, source, linear = _model("unbound")
    _, block, state_declaration = _references(model)
    program = adctime.Program("unbound")
    state = program.state(block, state_declaration).n

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
    module, block, state_declaration = _references(first)
    program = adctime.Program("macro")._bind_operators(module)
    with pytest.raises(ValueError, match="no operator registry is bound for owner"):
        libtime.explicit_rk(
            program, block, state_declaration, rhs_operator=foreign_rate,
            tableau=libtime.SSPRK2_TABLEAU)

    valid = adctime.Program("macro")._bind_operators(module)
    libtime.explicit_rk(
        valid, block, state_declaration, rhs_operator=first_rate,
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


def test_state_space_is_derived_from_the_model_declaration_not_the_call_site():
    module = Module("spaces")
    operator_space = module.state_space("U", ("a", "b"))
    operator = module.operator(
        name="L", kind="local_linear_operator",
        signature=Signature((), LocalLinearOperator(operator_space, operator_space)),
        expr="linear-map")
    # The rate declaration makes the Module's sole state descriptor available
    # to Program binding; callers cannot replace it with an ad-hoc same-named space.
    module.operator(
        name="R", kind="local_rate",
        signature=Signature((operator_space,), RateSpace(operator_space)), expr="rate")
    handle = OperatorHandle(
        operator.name, kind=operator.kind, owner=module.owner_path,
        signature=operator.signature)
    _, block, state_declaration = _references(module)
    program = adctime.Program("spaces")._bind_operators(module)
    temporal = program.state(block, state_declaration)
    state = temporal.n
    linear = program._call(handle)

    assert temporal.space is operator_space
    assert state.space is operator_space
    assert program.apply(linear, state=state).space == RateSpace(operator_space)
    with pytest.raises(TypeError, match="unexpected keyword argument 'space'"):
        program.state(block, state_declaration, space=operator_space)


def test_invalid_typed_state_identity_is_atomic():
    module = Module("invalid-state-model")
    state_space = module.state_space("U", ("u",))
    state_declaration = module.state_handle(state_space)
    _, block, _ = _references(module)
    qualified = block[state_declaration]
    wrong_kind = Handle("phi", kind="field", owner=module.owner_path)
    program = adctime.Program("invalid-state")

    def snapshot():
        return (
            dict(program._state_spaces),
            tuple(program._values),
            dict(program._time_states),
            program._case_owner_path,
        )

    invalid_calls = (
        (lambda: program.state("block", state_declaration), TypeError, "BlockHandle"),
        (lambda: program.state(block, "U"), TypeError, "declared Handle"),
        (lambda: program.state(block, wrong_kind), TypeError, "expected 'state'"),
        (lambda: program.state(block, qualified), DoubleOwnershipError, "already-qualified"),
    )
    for invoke, error, message in invalid_calls:
        before = snapshot()
        with pytest.raises(error, match=message):
            invoke()
        assert snapshot() == before
