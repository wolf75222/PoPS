"""Final IMEX, Lie/Strang and Adams--Bashforth factory contracts."""
from __future__ import annotations

from fractions import Fraction
import inspect

import pytest

import pops.lib.time as lt
from pops.time import Program


def _authoring(name="time"):
    from pops.descriptors import Descriptor
    from pops.fields import FieldDiscretization, FieldOperator
    from pops.math import ValueExpr
    from pops.math import laplacian
    from pops.model import LocalLinearOperator, Module, RateSpace, Signature
    from pops.problem import Case

    module = Module(name + "_model")
    state_space = module.state_space("U", ("u",))
    state = module.state_handle(state_space)
    field_space = module.field_space("potential", ("potential",))
    provider = module.operator(
        name="potential_provider",
        signature=Signature((state_space,), field_space),
        kind="field_operator",
        expr="potential_provider",
    )
    explicit = module.operator(
        name="explicit",
        signature=Signature((state_space, field_space), RateSpace(state_space)),
        kind="local_rate",
        expr="explicit",
    )
    implicit = module.operator(
        name="implicit",
        signature=Signature(
            (), LocalLinearOperator(state_space, state_space)),
        kind="local_linear_operator",
        expr="implicit",
    )

    class _Method(Descriptor):
        category = "field_method"

        def to_data(self):
            return {"type": "unit-second-order"}

    class _Solver(Descriptor):
        category = "elliptic_solver"

        def to_data(self):
            return {"type": "unit-krylov"}

    case = Case(name + "_case")
    block = case.block("plasma", module, states=(state,))
    unknown = block[module.field_handle(field_space)]
    fields = case.field(
        FieldOperator(
            "potential",
            unknown=unknown,
            equation=-laplacian(ValueExpr(unknown)) == ValueExpr(block[state]),
            providers=provider,
        ),
        FieldDiscretization(method=_Method(), boundaries=(), solver=_Solver()),
    )
    return block[state], explicit, implicit, fields


def test_imex_builds_one_typed_local_linear_stage():
    state, explicit, implicit, fields = _authoring("imex")
    program = lt.IMEX(
        state,
        explicit_operator=explicit,
        implicit_operator=implicit,
        fields_operator=fields,
    )
    assert type(program) is Program
    assert program.validate() is True
    operations = [value.op for value in program._values]
    assert operations.count("solve_local_linear") == 1
    retained = {
        value.attrs["operator_handle"]
        for value in program._values
        if "operator_handle" in value.attrs
    }
    assert {explicit, implicit} <= retained
    solve = next(value for value in program._values if value.op == "solve_fields")
    assert solve.attrs["field"] is fields


def test_imex_rejects_free_names_and_legacy_theta_knob():
    state, explicit, implicit, _ = _authoring("imex-guards")
    with pytest.raises(TypeError, match="OperatorHandle"):
        lt.IMEX(state, explicit_operator="explicit", implicit_operator=implicit)
    with pytest.raises(TypeError, match="unexpected keyword"):
        lt.IMEX(
            state, explicit_operator=explicit, implicit_operator=implicit, theta=Fraction(1, 2))


def test_lie_and_strang_pass_exact_partition_fractions_and_commit():
    state, _, _, _ = _authoring("split")
    seen = []

    def first(program, current, fraction, *, at):
        seen.append(("first", fraction, at))
        return program.value("first", 1 * current, at=at)

    def second(program, current, fraction, *, at):
        seen.append(("second", fraction, at))
        return program.value("second", 1 * current, at=at)

    strang = lt.Strang(state, first=first, second=second)
    assert strang.validate() is True
    assert [(name, fraction) for name, fraction, _ in seen] == [
        ("first", Fraction(1, 2)), ("second", 1), ("first", Fraction(1, 2))]
    assert len(strang.commits()) == 1

    seen.clear()
    lie = lt.Lie(state, first=first, second=second)
    assert lie.validate() is True
    assert [(name, fraction) for name, fraction, _ in seen] == [
        ("first", 1), ("second", 1)]


def test_split_subflow_must_materialize_the_supplied_endpoint():
    state, _, _, _ = _authoring("bad-split")

    def wrong(program, current, fraction, *, at):
        del fraction, at
        return program.value("wrong", 1 * current)

    with pytest.raises(ValueError, match="instead of"):
        lt.Lie(state, first=wrong, second=wrong)


@pytest.mark.parametrize("order", (1, 2, 3))
def test_adams_bashforth_orders_are_valid_and_history_depth_is_explicit(order):
    state, rate, _, fields = _authoring("ab%d" % order)
    program = lt.AdamsBashforth(state, rate=rate, order=order, fields=fields)
    assert program.validate() is True
    histories = program._serialize().get("histories", [])
    if order == 1:
        assert histories == []
    else:
        assert histories[0]["lag"] == order - 1
        assert any(value.op == "store_history" for value in program._values)


def test_adams_bashforth_one_matches_forward_euler_topology():
    state, rate, _, fields = _authoring("ab1")
    ab1 = lt.AdamsBashforth(state, rate=rate, order=1, fields=fields)
    euler = lt.ForwardEuler(state, rate=rate, fields=fields)
    def topology(program):
        return [(value.vtype, value.op, tuple(item.id for item in value.inputs))
                for value in program._values]
    assert topology(ab1) == topology(euler)


@pytest.mark.parametrize("order", (0, 4, True, 1.5, "2"))
def test_adams_bashforth_rejects_unsupported_orders(order):
    state, rate, _, _ = _authoring("bad-ab")
    with pytest.raises((TypeError, ValueError), match="order"):
        lt.AdamsBashforth(state, rate=rate, order=order)


def test_final_factory_signatures_have_no_in_place_program_or_free_selector_surface():
    assert tuple(inspect.signature(lt.IMEX).parameters) == (
        "state", "explicit_operator", "implicit_operator", "fields_operator", "tableau",
        "solve_action")
    assert tuple(inspect.signature(lt.AdamsBashforth).parameters) == (
        "state", "rate", "order", "fields", "solve_action")
    assert tuple(inspect.signature(lt.Lie).parameters) == ("state", "first", "second")
    assert tuple(inspect.signature(lt.Strang).parameters) == ("state", "first", "second")
