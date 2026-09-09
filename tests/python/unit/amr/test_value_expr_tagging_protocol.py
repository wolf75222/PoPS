from __future__ import annotations

import pytest

import pops
from pops.amr._resolution import AMRTaggingResolutionContext
from pops.math import ValueExpr
from pops.model import OwnerPath


def _state(case_name: str):
    model = pops.Model("model")
    state = model.state("U", components=("u",))
    case = pops.Case(case_name)
    block = case.block("fluid", model)
    return case, block[state]


def _owner_only_context(case_name: str) -> AMRTaggingResolutionContext:
    context = object.__new__(AMRTaggingResolutionContext)
    object.__setattr__(context, "owner", OwnerPath.case(case_name))
    object.__setattr__(context, "layout_plan", None)
    object.__setattr__(context, "numerics", ())
    object.__setattr__(context, "resolve", lambda value: value)
    return context


def test_value_expr_delegates_to_the_open_indicator_context() -> None:
    _case, state = _state("delegation")
    calls = []

    class Context:
        def resolve_value_indicator(self, **kwargs):
            calls.append(kwargs)
            return "resolved-above"

    result = ValueExpr(state).resolve_for_amr_tagging(
        Context(), action="refine", comparison="gt", threshold="threshold")

    assert result == "resolved-above"
    assert calls == [{
        "handle": state,
        "action": "refine",
        "comparison": "gt",
        "threshold": "threshold",
    }]


@pytest.mark.parametrize("component", [None, "u"])
def test_value_indicator_refuses_unresolved_and_foreign_case_handles(component) -> None:
    expected_case, unresolved = _state("expected")
    foreign_case, foreign = _state("foreign")
    context = _owner_only_context(expected_case.name)

    with pytest.raises(TypeError, match="owner-qualified block-state"):
        context.resolve_value_indicator(
            handle=unresolved, component=component, action="refine", comparison="gt", threshold=None)

    foreign_case.freeze()
    resolved_foreign = foreign_case.resolve(foreign)
    with pytest.raises(ValueError, match="different Case owner"):
        context.resolve_value_indicator(
            handle=resolved_foreign, component=component, action="refine", comparison="gt", threshold=None)


def _unit_typed_tagging_identity(units):
    import json

    from pops.amr import AMRTagging, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag

    model = pops.Model("model")
    state = model.state("U", components=("first", "second"), units=units)
    case = pops.Case("typed_units")
    block = case.block("fluid", model)
    case.freeze()
    subject = case.resolve(block[state])
    tagging = AMRTagging(
        rules=(Tag(ValueExpr(subject)["first"] > 1.), Buffer(cells=1)),
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS)
    return subject.qualified_id, json.dumps(tagging.inspect(), sort_keys=True)


def test_tagging_dimensions_keep_exact_powers_and_canonical_identity():
    from fractions import Fraction

    from pops._ir.quantity import PhysicalDimension

    units = PhysicalDimension((("T", -1), ("L", Fraction(1, 3))))
    equivalent = PhysicalDimension((("L", Fraction(2, 6)), ("T", -1)))
    left = _unit_typed_tagging_identity((units, PhysicalDimension(())))
    right = _unit_typed_tagging_identity((equivalent, PhysicalDimension(())))
    assert left == right
    assert '"kind": "physical_dimension"' in left[1]
    assert '["L", 1, 3]' in left[1]
    assert '["T", -1, 1]' in left[1]


def test_tagging_identity_distinguishes_unknown_dimensionless_and_all_component_units():
    from fractions import Fraction

    from pops._ir.quantity import PhysicalDimension

    dimensionless = PhysicalDimension(())
    third = PhysicalDimension((("L", Fraction(1, 3)),))
    half = PhysicalDimension((("L", Fraction(1, 2)),))
    values = [
        _unit_typed_tagging_identity(units)
        for units in ((None, None), (dimensionless, None), (third, None),
                      (half, None), (third, dimensionless))
    ]
    # Definition handles already hash the units independently. Normalize only that
    # handle contribution to prove that the serialized Space itself keeps the distinction.
    identities = {identity.replace(handle, "same-qualified-handle") for handle, identity in values}
    assert len(identities) == len(values)


def test_tagging_expression_identity_rejects_unknown_objects_with_serialization_hooks():
    from pops._ir import Expr
    from pops.amr.authoring import _expression_data

    class UnknownIdentity:
        def to_data(self):
            raise AssertionError("unknown serialization hooks must not run")

        def __repr__(self):
            raise AssertionError("repr must not authenticate an unknown identity")

        __str__ = __repr__

    class UnknownExpression(Expr):
        def __pops_ir_key__(self, recurse):
            return ("unknown", UnknownIdentity())

    with pytest.raises(TypeError, match="unsupported UnknownIdentity"):
        _expression_data(UnknownExpression())
