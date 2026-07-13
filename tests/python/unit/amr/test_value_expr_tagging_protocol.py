from __future__ import annotations

import pytest

import pops
from pops.amr import AMRTaggingResolutionContext
from pops.ir import ValueExpr
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


def test_value_indicator_refuses_unresolved_and_foreign_case_handles() -> None:
    expected_case, unresolved = _state("expected")
    foreign_case, foreign = _state("foreign")
    context = _owner_only_context(expected_case.name)

    with pytest.raises(TypeError, match="owner-qualified block-state"):
        context.resolve_value_indicator(
            handle=unresolved, action="refine", comparison="gt", threshold=None)

    foreign_case.freeze()
    resolved_foreign = foreign_case.resolve(foreign)
    with pytest.raises(ValueError, match="different Case owner"):
        context.resolve_value_indicator(
            handle=resolved_foreign, action="refine", comparison="gt", threshold=None)
