from __future__ import annotations

import sys
from types import ModuleType

import pytest

import pops
from pops import _api
import pops._native_selector as selector
import pops.codegen._phases as phases
import pops.codegen._plans as plans


def test_public_compile_selects_dimension_from_verified_resolved_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Resolved:
        resolved_dimension = 3

        @staticmethod
        def verify() -> None:
            events.append("verified")

    plan = Resolved()
    monkeypatch.setattr(plans, "ResolvedSimulationPlan", Resolved)
    monkeypatch.setattr(
        selector,
        "select_native_dimension",
        lambda dimension: events.append(("selected", dimension)),
    )
    bootstrap = ModuleType("pops._bootstrap")
    monkeypatch.setitem(sys.modules, "pops._bootstrap", bootstrap)
    monkeypatch.setattr(pops, "_bootstrap", bootstrap, raising=False)
    result = object()

    def compile_phase(value: object) -> object:
        events.append(("compiled", value))
        return result

    monkeypatch.setattr(phases, "compile", compile_phase)

    assert _api.compile(plan) is result
    assert events == ["verified", ("selected", 3), ("compiled", plan)]


def test_public_compile_rejects_foreign_plan_before_native_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[int] = []
    monkeypatch.setattr(
        selector, "select_native_dimension", lambda dimension: selected.append(dimension)
    )

    with pytest.raises(TypeError, match="ResolvedSimulationPlan"):
        _api.compile(object())
    assert selected == []
