"""Numerical partitions cover the selected scientific balance exactly once."""
from dataclasses import dataclass

import pytest

import pops
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan
from pops.numerics.plan import UnsupportedBalanceRealizationError
from tests.python.unit.numerics.test_discretization_plan import _declarations


@dataclass(frozen=True)
class _InspectionMethod:
    def validate(self, context=None):
        return True

    def validate_rate_contract(self, contract):
        return True

    def resolve_references(self, resolver):
        return self

    def to_data(self):
        return {"method": "inspection-only"}

    def freeze(self):
        return self


def _plan(*rates):
    plan = DiscretizationPlan()
    for rate in rates:
        plan.rates.add(rate, _InspectionMethod())
    return plan


def test_views_cover_each_repeated_physical_occurrence_without_reauthoring_balance():
    _, model, state, flux, initial_rate, _ = _declarations()
    source = model.source("forcing", on=state, value=[state[0]])
    rate = model.rate("full", equation=ddt(state) == -div(flux) + source + source)
    model.select_balance(rate)
    transport = rate.select(flux)
    first = rate.select(rate.occurrences[1])
    second = rate.select(rate.occurrences[2])
    assert _plan(transport, first, second).validate_for(model)
    assert _plan(rate).validate_for(model)
    with pytest.raises(ValueError, match="duplicate physical balance occurrence"):
        _plan(rate, transport).validate_for(model)
    with pytest.raises(ValueError, match="missing=.*occurrences.*1, 2"):
        _plan(transport).validate_for(model)
    with pytest.raises(ValueError, match="extra=.*A"):
        _plan(initial_rate).validate_for(model)


def test_numerical_plan_requires_explicit_choice_between_alternative_balances():
    _, model, state, flux, original, _ = _declarations()
    other = model.rate("alternative", equation=ddt(state) == -div(flux))
    with pytest.raises(ValueError, match="select_balance"):
        _plan(original).validate_for(model)
    model.select_balance(other)
    assert _plan(other).validate_for(model)


def test_unknown_signed_numerical_realization_is_diagnosed_at_resolution():
    _, model, state, flux, _original, _ = _declarations()
    rate = model.rate("positive_divergence", equation=ddt(state) == div(flux))
    model.select_balance(rate)
    plan = _plan(rate)
    assert plan.validate_for(model)
    case = pops.Case("unsupported")
    block = case.block("fluid", model)
    with pytest.raises(UnsupportedBalanceRealizationError) as caught:
        plan.resolve_for(case, block)
    assert caught.value.code == "unsupported_balance_realization"
    assert caught.value.phase == "resolve"
    assert caught.value.context["rate"] == rate.local_id
