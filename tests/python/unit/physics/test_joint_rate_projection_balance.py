"""Whole joint outputs retain target, multiplicity and one captured application."""
from fractions import Fraction

import pytest

import pops
from pops._ir.application import ApplicationProjection, OperatorApplication, RateApplicationProjection
from pops.math import ddt
from pops.model import Rate, RateBundle, Signature
from pops.numerics.plan import UnsupportedBalanceRealizationError
from tests.python.unit.numerics.test_balance_partition_coverage import _plan


def _joint_model():
    model = pops.Model("joint_balances")
    left = model.species("left", state=["rho"])
    right = model.species("right", state=["rho", "energy"])
    module = model.module
    captures = []
    signature = Signature((left.space, right.space), RateBundle({
        "particle_exchange": Rate(left.space), "gas_exchange": Rate(right.space)}))

    @module.operator("collision", signature=signature, kind="coupled_rate")
    def collision(a, b):
        captures.append(1)
        return {"particle_exchange": [b[0] - a[0]],
                "gas_exchange": [a[0] - b[0], b[1]]}

    application = module.apply(collision, left, right)
    return model, left, right, application, captures


def test_target_projection_keeps_component_api_and_does_not_guess_from_output_name():
    _model, left, right, application, captures = _joint_model()
    projected = application[left]
    assert isinstance(projected, RateApplicationProjection)
    assert projected.target is left
    assert projected.output == "particle_exchange"
    assert projected.rate_space == Rate(left.space)
    assert isinstance(projected[0], ApplicationProjection)
    assert len(application[right]) == 2
    assert application["gas_exchange"][1].application is application
    assert tuple(projected)[0].application is application
    assert captures == [1]


def test_two_balances_keep_one_joint_application_and_each_signed_occurrence():
    model, left, right, application, captures = _joint_model()
    left_rate = model.rate("left_balance", equation=ddt(left) == (
        application[left] + Fraction(2, 3) * application[left] - application[left]))
    right_rate = model.rate("right_balance", equation=ddt(right) == application[right])
    assert [item.coefficient for item in left_rate.occurrences] == [1, Fraction(2, 3), -1]
    assert len({item.identity for item in left_rate.occurrences}) == 3
    assert all(item.payload.application is application for item in (
        *left_rate.occurrences, *right_rate.occurrences))
    assert left_rate.select(application[left]) is left_rate
    assert left_rate.select(left_rate.occurrences[1]).occurrences[0] is left_rate.occurrences[1]
    assert model.validate_balance_selection()
    assert captures == [1]
    module = model.module
    operator = module.operator_registry().get(left_rate.local_id)
    assert operator.signature.inputs == (left.space, right.space)
    assert operator.signature.output == Rate(left.space)
    assert operator.lowering["physical_balance"] is left_rate.view
    assert "M4" in operator.lowering["native_unsupported"]["reason"]
    assert "sources" not in operator.lowering
    assert left_rate.local_id not in model._dsl._m._rate_operators


def test_canonical_balance_resolution_keeps_shared_application_in_repeated_projections():
    model, left, _right, application, _captures = _joint_model()
    rate = model.rate("balance", equation=ddt(left) == application[left] + application[left])
    _ = model.module
    resolved = rate.balance.resolve_references(lambda handle: handle._resolved())
    first, second = resolved.occurrences
    assert first.payload.application is second.payload.application
    assert first.payload.target.is_resolved
    assert first.payload.application_identity == second.payload.application_identity
    assert first.identity != second.identity
    assert resolved.to_data()["occurrences"][0]["kind"] == "projection"


def test_wrong_target_and_foreign_application_are_rejected_before_registration():
    model, left, right, application, _captures = _joint_model()
    with pytest.raises(ValueError, match="differentiated quantity"):
        model.rate("wrong_target", equation=ddt(left) == application[right])
    other, foreign_left, _foreign_right, foreign_application, _ = _joint_model()
    with pytest.raises(ValueError, match="application owner"):
        foreign_application[left]
    with pytest.raises(ValueError, match="differentiated quantity"):
        model.rate("foreign", equation=ddt(left) == foreign_application[foreign_left])
    assert other is not model
    assert not model._rate_contracts


def test_forged_application_output_cannot_replace_the_registered_joint_body():
    model, left, _right, application, _captures = _joint_model()
    forged = OperatorApplication(application.operator, application.inputs,
        {"particle_exchange": [left[0] + 123], "gas_exchange": application.outputs["gas_exchange"]},
        occurrence=application.occurrence)
    with pytest.raises(ValueError, match="captured registered body"):
        model.rate("forged", equation=ddt(left) == forged[left])
    assert not model._rate_contracts


def test_joint_rate_projection_is_representable_but_has_an_explicit_native_resolution_refusal():
    model, left, _right, application, captures = _joint_model()
    rate = model.rate("balance", equation=ddt(left) == application[left])
    plan = _plan(rate)
    assert plan.validate_for(model, states=(left,))
    case = pops.Case("joint_projection")
    block = case.block("left", model, states=(left,))
    with pytest.raises(UnsupportedBalanceRealizationError, match="native interaction") as error:
        plan.resolve_for(case, block)
    assert error.value.phase == "resolve"
    assert captures == [1]
