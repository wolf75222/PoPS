"""Typed field materialization cadence and off-cadence consumer decisions."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pops.fields import (
    Accepted,
    FailFieldRead,
    FieldConsumer,
    FieldContext,
    FieldFailureAction,
    FieldInput,
    FieldReadPolicy,
    FieldValidity,
    HoldLastValue,
    LayoutBinding,
    Provisional,
    RecomputeAtDiagnostic,
    RecomputeAtOutput,
    RejectFieldAttempt,
)
from pops.fields.context import RecomputeField, UseHeldField, UseMaterializedField
from pops.fields.policies import FieldAttemptRejected, FieldReadError
from pops.identity import make_identity
from pops.model import Handle, OwnerPath
from pops.time import Clock, TimePoint, every


def _context(*, provisional: bool = False):
    owner = OwnerPath.model("electrostatic")
    clock = Clock("main", owner=owner)
    point = TimePoint(clock, step=4)
    layout = LayoutBinding(
        Handle("mesh", kind="layout", owner=OwnerPath.layout("mesh")), generation=2)
    context = FieldContext(
        operator=Handle("poisson", kind="field_operator", owner=owner),
        inputs=(FieldInput(
            Handle("rho", kind="state", owner=owner),
            make_identity("rho-version", {"step": 4}),
        ),),
        clock=clock,
        point=point,
        layout=layout,
        materialization=Provisional("attempt-4") if provisional else Accepted(),
        validity=FieldValidity.valid_at(point, layout),
    )
    return context, every(4, clock=clock)


def test_due_materialization_is_read_directly_and_off_cadence_hold_is_explicit():
    context, cadence = _context()
    assert cadence.to_data()["trigger"] == {"type": "every", "n": 4}
    assert isinstance(
        context.resolve_read(
            FieldConsumer.PROGRAM, at=context.point, layout=context.layout),
        UseMaterializedField,
    )

    requested = TimePoint(context.clock, step=5)
    policy = HoldLastValue(on_failure=FailFieldRead())
    decision = context.resolve_read(
        FieldConsumer.OUTPUT, at=requested, layout=context.layout, policy=policy)
    assert isinstance(decision, UseHeldField)
    assert decision.source_point == context.point
    assert decision.requested_point == requested


def test_hold_never_crosses_regrid_or_provisional_attempt():
    context, _ = _context()
    requested = TimePoint(context.clock, step=5)
    policy = HoldLastValue(on_failure=FailFieldRead())
    regridded = LayoutBinding(context.layout.layout, generation=3)
    with pytest.raises(FieldReadError, match="across regrid"):
        context.resolve_read(
            FieldConsumer.OUTPUT, at=requested, layout=regridded, policy=policy)

    provisional, _ = _context(provisional=True)
    with pytest.raises(FieldReadError, match="provisional values"):
        provisional.resolve_read(
            FieldConsumer.OUTPUT,
            at=requested,
            layout=provisional.layout,
            policy=policy,
        )


def test_recompute_policy_is_consumer_specific_and_carries_failure_action():
    context, _ = _context()
    requested = TimePoint(context.clock, step=5)
    policy = RecomputeAtOutput(on_failure=RejectFieldAttempt())
    decision = context.resolve_read(
        FieldConsumer.OUTPUT, at=requested, layout=context.layout, policy=policy)
    assert isinstance(decision, RecomputeField)
    assert decision.consumer is FieldConsumer.OUTPUT
    assert isinstance(decision.on_failure, RejectFieldAttempt)
    assert policy.to_data() == {
        "policy": "recompute",
        "consumer": "output",
        "on_failure": {"action": "reject_field_attempt"},
    }

    with pytest.raises(FieldAttemptRejected, match="belongs to output"):
        context.resolve_read(
            FieldConsumer.DIAGNOSTIC,
            at=requested,
            layout=context.layout,
            policy=policy,
        )


def test_policy_contract_is_typed_immutable_and_rejects_legacy_shortcuts():
    assert issubclass(HoldLastValue, FieldReadPolicy)
    assert issubclass(FailFieldRead, FieldFailureAction)
    policy = RecomputeAtDiagnostic(on_failure=FailFieldRead())
    with pytest.raises(FrozenInstanceError):
        policy.on_failure = RejectFieldAttempt()
    with pytest.raises(TypeError, match="FieldFailureAction"):
        HoldLastValue(on_failure="hold")
    context, _ = _context()
    with pytest.raises(TypeError, match="unsupported FieldReadPolicy"):
        context.resolve_read(
            FieldConsumer.OUTPUT,
            at=TimePoint(context.clock, step=5),
            layout=context.layout,
            policy=object(),
        )
