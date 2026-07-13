from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pops.fields import (
    Accepted,
    FailFieldRead,
    FieldConsumer,
    FieldContext,
    FieldInput,
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
from pops.time import Clock, TimePoint


def _context(*, materialization: object | None = None) -> FieldContext:
    owner = OwnerPath.model("electrostatic")
    clock = Clock("main", owner=owner)
    point = TimePoint(clock, step=4)
    layout = LayoutBinding(
        Handle("mesh", kind="layout", owner=OwnerPath.layout("mesh")), generation=2
    )
    return FieldContext(
        operator=Handle("poisson", kind="field_operator", owner=owner),
        inputs=(
            FieldInput(
                Handle("rho", kind="state", owner=owner),
                make_identity("field-input-version", {"step": 4}),
            ),
        ),
        clock=clock,
        point=point,
        layout=layout,
        materialization=materialization or Accepted(),
        validity=FieldValidity.valid_at(point, layout),
    )


def test_current_accepted_context_reads_without_policy() -> None:
    context = _context()
    decision = context.resolve_read(FieldConsumer.PROGRAM, at=context.point, layout=context.layout)

    assert isinstance(decision, UseMaterializedField)
    assert context.inspect()["identity"] == context.identity.token


def test_stale_or_off_schedule_read_without_policy_fails_before_runtime() -> None:
    context = _context()
    later = TimePoint(context.clock, step=5)

    with pytest.raises(FieldReadError, match="explicit typed field read policy"):
        context.resolve_read(FieldConsumer.OUTPUT, at=later, layout=context.layout)


def test_hold_last_value_is_explicit_and_never_crosses_regrid_or_provisional_state() -> None:
    context = _context()
    later = TimePoint(context.clock, step=5)
    policy = HoldLastValue(on_failure=FailFieldRead())

    assert isinstance(
        context.resolve_read(FieldConsumer.OUTPUT, at=later, layout=context.layout, policy=policy),
        UseHeldField,
    )

    regridded = LayoutBinding(context.layout.layout, generation=3)
    with pytest.raises(FieldReadError, match="across regrid"):
        context.resolve_read(FieldConsumer.OUTPUT, at=later, layout=regridded, policy=policy)

    provisional = _context(materialization=Provisional("attempt-5"))
    with pytest.raises(FieldReadError, match="provisional values"):
        provisional.resolve_read(
            FieldConsumer.OUTPUT, at=later, layout=provisional.layout, policy=policy
        )


def test_consumer_specific_recompute_and_failure_action_are_explicit() -> None:
    context = _context()
    later = TimePoint(context.clock, step=5)
    output_policy = RecomputeAtOutput(on_failure=RejectFieldAttempt())

    decision = context.resolve_read(
        FieldConsumer.OUTPUT, at=later, layout=context.layout, policy=output_policy
    )
    assert isinstance(decision, RecomputeField)
    assert isinstance(decision.on_failure, RejectFieldAttempt)

    with pytest.raises(FieldAttemptRejected, match="belongs to output"):
        context.resolve_read(
            FieldConsumer.DIAGNOSTIC,
            at=later,
            layout=context.layout,
            policy=output_policy,
        )


def test_context_identity_observes_inputs_clock_point_layout_state_and_validity() -> None:
    context = _context()
    baseline = context.identity
    owner = OwnerPath.model("electrostatic")
    other_clock = Clock("subcycle", owner=owner)
    other_point = TimePoint(other_clock, step=4)
    other_layout = LayoutBinding(context.layout.layout, generation=3)
    other_input = FieldInput(
        context.inputs[0].reference,
        make_identity("field-input-version", {"step": 3}),
    )

    candidates = (
        replace(context, inputs=(other_input,)),
        replace(
            context,
            clock=other_clock,
            point=other_point,
            validity=FieldValidity.valid_at(other_point, context.layout),
        ),
        replace(
            context,
            point=TimePoint(context.clock, step=5),
            validity=FieldValidity.valid_at(TimePoint(context.clock, step=5), context.layout),
        ),
        replace(
            context,
            layout=other_layout,
            validity=FieldValidity.valid_at(context.point, other_layout),
        ),
        replace(context, materialization=Provisional("attempt-4")),
        context.invalidate("regrid"),
    )
    assert all(candidate.identity != baseline for candidate in candidates)


def test_context_and_policy_are_immutable() -> None:
    context = _context()
    policy = RecomputeAtDiagnostic(on_failure=FailFieldRead())

    with pytest.raises(FrozenInstanceError):
        context.point = context.point
    with pytest.raises(FrozenInstanceError):
        policy.on_failure = RejectFieldAttempt()
