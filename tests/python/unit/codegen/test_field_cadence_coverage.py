"""Final field-cadence coverage: typed schedule plus explicit stale-read policy."""
from __future__ import annotations

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
    RecomputeAtOutput,
)
from pops.fields.context import RecomputeField, UseHeldField, UseMaterializedField
from pops.fields.policies import FieldReadError
from pops.identity import make_identity
from pops.model import Handle, OwnerPath
from pops.time import Clock, TimePoint, always, every
from pops.time._schedule.ir import ScheduleDueKind


def _scheduled_field(step: int = 4):
    owner = OwnerPath.model("cadence-model")
    clock = Clock("main", owner=owner)
    point = TimePoint(clock, step=step)
    layout = LayoutBinding(
        Handle("mesh", kind="layout", owner=OwnerPath.layout("mesh")), generation=0)
    context = FieldContext(
        operator=Handle("poisson", kind="field_operator", owner=owner),
        inputs=(FieldInput(
            Handle("rho", kind="state", owner=owner),
            make_identity("field-input", {"step": step}),
        ),),
        clock=clock,
        point=point,
        layout=layout,
        materialization=Accepted(),
        validity=FieldValidity.valid_at(point, layout),
    )
    return clock, context


def test_every_schedule_and_field_read_policy_cover_due_and_off_cadence_paths():
    clock, context = _scheduled_field()
    cadence = every(4, clock=clock)
    lowering = cadence.native_schedule_ir(where="field materialization")
    assert lowering.due.kind is ScheduleDueKind.CACHE_PERIOD
    assert lowering.due.period == 4

    assert isinstance(
        context.resolve_read(
            FieldConsumer.PROGRAM, at=context.point, layout=context.layout),
        UseMaterializedField,
    )
    off_cadence = TimePoint(clock, step=5)
    with pytest.raises(FieldReadError, match="explicit typed field read policy"):
        context.resolve_read(
            FieldConsumer.OUTPUT, at=off_cadence, layout=context.layout)

    held = context.resolve_read(
        FieldConsumer.OUTPUT,
        at=off_cadence,
        layout=context.layout,
        policy=HoldLastValue(on_failure=FailFieldRead()),
    )
    recomputed = context.resolve_read(
        FieldConsumer.OUTPUT,
        at=off_cadence,
        layout=context.layout,
        policy=RecomputeAtOutput(on_failure=FailFieldRead()),
    )
    assert isinstance(held, UseHeldField)
    assert isinstance(recomputed, RecomputeField)


def test_always_schedule_has_no_off_cadence_policy_slot():
    clock, _ = _scheduled_field()
    schedule = always(clock=clock)
    assert schedule.is_always() is True
    assert schedule.to_data()["off"] is None
