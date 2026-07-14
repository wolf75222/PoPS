"""ADC-685: deterministic ConsumerGraph and accepted-only publication transactions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pops.fields import (
    Accepted,
    FailFieldRead,
    FieldContext,
    FieldInput,
    FieldValidity,
    LayoutBinding,
    RecomputeAtOutput,
)
from pops.identity import make_identity
from pops.model import Handle, OwnerPath
from pops.output import HDF5, NPZ
from pops.output._consumer_contracts import (
    ConsumerCursorSet,
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerMoment,
    ConsumerQuantity,
    FailRun,
    ParallelMode,
    Retry,
    SkipSampleReported,
)
from pops.runtime._runtime_plan_contracts import RuntimePlanningError
from pops.runtime._runtime_planning import build_runtime_plans
from pops.runtime._consumer import (
    ConsumerPublicationError,
    ConsumerPublisher,
    ConsumerTransaction,
    PreparedPublication,
    PublicationReceipt,
    plan_accepted_side_effects,
)
from pops.time import AcceptedStep, Clock, Every, Schedule, TimePoint
from tests.python.unit.runtime.test_runtime_planning import _install, _manifest


def _runtime(*, collective: bool = False):
    install = _install()
    requirements = ()
    if collective:
        requirements = ({
            "capability": "collective",
            "resource": "state:u",
            "operation": "gather",
            "strategy": "ordered_tree",
        },)
    runtime = build_runtime_plans(install, {
        "fluid": _manifest(
            "fluid",
            reads=({"resource": "state:u"},),
            writes=({"resource": "rate:u"},),
            requirements=requirements,
        ),
    })
    return install, runtime


def _manifest_for(
    runtime,
    name: str,
    clock: Clock,
    *,
    n: int = 2,
    resource: str = "state:u",
    dependency: Handle | None = None,
    action=None,
    parallel_mode=ParallelMode.SERIAL,
) -> ConsumerManifest:
    owner = OwnerPath.consumer("adc-685")
    handle = Handle(name, kind="consumer", owner=owner)
    quantity = ConsumerQuantity(
        Handle(name + "-value", kind="state", owner=OwnerPath.model("adc-685")),
        resource,
        runtime.calls[0].layout_id,
    )
    dependencies = (dependency,) if dependency is not None else ()
    if action is None:
        action = FailRun()
    return ConsumerManifest(
        handle=handle,
        kind=ConsumerKind.SCIENTIFIC_OUTPUT,
        quantities=(quantity,),
        schedule=Schedule(Every(AcceptedStep(clock), n)),
        target_uri="file:///adc-685/%s" % name,
        output_format=(
            HDF5(parallel=True) if parallel_mode is ParallelMode.COLLECTIVE else NPZ()),
        parallel_mode=parallel_mode,
        dependencies=dependencies,
        failure_action=action,
    )


def _moment(clock: Clock, *, step: int = 2, layouts=()) -> ConsumerMoment:
    return ConsumerMoment(
        TimePoint(clock, step=step),
        accepted_step=step,
        attempt=1,
        layouts=tuple(layouts),
    )


class _Prepared(PreparedPublication):
    def __init__(self, effect, publisher):
        self.effect = effect
        self.publisher = publisher
        self.temp_id = "temp-%d" % publisher.prepare_calls
        publisher.temporaries.add(self.temp_id)

    @property
    def effect_identity(self):
        return self.effect.identity

    @property
    def payload_identity(self):
        return self.effect.payload.identity

    def publish(self):
        self.publisher.publish_calls += 1
        if self.publisher.publish_calls in self.publisher.fail_on:
            raise OSError("injected publication failure")
        self.publisher.temporaries.remove(self.temp_id)
        artifact = "artifact-%s" % self.effect.payload.identity.hexdigest[:12]
        self.publisher.artifacts.add(artifact)
        return PublicationReceipt(
            self.effect.identity,
            self.effect.payload.identity,
            "test-publisher",
            artifact,
        )

    def discard(self):
        self.publisher.temporaries.discard(self.temp_id)

    def rollback(self):
        self.publisher.temporaries.discard(self.temp_id)
        artifact = "artifact-%s" % self.effect.payload.identity.hexdigest[:12]
        self.publisher.artifacts.discard(artifact)


class _Publisher(ConsumerPublisher):
    def __init__(self, *, fail_publications=0, fail_on=()):
        self.fail_on = set(fail_on) or set(range(1, fail_publications + 1))
        self.prepare_calls = 0
        self.publish_calls = 0
        self.temporaries = set()
        self.artifacts = set()

    def prepare(self, effect):
        self.prepare_calls += 1
        return _Prepared(effect, self)


def test_graph_and_plan_are_semantic_and_insertion_order_independent():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685"))
    first = _manifest_for(runtime, "diagnostic", clock)
    second = _manifest_for(runtime, "output", clock, dependency=first.handle)

    forward = ConsumerGraph((first, second))
    reversed_graph = ConsumerGraph((second, first))
    assert forward.identity == reversed_graph.identity
    assert forward.topology == (first, second)

    changed_selection = replace(
        first,
        quantities=(replace(first.quantities[0], runtime_resource="rate:u"),),
    )
    changed_schedule = replace(first, schedule=Schedule(Every(AcceptedStep(clock), 3)))
    changed_action = replace(first, failure_action=Retry(2))
    assert len({
        first.identity,
        changed_selection.identity,
        changed_schedule.identity,
        changed_action.identity,
    }) == 4

    plan = plan_accepted_side_effects(runtime, reversed_graph, _moment(clock))
    assert [value.consumer_id for value in plan.effects] == [
        first.qualified_id,
        second.qualified_id,
    ]
    assert len({value.payload.identity for value in plan.effects}) == 2
    assert len(plan.lowering_coverage.rows) == 2


def test_collective_mode_requires_a_nonserial_context_before_planning():
    _, serial_runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-collective"))
    manifest = _manifest_for(
        serial_runtime, "collective", clock, parallel_mode=ParallelMode.COLLECTIVE)
    with pytest.raises(RuntimePlanningError) as error:
        plan_accepted_side_effects(serial_runtime, ConsumerGraph((manifest,)), _moment(clock))
    assert error.value.code == "collective_consumer_requires_distributed_context"

    _, collective_runtime = _runtime(collective=True)
    manifest = _manifest_for(
        collective_runtime, "collective", clock, parallel_mode=ParallelMode.COLLECTIVE)
    with pytest.raises(RuntimePlanningError) as error:
        plan_accepted_side_effects(
            collective_runtime, ConsumerGraph((manifest,)), _moment(clock))
    assert error.value.code == "collective_consumer_requires_distributed_context"
    assert error.value.evidence == {"communicator": "serial"}


def test_stale_field_requires_explicit_policy_and_records_recompute_without_solving():
    install, runtime = _runtime()
    owner = OwnerPath.model("adc-685-field")
    clock = Clock("solution", owner=owner)
    current = TimePoint(clock, step=1)
    layout = LayoutBinding(install.artifact.plan.layout_plan.layouts[0].handle, generation=4)
    context = FieldContext(
        operator=Handle("phi", kind="field_operator", owner=owner),
        inputs=(FieldInput(
            Handle("rho", kind="state", owner=owner),
            make_identity("field-input-version", {"step": 1}),
        ),),
        clock=clock,
        point=current,
        layout=layout,
        materialization=Accepted(),
        validity=FieldValidity.valid_at(current, layout),
    )
    quantity = ConsumerQuantity(
        context.operator,
        "state:u",
        runtime.calls[0].layout_id,
        field_context=context,
    )
    manifest = ConsumerManifest(
        Handle("field-output", kind="consumer", owner=OwnerPath.consumer("adc-685-field")),
        ConsumerKind.SCIENTIFIC_OUTPUT,
        (quantity,),
        Schedule(Every(AcceptedStep(clock), 1)),
        "file:///adc-685/field",
        NPZ(),
        ParallelMode.SERIAL,
    )
    moment = _moment(clock, step=2, layouts=(layout,))
    with pytest.raises(RuntimePlanningError) as error:
        plan_accepted_side_effects(runtime, ConsumerGraph((manifest,)), moment)
    assert error.value.code == "consumer_field_not_fresh"

    explicit = replace(
        manifest,
        quantities=(replace(
            quantity,
            field_policy=RecomputeAtOutput(on_failure=FailFieldRead()),
        ),),
    )
    plan = plan_accepted_side_effects(runtime, ConsumerGraph((explicit,)), moment)
    assert plan.effects[0].payload.fields[0].action == "recompute"
    assert plan.effects[0].payload.fields[0].evidence["explicit"] is True


def test_rejected_attempt_discards_temporaries_without_publication_or_cursor_advance():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-reject"))
    manifest = _manifest_for(runtime, "reject", clock)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(runtime, ConsumerGraph((manifest,)), _moment(clock), cursors)
    publisher = _Publisher()

    report = ConsumerTransaction(plan, cursors, publisher).reject()

    assert report.status == "rejected"
    assert report.published == ()
    assert report.cursors.to_data() == cursors.to_data()
    assert publisher.publish_calls == 0
    assert publisher.temporaries == set()
    assert publisher.artifacts == set()


def test_failure_actions_have_exact_cursor_and_artifact_semantics():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-failures"))
    cursors = ConsumerCursorSet()

    fail = _manifest_for(runtime, "fail", clock, action=FailRun())
    fail_plan = plan_accepted_side_effects(runtime, ConsumerGraph((fail,)), _moment(clock))
    fail_publisher = _Publisher(fail_publications=1)
    with pytest.raises(ConsumerPublicationError) as failure:
        ConsumerTransaction(fail_plan, cursors, fail_publisher).accept()
    assert failure.value.report.cursors.to_data() == cursors.to_data()
    assert failure.value.report.published == ()
    assert fail_publisher.temporaries == set()
    assert fail_publisher.artifacts == set()

    retry = _manifest_for(runtime, "retry", clock, action=Retry(2))
    retry_plan = plan_accepted_side_effects(runtime, ConsumerGraph((retry,)), _moment(clock))
    retry_publisher = _Publisher(fail_publications=1)
    retried = ConsumerTransaction(retry_plan, cursors, retry_publisher).accept()
    assert retried.status == "accepted"
    assert len(retried.published) == 1
    assert retried.cursors.for_consumer(retry.qualified_id).committed_samples == 1
    assert retry_publisher.prepare_calls == 2
    assert len(retry_publisher.artifacts) == 1

    skip = _manifest_for(runtime, "skip", clock, action=SkipSampleReported())
    skip_plan = plan_accepted_side_effects(runtime, ConsumerGraph((skip,)), _moment(clock))
    skip_publisher = _Publisher(fail_publications=1)
    skipped = ConsumerTransaction(skip_plan, cursors, skip_publisher).accept()
    assert skipped.status == "accepted"
    assert skipped.published == ()
    assert len(skipped.skipped) == 1
    assert skipped.cursors.to_data() == cursors.to_data()
    assert skip_publisher.artifacts == set()


def test_success_receipt_is_the_only_cursor_commit_and_deduplicates_occurrence():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-success"))
    manifest = _manifest_for(runtime, "success", clock)
    graph, moment = ConsumerGraph((manifest,)), _moment(clock)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(runtime, graph, moment, cursors)
    publisher = _Publisher()

    report = ConsumerTransaction(plan, cursors, publisher).accept()

    assert len(report.published) == 1
    assert report.published[0].payload_identity == plan.effects[0].payload.identity
    assert report.cursors.for_consumer(manifest.qualified_id) == plan.effects[0].cursor_after
    duplicate = plan_accepted_side_effects(runtime, graph, moment, report.cursors)
    assert duplicate.effects == ()


def test_accepted_publications_remain_compensatable_until_the_outer_transaction_seals():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-compensate"))
    manifest = _manifest_for(runtime, "compensate", clock)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(
        runtime, ConsumerGraph((manifest,)), _moment(clock), cursors)
    publisher = _Publisher()
    transaction = ConsumerTransaction(plan, cursors, publisher)

    accepted = transaction.accept()
    assert accepted.published
    assert publisher.artifacts

    compensated = transaction.rollback_accepted()
    assert compensated.status == "rejected"
    assert compensated.published == ()
    assert compensated.cursors.to_data() == cursors.to_data()
    assert publisher.temporaries == set()
    assert publisher.artifacts == set()


def test_later_publication_failure_compensates_every_earlier_artifact():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-partial"))
    first = _manifest_for(runtime, "first", clock, n=1)
    second = _manifest_for(runtime, "second", clock, n=1, dependency=first.handle)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(
        runtime, ConsumerGraph((first, second)), _moment(clock, step=1), cursors)
    publisher = _Publisher(fail_on=(2,))

    with pytest.raises(ConsumerPublicationError) as failure:
        ConsumerTransaction(plan, cursors, publisher).accept()

    assert failure.value.report.published == ()
    assert failure.value.report.cursors.to_data() == cursors.to_data()
    assert publisher.temporaries == set()
    assert publisher.artifacts == set()
