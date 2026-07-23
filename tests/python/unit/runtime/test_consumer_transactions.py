"""ADC-685: deterministic ConsumerGraph and accepted-only publication transactions."""

from __future__ import annotations

from dataclasses import replace
import math

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
from pops.output._console_monitor import ConsolePresentation
from pops.output._restart_provider import RestartV3
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
from pops.time import AcceptedStep, Clock, Every, Schedule, TimePoint, every_dt
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
            HDF5(mode=ParallelMode.COLLECTIVE)
            if parallel_mode is ParallelMode.COLLECTIVE else NPZ()),
        parallel_mode=parallel_mode,
        dependencies=dependencies,
        failure_action=action,
    )


def _moment(
    clock: Clock, *, step: int = 2, physical_time: float | None = None, layouts=()
) -> ConsumerMoment:
    return ConsumerMoment(
        TimePoint(clock, step=step),
        accepted_step=step,
        attempt=1,
        physical_time_hex=float(step if physical_time is None else physical_time).hex(),
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

    def finalize(self):
        self.publisher.finalize_calls += 1
        if self.publisher.finalize_calls <= self.publisher.fail_finalizations:
            raise OSError("injected finalization failure")
        if self.publisher.non_none_finalize:
            return "invalid-finalize-result"
        return None


class _Publisher(ConsumerPublisher):
    def __init__(
        self, *, fail_publications=0, fail_on=(), fail_finalizations=0,
        non_none_finalize=False,
    ):
        self.fail_on = set(fail_on) or set(range(1, fail_publications + 1))
        self.prepare_calls = 0
        self.publish_calls = 0
        self.finalize_calls = 0
        self.fail_finalizations = fail_finalizations
        self.non_none_finalize = non_none_finalize
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


@pytest.mark.parametrize(
    "parallel_mode",
    (ParallelMode.ROOT, ParallelMode.COLLECTIVE, ParallelMode.PER_RANK),
)
def test_distributed_modes_require_a_nonserial_context_before_planning(parallel_mode):
    _, serial_runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-collective"))
    output_format = (
        HDF5(mode=parallel_mode)
        if parallel_mode is not ParallelMode.PER_RANK
        else NPZ(mode=ParallelMode.PER_RANK)
    )
    manifest = replace(
        _manifest_for(serial_runtime, "distributed", clock),
        output_format=output_format,
        parallel_mode=parallel_mode,
    )
    with pytest.raises(RuntimePlanningError) as error:
        plan_accepted_side_effects(serial_runtime, ConsumerGraph((manifest,)), _moment(clock))
    assert error.value.code == "distributed_consumer_requires_distributed_context"

    _, collective_runtime = _runtime(collective=True)
    manifest = replace(
        _manifest_for(collective_runtime, "distributed", clock),
        output_format=output_format,
        parallel_mode=parallel_mode,
    )
    with pytest.raises(RuntimePlanningError) as error:
        plan_accepted_side_effects(
            collective_runtime, ConsumerGraph((manifest,)), _moment(clock))
    assert error.value.code == "distributed_consumer_requires_distributed_context"
    assert error.value.evidence == {"communicator": "serial"}


def test_singleton_collective_requires_an_explicit_provider_capability():
    _, serial_runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-singleton"))
    manifest = ConsumerManifest(
        handle=Handle(
            "checkpoint", kind="consumer", owner=OwnerPath.consumer("adc-685-singleton")),
        kind=ConsumerKind.CHECKPOINT,
        quantities=(),
        schedule=Schedule(Every(AcceptedStep(clock), 2)),
        target_uri="checkpoint/restart",
        output_format=None,
        parallel_mode=ParallelMode.COLLECTIVE,
        operation=RestartV3(bit_identical=True),
    )

    plan = plan_accepted_side_effects(
        serial_runtime, ConsumerGraph((manifest,)), _moment(clock))
    assert len(plan.effects) == 1
    assert manifest.operation_data["supports_singleton_collective"] is True


def test_console_root_is_the_serial_process_for_a_singleton_run():
    _, serial_runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("console-singleton"))
    manifest = ConsumerManifest(
        handle=Handle(
            "console", kind="consumer", owner=OwnerPath.consumer("console-singleton")),
        kind=ConsumerKind.DIAGNOSTIC,
        quantities=(),
        schedule=Schedule(Every(AcceptedStep(clock), 2)),
        target_uri="console/diagnostics",
        output_format=None,
        parallel_mode=ParallelMode.ROOT,
        operation=ConsolePresentation(template=None, handler=None),
    )

    plan = plan_accepted_side_effects(
        serial_runtime, ConsumerGraph((manifest,)), _moment(clock))
    assert len(plan.effects) == 1
    assert manifest.operation_data["supports_singleton_collective"] is True


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
    assert "OSError: injected publication failure" in str(failure.value)
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


def test_every_dt_is_due_only_on_reached_physical_thresholds_and_deduplicates():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("physical-output-cadence"))
    manifest = replace(
        _manifest_for(runtime, "physical", clock),
        schedule=every_dt(0.1, clock=clock),
    )
    graph = ConsumerGraph((manifest,))
    cursors = ConsumerCursorSet()

    assert (
        plan_accepted_side_effects(
            runtime, graph, _moment(clock, step=1, physical_time=0.1 - 1.0e-8), cursors
        ).effects
        == ()
    )
    due = plan_accepted_side_effects(
        runtime, graph, _moment(clock, step=2, physical_time=0.1), cursors
    )
    assert len(due.effects) == 1
    accepted = ConsumerTransaction(due, cursors, _Publisher()).accept()
    assert accepted.cursors.for_consumer(manifest.qualified_id).committed_samples == 1

    duplicate = plan_accepted_side_effects(
        runtime, graph, _moment(clock, step=2, physical_time=0.1), accepted.cursors
    )
    assert duplicate.effects == ()
    assert (
        plan_accepted_side_effects(
            runtime, graph, _moment(clock, step=3, physical_time=0.15), accepted.cursors
        ).effects
        == ()
    )
    second = plan_accepted_side_effects(
        runtime, graph, _moment(clock, step=4, physical_time=0.2), accepted.cursors
    )
    assert len(second.effects) == 1

    assert manifest.schedule.trigger.consumer_next_deadline(
        physical_time_hex=(0.2).hex()
    ) == (0.3).hex()
    assert (
        plan_accepted_side_effects(
            runtime,
            graph,
            _moment(clock, step=5, physical_time=math.nextafter(0.3, -math.inf)),
            accepted.cursors,
        ).effects
        == ()
    )
    third = plan_accepted_side_effects(
        runtime, graph, _moment(clock, step=6, physical_time=0.3), accepted.cursors
    )
    assert len(third.effects) == 1


def test_every_dt_deduplicates_one_lattice_index_across_nearby_deadlines():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("nearby-physical-cadences"))
    first = replace(
        _manifest_for(runtime, "decimal-cadence", clock),
        schedule=every_dt(0.1, clock=clock),
    )
    second = replace(
        _manifest_for(runtime, "nextafter-cadence", clock),
        schedule=every_dt(math.nextafter(0.1, math.inf), clock=clock),
    )
    graph = ConsumerGraph((first, second))
    cursors = ConsumerCursorSet()

    decimal_landing = plan_accepted_side_effects(
        runtime, graph, _moment(clock, step=3, physical_time=0.3), cursors)
    assert [effect.consumer_id for effect in decimal_landing.effects] == [
        first.qualified_id]
    accepted = ConsumerTransaction(decimal_landing, cursors, _Publisher()).accept()

    nextafter_landing = plan_accepted_side_effects(
        runtime,
        graph,
        _moment(clock, step=4, physical_time=math.nextafter(0.3, math.inf)),
        accepted.cursors,
    )
    assert [effect.consumer_id for effect in nextafter_landing.effects] == [
        second.qualified_id]


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


def test_seal_explicitly_finalizes_every_accepted_publication_once():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-finalize"))
    manifest = _manifest_for(runtime, "finalize", clock)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(
        runtime, ConsumerGraph((manifest,)), _moment(clock), cursors)
    publisher = _Publisher()
    transaction = ConsumerTransaction(plan, cursors, publisher)

    transaction.accept()
    assert publisher.finalize_calls == 0
    assert transaction.seal() == ()
    assert publisher.finalize_calls == 1
    assert transaction.abort() is None
    assert transaction.seal() == ()
    assert publisher.finalize_calls == 1


def test_seal_failure_is_non_compensating_diagnostic_and_retryable():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-finalize-retry"))
    manifest = _manifest_for(runtime, "finalize-retry", clock)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(
        runtime, ConsumerGraph((manifest,)), _moment(clock), cursors)
    publisher = _Publisher(fail_finalizations=1)
    transaction = ConsumerTransaction(plan, cursors, publisher)

    transaction.accept()
    artifact = next(iter(publisher.artifacts))
    diagnostics = transaction.seal()
    assert len(diagnostics) == 1 and "injected finalization failure" in diagnostics[0]
    assert publisher.artifacts == {artifact}
    assert transaction.abort() is None

    assert transaction.seal() == ()
    assert publisher.finalize_calls == 2
    assert publisher.artifacts == {artifact}


def test_seal_rejects_a_non_none_finalizer_result_without_compensation():
    _, runtime = _runtime()
    clock = Clock("solution", owner=OwnerPath.consumer("adc-685-finalize-result"))
    manifest = _manifest_for(runtime, "finalize-result", clock)
    cursors = ConsumerCursorSet()
    plan = plan_accepted_side_effects(
        runtime, ConsumerGraph((manifest,)), _moment(clock), cursors)
    publisher = _Publisher(non_none_finalize=True)
    transaction = ConsumerTransaction(plan, cursors, publisher)

    transaction.accept()
    diagnostics = transaction.seal()
    assert len(diagnostics) == 1
    assert "PreparedPublication.finalize() must return None" in diagnostics[0]
    assert publisher.artifacts
    assert transaction.abort() is None


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
