"""Pure ConsumerGraph lowering against one authenticated RuntimePlanBundle."""

from __future__ import annotations

from typing import Any

from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringCoverageRow
from pops.identity import Identity, make_identity
from pops.time.schedule_protocol import UnresolvedScheduleCondition

from ._consumer_contracts import (
    ConsumerCursorSet,
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerMoment,
    ConsumerQuantity,
    ParallelMode,
    ScheduleCursor,
)
from ._consumer_effects import (
    AcceptedSideEffect,
    ConsumerFieldResolution,
    ConsumerPayload,
    ConsumerResourceBinding,
    EffectPlan,
    PublicationTarget,
)
from ._runtime_plan_contracts import RuntimePlanBundle, refuse


def _schedule_coordinate(manifest: ConsumerManifest, moment: ConsumerMoment) -> int | None:
    domain = manifest.schedule.domain
    if domain.clock != moment.point.clock:
        return None
    coordinate = domain.consumer_coordinate(moment)
    if coordinate is not None and (isinstance(coordinate, bool) or type(coordinate) is not int):
        raise TypeError(
            "Domain.consumer_coordinate() must return an exact int or None, got %s"
            % type(coordinate).__name__)
    return coordinate


def _is_due(manifest: ConsumerManifest, moment: ConsumerMoment) -> bool:
    trigger = manifest.schedule.trigger
    coordinate = _schedule_coordinate(manifest, moment)
    if coordinate is None:
        return False
    try:
        due = trigger.consumer_due(coordinate, moment)
    except UnresolvedScheduleCondition as exc:
        refuse(
            "unresolved_consumer_condition",
            "consumer[%s].schedule.condition" % manifest.qualified_id,
            "schedule conditions must be resolved explicitly before ConsumerGraph planning",
            evidence=exc.condition_type,
        )
    if type(due) is not bool:
        raise TypeError(
            "Trigger.consumer_due() must return an exact bool, got %s"
            % type(due).__name__)
    return due


def _occurrence(manifest: ConsumerManifest, moment: ConsumerMoment) -> Identity:
    domain = manifest.schedule.domain
    evidence: dict[str, Any] = {
        "coordinate": _schedule_coordinate(manifest, moment),
        "point": moment.point.to_data(),
    }
    domain_evidence = domain.consumer_occurrence_evidence(moment)
    if type(domain_evidence) is not dict:
        raise TypeError(
            "Domain.consumer_occurrence_evidence() must return an exact dict")
    evidence.update(domain_evidence)
    return make_identity(
        "consumer-occurrence",
        {
            "consumer_id": manifest.qualified_id,
            "schedule": manifest.schedule.to_data(),
            "evidence": evidence,
        },
    )


def _resource_bindings(
    runtime: RuntimePlanBundle,
    manifest: ConsumerManifest,
) -> tuple[ConsumerResourceBinding, ...]:
    result = []
    for quantity in manifest.quantities:
        accesses = [
            access
            for call in runtime.calls
            if call.layout_id == quantity.layout_id
            for access in (*call.reads, *call.writes)
            if access.resource == quantity.runtime_resource
        ]
        if not accesses:
            refuse(
                "consumer_resource_unavailable",
                "consumer[%s].quantity[%s]" % (
                    manifest.qualified_id, quantity.identity.token),
                "consumer quantity has no exact runtime resource/layout provider",
                evidence={
                    "resource": quantity.runtime_resource,
                    "layout_id": quantity.layout_id,
                },
            )
        collective_ids = tuple(sorted(
            row.identity.token for row in runtime.communication.collectives
            if row.resource == quantity.runtime_resource
        ))
        if manifest.parallel_mode is ParallelMode.COLLECTIVE and not collective_ids:
            refuse(
                "consumer_collective_unavailable",
                "consumer[%s].quantity[%s]" % (
                    manifest.qualified_id, quantity.identity.token),
                "collective consumer quantity lacks an authenticated collective plan",
                evidence=quantity.runtime_resource,
            )
        result.append(ConsumerResourceBinding(
            quantity.identity,
            quantity.reference.qualified_id,
            quantity.runtime_resource,
            quantity.layout_id,
            quantity.levels,
            tuple(sorted({access.memory_space for access in accesses})),
            collective_ids,
        ))
    return tuple(result)


def _field_consumer(kind: ConsumerKind) -> Any:
    from pops.fields import FieldConsumer

    if kind in (ConsumerKind.DIAGNOSTIC, ConsumerKind.MONITOR):
        return FieldConsumer.DIAGNOSTIC
    return FieldConsumer.OUTPUT


def _field_resolution(
    manifest: ConsumerManifest,
    quantity: ConsumerQuantity,
    moment: ConsumerMoment,
) -> ConsumerFieldResolution | None:
    from pops.fields import (
        FieldReadError,
        RecomputeField,
        UseHeldField,
        UseMaterializedField,
    )

    context = quantity.field_context
    if context is None:
        return None
    try:
        layout = moment.layout_for(quantity.layout_id)
    except KeyError:
        refuse(
            "consumer_layout_generation_missing",
            "consumer[%s].quantity[%s].layout" % (
                manifest.qualified_id, quantity.identity.token),
            "field consumption requires the current exact LayoutBinding",
            evidence=quantity.layout_id,
        )
    try:
        resolution = context.resolve_read(
            _field_consumer(manifest.kind),
            at=moment.point,
            layout=layout,
            policy=quantity.field_policy,
        )
    except FieldReadError as exc:
        refuse(
            "consumer_field_not_fresh",
            "consumer[%s].quantity[%s].field_context" % (
                manifest.qualified_id, quantity.identity.token),
            str(exc),
            evidence=context.inspect(),
        )
    if type(resolution) is UseMaterializedField:
        return ConsumerFieldResolution(
            quantity.identity, resolution.context_identity, "materialized",
            {"requested_point": moment.point.to_data(), "layout": layout.to_data()},
        )
    if type(resolution) is UseHeldField:
        return ConsumerFieldResolution(
            quantity.identity, resolution.context_identity, "held",
            {"source_point": resolution.source_point.to_data(),
             "requested_point": resolution.requested_point.to_data(),
             "layout": layout.to_data()},
        )
    if type(resolution) is RecomputeField:
        return ConsumerFieldResolution(
            quantity.identity, resolution.context_identity, "recompute",
            {"consumer": resolution.consumer.value,
             "requested_point": resolution.requested_point.to_data(),
             "layout": layout.to_data(),
             "explicit": True},
        )
    raise TypeError("unsupported field resolution %s" % type(resolution).__name__)


def _coverage_active(
    manifests: tuple[ConsumerManifest, ...],
    bindings: dict[str, tuple[ConsumerResourceBinding, ...]],
) -> LoweringCoverageReport:
    rows = []
    for manifest in manifests:
        for binding in bindings[manifest.qualified_id]:
            targets = [
                "runtime-resource:%s@%s" % (binding.runtime_resource, binding.layout_id),
                *["runtime-collective:%s" % value for value in binding.collective_ids],
            ]
            rows.append(LoweringCoverageRow(
                source="consumer-quantity:%s:%s" % (
                    manifest.qualified_id, binding.quantity_identity.token),
                disposition="lowered",
                targets=tuple(targets),
            ))
    return LoweringCoverageReport(rows)


def plan_accepted_side_effects(
    runtime_plan: Any,
    graph: Any,
    moment: Any,
    cursors: Any = None,
) -> EffectPlan:
    """Plan due samples without preparing, publishing, or advancing any runtime state."""
    if type(runtime_plan) is not RuntimePlanBundle:
        raise TypeError("consumer planning requires an exact RuntimePlanBundle")
    if type(graph) is not ConsumerGraph:
        raise TypeError("consumer planning requires an exact ConsumerGraph")
    if not graph.is_resolved:
        raise TypeError(
            "consumer planning requires the resolved ConsumerGraph from pops.resolve")
    if type(moment) is not ConsumerMoment:
        raise TypeError("consumer planning requires an exact ConsumerMoment")
    if cursors is None:
        cursors = ConsumerCursorSet()
    if type(cursors) is not ConsumerCursorSet:
        raise TypeError("consumer planning requires an exact ConsumerCursorSet")

    active = tuple(
        manifest for manifest in graph.nodes
        if manifest.schedule.domain.clock == moment.point.clock
    )
    bindings = {
        manifest.qualified_id: _resource_bindings(runtime_plan, manifest)
        for manifest in active
    }
    coverage = _coverage_active(active, bindings)
    effects = []
    for manifest in graph.topology:
        if not _is_due(manifest, moment):
            continue
        occurrence = _occurrence(manifest, moment)
        before = cursors.for_consumer(manifest.qualified_id)
        if before.last_occurrence == occurrence.token:
            continue
        fields = tuple(
            resolution
            for quantity in manifest.quantities
            if (resolution := _field_resolution(manifest, quantity, moment)) is not None
        )
        payload = ConsumerPayload(
            runtime_plan.identity,
            occurrence,
            bindings[manifest.qualified_id],
            fields,
        )
        after = ScheduleCursor(
            manifest.qualified_id,
            occurrence.token,
            before.committed_samples + 1,
        )
        effects.append(AcceptedSideEffect(
            len(effects),
            manifest.qualified_id,
            manifest.identity,
            PublicationTarget(
                manifest.target_uri,
                manifest.output_format_data,
                manifest.operation_data,
                manifest.parallel_mode,
            ),
            payload,
            manifest.failure_action,
            before,
            after,
        ))
    return EffectPlan(graph.identity, runtime_plan.identity, tuple(effects), coverage)


__all__ = ["plan_accepted_side_effects"]
