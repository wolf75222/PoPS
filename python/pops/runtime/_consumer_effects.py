"""Immutable payload and effect values derived from a ConsumerGraph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pops.codegen.lowering_coverage import LoweringCoverageReport
from pops.identity import Identity, make_identity

from ._consumer_contracts import (
    ConsumerFailureAction,
    FailRun,
    ParallelMode,
    Retry,
    ScheduleCursor,
    SkipSampleReported,
)
from ._runtime_plan_io import freeze_data, thaw_data


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _index(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("%s must be an integer >= 0" % where)
    return value


def _identity(value: Any, domain: str, where: str) -> Identity:
    if type(value) is not Identity or value.domain != domain:
        raise TypeError("%s must be an exact %s Identity" % (where, domain))
    return value


_FAILURE_ACTIONS = (FailRun, Retry, SkipSampleReported)


@dataclass(frozen=True, slots=True)
class ConsumerResourceBinding:
    quantity_identity: Identity
    reference_id: str
    runtime_resource: str
    layout_id: str
    levels: tuple[int, ...]
    memory_spaces: tuple[str, ...]
    collective_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.quantity_identity, "consumer-quantity", "ConsumerResourceBinding.quantity_identity")
        for name in ("reference_id", "runtime_resource", "layout_id"):
            _text(getattr(self, name), "ConsumerResourceBinding.%s" % name)
        for name in ("levels", "memory_spaces", "collective_ids"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError("ConsumerResourceBinding.%s must be a tuple" % name)
        if self.levels != tuple(sorted(set(self.levels))) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.levels):
            raise ValueError("ConsumerResourceBinding.levels must be sorted unique integers >= 0")
        for name in ("memory_spaces", "collective_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))) or any(
                    not isinstance(value, str) or not value for value in values):
                raise ValueError(
                    "ConsumerResourceBinding.%s must be sorted unique text" % name)
        if not self.memory_spaces:
            raise ValueError("ConsumerResourceBinding.memory_spaces cannot be empty")

    def to_data(self) -> dict[str, Any]:
        return {
            "quantity_identity": self.quantity_identity.to_data(),
            "reference_id": self.reference_id,
            "runtime_resource": self.runtime_resource,
            "layout_id": self.layout_id,
            "levels": list(self.levels),
            "memory_spaces": list(self.memory_spaces),
            "collective_ids": list(self.collective_ids),
        }


@dataclass(frozen=True, slots=True)
class ConsumerFieldResolution:
    quantity_identity: Identity
    context_identity: Identity
    action: str
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identity(self.quantity_identity, "consumer-quantity", "ConsumerFieldResolution.quantity_identity")
        if type(self.context_identity) is not Identity:
            raise TypeError("ConsumerFieldResolution.context_identity must be an exact Identity")
        if self.action not in ("materialized", "held", "recompute"):
            raise ValueError("ConsumerFieldResolution.action is unsupported")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("ConsumerFieldResolution.evidence must be a mapping")
        object.__setattr__(
            self, "evidence", freeze_data(self.evidence, "ConsumerFieldResolution.evidence"))

    def to_data(self) -> dict[str, Any]:
        return {
            "quantity_identity": self.quantity_identity.to_data(),
            "context_identity": self.context_identity.to_data(),
            "action": self.action,
            "evidence": thaw_data(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class PublicationTarget:
    uri: str
    output_format: str
    parallel_mode: ParallelMode

    def __post_init__(self) -> None:
        _text(self.uri, "PublicationTarget.uri")
        _text(self.output_format, "PublicationTarget.output_format")
        if type(self.parallel_mode) is not ParallelMode:
            raise TypeError("PublicationTarget.parallel_mode must be an exact ParallelMode")

    def to_data(self) -> dict[str, Any]:
        return {"uri": self.uri, "format": self.output_format,
                "parallel_mode": self.parallel_mode.value}


@dataclass(frozen=True, slots=True)
class ConsumerPayload:
    runtime_plan_identity: Identity
    occurrence_identity: Identity
    resources: tuple[ConsumerResourceBinding, ...]
    fields: tuple[ConsumerFieldResolution, ...]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.runtime_plan_identity, "runtime-plan-bundle", "ConsumerPayload.runtime_plan_identity")
        _identity(self.occurrence_identity, "consumer-occurrence", "ConsumerPayload.occurrence_identity")
        if not isinstance(self.resources, tuple) or any(
                type(value) is not ConsumerResourceBinding for value in self.resources):
            raise TypeError("ConsumerPayload.resources must contain exact resource bindings")
        if not isinstance(self.fields, tuple) or any(
                type(value) is not ConsumerFieldResolution for value in self.fields):
            raise TypeError("ConsumerPayload.fields must contain exact field resolutions")
        object.__setattr__(self, "identity", make_identity("consumer-payload", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "runtime_plan_identity": self.runtime_plan_identity.to_data(),
            "occurrence_identity": self.occurrence_identity.to_data(),
            "resources": [value.to_data() for value in self.resources],
            "fields": [value.to_data() for value in self.fields],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


@dataclass(frozen=True, slots=True)
class AcceptedSideEffect:
    ordinal: int
    consumer_id: str
    manifest_identity: Identity
    target: PublicationTarget
    payload: ConsumerPayload
    failure_action: ConsumerFailureAction
    cursor_before: ScheduleCursor
    cursor_after: ScheduleCursor
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _index(self.ordinal, "AcceptedSideEffect.ordinal")
        _text(self.consumer_id, "AcceptedSideEffect.consumer_id")
        _identity(self.manifest_identity, "consumer-manifest", "AcceptedSideEffect.manifest_identity")
        if type(self.target) is not PublicationTarget or type(self.payload) is not ConsumerPayload:
            raise TypeError("AcceptedSideEffect requires exact target and payload values")
        if type(self.failure_action) not in _FAILURE_ACTIONS:
            raise TypeError("AcceptedSideEffect.failure_action is unsupported")
        if type(self.cursor_before) is not ScheduleCursor or type(self.cursor_after) is not ScheduleCursor:
            raise TypeError("AcceptedSideEffect cursors must be exact ScheduleCursor values")
        if self.cursor_before.consumer_id != self.consumer_id or self.cursor_after.consumer_id != self.consumer_id:
            raise ValueError("AcceptedSideEffect cursor consumer ids disagree")
        if self.cursor_after.last_occurrence != self.payload.occurrence_identity.token \
                or self.cursor_after.committed_samples != self.cursor_before.committed_samples + 1:
            raise ValueError("AcceptedSideEffect cursor advancement is not the exact next sample")
        object.__setattr__(self, "identity", make_identity("accepted-side-effect", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "consumer_id": self.consumer_id,
            "manifest_identity": self.manifest_identity.to_data(),
            "target": self.target.to_data(),
            "payload": self.payload.to_data(),
            "failure_action": self.failure_action.to_data(),
            "cursor_before": self.cursor_before.to_data(),
            "cursor_after": self.cursor_after.to_data(),
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


@dataclass(frozen=True, slots=True)
class EffectPlan:
    graph_identity: Identity
    runtime_plan_identity: Identity
    effects: tuple[AcceptedSideEffect, ...]
    lowering_coverage: LoweringCoverageReport
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.graph_identity, "consumer-graph", "EffectPlan.graph_identity")
        _identity(self.runtime_plan_identity, "runtime-plan-bundle", "EffectPlan.runtime_plan_identity")
        if not isinstance(self.effects, tuple) or any(
                type(value) is not AcceptedSideEffect for value in self.effects):
            raise TypeError("EffectPlan.effects must contain exact AcceptedSideEffect values")
        if tuple(value.ordinal for value in self.effects) != tuple(range(len(self.effects))):
            raise ValueError("EffectPlan effect ordinals must be contiguous")
        payloads = [value.payload.identity for value in self.effects]
        if len(payloads) != len(set(payloads)):
            raise ValueError("EffectPlan contains a duplicate consumer payload")
        if type(self.lowering_coverage) is not LoweringCoverageReport:
            raise TypeError("EffectPlan requires an exact LoweringCoverageReport")
        object.__setattr__(self, "identity", make_identity("consumer-effect-plan", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "graph_identity": self.graph_identity.to_data(),
            "runtime_plan_identity": self.runtime_plan_identity.to_data(),
            "effects": [value.to_data() for value in self.effects],
            "lowering_coverage": self.lowering_coverage.to_data(),
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


__all__ = [
    "AcceptedSideEffect", "ConsumerFieldResolution", "ConsumerPayload",
    "ConsumerResourceBinding", "EffectPlan", "PublicationTarget",
]
