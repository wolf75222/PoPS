"""Immutable contracts produced by the pure RuntimeInstance planning phase."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pops.identity import Identity

from ._runtime_plan_io import (
    DataContract as _DataContract,
    RuntimePlanningError,
    canonical_text as _text,
    freeze_data as _freeze,
    frozen_rows as _rows,
    nonnegative_integer as _index,
    positive_integer as _positive,
    string_tuple as _strings,
    thaw_data as _thaw,
    refuse,
)


RUNTIME_PLAN_SCHEMA_VERSION = 1
_ACCESS_MODES = frozenset(("read", "write"))
_DETERMINISM = frozenset(("bitwise", "reproducible", "statistical", "nondeterministic"))


@dataclass(frozen=True, slots=True)
class FieldAccess:
    resource: str
    mode: str
    memory_space: str

    def __post_init__(self) -> None:
        _text(self.resource, "FieldAccess.resource")
        if self.mode not in _ACCESS_MODES:
            raise ValueError("FieldAccess.mode must be read or write")
        _text(self.memory_space, "FieldAccess.memory_space")

    def to_data(self) -> dict[str, str]:
        return {"resource": self.resource, "mode": self.mode, "memory_space": self.memory_space}


@dataclass(frozen=True, slots=True)
class RuntimeCall(_DataContract):
    _domain: ClassVar[str] = "runtime-call"
    ordinal: int
    block_id: str
    component_id: str
    component_type: str
    component_manifest_identity: Identity
    layout_id: str
    entry_point: str
    reads: tuple[FieldAccess, ...]
    writes: tuple[FieldAccess, ...]
    requirements: tuple[Any, ...]
    effects: tuple[Any, ...]
    clocks: tuple[Any, ...]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _index(self.ordinal, "RuntimeCall.ordinal")
        for name in ("block_id", "component_id", "component_type", "layout_id", "entry_point"):
            _text(getattr(self, name), "RuntimeCall.%s" % name)
        if type(self.component_manifest_identity) is not Identity or self.component_manifest_identity.domain != "component-semantics":
            raise TypeError("RuntimeCall requires a component-semantics Identity")
        for name, mode in (("reads", "read"), ("writes", "write")):
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or any(type(row) is not FieldAccess or row.mode != mode for row in rows):
                raise TypeError("RuntimeCall.%s must contain exact %s FieldAccess values" % (name, mode))
        for name in ("requirements", "effects", "clocks"):
            object.__setattr__(self, name, _rows(getattr(self, name), "RuntimeCall.%s" % name))
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "block_id": self.block_id,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_manifest_identity": self.component_manifest_identity.to_data(),
            "layout_id": self.layout_id,
            "entry_point": self.entry_point,
            "reads": [row.to_data() for row in self.reads],
            "writes": [row.to_data() for row in self.writes],
            "requirements": _thaw(self.requirements),
            "effects": _thaw(self.effects),
            "clocks": _thaw(self.clocks),
        }


@dataclass(frozen=True, slots=True)
class HaloExchange(_DataContract):
    _domain: ClassVar[str] = "runtime-halo"
    call_id: str
    resource: str
    layout_id: str
    depth: int
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        for name in ("call_id", "resource", "layout_id"):
            _text(getattr(self, name), "HaloExchange.%s" % name)
        _positive(self.depth, "HaloExchange.depth")
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "resource": self.resource,
            "layout_id": self.layout_id,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class LayoutTransfer(_DataContract):
    _domain: ClassVar[str] = "runtime-layout-transfer"
    mapping_id: str
    provider_id: str
    source_layout_id: str
    target_layout_id: str
    channel: str
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "mapping_id",
            "provider_id",
            "source_layout_id",
            "target_layout_id",
            "channel",
        ):
            _text(getattr(self, name), "LayoutTransfer.%s" % name)
        if self.source_layout_id == self.target_layout_id:
            raise ValueError("LayoutTransfer must cross distinct layouts")
        self._seal(self._payload())

    def _payload(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in (
                "mapping_id",
                "provider_id",
                "source_layout_id",
                "target_layout_id",
                "channel",
            )
        }


@dataclass(frozen=True, slots=True)
class Collective(_DataContract):
    _domain: ClassVar[str] = "runtime-collective"
    call_id: str
    resource: str
    operation: str
    strategy: str
    communicator_id: str
    sequence: int
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        for name in ("call_id", "resource", "operation", "strategy", "communicator_id"):
            _text(getattr(self, name), "Collective.%s" % name)
        _index(self.sequence, "Collective.sequence")
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "resource": self.resource,
            "operation": self.operation,
            "strategy": self.strategy,
            "communicator_id": self.communicator_id,
            "sequence": self.sequence,
        }


@dataclass(frozen=True, slots=True)
class Fence(_DataContract):
    _domain: ClassVar[str] = "runtime-fence"
    resource: str
    before_call_id: str
    after_call_id: str
    source_space: str
    target_space: str
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        for name in ("resource", "before_call_id", "after_call_id", "source_space", "target_space"):
            _text(getattr(self, name), "Fence.%s" % name)
        if self.before_call_id == self.after_call_id:
            raise ValueError("Fence cannot split one opaque RuntimeCall")
        if self.source_space == self.target_space:
            raise ValueError("Fence requires distinct memory spaces")
        self._seal(self._payload())

    def _payload(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in (
                "resource",
                "before_call_id",
                "after_call_id",
                "source_space",
                "target_space",
            )
        }


@dataclass(frozen=True, slots=True)
class ClockJoin(_DataContract):
    _domain: ClassVar[str] = "runtime-clock-join"
    call_id: str
    source_clock: str
    target_clock: str
    policy: str
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        for name in ("call_id", "source_clock", "target_clock", "policy"):
            _text(getattr(self, name), "ClockJoin.%s" % name)
        if self.source_clock == self.target_clock:
            raise ValueError("ClockJoin requires distinct clocks")
        self._seal(self._payload())

    def _payload(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in ("call_id", "source_clock", "target_clock", "policy")}


@dataclass(frozen=True, slots=True)
class CommunicationPlan(_DataContract):
    _domain: ClassVar[str] = "communication-plan"
    layout_plan_id: str
    communicator_id: str
    halos: tuple[HaloExchange, ...] = ()
    transfers: tuple[LayoutTransfer, ...] = ()
    collectives: tuple[Collective, ...] = ()
    fences: tuple[Fence, ...] = ()
    clock_joins: tuple[ClockJoin, ...] = ()
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _text(self.layout_plan_id, "CommunicationPlan.layout_plan_id")
        _text(self.communicator_id, "CommunicationPlan.communicator_id")
        checks = (
            ("halos", HaloExchange),
            ("transfers", LayoutTransfer),
            ("collectives", Collective),
            ("fences", Fence),
            ("clock_joins", ClockJoin),
        )
        for name, kind in checks:
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or any(type(row) is not kind for row in rows):
                raise TypeError("CommunicationPlan.%s must contain exact %s values" % (name, kind.__name__))
        if tuple(row.sequence for row in self.collectives) != tuple(range(len(self.collectives))):
            raise ValueError("CommunicationPlan collective sequence must be contiguous")
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "layout_plan_id": self.layout_plan_id,
            "communicator_id": self.communicator_id,
            "halos": [row.to_data() for row in self.halos],
            "transfers": [row.to_data() for row in self.transfers],
            "collectives": [row.to_data() for row in self.collectives],
            "fences": [row.to_data() for row in self.fences],
            "clock_joins": [row.to_data() for row in self.clock_joins],
        }


@dataclass(frozen=True, slots=True)
class ResourceUse:
    resource: str
    memory_space: str
    first_call: int
    last_call: int
    modes: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.resource, "ResourceUse.resource")
        _text(self.memory_space, "ResourceUse.memory_space")
        _index(self.first_call, "ResourceUse.first_call")
        _index(self.last_call, "ResourceUse.last_call")
        if self.last_call < self.first_call:
            raise ValueError("ResourceUse.last_call precedes first_call")
        modes = _strings(self.modes, "ResourceUse.modes")
        if not modes or set(modes) - _ACCESS_MODES or modes != tuple(sorted(modes)):
            raise ValueError("ResourceUse.modes must be sorted read/write values")

    def to_data(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "memory_space": self.memory_space,
            "first_call": self.first_call,
            "last_call": self.last_call,
            "modes": list(self.modes),
        }


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    resource: str
    memory_space: str
    size_bytes: int
    first_call: int
    last_call: int

    def __post_init__(self) -> None:
        _text(self.resource, "BufferAllocation.resource")
        _text(self.memory_space, "BufferAllocation.memory_space")
        _positive(self.size_bytes, "BufferAllocation.size_bytes")
        _index(self.first_call, "BufferAllocation.first_call")
        _index(self.last_call, "BufferAllocation.last_call")
        if self.last_call < self.first_call:
            raise ValueError("BufferAllocation.last_call precedes first_call")

    def to_data(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "memory_space": self.memory_space,
            "size_bytes": self.size_bytes,
            "first_call": self.first_call,
            "last_call": self.last_call,
        }


@dataclass(frozen=True, slots=True)
class ResourcePlan(_DataContract):
    _domain: ClassVar[str] = "resource-plan"
    layout_plan_id: str
    execution_context_identity: Identity
    memory_spaces: tuple[str, ...]
    uses: tuple[ResourceUse, ...]
    buffers: tuple[BufferAllocation, ...]
    mapping_provider_ids: tuple[str, ...]
    fence_ids: tuple[str, ...]
    declared_requirements: tuple[Any, ...]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _text(self.layout_plan_id, "ResourcePlan.layout_plan_id")
        if type(self.execution_context_identity) is not Identity or self.execution_context_identity.domain != "execution-context":
            raise TypeError("ResourcePlan requires an execution-context Identity")
        object.__setattr__(self, "memory_spaces", _strings(self.memory_spaces, "ResourcePlan.memory_spaces"))
        for name, kind in (("uses", ResourceUse), ("buffers", BufferAllocation)):
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or any(type(row) is not kind for row in rows):
                raise TypeError("ResourcePlan.%s must contain exact %s values" % (name, kind.__name__))
        for name in ("mapping_provider_ids", "fence_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), "ResourcePlan.%s" % name))
        object.__setattr__(
            self,
            "declared_requirements",
            _rows(self.declared_requirements, "ResourcePlan.declared_requirements"),
        )
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "layout_plan_id": self.layout_plan_id,
            "execution_context_identity": self.execution_context_identity.to_data(),
            "memory_spaces": list(self.memory_spaces),
            "uses": [row.to_data() for row in self.uses],
            "buffers": [row.to_data() for row in self.buffers],
            "mapping_provider_ids": list(self.mapping_provider_ids),
            "fence_ids": list(self.fence_ids),
            "declared_requirements": _thaw(self.declared_requirements),
        }


@dataclass(frozen=True, slots=True)
class DeterminismGuarantee(_DataContract):
    _domain: ClassVar[str] = "determinism-guarantee"
    classification: str
    scope: tuple[str, ...]
    assumptions: Mapping[str, Any]
    component_evidence: Mapping[str, Any]
    execution_context_identity: Identity
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if self.classification not in _DETERMINISM:
            raise ValueError("unsupported determinism classification %r" % self.classification)
        object.__setattr__(self, "scope", _strings(self.scope, "DeterminismGuarantee.scope"))
        for name in ("assumptions", "component_evidence"):
            value = _freeze(getattr(self, name), "DeterminismGuarantee.%s" % name)
            if not isinstance(value, Mapping):
                raise TypeError("DeterminismGuarantee.%s must be a mapping" % name)
            object.__setattr__(self, name, value)
        if type(self.execution_context_identity) is not Identity or self.execution_context_identity.domain != "execution-context":
            raise TypeError("DeterminismGuarantee requires an execution-context Identity")
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "scope": list(self.scope),
            "assumptions": _thaw(self.assumptions),
            "component_evidence": _thaw(self.component_evidence),
            "execution_context_identity": self.execution_context_identity.to_data(),
        }

    def require_assumptions(self, actual: Mapping[str, Any]) -> None:
        frozen = _freeze(actual, "determinism actual assumptions")
        if not isinstance(frozen, Mapping) or frozen != self.assumptions:
            refuse(
                "determinism_assumption_mismatch",
                "determinism.assumptions",
                "runtime facts do not match the authenticated determinism assumptions",
                evidence={"expected": _thaw(self.assumptions), "actual": _thaw(frozen)},
            )


@dataclass(frozen=True, slots=True)
class RuntimePlanBundle(_DataContract):
    _domain: ClassVar[str] = "runtime-plan-bundle"
    install_identity: Identity
    platform_identity: Identity
    execution_context_identity: Identity
    layout_plan_id: str
    calls: tuple[RuntimeCall, ...]
    communication: CommunicationPlan
    resources: ResourcePlan
    determinism: DeterminismGuarantee
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        domains = (
            (self.install_identity, "bind"),
            (self.platform_identity, "platform-manifest"),
            (self.execution_context_identity, "execution-context"),
        )
        if any(type(value) is not Identity or value.domain != domain for value, domain in domains):
            raise TypeError("RuntimePlanBundle received an identity with the wrong domain")
        _text(self.layout_plan_id, "RuntimePlanBundle.layout_plan_id")
        if not isinstance(self.calls, tuple) or any(type(row) is not RuntimeCall for row in self.calls):
            raise TypeError("RuntimePlanBundle.calls must contain exact RuntimeCall values")
        if tuple(row.ordinal for row in self.calls) != tuple(range(len(self.calls))):
            raise ValueError("RuntimePlanBundle call ordinals must be contiguous")
        if type(self.communication) is not CommunicationPlan or type(self.resources) is not ResourcePlan or type(self.determinism) is not DeterminismGuarantee:
            raise TypeError("RuntimePlanBundle requires exact subordinate plans")
        if self.communication.layout_plan_id != self.layout_plan_id or self.resources.layout_plan_id != self.layout_plan_id:
            raise ValueError("RuntimePlanBundle subordinate layout identities disagree")
        if self.resources.execution_context_identity != self.execution_context_identity or self.determinism.execution_context_identity != self.execution_context_identity:
            raise ValueError("RuntimePlanBundle execution-context identities disagree")
        call_ids = {row.identity.token for row in self.calls}
        referenced = {row.call_id for row in self.communication.halos} | {row.call_id for row in self.communication.collectives} | {row.call_id for row in self.communication.clock_joins} | {row.before_call_id for row in self.communication.fences} | {row.after_call_id for row in self.communication.fences}
        if referenced - call_ids:
            raise ValueError("RuntimePlanBundle communication references unknown RuntimeCalls")
        expected_fences = tuple(row.identity.token for row in self.communication.fences)
        if self.resources.fence_ids != expected_fences:
            raise ValueError("ResourcePlan fence identities disagree with CommunicationPlan")
        self._seal(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
            "install_identity": self.install_identity.to_data(),
            "platform_identity": self.platform_identity.to_data(),
            "execution_context_identity": self.execution_context_identity.to_data(),
            "layout_plan_id": self.layout_plan_id,
            "calls": [row.to_data() for row in self.calls],
            "communication": self.communication.to_data(),
            "resources": self.resources.to_data(),
            "determinism": self.determinism.to_data(),
        }


__all__ = [
    "RUNTIME_PLAN_SCHEMA_VERSION",
    "RuntimePlanningError",
    "FieldAccess",
    "RuntimeCall",
    "HaloExchange",
    "LayoutTransfer",
    "Collective",
    "Fence",
    "ClockJoin",
    "CommunicationPlan",
    "ResourceUse",
    "BufferAllocation",
    "ResourcePlan",
    "DeterminismGuarantee",
    "RuntimePlanBundle",
]
