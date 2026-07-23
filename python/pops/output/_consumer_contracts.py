"""Immutable ConsumerGraph and accepted-side-effect planning values."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pops.identity import Identity, make_identity
from pops.model import Handle
from pops.time import EventHandle, Schedule, StagePoint, TimePoint

from pops._frozen_data import freeze_data, thaw_data

if TYPE_CHECKING:
    from pops.fields import FieldContext, FieldReadPolicy, LayoutBinding


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _index(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("%s must be an integer >= 0" % where)
    return value


def _nonnegative_binary64_hex(value: Any, where: str) -> str:
    """Normalize an exact finite binary64 value for identity-bearing manifests."""
    if isinstance(value, bool):
        raise TypeError("%s must be a finite number >= 0" % where)
    if isinstance(value, str):
        try:
            number = float.fromhex(value)
        except (OverflowError, ValueError) as exc:
            raise TypeError("%s must be a canonical float.hex() string" % where) from exc
        if number.hex() != value:
            raise ValueError("%s must be a canonical float.hex() string" % where)
    elif isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("%s must be a finite number >= 0" % where) from exc
    else:
        raise TypeError("%s must be a finite number >= 0" % where)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("%s must be a finite number >= 0" % where)
    return number.hex()


def _exact_handle(value: Any, kind: str | None, where: str) -> Handle:
    if not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s must be a canonical Handle" % where)
    if kind is not None and value.kind != kind:
        raise TypeError("%s Handle kind must be %r" % (where, kind))
    return value


def _provider_data(value: Any, *, where: str, methods: tuple[str, ...]) -> Mapping[str, Any]:
    if getattr(value, "__pops_ir_immutable__", False) is not True:
        raise TypeError("%s must declare immutable semantic state" % where)
    consumer_data = getattr(value, "consumer_data", None)
    if not callable(consumer_data) or any(not callable(getattr(value, name, None)) for name in methods):
        raise TypeError(
            "%s must implement consumer_data() and %s"
            % (where, "/".join("%s()" % name for name in methods))
        )
    first, second = consumer_data(), consumer_data()
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("%s consumer_data() must return one deterministic dict" % where)
    if first.get("schema_version") != 1:
        raise ValueError("%s consumer_data schema_version must be 1" % where)
    provider_id, extension = first.get("provider_id"), first.get("extension")
    _text(provider_id, "%s.consumer_data.provider_id" % where)
    if not isinstance(extension, str) or not extension.startswith(".") or "/" in extension:
        raise TypeError("%s consumer_data.extension must be a canonical file suffix" % where)
    return freeze_data(first, "%s.consumer_data" % where)


def _observer_provider_data(value: Any, *, where: str) -> Mapping[str, Any]:
    """Authenticate a non-file, irreversible monitor operation provider."""
    if getattr(value, "__pops_ir_immutable__", False) is not True:
        raise TypeError("%s must declare immutable semantic state" % where)
    consumer_data = getattr(value, "consumer_data", None)
    if not callable(consumer_data) or not callable(getattr(value, "open_session", None)):
        raise TypeError(
            "%s must implement consumer_data() and open_session()" % where)
    first, second = consumer_data(), consumer_data()
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("%s consumer_data() must return one deterministic dict" % where)
    expected = {
        "schema_version", "provider_id", "parallel_mode", "queue_capacity",
        "max_attempts", "on_failure", "durability", "observer",
    }
    if set(first) != expected or first["schema_version"] != 1:
        raise ValueError("%s consumer_data has an unsupported live-observer schema" % where)
    _text(first["provider_id"], "%s.consumer_data.provider_id" % where)
    if first["parallel_mode"] not in {"serial", "root", "per_rank", "collective"}:
        raise ValueError("%s parallel_mode is unsupported" % where)
    for name in ("queue_capacity", "max_attempts"):
        value = first[name]
        if isinstance(value, bool) or type(value) is not int or value < 1:
            raise ValueError("%s %s must be an integer >= 1" % (where, name))
    if first["on_failure"] not in (
            {"action": "raise_on_flush"}, {"action": "report_only"}):
        raise ValueError("%s on_failure has an unsupported policy" % where)
    durability = first["durability"]
    if durability is not None and (
            type(durability) is not dict
            or set(durability) != {
                "schema_version", "kind", "root", "sync", "recover", "delivery"
            }
            or durability["schema_version"] != 1
            or durability["kind"] != "durable_observer_journal"
            or durability["sync"] not in {"fsync", "none"}
            or durability["recover"] not in {"automatic", "manual"}
            or durability["delivery"] != "at_least_once_after_handoff"
            or not isinstance(durability["root"], str)
            or not durability["root"]):
        raise ValueError("%s durability has an unsupported journal policy" % where)
    if type(first["observer"]) is not dict:
        raise TypeError("%s observer declaration must be an exact dict" % where)
    return freeze_data(first, "%s.consumer_data" % where)


def _console_provider_data(value: Any, *, where: str) -> Mapping[str, Any]:
    """Authenticate the Python-only renderer of a rank-zero diagnostic consumer."""
    if getattr(value, "__pops_ir_immutable__", False) is not True:
        raise TypeError("%s must declare immutable semantic state" % where)
    consumer_data = getattr(value, "consumer_data", None)
    if not callable(consumer_data) or not callable(getattr(value, "emit", None)):
        raise TypeError("%s must implement consumer_data() and emit()" % where)
    first, second = consumer_data(), consumer_data()
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("%s consumer_data() must return one deterministic dict" % where)
    expected = {
        "schema_version", "provider_id", "parallel_mode", "template", "handler",
    }
    if set(first) != expected or first["schema_version"] != 1:
        raise ValueError("%s consumer_data has an unsupported console schema" % where)
    if first["provider_id"] != "pops.output.console-presentation.v1":
        raise ValueError("%s has an unsupported console provider" % where)
    if first["parallel_mode"] != "root":
        raise ValueError("%s console parallel_mode must be root" % where)
    template, handler = first["template"], first["handler"]
    if (template is None) == (handler is None):
        raise ValueError("%s must declare exactly one console presentation" % where)
    if template is not None and (not isinstance(template, str) or not template):
        raise TypeError("%s template must be non-empty text" % where)
    if handler is not None and (
            type(handler) is not dict
            or set(handler) != {"module", "qualname"}
            or any(not isinstance(item, str) or not item for item in handler.values())):
        raise TypeError("%s handler must be a canonical function reference" % where)
    return freeze_data(first, "%s.consumer_data" % where)


def validate_checkpoint_snapshot(value: Any, *, where: str = "checkpoint snapshot") -> Any:
    """Require the compensating protocol used before and after checkpoint publication."""
    missing = tuple(
        name for name in ("discard", "rollback")
        if not callable(getattr(value, name, None))
    )
    if missing:
        raise TypeError(
            "%s must implement compensating %s()"
            % (where, "/".join(missing)))
    return value


class ConsumerKind(Enum):
    DIAGNOSTIC = "diagnostic"
    SCIENTIFIC_OUTPUT = "scientific_output"
    CHECKPOINT = "checkpoint"
    MONITOR = "monitor"


class ParallelMode(Enum):
    SERIAL = "serial"
    ROOT = "root"
    COLLECTIVE = "collective"
    PER_RANK = "per_rank"


class ConsumerFailureAction:
    """Closed failure decision applied to one consumer sample."""

    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FailRun(ConsumerFailureAction):
    def to_data(self) -> dict[str, Any]:
        return {"action": "fail_run"}


@dataclass(frozen=True, slots=True)
class Retry(ConsumerFailureAction):
    max_attempts: int

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) \
                or self.max_attempts < 2:
            raise ValueError("Retry.max_attempts must be an integer >= 2")

    def to_data(self) -> dict[str, Any]:
        return {"action": "retry", "max_attempts": self.max_attempts}


@dataclass(frozen=True, slots=True)
class SkipSampleReported(ConsumerFailureAction):
    def to_data(self) -> dict[str, Any]:
        return {"action": "skip_sample_reported"}


_FAILURE_ACTIONS = (FailRun, Retry, SkipSampleReported)


@dataclass(frozen=True, slots=True)
class ConsumerQuantity:
    """One owner-qualified runtime resource selected by a consumer."""

    reference: Handle
    runtime_resource: str
    layout_id: str
    levels: tuple[int, ...] = ()
    field_context: FieldContext | None = None
    field_policy: FieldReadPolicy | None = None
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _exact_handle(self.reference, None, "ConsumerQuantity.reference")
        _text(self.runtime_resource, "ConsumerQuantity.runtime_resource")
        _text(self.layout_id, "ConsumerQuantity.layout_id")
        if not isinstance(self.levels, tuple):
            raise TypeError("ConsumerQuantity.levels must be a tuple")
        levels = tuple(_index(value, "ConsumerQuantity.levels[]") for value in self.levels)
        if levels != tuple(sorted(set(levels))):
            raise ValueError("ConsumerQuantity.levels must be sorted and unique")
        if self.field_context is not None:
            from pops.fields import FieldContext

            if type(self.field_context) is not FieldContext:
                raise TypeError("ConsumerQuantity.field_context must be an exact FieldContext")
            if self.reference != self.field_context.operator:
                raise ValueError("ConsumerQuantity field reference and FieldContext disagree")
            if self.field_context.layout.layout.qualified_id != self.layout_id:
                raise ValueError("ConsumerQuantity layout and FieldContext layout disagree")
        if self.field_policy is not None:
            from pops.fields import FieldReadPolicy

            if not isinstance(self.field_policy, FieldReadPolicy):
                raise TypeError("ConsumerQuantity.field_policy must be a FieldReadPolicy")
            if self.field_context is None:
                raise ValueError("a field_policy requires an exact field_context")
        object.__setattr__(self, "identity", make_identity("consumer-quantity", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "reference": self.reference.canonical_identity(),
            "runtime_resource": self.runtime_resource,
            "layout_id": self.layout_id,
            "levels": list(self.levels),
            "field_context": self.field_context.to_data() if self.field_context else None,
            "field_policy": self.field_policy.to_data() if self.field_policy else None,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


_DIAGNOSTIC_REDUCTIONS = frozenset({
    "sum", "abs_sum", "sum_sq", "min", "max", "abs_max", "step_change_l2",
})
_DIAGNOSTIC_TRANSFORMS = frozenset({"identity", "sqrt"})
_DIAGNOSTIC_COLLECTIVES = {
    "sum": "global_sum",
    "abs_sum": "global_sum",
    "sum_sq": "global_sum",
    "min": "global_min",
    "max": "global_max",
    "abs_max": "global_max",
    "step_change_l2": "global_sum",
}


def _diagnostic_execution(value: Any) -> Mapping[str, Any]:
    """Validate and freeze the closed diagnostic reduction instruction schema."""
    if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "role", "operations", "conservation"}:
        raise TypeError("DiagnosticQuantity.execution has an unknown schema")
    if value["schema_version"] != 1:
        raise ValueError("DiagnosticQuantity.execution schema_version must be 1")
    role = value["role"]
    if role is not None:
        _text(role, "DiagnosticQuantity.execution.role")
    operations = value["operations"]
    if not isinstance(operations, (tuple, list)) or not operations:
        raise TypeError("DiagnosticQuantity.execution.operations must be a non-empty sequence")
    normalized = []
    for index, operation in enumerate(operations):
        where = "DiagnosticQuantity.execution.operations[%d]" % index
        if not isinstance(operation, Mapping) or set(operation) != {
                "name", "reduction", "transform", "metric_weighted"}:
            raise TypeError("%s has an unknown schema" % where)
        name = _text(operation["name"], "%s.name" % where)
        reduction = operation["reduction"]
        if reduction not in _DIAGNOSTIC_REDUCTIONS:
            raise ValueError("%s.reduction is not a supported native reduction" % where)
        transform = operation["transform"]
        if transform not in _DIAGNOSTIC_TRANSFORMS:
            raise ValueError("%s.transform is not a supported scalar transform" % where)
        weighted = operation["metric_weighted"]
        if type(weighted) is not bool:
            raise TypeError("%s.metric_weighted must be an exact bool" % where)
        if weighted and reduction not in {"sum", "abs_sum", "sum_sq"}:
            raise ValueError("only additive diagnostic reductions may be metric-weighted")
        normalized.append({
            "name": name,
            "reduction": reduction,
            "transform": transform,
            "metric_weighted": weighted,
        })
    if len({row["name"] for row in normalized}) != len(normalized):
        raise ValueError("DiagnosticQuantity execution operation names must be unique")
    conservation = value["conservation"]
    normalized_conservation = None
    if conservation is not None:
        if not isinstance(conservation, Mapping) or set(conservation) != {"tolerance"}:
            raise TypeError("DiagnosticQuantity.execution.conservation has an unknown schema")
        tolerance = _nonnegative_binary64_hex(
            conservation["tolerance"], "diagnostic conservation tolerance")
        if len(normalized) != 1:
            raise ValueError("a conservation check requires exactly one scalar operation")
        normalized_conservation = {"tolerance": tolerance}
    return freeze_data({
        "schema_version": 1,
        "role": role,
        "operations": normalized,
        "conservation": normalized_conservation,
    }, "DiagnosticQuantity.execution")


def diagnostic_collective_operations(execution: Any) -> tuple[str, ...]:
    """Project one closed execution plan onto its exact global-reduction semantics."""
    canonical = _diagnostic_execution(execution)
    return tuple(sorted({
        _DIAGNOSTIC_COLLECTIVES[operation["reduction"]]
        for operation in canonical["operations"]
    }))


@dataclass(frozen=True, slots=True)
class DiagnosticQuantity:
    """One exact typed native reduction embedded in a scientific-output consumer."""

    handle: Handle
    reference: Handle
    runtime_resource: str
    layout_id: str
    levels: tuple[int, ...]
    execution: Mapping[str, Any]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _exact_handle(self.handle, "diagnostic", "DiagnosticQuantity.handle")
        _exact_handle(self.reference, "state", "DiagnosticQuantity.reference")
        _text(self.runtime_resource, "DiagnosticQuantity.runtime_resource")
        _text(self.layout_id, "DiagnosticQuantity.layout_id")
        if not isinstance(self.levels, tuple):
            raise TypeError("DiagnosticQuantity.levels must be a tuple")
        levels = tuple(_index(value, "DiagnosticQuantity.levels[]") for value in self.levels)
        if levels != tuple(sorted(set(levels))):
            raise ValueError("DiagnosticQuantity.levels must be sorted and unique")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "execution", _diagnostic_execution(self.execution))
        # Diagnostics are executable consumer quantities, not a second resource
        # namespace.  ConsumerResourceBinding and ConsumerFieldResolution use the
        # shared ``consumer-quantity`` domain to authenticate every field read;
        # keeping a diagnostic-only domain here would make the exact value
        # impossible to install without an alias or identity rewrite.
        object.__setattr__(self, "identity", make_identity(
            "consumer-quantity", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "handle": self.handle.canonical_identity(),
            "reference": self.reference.canonical_identity(),
            "runtime_resource": self.runtime_resource,
            "layout_id": self.layout_id,
            "levels": list(self.levels),
            "execution": thaw_data(self.execution),
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


@dataclass(frozen=True, slots=True)
class ConsumerManifest:
    """Semantic declaration of one distinct ConsumerGraph node."""

    handle: Handle
    kind: ConsumerKind
    quantities: tuple[ConsumerQuantity, ...]
    schedule: Schedule
    target_uri: str
    output_format: Any
    parallel_mode: ParallelMode
    diagnostics: tuple[Any, ...] = ()
    diagnostic_quantities: tuple[DiagnosticQuantity, ...] = ()
    dependencies: tuple[Handle, ...] = ()
    failure_action: ConsumerFailureAction = field(default_factory=FailRun)
    operation: Any = None
    output_format_data: Mapping[str, Any] | None = field(init=False, repr=False)
    operation_data: Mapping[str, Any] | None = field(init=False, repr=False)
    diagnostics_data: tuple[Mapping[str, Any], ...] = field(init=False, repr=False)
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _exact_handle(self.handle, "consumer", "ConsumerManifest.handle")
        if type(self.kind) is not ConsumerKind:
            raise TypeError("ConsumerManifest.kind must be an exact ConsumerKind")
        if not isinstance(self.quantities, tuple) or any(
                type(value) is not ConsumerQuantity for value in self.quantities):
            raise TypeError("ConsumerManifest.quantities must contain exact ConsumerQuantity values")
        quantities = tuple(sorted(self.quantities, key=lambda value: value.identity.token))
        if len({value.identity for value in quantities}) != len(quantities):
            raise ValueError("ConsumerManifest contains duplicate quantities")
        object.__setattr__(self, "quantities", quantities)
        if type(self.schedule) is not Schedule:
            raise TypeError("ConsumerManifest.schedule must be an exact Schedule")
        _text(self.target_uri, "ConsumerManifest.target_uri")
        if type(self.parallel_mode) is not ParallelMode:
            raise TypeError("ConsumerManifest.parallel_mode must be an exact ParallelMode")
        if self.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            from pops.output.provider import consumer_format_data
            format_data = freeze_data(
                consumer_format_data(
                    self.output_format, where="ConsumerManifest.output_format"),
                "ConsumerManifest.output_format.consumer_data",
            )
            if format_data["parallel_mode"] != self.parallel_mode.value:
                raise ValueError(
                    "ConsumerManifest parallel mode differs from its scientific format provider"
                )
            if self.operation is not None:
                raise ValueError("ScientificOutput carries its writer in output_format, not operation")
            operation_data = None
        elif self.kind is ConsumerKind.CHECKPOINT:
            if self.output_format is not None:
                raise ValueError("Checkpoint has no scientific output_format")
            format_data = None
            operation_data = _provider_data(
                self.operation,
                where="ConsumerManifest.operation",
                methods=("snapshot", "validate_snapshot", "write", "reopen", "restore"),
            )
        elif self.kind is ConsumerKind.MONITOR:
            if self.output_format is not None:
                raise ValueError("Monitor has no scientific output_format")
            format_data = None
            operation_data = _observer_provider_data(
                self.operation, where="ConsumerManifest.operation")
            if operation_data["parallel_mode"] != self.parallel_mode.value:
                raise ValueError(
                    "ConsumerManifest parallel mode differs from its live observer provider")
        elif self.kind is ConsumerKind.DIAGNOSTIC:
            if self.output_format is not None:
                raise ValueError("Diagnostic consumers have no scientific output_format")
            format_data = None
            operation_data = _console_provider_data(
                self.operation, where="ConsumerManifest.operation")
            if self.parallel_mode is not ParallelMode.ROOT:
                raise ValueError("Console diagnostic consumers must use root parallel mode")
        else:
            raise ValueError("ConsumerManifest has an unsupported consumer kind")
        object.__setattr__(self, "output_format_data", format_data)
        object.__setattr__(self, "operation_data", operation_data)
        if not isinstance(self.diagnostics, tuple):
            raise TypeError("ConsumerManifest.diagnostics must be a tuple")
        diagnostic_rows = []
        for index, diagnostic in enumerate(self.diagnostics):
            where = "ConsumerManifest.diagnostics[%d]" % index
            if not callable(getattr(diagnostic, "consumer_data", None)):
                raise TypeError("%s must implement consumer_data()" % where)
            first, second = diagnostic.consumer_data(), diagnostic.consumer_data()
            if type(first) is not dict or first != second:
                raise TypeError("%s consumer_data() must return one deterministic dict" % where)
            references = getattr(diagnostic, "declaration_references", None)
            if not callable(references):
                raise TypeError("%s must implement declaration_references()" % where)
            resolved_references = references()
            if not isinstance(resolved_references, tuple) or any(
                    not isinstance(value, Handle) or not value.is_resolved
                    for value in resolved_references):
                raise TypeError("%s references must be canonical Handles" % where)
            diagnostic_rows.append(freeze_data({
                "descriptor": first,
                "references": [value.canonical_identity() for value in resolved_references],
            }, "%s.consumer_data" % where))
        if diagnostic_rows and self.kind not in {
                ConsumerKind.DIAGNOSTIC, ConsumerKind.SCIENTIFIC_OUTPUT}:
            raise ValueError(
                "only ConsoleMonitor or ScientificOutput can embed diagnostic providers")
        object.__setattr__(self, "diagnostics_data", tuple(diagnostic_rows))
        if not isinstance(self.diagnostic_quantities, tuple) or any(
                type(value) is not DiagnosticQuantity
                for value in self.diagnostic_quantities):
            raise TypeError(
                "ConsumerManifest.diagnostic_quantities must contain exact "
                "DiagnosticQuantity values")
        diagnostic_quantities = tuple(sorted(
            self.diagnostic_quantities, key=lambda value: value.identity.token))
        if len({value.identity for value in diagnostic_quantities}) \
                != len(diagnostic_quantities):
            raise ValueError("ConsumerManifest contains duplicate diagnostic quantities")
        if len(diagnostic_quantities) != len(self.diagnostics):
            raise ValueError(
                "ConsumerManifest must lower every diagnostic descriptor exactly once")
        if diagnostic_quantities and self.kind not in {
                ConsumerKind.DIAGNOSTIC, ConsumerKind.SCIENTIFIC_OUTPUT}:
            raise ValueError(
                "only ConsoleMonitor or ScientificOutput can carry diagnostic quantities")
        object.__setattr__(self, "diagnostic_quantities", diagnostic_quantities)
        if not isinstance(self.dependencies, tuple):
            raise TypeError("ConsumerManifest.dependencies must be a tuple")
        dependencies = tuple(sorted(
            (_exact_handle(value, "consumer", "ConsumerManifest.dependencies[]")
             for value in self.dependencies), key=lambda value: value.qualified_id))
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("ConsumerManifest contains duplicate dependencies")
        if self.handle in dependencies:
            raise ValueError("ConsumerManifest cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
        if type(self.failure_action) not in _FAILURE_ACTIONS:
            raise TypeError("ConsumerManifest.failure_action must be FailRun, Retry, or SkipSampleReported")
        object.__setattr__(self, "identity", make_identity("consumer-manifest", self._payload()))

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "handle": self.handle.canonical_identity(),
            "kind": self.kind.value,
            "quantities": [value.to_data() for value in self.quantities],
            "schedule": self.schedule.to_data(),
            "target_uri": self.target_uri,
            "output_format": None if self.output_format_data is None
            else thaw_data(self.output_format_data),
            "operation": None if self.operation_data is None
            else thaw_data(self.operation_data),
            "diagnostics": [thaw_data(value) for value in self.diagnostics_data],
            "diagnostic_quantities": [
                value.to_data() for value in self.diagnostic_quantities],
            "parallel_mode": self.parallel_mode.value,
            "dependencies": [value.canonical_identity() for value in self.dependencies],
            "failure_action": self.failure_action.to_data(),
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


class ConsumerGraph:
    """Immutable authoring or resolved DAG with an explicit phase boundary.

    ``from_consumers`` captures callback-free authoring nodes. ``resolve`` is the only route that
    combines them with Case ownership and a LayoutPlan. The ordinary constructor remains the
    low-level resolved form used by runtime planning.
    """

    __slots__ = ("nodes", "topology", "identity", "_by_id", "_authoring", "_sealed")

    def __init__(self, manifests: Iterable[ConsumerManifest]) -> None:
        supplied = tuple(manifests)
        if any(type(value) is not ConsumerManifest for value in supplied):
            raise TypeError("ConsumerGraph requires exact ConsumerManifest values")
        nodes = tuple(sorted(supplied, key=lambda value: value.qualified_id))
        ids = [value.qualified_id for value in nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("ConsumerGraph contains duplicate consumer handles")
        by_id = {value.qualified_id: value for value in nodes}
        indegree = {value.qualified_id: len(value.dependencies) for value in nodes}
        followers: dict[str, list[str]] = {value.qualified_id: [] for value in nodes}
        for value in nodes:
            for dependency in value.dependencies:
                if dependency.qualified_id not in by_id or by_id[dependency.qualified_id].handle != dependency:
                    raise ValueError("ConsumerGraph dependency %s is not an exact graph node" % dependency.qualified_id)
                followers[dependency.qualified_id].append(value.qualified_id)
        ready = [consumer_id for consumer_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        topology = []
        while ready:
            current = heapq.heappop(ready)
            topology.append(by_id[current])
            for follower in sorted(followers[current]):
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    heapq.heappush(ready, follower)
        if len(topology) != len(nodes):
            cycle = sorted(consumer_id for consumer_id, degree in indegree.items() if degree)
            raise ValueError("ConsumerGraph contains a dependency cycle: %s" % cycle)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "topology", tuple(topology))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_authoring", ())
        object.__setattr__(self, "identity", make_identity("consumer-graph", self._payload()))
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def from_consumers(cls, consumers: Iterable[Any]) -> ConsumerGraph:
        """Capture typed consumers through their universal ``consumer_authoring()`` protocol."""
        try:
            supplied = tuple(consumers)
        except TypeError as exc:
            raise TypeError("ConsumerGraph.from_consumers requires an iterable of consumers") from exc
        if not supplied:
            raise ValueError("ConsumerGraph.from_consumers requires at least one consumer")
        from ._consumer_authoring import ConsumerAuthoringNode

        nodes = []
        for index, consumer in enumerate(supplied):
            protocol = getattr(consumer, "consumer_authoring", None)
            if not callable(protocol):
                raise TypeError(
                    "consumer %d (%s) must implement consumer_authoring()"
                    % (index, type(consumer).__name__))
            authored = protocol()
            if not isinstance(authored, tuple) or any(
                    type(value) is not ConsumerAuthoringNode for value in authored):
                raise TypeError(
                    "%s.consumer_authoring() must return a tuple of exact "
                    "ConsumerAuthoringNode values" % type(consumer).__name__)
            nodes.extend(authored)
        if not nodes:
            raise ValueError("consumers produced no ConsumerGraph nodes")
        result = object.__new__(cls)
        object.__setattr__(result, "nodes", ())
        object.__setattr__(result, "topology", ())
        object.__setattr__(result, "identity", None)
        object.__setattr__(result, "_by_id", MappingProxyType({}))
        object.__setattr__(result, "_authoring", tuple(nodes))
        object.__setattr__(result, "_sealed", True)
        return result

    @property
    def is_resolved(self) -> bool:
        return not self._authoring

    def validate_references(self, resolver: Any) -> bool:
        """Authenticate every declared reference without choosing a layout or runtime route."""
        if not callable(resolver):
            raise TypeError("ConsumerGraph reference resolver must be callable")
        references = (
            (reference for node in self._authoring for reference in node.declaration_references())
            if self._authoring else
            (quantity.reference for manifest in self.nodes for quantity in manifest.quantities)
        )
        for reference in references:
            resolved = resolver(reference)
            if not isinstance(resolved, Handle) or not resolved.is_resolved:
                raise TypeError("consumer reference resolver must return canonical Handles")
        return True

    def authoring_data(self, resolver: Any) -> dict[str, Any]:
        """Canonical Case-snapshot projection before layout resolution."""
        if not self._authoring:
            return self.to_data()
        rows = [node.canonical_data(resolver) for node in self._authoring]
        rows.sort(key=lambda row: make_identity("consumer-authoring-node", row).token)
        identities = [make_identity("consumer-authoring-node", row).token for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError("ConsumerGraph.from_consumers contains a duplicate consumer declaration")
        return {"schema_version": 1, "phase": "authoring", "nodes": rows}

    def resolve(self, resolver: Any, layout_plan: Any, *, owner: Any) -> ConsumerGraph:
        """Return the canonical runtime graph for one Case and exact LayoutPlan."""
        from pops.mesh import LayoutPlan
        from pops.model import OwnerKind, OwnerPath

        if not callable(resolver):
            raise TypeError("ConsumerGraph resolver must be callable")
        if type(layout_plan) is not LayoutPlan:
            raise TypeError("ConsumerGraph.resolve requires an exact LayoutPlan")
        case_owner = OwnerPath.coerce(owner)
        if not case_owner.is_canonical or case_owner.kind is not OwnerKind.CASE:
            raise TypeError("ConsumerGraph.resolve owner must be a canonical Case OwnerPath")
        if layout_plan.owner != case_owner:
            raise ValueError("ConsumerGraph and LayoutPlan belong to different Case authorities")
        if self._authoring:
            resolved = tuple(
                node.resolve(resolver, layout_plan, owner=case_owner)
                for node in self._authoring)
            # A checkpoint authenticates the post-publication cursor set, so it is necessarily the
            # final effect for its clock. Derive that ordering from semantics instead of asking users
            # to repeat dependency handles that do not exist until after resolution.
            ordered = []
            for manifest in resolved:
                if manifest.kind is not ConsumerKind.CHECKPOINT:
                    ordered.append(manifest)
                    continue
                predecessors = tuple(
                    row.handle for row in resolved
                    if row.kind is not ConsumerKind.CHECKPOINT
                    and row.schedule.domain.clock == manifest.schedule.domain.clock
                )
                ordered.append(replace(
                    manifest,
                    dependencies=tuple({*manifest.dependencies, *predecessors}),
                ))
            return type(self)(ordered)
        self.validate_references(resolver)
        expected_owner = case_owner.child(OwnerKind.CONSUMER, "graph")
        if any(manifest.handle.owner_path != expected_owner for manifest in self.nodes):
            raise ValueError(
                "a resolved ConsumerGraph attached to a Case must use that Case consumer owner")
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ConsumerGraph is immutable")
        object.__setattr__(self, name, value)

    def _payload(self) -> dict[str, Any]:
        if self._authoring:
            raise TypeError("an authoring ConsumerGraph has no resolved runtime payload")
        return {
            "schema_version": 2,
            "nodes": [value.to_data() for value in self.nodes],
            "topology": [value.qualified_id for value in self.topology],
        }

    def to_data(self) -> dict[str, Any]:
        if self._authoring:
            return self.inspect()
        return {**self._payload(), "identity": self.identity.to_data()}

    def inspect(self) -> dict[str, Any]:
        if self._authoring:
            return {
                "schema_version": 1,
                "phase": "authoring",
                "nodes": [node.inspect() for node in self._authoring],
            }
        return {**self.to_data(), "phase": "resolved"}


@dataclass(frozen=True, slots=True)
class ConsumerMoment:
    """Exact runtime evidence used to evaluate every typed schedule domain."""

    point: TimePoint
    accepted_step: int
    attempt: int
    physical_time_hex: str
    clock_tick: int = 0
    wall_tick: int = 0
    stage: StagePoint | None = None
    level: int | None = None
    events: tuple[EventHandle, ...] = ()
    layouts: tuple[LayoutBinding, ...] = ()
    at_start: bool = False
    at_end: bool = False

    def __post_init__(self) -> None:
        from pops.fields import LayoutBinding

        if type(self.point) is not TimePoint:
            raise TypeError("ConsumerMoment.point must be an exact TimePoint")
        for name in ("accepted_step", "attempt", "clock_tick", "wall_tick"):
            _index(getattr(self, name), "ConsumerMoment.%s" % name)
        object.__setattr__(
            self,
            "physical_time_hex",
            _nonnegative_binary64_hex(self.physical_time_hex, "ConsumerMoment.physical_time_hex"),
        )
        if self.stage is not None and type(self.stage) is not StagePoint:
            raise TypeError("ConsumerMoment.stage must be an exact StagePoint or None")
        if self.level is not None:
            _index(self.level, "ConsumerMoment.level")
        if not isinstance(self.events, tuple) or any(type(value) is not EventHandle for value in self.events):
            raise TypeError("ConsumerMoment.events must contain exact EventHandle values")
        events = tuple(sorted(self.events, key=lambda value: (str(value.owner), value.local_id)))
        if len(set(events)) != len(events):
            raise ValueError("ConsumerMoment.events must be unique")
        object.__setattr__(self, "events", events)
        if not isinstance(self.layouts, tuple) or any(type(value) is not LayoutBinding for value in self.layouts):
            raise TypeError("ConsumerMoment.layouts must contain exact LayoutBinding values")
        layouts = tuple(sorted(self.layouts, key=lambda value: value.layout.qualified_id))
        if len({value.layout.qualified_id for value in layouts}) != len(layouts):
            raise ValueError("ConsumerMoment.layouts contains duplicate layouts")
        object.__setattr__(self, "layouts", layouts)
        if type(self.at_start) is not bool or type(self.at_end) is not bool:
            raise TypeError("ConsumerMoment at_start/at_end must be bool")

    def layout_for(self, layout_id: str) -> LayoutBinding:
        matches = [value for value in self.layouts if value.layout.qualified_id == layout_id]
        if not matches:
            raise KeyError(layout_id)
        return matches[0]

    def to_data(self) -> dict[str, Any]:
        return {
            "point": self.point.to_data(),
            "accepted_step": self.accepted_step,
            "attempt": self.attempt,
            "physical_time": self.physical_time_hex,
            "clock_tick": self.clock_tick,
            "wall_tick": self.wall_tick,
            "stage": self.stage.to_data() if self.stage else None,
            "level": self.level,
            "events": [value.to_data() for value in self.events],
            "layouts": [value.to_data() for value in self.layouts],
            "at_start": self.at_start,
            "at_end": self.at_end,
        }


@dataclass(frozen=True, slots=True)
class ScheduleCursor:
    consumer_id: str
    last_occurrence: str | None = None
    committed_samples: int = 0

    def __post_init__(self) -> None:
        _text(self.consumer_id, "ScheduleCursor.consumer_id")
        if self.last_occurrence is not None:
            _text(self.last_occurrence, "ScheduleCursor.last_occurrence")
        _index(self.committed_samples, "ScheduleCursor.committed_samples")

    def to_data(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "last_occurrence": self.last_occurrence,
            "committed_samples": self.committed_samples,
        }


class ConsumerCursorSet:
    __slots__ = ("rows", "_by_id")

    def __init__(self, rows: Iterable[ScheduleCursor] = ()) -> None:
        supplied = tuple(rows)
        if any(type(value) is not ScheduleCursor for value in supplied):
            raise TypeError("ConsumerCursorSet requires exact ScheduleCursor values")
        values = tuple(sorted(supplied, key=lambda value: value.consumer_id))
        if len({value.consumer_id for value in values}) != len(values):
            raise ValueError("ConsumerCursorSet contains duplicate consumer ids")
        object.__setattr__(self, "rows", values)
        object.__setattr__(self, "_by_id", MappingProxyType({value.consumer_id: value for value in values}))

    def for_consumer(self, consumer_id: str) -> ScheduleCursor:
        _text(consumer_id, "consumer_id")
        return self._by_id.get(consumer_id, ScheduleCursor(consumer_id))

    def replace(self, cursor: ScheduleCursor) -> ConsumerCursorSet:
        if type(cursor) is not ScheduleCursor:
            raise TypeError("ConsumerCursorSet.replace requires an exact ScheduleCursor")
        values = {value.consumer_id: value for value in self.rows}
        values[cursor.consumer_id] = cursor
        return ConsumerCursorSet(tuple(values.values()))

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "rows": [value.to_data() for value in self.rows]}


__all__ = [
    "ConsumerCursorSet", "ConsumerFailureAction", "ConsumerGraph", "ConsumerKind",
    "ConsumerManifest", "ConsumerMoment", "ConsumerQuantity", "DiagnosticQuantity", "FailRun", "ParallelMode",
    "Retry", "ScheduleCursor", "SkipSampleReported", "diagnostic_collective_operations",
    "validate_checkpoint_snapshot",
]
