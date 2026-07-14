"""Private extension resolution and persistent state for AMR tagging graphs."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass
from enum import Enum
import hashlib
import inspect
import json
from typing import TYPE_CHECKING, Any

from .tagging_graph import (
    Above, AllOf, AnyOf, Below, GradientAbove, GradientBelow, MagnitudeAbove, Not, TagExpr,
    TaggingGraph)

if TYPE_CHECKING:
    from pops.model import Handle


_SCHEMA_VERSION = 1
_BUILTIN_NODES = (
    Above, Below, MagnitudeAbove, GradientAbove, GradientBelow, AnyOf, AllOf, Not)


def _handle(value: Any, *, where: str, kind: str | None = None) -> Handle:
    from pops.model import Handle

    if not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if kind is not None and value.kind != kind:
        raise TypeError("%s requires Handle.kind=%r" % (where, kind))
    identity = value.canonical_identity()
    if identity.get("qualified_id") != value.qualified_id:
        raise ValueError("%s Handle identity does not authenticate qualified_id" % where)
    return value


def _strict_data(value: Any, *, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("%s contains a non-finite float" % where)
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mappings require non-empty string keys" % where)
        return {key: _strict_data(item, where="%s.%s" % (where, key))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_data(item, where="%s[]" % where) for item in value]
    raise TypeError("%s contains non-data value %s; callbacks are forbidden" %
                    (where, type(value).__name__))


def _audit_storage(value: Any, *, where: str) -> None:
    from pops.model import Handle

    if callable(value):
        raise TypeError("%s stores a Python callback; callbacks are forbidden" % where)
    if isinstance(value, Handle):
        _handle(value, where=where)
    elif isinstance(value, TagExpr) or isinstance(value, Enum):
        return
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("%s mappings require non-empty string keys" % where)
            _audit_storage(item, where="%s.%s" % (where, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _audit_storage(item, where="%s[%d]" % (where, index))
    elif hasattr(value, "canonical_identity"):
        _strict_data(value.canonical_identity(), where=where)
    else:
        _strict_data(value, where=where)


def _audit_node_storage(node: TagExpr) -> None:
    params = getattr(type(node), "__dataclass_params__", None)
    if not is_dataclass(node) or params is None or params.frozen is not True:
        raise TypeError("registered TagExpr %s must be a frozen dataclass" % type(node).__name__)
    for field in fields(node):
        _audit_storage(getattr(node, field.name), where="TagExpr.%s" % field.name)
    _strict_data(node.canonical_identity(), where="TagExpr.canonical_identity()")


def _stored_operands(node: TagExpr) -> tuple[TagExpr, ...]:
    result = []
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, TagExpr):
            result.append(value)
        elif isinstance(value, (list, tuple)):
            result.extend(item for item in value if isinstance(item, TagExpr))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TagNodeRegistration:
    """One exact Python node type mapped to an owner-qualified lowering authority."""

    node_class: type[TagExpr]
    lowering: Handle
    node_type: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not inspect.isclass(self.node_class) or not issubclass(self.node_class, TagExpr):
            raise TypeError("TagNodeRegistration.node_class must be a TagExpr class")
        if inspect.isabstract(self.node_class):
            raise TypeError("TagNodeRegistration.node_class must be concrete")
        _handle(self.lowering, where="TagNodeRegistration.lowering", kind="tag_lowering")
        value = getattr(self.node_class, "node_type", None)
        if not isinstance(value, str) or not value:
            raise TypeError("registered TagExpr class must expose a non-empty node_type")
        object.__setattr__(self, "node_type", value)

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "node_type": self.node_type,
                "python_type": "%s.%s" %
                               (self.node_class.__module__, self.node_class.__qualname__),
                "lowering": self.lowering.canonical_identity()}


@dataclass(frozen=True, slots=True, init=False)
class TagNodeRegistry:
    registrations: tuple[TagNodeRegistration, ...]

    def __init__(self, *registrations: TagNodeRegistration) -> None:
        rows = tuple(registrations)
        if any(not isinstance(row, TagNodeRegistration) for row in rows):
            raise TypeError("TagNodeRegistry accepts only TagNodeRegistration objects")
        classes = [row.node_class for row in rows]
        node_types = [row.node_type for row in rows]
        if len(classes) != len(set(classes)) or len(node_types) != len(set(node_types)):
            raise ValueError("TagNodeRegistry registrations must be unique by class and node_type")
        object.__setattr__(self, "registrations", rows)

    @classmethod
    def builtins(cls) -> TagNodeRegistry:
        from pops.model import Handle, OwnerPath

        owner = OwnerPath.shared("pops.amr.tagging.v1")
        return cls(*(TagNodeRegistration(
            node, Handle(node.node_type, kind="tag_lowering", owner=owner))
            for node in _BUILTIN_NODES))

    def registration_for(self, node: TagExpr) -> TagNodeRegistration:
        matches = [row for row in self.registrations if type(node) is row.node_class]
        if len(matches) != 1:
            raise ValueError("unregistered TagExpr node type %s" % type(node).__name__)
        _audit_node_storage(node)
        if node.canonical_identity().get("node_type") != matches[0].node_type:
            raise ValueError("TagExpr canonical identity does not authenticate registered node_type")
        return matches[0]


def _resolve_registrations(
    graph: TaggingGraph, registry: TagNodeRegistry,
) -> tuple[TagNodeRegistration, ...]:
    used = {}

    def visit(node: TagExpr) -> None:
        registration = registry.registration_for(node)
        used[registration.node_type] = registration
        operands = node.operands()
        if not isinstance(operands, tuple) or any(not isinstance(row, TagExpr) for row in operands):
            raise TypeError("TagExpr.operands() must return a tuple of TagExpr nodes")
        if operands != _stored_operands(node):
            raise ValueError("TagExpr.operands() must expose every stored child exactly once")
        for child in operands:
            visit(child)

    visit(graph.refine)
    if graph.coarsen is not None:
        visit(graph.coarsen)
    return tuple(used[key] for key in sorted(used))


@dataclass(frozen=True, slots=True)
class ResolvedTaggingGraph:
    graph: TaggingGraph
    registrations: tuple[TagNodeRegistration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, TaggingGraph):
            raise TypeError("ResolvedTaggingGraph.graph must be a TaggingGraph")
        if not isinstance(self.registrations, tuple):
            raise TypeError("ResolvedTaggingGraph.registrations must be a tuple")
        registry = TagNodeRegistry(*self.registrations)
        if _resolve_registrations(self.graph, registry) != self.registrations:
            raise ValueError("ResolvedTaggingGraph registrations are incomplete or non-canonical")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "resolved_type": "amr_tagging_graph",
                "graph": self.graph.canonical_identity(),
                "lowerings": [row.canonical_identity() for row in self.registrations]}

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def qualified_id(self) -> str:
        return "pops.resolved-amr-tagging-graph.v1::%s" % self.canonical_id

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "resolved_amr_tagging_graph",
                "canonical_id": self.canonical_id, "qualified_id": self.qualified_id,
                **self.canonical_identity()}

    def runtime_tagging_data(self, params: Any = None) -> dict[str, Any]:
        """Return provider-authenticated runtime data without dispatching on node classes."""
        data = self.graph.runtime_tagging_data(params)
        data["lowerings"] = [row.canonical_identity() for row in self.registrations]
        return data


def resolve_tagging_graph(graph: Any, *, registry: Any = None) -> ResolvedTaggingGraph:
    if not isinstance(graph, TaggingGraph):
        raise TypeError("resolve_tagging_graph requires a TaggingGraph")
    if registry is None:
        registry = TagNodeRegistry.builtins()
    if not isinstance(registry, TagNodeRegistry):
        raise TypeError("tagging resolution requires a TagNodeRegistry")
    return ResolvedTaggingGraph(graph, _resolve_registrations(graph, registry))


class TagDecision(Enum):
    HOLD = "hold"
    REFINE = "refine"
    COARSEN = "coarsen"


@dataclass(frozen=True, slots=True)
class TaggingState:
    """Persistent hysteresis state for one owner-qualified tagging scope."""

    scope: Handle
    graph: ResolvedTaggingGraph
    cycle: int
    last_decision_cycle: int
    last_decision: TagDecision

    def __post_init__(self) -> None:
        _handle(self.scope, where="TaggingState.scope", kind="tagging_state")
        if not isinstance(self.graph, ResolvedTaggingGraph):
            raise TypeError("TaggingState.graph must be a ResolvedTaggingGraph")
        for key in ("cycle", "last_decision_cycle"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("TaggingState.%s must be an integer >= 0" % key)
        if self.last_decision_cycle > self.cycle:
            raise ValueError("last_decision_cycle cannot be after cycle")
        if not isinstance(self.last_decision, TagDecision):
            raise TypeError("TaggingState.last_decision must be a TagDecision")

    @property
    def cycles_since_decision(self) -> int:
        return self.cycle - self.last_decision_cycle

    def transition_allowed(self) -> bool:
        return self.cycles_since_decision >= self.graph.graph.hysteresis.min_cycles

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "state_type": "amr_tagging",
                "scope": self.scope.canonical_identity(),
                "graph": self.graph.canonical_identity(), "cycle": self.cycle,
                "last_decision_cycle": self.last_decision_cycle,
                "last_decision": self.last_decision.value}

    @classmethod
    def from_canonical_identity(
        cls, data: Any, *, graph: ResolvedTaggingGraph,
    ) -> TaggingState:
        from pops.model import Handle

        required = {"schema_version", "state_type", "scope", "graph", "cycle",
                    "last_decision_cycle", "last_decision"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("TaggingState canonical identity has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["state_type"] != "amr_tagging":
            raise ValueError("unsupported TaggingState schema or state type")
        if data["graph"] != graph.canonical_identity():
            raise ValueError("TaggingState graph identity does not authenticate supplied graph")
        result = cls(scope=Handle.from_canonical_identity(data["scope"]), graph=graph,
                     cycle=data["cycle"], last_decision_cycle=data["last_decision_cycle"],
                     last_decision=TagDecision(data["last_decision"]))
        if result.canonical_identity() != dict(data):
            raise ValueError("TaggingState canonical identity failed round-trip authentication")
        return result

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "amr_tagging_state",
                "transition_allowed": self.transition_allowed(),
                "cycles_since_decision": self.cycles_since_decision,
                **self.canonical_identity()}


__all__ = [
    "ResolvedTaggingGraph", "TagDecision", "TagNodeRegistration", "TagNodeRegistry",
    "TaggingState", "resolve_tagging_graph",
]
