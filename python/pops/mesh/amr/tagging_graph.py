"""Canonical, inert AMR tagging graphs.

The graph describes decisions only.  It never calls Python during tagging and never discretises an
indicator: gradient predicates therefore carry the exact resolved discrete context they require.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pops.model import Handle, ParamHandle


_SCHEMA_VERSION = 1
_INDICATOR_KINDS = frozenset(("state", "field"))


def _canonical_handle(value: Any, *, where: str, kinds: frozenset[str]) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle):
        raise TypeError("%s requires an owner-qualified Handle, never a string or callback" % where)
    if not value.is_resolved:
        raise TypeError("%s requires a post-resolution canonical Handle" % where)
    if value.kind not in kinds:
        raise TypeError("%s requires Handle.kind in %s, got %r" %
                        (where, sorted(kinds), value.kind))
    identity = value.canonical_identity()
    if identity.get("qualified_id") != value.qualified_id:
        raise ValueError("%s Handle identity does not authenticate qualified_id" % where)
    return value


def _indicator(value: Any, *, where: str) -> Handle:
    return _canonical_handle(value, where=where, kinds=_INDICATOR_KINDS)


def _threshold(value: Any, *, where: str) -> ParamHandle:
    from pops.model import ParamHandle

    if not isinstance(value, ParamHandle):
        raise TypeError("%s threshold must be an owner-qualified ParamHandle" % where)
    _canonical_handle(value, where="%s threshold" % where, kinds=frozenset(("parameter",)))
    return value


def _digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class EqualityPolicy(Enum):
    """Action when an indicator is exactly on a strict hysteresis threshold."""

    HOLD = "hold"
    REFINE = "refine"
    COARSEN = "coarsen"


class ConflictPolicy(Enum):
    """Action when refine and coarsen roots both match the same cell."""

    ERROR = "error"
    HOLD = "hold"
    REFINE_WINS = "refine_wins"
    COARSEN_WINS = "coarsen_wins"


@dataclass(frozen=True, slots=True)
class DiscreteIndicatorContext:
    """Resolved authority for a discrete gradient indicator."""

    layout: Handle
    discretization: Handle
    stencil: Handle

    def __post_init__(self) -> None:
        _canonical_handle(self.layout, where="DiscreteIndicatorContext.layout",
                          kinds=frozenset(("layout",)))
        _canonical_handle(self.discretization, where="DiscreteIndicatorContext.discretization",
                          kinds=frozenset(("discretization",)))
        _canonical_handle(self.stencil, where="DiscreteIndicatorContext.stencil",
                          kinds=frozenset(("stencil",)))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "context_type": "discrete_indicator",
                "layout": self.layout.canonical_identity(),
                "discretization": self.discretization.canonical_identity(),
                "stencil": self.stencil.canonical_identity()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "discrete_indicator_context", **self.canonical_identity()}


class TagExpr(ABC):
    """Closed marker protocol for inert tagging-expression nodes."""

    __slots__ = ()

    @abstractmethod
    def canonical_identity(self) -> dict[str, Any]: ...

    @abstractmethod
    def operands(self) -> tuple[TagExpr, ...]: ...

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "amr_tag_expression", **self.canonical_identity()}

    def runtime_tagging_data(self, params: Any = None) -> dict[str, Any]:
        """Project this node through the open runtime-tagging protocol."""
        raise NotImplementedError(
            "%s has no runtime_tagging_data(params) provider" % type(self).__name__)


@dataclass(frozen=True, slots=True)
class _ThresholdPredicate(TagExpr):
    indicator: Handle
    threshold: ParamHandle

    node_type: ClassVar[str]
    comparison: ClassVar[str]
    transform: ClassVar[str] = "identity"
    polarity: ClassVar[str]
    equality_matches: ClassVar[bool] = False

    def __post_init__(self) -> None:
        _indicator(self.indicator, where="%s.indicator" % type(self).__name__)
        _threshold(self.threshold, where=type(self).__name__)

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "node_type": self.node_type,
                "indicator": self.indicator.canonical_identity(),
                "threshold": self.threshold.canonical_identity(),
                "comparison": self.comparison, "transform": self.transform,
                "polarity": self.polarity, "equality_matches": self.equality_matches}

    def operands(self) -> tuple[TagExpr, ...]:
        return ()

    def runtime_tagging_data(self, params: Any = None) -> dict[str, Any]:
        threshold: Any = self.threshold.canonical_identity()
        if params is not None:
            if self.threshold not in params:
                raise ValueError(
                    "runtime tagging threshold %s is missing" % self.threshold.qualified_id)
            threshold = params[self.threshold]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                    or not math.isfinite(float(threshold)):
                raise TypeError("runtime tagging thresholds must be finite scalars")
            threshold = float(threshold)
        data = {
            "schema_version": _SCHEMA_VERSION,
            "node_type": self.node_type,
            "comparison": self.comparison,
            "transform": self.transform,
            "polarity": self.polarity,
            "equality_matches": self.equality_matches,
            "indicator": self.indicator.canonical_identity(),
            "threshold": threshold,
        }
        context = getattr(self, "context", None)
        if context is not None:
            data["discrete_context"] = context.canonical_identity()
        return data


@dataclass(frozen=True, slots=True)
class Above(_ThresholdPredicate):
    node_type = "above"
    comparison = "strict_greater_than"
    polarity = "high"


@dataclass(frozen=True, slots=True)
class Below(_ThresholdPredicate):
    node_type = "below"
    comparison = "strict_less_than"
    polarity = "low"


@dataclass(frozen=True, slots=True)
class MagnitudeAbove(_ThresholdPredicate):
    node_type = "magnitude_above"
    comparison = "strict_greater_than"
    transform = "magnitude"
    polarity = "high"


@dataclass(frozen=True, slots=True)
class GradientAbove(_ThresholdPredicate):
    context: DiscreteIndicatorContext

    node_type = "gradient_above"
    comparison = "strict_greater_than"
    transform = "discrete_gradient_magnitude"
    polarity = "high"

    def __post_init__(self) -> None:
        _ThresholdPredicate.__post_init__(self)
        if not isinstance(self.context, DiscreteIndicatorContext):
            raise TypeError("GradientAbove requires a DiscreteIndicatorContext")

    def canonical_identity(self) -> dict[str, Any]:
        result = _ThresholdPredicate.canonical_identity(self)
        result["discrete_context"] = self.context.canonical_identity()
        return result


@dataclass(frozen=True, slots=True)
class GradientBelow(_ThresholdPredicate):
    """Strict coarsening predicate on the same authenticated discrete gradient."""

    context: DiscreteIndicatorContext

    node_type = "gradient_below"
    comparison = "strict_less_than"
    transform = "discrete_gradient_magnitude"
    polarity = "low"

    def __post_init__(self) -> None:
        _ThresholdPredicate.__post_init__(self)
        if not isinstance(self.context, DiscreteIndicatorContext):
            raise TypeError("GradientBelow requires a DiscreteIndicatorContext")

    def canonical_identity(self) -> dict[str, Any]:
        result = _ThresholdPredicate.canonical_identity(self)
        result["discrete_context"] = self.context.canonical_identity()
        return result


def _children(values: Any, *, where: str, minimum: int) -> tuple[TagExpr, ...]:
    rows = tuple(values)
    if len(rows) < minimum:
        raise ValueError("%s requires at least %d tagging-expression children" % (where, minimum))
    for row in rows:
        if not isinstance(row, TagExpr):
            raise TypeError("%s children must be TagExpr nodes; strings/callbacks are forbidden" %
                            where)
    return rows


@dataclass(frozen=True, slots=True, init=False)
class _Nary(TagExpr):
    children: tuple[TagExpr, ...]
    node_type: ClassVar[str]

    def __init__(self, *children: TagExpr) -> None:
        object.__setattr__(self, "children", _children(
            children, where=type(self).__name__, minimum=2))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "node_type": self.node_type,
                "children": [child.canonical_identity() for child in self.children]}

    def operands(self) -> tuple[TagExpr, ...]:
        return self.children

    def runtime_tagging_data(self, params: Any = None) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "node_type": self.node_type,
            "children": [child.runtime_tagging_data(params) for child in self.children],
        }


@dataclass(frozen=True, slots=True, init=False)
class AnyOf(_Nary):
    """Logical union retaining every authored child and its order."""

    node_type = "any_of"


@dataclass(frozen=True, slots=True, init=False)
class AllOf(_Nary):
    """Logical intersection retaining every authored child and its order."""

    node_type = "all_of"


@dataclass(frozen=True, slots=True)
class Not(TagExpr):
    child: TagExpr
    node_type: ClassVar[str] = "not"

    def __post_init__(self) -> None:
        if not isinstance(self.child, TagExpr):
            raise TypeError("Not.child must be a TagExpr; strings/callbacks are forbidden")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "node_type": self.node_type,
                "child": self.child.canonical_identity()}

    def operands(self) -> tuple[TagExpr, ...]:
        return (self.child,)

    def runtime_tagging_data(self, params: Any = None) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "node_type": self.node_type,
            "child": self.child.runtime_tagging_data(params),
        }


@dataclass(frozen=True, slots=True)
class Hysteresis:
    """Temporal stability and equality semantics for refine/coarsen transitions."""

    min_cycles: int
    equality: EqualityPolicy

    def __post_init__(self) -> None:
        if isinstance(self.min_cycles, bool) or not isinstance(self.min_cycles, int) \
                or self.min_cycles < 0:
            raise ValueError("Hysteresis.min_cycles must be an integer >= 0")
        if not isinstance(self.equality, EqualityPolicy):
            raise TypeError("Hysteresis.equality must be an explicit EqualityPolicy")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "hysteresis_type": "min_cycles",
                "min_cycles": self.min_cycles, "equality": self.equality.value}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "amr_hysteresis", **self.canonical_identity()}


@dataclass(frozen=True, slots=True, kw_only=True)
class TaggingGraph:
    """Complete, post-resolution refine/coarsen decision contract."""

    refine: TagExpr
    coarsen: TagExpr | None
    hysteresis: Hysteresis
    conflict_policy: ConflictPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.refine, TagExpr):
            raise TypeError("TaggingGraph.refine must be a TagExpr")
        if self.coarsen is not None and not isinstance(self.coarsen, TagExpr):
            raise TypeError("TaggingGraph.coarsen must be a TagExpr or None")
        if not isinstance(self.hysteresis, Hysteresis):
            raise TypeError("TaggingGraph.hysteresis must be explicit")
        if not isinstance(self.conflict_policy, ConflictPolicy):
            raise TypeError("TaggingGraph.conflict_policy must be an explicit ConflictPolicy")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "graph_type": "amr_tagging",
                "refine": self.refine.canonical_identity(),
                "coarsen": (None if self.coarsen is None else
                             self.coarsen.canonical_identity()),
                "hysteresis": self.hysteresis.canonical_identity(),
                "conflict_policy": self.conflict_policy.value}

    @property
    def canonical_id(self) -> str:
        return _digest(self.canonical_identity())

    @property
    def qualified_id(self) -> str:
        return "pops.amr-tagging-graph.v1::%s" % self.canonical_id

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "amr_tagging_graph", "canonical_id": self.canonical_id,
                "qualified_id": self.qualified_id, **self.canonical_identity()}

    def resolve(self, registry: Any = None) -> Any:
        """Authenticate every node against explicit lowering registrations."""
        from .tagging_resolution import resolve_tagging_graph

        return resolve_tagging_graph(self, registry=registry)

    def runtime_tagging_data(self, params: Any = None) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "graph_type": "amr_tagging_runtime",
            "refine": self.refine.runtime_tagging_data(params),
            "coarsen": (
                None if self.coarsen is None else self.coarsen.runtime_tagging_data(params)),
            "hysteresis": self.hysteresis.canonical_identity(),
            "conflict_policy": self.conflict_policy.value,
        }


__all__ = [
    "Above", "AllOf", "AnyOf", "Below", "ConflictPolicy", "DiscreteIndicatorContext",
    "EqualityPolicy", "GradientAbove", "GradientBelow", "Hysteresis", "MagnitudeAbove", "Not", "TagExpr",
    "TaggingGraph",
]
