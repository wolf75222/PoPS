"""Final object-level AMR authoring values."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar

from pops._ir import Expr
from pops._ir.visitors import _key
from pops.mesh._amr.tagging_graph import ConflictPolicy, Hysteresis
from pops.time import Schedule


def _positive_int(value: Any, *, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (where, minimum))
    return value


def _strict_key_data(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("expression identity contains a non-finite float")
        return {"binary64": value.hex()}
    if isinstance(value, (tuple, list)):
        return [_strict_key_data(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("expression identity mappings require non-empty string keys")
        return {key: _strict_key_data(item) for key, item in value.items()}
    raise TypeError("expression identity contains unsupported %s" % type(value).__name__)


def _expression_data(value: Expr) -> dict[str, Any]:
    return {
        "protocol": "pops.expr.key.v1",
        "value": _strict_key_data(_key(value)),
    }


@dataclass(frozen=True, slots=True)
class PatchLayout:
    """Public authority for the coarse-level patch distribution.

    ``coarse_max_grid=None`` delegates the tile-size choice to the selected native patch
    provider. The public value deliberately never exposes the provider's integer sentinel.
    """

    distribute_coarse: bool = False
    coarse_max_grid: int | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.distribute_coarse) is not bool:
            raise TypeError("PatchLayout.distribute_coarse must be an exact bool")
        if self.coarse_max_grid is not None:
            if type(self.coarse_max_grid) is not int:
                raise TypeError(
                    "PatchLayout.coarse_max_grid must be None or an exact non-bool integer"
                )
            if self.coarse_max_grid < 1:
                raise ValueError("PatchLayout.coarse_max_grid must be positive when provided")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "amr_patch_layout",
            "distribute_coarse": self.distribute_coarse,
            "coarse_max_grid": self.coarse_max_grid,
        }

    inspect = to_data
    canonical_identity = to_data


@dataclass(frozen=True, slots=True)
class AMRHierarchy:
    """Explicit level topology; one ratio is declared per coarse/fine transition."""

    max_levels: int
    ratios: tuple[int, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        levels = _positive_int(self.max_levels, where="AMRHierarchy.max_levels")
        ratios = tuple(self.ratios)
        if len(ratios) != levels - 1:
            raise ValueError(
                "AMRHierarchy.ratios must contain max_levels - 1 transitions")
        for index, ratio in enumerate(ratios):
            _positive_int(ratio, where="AMRHierarchy.ratios[%d]" % index, minimum=2)
        object.__setattr__(self, "ratios", ratios)

    @property
    def uniform_ratio(self) -> int | None:
        if not self.ratios:
            return 1
        return self.ratios[0] if len(set(self.ratios)) == 1 else None

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "amr_hierarchy",
            "max_levels": self.max_levels,
            "ratios": list(self.ratios),
        }

    inspect = to_data


@dataclass(frozen=True, slots=True)
class AMRRegrid:
    """Accepted-step schedule for transactional hierarchy changes."""

    schedule: Schedule
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.schedule) is not Schedule:
            raise TypeError("AMRRegrid.schedule must be an exact typed Schedule")
        data = self.schedule.to_data()
        if data["domain"]["type"] != "accepted_step" \
                or data["trigger"]["type"] not in {"always", "every"}:
            raise ValueError("AMRRegrid requires an always/every AcceptedStep schedule")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "amr_regrid",
            "schedule": self.schedule.to_data(),
        }

    inspect = to_data


class AMRRemainderPolicy(Enum):
    """Exact policy for closing a non-integral parent/child level-clock window."""

    INTEGRAL_ONLY = "integral_only"
    EXPLICIT_FINAL_SUBSTEP = "explicit_final_substep"


@dataclass(frozen=True, slots=True, init=False)
class AMRClockRelation:
    """One level-clock relation, independent from spatial refinement."""

    parent_level: int
    child_level: int
    temporal_ratio: Fraction
    remainder_policy: AMRRemainderPolicy
    __pops_ir_immutable__ = True

    def __init__(
        self, parent_level: int, child_level: int, temporal_ratio: int | Fraction,
        remainder_policy: AMRRemainderPolicy = AMRRemainderPolicy.INTEGRAL_ONLY,
    ) -> None:
        if (isinstance(parent_level, bool) or not isinstance(parent_level, int)
                or parent_level < 0 or child_level != parent_level + 1):
            raise ValueError("AMRClockRelation requires adjacent non-negative levels")
        if isinstance(temporal_ratio, bool) or type(temporal_ratio) not in {int, Fraction}:
            raise TypeError("AMRClockRelation temporal_ratio must be an int or Fraction")
        ratio = Fraction(temporal_ratio)
        if ratio < 1:
            raise ValueError("AMRClockRelation temporal_ratio must be >= 1")
        native_limit = (1 << 63) - 1
        if ratio.numerator > native_limit or ratio.denominator > native_limit:
            raise OverflowError(
                "AMRClockRelation temporal_ratio exceeds the native exact-clock range")
        if type(remainder_policy) is not AMRRemainderPolicy:
            raise TypeError("AMRClockRelation remainder_policy must be AMRRemainderPolicy")
        if ratio.denominator != 1 and remainder_policy is AMRRemainderPolicy.INTEGRAL_ONLY:
            raise ValueError(
                "a non-integral AMR temporal relation requires EXPLICIT_FINAL_SUBSTEP")
        object.__setattr__(self, "parent_level", parent_level)
        object.__setattr__(self, "child_level", child_level)
        object.__setattr__(self, "temporal_ratio", ratio)
        object.__setattr__(self, "remainder_policy", remainder_policy)

    def to_data(self) -> dict[str, Any]:
        return {
            "parent_level": self.parent_level,
            "child_level": self.child_level,
            "temporal_ratio": {
                "numerator": self.temporal_ratio.numerator,
                "denominator": self.temporal_ratio.denominator,
            },
            "remainder_policy": self.remainder_policy.value,
        }


@dataclass(frozen=True, slots=True, init=False)
class AMRExecution:
    """How levels advance in time, independently of the temporal Program graph."""

    mode: str
    relations: tuple[AMRClockRelation, ...]
    __pops_ir_immutable__ = True

    def __init__(self, mode: str, relations: tuple[AMRClockRelation, ...] = ()) -> None:
        if mode not in {"subcycled", "synchronous"}:
            raise ValueError("AMRExecution mode must be subcycled or synchronous")
        rows = tuple(relations)
        if any(type(row) is not AMRClockRelation for row in rows):
            raise TypeError("AMRExecution relations must be exact AMRClockRelation values")
        if mode == "synchronous" and rows:
            raise ValueError("synchronous AMRExecution derives ratio-one clocks and accepts no relations")
        children = [row.child_level for row in rows]
        if len(children) != len(set(children)):
            raise ValueError("AMRExecution declares one clock relation per child level")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "relations", rows)

    @classmethod
    def subcycled(
        cls, relations: tuple[AMRClockRelation, ...] = (),
    ) -> AMRExecution:
        return cls("subcycled", relations)

    @classmethod
    def synchronous(cls) -> AMRExecution:
        return cls("synchronous")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "authority_type": "amr_execution",
            "mode": self.mode,
            "relations": [row.to_data() for row in self.relations],
        }

    inspect = to_data
    runtime_execution_data = to_data


@dataclass(frozen=True, slots=True)
class Tag:
    predicate: Expr
    action: ClassVar[str] = "refine"
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, Expr) or not callable(
                getattr(self.predicate, "resolve_for_amr_predicate", None)):
            raise TypeError("Tag requires a typed symbolic Boolean expression")

    def resolve_references(self, resolver: Any) -> Tag:
        return type(self)(self.predicate.resolve_references(resolver))

    def inspect(self) -> dict[str, Any]:
        return {"action": self.action, "predicate": _expression_data(self.predicate)}


@dataclass(frozen=True, slots=True)
class Coarsen:
    predicate: Expr
    action: ClassVar[str] = "coarsen"
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, Expr) or not callable(
                getattr(self.predicate, "resolve_for_amr_predicate", None)):
            raise TypeError("Coarsen requires a typed symbolic Boolean expression")

    def resolve_references(self, resolver: Any) -> Coarsen:
        return type(self)(self.predicate.resolve_references(resolver))

    def inspect(self) -> dict[str, Any]:
        return {"action": self.action, "predicate": _expression_data(self.predicate)}


@dataclass(frozen=True, slots=True)
class Buffer:
    cells: int
    action: ClassVar[str] = "buffer"
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _positive_int(self.cells, where="Buffer.cells", minimum=0)

    def inspect(self) -> dict[str, Any]:
        return {"action": self.action, "cells": self.cells}


@dataclass(frozen=True, slots=True)
class AMRTagging:
    """Callback-free tag graph with explicit stability, equality and conflict semantics."""

    rules: tuple[Any, ...]
    hysteresis: Hysteresis
    conflict_policy: ConflictPolicy
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if not rules or any(type(rule) not in (Tag, Coarsen, Buffer) for rule in rules):
            raise TypeError("AMRTagging.rules must contain Tag, Coarsen or Buffer values")
        if not any(type(rule) is Tag for rule in rules):
            raise ValueError("AMRTagging requires at least one Tag rule")
        if sum(type(rule) is Buffer for rule in rules) != 1:
            raise ValueError("AMRTagging requires exactly one Buffer rule")
        if type(self.hysteresis) is not Hysteresis:
            raise TypeError("AMRTagging.hysteresis must be an exact Hysteresis")
        if type(self.conflict_policy) is not ConflictPolicy:
            raise TypeError("AMRTagging.conflict_policy must be an exact ConflictPolicy")
        object.__setattr__(self, "rules", rules)

    @property
    def buffer_cells(self) -> int:
        return next(rule.cells for rule in self.rules if type(rule) is Buffer)

    def resolve_references(self, resolver: Any) -> AMRTagging:
        if not callable(resolver):
            raise TypeError("AMRTagging.resolve_references requires a callable resolver")
        return type(self)(
            rules=tuple(
                rule.resolve_references(resolver)
                if callable(getattr(rule, "resolve_references", None)) else rule
                for rule in self.rules
            ),
            hysteresis=self.hysteresis,
            conflict_policy=self.conflict_policy,
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "amr_tagging_authoring",
            "rules": [rule.inspect() for rule in self.rules],
            "hysteresis": self.hysteresis.canonical_identity(),
            "conflict_policy": self.conflict_policy.value,
        }

    def resolve(self, context: Any) -> Any:
        from ._resolution import resolve_tagging

        return resolve_tagging(self, context)


__all__ = [
    "AMRClockRelation",
    "AMRExecution",
    "AMRHierarchy",
    "AMRRegrid",
    "AMRTagging",
    "AMRRemainderPolicy",
    "Buffer",
    "Coarsen",
    "PatchLayout",
    "Tag",
]
