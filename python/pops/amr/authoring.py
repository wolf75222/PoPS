"""Final object-level AMR authoring values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pops.ir import Expr
from pops.ir.visitors import _key
from pops.mesh.amr.tagging_graph import ConflictPolicy, Hysteresis
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


@dataclass(frozen=True, slots=True, init=False)
class AMRExecution:
    """How levels advance in time, independently of the temporal Program graph."""

    mode: str
    __pops_ir_immutable__ = True

    def __init__(self, mode: str) -> None:
        if mode not in {"subcycled", "synchronous"}:
            raise ValueError("AMRExecution mode must be subcycled or synchronous")
        object.__setattr__(self, "mode", mode)

    @classmethod
    def subcycled(cls) -> AMRExecution:
        return cls("subcycled")

    @classmethod
    def synchronous(cls) -> AMRExecution:
        return cls("synchronous")

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "authority_type": "amr_execution", "mode": self.mode}

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
        from .resolution import resolve_tagging

        return resolve_tagging(self, context)


@dataclass(frozen=True, slots=True)
class ResolvedAMRAuthorities:
    hierarchy: Any
    transfer: Any
    tagging: Any
    initial_conditions: Any
    bootstrap: Any
    execution: Any

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "resolved_amr_authorities",
            "hierarchy": self.hierarchy.canonical_identity(),
            "transfer": self.transfer.canonical_identity(),
            "tagging": self.tagging.canonical_identity(),
            "initial_conditions": self.initial_conditions.canonical_identity(),
            "bootstrap": self.bootstrap.canonical_identity(),
            "execution": self.execution.to_data(),
        }


__all__ = [
    "AMRExecution",
    "AMRHierarchy",
    "AMRRegrid",
    "AMRTagging",
    "Buffer",
    "Coarsen",
    "ResolvedAMRAuthorities",
    "Tag",
]
