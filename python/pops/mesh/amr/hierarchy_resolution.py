"""Provider capability resolution for canonical AMR hierarchy plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops.identity import Identity, make_identity

from ._contracts import clock_data

from .hierarchy import (
    _SCHEMA_VERSION,
    _handle,
    _positive_int,
    HierarchyPlan,
    RegridSchedule,
    CanonicalOptions,
)


class HierarchyCapabilityError(ValueError):
    """A provider cannot realize an authored hierarchy without changing semantics."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


@dataclass(frozen=True, slots=True)
class HierarchyProviderCapabilities:
    provider: Any
    supported_dimensions: tuple[int, ...]
    supports_anisotropic_ratio: bool
    max_materialized_level_count: int
    supports_transactional_regrid: bool
    supports_lifecycle_events: bool
    options: CanonicalOptions = CanonicalOptions()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _handle(
            self.provider,
            where="HierarchyProviderCapabilities.provider",
            kind="amr_hierarchy_provider",
        )
        dimensions = tuple(self.supported_dimensions)
        if not dimensions or any(item not in (1, 2, 3) for item in dimensions):
            raise ValueError("supported_dimensions must contain values from {1, 2, 3}")
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("supported_dimensions must be unique")
        object.__setattr__(self, "supported_dimensions", tuple(sorted(dimensions)))
        for name in (
            "supports_anisotropic_ratio",
            "supports_transactional_regrid",
            "supports_lifecycle_events",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("%s must be an exact bool" % name)
        _positive_int(
            self.max_materialized_level_count,
            where="max_materialized_level_count",
        )
        if type(self.options) is not CanonicalOptions:
            raise TypeError("HierarchyProviderCapabilities.options must be CanonicalOptions")

    @property
    def identity(self) -> Identity:
        return make_identity("amr-hierarchy-provider", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "provider": self.provider.canonical_identity(),
            "supported_dimensions": list(self.supported_dimensions),
            "supports_anisotropic_ratio": self.supports_anisotropic_ratio,
            "max_materialized_level_count": self.max_materialized_level_count,
            "supports_transactional_regrid": self.supports_transactional_regrid,
            "supports_lifecycle_events": self.supports_lifecycle_events,
            "options": self.options.to_data(),
        }


@dataclass(frozen=True, slots=True)
class HierarchyResolutionContext:
    clock: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        clock_data(self.clock, where="HierarchyResolutionContext.clock")


@dataclass(frozen=True, slots=True)
class ResolvedHierarchy:
    plan: HierarchyPlan
    provider: HierarchyProviderCapabilities
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.plan) is not HierarchyPlan:
            raise TypeError("ResolvedHierarchy.plan must be HierarchyPlan")
        if type(self.provider) is not HierarchyProviderCapabilities:
            raise TypeError("ResolvedHierarchy.provider must be HierarchyProviderCapabilities")

    @property
    def identity(self) -> Identity:
        return make_identity("resolved-amr-hierarchy", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "plan": self.plan.canonical_identity(),
            "provider": self.provider.canonical_identity(),
        }


def _capability_error(message: str, **evidence: Any) -> None:
    raise HierarchyCapabilityError(message, evidence=evidence)


def resolve_hierarchy(
    plan: Any,
    provider: Any,
    context: Any,
) -> ResolvedHierarchy:
    """Resolve immutable authoring intent within provider capabilities before runtime."""
    if type(plan) is not HierarchyPlan:
        raise TypeError("resolve_hierarchy plan must be HierarchyPlan")
    if type(provider) is not HierarchyProviderCapabilities:
        raise TypeError("resolve_hierarchy provider must be HierarchyProviderCapabilities")
    if type(context) is not HierarchyResolutionContext:
        raise TypeError("resolve_hierarchy context must be HierarchyResolutionContext")
    if type(plan.regrid) is RegridSchedule and plan.regrid.schedule.clock != context.clock:
        _capability_error(
            "regrid schedule is not synchronized with the resolution clock",
            schedule_clock=plan.regrid.schedule.clock.qualified_id,
            resolution_clock=context.clock.qualified_id,
        )
    if plan.dimension not in provider.supported_dimensions:
        _capability_error(
            "hierarchy dimension is not supported by the provider",
            requested_dimension=plan.dimension,
            supported_dimensions=list(provider.supported_dimensions),
        )
    if any(row.anisotropic for row in plan.transitions) and not provider.supports_anisotropic_ratio:
        _capability_error(
            "anisotropic refinement ratio is not supported by the provider",
            requested_ratios=[list(row.ratio) for row in plan.transitions],
            supports_anisotropic_ratio=False,
        )
    if plan.level_count > provider.max_materialized_level_count:
        _capability_error(
            "provider cannot materialize the requested derived hierarchy level count",
            requested_level_count=plan.level_count,
            supported_level_count=provider.max_materialized_level_count,
        )
    if type(plan.regrid) is RegridSchedule and not provider.supports_transactional_regrid:
        _capability_error(
            "provider does not support transactional regrid",
            supports_transactional_regrid=False,
        )
    if type(plan.regrid) is RegridSchedule and not provider.supports_lifecycle_events:
        _capability_error(
            "provider does not expose hierarchy lifecycle events",
            supports_lifecycle_events=False,
        )
    for row in plan.transitions:
        insufficient_axes = [
            axis
            for axis, value in enumerate(row.buffer)
            if value < plan.nesting.minimum_buffer[axis]
        ]
        if insufficient_axes or row.lookahead < plan.nesting.minimum_lookahead:
            _capability_error(
                "transition does not satisfy derived nesting requirements",
                transition=row.canonical_identity(),
                derived_minimum_buffer=list(plan.nesting.minimum_buffer),
                derived_minimum_lookahead=plan.nesting.minimum_lookahead,
                insufficient_axes=insufficient_axes,
            )
    return ResolvedHierarchy(plan, provider)


__all__ = [
    "HierarchyCapabilityError",
    "HierarchyProviderCapabilities",
    "HierarchyResolutionContext",
    "ResolvedHierarchy",
    "resolve_hierarchy",
]
