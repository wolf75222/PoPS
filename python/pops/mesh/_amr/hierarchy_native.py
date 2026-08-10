"""Open native-lowering providers for resolved AMR hierarchy authorities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedHierarchyNativeLowering:
    """Provider-authenticated rank-generic hierarchy values carried to native bind."""

    provider: Mapping[str, Any]
    dimension: int
    level_count: int
    transition_ratios: tuple[tuple[int, ...], ...]
    transition_buffers: tuple[tuple[int, ...], ...]
    transition_lookaheads: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provider, Mapping):
            raise TypeError("hierarchy native lowering requires provider authority")
        object.__setattr__(self, "provider", MappingProxyType(dict(self.provider)))
        if type(self.dimension) is not int or self.dimension not in (1, 2, 3):
            raise ValueError("hierarchy native lowering dimension must be 1, 2, or 3")
        if type(self.level_count) is not int or self.level_count < 1:
            raise ValueError("hierarchy native lowering level_count must be positive")
        ratios = tuple(tuple(row) for row in self.transition_ratios)
        buffers = tuple(tuple(row) for row in self.transition_buffers)
        lookaheads = tuple(self.transition_lookaheads)
        transition_count = self.level_count - 1
        if len(ratios) != transition_count \
                or len(buffers) != transition_count \
                or len(lookaheads) != transition_count:
            raise ValueError(
                "hierarchy native lowering must preserve one contract per level transition"
            )
        for index, row in enumerate(ratios):
            if len(row) != self.dimension or any(
                type(value) is not int or value < 1 for value in row
            ) or not any(value > 1 for value in row):
                raise ValueError(
                    "hierarchy native lowering transition_ratios[%d] must contain %d positive "
                    "integers and refine at least one axis" % (index, self.dimension)
                )
        for index, row in enumerate(buffers):
            if len(row) != self.dimension or any(
                type(value) is not int or value < 0 for value in row
            ):
                raise ValueError(
                    "hierarchy native lowering transition_buffers[%d] must contain %d "
                    "non-negative integers" % (index, self.dimension)
                )
        if any(type(value) is not int or value < 0 for value in lookaheads):
            raise ValueError(
                "hierarchy native lowering transition_lookaheads must be non-negative integers"
            )
        object.__setattr__(self, "transition_ratios", ratios)
        object.__setattr__(self, "transition_buffers", buffers)
        object.__setattr__(self, "transition_lookaheads", lookaheads)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "provider": dict(self.provider),
            "dimension": self.dimension,
            "level_count": self.level_count,
            "transition_ratios": [list(row) for row in self.transition_ratios],
            "transition_buffers": [list(row) for row in self.transition_buffers],
            "transition_lookaheads": list(self.transition_lookaheads),
        }


HierarchyNativeLowerer = Callable[[Any, Mapping[str, Any]], PreparedHierarchyNativeLowering]


def _exact_identity(value: Any, *, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be an exact non-empty string" % where)
    return value


@dataclass(frozen=True, slots=True)
class PreparedHierarchyNativeProvider:
    """Provider-owned native validation for one opaque hierarchy lowering route."""

    route_id: str
    version: int
    lowerer: HierarchyNativeLowerer

    def __post_init__(self) -> None:
        _exact_identity(self.route_id, where="hierarchy native route_id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("hierarchy native provider version must be positive")
        if not callable(self.lowerer):
            raise TypeError("hierarchy native provider lowerer must be callable")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "interface": "pops.amr.hierarchy-native-provider@1",
            "route_id": self.route_id,
            "version": self.version,
        }

    def lower(self, hierarchy: Any) -> PreparedHierarchyNativeLowering:
        authority = self.authority()
        first = self.lowerer(hierarchy, authority)
        second = self.lowerer(hierarchy, authority)
        if type(first) is not PreparedHierarchyNativeLowering \
                or type(second) is not PreparedHierarchyNativeLowering:
            raise TypeError(
                "hierarchy native provider must return PreparedHierarchyNativeLowering"
            )
        if first.to_data() != second.to_data():
            raise ValueError("hierarchy native provider lowering is non-deterministic")
        if dict(first.provider) != authority:
            raise ValueError("hierarchy native lowering authenticates another provider")
        return first


_registry_lock = RLock()
_providers: dict[str, PreparedHierarchyNativeProvider] = {}


def register_prepared_hierarchy_native_provider(
    provider: PreparedHierarchyNativeProvider,
) -> PreparedHierarchyNativeProvider:
    if type(provider) is not PreparedHierarchyNativeProvider:
        raise TypeError("hierarchy native plugins must register an exact Provider")
    with _registry_lock:
        if provider.route_id in _providers:
            raise ValueError(
                "hierarchy native route %r is already registered" % provider.route_id
            )
        _providers[provider.route_id] = provider
    return provider


def prepared_hierarchy_native_provider(
    route_id: Any,
) -> PreparedHierarchyNativeProvider:
    route = _exact_identity(route_id, where="hierarchy native route")
    with _registry_lock:
        provider = _providers.get(route)
    if provider is None:
        raise NotImplementedError(
            "hierarchy native route %r has no registered provider" % route
        )
    return provider


def validate_native_hierarchy(hierarchy: Any) -> None:
    """Dispatch native compatibility to the provider selected by canonical plan data."""

    from .hierarchy_resolution import ResolvedHierarchy

    if type(hierarchy) is not ResolvedHierarchy:
        raise TypeError("native hierarchy validation requires an exact ResolvedHierarchy")
    lower_native_hierarchy(hierarchy)


def lower_native_hierarchy(hierarchy: Any) -> PreparedHierarchyNativeLowering:
    """Lower through the selected provider without a route switch in compiler/runtime core."""

    from .hierarchy_resolution import ResolvedHierarchy

    if type(hierarchy) is not ResolvedHierarchy:
        raise TypeError("native hierarchy lowering requires an exact ResolvedHierarchy")
    options = hierarchy.provider.options.to_data()
    if not isinstance(options, Mapping):
        raise TypeError("resolved hierarchy provider options must be a canonical mapping")
    route = options.get("native_route")
    provider = prepared_hierarchy_native_provider(route)
    if options.get("native_provider") != provider.authority():
        raise ValueError(
            "resolved hierarchy does not authenticate the selected native provider"
        )
    return provider.lower(hierarchy)


def _lower_shared_n_level(
    hierarchy: Any, authority: Mapping[str, Any]
) -> PreparedHierarchyNativeLowering:
    options = hierarchy.provider.options.to_data()
    if options != {
        "native_route": "shared_n_level",
        "native_provider": prepared_hierarchy_native_provider(
            "shared_n_level"
        ).authority(),
    }:
        raise ValueError("shared_n_level hierarchy provider options are not canonical")
    transitions = hierarchy.plan.transitions
    return PreparedHierarchyNativeLowering(
        authority,
        hierarchy.plan.dimension,
        hierarchy.plan.level_count,
        tuple(row.ratio for row in transitions),
        tuple(row.buffer for row in transitions),
        tuple(row.lookahead for row in transitions),
    )


register_prepared_hierarchy_native_provider(
    PreparedHierarchyNativeProvider("shared_n_level", 3, _lower_shared_n_level)
)


__all__ = [
    "PreparedHierarchyNativeProvider",
    "PreparedHierarchyNativeLowering",
    "lower_native_hierarchy",
    "prepared_hierarchy_native_provider",
    "register_prepared_hierarchy_native_provider",
    "validate_native_hierarchy",
]
