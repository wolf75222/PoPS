"""Open native-lowering providers for resolved AMR hierarchy authorities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedHierarchyNativeLowering:
    """Provider-authenticated values consumed by the current native AMR runtime ABI."""

    provider: Mapping[str, Any]
    level_count: int
    nesting_buffer: int
    nesting_lookahead: int

    def __post_init__(self) -> None:
        if not isinstance(self.provider, Mapping):
            raise TypeError("hierarchy native lowering requires provider authority")
        object.__setattr__(self, "provider", MappingProxyType(dict(self.provider)))
        if type(self.level_count) is not int or self.level_count < 1:
            raise ValueError("hierarchy native lowering level_count must be positive")
        for name in ("nesting_buffer", "nesting_lookahead"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("hierarchy native lowering %s must be non-negative" % name)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": dict(self.provider),
            "level_count": self.level_count,
            "nesting_buffer": self.nesting_buffer,
            "nesting_lookahead": self.nesting_lookahead,
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
    if any(row.dimension != 2 or row.ratio != (2, 2) for row in transitions):
        raise NotImplementedError(
            "shared_n_level implements only exact two-dimensional ratio-(2,2) transitions"
        )
    buffers = {row.buffer for row in transitions}
    lookaheads = {row.lookahead for row in transitions}
    if len(buffers) != 1 or len(lookaheads) != 1 \
            or any(len(set(row)) != 1 for row in buffers):
        raise NotImplementedError(
            "shared_n_level requires one isotropic buffer and one lookahead across transitions"
        )
    buffer = next(iter(buffers), (0, 0))
    lookahead = next(iter(lookaheads), 0)
    return PreparedHierarchyNativeLowering(
        authority,
        hierarchy.plan.level_count,
        buffer[0],
        lookahead,
    )


register_prepared_hierarchy_native_provider(
    PreparedHierarchyNativeProvider("shared_n_level", 1, _lower_shared_n_level)
)


__all__ = [
    "PreparedHierarchyNativeProvider",
    "PreparedHierarchyNativeLowering",
    "lower_native_hierarchy",
    "prepared_hierarchy_native_provider",
    "register_prepared_hierarchy_native_provider",
    "validate_native_hierarchy",
]
