"""Exact component-provider packs shared by every compiler entry route.

The operator-first :class:`pops.model.Module` is the authority for provider identity.  Kernel
emitters must not rediscover providers from the legacy auxiliary layout: this module resolves the
full pack, every per-operator subset, and the physical-flux subset once and passes that immutable
value through the explicit compiler-emitter protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops.model.provider_pack import (
    ProviderPack,
    build_operator_provider_pack,
    build_provider_pack,
)


@dataclass(frozen=True, slots=True)
class ComponentProviderPacks:
    """One immutable provider resolution for a canonical Module."""

    complete: ProviderPack
    by_operator: Mapping[str, ProviderPack]
    physical_flux: ProviderPack

    def __post_init__(self) -> None:
        if type(self.complete) is not ProviderPack:
            raise TypeError("ComponentProviderPacks.complete must be an exact ProviderPack")
        rows = dict(self.by_operator)
        if any(not isinstance(name, str) or not name for name in rows):
            raise TypeError(
                "ComponentProviderPacks.by_operator keys must be non-empty strings"
            )
        if any(type(pack) is not ProviderPack for pack in rows.values()):
            raise TypeError(
                "ComponentProviderPacks.by_operator values must be exact ProviderPack values"
            )
        object.__setattr__(self, "by_operator", MappingProxyType(rows))
        if type(self.physical_flux) is not ProviderPack:
            raise TypeError(
                "ComponentProviderPacks.physical_flux must be an exact ProviderPack"
            )

    def attach(self, target: Any) -> None:
        """Attach compiler-owned immutable evidence to one emitter carrier.

        Reattachment is idempotent and verifies byte-for-byte logical equality.  This is needed
        because a facade and its private formula carrier are distinct Python objects but emit one
        native package; neither may retain a different provider resolution.
        """
        values = {
            "_component_provider_pack": self.complete,
            "_component_provider_metadata": self.complete.to_data(),
            "_component_operator_provider_packs": self.by_operator,
            "_component_operator_provider_metadata": MappingProxyType({
                name: pack.to_data() for name, pack in self.by_operator.items()
            }),
            "_component_flux_provider_pack": self.physical_flux,
            "_component_flux_provider_metadata": self.physical_flux.to_data(),
        }

        def canonical(value: Any) -> Any:
            if isinstance(value, ProviderPack):
                return value.to_data()
            if isinstance(value, Mapping):
                return {
                    key: canonical(item)
                    for key, item in value.items()
                }
            return value

        for name, value in values.items():
            previous = getattr(target, name, None)
            if previous is not None and canonical(previous) != canonical(value):
                raise ValueError(
                    "compiler emitter retained a conflicting component-provider pack"
                )
            object.__setattr__(target, name, value)


def resolve_component_provider_packs(module: Any) -> ComponentProviderPacks:
    """Resolve all exact provider packs from one canonical Module authority."""
    complete = build_provider_pack(module)
    by_operator = {
        operator.name: build_operator_provider_pack(module, operator)
        for operator in module.operator_registry()
    }
    flux_requirements = []
    for operator in module.operator_registry():
        if operator.kind == "grid_operator":
            flux_requirements.extend(by_operator[operator.name])
    physical_flux = complete.select(flux_requirements)
    return ComponentProviderPacks(
        complete=complete,
        by_operator=by_operator,
        physical_flux=physical_flux,
    )


__all__ = ["ComponentProviderPacks", "resolve_component_provider_packs"]
