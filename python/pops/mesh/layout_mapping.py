"""Typed external-component provider for cross-layout field transfer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pops.identity import canonical_bytes

from ._layout_plan_contracts import LayoutMappingRequirement


@dataclass(frozen=True, slots=True)
class NativeLayoutMapping:
    """Bind exact mapping requirements to one authenticated Transfer component.

    The provider is a resolve-time value only.  Runtime installation authenticates its complete
    component identity against the loaded ``InstalledComponent`` before exposing any native field.
    """

    component: Any
    requirements: tuple[LayoutMappingRequirement, ...]
    qualified_id: str = field(init=False)

    def __post_init__(self) -> None:
        from pops.external import ExternalComponent
        from pops.interfaces import Transfer

        if type(self.component) is not ExternalComponent:
            raise TypeError("NativeLayoutMapping.component must be an exact ExternalComponent")
        interface = self.component.component_type.interface
        if interface != Transfer:
            raise TypeError("NativeLayoutMapping requires the generated Transfer interface")
        requirements = tuple(self.requirements)
        if not requirements or any(type(row) is not LayoutMappingRequirement
                                   for row in requirements):
            raise TypeError(
                "NativeLayoutMapping.requirements must contain exact mapping requirements")
        ids = tuple(row.qualified_id for row in requirements)
        if len(ids) != len(set(ids)):
            raise ValueError("NativeLayoutMapping requirements cannot contain duplicates")
        object.__setattr__(self, "requirements", requirements)
        import hashlib

        digest = hashlib.sha256(canonical_bytes({
            "component": self.component.to_data(),
            "requirements": sorted(ids),
        })).hexdigest()
        object.__setattr__(self, "qualified_id", "pops.native-layout-mapping.v1::" + digest)

    @property
    def component_id(self) -> str:
        return self.component.component_manifest.component_id

    def supports_layout_mapping(self, requirement: LayoutMappingRequirement) -> bool:
        if type(requirement) is not LayoutMappingRequirement:
            raise TypeError("supports_layout_mapping requires an exact LayoutMappingRequirement")
        return requirement.qualified_id in {row.qualified_id for row in self.requirements}

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "qualified_id": self.qualified_id,
            "provider_type": "native_transfer_component",
            "component_id": self.component_id,
            "component": self.component.to_data(),
            "requirements": sorted(row.qualified_id for row in self.requirements),
        }


__all__ = ["NativeLayoutMapping"]
