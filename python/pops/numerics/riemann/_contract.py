"""Capability contract carried by every typed Riemann-flux descriptor.

The runtime must decide which model hooks a numerical flux needs without
recognising the flux class, factory name or native wire token.  This module
normalises the descriptor protocol into one immutable value that can cross the
private authoring/runtime boundary unchanged.  External C++ brick descriptors
participate through the same ``requirements["capabilities"]`` metadata as the
builtins.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class RiemannCapabilityContract:
    """Canonical model requirements and optional signed-speed authority."""

    __pops_ir_immutable__: ClassVar[bool] = True

    required_capabilities: tuple[str, ...]
    wave_speed_provider: str | None = None

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(value, str) or not value for value in capabilities):
            raise TypeError(
                "Riemann capability requirements must be non-empty strings")
        canonical = tuple(sorted(set(capabilities)))
        if capabilities != canonical:
            object.__setattr__(self, "required_capabilities", canonical)
        provider = self.wave_speed_provider
        if provider is not None and (not isinstance(provider, str) or not provider):
            raise TypeError("Riemann wave-speed provider must be non-empty text or None")
        if provider is not None and "wave_speeds" not in canonical:
            raise ValueError(
                "a Riemann wave-speed provider requires the declared wave_speeds capability")

    def requires(self, capability: str) -> bool:
        return capability in self.required_capabilities

    def to_data(self) -> dict[str, Any]:
        return {
            "required_capabilities": list(self.required_capabilities),
            "wave_speed_provider": self.wave_speed_provider,
        }


def riemann_capability_contract(descriptor: Any) -> RiemannCapabilityContract:
    """Read one typed descriptor structurally, without a class/name switch."""

    if isinstance(descriptor, str) or getattr(descriptor, "category", None) != "riemann":
        raise TypeError("Riemann capability contract requires a typed riemann descriptor")
    requirements = getattr(descriptor, "requirements", None)
    options = getattr(descriptor, "options", None)
    if not isinstance(requirements, Mapping):
        raise TypeError("Riemann descriptor requirements must be a mapping")
    if not isinstance(options, Mapping):
        raise TypeError("Riemann descriptor options must be a mapping")
    capabilities = requirements.get("capabilities", ())
    if not isinstance(capabilities, (tuple, list)):
        raise TypeError("Riemann descriptor capability requirements must be a sequence")
    return RiemannCapabilityContract(
        tuple(capabilities),
        options.get("waves"),
    )


__all__ = ["RiemannCapabilityContract", "riemann_capability_contract"]
