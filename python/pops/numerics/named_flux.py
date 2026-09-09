"""Program-owned centered divergence for exact named physical fluxes."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.descriptors import Descriptor


class NamedCenteredDivergence(Descriptor):
    """Select the native centered divergence of an ordered named-flux sum.

    The compiled Program materializes the named physical fluxes and owns the single centered
    divergence stencil.  The block package supplies state storage and halo preparation only.
    """

    category = "named_centered_divergence"
    native_id = "pops::runtime::program::ProgramContext::neg_div_named_flux_into"
    formal_order = 2
    ghost_depth = 1

    def __init__(self, *, flux: Any) -> None:
        from pops.model import Handle

        if not isinstance(flux, tuple) or not flux:
            raise TypeError(
                "NamedCenteredDivergence flux must be a non-empty ordered tuple"
            )
        if any(not isinstance(item, Handle) or item.kind != "grid_operator" for item in flux):
            raise TypeError(
                "NamedCenteredDivergence flux must contain typed grid_operator handles"
            )
        owner = flux[0].owner_path
        if any(item.owner_path != owner for item in flux[1:]):
            raise ValueError("NamedCenteredDivergence fluxes belong to different Models")
        self.flux = flux

    def validate(self, context: Any = None) -> bool:
        del context
        return True

    def validate_rate_contract(self, contract: Any) -> bool:
        if not isinstance(contract, Mapping) or "state" not in contract or "flux" not in contract:
            raise TypeError(
                "NamedCenteredDivergence requires a rate contract containing state and flux"
            )
        if contract["flux"] != self.flux:
            raise ValueError(
                "NamedCenteredDivergence flux does not match the named fluxes referenced by the rate"
            )
        return True

    def resolve_references(self, resolver: Any) -> NamedCenteredDivergence:
        if not callable(resolver):
            raise TypeError("NamedCenteredDivergence.resolve_references requires a resolver")
        return type(self)(flux=tuple(resolver(item) for item in self.flux))

    def to_data(self) -> dict[str, Any]:
        if any(not item.is_resolved for item in self.flux):
            raise ValueError("NamedCenteredDivergence.to_data requires resolved flux handles")
        return {
            "schema_version": 1,
            "method": "native_named_centered_divergence",
            "flux": [item.canonical_identity() for item in self.flux],
            "formal_order": self.formal_order,
            "ghost_depth": self.ghost_depth,
        }

    def runtime_configuration(self) -> dict[str, Any]:
        from .state_storage import StateStorage

        return StateStorage().runtime_configuration()

    def runtime_spatial(self) -> Any:
        from pops.runtime._state_storage import StateStorageSpatial

        return StateStorageSpatial()


__all__ = ["NamedCenteredDivergence"]
