"""Native state storage for rates evaluated by the whole-system Program."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.descriptors import Descriptor


class StateStorage(Descriptor):
    """Allocate an evolved state without selecting a hyperbolic numerical flux.

    Source and interaction rates may select this method directly. Spatial operators such as
    diffusion retain their own method and use the same private storage adapter at installation.
    The generated model must independently authenticate Program-only storage support.
    """

    category = "state_storage"
    native_id = "pops::prepare_generated_system_block"
    formal_order = 1
    ghost_depth = 1

    def validate_rate_contract(self, contract: Any) -> bool:
        if not isinstance(contract, Mapping) or "state" not in contract:
            raise TypeError("StateStorage requires a rate contract containing its evolved state")
        if contract.get("flux") not in (None, ()):
            raise ValueError("StateStorage cannot discretize a physical hyperbolic flux")
        return True

    def validate_balance_view(self, view: Any) -> bool:
        from pops._ir.balance import source_balance_supported

        return source_balance_supported(view)

    def resolve_references(self, resolver: Any) -> StateStorage:
        if not callable(resolver):
            raise TypeError("StateStorage.resolve_references requires a resolver")
        return type(self)()

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": "state_storage",
            "formal_order": self.formal_order,
            "ghost_depth": self.ghost_depth,
        }

    def runtime_configuration(self) -> dict[str, Any]:
        return self.to_data()

    def runtime_spatial(self) -> Any:
        from pops.runtime._state_storage import StateStorageSpatial

        return StateStorageSpatial()


__all__ = ["StateStorage"]
