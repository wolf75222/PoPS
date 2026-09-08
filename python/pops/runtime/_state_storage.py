"""Exact private native adapter for authenticated Program-only state storage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class StateStorageSpatial:
    """Carry the storage ABI without manufacturing a finite-volume/Riemann authority."""

    limiter: ClassVar[str] = "state_storage"
    flux: ClassVar[str] = "unavailable"
    recon: ClassVar[str] = "conservative"
    positivity_floor: ClassVar[float] = 0.0
    wave_speed_cache: ClassVar[bool] = False
    weno_epsilon: ClassVar[None] = None
    external_flux_id: ClassVar[None] = None
    waves_provider: ClassVar[None] = None

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "family": "state_storage",
            "limiter": self.limiter,
            "flux": self.flux,
            "recon": self.recon,
            "ghost_depth": 1,
        }

    def identity(self) -> Any:
        from pops.identity import make_identity

        return make_identity("spatial", self.to_data(), schema_version=1)

    def routes(self) -> dict[str, Any]:
        return {"storage": self.to_data()}

    def validate(self, ghost_depth: Any = None, block: Any = None) -> bool:
        if ghost_depth is not None and (type(ghost_depth) is not int or ghost_depth < 1):
            raise ValueError("StateStorage requires at least one ghost cell for block %r" % block)
        return True


def require_state_storage_model(model: Any, spatial: Any, *, where: str) -> bool:
    """Authenticate the exact adapter and compiled fact before native storage installation."""
    if type(spatial) is not StateStorageSpatial:
        return False
    from pops.codegen._compiled_model_boundary import validate_compiled_model_result

    validate_compiled_model_result(model, allow_install_plan=True)
    if model.caps.get("program_only_storage") is not True:
        raise ValueError(
            "%s: StateStorage requires an authenticated Program-only generated model" % where
        )
    return True


__all__ = ["StateStorageSpatial", "require_state_storage_model"]
