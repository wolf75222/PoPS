"""Canonical finite-volume discretization descriptors.

The physical flux, reconstructed variables, reconstruction and numerical Riemann flux are four
distinct authorities. This module stores them as inert Python values; native runtime types are
created only by :meth:`FiniteVolume.runtime_spatial` after the compile/bind boundary.
"""
from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any

from pops.descriptors import Descriptor, _native


spatial = SimpleNamespace(
    FiniteVolumeResidual=lambda **options: _native(
        "fv_residual", "pops::SpatialDiscretisation", "fv", category="spatial", **options),
    FluxDivergence=lambda **options: _native(
        "flux_divergence", "pops::SpatialDiscretisation", "fv", category="spatial", **options),
    SourceAssembly=lambda **options: _native(
        "source_assembly", "pops::SpatialDiscretisation", "fv", category="spatial", **options),
)


def _brick(value: Any, *, category: str, where: str) -> Any:
    if isinstance(value, str) or getattr(value, "category", None) != category:
        raise TypeError("%s requires a typed %s descriptor" % (where, category))
    validate = getattr(value, "validate", None)
    if not callable(validate):
        raise TypeError("%s descriptor does not implement validate()" % where)
    return value


def _resolved_value(value: Any, resolver: Any) -> Any:
    from pops.model import Handle

    if isinstance(value, Handle):
        return resolver(value)
    if isinstance(value, dict):
        return {key: _resolved_value(item, resolver) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_resolved_value(item, resolver) for item in value)
    return value


def _resolved_brick(value: Any, resolver: Any) -> Any:
    options = getattr(value, "options", None)
    if not isinstance(options, dict) or not options:
        return value
    result = copy(value)
    if hasattr(result, "_frozen"):
        object.__setattr__(result, "_frozen", False)
    object.__setattr__(result, "options", _resolved_value(options, resolver))
    return result


class FiniteVolume(Descriptor):
    """Finite-volume method for one explicit physical :class:`FluxHandle`.

    No public ``order`` or ``ghost_depth`` argument exists: both are derived from
    ``reconstruction``. Likewise, the CFL provider is derived from ``riemann`` and the model's
    physical flux capabilities.
    """

    category = "finite_volume"
    native_id = "pops::SpatialDiscretisation"

    def __init__(
        self,
        *,
        flux: Any,
        variables: Any,
        reconstruction: Any,
        riemann: Any,
        positivity_floor: Any = None,
    ) -> None:
        from pops.model import Handle

        if not isinstance(flux, Handle) or flux.kind != "flux":
            raise TypeError("FiniteVolume(flux=) requires a physical FluxHandle")
        self.flux = flux
        self.variables = _brick(
            variables, category="variables", where="FiniteVolume.variables")
        self.reconstruction = _brick(
            reconstruction, category="reconstruction", where="FiniteVolume.reconstruction")
        self.riemann = _brick(riemann, category="riemann", where="FiniteVolume.riemann")
        if positivity_floor is not None:
            if isinstance(positivity_floor, bool) or not isinstance(positivity_floor, (int, float)):
                raise TypeError("FiniteVolume.positivity_floor must be a non-negative scalar or None")
            if positivity_floor < 0:
                raise ValueError("FiniteVolume.positivity_floor must be >= 0")
        self.positivity_floor = positivity_floor
        state = self.variables.options.get("state")
        if state is not None and state.owner_path != flux.owner_path:
            raise ValueError("FiniteVolume variables and physical flux belong to different Models")
        velocity = self.riemann.options.get("velocity")
        if velocity is not None and velocity.owner_path != flux.owner_path:
            raise ValueError("FiniteVolume Riemann velocity and physical flux belong to different Models")

    def options(self) -> dict[str, Any]:
        return {
            "flux": self.flux,
            "variables": self.variables,
            "reconstruction": self.reconstruction,
            "riemann": self.riemann,
            "positivity_floor": self.positivity_floor,
        }

    @property
    def formal_order(self) -> int:
        value = self.reconstruction.options.get("formal_order")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("reconstruction does not declare a valid formal_order")
        return value

    @property
    def ghost_depth(self) -> int:
        value = self.reconstruction.options.get("ghost_depth")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("reconstruction does not declare a valid ghost_depth")
        return value

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet

        return RequirementSet({
            "physical_flux": self.flux.qualified_id,
            "ghost_depth": self.ghost_depth,
            "riemann_capabilities": tuple(
                self.riemann.requirements.get("capabilities", ())),
        })

    def capabilities(self) -> Any:
        from pops.descriptors_report import CapabilitySet

        return CapabilitySet({
            "formal_order": self.formal_order,
            "ghost_depth": self.ghost_depth,
            "conservative_variables": self.variables.scheme == "conservative",
            "cfl_provider": self.riemann.name,
        })

    def validate(self, context: Any = None) -> bool:
        self.variables.validate(context)
        self.reconstruction.validate(context)
        self.riemann.validate(context)
        self.formal_order
        self.ghost_depth
        return True

    def resolve_references(self, resolver: Any) -> "FiniteVolume":
        if not callable(resolver):
            raise TypeError("FiniteVolume.resolve_references requires a resolver")
        return type(self)(
            flux=resolver(self.flux),
            variables=_resolved_brick(self.variables, resolver),
            reconstruction=_resolved_brick(self.reconstruction, resolver),
            riemann=_resolved_brick(self.riemann, resolver),
            positivity_floor=self.positivity_floor,
        )

    def to_data(self) -> dict[str, Any]:
        if not self.flux.is_resolved:
            raise ValueError("FiniteVolume.to_data requires resolved physical handles")
        return {
            "schema_version": 1,
            "method": "finite_volume",
            "flux": self.flux.canonical_identity(),
            "variables": self.variables.inspect(),
            "reconstruction": self.reconstruction.inspect(),
            "riemann": self.riemann.inspect(),
            "formal_order": self.formal_order,
            "ghost_depth": self.ghost_depth,
            "positivity_floor": self.positivity_floor,
        }

    def runtime_spatial(self) -> Any:
        """Lower at the native boundary to the existing optimized runtime value."""
        from pops.runtime._bricks_scheme import Spatial

        return Spatial(
            limiter=self.reconstruction,
            flux=self.riemann,
            recon=self.variables,
            positivity_floor=self.positivity_floor,
        )


__all__ = ["FiniteVolume", "spatial"]
