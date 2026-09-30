"""Typed spatial methods for field operators."""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet


class PreparedFieldMethod(Descriptor):
    """Generic method descriptor backed by one registered lowering provider."""

    category = "field_method"

    def __init__(self, provider: Any, **options: Any) -> None:
        from ._prepared_field_lowering_registry import (
            PreparedFieldLoweringProvider,
            prepared_field_lowering_provider_by_resolver_id,
        )

        if type(provider) is not PreparedFieldLoweringProvider:
            raise TypeError("PreparedFieldMethod requires an exact registered Provider")
        if prepared_field_lowering_provider_by_resolver_id(
            provider.resolver_id
        ) is not provider:
            raise ValueError("PreparedFieldMethod provider is not the registered authority")
        self.provider = provider
        self.provider_options = dict(options)

    @property
    def name(self) -> str:
        return self.provider.provider_id

    def options(self) -> dict[str, Any]:
        return dict(self.provider_options)

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "provider": self.provider.authority(),
            "options": self.options(),
        }

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(dict(self.provider.capabilities))

    def _prepared_field_lowering(self) -> tuple[Any, dict[str, Any]]:
        return self.provider, self.options()


class CellCenteredSecondOrder(Descriptor):
    """Native cell-centred second-order elliptic stencil.

    Order and halo depth are consequences of this method and are therefore
    capabilities, never duplicate constructor arguments on FieldDiscretization.
    """

    category = "field_method"
    native_id = "pops::CellCenteredEllipticOperator"

    def options(self) -> dict[str, Any]:
        return {"method": "cell_centered_second_order"}

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({
            "native_cell_centered_elliptic": True,
            "order": 2,
            "ghost_depth": 1,
        })

    def _prepared_field_lowering(self) -> tuple[Any, dict[str, Any]]:
        """Bind authoring to the authenticated complete lowering provider.

        The descriptor deliberately carries no target/layout branches.  Those decisions belong to
        the selected provider and its versioned capability contract.
        """
        from pops.codegen._cell_centered_field_lowering import (
            cell_centered_second_order_field_lowering_provider,
        )
        return cell_centered_second_order_field_lowering_provider(), {}


class CellCenteredGeneralCoupled(CellCenteredSecondOrder):
    """Cell-centred matrix-free stencil with finite, possibly nonsymmetric coupling.

    This realization uses a general Krylov method.  It carries no SPD/energy
    certificate and does not alter the default positive-definite field route.
    """

    def options(self) -> dict[str, Any]:
        return {"method": "cell_centered_second_order",
                "coefficient_admissibility": "finite_general"}


class CellCenteredNonlinearCoupled(CellCenteredGeneralCoupled):
    """Original mixed residual with explicitly selected central finite differences.

    The first realization admits constant diffusion and local nonlinear reactions,
    on Cartesian Uniform or synchronized AMR layouts. The AMR realization applies
    one covered composite operator and solves the full coupled hierarchy.
    """

    def __init__(self, *, finite_difference_step: Any) -> None:
        import math
        from pops.identity.scalar import exact_numeric_scalar

        step = exact_numeric_scalar(finite_difference_step, where="field residual FD step")
        if not math.isfinite(float(step)) or step <= 0:
            raise ValueError("field residual FD step must be positive and finite")
        self.finite_difference_step = step

    def options(self) -> dict[str, Any]:
        from pops.identity.scalar import scalar_data

        return {**super().options(), "contract": "pops.spatial-field-residual@1",
                "derivative": {"route": "finite_difference", "scheme": "central_full_residual",
                               "step": scalar_data(self.finite_difference_step)}}


__all__ = ["CellCenteredSecondOrder", "CellCenteredGeneralCoupled",
           "CellCenteredNonlinearCoupled", "PreparedFieldMethod"]
