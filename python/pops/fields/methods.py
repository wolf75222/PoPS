"""Typed spatial methods for field operators."""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet


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

    def lower_field_method(self, *, target: str, layout: Any) -> dict[str, Any]:
        del target, layout
        return {
            "native_method": "cell_centered_second_order",
            "order": 2,
            "ghost_depth": 1,
        }


__all__ = ["CellCenteredSecondOrder"]
