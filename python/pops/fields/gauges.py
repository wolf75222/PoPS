"""Typed gauge choices, separate from mathematical nullspace declarations."""

from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.model import Handle

from ._identity import strict_field_data
from ._references import reference_label, resolve_handle


class FieldGauge(Descriptor):
    category = "field_gauge"

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}


class MeanValueGauge(FieldGauge):
    """Fix the domain mean of the unknown to an explicit value."""

    def __init__(self, value: Any = 0) -> None:
        self.value = value
        strict_field_data(value)

    def options(self) -> dict[str, Any]:
        return {"gauge": "mean_value", "value": strict_field_data(self.value)}


class PinnedValueGauge(FieldGauge):
    """Fix the unknown at a model-owned point/degree-of-freedom handle."""

    def __init__(self, location: Any, value: Any = 0) -> None:
        if not isinstance(location, Handle):
            raise TypeError("PinnedValueGauge location must be a declaration Handle")
        self.location = location
        self.value = value
        strict_field_data(value)

    def options(self) -> dict[str, Any]:
        return {
            "gauge": "pinned_value",
            "location": reference_label(self.location, where="gauge location"),
            "value": strict_field_data(self.value),
        }

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.location,)

    def resolve_references(self, resolver: Any) -> PinnedValueGauge:
        return PinnedValueGauge(
            resolve_handle(self.location, resolver, where="gauge location"), self.value
        )


__all__ = ["FieldGauge", "MeanValueGauge", "PinnedValueGauge"]
