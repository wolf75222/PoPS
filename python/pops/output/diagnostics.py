"""Composite AMR scientific reductions and explicit balance accounting."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops.identity import make_identity

from .data import OutputRequest, OutputSnapshot


def _finite(value: Any, where: str) -> float:
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError("%s must be finite" % where)
    return result


def composite_integrals(snapshot: OutputSnapshot, request: OutputRequest) -> Any:
    """Metric-weighted integrals, excluding every covered coarse cell.

    Results are grouped by the exact owner-qualified field, component manifest, layout and accepted
    state while levels are reduced together. Vector/component fields are intentionally refused: a
    caller must select a scalar component explicitly rather than receive an implicit reduction.
    """
    import numpy as np

    totals: dict[str, float] = {}
    for field in snapshot.select(request):
        if field.component_names:
            raise ValueError(
                "composite_integrals requires scalar selections; select a component explicitly")
        geometry = snapshot.geometry(field.key)
        values = field.materialize()
        active = np.logical_and(geometry.valid_cells, np.logical_not(geometry.coverage))
        value = float(np.sum(values[active] * geometry.cell_volumes[active], dtype=np.float64))
        family = make_identity("output-field-family", {
            "reference": field.key.reference.canonical_identity(),
            "component_manifest_identity": field.key.component_manifest_identity.token,
            "layout_identity": field.key.layout_identity.token,
            "state_id": field.key.state_id,
        }).token
        totals[family] = totals.get(family, 0.0) + value
    return MappingProxyType(dict(sorted(totals.items())))


@dataclass(frozen=True, slots=True)
class BalanceTerms:
    """All signed terms of an open-domain discrete balance.

    Convention: ``residual = storage_change + outward_boundary_flux - sources - reflux -
    projection``. Nothing here is called an invariant: boundary/source terms are mandatory inputs.
    """

    storage_change: float
    outward_boundary_flux: float
    sources: float
    reflux: float
    projection: float

    def __post_init__(self) -> None:
        for name in (
            "storage_change", "outward_boundary_flux", "sources", "reflux", "projection",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    @property
    def residual(self) -> float:
        return (self.storage_change + self.outward_boundary_flux - self.sources
                - self.reflux - self.projection)

    def to_data(self) -> dict[str, str]:
        return {
            "storage_change": self.storage_change.hex(),
            "outward_boundary_flux": self.outward_boundary_flux.hex(),
            "sources": self.sources.hex(), "reflux": self.reflux.hex(),
            "projection": self.projection.hex(), "residual": self.residual.hex(),
        }


__all__ = ["BalanceTerms", "composite_integrals"]
