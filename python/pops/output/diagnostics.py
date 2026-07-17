"""Composite AMR scientific reductions and explicit balance accounting."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .data import (
    OutputRequest,
    OutputSnapshot,
    _CARTESIAN_CELL_AREA,
    _composite_integral_authority_identity,
    _field_family_identity,
)


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
    if type(snapshot) is not OutputSnapshot or type(request) is not OutputRequest:
        raise TypeError(
            "composite_integrals requires exact OutputSnapshot and OutputRequest values")
    selected: dict[str, dict[str, Any]] = {}
    for field in snapshot.select(request):
        if len(field.component_names) > 1:
            raise ValueError(
                "composite_integrals requires scalar selections; select a component explicitly")
        geometry = snapshot.geometry(field.key)
        if geometry.layout_kind != "amr":
            raise ValueError("composite_integrals requires an adaptive AMR layout")
        if geometry.cell_measure != _CARTESIAN_CELL_AREA:
            raise NotImplementedError(
                "composite_integrals currently supports only the native Cartesian cell-area "
                "metric; non-Cartesian measures require a typed native metric provider")
        family = _field_family_identity(field.key).token
        row = selected.setdefault(family, {
            "components": field.component_names,
            "levels": [],
            "family_identity": _field_family_identity(field.key),
        })
        if row["components"] != field.component_names:
            raise ValueError("one output field family has inconsistent component metadata")
        row["levels"].append(field.key.level)
    evidence = {
        item.authority_identity.token: item.value
        for item in snapshot._native_composite_integrals
    }
    requested = {
        family: _composite_integral_authority_identity(
            row["family_identity"], tuple(sorted(row["levels"]))).token
        for family, row in selected.items()
    }
    missing = sorted(
        family for family, authority in requested.items() if authority not in evidence)
    if missing:
        raise RuntimeError(
            "composite_integrals requires accepted-state native C++/Kokkos reduction evidence "
            "for the exact selected level tuple; detached, under-selected, or over-selected "
            "snapshots cannot be reduced")
    return MappingProxyType({
        family: evidence[requested[family]] for family in sorted(selected)
    })


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
