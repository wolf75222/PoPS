"""The static reading of the geometry evolution contract (N17, C1).

A coordinate map that reads neither the clock nor a discrete input is served by static geometry
arrays (design note N17, section 4, « How C1 reads the geometry-evolution contract »): the reference
box of the layout is logical, the physical domain is the image of the map, fixed in time, and no
coupled ``geometry_state`` is needed. A map that reads the clock or a discrete input keeps the
moving route and its refusals.

The reading is structural and states no heuristic. A map whose component ``a`` reads the logical
coordinate ``a`` alone is read as node positions with exact node differences (``nodes``), at any
rank: a graded box, which ``pops.Graded`` elaborates to (design note N17, DN3), whose end nodes
keep the bounds of the box (certified at bind); any other map is a chart, of rank two, whose
measures need the stated quadrature rule of its layout (D0b; omitted is refused, SU9). A chart of
rank three is planned, not covered (design note N17, 3.10).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

STATIC_GEOMETRY_CONTRACT = "pops://geometry/static-arrays@1"
# The ranks of node positions (graded boxes) and of charts.
STATIC_GEOMETRY_RANKS = (1, 2, 3)
STATIC_GEOMETRY_CHART_RANKS = (2,)


class StaticGeometryRefusal(ValueError):
    """A clock-free coordinate map that the static arrays do not serve, with its exact reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _expression_nodes(data: Any):
    pending = [data]
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            if node.get("kind") in ("scalar", "predicate") and isinstance(node.get("op"), str):
                yield node
            pending.extend(node.values())
        elif isinstance(node, (list, tuple)):
            pending.extend(node)


def coordinate_map_reads(evolution: Mapping[str, Any]) -> frozenset[str]:
    """The leaf kinds that a serialized coordinate map reads (``coordinate``, ``time``, ...)."""
    return frozenset(node["op"] for node in _expression_nodes(evolution.get("coordinate_map", ()))
                     if node["op"] in ("coordinate", "time", "input", "parameter"))


def is_static_coordinate_map(evolution: Mapping[str, Any] | None) -> bool:
    """Whether a geometry evolution is clock-free: it reads neither the clock nor an input."""
    if evolution is None:
        return False
    return not ({"time", "input"} & coordinate_map_reads(evolution))


def _component_axes(component: Any, frame_id: str) -> frozenset[int]:
    axes = set()
    for node in _expression_nodes(component):
        if node["op"] != "coordinate":
            continue
        if node.get("frame_id") != frame_id:
            raise StaticGeometryRefusal(
                "static_geometry_frame_mismatch",
                "a static coordinate map reads one reference frame; a component reads %r, "
                "not %r" % (node.get("frame_id"), frame_id))
        axes.add(int(node["axis"]["index"]))
    return frozenset(axes)


@dataclass(frozen=True)
class StaticGeometryReading:
    """What the static arrays of one layout are generated from: data only, no callback."""

    frame_id: str
    rank: int
    kind: str
    components: tuple[Any, ...]
    rule: Mapping[str, Any] | None

    def to_data(self) -> dict[str, Any]:
        return {"contract": STATIC_GEOMETRY_CONTRACT, "frame_id": self.frame_id,
                "rank": self.rank, "kind": self.kind, "components": list(self.components),
                "rule": None if self.rule is None else dict(self.rule)}


def static_geometry_reading(evolution: Mapping[str, Any], rule: Mapping[str, Any] | None, *,
                            dimension: int) -> StaticGeometryReading:
    """Read one clock-free geometry evolution as static arrays, or refuse it precisely."""
    if not is_static_coordinate_map(evolution):
        raise StaticGeometryRefusal(
            "static_geometry_reads_clock",
            "a coordinate map that reads the clock or a discrete input is never served by the "
            "static arrays; it needs the moving geometry route")
    components = tuple(evolution["coordinate_map"])
    frame_id = evolution["frame_id"]
    if len(components) != dimension:
        raise StaticGeometryRefusal(
            "static_geometry_rank_mismatch",
            "the coordinate map has %d components for a layout of rank %d"
            % (len(components), dimension))
    if dimension not in STATIC_GEOMETRY_RANKS:
        raise StaticGeometryRefusal(
            "static_geometry_rank_unavailable",
            "static geometry arrays serve layouts of rank %s, not %d"
            % (STATIC_GEOMETRY_RANKS, dimension))
    if "parameter" in coordinate_map_reads(evolution):
        raise StaticGeometryRefusal(
            "static_geometry_parameter_unavailable",
            "a coordinate map that reads a parameter needs its bound value at bind; the static "
            "arrays of this lot take a map of numbers and logical coordinates only")
    axes = tuple(_component_axes(component, frame_id) for component in components)
    if any(not read for read in axes):
        raise StaticGeometryRefusal(
            "static_geometry_degenerate_map",
            "every component of a coordinate map must read a logical coordinate; a constant "
            "component has no cell measure")
    separable = all(read == {index} for index, read in enumerate(axes))
    if separable:
        if rule is not None:
            raise StaticGeometryRefusal(
                "static_geometry_rule_unused",
                "a map whose component a reads the logical coordinate a alone is read as node "
                "positions with exact node differences; a stated quadrature rule would state "
                "nothing, so it is refused rather than ignored")
        return StaticGeometryReading(frame_id, dimension, "nodes", components, None)
    if dimension not in STATIC_GEOMETRY_CHART_RANKS:
        raise StaticGeometryRefusal(
            "static_geometry_rank_unavailable",
            "a chart of rank %d is planned, not covered (design note N17, 3.10): a chart has rank "
            "two, and a map whose component a reads the logical coordinate a alone (node "
            "positions, a graded box) is served at every rank" % dimension)
    if rule is None:
        raise StaticGeometryRefusal(
            "static_geometry_rule_missing",
            "the measures of a chart are generated with a stated quadrature (D0b); give the "
            "layout one, MovingControlVolumes(..., geometry=pops.lib.quadrature.GaussLegendre(n))")
    return StaticGeometryReading(frame_id, dimension, "chart", components, dict(rule))


def layout_static_geometry(layout: Any) -> StaticGeometryReading | None:
    """The static reading of one normalized layout row, or ``None`` for a box or a moving map."""
    from pops._frozen_data import thaw_data

    evolution = layout.requirements.get("geometry_evolution")
    if evolution is None:
        return None
    evolution = thaw_data(evolution)
    if not is_static_coordinate_map(evolution):
        return None
    rule = layout.requirements.get("geometry_rule")
    native = layout.native_spatial_layout
    if native is None:
        raise StaticGeometryRefusal(
            "static_geometry_layout_unavailable",
            "static geometry arrays need a layout with a native spatial projection")
    return static_geometry_reading(evolution, None if rule is None else thaw_data(rule),
                                   dimension=native.dimension)


__all__ = ["STATIC_GEOMETRY_CHART_RANKS", "STATIC_GEOMETRY_CONTRACT", "STATIC_GEOMETRY_RANKS",
           "StaticGeometryReading", "StaticGeometryRefusal", "coordinate_map_reads",
           "is_static_coordinate_map", "layout_static_geometry", "static_geometry_reading"]
