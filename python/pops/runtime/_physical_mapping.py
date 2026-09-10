"""Authenticated physical geometry and effect-ordered layout transactions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

_BEFORE = "pops://synchronization/before-step@1"
_AFTER = "pops://synchronization/after-source-step@1"


@dataclass(frozen=True, slots=True)
class PhysicalMappingSchedule:
    """Accepted-source captures followed by a stable map/child-step dependency DAG.

    A child Program remains an atomic native step. Internal Program stage points
    are preserved; this schedule does not claim inter-stage communication.
    """
    accepted_captures: tuple[str, ...]
    events: tuple[tuple[str, str], ...]


def physical_mapping_schedule(transfers: Any, layouts: Any) -> PhysicalMappingSchedule | None:
    rows = tuple(transfers)
    if not any(row.operation_abi in (2, 3) for row in rows):
        return None
    layout_ids = set(layouts)
    by_id = {row.mapping_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("physical mapping schedule repeats a mapping identity")
    steps = {("step", layout) for layout in layout_ids}
    dependencies = {event: set() for event in steps}
    captures = []
    targets = {}
    for row in rows:
        if row.source_layout_id not in layout_ids or row.target_layout_id not in layout_ids:
            raise ValueError("physical mapping schedule names an unknown layout")
        target = row.target_layout_id, row.target_subject_id
        if target in targets:
            raise ValueError("multiple physical mappings overwrite one subject in an atomic child step; "
                             "an explicit merge or Program continuation is required")
        targets[target] = row.mapping_id
        event = ("map", row.mapping_id)
        dependencies[event] = set()
        dependencies[("step", row.target_layout_id)].add(event)
        if row.synchronization_uri == _BEFORE:
            # All accepted reads are captured before any mapped write or child commit.
            captures.append(row.mapping_id)
        elif row.synchronization_uri == _AFTER:
            dependencies[event].add(("step", row.source_layout_id))
        else:
            raise ValueError("physical mapping schedule requires an explicit supported synchronization")
    events = []
    while dependencies:
        ready = sorted(event for event, needed in dependencies.items() if not needed)
        if not ready:
            raise ValueError("physical mapping Program dependencies form a cycle across atomic child steps; "
                             "inter-stage communication requires a native Program continuation")
        for event in ready:
            events.append(event)
            del dependencies[event]
        for needed in dependencies.values():
            needed.difference_update(ready)
    return PhysicalMappingSchedule(tuple(sorted(captures)), tuple(events))


def validate_physical_geometry(requirement: Any, source: Any, target: Any, *,
                               composite: bool = False) -> None:
    physical = requirement.physical_map
    if physical is None:
        raise ValueError("physical Transfer lost its resolved physical map")
    dimension = physical.native_dimension
    if len(source.shape) != dimension or len(target.shape) != dimension:
        raise ValueError("physical map native dimension disagrees with its explicit storage embedding")
    if source.coordinate_system != target.coordinate_system:
        raise ValueError("physical maps require matching Cartesian coordinate systems")
    for source_axis, target_axis in enumerate(physical.source_to_target):
        if target_axis < 0:
            continue
        if ((not composite and source.shape[source_axis] != target.shape[target_axis]) or
                source.lower[source_axis] != target.lower[target_axis] or
                source.upper[source_axis] != target.upper[target_axis] or
                source.periodicity[source_axis] != target.periodicity[target_axis]):
            raise ValueError("physical maps require exactly aligned shared coordinate geometry/topology")
    for layout, active in ((source, physical.source_axes), (target, physical.target_axes)):
        for axis in range(dimension):
            if axis not in active and (layout.shape[axis] != 1 or layout.lower[axis] != 0 or
                                      layout.upper[axis] != 1 or not layout.periodicity[axis]):
                raise ValueError("physical map requires periodic unit-measure singleton hidden storage axes")
    for quadrature in physical.reductions:
        axis = quadrature.axis
        if (source.shape[axis] != quadrature.cells or source.lower[axis] != float(quadrature.lower) or
                source.upper[axis] != float(quadrature.upper)):
            raise ValueError("quadrature does not authenticate its source axis geometry")
