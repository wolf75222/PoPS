"""Authenticated bounded physical mapping geometry and ordered execution."""
from __future__ import annotations
from typing import Any


def physical_mapping_order(transfers: Any) -> tuple[str, str] | None:
    rows = tuple(transfers)
    if not any(row.operation_abi in (2, 3) for row in rows):
        return None
    if len(rows) != 2 or sorted(row.operation_abi for row in rows) != [2, 3]:
        raise NotImplementedError("bounded physical coupling requires one moment and one pullback")
    moment = next(row for row in rows if row.operation_abi == 2)
    pullback = next(row for row in rows if row.operation_abi == 3)
    if (moment.source_layout_id, moment.target_layout_id) != (
            pullback.target_layout_id, pullback.source_layout_id):
        raise ValueError("physical moment and pullback must explicitly close the same layout pair")
    if moment.source_subject_id == pullback.target_subject_id:
        raise ValueError("pullback cannot overwrite the distribution as an inferred inverse closure")
    if moment.target_subject_id == pullback.source_subject_id:
        raise ValueError("pullback requires a separate field observation, not the moment source")
    if (moment.synchronization_uri != "pops://synchronization/before-step@1" or
            pullback.synchronization_uri != "pops://synchronization/after-source-step@1"):
        raise ValueError("physical maps require moment -> field step -> pullback -> distribution step")
    return moment.target_layout_id, moment.source_layout_id


def validate_physical_geometry(requirement: Any, source: Any, target: Any) -> None:
    physical = requirement.physical_map
    if physical is None:
        raise ValueError("physical Transfer lost its resolved physical map")
    moment = physical.operation_abi == 2
    phase, field = (source, target) if moment else (target, source)
    if len(phase.shape) != 2 or len(field.shape) != 2:
        raise NotImplementedError("1x1v/1x physical maps require one Dim=2 native artifact")
    if (phase.shape[0] != field.shape[0] or phase.lower[0] != field.lower[0] or
            phase.upper[0] != field.upper[0] or
            phase.periodicity[0] != field.periodicity[0] or
            phase.coordinate_system != field.coordinate_system):
        raise ValueError("physical maps require exactly aligned physical x geometry/topology")
    if (field.shape[1] != 1 or field.lower[1] != 0 or field.upper[1] != 1 or
            not field.periodicity[1]):
        raise ValueError("1x field requires a periodic unit-measure singleton hidden storage axis")
    if moment:
        quad = physical.quadrature
        if (phase.shape[1] != quad.cells or phase.lower[1] != float(quad.lower) or
                phase.upper[1] != float(quad.upper)):
            raise ValueError("velocity quadrature does not authenticate the distribution geometry")
