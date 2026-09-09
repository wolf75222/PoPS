"""Resolve physical support maps against actual Program reads, commits and timing."""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any


def validate_physical_mapping_geometry(plan: Any) -> None:
    from pops.runtime._physical_mapping import validate_physical_geometry
    for row in plan.mappings:
        requirement = row.requirement
        if requirement.physical_map is None:
            source = getattr(requirement.source_port.subject, "space", None)
            target = getattr(requirement.target_port.subject, "space", None)
            if source is not None and target is not None and source.support != target.support:
                raise ValueError("different physical supports require an explicit physical map: "
                                 + requirement.qualified_id)
            continue
        source = plan.normalized(requirement.source_layout).native_spatial_layout
        target = plan.normalized(requirement.target_layout).native_spatial_layout
        validate_physical_geometry(requirement, source, target)


def validate_physical_mapping_program(plan: Any, program: Any, field_plans: Any,
                                      *, resolve: Any) -> None:
    from pops.codegen.program_field_plan import _nodes, _reachable
    from pops.runtime._physical_mapping import physical_mapping_schedule
    mappings = tuple(row.requirement for row in plan.mappings)
    if not any(row.physical_map is not None for row in mappings):
        return
    # Field solve count and stage points are properties of the authored Program.
    # The field resolver authenticates each solve and its consumed publication.
    # A support map has no reason to prescribe either a field equation or c=0.
    del field_plans
    from pops.codegen.program_mapping_regions import (
        resolve_program_map_invocations, plan_program_mapping_regions)
    from pops.mesh import LayoutSynchronization
    invocations = resolve_program_map_invocations(program, plan, resolve=resolve)
    if invocations:
        plan_program_mapping_regions(program, {
            row.subject.local_id: row.layout.qualified_id for row in plan.assignments
            if row.subject_kind == "block"})
    all_nodes = _nodes(program)
    commits = {}
    readers = {}
    for state, value in program._commits.items():
        subject = resolve(state)
        commits.setdefault(subject, []).append(value)
        for node in _reachable(value, all_nodes):
            if node.op == "state":
                readers.setdefault(resolve(node.state_ref), set()).add(subject)
    assignments = {row.subject: row.layout for row in plan.assignments
                   if row.subject_kind == "state"}
    transfers = []
    for requirement in mappings:
        if requirement.synchronization is LayoutSynchronization.PROGRAM_POINT_V1:
            continue
        source = requirement.source_port.subject
        target = requirement.target_port.subject
        if requirement.physical_map is not None:
            consumers = readers.get(target, set())
            if not consumers:
                raise ValueError("physical mapping target is not read by any committed Program value")
            if any(assignments.get(consumer) != requirement.target_layout for consumer in consumers):
                raise ValueError("physical map target is consumed outside its declared native layout")
            if requirement.synchronization.value == "pops://synchronization/after-source-step@1":
                if len(commits.get(source, ())) != 1:
                    raise ValueError("after-source-step physical mapping requires one committed source value")
        transfers.append(SimpleNamespace(mapping_id=requirement.qualified_id,
            operation_abi=int(requirement.operation), source_layout_id=requirement.source_layout.qualified_id,
            target_layout_id=requirement.target_layout.qualified_id,
            source_subject_id=source.qualified_id, target_subject_id=target.qualified_id,
            synchronization_uri=requirement.synchronization.value))
    physical_mapping_schedule(transfers, (row.handle.qualified_id for row in plan.layouts))
