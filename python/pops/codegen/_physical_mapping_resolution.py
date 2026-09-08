"""Resolve the bounded physical-map chain against actual Program dataflow."""
from __future__ import annotations
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
        if any(len(layout.decomposition["boxes"]) != 1 for layout in (source, target)):
            raise NotImplementedError("bounded physical maps require one native patch per layout")


def validate_physical_mapping_program(plan: Any, program: Any, field_plans: Any,
                                      *, resolve: Any) -> None:
    from pops.codegen.program_field_plan import _nodes, _reachable, _solve_nodes
    mappings = tuple(row.requirement for row in plan.mappings if row.requirement.physical_map)
    if not mappings:
        return
    if len(mappings) != 2 or sorted(int(row.operation) for row in mappings) != [2, 3]:
        raise ValueError("physical coupling requires exactly one explicit moment and pullback")
    moment = next(row for row in mappings if int(row.operation) == 2)
    pullback = next(row for row in mappings if int(row.operation) == 3)
    if (moment.source_layout, moment.target_layout) != (pullback.target_layout, pullback.source_layout):
        raise ValueError("physical coupling maps do not close their declared layout pair")
    if moment.source_port.subject == pullback.target_port.subject:
        raise ValueError("physical pullback cannot reconstruct or overwrite a distribution")
    if len(field_plans) != 1:
        raise ValueError("bounded physical coupling requires one explicit generic field solve")
    field_plan = next(iter(field_plans.values()))
    if field_plan.storage.layout != moment.target_layout:
        raise ValueError("physical field solve belongs to a different mapped field layout")
    solves = _solve_nodes(program, field_plan.handle)
    if len(solves) != 1:
        raise ValueError("bounded physical coupling requires one accepted-state field solve per step")
    solve = solves[0]
    coordinates = solve.point.to_data()
    coordinates = tuple(coordinates.get("partitions", {"main": coordinates}).values())
    if any(row["step"] != 0 or row["offset"] != {"kind": "integer", "value": "0"}
           for row in coordinates):
        raise ValueError("physical mapping field solve requires accepted-state time c=0")
    all_nodes = _nodes(program)
    reads = {resolve(row.state_ref) for row in _reachable(solve, all_nodes) if row.op == "state"}
    if reads != {moment.target_port.subject}:
        raise ValueError("physical field equation must read its explicitly mapped moment quantity")
    commits = tuple(value for state, value in program._commits.items()
                    if resolve(state) == pullback.source_port.subject)
    if len(commits) != 1:
        raise ValueError("physical pullback source requires one committed field observation")
    closure = _reachable(commits[0], all_nodes)
    if solve.id not in {row.id for row in closure} or not any(
            row.op == "field_publication" for row in closure):
        raise ValueError("physical pullback source is not a consumed fresh field observation")
