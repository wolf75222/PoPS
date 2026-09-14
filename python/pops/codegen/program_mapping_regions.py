"""Immutable region dependencies for stage-qualified physical map invocations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops.identity.encoding import canonical_bytes
from pops.time.canonical_data import _json_ready
from pops.time.values import ProgramValue

MAP_EXPORT = "layout_map_export"
MAP_IMPORT = "layout_map_import"
MAP_OPS = frozenset((MAP_EXPORT, MAP_IMPORT))


@dataclass(frozen=True)
class ProgramMapInvocation:
    identity: str
    source: Any
    target: Any
    requirement: Any = None


@dataclass(frozen=True)
class ProgramMappingRegion:
    layout: str
    index: int
    values: tuple[Any, ...]
    barrier: str | None
    direction: str | None


@dataclass(frozen=True)
class ProgramMappingRegions:
    invocations: tuple[ProgramMapInvocation, ...]
    regions: tuple[ProgramMappingRegion, ...]
    events: tuple[tuple[str, str, int], ...]


def program_map_invocations(program: Any, *, allow_partition: bool = False) -> tuple[ProgramMapInvocation, ...]:
    ports: dict[str, dict[str, Any]] = {}
    for node in program._values:
        if node.op not in MAP_OPS:
            continue
        if node.region != 0:
            raise ValueError("physical maps require top-level Program barriers")
        identity = node.attrs.get("invocation")
        if not isinstance(identity, str) or not identity.startswith("pops.program-map.v1::"):
            raise ValueError("Program map lacks an authenticated invocation identity")
        row = ports.setdefault(identity, {})
        if node.op in row:
            raise ValueError("Program map duplicates a directional invocation port")
        row[node.op] = node
    result = []
    for identity, ports_by_direction in sorted(ports.items()):
        source, target = ports_by_direction.get(MAP_EXPORT), ports_by_direction.get(MAP_IMPORT)
        if source is None or target is None:
            if allow_partition:
                continue
            raise ValueError("Program map requires exactly one source and one target port")
        # Compare canonical source metadata without relying on live object equality after detach.
        if canonical_bytes(_json_ready(source.attrs)) != canonical_bytes(_json_ready(target.attrs)):
            raise ValueError("Program map directional ports disagree on their exact contract")
        if len(source.inputs) != 1 or source.inputs[0].state_ref != source.attrs["source_state"]:
            raise ValueError("Program map source port does not read its declared qualified state")
        if source.state_ref != source.attrs["source_state"] or target.state_ref != target.attrs["target_state"]:
            raise ValueError("Program map port state provenance differs from its declaration")
        result.append(ProgramMapInvocation(identity, source, target))
    return tuple(result)


def resolve_program_map_invocations(program: Any, plan: Any, *, resolve: Any) -> tuple[ProgramMapInvocation, ...]:
    from pops.mesh import LayoutSynchronization
    matched = set()
    resolved = []
    for invocation in program_map_invocations(program):
        attrs = invocation.source.attrs
        source, target = resolve(attrs["source_state"]), resolve(attrs["target_state"])
        candidates = [row.requirement for row in plan.mappings
            if row.requirement.source_port.subject == source
            and row.requirement.target_port.subject == target
            and row.requirement.synchronization is LayoutSynchronization.PROGRAM_POINT_V1
            and row.requirement.physical_map is not None
            and canonical_bytes(row.requirement.physical_map.to_data()) == canonical_bytes(_json_ready(attrs["physical_map"]))]
        if len(candidates) != 1:
            raise ValueError("Program.map must resolve to exactly one matching program-point layout mapping")
        requirement = candidates[0]
        matched.add(requirement.qualified_id)
        resolved.append(ProgramMapInvocation(invocation.identity, invocation.source,
                                            invocation.target, requirement))
    declared = {row.requirement.qualified_id for row in plan.mappings
                if row.requirement.synchronization is LayoutSynchronization.PROGRAM_POINT_V1}
    if matched != declared:
        raise ValueError("program-point layout mapping has no authenticated Program.map invocation")
    return tuple(resolved)


def compiled_program_map_invocations(artifact: Any) -> tuple[ProgramMapInvocation, ...]:
    from types import SimpleNamespace
    from pops.time.references import canonical_handle
    from pops.mesh import LayoutSynchronization
    if not any(row.requirement.synchronization is LayoutSynchronization.PROGRAM_POINT_V1
               for row in artifact.layout_plan.mappings):
        return ()
    values = tuple(value for row in artifact.layout_programs for value in row.program.program._values)
    return resolve_program_map_invocations(SimpleNamespace(_values=values), artifact.layout_plan,
                                           resolve=canonical_handle)


def plan_program_mapping_regions(program: Any, layout_by_block: Mapping[str, str]) -> ProgramMappingRegions:
    """Topologically order native regions and explicit transfer edges, preserving local effects.

    Each layout's authored operations retain their order. Both peers stop at the matching port;
    transfer completes before either continuation resumes. No relation is inferred from physical
    opcode, stage label, map declaration order, or equality of unrelated stage coordinates.
    """
    invocations = program_map_invocations(program)
    if not invocations:
        return ProgramMappingRegions((), (), ())
    layouts = tuple(sorted(set(layout_by_block.values())))
    if not layouts:
        raise ValueError("Program mapping regions require resolved layout ownership")
    values: dict[str, tuple[ProgramValue, ...]] = {}
    # Slicing owns scalar/control dependency projection. Reuse it rather than duplicating an SSA
    # closure here; its preserved invocation token joins renamed per-layout SSA namespaces.
    from pops.codegen.program_slicing import slice_program
    for layout in layouts:
        names = [block for block, owner in layout_by_block.items() if owner == layout]
        values[layout] = tuple(slice_program(program, names)._values)
    regions = []
    dependencies: dict[tuple[str, str, int], set[tuple[str, str, int]]] = {}
    port_regions = {}
    for layout in layouts:
        pending = []
        index = 0
        prior = None
        for value in values[layout]:
            pending.append(value)
            if value.op not in MAP_OPS:
                continue
            identity = value.attrs["invocation"]
            event = ("region", layout, index)
            dependencies[event] = set() if prior is None else {prior}
            regions.append(ProgramMappingRegion(layout, index, tuple(pending), identity, value.op))
            key = (identity, value.op)
            if key in port_regions:
                raise ValueError("Program map port appears in multiple layout regions")
            port_regions[key] = event
            prior = ("map", identity, 0)
            pending = []
            index += 1
        event = ("region", layout, index)
        dependencies[event] = set() if prior is None else {prior}
        regions.append(ProgramMappingRegion(layout, index, tuple(pending), None, None))
    for invocation in invocations:
        source = port_regions.get((invocation.identity, MAP_EXPORT))
        target = port_regions.get((invocation.identity, MAP_IMPORT))
        if source is None or target is None or source[1] == target[1]:
            raise ValueError("Program map requires one port on each of two distinct layouts")
        dependencies[("map", invocation.identity, 0)] = {source, target}
    events = []
    while dependencies:
        ready = sorted(key for key, parents in dependencies.items() if not parents)
        if not ready:
            raise ValueError("Program map region dependencies contain a communication cycle")
        for event in ready:
            events.append(event)
            del dependencies[event]
        for parents in dependencies.values():
            parents.difference_update(ready)
    return ProgramMappingRegions(invocations, tuple(regions), tuple(events))
