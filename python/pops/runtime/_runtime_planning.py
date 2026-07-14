"""Pure, fail-closed derivation of RuntimeInstance planning values.

This module consumes authenticated phase records only.  It does not load binaries, initialize a
backend, inspect process-global state, or call a runtime adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops._platform_contracts import ExecutionContext
from pops.codegen._plans import InstallPlan, require_install_plan
from pops.mesh import LayoutPlan
from pops.model import ComponentManifest

from ._runtime_plan_contracts import (
    BufferAllocation,
    ClockJoin,
    Collective,
    CommunicationPlan,
    DeterminismGuarantee,
    Fence,
    FieldAccess,
    HaloExchange,
    LayoutTransfer,
    ResourcePlan,
    ResourceUse,
    RuntimeCall,
    RuntimePlanBundle,
    refuse,
)
from ._runtime_plan_io import component_features as _features
from ._runtime_plan_io import component_map as _component_map
from ._runtime_plan_io import proved_platform as _proved_platform
from ._runtime_plan_io import validate_component_platform as _validate_component_platform


_RUNTIME_REQUIREMENTS = frozenset(("halo", "collective", "buffer"))
_DETERMINISM_ORDER = {"bitwise": 0, "reproducible": 1, "statistical": 2, "nondeterministic": 3}


def _plain(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        refuse(
            "noncanonical_runtime_evidence",
            path,
            "%s cannot contain binary floats" % path,
            evidence=value,
        )
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            refuse("noncanonical_runtime_evidence", path, "%s requires non-empty string keys" % path)
        return {key: _plain(item, "%s.%s" % (path, key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, "%s[]" % path) for item in value]
    refuse("opaque_runtime_evidence", path, "%s contains opaque %s" % (path, type(value).__name__))


def _exact(value: Any, required: set[str], optional: set[str], path: str) -> dict[str, Any]:
    row = _plain(value, path)
    if not isinstance(row, dict):
        refuse("runtime_row_not_mapping", path, "%s must be a mapping" % path, evidence=row)
    missing, unknown = sorted(required - set(row)), sorted(set(row) - required - optional)
    if missing or unknown:
        refuse(
            "runtime_row_fields_mismatch",
            path,
            "%s fields mismatch: missing=%s, unknown=%s" % (path, missing, unknown),
            evidence={"missing": missing, "unknown": unknown},
        )
    return row


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        refuse(
            "invalid_runtime_text",
            path,
            "%s must be non-empty canonical text" % path,
            evidence=value,
        )
    return value


def _positive(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        refuse("invalid_runtime_integer", path, "%s must be an integer >= 1" % path, evidence=value)
    return value


def _block_layouts(plan: InstallPlan) -> tuple[LayoutPlan, Mapping[str, tuple[str, str]]]:
    layout_plan = plan.artifact.plan.layout_plan
    if type(layout_plan) is not LayoutPlan:
        raise TypeError("runtime planning requires an exact LayoutPlan from InstallPlan")
    blocks: dict[str, tuple[str, str]] = {}
    for assignment in layout_plan.assignments:
        if assignment.subject_kind != "block":
            continue
        name = assignment.subject.local_id
        if name in blocks:
            refuse(
                "duplicate_block_layout",
                "layout_plan.assignments",
                "block %r has multiple layout assignments" % name,
            )
        blocks[name] = (assignment.subject_id, assignment.layout.qualified_id)
    expected = tuple(block.name for block in plan.artifact.blocks)
    missing, extra = sorted(set(expected) - set(blocks)), sorted(set(blocks) - set(expected))
    if missing or extra:
        refuse(
            "block_layout_set_mismatch",
            "layout_plan.assignments",
            "layout plan must assign exactly the compiled blocks",
            evidence={"missing": missing, "extra": extra},
        )
    return layout_plan, blocks


def _memory_space(row: Mapping[str, Any], spaces: tuple[str, ...], path: str) -> str:
    value = row.get("memory_space")
    if value is None:
        if len(spaces) != 1:
            refuse(
                "ambiguous_memory_space",
                path,
                "multi-space platforms require an explicit memory_space",
                evidence=spaces,
            )
        value = spaces[0]
    value = _text(value, "%s.memory_space" % path)
    if value not in spaces:
        refuse(
            "unsupported_memory_space",
            "%s.memory_space" % path,
            "access names a memory space absent from the platform proof",
            evidence={"requested": value, "proved": spaces},
        )
    return value


def _accesses(values: tuple[Any, ...], mode: str, spaces: tuple[str, ...], path: str) -> tuple[FieldAccess, ...]:
    result = []
    for index, value in enumerate(values):
        item_path = "%s[%d]" % (path, index)
        row = _exact(value, {"resource"}, {"memory_space"}, item_path)
        result.append(
            FieldAccess(
                _text(row["resource"], "%s.resource" % item_path),
                mode,
                _memory_space(row, spaces, item_path),
            )
        )
    keys = [(row.resource, row.memory_space) for row in result]
    if len(keys) != len(set(keys)):
        refuse("duplicate_runtime_access", path, "%s contains duplicate accesses" % path)
    return tuple(result)


def _entry_point(manifest: ComponentManifest) -> str:
    if len(manifest.entry_points) != 1:
        refuse(
            "ambiguous_component_entry_point",
            "component[%s].entry_points" % manifest.component_id,
            "runtime planning requires exactly one authenticated native entry point",
            evidence=dict(manifest.entry_points),
        )
    return next(iter(manifest.entry_points.values()))


def _requirement_rows(call: RuntimeCall, spaces: tuple[str, ...], communicator: str, sequence: int) -> tuple[list[HaloExchange], list[Collective], list[tuple[int, str, str, int]], int]:
    reads = {row.resource for row in call.reads}
    resources = reads | {row.resource for row in call.writes}
    halo_depths: dict[str, int] = {}
    collectives: list[Collective] = []
    buffers: list[tuple[int, str, str, int]] = []
    for index, value in enumerate(call.requirements):
        path = "call[%d].requirements[%d]" % (call.ordinal, index)
        base = _exact(
            value,
            {"capability"},
            {"resource", "depth", "operation", "strategy", "bytes", "memory_space"},
            path,
        )
        capability = _text(base["capability"], "%s.capability" % path)
        if capability not in _RUNTIME_REQUIREMENTS:
            refuse(
                "unsupported_runtime_requirement",
                "%s.capability" % path,
                "runtime planner cannot prove capability %r" % capability,
                evidence=base,
            )
        if capability == "halo":
            row = _exact(value, {"capability", "depth"}, {"resource"}, path)
            resource = row.get("resource")
            if resource is None:
                if len(reads) != 1:
                    refuse(
                        "ambiguous_halo_resource",
                        path,
                        "a halo requirement without resource requires exactly one read",
                        evidence=sorted(reads),
                    )
                resource = next(iter(reads))
            resource = _text(resource, "%s.resource" % path)
            if resource not in reads:
                refuse(
                    "halo_without_declared_read",
                    path,
                    "halo requirement does not name a declared read",
                    evidence=resource,
                )
            halo_depths[resource] = max(halo_depths.get(resource, 0), _positive(row["depth"], "%s.depth" % path))
        elif capability == "collective":
            row = _exact(value, {"capability", "resource", "operation", "strategy"}, set(), path)
            resource = _text(row["resource"], "%s.resource" % path)
            if resource not in resources:
                refuse(
                    "collective_without_declared_access",
                    path,
                    "collective resource has no declared read or write",
                    evidence=resource,
                )
            collectives.append(
                Collective(
                    call.identity.token,
                    resource,
                    _text(row["operation"], "%s.operation" % path),
                    _text(row["strategy"], "%s.strategy" % path),
                    communicator,
                    sequence,
                )
            )
            sequence += 1
        else:
            row = _exact(value, {"capability", "resource", "bytes"}, {"memory_space"}, path)
            memory = _memory_space(row, spaces, path)
            buffers.append(
                (
                    call.ordinal,
                    _text(row["resource"], "%s.resource" % path),
                    memory,
                    _positive(row["bytes"], "%s.bytes" % path),
                )
            )
    halos = [HaloExchange(call.identity.token, resource, call.layout_id, depth) for resource, depth in sorted(halo_depths.items())]
    return halos, collectives, buffers, sequence


def _clock_rows(call: RuntimeCall) -> tuple[set[str], list[ClockJoin]]:
    clocks: set[str] = set()
    joins: list[ClockJoin] = []
    for index, value in enumerate(call.clocks):
        path = "call[%d].clocks[%d]" % (call.ordinal, index)
        initial = _exact(value, {"clock", "access"}, {"target", "policy"}, path)
        clock = _text(initial["clock"], "%s.clock" % path)
        access = _text(initial["access"], "%s.access" % path)
        clocks.add(clock)
        if access == "join":
            row = _exact(value, {"clock", "access", "target", "policy"}, set(), path)
            target = _text(row["target"], "%s.target" % path)
            clocks.add(target)
            joins.append(ClockJoin(call.identity.token, clock, target, _text(row["policy"], "%s.policy" % path)))
        elif access not in {"stage", "observe", "advance", "read"}:
            refuse(
                "unsupported_clock_access",
                "%s.access" % path,
                "unsupported clock access %r" % access,
            )
        elif set(initial) != {"clock", "access"}:
            refuse(
                "clock_join_fields_without_join",
                path,
                "target/policy are valid only for access='join'",
                evidence=initial,
            )
    return clocks, joins


def _require_connected_clocks(clocks: set[str], joins: tuple[ClockJoin, ...]) -> None:
    if len(clocks) < 2:
        return
    graph = {clock: set() for clock in clocks}
    for row in joins:
        graph[row.source_clock].add(row.target_clock)
        graph[row.target_clock].add(row.source_clock)
    reached, pending = set(), [min(clocks)]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(graph[current] - reached)
    if reached != clocks:
        refuse(
            "missing_clock_join",
            "component_manifests.clocks",
            "distinct runtime clocks require an explicit connected join graph",
            evidence={"clocks": sorted(clocks), "joined": sorted(reached)},
        )


def _resource_uses_and_fences(
    calls: tuple[RuntimeCall, ...],
) -> tuple[tuple[ResourceUse, ...], tuple[Fence, ...]]:
    uses: dict[tuple[str, str], dict[str, Any]] = {}
    previous: dict[str, tuple[RuntimeCall, str, str]] = {}
    fences: list[Fence] = []
    for call in calls:
        grouped: dict[str, list[FieldAccess]] = {}
        for access in call.reads + call.writes:
            grouped.setdefault(access.resource, []).append(access)
            key = (access.resource, access.memory_space)
            row = uses.setdefault(key, {"first": call.ordinal, "last": call.ordinal, "modes": set()})
            row["last"] = call.ordinal
            row["modes"].add(access.mode)
        for resource in sorted(grouped):
            accesses = grouped[resource]
            spaces = {row.memory_space for row in accesses}
            modes = {row.mode for row in accesses}
            if len(spaces) != 1:
                refuse(
                    "intra_call_cross_space_access",
                    "call[%d]" % call.ordinal,
                    "one opaque RuntimeCall cannot hide a cross-space transition",
                    evidence={"resource": resource, "spaces": sorted(spaces)},
                )
            space = next(iter(spaces))
            mode = "write" if "write" in modes else "read"
            prior = previous.get(resource)
            if prior is not None and prior[1] != space and (prior[2] == "write" or mode == "write"):
                fences.append(Fence(resource, prior[0].identity.token, call.identity.token, prior[1], space))
            previous[resource] = (call, space, mode)
    rows = tuple(ResourceUse(resource, space, item["first"], item["last"], tuple(sorted(item["modes"]))) for (resource, space), item in sorted(uses.items()))
    return rows, tuple(fences)


def _allocations(specs: list[tuple[int, str, str, int]], uses: tuple[ResourceUse, ...]) -> tuple[BufferAllocation, ...]:
    lifetimes = {(row.resource, row.memory_space): (row.first_call, row.last_call) for row in uses}
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for ordinal, resource, space, size in specs:
        first, last = lifetimes.get((resource, space), (ordinal, ordinal))
        row = grouped.setdefault((resource, space), {"size": size, "first": first, "last": last})
        row["size"] = max(row["size"], size)
        row["first"] = min(row["first"], first)
        row["last"] = max(row["last"], last)
    return tuple(BufferAllocation(resource, space, row["size"], row["first"], row["last"]) for (resource, space), row in sorted(grouped.items()))


def _assumption(name: str, context: ExecutionContext, communication: CommunicationPlan) -> Any:
    base = {
        "device": context.device.identity,
        "communicator": context.communicator.identity,
        "reduction_order": [row.identity.token for row in communication.collectives],
        "reduction_strategy": ["%s:%s" % (row.operation, row.strategy) for row in communication.collectives],
    }
    if name in base:
        return base[name]
    if name == "rank_count" and context.communicator.identity == "serial":
        return 1
    proof = context.backend.capabilities.get(name)
    if proof is None or not proof.known:
        refuse(
            "unknown_determinism_assumption",
            "determinism.scope[%s]" % name,
            "determinism scope lacks explicit runtime proof",
            evidence=name,
        )
    return _plain(proof.require("runtime.capabilities.%s" % name), "runtime.capabilities.%s" % name)


def _determinism(
    calls: tuple[RuntimeCall, ...],
    manifests: Mapping[str, ComponentManifest],
    context: ExecutionContext,
    communication: CommunicationPlan,
) -> DeterminismGuarantee:
    classifications = [manifest.determinism["classification"] for manifest in manifests.values()]
    if "unspecified" in classifications:
        refuse(
            "unspecified_component_determinism",
            "component_manifests.determinism",
            "every runtime component must declare a determinism classification",
        )
    classification = max(classifications, key=lambda item: _DETERMINISM_ORDER[item])
    scope = {item for manifest in manifests.values() for item in manifest.determinism["scope"]}
    required = set(scope)
    if classification == "bitwise":
        required.update(("rank_count", "device", "reduction_order", "reduction_strategy"))
    assumptions = {name: _assumption(name, context, communication) for name in sorted(required)}
    evidence = {
        call.block_id: {
            "component_id": call.component_id,
            "manifest_identity": call.component_manifest_identity.to_data(),
            "classification": manifests[next(name for name, manifest in manifests.items() if manifest.component_id == call.component_id)].determinism["classification"],
            "scope": list(manifests[next(name for name, manifest in manifests.items() if manifest.component_id == call.component_id)].determinism["scope"]),
        }
        for call in calls
    }
    return DeterminismGuarantee(classification, tuple(sorted(scope)), assumptions, evidence, context.identity)


def build_runtime_plans(install_plan: Any, component_manifests: Any) -> RuntimePlanBundle:
    """Derive exact calls, communication, resources and determinism from one InstallPlan."""
    plan = require_install_plan(install_plan)
    manifests = _component_map(plan, component_manifests)
    platform, context, spaces, facts = _proved_platform(plan)
    features = _features(platform, manifests)
    layout_plan, block_layouts = _block_layouts(plan)
    calls: list[RuntimeCall] = []
    halos: list[HaloExchange] = []
    collectives: list[Collective] = []
    buffers: list[tuple[int, str, str, int]] = []
    clock_joins: list[ClockJoin] = []
    clocks: set[str] = set()
    declared: list[Any] = []
    sequence = 0
    for ordinal, block in enumerate(plan.artifact.blocks):
        manifest = manifests[block.name]
        _validate_component_platform(manifest, facts, features)
        block_id, layout_id = block_layouts[block.name]
        call = RuntimeCall(
            ordinal,
            block_id,
            manifest.component_id,
            manifest.component_type,
            manifest.semantic_digest,
            layout_id,
            _entry_point(manifest),
            _accesses(manifest.reads, "read", spaces, "component[%s].reads" % block.name),
            _accesses(manifest.writes, "write", spaces, "component[%s].writes" % block.name),
            tuple(_plain(manifest.requirements, "component.requirements")),
            tuple(_plain(manifest.effects, "component.effects")),
            tuple(_plain(manifest.clocks, "component.clocks")),
        )
        calls.append(call)
        new_halos, new_collectives, new_buffers, sequence = _requirement_rows(call, spaces, facts["communicator"], sequence)
        halos.extend(new_halos)
        collectives.extend(new_collectives)
        buffers.extend(new_buffers)
        call_clocks, joins = _clock_rows(call)
        clocks.update(call_clocks)
        clock_joins.extend(joins)
        declared.extend(
            {
                "block_id": block_id,
                "component_id": manifest.component_id,
                "requirement": _plain(row, "component.requirement"),
            }
            for row in manifest.requirements
        )
    call_rows = tuple(calls)
    join_rows = tuple(clock_joins)
    _require_connected_clocks(clocks, join_rows)
    transfers = tuple(
        LayoutTransfer(
            row.requirement.qualified_id,
            row.provider_id,
            row.provider_identity["component_id"],
            row.requirement.source_layout.qualified_id,
            row.requirement.target_layout.qualified_id,
            row.requirement.source_port.subject.qualified_id,
            row.requirement.target_port.subject.qualified_id,
            row.requirement.source_port.representation.value,
            row.requirement.target_port.representation.value,
            int(row.requirement.operation),
            row.requirement.synchronization.value,
        )
        for row in layout_plan.mappings
    )
    uses, fences = _resource_uses_and_fences(call_rows)
    communication = CommunicationPlan(
        layout_plan.qualified_id,
        facts["communicator"],
        tuple(halos),
        transfers,
        tuple(collectives),
        fences,
        join_rows,
    )
    resources = ResourcePlan(
        layout_plan.qualified_id,
        context.identity,
        spaces,
        uses,
        _allocations(buffers, uses),
        tuple(sorted({row.provider_id for row in layout_plan.mappings})),
        tuple(row.identity.token for row in fences),
        tuple(declared),
    )
    determinism = _determinism(call_rows, manifests, context, communication)
    return RuntimePlanBundle(
        plan.bind_identity,
        platform.identity,
        context.identity,
        layout_plan.qualified_id,
        call_rows,
        communication,
        resources,
        determinism,
    )


__all__ = ["build_runtime_plans"]
