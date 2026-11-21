"""Read-only logical inventory of the qualified subset of named Uniform CP9."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import TYPE_CHECKING, Any

from pops.identity import Identity, canonical_bytes, make_identity

if TYPE_CHECKING:
    from pops.codegen._plans import InstallPlan
    from pops.runtime._bound_snapshot import BoundSnapshot
    from pops.runtime._runtime_instance import RuntimeInstance


class _ProjectionUnavailable(ValueError):
    """The installed or recorded authority lies outside this projection."""


@dataclass(frozen=True, slots=True)
class _BlockRole:
    position: int
    name: str
    state_name: str
    components: tuple[str, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BindProjection:
    owner: Identity
    program: Identity
    identity: Identity
    blocks: tuple[_BlockRole, ...]


@dataclass(frozen=True, slots=True)
class _Member:
    path: tuple[str | int, ...]
    kind: str
    physical_name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Inventory:
    binding: _BindProjection
    members: tuple[_Member, ...]


def _parameters(plan: InstallPlan, snapshot: BoundSnapshot, presentation: Any) -> list[Any]:
    from pops.problem._snapshot_presentation import project_parameter_expression
    from pops.runtime._bound_snapshot import _thaw

    rows = []
    for slot, effective in zip(plan.params.schema.slots, snapshot.params, strict=True):
        report = slot.to_dict()
        declaration = report["declaration"]
        project_parameter_expression(declaration, presentation)
        del declaration["provenance"]
        rows.append({
            "ordinal": slot.ordinal,
            "handle": presentation.project(report["handle"]),
            "declaration": declaration,
            "source": effective["source"],
            "value": _thaw(effective["value"]),
        })
    return rows


def _initial(snapshot: BoundSnapshot, presentation: Any) -> Any:
    from pops.runtime._bound_snapshot import _thaw

    evidence = _thaw(snapshot.initial_evidence)
    if evidence["legacy_state"]:
        raise _ProjectionUnavailable("initial_state has no resolved declaration authority")
    resolved = evidence["resolved_plan"]
    if resolved is None or not resolved["bindings"]:
        raise _ProjectionUnavailable("initial_values lack the resolved initial-condition plan")
    rows = []
    for row in resolved["bindings"].values():
        # The compiled Plan binds the source, projection and native route. Only the actual
        # bound value is new at installation. Keep its exact subject and array evidence here.
        rows.append((presentation.project(row["subject"]), row["bound_value"]))
    return sorted(rows, key=canonical_bytes)


def _auxiliary(plan: InstallPlan, snapshot: BoundSnapshot, positions: dict[str, int]) -> Any:
    from pops.codegen.provider_instances import require_instance_contract, runtime_key
    from pops.model.provider_pack import ComponentKey, ProviderPack
    from pops.runtime._bound_snapshot import _thaw

    if not plan.aux:
        return []
    routes = {}
    for block in plan.artifact.plan.blocks:
        operations = block.resolved_operations
        if operations is None:
            raise _ProjectionUnavailable("auxiliary_checkpoint lacks resolved provider operations")
        pack = ProviderPack.from_data(operations.to_data()["provider_evidence"]["auxiliary"])
        instance = require_instance_contract(operations, owner_qid=block.instance_owner_qid)
        for key in pack:
            native_key = runtime_key(key, instance)
            if native_key not in plan.aux:
                continue
            if native_key in routes:
                raise _ProjectionUnavailable("auxiliary_checkpoint has an ambiguous instance owner")
            routes[native_key] = (positions[block.name], key.space_kind,
                                  key.space_name, key.component)
    if set(routes) != set(plan.aux):
        raise _ProjectionUnavailable("auxiliary_checkpoint has an unknown component owner")
    return sorted(
        [(routes[ComponentKey(**row["key"])], row["array"])
         for row in _thaw(snapshot.aux_evidence)["components"]],
        key=canonical_bytes,
    )


def _parameter_storage(plan: InstallPlan, positions: dict[str, int], layout_id: str) -> Any:
    storage = plan.runtime_parameter_storage
    routes = {"model:" + name: ("model", index) for name, index in positions.items()}
    routes.update({"program:" + layout_id + ":" + str(index): ("program", index)
                   for index in positions.values()})
    counts, sizes = storage["cardinalities"], storage["carrier_bytes"]
    if set(counts) != set(sizes) or not set(counts) <= set(routes):
        raise _ProjectionUnavailable("runtime_parameter_storage has an unknown carrier route")
    return {
        "contract": storage["contract"],
        "scope": storage["scope"],
        "byte_capacity": storage["byte_capacity"],
        "initial_storage_bytes": storage["initial_storage_bytes"],
        "carriers": sorted((routes[key], counts[key], sizes[key]) for key in counts),
    }


def _describe(engine: Any, plan: InstallPlan, snapshot: BoundSnapshot) -> _BindProjection:
    from pops.mesh._static_geometry import layout_static_geometry
    from pops.runtime._checkpoint_embedded_boundary import current_checkpoint_embedded_boundary_contract
    from pops.runtime._checkpoint_finite_continuation import installed_capacity
    from pops.runtime._checkpoint_spatial import require_checkpoint_spatial_contract

    artifact = plan.artifact
    layouts = artifact.layout_plan.layouts
    if len(layouts) != 1 or layouts[0].adaptive or len(layouts[0].levels) != 1:
        raise _ProjectionUnavailable("projection requires a single Uniform layout")
    communicator = plan.execution_context.communicator
    if communicator.identity != "serial" or communicator.handle is not None:
        raise _ProjectionUnavailable("projection requires a serial checkpoint lane")
    layout, = layouts
    native_layout = layout.native_spatial_layout
    if native_layout is None or layout_static_geometry(layout) is not None:
        raise _ProjectionUnavailable("pops_spatial_contract lacks qualified static box geometry")
    spatial = require_checkpoint_spatial_contract(engine)
    if spatial.native_layout_identity != native_layout.identity.token:
        raise _ProjectionUnavailable("pops_spatial_contract differs from the installed Layout")
    embedded = current_checkpoint_embedded_boundary_contract(engine)
    if embedded.enabled:
        raise _ProjectionUnavailable("pops_embedded_boundary_contract is enabled")
    if plan.components:
        raise _ProjectionUnavailable("projection of installed external components is unavailable")
    unknown_resources = set(plan.resources) - {"execution_context", "runtime_parameter_byte_capacity"}
    if unknown_resources:
        raise _ProjectionUnavailable("bind resources lack a logical association: %s"
                                     % sorted(unknown_resources))
    if artifact.plan.consumer_graph is not None and artifact.plan.consumer_graph.nodes:
        raise _ProjectionUnavailable("runtime_consumer_graph is not empty")
    if artifact.program is None or artifact.program.program is None:
        raise _ProjectionUnavailable("projection lacks an emitted Program")
    if engine._s.installed_program_hash() != artifact.program_hash:
        raise _ProjectionUnavailable("program_hash differs from the emitted Program")
    program = artifact.program.program
    if program._integral_states or getattr(program, "_geometry_states", {}):
        raise _ProjectionUnavailable("program_exchange_state has unsupported declarations")
    if artifact.plan.field_plans or engine._s.field_provider_slots():
        raise _ProjectionUnavailable("field_provider_slots lacks a logical Field association")
    if program._histories:
        raise _ProjectionUnavailable("history_names lacks a logical ring association")
    if installed_capacity(engine)[0]:
        raise _ProjectionUnavailable("finite_continuation_schema has no logical association")

    presentation = artifact.plan.snapshot.presentation
    if presentation is None or len(presentation.layouts) != 1:
        raise _ProjectionUnavailable("projection lacks authenticated presentation roles")
    routes = artifact.program_block_routes
    names = tuple(engine._s.block_names())
    mapping = tuple(engine._s.program_block_map())
    authored = {block.name: block for block in artifact.plan.blocks}
    if not names or set(names) != set(authored) or len(set(names)) != len(names) \
            or len(mapping) != len(names) or set(mapping) != set(range(len(names))):
        raise _ProjectionUnavailable("blocks lack a complete bijection to native storage")
    ordered = tuple(names[index] for index in mapping)
    if routes != tuple(enumerate(ordered)) or ordered != presentation.blocks:
        raise _ProjectionUnavailable("blocks disagree with the emitted Program or Plan role order")
    blocks = []
    for position, name in enumerate(ordered):
        components = tuple(engine._s.variable_names(name, "conservative"))
        if not components or len(set(components)) != len(components) \
                or len(components) != engine._s.n_vars(name):
            raise _ProjectionUnavailable("names_%s lacks its exact component schema" % name)
        state_name, = authored[name].state_spaces
        blocks.append(_BlockRole(position, name, state_name, components,
                                 (len(components), *reversed(spatial.shape))))

    positions = {block.name: block.position for block in blocks}
    # program_id already covers the projected full Plan: resolved block models and
    # their variable units and policies, capabilities, initialization, continuation
    # transitions and provider contracts. Reuse that authority before adding bind data.
    identity = make_identity("restart-bind", {
        "schema_version": 1,
        "program_id": artifact.program_id.to_data(),
        "spatial": replace(native_layout, layout_id="$layout0").to_data(),
        "embedded_boundary": embedded.to_data(),
        "blocks": [(row.position, row.state_name, row.components, row.shape) for row in blocks],
        "parameters": _parameters(plan, snapshot, presentation),
        "initial_values": _initial(snapshot, presentation),
        "auxiliary": _auxiliary(plan, snapshot, positions),
        "execution_context": plan.execution_context.to_data(),
        "runtime_parameter_storage": _parameter_storage(plan, positions, layout.handle.qualified_id),
    })
    return _BindProjection(snapshot.bind_identity, artifact.program_id, identity, tuple(blocks))


def describe_uniform_checkpoint(engine: Any, plan: InstallPlan,
                                snapshot: BoundSnapshot) -> _BindProjection | str:
    """Capture the cold bind authority without changing admission to the named runtime."""
    from pops.identity.presentation import PresentationProjectionError

    try:
        return _describe(engine, plan, snapshot)
    except (_ProjectionUnavailable, PresentationProjectionError) as error:
        return str(error)


def inspect_uniform_checkpoint(runtime: RuntimeInstance, archive: bytes) -> _Inventory:
    """Authenticate named CP9 and describe its logical leaves without opening a transaction."""
    import numpy as np

    from pops.identity.checkpoint_origin import RUN_ORIGIN_CHECKPOINT_SCHEMA_VERSION
    from pops.output._checkpoint_contract import IDENTITY_KEY, MANIFEST_KEY
    from pops.runtime._checkpoint_program_diagnostics import PROGRAM_DIAGNOSTIC_CHECKPOINT_KEYS
    from pops.runtime._checkpoint_exchanges import CONTINUATION_CHECKPOINT_KEYS
    from pops.runtime._program_cadence_checkpoint import PROGRAM_CADENCE_CHECKPOINT_KEYS

    binding = getattr(runtime.bound_snapshot, "_uniform_checkpoint_projection", None)
    if not isinstance(binding, _BindProjection):
        raise _ProjectionUnavailable(binding or "named bind has no Uniform checkpoint projection")
    if binding.owner != runtime.bind_identity:
        raise _ProjectionUnavailable("logical projection belongs to a different named bind")
    prepared = runtime._executor._prepare_checkpoint_restart(archive, bit_identical=True)
    payload = prepared.payload

    manifest = json.loads(str(payload[MANIFEST_KEY]))
    if (manifest["schema_version"] != RUN_ORIGIN_CHECKPOINT_SCHEMA_VERSION
            or "checkpoint_migration" in payload):
        raise _ProjectionUnavailable("projection requires an original run-origin CP9 archive")
    if not prepared.state_carriers or not prepared.state_carriers.startswith(b"POPSCAR2"):
        raise _ProjectionUnavailable("state_carriers_checkpoint has no position authority")
    if not prepared.auxiliary_checkpoint.startswith((b"POPSAUX4", b"POPSAUX5")):
        raise _ProjectionUnavailable("auxiliary_checkpoint has no position authority")
    if prepared.exchange_checkpoint != b"POPSEX01" + bytes(8):
        raise _ProjectionUnavailable("program_exchange_state is not the exact empty mailbox")
    diagnostic = prepared.program_diagnostic_checkpoint
    if len(diagnostic) != 40 or diagnostic[32:] != bytes(8):
        raise _ProjectionUnavailable("program_diagnostics_state is not empty")
    for name in ("field_provider_slots", "history_names", "cache_nodes", "cache_names"):
        if payload[name].size:
            raise _ProjectionUnavailable("%s lacks a logical role" % name)
    temporal = prepared.temporal_state
    schedule = temporal.program_schedule
    if (schedule is None or len(schedule["clocks"]) != 1
            or any(schedule[name] for name in ("subcycles", "synchronizations", "schedules"))):
        raise _ProjectionUnavailable("temporal_restart_state lacks a qualified single-clock schedule")
    clock_owner = schedule["clocks"][0]["descriptor"]["owner"]
    if clock_owner is not None and any(
            row["kind"] != "consumer" for row in clock_owner["nodes"]):
        raise _ProjectionUnavailable("temporal_restart_state has a presentation-owned clock")
    if runtime._inspect_checkpoint_consumer_state(payload).rows:
        raise _ProjectionUnavailable("runtime_consumer_cursors is not empty")

    members = []

    def add(path: tuple[str | int, ...], kind: str, name: str) -> None:
        array = payload[name]
        members.append(_Member(path, kind, name, array.dtype.str, array.shape))

    for block in binding.blocks:
        prefix = ("state", block.position, block.state_name)
        ncomp, names, values = ("ncomp_" + block.name, "names_" + block.name,
                                "state_" + block.name)
        if int(payload[ncomp]) != len(block.components) \
                or tuple(payload[names].tolist()) != block.components \
                or payload[values].dtype != np.dtype(np.float64) \
                or payload[values].shape != block.shape:
            raise _ProjectionUnavailable("state role %d differs from its bound schema" % block.position)
        add((*prefix, "component_count"), "contract", ncomp)
        add((*prefix, "components"), "contract", names)
        add((*prefix, "values"), "continuation", values)

    groups = {
        "continuation": {"t", "macro_step", "temporal_restart_state",
                         *PROGRAM_CADENCE_CHECKPOINT_KEYS},
        "contract": {"pops_spatial_contract", "pops_embedded_boundary_contract",
                     "history_names", "cache_names", "cache_nodes", "field_provider_slots",
                     "runtime_consumer_graph", "runtime_consumer_cursors",
                     "runtime_consumer_diagnostics", *CONTINUATION_CHECKPOINT_KEYS,
                     *PROGRAM_DIAGNOSTIC_CHECKPOINT_KEYS},
        "codec": {"state_carriers_checkpoint", "auxiliary_checkpoint"},
        "alias": {"phi"},
        "provenance": {MANIFEST_KEY, IDENTITY_KEY, "abi_key", "pops_checkpoint_version",
                       "blocks", "program_hash"},
    }
    for kind, keys in groups.items():
        for name in sorted(keys.intersection(payload)):
            add((kind, name), kind, name)
    names = {row.physical_name for row in members}
    if len(names) != len(members) or len({row.path for row in members}) != len(members):
        raise _ProjectionUnavailable("checkpoint members do not form a logical bijection")
    unknown = set(payload) - names
    if unknown:
        raise _ProjectionUnavailable("checkpoint members lack a logical role: %s" % sorted(unknown))
    return _Inventory(binding, tuple(members))
