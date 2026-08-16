"""Private live resource authority for bounded accepted-state NPZ decoding."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
import sys
from typing import Any, cast

from pops._checkpoint_migration_protocol import (
    _CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS,
    _checkpoint_migration_provenance_characters,
)
from pops.identity import canonical_bytes
from pops.output._checkpoint_contract import (
    CheckpointResourceBudget,
    _capacity,
    require_checkpoint_resource_budget,
)


def _add(left: int, right: int, *, where: str) -> int:
    if left < 0 or right < 0 or right > sys.maxsize - left:
        raise OverflowError("%s exceeds the native addressable range" % where)
    return left + right


def _mul(left: int, right: int, *, where: str) -> int:
    if left < 0 or right < 0 or (left and right > sys.maxsize // left):
        raise OverflowError("%s exceeds the native addressable range" % where)
    return left * right


def _sum(values: Any, *, where: str) -> int:
    result = 0
    for value in values:
        result = _add(result, _capacity(value, where=where), where=where)
    return result


def _product(values: Any, *, where: str) -> int:
    result = 1
    for value in values:
        result = _mul(result, _capacity(value, where=where, positive=True), where=where)
    return result


def _require_iterable(value: object, *, where: str) -> Iterable[object]:
    if not isinstance(value, Iterable):
        raise TypeError("%s must be iterable" % where)
    return value


def _archive_byte_capacity(
    uncompressed_bytes: int, member_names: tuple[str, ...], *, where: str
) -> int:
    """Bound the ZIP64/DEFLATE carrier without consulting archive-authored sizes."""
    if (
        type(member_names) is not tuple
        or len(member_names) != len(set(member_names))
        or any(not isinstance(name, str) or not name for name in member_names)
    ):
        raise TypeError("%s requires exact unique member names" % where)
    members = _capacity(len(member_names), where=where + " member count", positive=True)
    payload = _capacity(uncompressed_bytes, where=where + " uncompressed bytes", positive=True)
    # The reviewed bound is U + (U >> 12) + (U >> 14) + (U >> 25)
    # + 145*M + 2*N + 98. Every term crosses the native addressable range through the checked
    # helpers; neither an archive header nor a central-directory size participates in this budget.
    result = payload
    for shift in (12, 14, 25):
        result = _add(result, payload >> shift, where=where + " DEFLATE bound")
    result = _add(
        result,
        _mul(members, 145, where=where + " ZIP64 member overhead"),
        where=where,
    )
    name_characters = _sum(
        (len(name.encode("utf-8")) for name in member_names),
        where=where + " member names",
    )
    result = _add(
        result,
        _mul(name_characters, 2, where=where + " ZIP member names"),
        where=where,
    )
    return _add(result, 98, where=where + " ZIP64 directory")


def _program_for_install(install_plan: Any) -> tuple[Any, dict[str, int]]:
    artifact = install_plan.artifact
    program_handle = artifact.program
    if program_handle is None or getattr(program_handle, "program", None) is None:
        raise RuntimeError("checkpoint budget requires the artifact's exact compiled Program")
    program = program_handle.program
    instances = dict(artifact.arguments().instances)
    block_nvars = {}
    for block in artifact.blocks:
        row = instances.get(block.name)
        if not isinstance(row, dict):
            raise ValueError("checkpoint budget lacks exact block instance metadata")
        block_nvars[block.name] = _capacity(
            int(row.get("components", 0)),
            where="block %r component count" % block.name,
            positive=True,
        )
    return program, block_nvars


def _history_capacity(
    program: Any,
    *,
    cells: tuple[int, ...],
    amr: bool,
    block_nvars: dict[str, int],
    native: Any = None,
) -> tuple[tuple[str, ...], int, tuple[tuple[str, int, int], ...]]:
    """Budget slots from the installed native ring registry when it is available.

    The Program declares logical history names and owners, while the native ring owns its actual
    storage depth (for example AB2's two retained slots).  The two are authenticated here, before
    a checkpoint archive exists; runtime fill state is deliberately not consulted.
    """
    histories = dict(getattr(program, "_histories", None) or {})
    ncomps = dict(getattr(program, "_histories_ncomp", None) or {})
    owners = dict(getattr(program, "_history_blocks", None) or {})
    declared_names = tuple(sorted(histories))
    native_names = None
    native_depth: Callable[[str], object] | None = None
    native_ncomp: Callable[[str], object] | None = None
    if native is not None:
        providers = (
            getattr(native, "history_names", None),
            getattr(native, "history_depth", None),
            getattr(native, "history_ncomp", None),
        )
        if all(callable(provider) for provider in providers):
            name_provider = cast(Callable[[], object], providers[0])
            raw_names = name_provider()
            if not isinstance(raw_names, (tuple, list)) or any(
                type(name) is not str or not name for name in raw_names
            ):
                raise TypeError("native checkpoint history registry has invalid names")
            native_names = tuple(sorted(raw_names))
            if len(native_names) != len(set(native_names)) or native_names != declared_names:
                raise ValueError("native checkpoint history names differ from the Program declaration")
            native_depth = cast(Callable[[str], object], providers[1])
            native_ncomp = cast(Callable[[str], object], providers[2])

    member_names = ["history_names"]
    data_bytes = 0
    evidence = []
    for name in declared_names:
        depth_value = histories[name]
        logical_depth = _capacity(
            int(depth_value), where="history %r logical depth" % name, positive=True
        )
        ncomp = ncomps.get(name)
        if ncomp is None:
            raw_owner = owners.get(name)
            owner_name = None
            if raw_owner is not None:
                from pops.time.references import block_name

                owner_name = block_name(raw_owner)
            ncomp = block_nvars.get(owner_name) if owner_name is not None else None
        if ncomp is None:
            raise ValueError("checkpoint history %r has no component authority" % name)
        ncomp = _capacity(int(ncomp), where="history %r components" % name, positive=True)
        if native_depth is not None and native_ncomp is not None:
            depth_value = native_depth(name)
            ncomp_value = native_ncomp(name)
            if type(depth_value) is not int or depth_value < 1:
                raise ValueError("native checkpoint history %r has an invalid storage depth" % name)
            if type(ncomp_value) is not int or ncomp_value < 1:
                raise ValueError("native checkpoint history %r has an invalid component width" % name)
            if ncomp_value != ncomp:
                raise ValueError("native checkpoint history %r component width differs from Program" % name)
            depth = depth_value
        else:
            depth = logical_depth
        evidence.append((name, logical_depth, depth))
        member_names.extend(
            (
                "history_depth_" + name,
                "history_ncomp_" + name,
                "history_policy_" + name,
                "history_requested_stored_slots_" + name,
                "history_stored_slots_" + name,
                "history_storage_mode_" + name,
                "history_regrid_steps_" + name,
            )
        )
        levels: tuple[int | None, ...] = tuple(range(len(cells))) if amr else (None,)
        if amr:
            member_names.append("history_levels_" + name)
        for level, level_cells in zip(levels, cells, strict=True):
            suffix = name if level is None else "%s_level_%d" % (name, level)
            member_names.extend(
                (
                    "history_init_" + suffix,
                    "history_fill_count_" + suffix,
                    "history_slot_dt_" + suffix,
                )
            )
            for slot in range(depth):
                member_names.append(
                    "history_%s_%d" % (name, slot)
                    if level is None
                    else "history_%s_level_%d_%d" % (name, level, slot)
                )
            values = _mul(level_cells, ncomp, where="history scalar budget")
            values = _mul(values, depth, where="history scalar budget")
            data_bytes = _add(
                data_bytes,
                _mul(values, 8, where="history byte budget"),
                where="history byte budget",
            )
    return tuple(member_names), data_bytes, tuple(evidence)


def _cache_capacity(
    owner: Any,
    shape: tuple[int, ...],
    *,
    program: Any,
    block_nvars_by_name: dict[str, int],
) -> tuple[tuple[str, ...], int, Any]:
    """Reserve every cacheable schedule node from the immutable Program authority.

    Native cache nodes are intentionally lazy: before the first accepted execution they need not
    exist at all.  Their mutable materialization state is therefore validation evidence, not the
    source of the resource envelope or its bind identity.
    """
    native = owner._s
    temporal = program.temporal_manifest()
    if not isinstance(temporal, Mapping):
        raise TypeError("checkpoint cache authority requires a temporal manifest mapping")
    schedules = temporal.get("schedules")
    if not isinstance(schedules, list):
        raise TypeError("checkpoint cache authority requires temporal schedule rows")
    declared_nodes = []
    potential_nodes = []
    for index, row in enumerate(schedules):
        if not isinstance(row, Mapping):
            raise TypeError("checkpoint cache schedule %d must be a mapping" % index)
        raw_node = row.get("node_id")
        if type(raw_node) is not int:
            raise TypeError("checkpoint cache schedule node id must be an exact integer")
        node = _capacity(
            raw_node,
            where="checkpoint cache schedule node id",
        )
        declared_nodes.append(node)
        cache_required = row.get("cache_required")
        if type(cache_required) is not bool:
            raise TypeError("checkpoint cache schedule cache_required must be bool")
        if cache_required:
            potential_nodes.append(node)
    potential = tuple(sorted(potential_nodes))
    if len(declared_nodes) != len(set(declared_nodes)):
        raise ValueError("checkpoint cache authority has duplicate schedule node ids")

    provider = getattr(native, "program_cache_nodes", None)
    raw_nodes = (
        _require_iterable(provider(), where="checkpoint cache native node evidence")
        if callable(provider)
        else ()
    )
    nodes = []
    for node in raw_nodes:
        if type(node) is not int or node < 0:
            raise ValueError("checkpoint cache authority has invalid exact node ids")
        nodes.append(node)
    nodes = tuple(nodes)
    if nodes != tuple(sorted(set(nodes))):
        raise ValueError("checkpoint cache authority has invalid exact node ids")
    unknown = tuple(node for node in nodes if node not in potential)
    if unknown:
        raise ValueError("checkpoint cache authority has live nodes outside the Program schedule")

    width = _sum(block_nvars_by_name.values(), where="checkpoint cache scalar width")
    valid_cells = _product(shape, where="checkpoint cache valid cells")
    names = ["cache_nodes", "cache_names"]
    data_bytes = 0
    evidence = []
    for node in potential:
        names.extend(
            (
                "cache_ncomp_%d" % node,
                "cache_ngrow_%d" % node,
                "cache_last_update_%d" % node,
                "cache_accum_dt_%d" % node,
                "cache_value_%d" % node,
            )
        )
        data_bytes = _add(
            data_bytes,
            _mul(
                _mul(valid_cells, width, where="checkpoint cache scalar budget"),
                8,
                where="checkpoint cache byte budget",
            ),
            where="checkpoint cache byte budget",
        )
        # This stable potential-node evidence deliberately does not depend on whether native
        # execution has materialized the lazy cache yet.
        evidence.append((node, "node_%d" % node, width))

    for node in nodes:
        name = native.program_cache_name(node)
        ncomp = _capacity(
            native.program_cache_ncomp(node),
            where="checkpoint cache component count",
            positive=True,
        )
        _capacity(native.program_cache_ngrow(node), where="checkpoint cache ghost width")
        if name != "node_%d" % node:
            raise ValueError("checkpoint cache authority has a noncanonical exact node name")
        if ncomp > width:
            raise ValueError("checkpoint cache component count exceeds the Program width")
    return tuple(names), data_bytes, evidence


def _consumer_evidence(install_plan: Any) -> tuple[str, int, Any]:
    graph = install_plan.artifact.plan.consumer_graph
    if graph is None:
        return "none", 0, {"nodes": []}
    identity = getattr(getattr(graph, "identity", None), "token", None)
    to_data = getattr(graph, "to_data", None)
    nodes = getattr(graph, "nodes", None)
    if (
        not isinstance(identity, str)
        or not identity
        or not callable(to_data)
        or not isinstance(nodes, tuple)
    ):
        raise TypeError("checkpoint budget requires the exact sealed ConsumerGraph")
    data = to_data()
    # Consumer evidence permits opaque byte strings.  Project the complete canonical CBOR payload
    # into a tagged JSON-safe envelope for this budget identity rather than decoding binary data or
    # weakening the sealed ConsumerGraph contract.
    evidence = {
        "encoding": "pops-canonical-cbor-hex-v1",
        "payload": canonical_bytes(data).hex(),
    }
    json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return identity, len(nodes), evidence


def _amr_field_provider_manifest_capacity(
    owner: Any, *, configured_levels: int
) -> tuple[tuple[str, ...], int, int]:
    """Bound the v11 field-provider manifest from its live immutable native rows."""
    provider = getattr(owner._s, "field_provider_checkpoint_manifest", None)
    if not callable(provider):
        raise TypeError("AMR checkpoint budget requires the native field-provider manifest")
    raw_rows = _require_iterable(
        provider(), where="AMR checkpoint field-provider manifest"
    )
    rows = []
    for raw_row in raw_rows:
        typed_row = []
        for value in _require_iterable(
            raw_row, where="AMR checkpoint field-provider manifest row"
        ):
            if not isinstance(value, str) or not value:
                raise TypeError("AMR checkpoint field-provider manifest has an invalid row")
            typed_row.append(value)
        rows.append(tuple(typed_row))
    rows = tuple(rows)

    maximum_uint64_text = str((1 << 64) - 1)
    bounded_rows = []
    slots = []
    for row in rows:
        if len(row) < 14:
            raise TypeError("AMR checkpoint field-provider manifest has an invalid row")
        if row[0] != "pops.amr.field-provider-checkpoint-manifest@1":
            raise ValueError("AMR checkpoint field-provider manifest has an unknown schema")
        if row[1] in slots:
            raise ValueError("AMR checkpoint field-provider manifest has a duplicate slot")

        def decimal(value: str, *, where: str) -> int:
            if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
                raise ValueError("%s must be canonical unsigned decimal text" % where)
            return int(value)

        depth = decimal(row[2], where="AMR field-provider depth")
        topology_epoch = decimal(row[6], where="AMR field-provider topology epoch")
        materialization_generation = decimal(
            row[7], where="AMR field-provider materialization generation"
        )
        if depth < 1 or depth > configured_levels:
            raise ValueError("AMR field-provider depth exceeds its configured hierarchy")
        if topology_epoch > (1 << 64) - 1 or materialization_generation > (1 << 64) - 1:
            raise ValueError("AMR field-provider generation exceeds its native uint64 domain")
        if row[8] not in {"materialized", "unmaterialized"}:
            raise ValueError("AMR field-provider manifest has an invalid materialization state")

        dependency_count = decimal(row[12], where="AMR field-provider dependency count")
        boundary_count_index = 13 + 3 * dependency_count
        if boundary_count_index >= len(row):
            raise ValueError("AMR field-provider manifest truncates its dependency rows")
        boundary_count = decimal(
            row[boundary_count_index], where="AMR field-provider boundary dependency count"
        )
        if len(row) != boundary_count_index + 1 + 2 * boundary_count:
            raise ValueError("AMR field-provider manifest has an invalid structural width")

        bounded = list(row)
        bounded[2] = str(configured_levels)
        bounded[6] = maximum_uint64_text
        bounded[7] = maximum_uint64_text
        bounded[8] = "unmaterialized"
        bounded_rows.append(tuple(bounded))
        slots.append(row[1])

    text = json.dumps(tuple(bounded_rows), separators=(",", ":"), ensure_ascii=True)
    characters = len(text)
    import numpy as np

    structural_bytes = _mul(
        characters,
        int(np.dtype("U1").itemsize),
        where="AMR field-provider manifest byte capacity",
    )
    return tuple(slots), characters, structural_bytes


def _checkpoint_member_names(
    *,
    runtime_kind: str,
    block_names: tuple[str, ...],
    field_names: tuple[str, ...],
    history_names: tuple[str, ...],
    cache_names: tuple[str, ...],
    levels: int,
    rank_capacity: int,
    has_amr_legacy_phi: bool = False,
) -> tuple[str, ...]:
    from pops.runtime._checkpoint_embedded_boundary import EMBEDDED_BOUNDARY_CONTRACT_KEY
    from pops.output._checkpoint_contract import IDENTITY_KEY, MANIFEST_KEY
    from pops.runtime._checkpoint_spatial import SPATIAL_CONTRACT_KEY

    cadence = (
        "program_cadence_substeps",
        "program_cadence_stride",
        "program_cadence_window_steps",
        "program_cadence_window_dt",
        "program_cadence_window_start_time",
        "program_last_dt",
    )
    runtime = (
        "runtime_consumer_graph",
        "runtime_consumer_cursors",
        "runtime_consumer_diagnostics",
    )
    if runtime_kind == "uniform":
        names = [
            "pops_checkpoint_version",
            "t",
            "macro_step",
            "abi_key",
            "blocks",
            "temporal_restart_state",
            SPATIAL_CONTRACT_KEY,
            EMBEDDED_BOUNDARY_CONTRACT_KEY,
            *cadence,
            "field_provider_slots",
            "program_hash",
            "phi",
            "auxiliary_checkpoint",
            "checkpoint_migration",
            *cache_names,
            *history_names,
        ]
        for block in block_names:
            names.extend(("ncomp_" + block, "names_" + block, "state_" + block))
        for index in range(len(field_names)):
            names.append("field_potential_%d" % index)
    else:
        names = [
            "pops_amr_checkpoint_version",
            "t",
            "macro_step",
            "regrid_every",
            "abi_key",
            "blocks",
            "n_levels",
            "configured_n_levels",
            "n_ranks",
            "patch_boxes",
            "temporal_restart_state",
            "regrid_count",
            "topology_epoch",
            "amr_accepted_contract",
            "program_hash",
            "field_provider_slots",
            "field_provider_manifest",
            SPATIAL_CONTRACT_KEY,
            *cadence,
            "program_accepted_state_source_authority",
            *history_names,
        ]
        for block in block_names:
            names.append("n_vars_" + block)
            for level in range(levels):
                names.append("state_%s_%d" % (block, level))
        for level in range(levels):
            if has_amr_legacy_phi:
                names.append("phi_%d" % level)
            names.append("auxiliary_checkpoint_%d" % level)
        for index in range(len(field_names)):
            names.append("field_provider_levels_%d" % index)
            for level in range(levels):
                names.append("field_provider_phi_%d_%d" % (index, level))
        names.append("program_accepted_state")
        for level in range(levels):
            names.extend(("distribution_mode_%d" % level, "dmap_%d" % level))
    names.extend((*runtime, MANIFEST_KEY, IDENTITY_KEY))
    if not names or len(names) != len(set(names)):
        raise RuntimeError("checkpoint resource authority produced duplicate member names")
    return tuple(names)


def _common_budget(
    owner: Any,
    install_plan: Any,
    *,
    runtime_kind: str,
    cells: tuple[int, ...],
    shape: tuple[int, ...],
    rank_capacity: int,
    auxiliary_metadata_bytes: int,
    auxiliary_components: int,
    accepted_program_bytes: int,
    source_authority_bytes: int,
    structural_bytes: int,
    field_provider_manifest_characters: int,
    program: Any,
    block_nvars_by_name: dict[str, int],
    field_names: tuple[str, ...],
) -> CheckpointResourceBudget:
    from pops.identity import make_identity
    from pops.output._checkpoint_collective import _NPY_HEADER_BUDGET, _manifest_character_budget

    if runtime_kind not in {"uniform", "amr"}:
        raise ValueError("checkpoint common budget requires Uniform or AMR")
    rank_capacity = _capacity(rank_capacity, where="checkpoint rank capacity", positive=True)
    block_names = tuple(block_nvars_by_name)
    block_nvars = tuple(block_nvars_by_name[name] for name in block_names)
    total_cells = _sum(cells, where="checkpoint configured cell capacity")
    state_scalars = _mul(
        total_cells,
        _sum(block_nvars, where="block scalar width"),
        where="checkpoint state scalar budget",
    )
    field_scalars = _mul(total_cells, len(field_names), where="checkpoint field scalar budget")
    scientific_bytes = _mul(
        _add(
            _add(state_scalars, total_cells, where="checkpoint scalar budget"),
            field_scalars,
            where="checkpoint scalar budget",
        ),
        8,
        where="checkpoint scientific byte budget",
    )
    history_names, history_bytes, history_evidence = _history_capacity(
        program,
        cells=cells,
        amr=runtime_kind == "amr",
        block_nvars=block_nvars_by_name,
        native=owner._s,
    )
    cache_names, cache_bytes, cache_evidence = (
        _cache_capacity(
            owner,
            shape,
            program=program,
            block_nvars_by_name=block_nvars_by_name,
        )
        if runtime_kind == "uniform"
        else ((), 0, ())
    )
    auxiliary_bytes = _mul(auxiliary_components, 8, where="auxiliary scalar width")
    auxiliary_bytes = _mul(
        total_cells, auxiliary_bytes, where="auxiliary checkpoint payload budget"
    )
    auxiliary_bytes = _add(
        auxiliary_bytes,
        _mul(len(cells), auxiliary_metadata_bytes, where="auxiliary metadata budget"),
        where="auxiliary checkpoint payload budget",
    )
    program_bytes = _mul(
        accepted_program_bytes,
        1,
        where="accepted Program checkpoint byte budget",
    )
    migration_bytes = 0
    if runtime_kind == "uniform":
        import numpy as np

        migration_bytes = _mul(
            _CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS,
            int(np.dtype("U1").itemsize),
            where="checkpoint migration provenance byte capacity",
        )
    payload_bytes = 0
    for addition in (
        scientific_bytes,
        history_bytes,
        cache_bytes,
        auxiliary_bytes,
        program_bytes,
        source_authority_bytes,
        structural_bytes,
        migration_bytes,
    ):
        payload_bytes = _add(payload_bytes, addition, where="checkpoint payload byte budget")

    names = _checkpoint_member_names(
        runtime_kind=runtime_kind,
        block_names=block_names,
        field_names=field_names,
        history_names=history_names,
        cache_names=cache_names,
        levels=len(cells),
        rank_capacity=rank_capacity,
        has_amr_legacy_phi=runtime_kind == "amr" and bool(field_names),
    )
    consumer_identity, consumer_count, consumer_data = _consumer_evidence(install_plan)
    temporal_manifest = program.temporal_manifest()
    control_data = {
        "artifact": install_plan.artifact.artifact_identity.token,
        "bind": install_plan.bind_identity.token,
        "runtime_kind": runtime_kind,
        "blocks": list(zip(block_names, block_nvars, strict=True)),
        "block_variables": {
            name: list(owner._s.variable_names(name, "conservative")) for name in block_names
        },
        "fields": list(field_names),
        "histories": sorted(getattr(program, "_histories", {})),
        "history_storage": [list(row) for row in history_evidence],
        "cache": cache_evidence,
        "temporal": temporal_manifest,
        "consumer_graph": consumer_data,
        "consumer_identity": consumer_identity,
        "consumer_count": consumer_count,
        "cells": list(cells),
        "rank_capacity": rank_capacity,
    }
    control_characters = len(
        json.dumps(control_data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    max_manifest_characters = _manifest_character_budget(names)
    # Every variable-sized text value is authored by one of the immutable controls above. Reserve
    # its complete canonical spelling for the temporal image, consumer cursors/diagnostics and
    # native metadata, in addition to the independently derived exact manifest schema budget.
    text_characters = _add(
        max_manifest_characters,
        _mul(control_characters, 8, where="checkpoint control text budget"),
        where="checkpoint text budget",
    )
    text_characters = _add(
        text_characters,
        _mul(
            _sum((len(name) for name in names), where="checkpoint member-name budget"),
            4,
            where="checkpoint member-name budget",
        ),
        where="checkpoint text budget",
    )
    text_bytes = _mul(text_characters, 4, where="checkpoint text byte budget")
    uncompressed = _add(payload_bytes, text_bytes, where="checkpoint aggregate byte budget")
    uncompressed = _add(
        uncompressed,
        _mul(len(names), _NPY_HEADER_BUDGET, where="checkpoint header budget"),
        where="checkpoint aggregate byte budget",
    )
    authority = make_identity(
        "checkpoint-resource-budget",
        {
            **control_data,
            "auxiliary": [auxiliary_metadata_bytes, auxiliary_components],
            "accepted_program_bytes": accepted_program_bytes,
            "source_authority_bytes": source_authority_bytes,
            "structural_bytes": structural_bytes,
            "field_provider_manifest_characters": field_provider_manifest_characters,
            "members": list(names),
            "manifest_characters": max_manifest_characters,
            "uncompressed_bytes": uncompressed,
            "archive_bytes": _archive_byte_capacity(
                uncompressed, names, where="checkpoint archive byte budget"
            ),
        },
    ).token
    archive_bytes = _archive_byte_capacity(
        uncompressed, names, where="checkpoint archive byte budget"
    )
    return CheckpointResourceBudget(
        runtime_kind,
        len(names),
        max_manifest_characters,
        max(
            payload_bytes,
            text_bytes,
            accepted_program_bytes,
            source_authority_bytes,
            migration_bytes,
            1,
        ),
        uncompressed,
        archive_bytes,
        authority,
    )


def install_uniform_checkpoint_resource_budget(owner: Any, install_plan: Any) -> None:
    from pops.runtime._checkpoint_spatial import require_checkpoint_spatial_contract

    spatial = require_checkpoint_spatial_contract(owner)
    capacity = owner._s._checkpoint_auxiliary_capacity()
    if not isinstance(capacity, tuple) or len(capacity) != 2:
        raise TypeError("Uniform native auxiliary checkpoint capacity has an invalid schema")
    program, block_nvars = _program_for_install(install_plan)
    artifact = install_plan.artifact
    candidate = _common_budget(
        owner,
        install_plan,
        runtime_kind="uniform",
        cells=(spatial.cells_at_level(0),),
        shape=spatial.shape,
        rank_capacity=1,
        auxiliary_metadata_bytes=_capacity(capacity[0], where="auxiliary metadata capacity"),
        auxiliary_components=_capacity(capacity[1], where="auxiliary component capacity"),
        accepted_program_bytes=0,
        source_authority_bytes=0,
        structural_bytes=0,
        field_provider_manifest_characters=0,
        program=program,
        block_nvars_by_name=block_nvars,
        field_names=tuple(sorted(artifact.plan.field_plans)),
    )
    existing = getattr(owner, "_checkpoint_resource_budget", None)
    if existing is not None and existing != candidate:
        raise RuntimeError("Uniform checkpoint resource authority changed during bind")
    owner._checkpoint_resource_budget = candidate


def install_amr_checkpoint_resource_budget(owner: Any, install_plan: Any) -> None:
    from pops.output._checkpoint_collective import checkpoint_topology
    from pops.runtime._checkpoint_spatial import require_checkpoint_spatial_contract

    spatial = require_checkpoint_spatial_contract(owner)
    configured_levels = _capacity(
        int(owner._s.configured_n_levels()), where="configured AMR checkpoint levels", positive=True
    )
    if configured_levels > 1 + len(spatial.refinement_ratios):
        raise ValueError("AMR checkpoint resource authority lacks configured refinement shapes")
    cells = tuple(spatial.cells_at_level(level) for level in range(configured_levels))
    capacity = owner._s._checkpoint_auxiliary_level_capacity()
    program_state = tuple(owner._s._checkpoint_program_state_capacity())
    if not isinstance(capacity, tuple) or len(capacity) != 2:
        raise TypeError("AMR native auxiliary checkpoint capacity has an invalid schema")
    if len(program_state) != 2:
        raise TypeError("AMR native Program checkpoint capacity has an invalid schema")
    topology = checkpoint_topology(owner)
    patch_capacity = _sum(cells, where="AMR patch capacity")
    rank_capacity = topology.size
    program, block_nvars = _program_for_install(install_plan)
    accepted_capacity = _capacity(
        program_state[0], where="native accepted Program image capacity", positive=True
    )
    source_authority_bytes = _capacity(
        program_state[1], where="native source Program authority capacity", positive=True
    )
    artifact = install_plan.artifact
    field_slots, field_manifest_characters, field_manifest_bytes = (
        _amr_field_provider_manifest_capacity(owner, configured_levels=configured_levels)
    )
    if len(field_slots) != len(artifact.plan.field_plans):
        raise ValueError("AMR checkpoint field-provider authority differs from its artifact plans")
    structural_bytes = _add(
        _add(
            _mul(
                _mul(
                    patch_capacity,
                    1 + 2 * spatial.dimension,
                    where="AMR patch-box scalar capacity",
                ),
                8,
                where="AMR patch-box byte capacity",
            ),
            _add(
                _mul(patch_capacity, 8, where="AMR owner-map byte capacity"),
                _add(
                    _mul(
                        patch_capacity if field_slots else 0,
                        8,
                        where="AMR legacy phi byte capacity",
                    ),
                    _mul(
                        configured_levels,
                        44,
                        where="AMR distribution-mode Unicode byte capacity",
                    ),
                    where="AMR structural checkpoint capacity",
                ),
                where="AMR structural checkpoint capacity",
            ),
            where="AMR structural checkpoint capacity",
        ),
        field_manifest_bytes,
        where="AMR structural checkpoint capacity",
    )
    candidate = _common_budget(
        owner,
        install_plan,
        runtime_kind="amr",
        cells=cells,
        shape=spatial.shape,
        rank_capacity=rank_capacity,
        auxiliary_metadata_bytes=_capacity(capacity[0], where="auxiliary metadata capacity"),
        auxiliary_components=_capacity(capacity[1], where="auxiliary component capacity"),
        accepted_program_bytes=accepted_capacity,
        source_authority_bytes=source_authority_bytes,
        structural_bytes=structural_bytes,
        field_provider_manifest_characters=field_manifest_characters,
        program=program,
        block_nvars_by_name=block_nvars,
        field_names=field_slots,
    )
    existing = getattr(owner, "_checkpoint_resource_budget", None)
    if existing is not None and existing != candidate:
        raise RuntimeError("AMR checkpoint resource authority changed during bind")
    owner._checkpoint_resource_budget = candidate


def install_layout_checkpoint_resource_budget(
    owner: Any,
    *,
    program: Any,
    block_names: tuple[str, ...],
    artifact_identity: Any,
    bind_identity: Any,
    install_plan: Any,
) -> None:
    """Install one child Uniform budget under the aggregate artifact/bind authorities."""
    from pops.runtime._checkpoint_spatial import require_checkpoint_spatial_contract

    if not block_names or len(block_names) != len(set(block_names)):
        raise ValueError("layout checkpoint budget requires unique child block names")
    spatial = require_checkpoint_spatial_contract(owner)
    capacity = owner._s._checkpoint_auxiliary_capacity()
    block_nvars = {
        name: _capacity(
            int(owner._s.n_vars(name)), where="layout block component count", positive=True
        )
        for name in block_names
    }
    candidate = _common_budget(
        owner,
        install_plan,
        runtime_kind="uniform",
        cells=(spatial.cells_at_level(0),),
        shape=spatial.shape,
        rank_capacity=1,
        auxiliary_metadata_bytes=_capacity(capacity[0], where="auxiliary metadata capacity"),
        auxiliary_components=_capacity(capacity[1], where="auxiliary component capacity"),
        accepted_program_bytes=0,
        source_authority_bytes=0,
        structural_bytes=0,
        field_provider_manifest_characters=0,
        program=program,
        block_nvars_by_name=block_nvars,
        field_names=(),
    )
    if artifact_identity.token != install_plan.artifact.artifact_identity.token or (
        bind_identity.token != install_plan.bind_identity.token
    ):
        raise ValueError("layout checkpoint resource identity differs from its install plan")
    existing = getattr(owner, "_checkpoint_resource_budget", None)
    if existing is not None and existing != candidate:
        raise RuntimeError("layout checkpoint resource authority changed during bind")
    owner._checkpoint_resource_budget = candidate


def aggregate_checkpoint_resource_budgets(
    budgets: Any,
    *,
    authority: str,
    install_plan: Any,
    layout_ids: tuple[str, ...],
    mapping_ids: tuple[str, ...],
) -> CheckpointResourceBudget:
    from pops.output._checkpoint_collective import _NPY_HEADER_BUDGET, _manifest_character_budget

    rows = tuple(budgets)
    if not rows or any(type(row) is not CheckpointResourceBudget for row in rows):
        raise TypeError("multi-layout checkpoint budget requires exact child budgets")
    if (
        type(layout_ids) is not tuple
        or len(layout_ids) != len(rows)
        or len(layout_ids) != len(set(layout_ids))
        or any(not isinstance(value, str) or not value for value in layout_ids)
        or type(mapping_ids) is not tuple
        or len(mapping_ids) != len(set(mapping_ids))
        or any(not isinstance(value, str) or not value for value in mapping_ids)
    ):
        raise TypeError("multi-layout checkpoint budget requires exact live layout/mapping ids")
    member_names = [
        "t",
        "macro_step",
        "abi_key",
        "layout_ids",
        "mapping_evaluations",
        "runtime_consumer_graph",
        "runtime_consumer_cursors",
        "runtime_consumer_diagnostics",
        "pops_checkpoint_manifest",
        "pops_restart_identity",
    ]
    maximum_array = 1
    # Scalar time/macro-step controls are two fixed eight-byte arrays. The remaining controls are
    # bounded by the live child manifest authorities below.
    total = 16
    for index, row in enumerate(rows):
        member_names.append("layout_checkpoint_%d" % index)
        archive_bytes = row.max_archive_bytes
        maximum_array = max(maximum_array, archive_bytes)
        total = _add(total, archive_bytes, where="multi-layout checkpoint byte budget")
    manifest = _manifest_character_budget(tuple(member_names))
    consumer_identity, consumer_count, consumer_data = _consumer_evidence(install_plan)
    control_characters = len(
        json.dumps(
            {
                "layouts": layout_ids,
                "mappings": mapping_ids,
                "consumer_graph": consumer_data,
                "consumer_identity": consumer_identity,
                "consumer_count": consumer_count,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    text_bytes = _mul(
        _add(
            _add(
                manifest,
                _sum(
                    (row.max_manifest_characters for row in rows),
                    where="multi-layout child manifest budget",
                ),
                where="multi-layout text budget",
            ),
            _mul(control_characters, 8, where="multi-layout control text budget"),
            where="multi-layout text budget",
        ),
        4,
        where="multi-layout text byte budget",
    )
    total = _add(total, text_bytes, where="multi-layout checkpoint byte budget")
    total = _add(
        total,
        _mul(len(member_names), _NPY_HEADER_BUDGET, where="multi-layout header budget"),
        where="multi-layout checkpoint byte budget",
    )
    archive_bytes = _archive_byte_capacity(
        total, tuple(member_names), where="multi-layout container archive budget"
    )
    return CheckpointResourceBudget(
        "multi_layout_uniform",
        len(member_names),
        manifest,
        max(maximum_array, text_bytes),
        total,
        archive_bytes,
        authority,
    )


def _producer_checkpoint_resource_budget(
    payload: Mapping[str, Any], *, runtime_kind: str, authority: str
) -> CheckpointResourceBudget:
    """Build a private exact budget from already-authenticated producer arrays."""
    import numpy as np
    from pops.output._checkpoint_collective import _NPY_HEADER_BUDGET, _manifest_character_budget

    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("producer checkpoint budget requires a non-empty exact array mapping")
    names = tuple(payload)
    if len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("producer checkpoint budget requires unique non-empty member names")
    migration_characters = _checkpoint_migration_provenance_characters(payload)
    migration_bytes = 0
    if "checkpoint_migration" in payload:
        migration_bytes = _mul(
            _CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS,
            int(np.dtype("U1").itemsize),
            where="checkpoint migration provenance byte capacity",
        )
    total = 0
    maximum = 1
    for name in names:
        value = np.asarray(payload[name])
        if value.dtype.hasobject or value.dtype.itemsize <= 0 or value.ndim > 4:
            raise TypeError("producer checkpoint member %r has no bounded primitive array" % name)
        size = _capacity(int(value.nbytes), where="producer checkpoint array bytes")
        if name == "checkpoint_migration":
            # Keep the producer envelope independent of the particular reviewed mapping spelling.
            # Decode therefore has the same fixed reservation for every accepted provenance.
            assert migration_characters <= _CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS
            size = migration_bytes
        total = _add(total, size, where="producer checkpoint aggregate bytes")
        maximum = max(maximum, size)
    total = _add(
        total,
        _mul(len(names), _NPY_HEADER_BUDGET, where="producer checkpoint header budget"),
        where="producer checkpoint aggregate bytes",
    )
    archive_bytes = _archive_byte_capacity(
        max(total, 1), names, where="producer checkpoint archive budget"
    )
    return CheckpointResourceBudget(
        runtime_kind,
        len(names),
        _manifest_character_budget(names),
        maximum,
        max(total, 1),
        archive_bytes,
        authority,
    )


def _reviewed_archive_checkpoint_resource_budget(
    *,
    content_sha256: str,
    runtime_kind: str,
    max_members: int,
    max_manifest_characters: int,
    max_array_bytes: int,
    max_uncompressed_bytes: int,
    max_archive_bytes: int,
) -> CheckpointResourceBudget:
    """Materialize one mapping-reviewed envelope before reading its pinned archive.

    The explicit limits are review input. In particular, this helper never opens the archive and
    never promotes central-directory claims into allocation authority.
    """
    if not isinstance(content_sha256, str) or len(content_sha256) != 64:
        raise TypeError("reviewed checkpoint SHA-256 must be canonical lowercase hexadecimal")
    try:
        raw_digest = bytes.fromhex(content_sha256)
    except ValueError:
        raise ValueError("reviewed checkpoint SHA-256 is not hexadecimal") from None
    if raw_digest.hex() != content_sha256:
        raise ValueError("reviewed checkpoint SHA-256 must be canonical lowercase hexadecimal")
    return CheckpointResourceBudget(
        runtime_kind,
        _capacity(max_members, where="reviewed checkpoint member capacity", positive=True),
        _capacity(
            max_manifest_characters,
            where="reviewed checkpoint manifest capacity",
            positive=True,
        ),
        _capacity(max_array_bytes, where="reviewed checkpoint array capacity", positive=True),
        _capacity(
            max_uncompressed_bytes,
            where="reviewed checkpoint uncompressed capacity",
            positive=True,
        ),
        _capacity(max_archive_bytes, where="reviewed checkpoint archive capacity", positive=True),
        "reviewed-checkpoint-archive-sha256:" + content_sha256,
    )


__all__ = [
    "CheckpointResourceBudget",
    "aggregate_checkpoint_resource_budgets",
    "install_amr_checkpoint_resource_budget",
    "install_layout_checkpoint_resource_budget",
    "install_uniform_checkpoint_resource_budget",
    "require_checkpoint_resource_budget",
]
