"""Fail-closed offline migration of the frozen Uniform v2 checkpoint format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
from pops._manifest_protocol import strict_json_loads


UNIFORM_V2_MIGRATION_SCHEMA_VERSION = 1
UNIFORM_V2_SOURCE_VERSION = 2
UNIFORM_V2_MIGRATION_PROTOCOL = "pops.uniform-checkpoint-v2-offline-migration.v1"
UNIFORM_V2_AUTHORITY_TRANSFERS = (
    "lifecycle_identities",
    "target_abi",
    "target_program",
    "temporal_restart_state",
    "program_cadence",
    "history_fill_count",
    "history_persistence",
    "consumer_state",
    "embedded_boundary_contract",
)


@dataclass(frozen=True, slots=True)
class UniformV2BlockMapping:
    """Reviewed one-to-one block and conservative-component correspondence."""

    source: str
    target: str
    components: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class UniformV2HistoryMapping:
    """Reviewed one-to-one history and component-index correspondence."""

    source: str
    target: str
    components: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class UniformV2MigrationMapping:
    """Complete authority pins and semantic mapping for one reviewed v2 artifact."""

    reviewed_mapping_id: str
    source_content_sha256: str
    source_abi_key: str
    source_program_hash: str
    authority_restart_identity: str
    target_semantic_identity: str
    target_artifact_identity: str
    target_bind_identity: str
    target_run_identity: str
    target_abi_key: str
    target_program_hash: str
    authority_transfers: tuple[str, ...]
    blocks: tuple[UniformV2BlockMapping, ...]
    histories: tuple[UniformV2HistoryMapping, ...]
    schema_version: int = UNIFORM_V2_MIGRATION_SCHEMA_VERSION
    source_version: int = UNIFORM_V2_SOURCE_VERSION
    target_version: int = UNIFORM_CHECKPOINT_PAYLOAD_VERSION


@dataclass(frozen=True, slots=True)
class UniformV2MigrationReport:
    source_content_sha256: str
    authority_restart_identity: str
    destination_restart_identity: str
    mapping_identity: str
    destination: str


@dataclass(frozen=True, slots=True)
class _LegacyUniformV2:
    payload: Mapping[str, Any]
    blocks: tuple[str, ...]
    block_components: Mapping[str, tuple[str, ...]]
    histories: tuple[str, ...]
    history_depth: Mapping[str, int]
    history_ncomp: Mapping[str, int]
    nx: int
    ny: int
    time: float
    macro_step: int
    abi_key: str
    program_hash: str


@dataclass(frozen=True, slots=True)
class _CurrentUniformAuthority:
    payload: Mapping[str, Any]
    manifest: Mapping[str, Any]
    restart_identity: Any
    identities: tuple[Any, Any, Any, Any]
    temporal: Any
    blocks: tuple[str, ...]
    block_components: Mapping[str, tuple[str, ...]]
    histories: tuple[str, ...]
    spatial: Any
    time: float
    macro_step: int
    abi_key: str
    program_hash: str


def _files(payload: Mapping[str, Any]) -> set[str]:
    return {str(name) for name in payload}


def _int_scalar(payload: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "iu":
        raise TypeError("%s must be an exact integer scalar" % key)
    result = int(value.item())
    if result < minimum:
        raise ValueError("%s must be >= %d" % (key, minimum))
    return result


def _float64_scalar(payload: Mapping[str, Any], key: str) -> float:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype != np.dtype(np.float64):
        raise TypeError("%s must be an exact binary64 scalar" % key)
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % key)
    return result


def _bool_scalar(payload: Mapping[str, Any], key: str) -> bool:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind != "b":
        raise TypeError("%s must be an exact boolean scalar" % key)
    return bool(value.item())


def _text_scalar(payload: Mapping[str, Any], key: str) -> str:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "US":
        raise TypeError("%s must be an exact text scalar" % key)
    result = str(value.item())
    if not result:
        raise ValueError("%s must be non-empty text" % key)
    return result


def _text_vector(
    payload: Mapping[str, Any],
    key: str,
    *,
    nonempty: bool,
) -> tuple[str, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise TypeError("%s must be an exact text vector" % key)
    result = tuple(str(item) for item in value.tolist())
    if (nonempty and not result) or any(not item for item in result):
        raise ValueError("%s contains an invalid empty name" % key)
    if len(result) != len(set(result)):
        raise ValueError("%s must contain unique names" % key)
    return result


def _int_vector(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise TypeError("%s must be an exact integer vector" % key)
    return tuple(int(item) for item in value.tolist())


def _float64_array(
    payload: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
) -> Any:
    import numpy as np

    value = np.asarray(payload[key])
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise TypeError("%s must be a binary64 array with shape %r" % (key, shape))
    if not np.all(np.isfinite(value)):
        raise ValueError("%s must contain only finite values" % key)
    return value


def _require_sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError("%s must be 64 lowercase hexadecimal characters" % where)
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        raise ValueError("%s is not hexadecimal" % where) from None
    if raw.hex() != value:
        raise ValueError("%s must be lowercase canonical hexadecimal" % where)
    return value


def _decode_file(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    from pops.output._checkpoint_collective import decode_checkpoint_bytes

    raw = path.read_bytes()
    return raw, decode_checkpoint_bytes(raw)


def _validate_legacy_v2(payload: Mapping[str, Any]) -> _LegacyUniformV2:
    version = _int_scalar(payload, "pops_checkpoint_version", minimum=1)
    if version != UNIFORM_V2_SOURCE_VERSION:
        raise ValueError(
            "offline Uniform migration accepts exactly payload version %d"
            % UNIFORM_V2_SOURCE_VERSION
        )
    nx = _int_scalar(payload, "nx", minimum=1)
    ny = _int_scalar(payload, "ny", minimum=1)
    time = _float64_scalar(payload, "t")
    if time < 0.0:
        raise ValueError("legacy Uniform time must be non-negative")
    macro_step = _int_scalar(payload, "macro_step")
    abi = _text_scalar(payload, "abi_key")
    program_hash = _require_sha256(
        _text_scalar(payload, "program_hash"),
        where="legacy Uniform program_hash",
    )
    blocks = _text_vector(payload, "blocks", nonempty=True)
    histories = _text_vector(payload, "history_names", nonempty=False)
    expected = {
        "pops_checkpoint_version",
        "t",
        "macro_step",
        "nx",
        "ny",
        "abi_key",
        "program_hash",
        "blocks",
        "phi",
        "history_names",
    }
    block_components: dict[str, tuple[str, ...]] = {}
    for block in blocks:
        ncomp = _int_scalar(payload, "ncomp_" + block, minimum=1)
        names = _text_vector(payload, "names_" + block, nonempty=True)
        if len(names) != ncomp:
            raise ValueError("legacy block %r component names do not match ncomp" % block)
        _float64_array(payload, "state_" + block, (ncomp, nx, ny))
        block_components[block] = names
        expected.update({"ncomp_" + block, "names_" + block, "state_" + block})
    _float64_array(payload, "phi", (nx, ny))

    history_depth: dict[str, int] = {}
    history_ncomp: dict[str, int] = {}
    for name in histories:
        depth = _int_scalar(payload, "history_depth_" + name, minimum=1)
        ncomp = _int_scalar(payload, "history_ncomp_" + name, minimum=1)
        initialized = _bool_scalar(payload, "history_init_" + name)
        policy = strict_json_loads(
            _text_scalar(payload, "history_policy_" + name),
            where="legacy Uniform history policy",
        )
        if policy != {"kind": "dense"}:
            raise ValueError("legacy Uniform migration supports only exact dense histories")
        stored = _int_vector(payload, "history_stored_slots_" + name)
        if stored != tuple(range(depth)):
            raise ValueError("legacy Uniform migration requires every history slot")
        slot_dt = _float64_array(payload, "history_slot_dt_" + name, (depth,))
        if initialized and any(float(value) <= 0.0 for value in slot_dt):
            raise ValueError("initialized legacy history slot durations must be positive")
        if not initialized and any(float(value) != 0.0 for value in slot_dt):
            raise ValueError("uninitialized legacy history slot durations must be zero")
        expected.update(
            {
                "history_depth_" + name,
                "history_ncomp_" + name,
                "history_init_" + name,
                "history_policy_" + name,
                "history_stored_slots_" + name,
                "history_slot_dt_" + name,
            }
        )
        for slot in stored:
            key = "history_%s_%d" % (name, slot)
            _float64_array(payload, key, (ncomp, nx, ny))
            expected.add(key)
        history_depth[name] = depth
        history_ncomp[name] = ncomp
    if _files(payload) != expected:
        raise ValueError("legacy Uniform v2 payload has ambiguous or unknown keys")
    return _LegacyUniformV2(
        payload,
        blocks,
        block_components,
        histories,
        history_depth,
        history_ncomp,
        nx,
        ny,
        time,
        macro_step,
        abi,
        program_hash,
    )


def _current_authority(payload: Mapping[str, Any]) -> _CurrentUniformAuthority:
    from pops.output._consumer_contracts import ConsumerGraph
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        _identity_from_json,
        inspect_checkpoint_payload_integrity,
        require_exact_payload_version,
    )
    from pops.runtime._checkpoint_embedded_boundary import (
        EMBEDDED_BOUNDARY_CONTRACT_KEY,
        inspect_checkpoint_embedded_boundary_contract,
    )
    from pops.runtime._checkpoint_spatial import (
        SPATIAL_CONTRACT_KEY,
        inspect_checkpoint_spatial_contract,
    )
    from pops.runtime._program_cadence_checkpoint import (
        PROGRAM_CADENCE_CHECKPOINT_KEYS,
        ProgramCadenceCheckpointState,
        _validate_state,
    )
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher
    from pops.runtime._runtime_instance import RuntimeInstance
    from pops.runtime._temporal_restart import TemporalRestartState
    from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
    from pops.runtime._system_io_history import validate_history_slot_dt_payload
    from pops.time._history.persistence import Dense, HistoryPersistence

    manifest, restart = inspect_checkpoint_payload_integrity(payload, runtime_kind="uniform")
    require_exact_payload_version(
        payload,
        key="pops_checkpoint_version",
        expected=UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        runtime_kind="Uniform migration authority",
    )
    preflight_uniform_restart(payload)
    spatial = inspect_checkpoint_spatial_contract(payload)
    inspect_checkpoint_embedded_boundary_contract(payload)
    time = _float64_scalar(payload, "t")
    macro_step = _int_scalar(payload, "macro_step")
    abi = _text_scalar(payload, "abi_key")
    program_hash = _require_sha256(
        _text_scalar(payload, "program_hash"),
        where="current Uniform program_hash",
    )
    blocks = _text_vector(payload, "blocks", nonempty=True)
    histories = _text_vector(payload, "history_names", nonempty=False)
    if _text_vector(payload, "field_provider_slots", nonempty=False):
        raise ValueError("Uniform v2 migration authority must have no field-provider slots")
    if _int_vector(payload, "cache_nodes") or _text_vector(payload, "cache_names", nonempty=False):
        raise ValueError("Uniform v2 migration authority must have no scheduled caches")

    expected = {
        "pops_checkpoint_version",
        "t",
        "macro_step",
        SPATIAL_CONTRACT_KEY,
        EMBEDDED_BOUNDARY_CONTRACT_KEY,
        "abi_key",
        "program_hash",
        "blocks",
        "phi",
        "field_provider_slots",
        "history_names",
        "cache_nodes",
        "cache_names",
        "temporal_restart_state",
        "runtime_consumer_graph",
        "runtime_consumer_cursors",
        "runtime_consumer_diagnostics",
        MANIFEST_KEY,
        IDENTITY_KEY,
    } | set(PROGRAM_CADENCE_CHECKPOINT_KEYS)
    block_components: dict[str, tuple[str, ...]] = {}
    for block in blocks:
        ncomp = _int_scalar(payload, "ncomp_" + block, minimum=1)
        names = _text_vector(payload, "names_" + block, nonempty=True)
        if len(names) != ncomp:
            raise ValueError("authority block %r component names do not match ncomp" % block)
        _float64_array(payload, "state_" + block, (ncomp, *spatial.shape))
        block_components[block] = names
        expected.update({"ncomp_" + block, "names_" + block, "state_" + block})
    _float64_array(payload, "phi", spatial.shape)

    for name in histories:
        depth = _int_scalar(payload, "history_depth_" + name, minimum=1)
        ncomp = _int_scalar(payload, "history_ncomp_" + name, minimum=1)
        initialized = _bool_scalar(payload, "history_init_" + name)
        fill_count = _int_scalar(payload, "history_fill_count_" + name)
        if fill_count > depth or initialized != (fill_count > 0):
            raise ValueError("authority history %r has inconsistent fill metadata" % name)
        policy = HistoryPersistence.from_json(_text_scalar(payload, "history_policy_" + name))
        if type(policy) is not Dense:
            raise ValueError("Uniform v2 migration authority must use Dense histories")
        requested = _int_vector(payload, "history_requested_stored_slots_" + name)
        stored = _int_vector(payload, "history_stored_slots_" + name)
        if requested != tuple(range(depth)) or stored != requested:
            raise ValueError("Uniform v2 migration authority must store every history slot")
        if _text_scalar(payload, "history_storage_mode_" + name) != "policy":
            raise ValueError("Uniform v2 migration authority history must use policy storage")
        validate_history_slot_dt_payload(payload, name, depth, fill_count)
        expected.update(
            {
                "history_depth_" + name,
                "history_ncomp_" + name,
                "history_init_" + name,
                "history_fill_count_" + name,
                "history_policy_" + name,
                "history_requested_stored_slots_" + name,
                "history_stored_slots_" + name,
                "history_storage_mode_" + name,
                "history_slot_dt_" + name,
            }
        )
        for slot in stored:
            key = "history_%s_%d" % (name, slot)
            _float64_array(payload, key, (ncomp, *spatial.shape))
            expected.add(key)
    if _files(payload) != expected:
        raise ValueError("current Uniform migration authority has ambiguous or unknown keys")

    cadence = ProgramCadenceCheckpointState(
        _int_scalar(payload, "program_cadence_substeps", minimum=1),
        _int_scalar(payload, "program_cadence_stride", minimum=1),
        _int_scalar(payload, "program_cadence_window_steps"),
        _float64_scalar(payload, "program_cadence_window_dt"),
        _float64_scalar(payload, "program_cadence_window_start_time"),
        _float64_scalar(payload, "program_last_dt"),
    )
    _validate_state(cadence, macro_step=macro_step, accepted_time=time, phase="migration")
    temporal = TemporalRestartState.from_json(
        payload["temporal_restart_state"],
        time=time,
        macro_step=macro_step,
    )
    scheduled_histories = tuple(
        str(row["name"]) for row in (temporal.program_schedule or {}).get("histories", ())
    )
    if scheduled_histories != histories:
        raise ValueError("authority histories differ from its temporal Program schedule")

    if _text_scalar(payload, "runtime_consumer_graph") != ConsumerGraph(()).identity.token:
        raise ValueError("Uniform v2 migration authority requires an empty ConsumerGraph")
    cursor_data = strict_json_loads(
        _text_scalar(payload, "runtime_consumer_cursors"),
        where="authority consumer cursors",
    )
    if RuntimeInstance._checkpoint_cursors_from_data(cursor_data).to_data() != {
        "schema_version": 1,
        "rows": [],
    }:
        raise ValueError("Uniform v2 migration authority requires empty consumer cursors")
    diagnostic_data = strict_json_loads(
        _text_scalar(payload, "runtime_consumer_diagnostics"),
        where="authority consumer diagnostics",
    )
    canonical_diagnostics = RuntimeConsumerPublisher.validate_diagnostic_restart_state(
        diagnostic_data
    )
    if canonical_diagnostics != {
        "schema_version": 2,
        "baselines": {},
        "diagnostics": [],
    }:
        raise ValueError("Uniform v2 migration authority requires empty diagnostics")

    identities = (
        _identity_from_json(manifest["semantic_identity"]),
        _identity_from_json(manifest["artifact_identity"]),
        _identity_from_json(manifest["bind_identity"]),
        _identity_from_json(manifest["run_identity"]),
    )
    return _CurrentUniformAuthority(
        payload,
        manifest,
        restart,
        identities,
        temporal,
        blocks,
        block_components,
        histories,
        spatial,
        time,
        macro_step,
        abi,
        program_hash,
    )


def _mapping_data(mapping: UniformV2MigrationMapping) -> dict[str, Any]:
    return {
        "schema_version": mapping.schema_version,
        "source_version": mapping.source_version,
        "target_version": mapping.target_version,
        "reviewed_mapping_id": mapping.reviewed_mapping_id,
        "source_content_sha256": mapping.source_content_sha256,
        "source_abi_key": mapping.source_abi_key,
        "source_program_hash": mapping.source_program_hash,
        "authority_restart_identity": mapping.authority_restart_identity,
        "target_semantic_identity": mapping.target_semantic_identity,
        "target_artifact_identity": mapping.target_artifact_identity,
        "target_bind_identity": mapping.target_bind_identity,
        "target_run_identity": mapping.target_run_identity,
        "target_abi_key": mapping.target_abi_key,
        "target_program_hash": mapping.target_program_hash,
        "authority_transfers": list(mapping.authority_transfers),
        "blocks": [
            {
                "source": row.source,
                "target": row.target,
                "components": [list(pair) for pair in row.components],
            }
            for row in mapping.blocks
        ],
        "histories": [
            {
                "source": row.source,
                "target": row.target,
                "components": [list(pair) for pair in row.components],
            }
            for row in mapping.histories
        ],
    }


def _validate_mapping(mapping: UniformV2MigrationMapping) -> dict[str, Any]:
    if type(mapping) is not UniformV2MigrationMapping:
        raise TypeError("Uniform v2 migration requires an exact reviewed mapping")
    if (
        mapping.schema_version != UNIFORM_V2_MIGRATION_SCHEMA_VERSION
        or mapping.source_version != UNIFORM_V2_SOURCE_VERSION
        or mapping.target_version != UNIFORM_CHECKPOINT_PAYLOAD_VERSION
    ):
        raise ValueError("Uniform v2 migration mapping pins unsupported schema versions")
    if not isinstance(mapping.reviewed_mapping_id, str) or not mapping.reviewed_mapping_id:
        raise ValueError("Uniform v2 migration mapping needs a reviewed_mapping_id")
    _require_sha256(mapping.source_content_sha256, where="mapping source_content_sha256")
    _require_sha256(mapping.source_program_hash, where="mapping source_program_hash")
    _require_sha256(mapping.target_program_hash, where="mapping target_program_hash")
    text_fields = (
        mapping.source_abi_key,
        mapping.authority_restart_identity,
        mapping.target_semantic_identity,
        mapping.target_artifact_identity,
        mapping.target_bind_identity,
        mapping.target_run_identity,
        mapping.target_abi_key,
    )
    if any(not isinstance(value, str) or not value for value in text_fields):
        raise TypeError("Uniform v2 migration identity/ABI pins must be non-empty text")
    if type(mapping.blocks) is not tuple or type(mapping.histories) is not tuple:
        raise TypeError("Uniform v2 migration block/history maps must be exact tuples")
    if mapping.authority_transfers != UNIFORM_V2_AUTHORITY_TRANSFERS:
        raise ValueError(
            "Uniform v2 migration must explicitly attest the complete authority transfer set"
        )
    for row in mapping.blocks:
        if (
            type(row) is not UniformV2BlockMapping
            or not row.source
            or not row.target
            or type(row.components) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(not isinstance(value, str) or not value for value in pair)
                for pair in row.components
            )
        ):
            raise TypeError("Uniform v2 block mappings must be complete exact tuples")
    for row in mapping.histories:
        if (
            type(row) is not UniformV2HistoryMapping
            or not row.source
            or not row.target
            or type(row.components) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
                for pair in row.components
            )
        ):
            raise TypeError("Uniform v2 history mappings must be complete exact tuples")
    return _mapping_data(mapping)


def _pin_authorities(
    source: _LegacyUniformV2,
    authority: _CurrentUniformAuthority,
    source_sha256: str,
    mapping: UniformV2MigrationMapping,
) -> None:
    if source_sha256 != mapping.source_content_sha256:
        raise ValueError("legacy source content SHA-256 differs from the reviewed mapping")
    if source.abi_key != mapping.source_abi_key:
        raise ValueError("legacy source ABI key differs from the reviewed mapping")
    if source.program_hash != mapping.source_program_hash:
        raise ValueError("legacy source Program hash differs from the reviewed mapping")
    semantic, artifact, bind, run = authority.identities
    observed = (
        authority.restart_identity.token,
        semantic.token,
        artifact.token,
        bind.token,
        run.token,
        authority.abi_key,
        authority.program_hash,
    )
    expected = (
        mapping.authority_restart_identity,
        mapping.target_semantic_identity,
        mapping.target_artifact_identity,
        mapping.target_bind_identity,
        mapping.target_run_identity,
        mapping.target_abi_key,
        mapping.target_program_hash,
    )
    if observed != expected:
        raise ValueError("current authority or target lifecycle pins differ from the mapping")
    if ((source.nx, source.ny), source.time, source.macro_step) != (
        authority.spatial.shape,
        authority.time,
        authority.macro_step,
    ):
        raise ValueError(
            "legacy source and current authority must have the exact same grid and accepted clock"
        )


def _require_bijection(
    pairs: tuple[tuple[Any, Any], ...],
    source: tuple[Any, ...],
    target: tuple[Any, ...],
    *,
    where: str,
) -> None:
    if tuple(left for left, _ in pairs) != source:
        raise ValueError("%s does not cover the source exactly in canonical order" % where)
    if tuple(right for _, right in pairs) != target:
        raise ValueError("%s does not cover the target exactly in canonical order" % where)


def _migrate_payload(
    source: _LegacyUniformV2,
    authority: _CurrentUniformAuthority,
    mapping: UniformV2MigrationMapping,
    mapping_data: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    from pops.identity import make_identity
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        _seal_checkpoint_payload_with_identities,
    )

    block_pairs = tuple((row.source, row.target) for row in mapping.blocks)
    _require_bijection(block_pairs, source.blocks, authority.blocks, where="block mapping")
    history_pairs = tuple((row.source, row.target) for row in mapping.histories)
    _require_bijection(
        history_pairs,
        source.histories,
        authority.histories,
        where="history mapping",
    )
    output = {
        name: np.array(value, copy=True)
        for name, value in authority.payload.items()
        if name not in {MANIFEST_KEY, IDENTITY_KEY}
    }
    for row in mapping.blocks:
        source_names = source.block_components[row.source]
        target_names = authority.block_components[row.target]
        _require_bijection(
            row.components,
            source_names,
            target_names,
            where="block %r component mapping" % row.source,
        )
        source_state = np.asarray(source.payload["state_" + row.source])
        target_state = np.empty_like(np.asarray(output["state_" + row.target]))
        for source_name, target_name in row.components:
            target_state[target_names.index(target_name)] = source_state[
                source_names.index(source_name)
            ]
        output["state_" + row.target] = target_state
    output["phi"] = np.array(source.payload["phi"], copy=True)

    for row in mapping.histories:
        source_depth = source.history_depth[row.source]
        target_depth = int(output["history_depth_" + row.target])
        source_ncomp = source.history_ncomp[row.source]
        target_ncomp = int(output["history_ncomp_" + row.target])
        if source_depth != target_depth:
            raise ValueError("history %r depth differs from its current authority" % row.source)
        source_slot_dt = np.asarray(source.payload["history_slot_dt_" + row.source])
        authority_slot_dt = np.asarray(output["history_slot_dt_" + row.target])
        if not np.array_equal(source_slot_dt, authority_slot_dt):
            raise ValueError(
                "history %r outgoing-dt ledger differs from its reviewed authority" % row.source
            )
        if int(output["history_fill_count_" + row.target]) != min(
            source.macro_step,
            source_depth,
        ):
            raise ValueError(
                "history %r fill count is inconsistent with the legacy accepted clock" % row.source
            )
        if source_depth and float(source_slot_dt[0]) != float(output["program_last_dt"]):
            raise ValueError(
                "history %r newest outgoing dt differs from the authority cadence" % row.source
            )
        _require_bijection(
            row.components,
            tuple(range(source_ncomp)),
            tuple(range(target_ncomp)),
            where="history %r component mapping" % row.source,
        )
        if bool(source.payload["history_init_" + row.source]) != bool(
            output["history_init_" + row.target]
        ):
            raise ValueError("history %r initialization differs from its authority" % row.source)
        output["history_slot_dt_" + row.target] = np.array(source_slot_dt, copy=True)
        for slot in range(source_depth):
            source_values = np.asarray(source.payload["history_%s_%d" % (row.source, slot)])
            target_values = np.empty_like(np.asarray(output["history_%s_%d" % (row.target, slot)]))
            for source_index, target_index in row.components:
                target_values[target_index] = source_values[source_index]
            output["history_%s_%d" % (row.target, slot)] = target_values

    mapping_identity = make_identity("uniform-v2-migration-map", mapping_data)
    output["checkpoint_migration"] = np.asarray(
        json.dumps(
            {
                "protocol": UNIFORM_V2_MIGRATION_PROTOCOL,
                "mapping_identity": mapping_identity.token,
                "mapping": mapping_data,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    restart = _seal_checkpoint_payload_with_identities(
        output,
        runtime_kind="uniform",
        semantic=authority.identities[0],
        artifact=authority.identities[1],
        bind=authority.identities[2],
        run=authority.identities[3],
    )
    return output, (restart, mapping_identity)


def _encode(payload: Mapping[str, Any]) -> bytes:
    import numpy as np

    stream = BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def _validate_migrated_bytes(
    raw: bytes,
    *,
    authority: _CurrentUniformAuthority,
    expected_restart: Any,
    expected_mapping_identity: Any,
) -> None:
    from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
    from pops.output._consumer_contracts import ConsumerGraph
    from pops.output._checkpoint_collective import decode_checkpoint_bytes
    from pops.runtime._checkpoint_manifest import (
        inspect_checkpoint_payload_integrity,
        require_exact_payload_version,
    )
    from pops.runtime._program_cadence_checkpoint import (
        ProgramCadenceCheckpointState,
        _validate_state,
    )
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher
    from pops.runtime._runtime_instance import RuntimeInstance
    from pops.runtime._temporal_restart import TemporalRestartState
    from pops.runtime._uniform_restart_preflight import preflight_uniform_restart

    payload = decode_checkpoint_bytes(raw)
    _, restart = inspect_checkpoint_payload_integrity(payload, runtime_kind="uniform")
    if restart.token != expected_restart.token:
        raise RuntimeError("migrated checkpoint restart identity changed during encoding")
    require_exact_payload_version(
        payload,
        key="pops_checkpoint_version",
        expected=UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        runtime_kind="migrated Uniform",
    )
    preflight_uniform_restart(payload)
    cadence = ProgramCadenceCheckpointState(
        _int_scalar(payload, "program_cadence_substeps", minimum=1),
        _int_scalar(payload, "program_cadence_stride", minimum=1),
        _int_scalar(payload, "program_cadence_window_steps"),
        _float64_scalar(payload, "program_cadence_window_dt"),
        _float64_scalar(payload, "program_cadence_window_start_time"),
        _float64_scalar(payload, "program_last_dt"),
    )
    _validate_state(
        cadence,
        macro_step=_int_scalar(payload, "macro_step"),
        accepted_time=_float64_scalar(payload, "t"),
        phase="migrated checkpoint",
    )
    TemporalRestartState.from_json(
        payload["temporal_restart_state"],
        time=payload["t"],
        macro_step=payload["macro_step"],
        program_schedule=authority.temporal.program_schedule,
    )
    if _text_scalar(payload, "runtime_consumer_graph") != ConsumerGraph(()).identity.token:
        raise ValueError("migrated checkpoint ConsumerGraph differs from its authority")
    RuntimeInstance._checkpoint_cursors_from_data(
        strict_json_loads(
            _text_scalar(payload, "runtime_consumer_cursors"),
            where="migrated consumer cursors",
        )
    )
    RuntimeConsumerPublisher.validate_diagnostic_restart_state(
        strict_json_loads(
            _text_scalar(payload, "runtime_consumer_diagnostics"),
            where="migrated consumer diagnostics",
        )
    )
    provenance = strict_json_loads(
        _text_scalar(payload, "checkpoint_migration"),
        where="migrated checkpoint provenance",
    )
    if (
        provenance.get("protocol") != UNIFORM_V2_MIGRATION_PROTOCOL
        or provenance.get("mapping_identity") != expected_mapping_identity.token
    ):
        raise ValueError("migrated checkpoint provenance differs from its reviewed mapping")


def _path(value: Any, *, where: str) -> Path:
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise TypeError("%s must be non-empty filesystem text" % where)
    return Path(os.path.abspath(os.path.normpath(text)))


def migrate_uniform_v2_checkpoint(
    source: Any,
    destination: Any,
    *,
    current_authority: Any,
    mapping: UniformV2MigrationMapping,
) -> UniformV2MigrationReport:
    """Publish one current checkpoint from a true v2 artifact, entirely offline.

    ``current_authority`` is a complete, authenticated v5 checkpoint captured from the exact
    target runtime. The explicit mapping pins both artifacts and supplies every semantic
    correspondence absent from v2. Runtime restart paths never call this function.
    """
    source_path = _path(source, where="legacy source")
    authority_path = _path(current_authority, where="current authority")
    destination_path = _path(destination, where="migration destination")
    if destination_path in {source_path, authority_path}:
        raise ValueError("offline checkpoint migration refuses to overwrite either input")
    if destination_path.exists():
        raise FileExistsError("offline checkpoint migration refuses to replace its destination")

    mapping_data = _validate_mapping(mapping)
    source_raw = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if source_sha256 != mapping.source_content_sha256:
        raise ValueError("legacy source content SHA-256 differs from the reviewed mapping")
    from pops.output._checkpoint_collective import decode_checkpoint_bytes

    source_payload = decode_checkpoint_bytes(source_raw)
    legacy = _validate_legacy_v2(source_payload)
    _, authority_payload = _decode_file(authority_path)
    authority = _current_authority(authority_payload)
    _pin_authorities(legacy, authority, source_sha256, mapping)
    migrated, (restart, mapping_identity) = _migrate_payload(
        legacy,
        authority,
        mapping,
        mapping_data,
    )
    output = _encode(migrated)
    _validate_migrated_bytes(
        output,
        authority=authority,
        expected_restart=restart,
        expected_mapping_identity=mapping_identity,
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=".%s." % destination_path.name,
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_migrated_bytes(
            temporary.read_bytes(),
            authority=authority,
            expected_restart=restart,
            expected_mapping_identity=mapping_identity,
        )
        os.link(temporary, destination_path)
        directory_fd = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return UniformV2MigrationReport(
        source_sha256,
        authority.restart_identity.token,
        restart.token,
        mapping_identity.token,
        str(destination_path),
    )


__all__ = [
    "UNIFORM_V2_AUTHORITY_TRANSFERS",
    "UNIFORM_V2_MIGRATION_PROTOCOL",
    "UNIFORM_V2_MIGRATION_SCHEMA_VERSION",
    "UNIFORM_V2_SOURCE_VERSION",
    "UniformV2BlockMapping",
    "UniformV2HistoryMapping",
    "UniformV2MigrationMapping",
    "UniformV2MigrationReport",
    "migrate_uniform_v2_checkpoint",
]
