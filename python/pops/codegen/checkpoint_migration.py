"""Strict offline migration of one authenticated Uniform-v2 compatibility projection.

Runtime restart remains deliberately current-format-only. This module transforms only a sealed v2
projection that retains the current runtime's semantic identity and compiled Program hash. It
validates that envelope, requires the temporal facts omitted by v2, emits a complete current
payload, and runs both current restart preflights before publishing destination bytes.

This is not yet a release-to-release migration for artifacts emitted by the historical v2 writer:
those files were unsealed and omitted the identities and continuation facts needed to prove
semantic equivalence. They remain refused until an explicit, reviewed version map can supply that
evidence. AMR hierarchy migration and distributed publication atomicity remain owned by ADC-678
and ADC-666 respectively.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

from ._checkpoint_migration_uniform_v2 import (
    HISTORICAL_UNIFORM_PAYLOAD_VERSION,
    validate_compatible_v2_projection,
)
from pops.time import StepStrategy


MIGRATION_PROTOCOL = "pops.uniform-checkpoint-migration.v1"
MIGRATION_COMPATIBILITY_SCOPE = (
    "authenticated-v2-projection-same-semantic-and-program-hash"
)


@dataclass(frozen=True, slots=True)
class UniformCheckpointMigrationState:
    """Explicit temporal facts absent from a historical Uniform v2 checkpoint.

    ``last_accepted_dt`` and the transaction counters cannot be recovered from the old artifact.
    They are therefore mandatory caller evidence rather than inferred defaults.  ``controls`` are
    the exact runtime controls accepted by ``step_strategy``.
    """

    step_strategy: StepStrategy
    last_accepted_dt: float | None
    transaction_stats: Mapping[str, int]
    controls: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step_strategy, StepStrategy):
            raise TypeError("step_strategy must be a registered pops.time.StepStrategy")
        controls = {} if self.controls is None else dict(self.controls)
        self.step_strategy.validate_runtime_controls(controls)
        canonical_controls = self.step_strategy.restore_runtime_controls(
            self.step_strategy.runtime_controls_data(controls)
        )
        object.__setattr__(
            self, "controls", MappingProxyType(dict(canonical_controls)))
        last_dt = self.last_accepted_dt
        if last_dt is not None:
            if isinstance(last_dt, bool) or not isinstance(last_dt, (int, float)):
                raise TypeError("last_accepted_dt must be a positive finite scalar or None")
            last_dt = float(last_dt)
            if not math.isfinite(last_dt) or last_dt <= 0.0:
                raise ValueError("last_accepted_dt must be finite and > 0")
            object.__setattr__(self, "last_accepted_dt", last_dt)
        stats = dict(self.transaction_stats)
        if set(stats) != {"accepted", "rejected", "failed"}:
            raise ValueError(
                "transaction_stats must contain exactly accepted/rejected/failed")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in stats.values()):
            raise TypeError("transaction_stats values must be non-negative Python integers")
        object.__setattr__(self, "transaction_stats", MappingProxyType(stats))


@dataclass(frozen=True, slots=True)
class CheckpointMigrationReport:
    """Content identities and destination of one completed offline migration."""

    source_restart_identity: str
    migrated_restart_identity: str
    from_payload_version: int
    to_payload_version: int
    destination: str


def _current_temporal_json(
    runtime: Any,
    *,
    facts: Mapping[str, Any],
    state: UniformCheckpointMigrationState,
) -> str:
    from pops.runtime._step_strategy import run_control_payload
    from pops.runtime._temporal_restart import TemporalRestartState

    time = facts["time"]
    macro_step = facts["macro_step"]
    if macro_step == 0 and state.last_accepted_dt is not None:
        raise ValueError("a step-zero checkpoint must use last_accepted_dt=None")
    if macro_step > 0 and state.last_accepted_dt is None:
        raise ValueError(
            "last_accepted_dt is mandatory for a non-zero historical checkpoint")
    installed = getattr(runtime._executor, "_temporal_restart_state", None)
    schedule = getattr(installed, "program_schedule", None)
    temporal = TemporalRestartState(time_hex=time.hex(), macro_step=macro_step)
    if schedule is not None:
        temporal.configure_program(schedule, time=time, macro_step=macro_step)
    temporal.begin_run(
        run_control_payload(state.step_strategy, state.controls),
        time=time,
        macro_step=macro_step,
    )
    temporal.controller_state = {
        "last_accepted_dt": (
            None if state.last_accepted_dt is None
            else state.last_accepted_dt.hex()
        ),
    }
    from pops.time import ErrorControlledDt

    if type(state.step_strategy) is ErrorControlledDt:
        next_dt = (
            state.step_strategy.dt_init
            if state.last_accepted_dt is None
            else min(
                state.step_strategy.dt_max,
                state.last_accepted_dt * state.step_strategy.growth,
            )
        )
        temporal.queue_error_controlled_proposal(
            dt=next_dt,
            time=time,
            macro_step=macro_step,
        )
    temporal.transaction_stats = dict(state.transaction_stats)
    return temporal.checkpoint_json(time=time, macro_step=macro_step)


def _encode_npz(payload: Mapping[str, Any]) -> bytes:
    import numpy as np

    stream = BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def migrate_uniform_checkpoint(
    source: Any,
    destination: Any,
    *,
    runtime: Any,
    state: UniformCheckpointMigrationState,
) -> CheckpointMigrationReport:
    """Migrate one exact compatible Uniform v2 projection into the current format.

    ``runtime`` must be a bound current Uniform :class:`RuntimeInstance` for the same semantic
    Case/Program, including the same semantic identity and compiled Program hash sealed into the
    projection. It is used only as an authenticated schema/preflight authority; no restart
    transaction begins and no native state is mutated. Runtime loaders never call this function.
    Files produced by the actual unsealed historical v2 writer, and artifacts from an older
    semantic schema, are outside this API's explicit compatibility scope.
    """
    import numpy as np
    from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
    from pops.identity import Identity
    from pops.output._checkpoint_collective import (
        canonical_checkpoint_path,
        decode_checkpoint_bytes,
    )
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        _identity_from_json,
        _runtime_identities,
        _seal_checkpoint_payload_with_identities,
        inspect_checkpoint_payload_integrity,
    )
    from pops.runtime._engine_descriptors import abi_key
    from pops.runtime._runtime_instance import RuntimeInstance

    if type(runtime) is not RuntimeInstance:
        raise TypeError("runtime must be an exact bound RuntimeInstance")
    if type(state) is not UniformCheckpointMigrationState:
        raise TypeError("state must be an exact UniformCheckpointMigrationState")
    layouts = tuple(runtime._layout_plan.layouts)
    if len(layouts) != 1 or layouts[0].adaptive:
        raise NotImplementedError(
            "offline migration currently supports one Uniform layout only; "
            "complete AMR hierarchy migration remains ADC-678")
    if runtime.consumer_graph.nodes:
        raise NotImplementedError(
            "historical v2 checkpoints have no ConsumerGraph cursors; migrate only against "
            "a runtime with an empty ConsumerGraph")
    field_slots = tuple(runtime.field_provider_slots())
    if field_slots:
        raise NotImplementedError(
            "historical v2 checkpoints have no qualified field-provider potentials; "
            "migration requires a runtime with no field providers")

    # Reuse the public checkpoint path contract: a missing or different suffix is normalized to
    # ``.npz`` for both source lookup and destination publication.
    source_path = canonical_checkpoint_path(source)
    destination_path = canonical_checkpoint_path(destination)
    if source_path == destination_path:
        raise ValueError("offline checkpoint migration refuses to overwrite its source")
    raw = source_path.read_bytes()
    source_payload = decode_checkpoint_bytes(raw)
    source_manifest, source_restart = inspect_checkpoint_payload_integrity(
        source_payload, runtime_kind="uniform")
    facts = validate_compatible_v2_projection(source_payload)
    current_semantic, current_artifact, current_bind = _runtime_identities(runtime)
    source_semantic = _identity_from_json(source_manifest["semantic_identity"])
    if source_semantic.token != current_semantic.token:
        raise NotImplementedError(
            "Uniform v2 checkpoint is outside the authenticated compatibility subset: "
            "its semantic identity differs from the bound runtime; an explicit versioned "
            "semantic migration is required")
    if facts["program_hash"] != runtime.installed_program_hash():
        raise NotImplementedError(
            "Uniform v2 checkpoint is outside the authenticated compatibility subset: "
            "its compiled Program hash differs from the bound runtime")

    migrated = {
        name: np.array(source_payload[name], copy=True, order="C")
        for name in source_payload.files
        if name not in {MANIFEST_KEY, IDENTITY_KEY}
    }
    migrated["pops_checkpoint_version"] = np.asarray(
        UNIFORM_CHECKPOINT_PAYLOAD_VERSION, dtype=np.int64)
    migrated["abi_key"] = np.asarray(abi_key())
    migrated["field_provider_slots"] = np.asarray([], dtype="U1")
    if np.asarray(migrated["history_names"]).size == 0:
        migrated["history_names"] = np.asarray([], dtype="U1")
    if np.asarray(migrated["cache_names"]).size == 0:
        migrated["cache_names"] = np.asarray([], dtype="U1")
    migrated["temporal_restart_state"] = np.asarray(
        _current_temporal_json(runtime, facts=facts, state=state))
    for name, depth, policy, stored, initialized in facts["histories"]:
        migrated["history_policy_" + name] = np.asarray(json.dumps(
            policy.to_manifest(), sort_keys=True, separators=(",", ":")))
        fill_count = min(depth, facts["macro_step"]) if initialized else 0
        migrated["history_fill_count_" + name] = np.asarray(
            fill_count, dtype=np.int64)
        migrated["history_requested_stored_slots_" + name] = np.asarray(
            stored, dtype=np.int64)
        migrated["history_storage_mode_" + name] = np.asarray("policy")
    migrated["runtime_consumer_graph"] = np.asarray(
        runtime.consumer_graph.identity.token)
    migrated["runtime_consumer_cursors"] = np.asarray(json.dumps(
        {"schema_version": 1, "rows": []},
        sort_keys=True,
        separators=(",", ":"),
    ))
    migrated["runtime_consumer_diagnostics"] = np.asarray(json.dumps(
        {"schema_version": 2, "baselines": {}, "diagnostics": []},
        sort_keys=True,
        separators=(",", ":"),
    ))
    migration_controls = {} if state.controls is None else dict(state.controls)
    migrated["checkpoint_migration"] = np.asarray(json.dumps({
        "schema_version": 1,
        "protocol": MIGRATION_PROTOCOL,
        "source_restart_identity": source_restart.token,
        "from_payload_version": HISTORICAL_UNIFORM_PAYLOAD_VERSION,
        "explicit_temporal_state": {
            "strategy": state.step_strategy.to_data(),
            "controls": state.step_strategy.runtime_controls_data(migration_controls),
            "last_accepted_dt": (
                None if state.last_accepted_dt is None
                else state.last_accepted_dt.hex()
            ),
            "transaction_stats": dict(state.transaction_stats),
        },
        "scope": {
            "compatibility": MIGRATION_COMPATIBILITY_SCOPE,
            "runtime_kind": "uniform",
            "field_provider_slots": [],
            "consumer_graph": "empty",
        },
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))

    source_run = _identity_from_json(source_manifest["run_identity"])
    if type(source_run) is not Identity or source_run.domain != "run":
        raise ValueError("historical checkpoint has no exact run identity")
    migrated_restart = _seal_checkpoint_payload_with_identities(
        migrated,
        runtime_kind="uniform",
        semantic=current_semantic,
        artifact=current_artifact,
        bind=current_bind,
        run=source_run,
    )
    migrated_bytes = _encode_npz(migrated)

    # These are the exact two current read-only restart preflights.  Neither begins a transaction.
    runtime._inspect_checkpoint_payload(migrated_bytes)
    prepare = getattr(runtime._executor, "_prepare_checkpoint_restart", None)
    if not callable(prepare):
        raise TypeError("bound Uniform executor lacks the strict restart preflight")
    prepare(migrated_bytes)

    # Decode once more after sealing so an invalid/object archive can never reach publication.
    decode_checkpoint_bytes(migrated_bytes)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination_path.name,
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(migrated_bytes)
        os.replace(temporary, destination_path)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return CheckpointMigrationReport(
        source_restart_identity=source_restart.token,
        migrated_restart_identity=migrated_restart.token,
        from_payload_version=HISTORICAL_UNIFORM_PAYLOAD_VERSION,
        to_payload_version=UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        destination=str(destination_path),
    )


__all__ = [
    "CheckpointMigrationReport",
    "HISTORICAL_UNIFORM_PAYLOAD_VERSION",
    "MIGRATION_COMPATIBILITY_SCOPE",
    "MIGRATION_PROTOCOL",
    "UniformCheckpointMigrationState",
    "migrate_uniform_checkpoint",
]
