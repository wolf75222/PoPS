"""ADC-667: true Uniform v2 checkpoints migrate only through explicit offline authority."""

from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pops
from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
from pops._checkpoint_migration_protocol import _CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS
from pops.codegen.checkpoint_migration import (
    UNIFORM_V2_AUTHORITY_TRANSFERS,
    UniformV2BlockMapping,
    UniformV2HistoryMapping,
    UniformV2MigrationMapping,
    migrate_uniform_v2_checkpoint,
)
from pops.output._checkpoint_collective import decode_checkpoint_bytes
from pops.runtime._checkpoint_manifest import (
    IDENTITY_KEY,
    MANIFEST_KEY,
    inspect_checkpoint_payload_integrity,
    seal_checkpoint_payload,
)
from pops.runtime._checkpoint_resource_budget import (
    _producer_checkpoint_resource_budget,
    _reviewed_archive_checkpoint_resource_budget,
)
from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
from pops.solvers.elliptic import CartesianCG
from pops.time import FixedDt
from tests.python.integration._final_field_program import (
    passive_field_model,
    resolve_periodic_field_program,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.native_loader.test_uniform_restart_missing_history import (
    DT,
    _bound_history_runtime,
)


ROOT = Path(__file__).resolve().parents[4]
FROZEN_UNIFORM_V2_B64 = ROOT / "tests/data/adc667/uniform_v2_ab2_98b7ffe6.npz.b64"
FROZEN_UNIFORM_V2_SHA256 = "82490ddc97dbf37e6431c3c0ddb61c30439bdf4df9166f659146634d27766226"
_REVIEWED_RESOURCE_LIMITS = {
    "max_members": 256,
    "max_manifest_characters": 1 << 20,
    "max_array_bytes": 1 << 24,
    "max_uncompressed_bytes": 1 << 26,
    "max_archive_bytes": 1 << 26,
}


def _decode_reviewed(raw, *, unsealed=False):
    digest = hashlib.sha256(raw).hexdigest()
    budget = _reviewed_archive_checkpoint_resource_budget(
        content_sha256=digest,
        runtime_kind="uniform",
        **_REVIEWED_RESOURCE_LIMITS,
    )
    return decode_checkpoint_bytes(raw, budget, allow_reviewed_unsealed=unsealed)


def test_checkpoint_migration_provenance_has_a_fixed_budgeted_envelope():
    from pops.identity import make_identity

    mapping = {"reviewed_mapping_id": "test-envelope"}
    provenance = json.dumps(
        {
            "protocol": "pops.uniform-checkpoint-v2-offline-migration.v1",
            "mapping_identity": make_identity("uniform-v2-migration-map", mapping).token,
            "mapping": mapping,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = {
        "state": np.zeros((1,), dtype=np.float64),
        "checkpoint_migration": np.asarray(provenance),
    }
    budget = _producer_checkpoint_resource_budget(
        payload,
        runtime_kind="uniform",
        authority="test-migration-provenance",
    )
    assert budget.max_members == 2
    assert budget.max_archive_bytes > budget.max_uncompressed_bytes
    near_capacity = dict(payload)
    near_mapping = {
        "reviewed_mapping_id": "x" * (_CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS // 2)
    }
    near_capacity["checkpoint_migration"] = np.asarray(
        json.dumps(
            {
                "protocol": "pops.uniform-checkpoint-v2-offline-migration.v1",
                "mapping_identity": make_identity(
                    "uniform-v2-migration-map", near_mapping
                ).token,
                "mapping": near_mapping,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    near_capacity_budget = _producer_checkpoint_resource_budget(
        near_capacity,
        runtime_kind="uniform",
        authority="test-migration-provenance",
    )
    assert near_capacity_budget == budget
    payload["checkpoint_migration"] = np.asarray(
        "x" * (_CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS + 1)
    )
    with pytest.raises(ValueError, match="fixed character capacity"):
        _producer_checkpoint_resource_budget(
            payload,
            runtime_kind="uniform",
            authority="test-migration-provenance",
        )


def _migration_provenance(mapping=None):
    from pops.identity import make_identity

    mapping = {"reviewed_mapping_id": "test-provenance"} if mapping is None else mapping
    return {
        "protocol": "pops.uniform-checkpoint-v2-offline-migration.v1",
        "mapping_identity": make_identity("uniform-v2-migration-map", mapping).token,
        "mapping": mapping,
    }


def _strict_uniform_preflight_payload():
    return {
        "pops_checkpoint_version": np.asarray(
            UNIFORM_CHECKPOINT_PAYLOAD_VERSION, dtype=np.int64
        ),
        "t": np.asarray(0.0, dtype=np.float64),
        "macro_step": np.asarray(0, dtype=np.int64),
        "pops_spatial_contract": np.asarray("{}"),
        "pops_embedded_boundary_contract": np.asarray("{}"),
        "program_hash": np.asarray("ab" * 32),
        "history_names": np.asarray([], dtype="U1"),
        "cache_slots": np.asarray([], dtype=np.int64),
        "cache_plan_schema": np.asarray("program-resource-plan:v1"),
        "cache_plan_digest": np.asarray("a" * 64),
        "cache_names": np.asarray([], dtype="U1"),
        "cache_valid": np.asarray([], dtype=np.bool_),
        "cache_cold": np.asarray([], dtype=np.bool_),
        "temporal_restart_state": np.asarray("{}"),
        "program_cadence_substeps": np.asarray(1, dtype=np.int64),
        "program_cadence_stride": np.asarray(1, dtype=np.int64),
        "program_cadence_window_steps": np.asarray(0, dtype=np.int64),
        "program_cadence_window_dt": np.asarray(0.0, dtype=np.float64),
        "program_cadence_window_start_time": np.asarray(0.0, dtype=np.float64),
        "program_last_dt": np.asarray(0.0, dtype=np.float64),
    }


def test_checkpoint_migration_provenance_is_consumed_by_producer_and_restart():
    payload = _strict_uniform_preflight_payload()
    preflight_uniform_restart(payload)
    payload["checkpoint_migration"] = np.asarray(
        json.dumps(_migration_provenance(), sort_keys=True, separators=(",", ":"))
    )
    _producer_checkpoint_resource_budget(
        payload,
        runtime_kind="uniform",
        authority="test-migration-provenance",
    )
    preflight_uniform_restart(payload)

    cases = (
        (lambda record: {key: value for key, value in record.items() if key != "mapping"}, "exactly"),
        (
            lambda record: {**record, "mapping": {"reviewed_mapping_id": "altered"}},
            "does not match",
        ),
        (lambda record: {**record, "unreviewed": True}, "exactly"),
    )
    for mutate, match in cases:
        invalid = dict(payload)
        invalid["checkpoint_migration"] = np.asarray(
            json.dumps(mutate(_migration_provenance()), sort_keys=True, separators=(",", ":"))
        )
        with pytest.raises((TypeError, ValueError), match=match):
            _producer_checkpoint_resource_budget(
                invalid,
                runtime_kind="uniform",
                authority="test-migration-provenance",
            )
        with pytest.raises((TypeError, ValueError), match=match):
            preflight_uniform_restart(invalid)

    payload["unreviewed_archive_member"] = np.asarray("forbidden")
    with pytest.raises(ValueError, match="unknown archive members"):
        preflight_uniform_restart(payload)


def _write_source(tmp_path):
    path = tmp_path / "legacy-v2.npz"
    raw = base64.b64decode(
        b"".join(FROZEN_UNIFORM_V2_B64.read_bytes().split()),
        validate=True,
    )
    assert hashlib.sha256(raw).hexdigest() == FROZEN_UNIFORM_V2_SHA256
    path.write_bytes(raw)
    budget = _reviewed_archive_checkpoint_resource_budget(
        content_sha256=FROZEN_UNIFORM_V2_SHA256,
        runtime_kind="uniform",
        **_REVIEWED_RESOURCE_LIMITS,
    )
    return path, decode_checkpoint_bytes(raw, budget, allow_reviewed_unsealed=True)


@pytest.mark.parametrize("replacement", [1.0, -0.0, np.nan])
def test_frozen_v2_fixture_refuses_noncanonical_legacy_phi(replacement, tmp_path):
    from pops.codegen._checkpoint_migration_uniform_v2 import _validate_legacy_v2

    _path, source = _write_source(tmp_path)
    payload = {name: np.array(source[name], copy=True) for name in source.files}
    payload["phi"].flat[0] = replacement
    with pytest.raises(ValueError, match=r"canonical \+0.0"):
        _validate_legacy_v2(payload)


def _write_authority(tmp_path, native_cxx, *, history_slot_dt=(0.01, 0.01)):
    """Capture the authority from the real bound history runtime, never a Python imitation."""
    runtime = _bound_history_runtime(native_cxx)
    report = pops.run(runtime, t_end=0.03, max_steps=3)
    assert report.accepted_steps == 3
    assert report.final_time == 3 * DT
    path = Path(runtime.checkpoint(tmp_path / "authority-v9.npz"))
    if history_slot_dt != (0.01, 0.01):
        with np.load(path, allow_pickle=False) as stored:
            payload = {
                name: np.asarray(stored[name]).copy()
                for name in stored.files
                if name not in {MANIFEST_KEY, IDENTITY_KEY}
            }
        history = tuple(str(name) for name in payload["history_names"])
        assert len(history) == 1
        payload["history_slot_dt_" + history[0]] = np.asarray(history_slot_dt, dtype=np.float64)
        restart = seal_checkpoint_payload(runtime, payload, runtime_kind="uniform")
        np.savez_compressed(path, **payload)
    else:
        authority_payload = decode_checkpoint_bytes(
            path.read_bytes(), runtime._checkpoint_resource_budget
        )
        slots = np.asarray(authority_payload["field_provider_slots"])
        assert slots.shape == (0,)
        assert slots.dtype.kind in "US"
        _, restart = inspect_checkpoint_payload_integrity(authority_payload, runtime_kind="uniform")
        histories = tuple(str(name) for name in authority_payload["history_names"])
        assert histories == ("blk.rate",)
        history = histories[0]
        assert int(authority_payload["history_depth_" + history]) == 2
        assert int(authority_payload["history_fill_count_" + history]) == 2
        slot_dt = np.asarray(authority_payload["history_slot_dt_" + history])
        assert slot_dt.dtype == np.dtype(np.float64)
        assert np.array_equal(slot_dt, np.asarray((DT, DT), dtype=np.float64))
    return path, runtime, restart, None


@pytest.mark.compiler
@pytest.mark.native_loader
def test_dense_uniform_authority_refuses_spurious_regrid_replay_schedule(
    tmp_path, native_cxx
):
    from pops.codegen._checkpoint_migration_uniform_v2 import _current_authority

    authority, owner, _restart, _ = _write_authority(tmp_path, native_cxx)
    decoded = decode_checkpoint_bytes(authority.read_bytes(), owner._checkpoint_resource_budget)
    payload = {
        name: np.asarray(decoded[name]).copy()
        for name in decoded.files
        if name not in {MANIFEST_KEY, IDENTITY_KEY}
    }
    histories = tuple(str(name) for name in payload["history_names"])
    assert histories == ("blk.rate",)
    payload["history_regrid_steps_" + histories[0]] = np.asarray((1,), dtype=np.int64)
    seal_checkpoint_payload(owner, payload, runtime_kind="uniform")

    with pytest.raises(ValueError, match="Dense history .* spurious replay schedule"):
        _current_authority(payload)


def _mapping(source_payload, owner, restart, authority):
    authority_payload = decode_checkpoint_bytes(
        authority.read_bytes(), owner._checkpoint_resource_budget
    )


    source_blocks = tuple(str(name) for name in source_payload["blocks"])
    authority_blocks = tuple(str(name) for name in authority_payload["blocks"])
    source_histories = tuple(str(name) for name in source_payload["history_names"])
    authority_histories = tuple(str(name) for name in authority_payload["history_names"])
    assert source_blocks == ("blk",) and len(authority_blocks) == 1
    assert source_histories == ("blk.R",) and len(authority_histories) == 1
    semantic, artifact, bind = owner._checkpoint_identities()
    auxiliary_checkpoint = np.asarray(authority_payload["auxiliary_checkpoint"])
    from pops._native_selector import selected_native_module

    attest = getattr(
        selected_native_module(required=True),
        "_attest_empty_uniform_auxiliary_checkpoint",
        None,
    )
    assert callable(attest)
    attestation = attest(auxiliary_checkpoint.tobytes())
    assert type(attestation["registry_contract"]) is bytes
    assert type(attestation["accepted_generation"]) is int
    assert not isinstance(attestation["accepted_generation"], bool)
    assert 0 <= attestation["accepted_generation"] < (1 << 64) - 1
    return UniformV2MigrationMapping(
        reviewed_mapping_id="ADC-667-frozen-uniform-v2-to-current-v9",
        source_content_sha256=FROZEN_UNIFORM_V2_SHA256,
        source_max_members=_REVIEWED_RESOURCE_LIMITS["max_members"],
        source_max_manifest_characters=_REVIEWED_RESOURCE_LIMITS["max_manifest_characters"],
        source_max_array_bytes=_REVIEWED_RESOURCE_LIMITS["max_array_bytes"],
        source_max_uncompressed_bytes=_REVIEWED_RESOURCE_LIMITS["max_uncompressed_bytes"],
        source_max_archive_bytes=_REVIEWED_RESOURCE_LIMITS["max_archive_bytes"],
        source_abi_key=str(source_payload["abi_key"]),
        source_program_hash=str(source_payload["program_hash"]),
        authority_content_sha256=hashlib.sha256(authority.read_bytes()).hexdigest(),
        authority_auxiliary_checkpoint_sha256=hashlib.sha256(
            auxiliary_checkpoint.tobytes()
        ).hexdigest(),
        authority_auxiliary_registry_contract_sha256=hashlib.sha256(
            attestation["registry_contract"]
        ).hexdigest(),
        authority_max_members=_REVIEWED_RESOURCE_LIMITS["max_members"],
        authority_max_manifest_characters=_REVIEWED_RESOURCE_LIMITS["max_manifest_characters"],
        authority_max_array_bytes=_REVIEWED_RESOURCE_LIMITS["max_array_bytes"],
        authority_max_uncompressed_bytes=_REVIEWED_RESOURCE_LIMITS["max_uncompressed_bytes"],
        authority_max_archive_bytes=_REVIEWED_RESOURCE_LIMITS["max_archive_bytes"],
        authority_restart_identity=restart.token,
        target_semantic_identity=semantic.token,
        target_artifact_identity=artifact.token,
        target_bind_identity=bind.token,
        target_run_identity=owner.last_run_identity.token,
        target_abi_key=str(authority_payload["abi_key"]),
        target_program_hash=str(authority_payload["program_hash"]),
        authority_transfers=UNIFORM_V2_AUTHORITY_TRANSFERS,
        blocks=(UniformV2BlockMapping("blk", authority_blocks[0], (("rho", "rho"),)),),
        histories=(
            UniformV2HistoryMapping("blk.R", authority_histories[0], ((0, 0),)),
        ),
    )


def _captured_nonempty_auxiliary_checkpoint(native_cxx):
    """Capture a real registered FieldOutput image; this test never fabricates POPSAUX2."""
    from pops.lib.time import ForwardEuler
    from pops.time import FailRun

    def program(state, rate, field):
        result = ForwardEuler(state, rate=rate, fields=field, solve_action=FailRun())
        result.step_strategy(FixedDt(1.0e-4))
        return result

    model = passive_field_model("migration-nonempty-auxiliary", coefficient=0.0)
    resolved = resolve_periodic_field_program(
        model,
        program,
        name="migration-nonempty-auxiliary",
        block_name="material",
        target="system",
        n=4,
        field_solver=CartesianCG(),
        cxx=native_cxx,
        include=str(ROOT / "include"),
    )
    artifact = pops.compile(resolved)
    simulation = pops.bind(
        artifact,
        initial_state={"material": np.ones((1, 4, 4), dtype=np.float64)},
        resources={"execution_context": artifact_execution_context(artifact)},
    )
    payload = simulation._executor._s.capture_auxiliary_checkpoint_accepted_state()
    assert type(payload) is bytes and payload.startswith(b"POPSAUX2")
    return payload


@pytest.mark.compiler
@pytest.mark.native_loader
def test_true_frozen_v2_migrates_and_strict_uniform_restart_accepts(tmp_path, native_cxx):
    source, source_payload = _write_source(tmp_path)
    authority, owner, authority_restart, _ = _write_authority(tmp_path, native_cxx)
    authority_bytes = authority.read_bytes()
    destination = tmp_path / "migrated-v8.npz"

    report = migrate_uniform_v2_checkpoint(
        source,
        destination,
        current_authority=authority,
        mapping=_mapping(source_payload, owner, authority_restart, authority),
    )

    assert source.read_bytes() == base64.b64decode(
        b"".join(FROZEN_UNIFORM_V2_B64.read_bytes().split()),
        validate=True,
    )
    assert authority.read_bytes() == authority_bytes
    migrated = _decode_reviewed(destination.read_bytes())
    _, restart = inspect_checkpoint_payload_integrity(migrated, runtime_kind="uniform")
    assert report.destination_restart_identity == restart.token
    assert int(migrated["pops_checkpoint_version"]) == UNIFORM_CHECKPOINT_PAYLOAD_VERSION
    authority_payload = decode_checkpoint_bytes(authority_bytes, owner._checkpoint_resource_budget)
    assert str(migrated["program_hash"]) == str(authority_payload["program_hash"])
    assert np.array_equal(
        migrated["auxiliary_checkpoint"], authority_payload["auxiliary_checkpoint"]
    )
    assert np.array_equal(migrated["state_blk"], source_payload["state_blk"])
    assert np.array_equal(migrated["phi"], source_payload["phi"])
    assert migrated["phi"].dtype == np.dtype(np.float64)
    assert migrated["phi"].flags.c_contiguous
    assert np.all(migrated["phi"].view(np.uint64) == 0)
    for slot in (0, 1):
        assert np.array_equal(
            migrated["history_%s_%d" % (str(authority_payload["history_names"][0]), slot)],
            source_payload["history_blk.R_%d" % slot],
        )

    accepted = owner.restart(destination)
    assert accepted.token == report.destination_restart_identity


@pytest.mark.parametrize(
    "change, match",
    [
        (
            lambda mapping: replace(mapping, source_content_sha256="0" * 64),
            "content SHA-256",
        ),
        (
            lambda mapping: replace(mapping, source_program_hash="1" * 64),
            "Program hash",
        ),
        (
            lambda mapping: replace(mapping, source_max_archive_bytes=1),
            "live resource budget",
        ),
        (
            lambda mapping: replace(
                mapping, authority_restart_identity="restart:v1:sha256:" + "0" * 64
            ),
            "authority or target lifecycle pins",
        ),
        (
            lambda mapping: replace(mapping, authority_auxiliary_checkpoint_sha256="0" * 64),
            "auxiliary checkpoint SHA-256",
        ),
        (
            lambda mapping: replace(
                mapping,
                authority_auxiliary_registry_contract_sha256="0" * 64,
            ),
            "auxiliary registry contract",
        ),
        (
            lambda mapping: replace(mapping, blocks=()),
            "block mapping",
        ),
        (
            lambda mapping: replace(
                mapping,
                authority_transfers=mapping.authority_transfers[:-1],
            ),
            "complete authority transfer set",
        ),
    ],
)
@pytest.mark.compiler
@pytest.mark.native_loader
def test_refused_mapping_publishes_no_destination(tmp_path, native_cxx, change, match):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path, native_cxx)
    destination = tmp_path / "must-not-exist.npz"
    source_bytes = source.read_bytes()
    authority_bytes = authority.read_bytes()

    with pytest.raises(ValueError, match=match):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=change(_mapping(source_payload, owner, restart, authority)),
        )
    assert not destination.exists()
    assert source.read_bytes() == source_bytes
    assert authority.read_bytes() == authority_bytes


@pytest.mark.compiler
@pytest.mark.native_loader
def test_ambiguous_legacy_schema_is_refused_without_publication(tmp_path, native_cxx):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path, native_cxx)
    arrays = {name: np.array(value, copy=True) for name, value in source_payload.items()}
    arrays["implicit_alias"] = np.asarray("forbidden")
    with open(source, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    mapping = replace(
        _mapping(source_payload, owner, restart, authority),
        source_content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    destination = tmp_path / "must-not-exist.npz"

    with pytest.raises(ValueError, match="ambiguous or unknown keys"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=mapping,
        )
    assert not destination.exists()


@pytest.mark.compiler
@pytest.mark.native_loader
def test_tampered_current_authority_is_refused_without_publication(tmp_path, native_cxx):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path, native_cxx)
    mapping = _mapping(source_payload, owner, restart, authority)
    authority_payload = _decode_reviewed(authority.read_bytes())
    damaged = {name: np.array(value, copy=True) for name, value in authority_payload.items()}
    damaged["state_blk"][0, 0, 0] = 1.0
    with open(authority, "wb") as stream:
        np.savez_compressed(stream, **damaged)
    destination = tmp_path / "must-not-exist.npz"

    with pytest.raises(ValueError, match="digest mismatch"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=mapping,
        )
    assert not destination.exists()


@pytest.mark.compiler
@pytest.mark.native_loader
def test_resealed_tampered_auxiliary_authority_is_refused_without_publication(
    tmp_path, native_cxx
):
    source, source_payload = _write_source(tmp_path)
    authority, owner, authority_restart, _ = _write_authority(tmp_path, native_cxx)
    mapping = _mapping(source_payload, owner, authority_restart, authority)
    with np.load(authority, allow_pickle=False) as stored:
        payload = {
            name: np.asarray(stored[name]).copy()
            for name in stored.files
            if name not in {MANIFEST_KEY, IDENTITY_KEY}
        }
    auxiliary = np.asarray(payload["auxiliary_checkpoint"]).copy()
    auxiliary[-1] ^= np.uint8(1)
    payload["auxiliary_checkpoint"] = auxiliary
    restart = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
    np.savez_compressed(authority, **payload)
    source_bytes = source.read_bytes()
    authority_bytes = authority.read_bytes()
    destination = tmp_path / "must-not-exist.npz"
    mapping = replace(
        mapping,
        authority_content_sha256=hashlib.sha256(authority_bytes).hexdigest(),
        authority_auxiliary_checkpoint_sha256=hashlib.sha256(auxiliary.tobytes()).hexdigest(),
        authority_restart_identity=restart.token,
    )

    with pytest.raises((RuntimeError, ValueError), match="auxiliary"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=mapping,
        )
    assert not destination.exists()
    assert source.read_bytes() == source_bytes
    assert authority.read_bytes() == authority_bytes


@pytest.mark.compiler
@pytest.mark.native_loader
def test_resealed_nonempty_auxiliary_authority_is_refused_without_publication(
    tmp_path, native_cxx
):
    source, source_payload = _write_source(tmp_path)
    authority, owner, authority_restart, _ = _write_authority(tmp_path, native_cxx)
    mapping = _mapping(source_payload, owner, authority_restart, authority)
    nonempty = _captured_nonempty_auxiliary_checkpoint(native_cxx)
    with np.load(authority, allow_pickle=False) as stored:
        payload = {
            name: np.asarray(stored[name]).copy()
            for name in stored.files
            if name not in {MANIFEST_KEY, IDENTITY_KEY}
        }
    payload["auxiliary_checkpoint"] = np.frombuffer(nonempty, dtype=np.uint8).copy()
    restart = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
    np.savez_compressed(authority, **payload)
    source_bytes = source.read_bytes()
    authority_bytes = authority.read_bytes()
    destination = tmp_path / "must-not-exist.npz"
    mapping = replace(
        mapping,
        authority_content_sha256=hashlib.sha256(authority_bytes).hexdigest(),
        authority_auxiliary_checkpoint_sha256=hashlib.sha256(nonempty).hexdigest(),
        authority_restart_identity=restart.token,
    )

    with pytest.raises((RuntimeError, ValueError), match="auxiliary"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=mapping,
        )
    assert not destination.exists()
    assert source.read_bytes() == source_bytes
    assert authority.read_bytes() == authority_bytes


@pytest.mark.compiler
@pytest.mark.native_loader
def test_authority_history_ledger_must_match_the_legacy_trajectory(tmp_path, native_cxx):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(
        tmp_path, native_cxx,
        history_slot_dt=(0.02, 0.02),
    )
    destination = tmp_path / "must-not-exist.npz"

    with pytest.raises(ValueError, match="outgoing-dt ledger differs"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=_mapping(source_payload, owner, restart, authority),
        )
    assert not destination.exists()


@pytest.mark.compiler
@pytest.mark.native_loader
def test_existing_destination_is_never_replaced(tmp_path, native_cxx):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path, native_cxx)
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="refuses to replace"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=_mapping(source_payload, owner, restart, authority),
        )
    assert destination.read_bytes() == b"sentinel"


@pytest.mark.compiler
@pytest.mark.native_loader
def test_destination_race_at_atomic_link_is_no_clobber(tmp_path, monkeypatch, native_cxx):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path, native_cxx)
    destination = tmp_path / "raced.npz"
    real_link = os.link

    def publish_race(temporary, target):
        Path(target).write_bytes(b"concurrent-writer")
        return real_link(temporary, target)

    monkeypatch.setattr(os, "link", publish_race)
    with pytest.raises(FileExistsError):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=_mapping(source_payload, owner, restart, authority),
        )
    assert destination.read_bytes() == b"concurrent-writer"
    assert not tuple(tmp_path.glob(".raced.npz.*.tmp"))
