"""ADC-667: true Uniform v2 checkpoints migrate only through explicit offline authority."""

from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
from pops.codegen.checkpoint_migration import (
    UNIFORM_V2_AUTHORITY_TRANSFERS,
    UniformV2BlockMapping,
    UniformV2HistoryMapping,
    UniformV2MigrationMapping,
    migrate_uniform_v2_checkpoint,
)
from pops.identity import make_identity
from pops.mesh._layout_plan_contracts import NativeSpatialLayout
from pops.output._checkpoint_collective import decode_checkpoint_bytes
from pops.runtime._checkpoint_embedded_boundary import (
    CheckpointEmbeddedBoundaryContract,
    add_checkpoint_embedded_boundary_contract,
)
from pops.runtime._checkpoint_manifest import (
    inspect_checkpoint_payload_integrity,
    seal_checkpoint_payload,
)
from pops.runtime._checkpoint_spatial import (
    add_checkpoint_spatial_contract,
    install_checkpoint_spatial_contract,
)
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import Clock, FixedDt, TimePoint
from pops.time._history.persistence import Dense


ROOT = Path(__file__).resolve().parents[4]
FROZEN_UNIFORM_V2_B64 = ROOT / "tests/data/adc667/uniform_v2_ab2_98b7ffe6.npz.b64"
FROZEN_UNIFORM_V2_SHA256 = "82490ddc97dbf37e6431c3c0ddb61c30439bdf4df9166f659146634d27766226"
TARGET_PROGRAM_HASH = hashlib.sha256(b"adc667-current-program").hexdigest()
TARGET_ABI_KEY = "adc667-current-test-abi"


def _native_layout():
    return NativeSpatialLayout(
        layout_id="case:adc667/layout:grid",
        coordinate_system="pops://coordinates/cartesian-nd@1",
        cell_measure="pops://measures/cartesian-cell@1",
        axis_names=("x", "y"),
        shape=(4, 4),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        periodicity=(True, True),
        centering="cell",
        decomposition={"kind": "single_box", "shape": [4, 4]},
    )


def _schedule():
    macro = Clock("macro")
    block = {"kind": "block", "qualified_id": "case/blk"}
    state = {"kind": "state", "qualified_id": "case/blk/U", "block_ref": block}
    space = {"kind": "state", "name": "U", "components": ["rho"]}
    interpolation = {"kind": "linear", "schema_version": 1, "minimum_samples": 2}
    validity = {
        "schema_version": 1,
        "oldest": TimePoint(macro, step=-1).to_data(),
        "newest": TimePoint(macro).to_data(),
    }
    return {
        "schema_version": 1,
        "kind": "pops.temporal-program-schedule",
        "primary_clock": macro.qualified_id,
        "clocks": [{"id": macro.qualified_id, "descriptor": macro.to_data(), "ticks_per_macro": 1}],
        "subcycles": [],
        "synchronizations": [],
        "schedules": [],
        "histories": [
            {
                "name": "blk.R",
                "owner": block,
                "state": state,
                "space": space,
                "clock": macro.qualified_id,
                "depth": 1,
                "ring_slots": 2,
                "ncomp": 1,
                "validity": validity,
                "interpolation": interpolation,
                "checkpoint_policy": None,
            }
        ],
    }


def _temporal_json(schedule):
    state = TemporalRestartState()
    state.configure_program(schedule, time=0.0, macro_step=0)
    strategy = FixedDt(0.01)
    state.begin_run(
        {"strategy": strategy.to_data(), "controls": {}},
        time=0.0,
        macro_step=0,
    )
    before = 0.0
    for step, now in enumerate((0.01, 0.02, 0.03), start=1):
        state.accept(
            before_time=before,
            before_step=step - 1,
            time=now,
            macro_step=step,
        )
        before = now
    return state.checkpoint_json(time=0.03, macro_step=3)


class _AuthorityOwner:
    def __init__(self):
        self.identities = (
            make_identity("semantic", {"case": "adc667"}),
            make_identity("artifact", {"case": "adc667"}),
            make_identity("bind", {"case": "adc667"}),
        )
        self.last_run_identity = make_identity("run", {"case": "adc667"})

    def _checkpoint_identities(self):
        return self.identities


def _write_source(tmp_path):
    path = tmp_path / "legacy-v2.npz"
    raw = base64.b64decode(
        b"".join(FROZEN_UNIFORM_V2_B64.read_bytes().split()),
        validate=True,
    )
    assert hashlib.sha256(raw).hexdigest() == FROZEN_UNIFORM_V2_SHA256
    path.write_bytes(raw)
    return path, decode_checkpoint_bytes(raw)


def _write_authority(tmp_path, *, history_slot_dt=(0.01, 0.01)):
    from pops.output._consumer_contracts import ConsumerGraph

    schedule = _schedule()
    payload = {
        "pops_checkpoint_version": UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        "t": 0.03,
        "macro_step": 3,
        "abi_key": TARGET_ABI_KEY,
        "program_hash": TARGET_PROGRAM_HASH,
        "blocks": np.asarray(["blk"]),
        "ncomp_blk": 1,
        "names_blk": np.asarray(["rho"]),
        "state_blk": np.zeros((1, 4, 4), dtype=np.float64),
        "phi": np.zeros((4, 4), dtype=np.float64),
        "field_provider_slots": np.asarray([], dtype=str),
        "temporal_restart_state": np.asarray(_temporal_json(schedule)),
        "program_cadence_substeps": 1,
        "program_cadence_stride": 1,
        "program_cadence_window_steps": 0,
        "program_cadence_window_dt": 0.0,
        "program_cadence_window_start_time": 0.0,
        "program_last_dt": 0.01,
        "history_names": np.asarray(["blk.R"]),
        "history_depth_blk.R": 2,
        "history_ncomp_blk.R": 1,
        "history_init_blk.R": True,
        "history_fill_count_blk.R": 2,
        "history_policy_blk.R": np.asarray(
            json.dumps(Dense().to_manifest(), sort_keys=True, separators=(",", ":"))
        ),
        "history_requested_stored_slots_blk.R": np.asarray([0, 1], dtype=np.int64),
        "history_stored_slots_blk.R": np.asarray([0, 1], dtype=np.int64),
        "history_storage_mode_blk.R": np.asarray("policy"),
        "history_slot_dt_blk.R": np.asarray(history_slot_dt, dtype=np.float64),
        "history_blk.R_0": np.zeros((1, 4, 4), dtype=np.float64),
        "history_blk.R_1": np.zeros((1, 4, 4), dtype=np.float64),
        "cache_nodes": np.asarray([], dtype=np.int64),
        "cache_names": np.asarray([], dtype=str),
        "runtime_consumer_graph": np.asarray(ConsumerGraph(()).identity.token),
        "runtime_consumer_cursors": np.asarray(
            json.dumps({"schema_version": 1, "rows": []}, separators=(",", ":"))
        ),
        "runtime_consumer_diagnostics": np.asarray(
            json.dumps(
                {"schema_version": 2, "baselines": {}, "diagnostics": []},
                separators=(",", ":"),
            )
        ),
    }
    owner = _AuthorityOwner()
    spatial = install_checkpoint_spatial_contract(owner, _native_layout())
    add_checkpoint_spatial_contract(payload, spatial)
    add_checkpoint_embedded_boundary_contract(
        payload,
        CheckpointEmbeddedBoundaryContract(2, False, "none", 0.0, 0.0, 0.0, ""),
    )
    restart = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
    path = tmp_path / "authority-v7.npz"
    with open(path, "wb") as stream:
        np.savez_compressed(stream, **payload)
    return path, owner, restart, schedule


def _mapping(source_payload, owner, restart):
    return UniformV2MigrationMapping(
        reviewed_mapping_id="ADC-667-frozen-uniform-v2-to-current-v7",
        source_content_sha256=FROZEN_UNIFORM_V2_SHA256,
        source_abi_key=str(source_payload["abi_key"]),
        source_program_hash=str(source_payload["program_hash"]),
        authority_restart_identity=restart.token,
        target_semantic_identity=owner.identities[0].token,
        target_artifact_identity=owner.identities[1].token,
        target_bind_identity=owner.identities[2].token,
        target_run_identity=owner.last_run_identity.token,
        target_abi_key=TARGET_ABI_KEY,
        target_program_hash=TARGET_PROGRAM_HASH,
        authority_transfers=UNIFORM_V2_AUTHORITY_TRANSFERS,
        blocks=(UniformV2BlockMapping("blk", "blk", (("rho", "rho"),)),),
        histories=(UniformV2HistoryMapping("blk.R", "blk.R", ((0, 0),)),),
    )


class _NativeTarget:
    def __init__(self):
        self.state = np.zeros((1, 4, 4), dtype=np.float64)
        self.potential = np.zeros(16, dtype=np.float64)
        self.histories = {}
        self.history_dt = {}
        self.history_initialized = {}
        self.history_fill = {}
        self.clock = (0.0, 0)
        self.cadence = None
        self.transaction = None

    def program_substeps(self):
        return 1

    def program_stride(self):
        return 1

    def restore_program_cadence_window(self, *values):
        self.cadence = values

    def _prepare_checkpoint_spatial_contract(self, contract):
        return [int(np.prod(contract["shape"], dtype=np.int64))]

    def effective_options_report(self):
        return {
            "topology": {"dimension": 2, "periodicity": [True, True]},
            "eb": {
                "enabled": False,
                "geometry_mode": "none",
                "kappa_min": 0.01,
                "face_open_eps": 1.0e-6,
                "cut_theta_min": 1.0e-3,
                "semantic_digest": "",
                "materialization_digest": "",
                "generation": 0,
            },
        }

    def nx(self):
        return 4

    def ny(self):
        return 4

    def block_names(self):
        return ["blk"]

    def n_vars(self, block):
        assert block == "blk"
        return 1

    def field_provider_slots(self):
        return []

    def installed_program_hash(self):
        return TARGET_PROGRAM_HASH

    def history_names(self):
        return []

    def set_state(self, block, values):
        assert block == "blk"
        self.state = np.array(values, copy=True)

    def set_potential(self, values):
        self.potential = np.array(values, copy=True)

    def set_clock(self, time, macro_step):
        self.clock = (float(time), int(macro_step))

    def restore_history(self, name, slot, values):
        self.histories[name, int(slot)] = np.array(values, copy=True)

    def restore_history_slot_dt(self, name, slot, dt):
        self.history_dt[name, int(slot)] = float(dt)

    def set_history_initialized(self, name, initialized):
        self.history_initialized[name] = bool(initialized)

    def restore_history_fill_count(self, name, fill_count):
        self.history_fill[name] = int(fill_count)

    def _begin_step_transaction(self):
        assert self.transaction is None
        self.transaction = "active"

    def _commit_step_transaction(self):
        assert self.transaction == "active"
        self.transaction = "committed"

    def _finalize_step_transaction(self):
        assert self.transaction == "committed"
        self.transaction = None

    def _rollback_step_transaction(self):
        self.transaction = None


def _strict_uniform_target(owner, schedule, monkeypatch):
    engine_descriptors = ModuleType("pops.runtime._engine_descriptors")
    engine_descriptors.abi_key = lambda: TARGET_ABI_KEY
    monkeypatch.setitem(
        sys.modules,
        "pops.runtime._engine_descriptors",
        engine_descriptors,
    )
    module_name = "_pops_adc667_strict_system_io"
    module_path = ROOT / "python/pops/runtime/_system_io.py"
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    _SystemIO = module._SystemIO

    class _StrictUniformTarget(_SystemIO):
        def __init__(self):
            self._s = _NativeTarget()
            self._execution_context = SimpleNamespace(
                communicator=SimpleNamespace(identity="serial", handle=None)
            )
            self._identities = owner.identities
            self.last_run_identity = owner.last_run_identity
            self._temporal_restart_state = SimpleNamespace(program_schedule=schedule)
            install_checkpoint_spatial_contract(self, _native_layout())

        def _checkpoint_identities(self):
            return self._identities

    return _StrictUniformTarget()


def test_true_frozen_v2_migrates_and_strict_uniform_restart_accepts(tmp_path, monkeypatch):
    source, source_payload = _write_source(tmp_path)
    authority, owner, authority_restart, schedule = _write_authority(tmp_path)
    destination = tmp_path / "migrated-v7.npz"

    report = migrate_uniform_v2_checkpoint(
        source,
        destination,
        current_authority=authority,
        mapping=_mapping(source_payload, owner, authority_restart),
    )

    assert source.read_bytes() == base64.b64decode(
        b"".join(FROZEN_UNIFORM_V2_B64.read_bytes().split()),
        validate=True,
    )
    migrated = decode_checkpoint_bytes(destination.read_bytes())
    _, restart = inspect_checkpoint_payload_integrity(migrated, runtime_kind="uniform")
    assert report.destination_restart_identity == restart.token
    assert int(migrated["pops_checkpoint_version"]) == 7
    assert str(migrated["program_hash"]) == TARGET_PROGRAM_HASH
    assert np.array_equal(migrated["state_blk"], source_payload["state_blk"])
    assert np.array_equal(migrated["phi"], source_payload["phi"])
    for slot in (0, 1):
        assert np.array_equal(
            migrated["history_blk.R_%d" % slot],
            source_payload["history_blk.R_%d" % slot],
        )

    target = _strict_uniform_target(owner, schedule, monkeypatch)
    accepted = target.restart(destination, bit_identical=True)
    assert accepted.token == report.destination_restart_identity
    assert np.array_equal(target._s.state, source_payload["state_blk"])
    assert target._s.clock == (0.03, 3)
    assert target._s.history_fill == {"blk.R": 2}


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
            lambda mapping: replace(
                mapping, authority_restart_identity="restart:v1:sha256:" + "0" * 64
            ),
            "authority or target lifecycle pins",
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
def test_refused_mapping_publishes_no_destination(tmp_path, change, match):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path)
    destination = tmp_path / "must-not-exist.npz"

    with pytest.raises(ValueError, match=match):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=change(_mapping(source_payload, owner, restart)),
        )
    assert not destination.exists()


def test_ambiguous_legacy_schema_is_refused_without_publication(tmp_path):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path)
    arrays = {name: np.array(value, copy=True) for name, value in source_payload.items()}
    arrays["implicit_alias"] = np.asarray("forbidden")
    with open(source, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    mapping = replace(
        _mapping(source_payload, owner, restart),
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


def test_tampered_current_authority_is_refused_without_publication(tmp_path):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path)
    authority_payload = decode_checkpoint_bytes(authority.read_bytes())
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
            mapping=_mapping(source_payload, owner, restart),
        )
    assert not destination.exists()


def test_authority_history_ledger_must_match_the_legacy_trajectory(tmp_path):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(
        tmp_path,
        history_slot_dt=(0.02, 0.02),
    )
    destination = tmp_path / "must-not-exist.npz"

    with pytest.raises(ValueError, match="outgoing-dt ledger differs"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=_mapping(source_payload, owner, restart),
        )
    assert not destination.exists()


def test_existing_destination_is_never_replaced(tmp_path):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path)
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="refuses to replace"):
        migrate_uniform_v2_checkpoint(
            source,
            destination,
            current_authority=authority,
            mapping=_mapping(source_payload, owner, restart),
        )
    assert destination.read_bytes() == b"sentinel"


def test_destination_race_at_atomic_link_is_no_clobber(tmp_path, monkeypatch):
    source, source_payload = _write_source(tmp_path)
    authority, owner, restart, _ = _write_authority(tmp_path)
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
            mapping=_mapping(source_payload, owner, restart),
        )
    assert destination.read_bytes() == b"concurrent-writer"
    assert not tuple(tmp_path.glob(".raced.npz.*.tmp"))
