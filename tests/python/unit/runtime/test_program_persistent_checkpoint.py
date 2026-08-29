"""Lossless Python projection of the native Program persistent checkpoint carrier."""

from __future__ import annotations

import dataclasses
import io
from types import SimpleNamespace

import numpy as np
import pytest

from pops.identity import make_identity
from pops.output._checkpoint_contract import (
    PROGRAM_PERSISTENT_CHECKPOINT_KEY,
    PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA,
    PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY,
    PROGRAM_PERSISTENT_PLAN_DIGEST_KEY,
    PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY,
    PROGRAM_PERSISTENT_PLAN_SCHEMA,
    PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY,
    PROGRAM_PERSISTENT_SLOT_COUNT_KEY,
    CheckpointResourceBudget,
    PreparedProgramPersistentValueRestore,
    ProgramPersistentValueCheckpoint,
    capture_program_persistent_value_checkpoint,
    decode_program_persistent_value_checkpoint,
    encode_program_persistent_value_checkpoint,
    prepare_program_persistent_value_restore,
    program_persistent_checkpoint_from_payload,
    program_persistent_checkpoint_manifest_from_payload,
    publish_program_persistent_value_restore,
    require_program_persistent_checkpoint_plan,
    require_program_persistent_value_checkpoint_capture,
)
from pops.output._checkpoint_collective import decode_checkpoint_bytes
from pops.runtime._checkpoint_manifest import (
    _seal_checkpoint_payload_with_identities,
    inspect_checkpoint_payload_integrity,
)


def _row(*, slot=0, path="root/0", path_id=1, **changes):
    row = {
        "slot": slot,
        "key": {
            "value_id": 17,
            "occurrence_path_id": path_id,
            "owner": 2,
            "space": 3,
            "clock": 4,
            "level": None,
        },
        "identity": "program-resource:v1:row-%d" % slot,
        "occurrence_path": path,
        "owner_identity": "fluid",
        "space_identity": "cell-centered",
        "clock_identity": "accepted-step",
        "lifetime": 2,
        "centering": 1,
        "off_policy": 1,
        "spatial_transfer": 1,
        "components": 1,
        "ghosts": 0,
        "bytes": 8,
        "maximum_bytes": 16,
        "communicates": False,
        "restart_required": False,
        "communication": "none",
        "transfer_identity": "none",
        "restart_identity": "none",
        "component_names": "[\"value\"]",
        "shape": "[16]",
        "cells": 2,
        "itemsize": 8,
    }
    row.update(changes)
    return row


def _image(*rows, maximum_bytes=None, metadata=None):
    rows = tuple(rows or (_row(),))
    if maximum_bytes is None:
        maximum_bytes = sum(row["maximum_bytes"] for row in rows)
    if metadata is None:
        metadata = tuple(
            {
                "accepted_coordinate": 9,
                "cursor": 4,
                "accumulated_dt": 0.375,
                "topology_epoch": 11,
                "layout_generation": 7,
                "valid": False,
                "cold": True,
            }
            for _row_value in rows
        )
    storage = b"".join(bytes([index + 1]) * row["maximum_bytes"] for index, row in enumerate(rows))
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + row["maximum_bytes"])
    return ProgramPersistentValueCheckpoint(
        bound=True,
        schema=PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA,
        plan_schema=PROGRAM_PERSISTENT_PLAN_SCHEMA,
        plan_digest="a" * 64,
        maximum_bytes=maximum_bytes,
        slot_count=len(rows),
        rows=tuple(rows),
        metadata=tuple(metadata),
        offsets=tuple(offsets),
        value_bytes=tuple(
            row["bytes"] if slot_metadata["valid"] else 0
            for row, slot_metadata in zip(rows, metadata, strict=True)
        ),
        storage=storage,
    )


def _payload(image):
    encoded = encode_program_persistent_value_checkpoint(image)
    return {
        PROGRAM_PERSISTENT_CHECKPOINT_KEY: np.frombuffer(encoded, dtype=np.uint8).copy(),
        PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY: np.asarray(image.schema),
        PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY: np.asarray(image.plan_schema),
        PROGRAM_PERSISTENT_PLAN_DIGEST_KEY: np.asarray(image.plan_digest),
        PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY: np.asarray(image.maximum_bytes, dtype=np.uint64),
        PROGRAM_PERSISTENT_SLOT_COUNT_KEY: np.asarray(image.slot_count, dtype=np.uint32),
    }


def _budget(image, *, digest=None, maximum_bytes=None, slot_count=None):
    return CheckpointResourceBudget(
        "amr",
        100,
        100_000,
        1_000_000,
        2_000_000,
        3_000_000,
        "test-program-plan",
        program_resource_plan_schema=image.plan_schema,
        program_resource_plan_digest=image.plan_digest if digest is None else digest,
        program_resource_plan_maximum_bytes=(
            image.maximum_bytes if maximum_bytes is None else maximum_bytes
        ),
        program_resource_slot_count=image.slot_count if slot_count is None else slot_count,
    )


def test_roundtrip_is_bit_exact_and_keeps_cold_invalid_metadata():
    image = _image()
    encoded = encode_program_persistent_value_checkpoint(image)
    restored = decode_program_persistent_value_checkpoint(encoded)

    assert encode_program_persistent_value_checkpoint(restored) == encoded
    assert restored.storage == image.storage
    assert restored.metadata[0]["accumulated_dt"] == image.metadata[0]["accumulated_dt"]
    assert restored.metadata[0]["valid"] is False
    assert restored.metadata[0]["cold"] is True
    assert program_persistent_checkpoint_from_payload(_payload(image))[1] == restored
    assert program_persistent_checkpoint_manifest_from_payload(_payload(image))["plan_digest"] == "a" * 64


def test_valid_slot_retains_exact_value_bytes_while_invalid_slot_is_zero_sized():
    metadata = (
        {
            "accepted_coordinate": 9,
            "cursor": 4,
            "accumulated_dt": 0.375,
            "topology_epoch": 11,
            "layout_generation": 7,
            "valid": True,
            "cold": False,
        },
        {
            "accepted_coordinate": 9,
            "cursor": 4,
            "accumulated_dt": 0.375,
            "topology_epoch": 11,
            "layout_generation": 7,
            "valid": False,
            "cold": True,
        },
    )
    image = _image(_row(), _row(slot=1, path="root/1", path_id=2), metadata=metadata, maximum_bytes=32)
    assert image.value_bytes == (8, 0)
    restored = decode_program_persistent_value_checkpoint(
        encode_program_persistent_value_checkpoint(image)
    )
    assert restored.value_bytes == (8, 0)


@pytest.mark.parametrize(
    "metadata",
    (
        {
            "accepted_coordinate": 9,
            "cursor": 4,
            "accumulated_dt": 0.375,
            "topology_epoch": 11,
            "layout_generation": 7,
            "valid": False,
            "cold": False,
        },
        {
            "accepted_coordinate": 9,
            "cursor": 4,
            "accumulated_dt": 0.375,
            "topology_epoch": 11,
            "layout_generation": 7,
            "valid": True,
            "cold": True,
        },
    ),
)
def test_valid_and_cold_markers_must_be_complementary(metadata):
    with pytest.raises(ValueError, match="not complementary"):
        encode_program_persistent_value_checkpoint(_image(metadata=(metadata,)))


def test_manifest_binds_plan_digest_and_scalars_to_the_signed_envelope():
    image = _image()
    payload = _payload(image)
    payload.update({"t": np.asarray(1.0), "macro_step": np.asarray(3), "abi_key": np.asarray("abi")})
    _seal_checkpoint_payload_with_identities(
        payload,
        runtime_kind="uniform",
        semantic=make_identity("semantic", {"test": "semantic"}),
        artifact=make_identity("artifact", {"test": "artifact"}),
        bind=make_identity("bind", {"test": "bind"}),
        run=make_identity("run", {"test": "run"}),
    )
    manifest, _restart = inspect_checkpoint_payload_integrity(payload, runtime_kind="uniform")
    assert manifest[PROGRAM_PERSISTENT_CHECKPOINT_KEY] == {
        "schema": image.schema,
        "plan_schema": image.plan_schema,
        "plan_digest": image.plan_digest,
        "maximum_bytes": image.maximum_bytes,
        "slot_count": image.slot_count,
    }


@pytest.mark.parametrize("runtime_kind", ("uniform", "amr"))
def test_npz_decode_accepts_the_enriched_manifest_extension(runtime_kind):
    image = _image()
    payload = _payload(image)
    payload.update({"t": np.asarray(1.0), "macro_step": np.asarray(3), "abi_key": np.asarray("abi")})
    _seal_checkpoint_payload_with_identities(
        payload,
        runtime_kind=runtime_kind,
        semantic=make_identity("semantic", {"test": "semantic"}),
        artifact=make_identity("artifact", {"test": "artifact"}),
        bind=make_identity("bind", {"test": "bind"}),
        run=make_identity("run", {"test": "run"}),
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    budget = CheckpointResourceBudget(
        runtime_kind, 100, 100_000, 1_000_000, 2_000_000, 3_000_000, "test"
    )
    decoded = decode_checkpoint_bytes(stream.getvalue(), budget)
    assert PROGRAM_PERSISTENT_CHECKPOINT_KEY in decoded.files


def test_malformed_digest_offsets_duplicate_and_provider_are_refused():
    image = _image()
    encoded = bytearray(encode_program_persistent_value_checkpoint(image))
    encoded[20] ^= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        decode_program_persistent_value_checkpoint(bytes(encoded))

    duplicate = _row(slot=1)
    duplicate["identity"] = "program-resource:v1:row-duplicate"
    with pytest.raises(ValueError, match="duplicate complete key"):
        encode_program_persistent_value_checkpoint(_image(_row(), duplicate, maximum_bytes=32))

    collision = _row(slot=1, path="root/1", path_id=1)
    collision["key"] = dict(collision["key"], value_id=18)
    with pytest.raises(ValueError, match="digest collision"):
        encode_program_persistent_value_checkpoint(_image(_row(), collision, maximum_bytes=32))

    missing_provider = _row(spatial_transfer=2, transfer_identity="")
    with pytest.raises(ValueError, match="no provider"):
        encode_program_persistent_value_checkpoint(_image(missing_provider))

    unknown_bytes = _row(bytes=0)
    with pytest.raises(ValueError, match="incomplete byte"):
        encode_program_persistent_value_checkpoint(_image(unknown_bytes))

    malformed_offsets = dataclasses.replace(image, offsets=(0, 17))
    with pytest.raises(ValueError, match="offset"):
        encode_program_persistent_value_checkpoint(malformed_offsets)

    incomplete_ceiling = dataclasses.replace(
        image, maximum_bytes=image.maximum_bytes + 1
    )
    with pytest.raises(ValueError, match="exact checkpoint memory ceiling"):
        encode_program_persistent_value_checkpoint(incomplete_ceiling)


def test_payload_scalars_and_native_capability_are_strict():
    image = _image()
    payload = _payload(image)
    payload[PROGRAM_PERSISTENT_PLAN_DIGEST_KEY] = np.asarray("b" * 64)
    with pytest.raises(ValueError, match="digest differs"):
        program_persistent_checkpoint_from_payload(payload)

    class CompiledWithoutSeam:
        def installed_program_hash(self):
            return "compiled"

    with pytest.raises(RuntimeError, match="capture seam"):
        require_program_persistent_value_checkpoint_capture(CompiledWithoutSeam())


def test_prepare_publish_is_detached_and_rank_change_requires_specialized_seam():
    image = _image()
    encoded = encode_program_persistent_value_checkpoint(image)
    events = []

    class Native:
        def prepare_program_persistent_value_restore(self, value):
            events.append(("prepare", value))
            return "detached"

        def publish_program_persistent_value_restore(self, value):
            events.append(("publish", value))

    prepared = prepare_program_persistent_value_restore(Native(), encoded)
    assert isinstance(prepared, PreparedProgramPersistentValueRestore)
    assert events == [("prepare", encoded)]
    publish_program_persistent_value_restore(prepared)
    assert events[-1] == ("publish", "detached")

    with pytest.raises(RuntimeError, match="rank_change seam"):
        prepare_program_persistent_value_restore(Native(), encoded, mode="rank_change")

    captured = capture_program_persistent_value_checkpoint(
        type("NativeCapture", (), {"capture_program_persistent_value_checkpoint": lambda self: encoded})()
    )
    assert captured is not None and captured[0] == encoded and captured[1] == image


def test_specialized_restore_dispatches_without_same_layout_fallback():
    image = _image()
    encoded = encode_program_persistent_value_checkpoint(image)
    events = []

    class Native:
        def prepare_program_persistent_value_restore(self, _value):
            raise AssertionError("specialized restore fell back to the same-layout seam")

        def prepare_program_persistent_value_redistribution(self, value):
            events.append(("rank_change", value))
            return "rank-detached"

        def prepare_program_persistent_value_regrid(self, value):
            events.append(("regrid_on_restart", value))
            return "regrid-detached"

        def publish_program_persistent_value_restore(self, value):
            events.append(("publish", value))

    native = Native()
    rank = prepare_program_persistent_value_restore(native, encoded, mode="rank_change")
    regrid = prepare_program_persistent_value_restore(native, encoded, mode="regrid_on_restart")
    assert events == [("rank_change", encoded), ("regrid_on_restart", encoded)]
    publish_program_persistent_value_restore(rank)
    publish_program_persistent_value_restore(regrid)
    assert events[-2:] == [("publish", "rank-detached"), ("publish", "regrid-detached")]


def test_runtime_sized_target_plan_differs_only_on_specialized_restore():
    image = _image()
    changed_target = _budget(
        image,
        digest="b" * 64,
        maximum_bytes=image.maximum_bytes + 16,
        slot_count=image.slot_count + 1,
    )
    with pytest.raises(ValueError, match="differs from the bound live plan"):
        require_program_persistent_checkpoint_plan(image, changed_target)

    assert (
        require_program_persistent_checkpoint_plan(image, changed_target, mode="rank_change")
        is image
    )
    assert (
        require_program_persistent_checkpoint_plan(
            image, changed_target, mode="regrid_on_restart"
        )
        is image
    )


@pytest.mark.parametrize("mode", (None, [], "unsupported"))
def test_unknown_restore_mode_is_rejected_as_a_value_error(mode):
    image = _image()
    encoded = encode_program_persistent_value_checkpoint(image)
    with pytest.raises(ValueError, match="unknown Program persistent checkpoint restore mode"):
        prepare_program_persistent_value_restore(
            SimpleNamespace(), encoded, mode=mode
        )
    with pytest.raises(ValueError, match="unknown Program persistent checkpoint restore mode"):
        require_program_persistent_checkpoint_plan(image, _budget(image), mode=mode)


def test_specialized_restore_requires_complete_target_identity_before_native_prepare():
    image = _image()
    encoded = encode_program_persistent_value_checkpoint(image)
    events = []

    class Native:
        def prepare_program_persistent_value_redistribution(self, _value):
            events.append("prepare")
            return "must-not-be-reached"

    # A producer/archive budget without the bind-sealed target identity cannot authorize a
    # specialized source-to-target transfer.  In particular, the native detached seam must not
    # be used as a way to discover or repair a missing target identity.
    budget = CheckpointResourceBudget(
        "amr", 100, 100_000, 1_000_000, 2_000_000, 3_000_000, "missing-target-plan"
    )
    with pytest.raises(RuntimeError, match="lacks the sealed Program resource-plan identity"):
        require_program_persistent_checkpoint_plan(image, budget, mode="rank_change")
    assert events == []
    with pytest.raises(RuntimeError, match="lacks the detached"):
        prepare_program_persistent_value_restore(Native(), encoded, mode="regrid_on_restart")
    assert events == []


def test_specialized_restore_rejects_an_unbound_image_before_dispatch():
    image = ProgramPersistentValueCheckpoint(
        bound=False,
        schema=PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA,
        plan_schema="",
        plan_digest="",
        maximum_bytes=0,
        slot_count=0,
        rows=(),
        metadata=(),
        offsets=(),
        value_bytes=(),
        storage=b"",
    )
    with pytest.raises(ValueError, match="image is unbound"):
        require_program_persistent_checkpoint_plan(image, _budget(_image()), mode="rank_change")


def _prepare_v3_dispatch_fixture(
    monkeypatch,
    image,
    *,
    checkpoint_ranks,
    current_ranks,
    budget=None,
):
    """Install only data-only seams needed to exercise POPSPVS1 AMR preflight dispatch."""
    import pops.runtime._amr_checkpoint_v3 as checkpoint_v3
    from pops.runtime._temporal_restart import TemporalRestartState

    events = []

    class Native:
        def __init__(self):
            self.state = {"persistent": "accepted"}

        def installed_program_hash(self):
            return "compiled"

        def prepare_program_persistent_value_restore(self, _value):
            events.append("restore")
            return "restore-detached"

        def prepare_program_persistent_value_redistribution(self, _value):
            events.append("redistribution")
            return "redistribution-detached"

        def prepare_program_persistent_value_regrid(self, _value):
            events.append("regrid")
            return "regrid-detached"

        def publish_program_persistent_value_restore(self, value):
            events.append("publish")
            self.state["persistent"] = value

        def uses_runtime_engine(self):
            return True

        def block_names(self):
            return ["fluid"]

        def field_provider_slots(self):
            return []

        def block_n_vars(self, _block):
            return 1

        def checkpoint_phi_provider_slot(self):
            return ""

        def restore_restart_auxiliary_checkpoint_accepted_state(self, _payload):
            raise AssertionError("prepare_v3 must not mutate the native accepted state")

    native = Native()
    spatial = SimpleNamespace(dimension=1, cells_at_level=lambda _level: 1)
    topology = SimpleNamespace(size=current_ranks)
    recorded = SimpleNamespace(
        program_state=b"",
        level_distribution_modes=("partitioned",),
        level_owner_ranks=((0,),),
    )
    owner = SimpleNamespace(_checkpoint_resource_budget=budget or _budget(image))
    payload = _payload(image)
    payload.update(
        {
            "n_ranks": np.asarray(checkpoint_ranks, dtype=np.int64),
            "program_hash": np.asarray("compiled"),
            "blocks": np.asarray(["fluid"]),
            "macro_step": np.asarray(0, dtype=np.int64),
            "t": np.asarray(0.0),
            "temporal_restart_state": np.asarray("{}"),
            "field_provider_slots": np.asarray([], dtype=str),
            "field_provider_manifest": np.asarray("[]"),
            "patch_boxes": np.asarray([], dtype=np.int64),
            "n_vars_fluid": np.asarray(1, dtype=np.int64),
            "state_fluid_0": np.asarray([2.0], dtype=np.float64),
            "auxiliary_checkpoint_0": np.asarray([7], dtype=np.uint8),
        }
    )

    monkeypatch.setattr(
        "pops.output._checkpoint_collective.checkpoint_topology",
        lambda _owner: topology,
    )
    monkeypatch.setattr(
        "pops.runtime._checkpoint_spatial.authenticate_checkpoint_spatial_contract",
        lambda _owner, _payload: spatial,
    )
    monkeypatch.setattr(
        "pops.runtime._amr_checkpoint_contract.checkpoint_temporal_partition_kind",
        lambda _payload: "global",
    )
    monkeypatch.setattr(
        "pops.runtime._amr_checkpoint_contract.preflight_contract",
        lambda _sim, _payload: (b"", 0, 0),
    )
    monkeypatch.setattr(
        "pops.runtime._program_cadence_checkpoint.prepare_program_cadence",
        lambda *_args, **_kwargs: "cadence",
    )
    monkeypatch.setattr(
        TemporalRestartState,
        "from_json",
        classmethod(lambda _cls, *_args, **_kwargs: "temporal"),
    )
    monkeypatch.setattr(
        "pops.runtime._amr_checkpoint_topology.recorded_rank_topology",
        lambda *_args, **_kwargs: recorded,
    )
    monkeypatch.setattr(
        "pops.runtime._amr_checkpoint_topology.owner_ranks_for_boxes",
        lambda *_args, **_kwargs: (0,),
    )
    monkeypatch.setattr(
        checkpoint_v3,
        "_checkpoint_amr_level_envelope",
        lambda *_args, **_kwargs: (1, 1),
    )
    monkeypatch.setattr(
        checkpoint_v3,
        "_decode_ranked_patch_boxes",
        lambda *_args, **_kwargs: ((0, (0,), (1,)),),
    )
    monkeypatch.setattr(
        checkpoint_v3,
        "_preflight_current_base_distribution",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(checkpoint_v3, "_field_provider_manifest", lambda _sim: ())
    monkeypatch.setattr(
        checkpoint_v3,
        "_preflight_histories_v3",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pops.runtime._checkpoint_resource_budget.require_checkpoint_resource_budget",
        lambda _owner: owner._checkpoint_resource_budget,
    )
    return owner, native, payload, events


@pytest.mark.parametrize(
    (
        "checkpoint_ranks",
        "current_ranks",
        "hierarchy_mode",
        "expected_event",
        "expected_image",
    ),
    (
        (1, 1, "restore_recorded_hierarchy", "restore", "restore-detached"),
        (1, 2, "restore_recorded_hierarchy", "redistribution", "redistribution-detached"),
        (1, 1, "regrid_on_restart", "regrid", "regrid-detached"),
    ),
)
def test_prepare_v3_dispatches_popspvs1_without_native_mutation(
    monkeypatch,
    checkpoint_ranks,
    current_ranks,
    hierarchy_mode,
    expected_event,
    expected_image,
):
    image = _image()
    owner, native, payload, events = _prepare_v3_dispatch_fixture(
        monkeypatch,
        image,
        checkpoint_ranks=checkpoint_ranks,
        current_ranks=current_ranks,
    )

    from pops.runtime._amr_checkpoint_v3 import prepare_v3

    hierarchy_identity = None
    if hierarchy_mode == "regrid_on_restart":
        from pops.output import RegridOnRestart

        hierarchy_identity = RegridOnRestart().identity.token
    prepared = prepare_v3(
        owner,
        native,
        payload,
        bit_identical=False,
        hierarchy_mode=hierarchy_mode,
        hierarchy_identity=hierarchy_identity,
    )
    assert events == [expected_event]
    assert native.state == {"persistent": "accepted"}
    assert prepared.program_persistent_restore.native_image == expected_image


def test_prepare_v3_refuses_missing_target_identity_without_native_mutation(monkeypatch):
    image = _image()
    budget = CheckpointResourceBudget(
        "amr", 100, 100_000, 1_000_000, 2_000_000, 3_000_000, "missing-target-plan"
    )
    owner, native, payload, events = _prepare_v3_dispatch_fixture(
        monkeypatch,
        image,
        checkpoint_ranks=1,
        current_ranks=2,
        budget=budget,
    )

    from pops.runtime._amr_checkpoint_v3 import prepare_v3

    with pytest.raises(RuntimeError, match="lacks the sealed Program resource-plan identity"):
        prepare_v3(owner, native, payload, bit_identical=False)
    assert events == []
    assert native.state == {"persistent": "accepted"}
