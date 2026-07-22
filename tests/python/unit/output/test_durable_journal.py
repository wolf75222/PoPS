"""Exact archive and crash-window tests for post-commit observer durability."""

from __future__ import annotations

import hashlib
import os
import stat
import zipfile
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output._consumer_contracts import ParallelMode
from pops.output._durable_journal import DurableJournal
from pops.output._observer_archive import (
    decode_observer_frame,
    encode_observer_frame,
    observer_archive_identity,
    read_observer_archive,
    write_observer_archive,
)
from pops.output.data import (
    ArrayPiece,
    DiagnosticKey,
    DiagnosticPayload,
    FieldKey,
    FieldPayload,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    _NativeCompositeIntegral,
    _field_family_identity,
)
from pops.output.observers import ObserverFrame


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _field_key(name: str, component, layout) -> FieldKey:
    owner = OwnerPath.case("archive-case").child(OwnerKind.BLOCK, "heat")
    return FieldKey(Handle(name, kind="state", owner=owner), component, layout, 0, "accepted")


def _frame(*, unselected_offset: float = 0.0, macro_step: int = 11) -> ObserverFrame:
    layout = _identity("layout-plan", "two-patch-uniform")
    component = _identity("component-manifest", "heat-and-tracer")
    selected_key = _field_key("state", component, layout)
    unselected_key = _field_key("pressure", component, layout)

    geometry = LevelGeometry(
        layout,
        "uniform",
        0,
        (1.25, -2.5),
        (0.125, 0.25),
        (3, 3),
        ((0, 0, 2, 3), (2, 0, 3, 3)),
        np.asarray(
            [[False, False, True], [False, True, True], [False, False, False]],
            dtype=np.bool_,
        ),
        np.asarray(
            [[0.03125, 0.03125, 0.0625], [0.03125, 0.03125, 0.0625], [0.03125, 0.03125, 0.0625]],
            dtype=np.float64,
        ),
        coordinate_system="pops://coordinates/cartesian-2d@1",
        cell_measure="pops://cell-measures/cartesian-area@1",
        axis_names=("radius", "height"),
    )
    selected = FieldPayload(
        selected_key,
        "cell",
        "K",
        ("temperature", "tracer"),
        (3, 3),
        (
            ArrayPiece(
                (0, 0),
                (2, 3),
                np.arange(12, dtype=np.float64).reshape(2, 2, 3),
                0,
                0,
                False,
            ),
            ArrayPiece(
                (2, 0),
                (3, 3),
                (100.0 + np.arange(6, dtype=np.float64)).reshape(2, 1, 3),
                1,
                0,
                False,
            ),
        ),
    )
    unselected = FieldPayload(
        unselected_key,
        "cell",
        "Pa",
        (),
        (3, 3),
        (
            ArrayPiece(
                (0, 0),
                (3, 3),
                np.arange(9, dtype=np.int32).reshape(3, 3) + unselected_offset,
                0,
                0,
                False,
            ),
        ),
    )
    diagnostic_key = DiagnosticKey(
        Handle(
            "energy_balance",
            kind="diagnostic",
            owner=OwnerPath.consumer("archive-diagnostics"),
        ),
        component,
        layout,
        0,
        "accepted",
        "composite_metric_balance",
    )
    diagnostic = DiagnosticPayload(
        diagnostic_key,
        -0.125,
        "J",
        {"storage": 4.5, "flux": -4.625, "source": 0.0},
    )
    snapshot = OutputSnapshot(
        OutputClock.at(
            "macro",
            0.375,
            macro_step,
            stage="accepted",
            tick=19 + macro_step,
            level=0,
            substep=2,
            stage_index=3,
            fraction=(1, 1),
            dt=0.03125,
        ),
        OutputProvenance(
            _identity("resolved-plan", "archive-plan"),
            _identity("bind", "archive-bind"),
            _identity("run", "archive-run"),
            "accepted-step-transaction",
        ),
        (geometry,),
        (selected, unselected),
        {"case": "durable-archive", "restart": False, "schema": 1},
        diagnostics=(diagnostic,),
        _native_composite_integrals=(
            _NativeCompositeIntegral(_field_family_identity(selected_key), (0,), 12.5),
        ),
    )
    request = OutputRequest(
        "durable-live-state",
        (selected_key,),
        ParallelMode.SERIAL,
        diagnostics=(diagnostic_key,),
    )
    return ObserverFrame(snapshot, request)


def _assert_complete_frame_equal(actual: ObserverFrame, expected: ObserverFrame) -> None:
    assert actual.identity == expected.identity
    assert observer_archive_identity(actual) == observer_archive_identity(expected)
    assert actual.request.to_data() == expected.request.to_data()
    assert actual.snapshot.clock == expected.snapshot.clock
    assert actual.snapshot.provenance == expected.snapshot.provenance
    assert actual.snapshot.metadata == expected.snapshot.metadata
    assert len(actual.snapshot.geometries) == len(expected.snapshot.geometries)
    assert len(actual.snapshot.fields) == len(expected.snapshot.fields)
    assert len(actual.snapshot.diagnostics) == len(expected.snapshot.diagnostics)

    for left, right in zip(actual.snapshot.geometries, expected.snapshot.geometries, strict=True):
        assert left.to_data() == right.to_data()
        np.testing.assert_array_equal(left.valid_cells, right.valid_cells)
        np.testing.assert_array_equal(left.coverage, right.coverage)
        np.testing.assert_array_equal(left.cell_volumes, right.cell_volumes)
        assert not left.valid_cells.flags.writeable
        assert not left.coverage.flags.writeable
        assert not left.cell_volumes.flags.writeable
    for left, right in zip(actual.snapshot.fields, expected.snapshot.fields, strict=True):
        assert left.to_data() == right.to_data()
        assert len(left.pieces) == len(right.pieces)
        for left_piece, right_piece in zip(left.pieces, right.pieces, strict=True):
            np.testing.assert_array_equal(left_piece.values, right_piece.values)
            assert not left_piece.values.flags.writeable
    assert [item.to_data() for item in actual.snapshot.diagnostics] == [
        item.to_data() for item in expected.snapshot.diagnostics
    ]
    assert [
        (item.family_identity, item.levels, item.value.hex())
        for item in actual.snapshot._native_composite_integrals
    ] == [
        (item.family_identity, item.levels, item.value.hex())
        for item in expected.snapshot._native_composite_integrals
    ]


def _rewrite_archive(payload: bytes, replacements: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(BytesIO(payload), "r") as source:
        entries = [(item, source.read(item)) for item in source.infolist()]
    with zipfile.ZipFile(output, "w") as target:
        for information, member_payload in entries:
            target.writestr(
                information,
                replacements.get(information.filename, member_payload),
            )
    return output.getvalue()


def test_observer_archive_is_deterministic_pickle_free_and_exact() -> None:
    frame = _frame()

    first = encode_observer_frame(frame)
    second = encode_observer_frame(frame)

    assert first == second
    with zipfile.ZipFile(BytesIO(first), "r") as archive:
        members = archive.infolist()
        assert [item.filename for item in members][0] == "manifest.json"
        assert all(item.compress_type == zipfile.ZIP_STORED for item in members)
        assert all(
            item.filename == "manifest.json" or item.filename.endswith(".npy") for item in members
        )
        assert not any(
            "pickle" in item.filename or item.filename.endswith(".pkl") for item in members
        )
        for item in members[1:]:
            value = np.load(BytesIO(archive.read(item)), allow_pickle=False)
            assert isinstance(value, np.ndarray)
            assert not value.dtype.hasobject

    restored = decode_observer_frame(first)

    _assert_complete_frame_equal(restored, frame)
    assert encode_observer_frame(restored) == first


def test_observer_archive_file_sha_corruption_and_no_clobber(tmp_path: Path) -> None:
    frame = _frame()
    target = tmp_path / "frame.pfa"

    digest = write_observer_archive(target, frame)

    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    _assert_complete_frame_equal(read_observer_archive(target, expected_sha256=digest), frame)
    with pytest.raises(ValueError, match="SHA-256"):
        read_observer_archive(target, expected_sha256="0" * 64)
    with pytest.raises(TypeError, match="lowercase SHA-256"):
        read_observer_archive(target, expected_sha256=digest.upper())
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_observer_archive(target, frame)
    assert target.read_bytes() == original

    corrupted = bytearray(original)
    corrupted[len(corrupted) // 2] ^= 0x01
    with pytest.raises(ValueError):
        decode_observer_frame(bytes(corrupted))


def test_observer_archive_rejects_noncanonical_manifest_and_rehashed_array_tamper() -> None:
    encoded = encode_observer_frame(_frame())
    with zipfile.ZipFile(BytesIO(encoded), "r") as archive:
        manifest = archive.read("manifest.json")
        array_name = next(
            item.filename for item in archive.infolist() if item.filename.endswith(".npy"))
        value = np.load(BytesIO(archive.read(array_name)), allow_pickle=False).copy()

    with pytest.raises(ValueError, match="canonical JSON"):
        decode_observer_frame(_rewrite_archive(
            encoded, {"manifest.json": manifest + b"\n"}))

    value.flat[0] = value.flat[0] + 1
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256"):
        decode_observer_frame(_rewrite_archive(encoded, {array_name: stream.getvalue()}))


def test_durable_journal_prepare_commit_delivery_cycle_is_idempotent(
    tmp_path: Path,
) -> None:
    journal = DurableJournal(tmp_path / "journal")
    frame = _frame()

    prepared = journal.prepare(frame)

    assert prepared.path.parent.name == "prepared"
    assert prepared.path.exists()
    assert journal.list_pending() == ()

    pending = journal.commit(prepared)
    repeated_pending = journal.commit(prepared)

    assert pending.path.parent.name == "pending"
    assert repeated_pending.path == pending.path
    assert not prepared.path.exists()
    listed = journal.list_pending()
    assert len(listed) == 1
    assert listed[0].path == pending.path
    _assert_complete_frame_equal(listed[0].frame, frame)
    assert listed[0].file_sha256 == hashlib.sha256(pending.path.read_bytes()).hexdigest()

    completed = journal.delivered(pending)
    repeated_completed = journal.delivered(pending)

    assert completed.path.parent.name == "delivered"
    assert repeated_completed.path == completed.path
    assert completed.path.exists()
    assert not pending.path.exists()
    assert journal.list_pending() == ()

    rolled_back = journal.prepare(_frame(macro_step=12))
    assert journal.discard_prepared(rolled_back) is True
    assert journal.discard_prepared(rolled_back) is False
    assert not rolled_back.path.exists()


def test_automatic_recovery_closes_hard_link_crash_windows(tmp_path: Path) -> None:
    root = tmp_path / "journal"
    journal = DurableJournal(root)
    frame = _frame()
    prepared = journal.prepare(frame)
    pending_path = root / "pending" / prepared.path.name

    # Crash after publishing the committed hard link, before unlinking ``prepared``.
    os.link(prepared.path, pending_path)
    recovered = DurableJournal(root, recover="automatic")

    assert not prepared.path.exists()
    assert pending_path.exists()
    assert [item.path for item in recovered.list_pending()] == [pending_path]

    delivered_path = root / "delivered" / pending_path.name
    # Crash after publishing the delivery hard link, before unlinking ``pending``.
    os.link(pending_path, delivered_path)
    recovered_again = DurableJournal(root, recover="automatic")

    assert not pending_path.exists()
    assert delivered_path.exists()
    assert recovered_again.list_pending() == ()

    uncommitted = recovered_again.prepare(_frame(macro_step=12))
    assert uncommitted.path.exists()
    DurableJournal(root, recover="automatic")
    assert not uncommitted.path.exists()


def test_durable_journal_binds_exact_delivery_target_once(tmp_path: Path) -> None:
    journal = DurableJournal(tmp_path / "journal", sync="none")
    authority = {
        "schema_version": 1,
        "consumer_id": "consumer/temperature",
        "manifest_identity": make_identity(
            "consumer-manifest", {"name": "temperature"}).token,
        "target_uri": "temperature",
        "resolved_target": (tmp_path / "output" / "temperature").resolve().as_posix(),
    }

    journal.bind_delivery_authority(authority)
    journal.bind_delivery_authority(dict(authority))
    changed = dict(authority, resolved_target=(tmp_path / "other").resolve().as_posix())
    with pytest.raises(RuntimeError, match="different consumer/output authority"):
        journal.bind_delivery_authority(changed)


def test_durable_journal_validates_configuration_is_immutable_and_rejects_collision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    with pytest.raises(ValueError, match="sync"):
        DurableJournal(root, sync="eventual")
    with pytest.raises(ValueError, match="recover"):
        DurableJournal(root, recover="guess")

    journal = DurableJournal(root, sync="none", recover="manual")
    with pytest.raises(FrozenInstanceError):
        journal.root = tmp_path / "elsewhere"

    first = _frame(unselected_offset=0.0)
    collision = _frame(unselected_offset=0.5)
    assert first.identity == collision.identity
    assert observer_archive_identity(first) != observer_archive_identity(collision)
    first_record = journal.prepare(first)
    first_bytes = first_record.path.read_bytes()

    with pytest.raises((FileExistsError, ValueError), match="exist|collision|identity"):
        journal.prepare(collision)
    assert first_record.path.read_bytes() == first_bytes


def test_fsync_policy_covers_archive_files_and_state_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized_modes = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: synchronized_modes.append(
        os.fstat(descriptor).st_mode))

    journal = DurableJournal(tmp_path / "durable")
    prepared = journal.prepare(_frame())
    journal.delivered(journal.commit(prepared))

    assert any(stat.S_ISREG(mode) for mode in synchronized_modes)
    assert any(stat.S_ISDIR(mode) for mode in synchronized_modes)
    synchronized_modes.clear()
    non_durable = DurableJournal(tmp_path / "atomic-only", sync="none")
    non_durable.prepare(_frame())
    assert synchronized_modes == []
