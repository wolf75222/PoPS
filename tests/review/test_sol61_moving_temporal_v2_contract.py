"""Structural metadata counter-probes; no invented positive physical receipt."""

import ast
import copy
import os
from pathlib import Path

import pytest

from tests.review import sol61_moving_interval_offline_oracle as oracle


def boundary():
    """Temporal protocol primitive only, with no state, geometry or native wire."""
    return dict(
        schema_version=2,
        strategy={},
        program_schedule=dict(
            schema_version=1,
            kind="pops.temporal-program-schedule",
            primary_clock="primary",
            clocks=[dict(id="primary", descriptor={}, ticks_per_macro=1)],
            subcycles=[],
            synchronizations=[],
            schedules=[],
            histories=[],
        ),
        clock=dict(time=(0.001).hex(), macro_step=1),
        clock_cursors={"primary": dict(time=(0.001).hex(), tick=1, phase="accepted")},
        schedule_cursors={"macro_step": dict(macro_step=1, phase="accepted")},
        synchronization_cursors={},
        history_cursors={},
        cache_cursors={},
        controller_state={},
        event_queue=[],
        transaction_stats=dict(accepted=1, failed=0, rejected=0),
        status="accepted",
        synchronized=True,
    )


def test_schema_keys_match_actual_versioned_serializer_source():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "python/pops/runtime/_temporal_restart.py").read_text())
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemporalRestartState"
    )
    method = next(
        node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "to_data"
    )
    result = next(node.value for node in method.body if isinstance(node, ast.Return))
    assert {ast.literal_eval(key) for key in result.keys} == oracle.TEMPORAL_V2_KEYS
    version = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_SCHEMA_VERSION"
            for target in node.targets
        )
    )
    assert ast.literal_eval(version) == 2
    oracle.temporal_boundary(boundary(), 0.001, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("schema_version", True),
        lambda value: value.__setitem__("schema_version", 1),
        lambda value: value.__setitem__(
            "synchronization_state", value.pop("synchronization_cursors")
        ),
        lambda value: value.__setitem__("cache_generations", {}),
        lambda value: value.__setitem__("synchronization_cursors", {"foreign": dict(macro_step=1)}),
        lambda value: value.__setitem__("history_cursors", {"foreign": dict(position=0)}),
        lambda value: value.__setitem__("cache_cursors", {"foreign": dict(generation=0)}),
        lambda value: value["clock"].__setitem__("macro_step", True),
        lambda value: value["transaction_stats"].__setitem__("failed", True),
        lambda value: value["transaction_stats"].__setitem__("rejected", -1),
        lambda value: value.__setitem__("synchronized", 1),
        lambda value: value.__setitem__("event_queue", ["uncommitted-event"]),
    ],
)
def test_noncanonical_or_foreign_temporal_metadata_is_refused(mutation):
    value = copy.deepcopy(boundary())
    mutation(value)
    with pytest.raises(ValueError):
        oracle.temporal_boundary(value, 0.001, 1)


def test_restart_comparison_has_no_selective_history_or_cache_key_fallback():
    source = Path(oracle.__file__).read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "same_accepted"
    )
    text = ast.get_source_segment(source, method)
    assert "common.cbor(a.temporal) == common.cbor(b.temporal)" in text
    assert "for name in a.native_payload" in text
    assert "synchronization_state" not in text and "cache_generations" not in text
    assert ".get(" not in text


def test_bounded_reader_keeps_exact_external_hash_and_never_requests_eof(tmp_path, monkeypatch):
    path = tmp_path / "protocol-bytes.bin"
    value = b"structural metadata only"
    path.write_bytes(value)
    original = os.pread
    requests = []

    def tracked(descriptor, size, offset):
        requests.append((size, offset))
        assert 0 <= offset < len(value) and size <= len(value) - offset
        return original(descriptor, min(size, 3), offset)

    monkeypatch.setattr(oracle.os, "pread", tracked)
    assert (
        oracle.pinned_file(tmp_path, dict(path=path.name, sha256=oracle.digest(value)))[1] == value
    )
    assert requests and requests[-1][1] < len(value)
    with pytest.raises(ValueError, match="SHA256"):
        oracle.pinned_file(tmp_path, dict(path=path.name, sha256="0" * 64))


def test_bounded_reader_refuses_truncation_before_hash(tmp_path, monkeypatch):
    path = tmp_path / "protocol-bytes.bin"
    path.write_bytes(b"metadata")
    monkeypatch.setattr(oracle.os, "pread", lambda *args: b"")
    with pytest.raises(ValueError, match="truncated bounded"):
        oracle.file_bytes(path)


def test_bounded_reader_refuses_mutation_during_read(tmp_path, monkeypatch):
    path = tmp_path / "protocol-bytes.bin"
    path.write_bytes(b"metadata")
    original = os.pread

    def mutating(descriptor, size, offset):
        value = original(descriptor, size, offset)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return value

    monkeypatch.setattr(oracle.os, "pread", mutating)
    with pytest.raises(ValueError, match="changed during"):
        oracle.file_bytes(path)


def test_bounded_reader_refuses_nonregular_and_oversized_files(tmp_path, monkeypatch):
    fifo = tmp_path / "protocol-pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="nonregular"):
        oracle.file_bytes(fifo)
    path = tmp_path / "protocol-bytes.bin"
    path.write_bytes(b"metadata")
    monkeypatch.setattr(oracle.common, "MAX_FILE_BYTES", 2)
    with pytest.raises(ValueError, match="oversized"):
        oracle.file_bytes(path)
