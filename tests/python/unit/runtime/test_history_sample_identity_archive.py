"""Typed sample archives: exact identity, legacy reset, and pre-mutation rejection."""

import struct

import numpy as np
import pytest

from pops.runtime._history_sample_identity import (
    identity_key,
    prepare_identity_payload,
    validate_identity_bytes,
)
from pops.runtime._system_io_history import (
    capture_histories,
    prepare_history_capture,
    restore_histories,
)
from pops.time._history.persistence import Revolve


@pytest.fixture(autouse=True)
def _source_native_real_width(monkeypatch):
    # These are source protocol controls, not calls into a loaded native runtime.
    monkeypatch.setattr("pops.runtime._history_sample_identity.native_real_bytes", lambda: 8)


def _image(name="u", level=None, depth=3, rows=None):
    encoded = name.encode("utf-8")
    header = b"POPSHID1" + struct.pack("<Q", len(encoded)) + encoded
    header += struct.pack("<qQ", -1 if level is None else level, depth)
    if rows is None:
        start = struct.unpack("<Q", struct.pack("<d", 0.125))[0]
        dt = struct.unpack("<Q", struct.pack("<d", 0.0625))[0]
        rows = [(2, start, dt, i + 1) for i in range(depth)]
    return header + b"".join(struct.pack("<QQQQ", *row) for row in rows)


class _Runtime:
    def __init__(self, level):
        self.level = level
        self.events = []
        self.identities = {name: _image(name, level) for name in ("u", "v")}

    def history_names(self):
        return ["u", "v"]

    def history_levels(self, name):
        if self.level is None:
            raise NotImplementedError
        return [self.level]

    def history_depth(self, name):
        return 3

    def history_ncomp(self, name):
        return 1

    def history_initialized(self, *args):
        return True

    def history_fill_count(self, *args):
        return 3

    def history_slot_dt(self, *args):
        return 0.0625

    def history_sample_identity(self, name, *args):
        return self.identities[name]

    def history_global(self, name, *args):
        self.events.append(("gather", name, args[-1]))
        return np.asarray([float(args[-1])])

    def restore_history(self, name, *args):
        self.identities[name] = b"invalidated-anchor"
        self.events.append(("anchor", name))

    def restore_history_sample_identity(self, name, *args):
        self.identities[name] = bytes(args[-1])
        self.events.append(("identity", name))

    def restore_history_provenance(self, name, *args):
        self.identities[name] = b"invalidated-provenance"
        self.events.append(("provenance", name))

    def restore_history_slot_dt(self, *args):
        pass

    def set_history_initialized(self, *args):
        pass

    def restore_history_fill_count(self, name, *args):
        self.identities[name] = b"invalidated-fill"

    def rebuild_history_slots(self, name, stored):
        assert stored == [0, 2]
        self.events.append(("replay", name))
        return 1


def _capture(runtime):
    plan = prepare_history_capture(runtime, {name: Revolve(2) for name in runtime.history_names()})
    payload = {}
    capture_histories(runtime, plan, payload)
    return plan, payload


@pytest.mark.parametrize("level", [None, 0, 2])
def test_exact_full_ledger_is_restored_before_any_selective_replay(level):
    writer = _Runtime(level)
    plan, payload = _capture(writer)
    assert plan.to_data()[0]["levels"][0]["sample_identity"] == writer.identities["u"].hex()
    reader = _Runtime(level)
    reader.identities = {name: b"stale" for name in reader.identities}

    def before_replay():
        assert reader.identities == writer.identities
        assert not any(event[0] == "replay" for event in reader.events)
        reader.events.append(("before_replay", "all"))

    restore_histories(reader, payload, before_replay=before_replay)
    first_replay = next(i for i, event in enumerate(reader.events) if event[0] == "replay")
    assert sum(event[0] == "identity" for event in reader.events[:first_replay]) == 2
    assert reader.identities == writer.identities


@pytest.mark.parametrize("level", [None, 0])
def test_legacy_absence_clears_live_identity_before_replay(level):
    writer = _Runtime(level)
    _, payload = _capture(writer)
    for name in writer.identities:
        del payload[identity_key(name, level)]
    reader = _Runtime(level)

    def before_replay():
        assert reader.identities == {"u": b"", "v": b""}

    restore_histories(reader, payload, before_replay=before_replay)


@pytest.mark.parametrize("replacement", [
    _image("foreign"), _image("v", 0), _image("v", depth=2),
    b"", _image("v") + b"x", _image("v")[:-1],
    _image("v", rows=[(3, 0, 0, 0)] * 3),
    _image("v", rows=[(1, 0, 0, 1)] * 3),
    _image("v", rows=[(0, 1, 0, 0)] * 3),
    _image("v", rows=[(2, 0, 0, 1)] * 3),
    _image("v", rows=[(2, 0x7ff0000000000000, 1, 1)] * 3),
    np.asarray([1], dtype=np.int64), np.zeros((2, 2), dtype=np.uint8),
])
def test_malformed_or_foreign_second_ring_rejects_before_any_mutation(replacement):
    _, payload = _capture(_Runtime(None))
    payload[identity_key("v", None)] = (
        np.frombuffer(replacement, dtype=np.uint8) if isinstance(replacement, bytes) else replacement
    )
    reader = _Runtime(None)
    original = dict(reader.identities)
    with pytest.raises((ValueError, TypeError)):
        restore_histories(reader, payload)
    assert reader.events == []
    assert reader.identities == original


def test_identity_capture_rejects_foreign_ring_before_first_numeric_gather():
    runtime = _Runtime(1)
    runtime.identities["v"] = _image("u", 1)
    with pytest.raises(ValueError, match="ring"):
        _capture(runtime)
    assert runtime.events == []


def test_uint64_ordinal_and_signed_zero_window_bits_are_retained_exactly():
    raw = _image(rows=[(2, 1 << 63, 0x3fb0000000000000, (1 << 64) - 1)] * 3)
    assert validate_identity_bytes(raw, "u", None, 3) == raw
    assert prepare_identity_payload({identity_key("u", None): np.frombuffer(raw, dtype=np.uint8)},
                                    "u", None, 3) == raw


def test_missing_native_identity_seams_refuse_before_gather_or_restore():
    writer = _Runtime(None)
    writer.history_sample_identity = None
    with pytest.raises(TypeError, match="history_sample_identity"):
        _capture(writer)
    assert writer.events == []
    _, payload = _capture(_Runtime(None))
    reader = _Runtime(None)
    reader.restore_history_sample_identity = None
    with pytest.raises(TypeError, match="restore_history_sample_identity"):
        restore_histories(reader, payload)
    assert reader.events == []


def test_identity_npz_roundtrip_retains_exact_dtype_shape_and_bytes():
    import io

    _, payload = _capture(_Runtime(1))
    stream = io.BytesIO()
    np.savez(stream, **payload)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as restored:
        for name in ("u", "v"):
            values = restored[identity_key(name, 1)]
            assert values.dtype == np.dtype(np.uint8)
            assert values.ndim == 1
            assert prepare_identity_payload(restored, name, 1, 3) == _image(name, 1)


@pytest.mark.parametrize(("kind", "initialized"), [(1, True), (2, False)])
def test_identity_rejects_impossible_initialized_kind_before_capture(kind, initialized):
    raw = _image(rows=[(1, 0, 0, 0)] * 3) if kind == 1 else _image()
    with pytest.raises(ValueError, match="initialized metadata"):
        validate_identity_bytes(raw, "u", None, 3, initialized=initialized)
    runtime = _Runtime(None)
    runtime.identities["v"] = _image("v", rows=[(1, 0, 0, 0)] * 3)
    with pytest.raises(ValueError, match="initialized metadata"):
        _capture(runtime)
    assert runtime.events == []


def test_uniform_full_preflight_admits_optional_identity_and_refuses_foreign_or_invalid_context():
    from pops.runtime._checkpoint_exchanges import capture_checkpoint_continuation
    from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
    from tests.python.unit.runtime.test_continuation_transitions import _owner
    from tests.python.unit.runtime.test_uniform_continuation_preflight import _payload

    continuation = {}
    capture_checkpoint_continuation(_owner(), continuation)
    payload = _payload(continuation)
    _, histories = _capture(_Runtime(None))
    payload.update(histories)
    preflight_uniform_restart(payload)
    key = identity_key("v", None)
    saved = payload.pop(key)
    preflight_uniform_restart(payload)  # Old outer-version archives remain supported.
    for bad in (_image("u"), _image("v", rows=[(1, 0, 0, 0)] * 3)):
        payload[key] = np.frombuffer(bad, dtype=np.uint8)
        with pytest.raises(ValueError):
            preflight_uniform_restart(payload)
    payload[key] = saved
    preflight_uniform_restart(payload)


@pytest.mark.parametrize("width", [4, 8])
def test_publication_dt_uses_only_the_declared_native_width_before_mutation(monkeypatch, width):
    monkeypatch.setattr("pops.runtime._history_sample_identity.native_real_bytes", lambda: width)
    interval = 0.1
    rounded = float(np.float32(interval))
    expected_dt = rounded if width == 4 else interval
    other_dt = interval if width == 4 else rounded
    writer = _Runtime(None)
    bits = struct.unpack("<Q", struct.pack("<d", interval))[0]
    writer.identities = {name: _image(name, rows=[(2, 0, bits, 1)] * 3) for name in ("u", "v")}
    writer.history_slot_dt = lambda *args: expected_dt
    _, payload = _capture(writer)
    reader = _Runtime(None)
    restore_histories(reader, payload)
    assert reader.identities == writer.identities
    payload["history_slot_dt_v"] = np.asarray([other_dt] * 3, dtype=np.float64)
    rejected = _Runtime(None)
    with pytest.raises(ValueError, match="outgoing dt"):
        restore_histories(rejected, payload)
    assert rejected.events == []


def test_missing_native_width_cannot_authorize_a_publication_interval(monkeypatch):
    def unavailable():
        raise RuntimeError("native Real width unavailable")

    _, payload = _capture(_Runtime(None))
    monkeypatch.setattr("pops.runtime._history_sample_identity.native_real_bytes", unavailable)
    reader = _Runtime(None)
    with pytest.raises(RuntimeError, match="native Real width"):
        restore_histories(reader, payload)
    assert reader.events == []
