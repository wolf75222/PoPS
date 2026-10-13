"""Source-only admission/atomicity guards for empty AMR diagnostic rank adaptation."""
from types import SimpleNamespace
import struct

import numpy as np
import pytest

from pops.runtime import _checkpoint_program_diagnostics as codec
from pops.runtime import _checkpoint_resource_budget as budget
from pops.output import _checkpoint_collective as collective


def _image(rank=0, ranks=1, width=64, records=()):
    return b"POPSDIA1" + struct.pack("<QQQQ", width, rank, ranks, len(records)) + b"".join(
        struct.pack("<Q", len(name)) + name + struct.pack("<d", value) for name, value in records)


def _payload(images=None):
    images = (_image(0, 2), _image(1, 2)) if images is None else images
    offsets = [0]
    for image in images:
        offsets.append(offsets[-1] + len(image))
    return {"program_diagnostics_state": np.frombuffer(b"".join(images), dtype=np.uint8).copy(),
            "program_diagnostics_offsets": np.asarray(offsets, dtype=np.int64)}


def _owner(monkeypatch, *, kind="amr", target=None, capacity=256, bound=512):
    calls = []
    target = _image() if target is None else target
    def capture():
        calls.append(("capture", target))
        return target
    def validate(image):
        calls.append(("validate", image))
        if image and int.from_bytes(image[8:16], "little") != 64:
            raise ValueError("native Real width changed")
    owner = SimpleNamespace(_checkpoint_program_diagnostic_capacity_per_rank=capacity,
                            _checkpoint_program_diagnostic_byte_capacity=bound,
                            _s=SimpleNamespace(_checkpoint_program_diagnostics=capture,
                                               _validate_checkpoint_program_diagnostics=validate))
    monkeypatch.setattr(budget, "_require_checkpoint_diagnostic_authority", lambda o: None)
    monkeypatch.setattr(budget, "require_checkpoint_resource_budget", lambda o: SimpleNamespace(runtime_kind=kind))
    monkeypatch.setattr(collective, "checkpoint_topology", lambda o: SimpleNamespace(rank=0, size=1))
    return owner, calls


def _prepare(owner, payload, **kwargs):
    return codec.prepare_checkpoint_program_diagnostics(
        owner, payload, allow_empty_amr_rank_change=True, expected_source_ranks=2, **kwargs)


def test_empty_amr_uses_exact_native_target_image_without_mutating_source(monkeypatch):
    owner, calls = _owner(monkeypatch)
    payload = _payload()
    before = {key: value.copy() for key, value in payload.items()}
    result = _prepare(owner, payload)
    assert result == _image() and result != b""
    assert calls == [("capture", result), ("validate", result), ("validate", result)]
    assert all(np.array_equal(value, before[key]) for key, value in payload.items())


@pytest.mark.parametrize("kind,allow", [("amr", False), ("uniform", False), ("uniform", True)])
def test_strict_and_uniform_rank_changes_remain_refused(monkeypatch, kind, allow):
    owner, calls = _owner(monkeypatch, kind=kind)
    with pytest.raises(ValueError, match="rank count|requires an AMR"):
        codec.prepare_checkpoint_program_diagnostics(owner, _payload(),
            allow_empty_amr_rank_change=allow, expected_source_ranks=2 if allow else None)
    assert calls == []


@pytest.mark.parametrize("source_ranks", [True, 0, -1, 2.0, "2"])
def test_source_rank_authority_is_an_exact_positive_int(monkeypatch, source_ranks):
    owner, calls = _owner(monkeypatch)
    with pytest.raises(TypeError, match="exact positive int"):
        codec.prepare_checkpoint_program_diagnostics(owner, _payload(),
            allow_empty_amr_rank_change=True, expected_source_ranks=source_ranks)
    assert calls == []


def test_amr_three_diagnostic_two_authorities_are_not_mixed(monkeypatch):
    owner, calls = _owner(monkeypatch)
    with pytest.raises(ValueError, match="authenticated source rank count"):
        codec.prepare_checkpoint_program_diagnostics(owner, _payload(),
            allow_empty_amr_rank_change=True, expected_source_ranks=3)
    assert calls == []


@pytest.mark.parametrize("corruption", ["nonempty", "trailing", "count", "width", "magic", "rank", "offsets"])
def test_nonempty_and_malformed_sources_refuse_without_restore(monkeypatch, corruption):
    owner, calls = _owner(monkeypatch)
    first = _image(0, 2)
    if corruption == "nonempty": first = _image(0, 2, records=((b"opaque", -0.),))
    if corruption == "trailing": first += b"x"
    if corruption == "count": first = first[:32] + struct.pack("<Q", 1)
    if corruption == "width": first = _image(0, 2, width=32)
    if corruption == "magic": first = b"X" + first[1:]
    if corruption == "rank": first = _image(1, 2)
    payload = _payload((first, _image(1, 2)))
    if corruption == "offsets": payload["program_diagnostics_offsets"][1] = 0
    before = {key: value.copy() for key, value in payload.items()}
    with pytest.raises(ValueError):
        _prepare(owner, payload)
    assert all(np.array_equal(value, before[key]) for key, value in payload.items())
    assert all(call[0] in {"capture", "validate"} for call in calls)


def test_nonempty_target_refuses_without_erasing_opaque_records(monkeypatch):
    target = _image(records=((b"opaque", -0.),))
    owner, calls = _owner(monkeypatch, target=target)
    with pytest.raises(ValueError, match="empty target"):
        _prepare(owner, _payload())
    assert calls == [("capture", target)]
    assert owner._s._checkpoint_program_diagnostics() == target


def test_native_real_validation_precedes_source_adaptation(monkeypatch):
    owner, calls = _owner(monkeypatch, target=_image(width=32))
    with pytest.raises(ValueError, match="native Real width"):
        _prepare(owner, _payload((_image(0, 2, width=32), _image(1, 2, width=32))))
    assert calls == [("capture", _image(width=32)), ("validate", _image(width=32))]


@pytest.mark.parametrize("capacity,bound", [(39, 512), (256, 79)])
def test_capacity_and_aggregate_bounds_are_checked_before_native_codec(monkeypatch, capacity, bound):
    owner, calls = _owner(monkeypatch, capacity=capacity, bound=bound)
    with pytest.raises(ValueError, match="chosen resource capacity"):
        _prepare(owner, _payload())
    assert calls == []


def test_same_rank_and_legacy_absence_keep_existing_codec_routes(monkeypatch):
    owner, calls = _owner(monkeypatch)
    current = _image(records=((b"opaque", -0.),))
    assert codec.prepare_checkpoint_program_diagnostics(owner, _payload((current,))) == current
    assert codec.prepare_checkpoint_program_diagnostics(owner, {}) == b""
    assert calls == [("validate", current), ("validate", b"")]
