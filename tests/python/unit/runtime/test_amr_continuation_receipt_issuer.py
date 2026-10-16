"""Independent forest receipts distinguish own allocation from imported domains."""
import struct
from types import SimpleNamespace

import pytest

from tests.python.support import amr_continuation_receipts as reader


def _word(value):
    return struct.pack("<Q", value % (1 << 64))


def _blob(value):
    value = value.encode() if isinstance(value, str) else value
    return _word(len(value)) + value


def _image(magic, identity, words):
    return _blob(magic) + _blob(identity) + _word(2) + _word(1) + b"".join(map(_word, words))


def _forest(monkeypatch, *, high_water, own_version=2, imported_version=19, primary=1,
            imported_realm=1, second_issuer=(21, 22)):
    # Each realm has a distinct transport identity. The second domain is retained
    # from a different allocator and must not advance the first allocator's counter.
    child = _image("POPSCONT3", "child", [
        2, primary, 11, 12, 1, *second_issuer, 1,
        2, 0, 1, 1, 0, own_version, imported_realm, 1, 1, 1, imported_version,
        0, 0, 0, 0, 3, 2, 5, 0, 5, 1,
    ])
    root = _image("POPSCONT1", "root", [0, 1, 0])
    def domain(family):
        return dict(vtype="index_set", ownership="replicated", capacity=1, family=family)
    contract = dict(root_schema=dict(read=dict(vtype="int64")), level_schema=dict(read=dict(
        vtype="product", members=(("own", domain(0)), ("imported", domain(1))))))
    node = SimpleNamespace(op="post_synchronization", attrs=dict(continuation_forest=contract))
    spec = SimpleNamespace(case=SimpleNamespace(_time=SimpleNamespace(_values=[node])))
    image = b"POPSANDF" + _word(0) + _word(0) + _word(2)
    for level, counter, identity, payload in ((-1, 0, "root", root), (0, high_water, "child", child)):
        image += b"".join(map(_word, (level, 1, 1, counter)))
        image += _blob(identity) + _blob(bytes(32)) + _blob(payload)
    # Test the exact forest suffix independently of the preceding AMR/flux codec.
    monkeypatch.setattr(reader, "fragments", lambda image, _with_cursor: ([], [], 8))
    return spec, image


def test_imported_domain_does_not_advance_primary_allocator(monkeypatch):
    spec, image = _forest(monkeypatch, high_water=2)
    rows = reader.forest(spec, image)["records"]
    assert rows[1]["high_water"] == 2
    assert rows[1]["root"]["imported"]["version"] == 19


@pytest.mark.parametrize("high_water", (0, 1))
def test_primary_allocator_regression_is_still_rejected(monkeypatch, high_water):
    spec, image = _forest(monkeypatch, high_water=high_water)
    with pytest.raises(AssertionError, match="primary allocator counter regressed"):
        reader.forest(spec, image)


def test_import_only_image_keeps_its_zero_allocation_counter(monkeypatch):
    spec, image = _forest(monkeypatch, high_water=0, primary=0)
    assert reader.forest(spec, image)["records"][1]["high_water"] == 0


@pytest.mark.parametrize("mutation", (
    dict(imported_realm=2), dict(primary=3), dict(second_issuer=(11, 12)),
))
def test_missing_or_forged_realm_is_still_rejected(monkeypatch, mutation):
    spec, image = _forest(monkeypatch, high_water=19, **mutation)
    with pytest.raises(AssertionError):
        reader.forest(spec, image)


@pytest.mark.parametrize("magic, words", (
    ("POPSCONT1", [1, 1, 1, 0, 7, 5, 0]),
    ("POPSCONT2", [1, 11, 12, 1, 1, 0, 1, 1, 0, 7, 0, 0, 5, 0]),
))
def test_previous_wire_allocator_counter_remains_exact(magic, words):
    decoded = reader._decode(None, _image(magic, "legacy", words), schema=dict(
        vtype="index_set", ownership="replicated", capacity=1, family=0))
    assert reader._issuer_high_water(decoded) == 7
