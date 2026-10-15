"""Byte-level counterexamples for the independent CONT3 receipt inspector."""
import struct

import numpy as np
import pytest

from tests.python.support.api_v02_receipts import _decode


def _word(value):
    return struct.pack("<Q", value % (1 << 64))


def _text(value):
    value = value.encode()
    return _word(len(value))+value


def _header(wire):
    return _text(wire)+_text("declared-schema")+_word(2)+_word(7)


def _multiple_issuers(*, primary=1, second_issuer=(21, 22), second_registry=1):
    words = [2, primary, 11, 12, 1, *second_issuer, 1,
             2, 0, 2, 2, 0, 1, 1, 2, 2, 1, 1,
             2, 1, 31, 32, 4096, 32, 41, 42, 4096, 32,
             2, 0, 1, 4, 2, 3, 0, 1, 2, second_registry, 1, 5, 2, 3, 0, 1, 2,
             0, 3, 4, 7, 0, 2, 1, 1, 1, 2, 7, 1, 2, 1, 3, 1, 4, 9, 0, 9, 1]
    integer = dict(vtype="int64")
    members = [("a", dict(vtype="array", family=0, capacity=2, ownership="replicated", element=integer)),
               ("b", dict(vtype="array", family=1, capacity=2, ownership="replicated", element=integer)),
               ("c", dict(vtype="index_set", family=4, capacity=2, ownership="contiguous_distributed")),
               ("d", dict(vtype="index_set", family=5, capacity=2, ownership="contiguous_distributed"))]
    return _header("POPSCONT3")+b"".join(map(_word, words)), dict(vtype="product", members=members)


def test_cont3_keeps_equal_ordinals_from_distinct_issuers_distinct():
    image, schema = _multiple_issuers()
    packet = _decode(None, image, schema=schema)
    assert packet["wire"] == "POPSCONT3"
    assert packet["primary_finite"] == 0 and packet["primary_registry"] == 0
    assert packet["root"]["a"]["values"] == [1, 2]
    assert packet["root"]["b"]["values"] == [3, 4]
    assert packet["root"]["a"]["domain"]["realm"] != packet["root"]["b"]["domain"]["realm"]
    left, right = packet["root"]["c"], packet["root"]["d"]
    assert left["version"] == right["version"] == 1
    assert left["issuer"] == (31, 32) and right["issuer"] == (41, 42)


@pytest.mark.parametrize("mutation", (
    dict(primary=3), dict(second_issuer=(11, 12)), dict(second_registry=2),
))
def test_cont3_rejects_unbound_or_duplicate_authorities(mutation):
    image, schema = _multiple_issuers(**mutation)
    with pytest.raises(AssertionError):
        _decode(None, image, schema=schema)


def test_cont3_rejects_truncation_and_trailing_payload():
    image, schema = _multiple_issuers()
    for payload in (image[:-1], image+b"\0"):
        with pytest.raises(AssertionError):
            _decode(None, payload, schema=schema)


@pytest.mark.parametrize("wire", ("POPSCONT1", "POPSCONT2", "POPSCONT3"))
def test_previous_wires_and_empty_cont3_keep_scalar_bits(wire):
    tables = {"POPSCONT1": [0], "POPSCONT2": [0, 0, 0, 0],
              "POPSCONT3": [0, 0, 0, 0, 0, 0, 0]}[wire]
    image = _header(wire)+b"".join(map(_word, tables))+_word(2)+struct.pack("<d", -0.)
    decoded = _decode(None, image, schema=dict(vtype="scalar"))
    assert struct.pack("<d", decoded["root"]) == struct.pack("<d", -0.)


@pytest.mark.parametrize("vtype", ("state", "scalar_field"))
def test_field_receipt_preserves_rank_layout_and_payload_bits(vtype):
    # Two-dimensional one-cell valid box, one ghost in x, three actual storage cells.
    words = [0, 8, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 8, 0, 3]
    payload = _header("POPSCONT1")+b"".join(map(_word, words))
    payload += b"".join(struct.pack("<d", x) for x in (-0., 2., 3.))
    field = _decode(None, payload, schema=dict(vtype=vtype))["root"]
    assert field["axes"] == [(1, 0, 0, 1), (0, 0, 0, 1)]
    np.testing.assert_array_equal(field["patches"][0], [-0., 2., 3.])
    assert np.signbit(field["patches"][0][0])
