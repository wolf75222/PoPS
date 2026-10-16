"""CONT4 independently preserves expression aliases and rejects corrupt tables."""
import struct

import pytest

from tests.python.support.api_v02_receipts import _decode
from tests.python.support.field_expression_receipts import decode_field_expressions


def word(n):
    return struct.pack("<Q", n)


def blob(data):
    return word(len(data)) + data


def expression_image(*, coefficient=.25):
    # One common source basis even on a rank with no locally owned faces.
    ledger = b"POPSEX02" + word(0) * 3
    basis = blob(b"native/evaluation/7") + word(0) + blob(ledger) + word(0)
    image = word(0x3158504553504F50) + word(1) + blob(basis) + word(2)
    for weight in (1., coefficient):
        image += word(1) + word(1) + word(0) + struct.pack("<d", weight)
    return image + word(2) + word(1) + word(2)


def continuation_image(*, last_slot=2, coefficient=.25):
    image = blob(b"POPSCONT4") + blob(b"declared-schema") + word(2) + word(7)
    image += blob(expression_image(coefficient=coefficient)) + word(0) * 7
    image += word(3) + word(3)
    # 1 component, one valid cell, one x halo: three exact floating point cells.
    words = [8, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 8, 0, 3]
    field = b"".join(map(word, words)) + struct.pack("<ddd", -0., 2., 3.)
    return image + field + word(1) + field + word(1) + field + word(last_slot)


SCHEMA = dict(vtype="product", members=[(name, dict(vtype="state")) for name in ("a", "b", "c")])


def test_cont4_retains_expression_and_basis_aliases():
    packet = _decode(None, continuation_image(), schema=SCHEMA)
    a, b, c = (packet["root"][name]["field_expression"] for name in ("a", "b", "c"))
    assert a is b and a is not c
    assert a["terms"][0]["basis"] is c["terms"][0]["basis"]
    assert c["terms"][0]["coefficient"] == .25
    assert c["terms"][0]["basis"]["records"] == []


@pytest.mark.parametrize("slot", (0, 3))
def test_cont4_rejects_unreferenced_or_absent_expression(slot):
    with pytest.raises(AssertionError):
        _decode(None, continuation_image(last_slot=slot), schema=SCHEMA)


@pytest.mark.parametrize("coefficient", (0., float("nan"), float("inf")))
def test_cont4_rejects_invalid_exact_coefficient(coefficient):
    with pytest.raises(AssertionError):
        _decode(None, continuation_image(coefficient=coefficient), schema=SCHEMA)


@pytest.mark.parametrize("suffix", (b"", b"\0"))
def test_expression_reader_rejects_truncation_and_trailing_bytes(suffix):
    image = expression_image()
    corrupted = image[:-1] if not suffix else image + suffix
    with pytest.raises(AssertionError):
        decode_field_expressions(corrupted)
