"""Independent CONT5 framing preserves a producer descriptor before Field storage."""
import struct

import pytest

from tests.python.support.api_v02_receipts import _decode
from tests.python.unit.runtime.test_continuation_receipt_v4 import blob, word


def image(*, owner=0, level=2, present=1):
    header = blob(b"POPSCONT5") + blob(b"schema") + word(2) + word(7)
    header += word(0) + word(0)*7  # no expression table; empty authority tables
    descriptor = word(present)
    if present == 1:
        descriptor += blob(b"program") + blob(b"history/time=0.125/stage=2")
        descriptor += word(owner) + word(level) + word(7) + word(19)
    words = [8, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 8, 0, 3]
    field = b"".join(map(word, words)) + struct.pack("<ddd", -0., 2., 3.)
    return header + descriptor + field


def test_cont5_preserves_historical_producer_descriptor_and_bits():
    field = _decode(None, image(), schema=dict(vtype="state"))["root"]
    assert field["field_support"] == dict(program_identity="program",
        origin_point="history/time=0.125/stage=2", program_owner=0, producer_level=2,
        topology_epoch=7, materialization_generation=19)
    assert field["field_expression"] is None
    assert field["patches"][0].view("<u8").tolist() == [1 << 63, 1 << 62, 0x4008000000000000]


@pytest.mark.parametrize("corruption", ("tag", "owner", "level", "truncated", "absent"))
def test_cont5_rejects_invalid_producer_descriptor(corruption):
    payload = {"tag": lambda: image(present=2), "owner": lambda: image(owner=1 << 31),
        "level": lambda: image(level=1 << 31), "truncated": lambda: image()[:-1],
        "absent": lambda: image(present=0)}[corruption]()
    with pytest.raises(AssertionError):
        _decode(None, payload, schema=dict(vtype="state"))
