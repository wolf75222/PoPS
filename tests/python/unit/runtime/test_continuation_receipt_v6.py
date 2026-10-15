"""Independent CONT6 framing: scientific provider and frozen representation bytes."""
import json
import struct
from types import MappingProxyType

import pytest

from tests.python.support.api_v02_receipts import _decode
from tests.python.unit.runtime.test_continuation_receipt_v4 import blob, word

AUTHORITY = {"kind": "field_problem", "version": 1, "problem": "problem:A",
             "unknown": "potential", "components": 1, "space": None, "projection": "component"}
PROVIDER = json.dumps(AUTHORITY, sort_keys=True, separators=(",", ":"))


def image(*, provider=PROVIDER, owner=(1 << 64)-1, support_provider=None,
          support=1, contract="graph/capture:a=1", origin="evaluation:7", tag=1):
    header = blob(b"POPSCONT6") + blob(b"schema") + word(2) + word(7)
    header += word(0) + word(0)*7
    traits = blob(provider.encode()) + word(tag)
    if tag == 1:
        traits += blob(contract.encode()) + blob(origin.encode())
    descriptor = word(support)
    if support == 1:
        descriptor += blob(b"program") + blob(b"history/time=0.125/stage=2")
        descriptor += word(owner) + word(2) + word(7) + word(19)
        descriptor += blob((provider if support_provider is None else support_provider).encode())
    words = [8, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 8, 0, 3]
    field = b"".join(map(word, words)) + struct.pack("<ddd", -0., 2., 3.)
    return header + traits + descriptor + field


def test_cont6_independent_provider_and_closed_capture_are_distinct():
    schema = dict(vtype="scalar_field", field_authority=AUTHORITY)
    first = _decode(None, image(), schema=schema)["root"]
    second = _decode(None, image(contract="graph/capture:a=2"), schema=schema)["root"]
    assert first["field_provider"] == PROVIDER
    assert first["field_support"]["program_owner"] == -1
    assert first["field_support"]["field_provider"] == PROVIDER
    assert first["field_support"]["origin_point"] == "history/time=0.125/stage=2"
    assert first["field_representation"] != second["field_representation"]
    assert first["patches"][0].view("<u8").tolist() == [1 << 63, 1 << 62, 0x4008000000000000]


def test_cont6_state_representation_without_support_or_flux():
    result = _decode(None, image(provider="", owner=0, support=0), schema=dict(vtype="state"))
    assert result["root"]["field_support"] is None
    assert result["root"]["field_expression"] is None
    assert result["root"]["field_representation"]["origin"] == "evaluation:7"


def test_cont6_frozen_provider_schema_has_identical_wire_identity():
    authority = {**AUTHORITY, "space": {"dimension": 2, "axes": ("x", "y")}}
    provider = json.dumps(authority, sort_keys=True, separators=(",", ":"))
    frozen = MappingProxyType({**authority, "space": MappingProxyType(authority["space"])})
    result = _decode(None, image(provider=provider),
                     schema=MappingProxyType(dict(vtype="scalar_field", field_authority=frozen)))
    assert result["root"]["field_provider"] == provider


@pytest.mark.parametrize("change", [dict(owner=0), dict(support_provider="foreign"),
    dict(support=0), dict(contract=""), dict(origin=""), dict(tag=2),
    dict(provider="", owner=0, support=0, tag=0)])
def test_cont6_rejects_invalid_authority_or_noncanonical_envelope(change):
    schema = dict(vtype="scalar_field", field_authority=AUTHORITY)
    if change.get("provider") == "":
        schema = dict(vtype="state")
    with pytest.raises(AssertionError):
        _decode(None, image(**change), schema=schema)


@pytest.mark.parametrize("key,value", [("problem", "problem:B"), ("unknown", "other")])
def test_cont6_same_name_or_shape_never_substitutes_a_provider(key, value):
    with pytest.raises(AssertionError, match="provider differs"):
        _decode(None, image(), schema=dict(vtype="scalar_field", field_authority={**AUTHORITY, key: value}))


def test_cont6_every_truncated_prefix_is_rejected():
    data = image()
    for end in range(len(data)):
        with pytest.raises((AssertionError, UnicodeDecodeError)):
            _decode(None, data[:end], schema=dict(vtype="scalar_field", field_authority=AUTHORITY))
