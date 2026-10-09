"""Test-only wire: actual structured codec, no Native/MPI qualification."""
import copy
import struct
import sys

import numpy as np
import pytest

from pops._native_collectives import decode_value, encode_value
from tests.python.support import spatial_interaction_receipts as helper


def exact(left, right):
    assert type(left) is type(right)
    if type(left) is np.ndarray:
        assert left.dtype.str == right.dtype.str and left.shape == right.shape
        assert left.tobytes(order="C") == right.tobytes(order="C")
    elif type(left) is float:
        assert struct.pack(">d", left) == struct.pack(">d", right)
    elif type(left) is dict:
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif type(left) in (tuple, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            exact(a, b)
    else:
        assert left == right


def fixture_payload(rank):
    values = np.array([0., -0., np.inf, -np.inf, np.nan, 1.25], dtype="<f8").reshape(1, 2, 3)
    piece = dict(values=values, lower=(0, 1), upper=(3, 3), global_box_index=7,
                 owner_rank=rank, replicated=False)
    return dict(pieces={"rho": [piece], "active": [dict(piece, values=np.array([True, False]))],
                        "kappa": [dict(piece, values=np.array([.25, 1.], dtype="<f4"))],
                        "empty": []},
                history_values={"rho_slot_1": np.asfortranarray(values.astype(">f8"))},
                history_meta={"rho": dict(slot_dt=[.01, -0.], sample_hex="001122", initialized=True)},
                auxiliary=(b"POPSHID1\0\xff", b""))


@pytest.mark.parametrize("rank", (0, 1))
def test_actual_native_value_codec_accepts_wire_and_preserves_every_byte(rank):
    payload = fixture_payload(rank)
    with pytest.raises(TypeError, match="ndarray"):
        encode_value({"pieces": payload["pieces"]})
    with pytest.raises(TypeError, match="refuse bytes"):
        encode_value(payload["auxiliary"])
    wire = helper.array_wire_encode(payload)
    decoded = helper.array_wire_decode(decode_value(encode_value(wire)))
    exact(payload, decoded)
    decoded["pieces"]["rho"][0]["values"][0, 0, 0] = 99.
    assert payload["pieces"]["rho"][0]["values"][0, 0, 0] == 0.


@pytest.mark.parametrize("dtype", ("<f4", ">f4", "<f8", ">f8", "|b1", "|i1", "<u8", ">i8"))
@pytest.mark.parametrize("shape", ((), (0,), (2, 0, 3), (2, 3)))
def test_closed_plain_arrays_endian_zero_size_and_strided_input(dtype, shape):
    size = int(np.prod(shape)) if shape else 1
    data = np.arange(size, dtype=np.int64).astype(dtype).reshape(shape)
    if shape == (2, 3):
        data = data[:, ::-1]
    exact(data, helper.array_wire_decode(decode_value(encode_value(helper.array_wire_encode(data)))))


@pytest.mark.parametrize("value", (np.array([object()], dtype=object), np.array(["x"]),
                                    np.zeros(1, dtype=[("x", "f8")]), np.array([1.], dtype="f2"),
                                    {1: np.ones(2)}, np.float64(1.)))
def test_encoder_refuses_opaque_and_nonclosed_types(value):
    with pytest.raises((TypeError, ValueError)):
        helper.array_wire_encode(value)


@pytest.mark.parametrize("change", (
    {"dtype": "O"}, {"dtype": "float64"}, {"dtype": "=f8"}, {"dtype": True},
    {"shape": [True]}, {"shape": [1.]}, {"shape": [-1]}, {"shape": [sys.maxsize, 2]},
    {"shape": [0, sys.maxsize, 2]}, {"shape": [1]*33},
    {"Cbytes": "00"}, {"Cbytes": "00"*17}, {"Cbytes": "00 "*16},
    {"Cbytes": "AB"*16}, {"Cbytes": "gg"*16}, {"Cbytes": b"00"*16},
    {"extra": 1},
))
def test_decoder_refuses_before_array_allocation(monkeypatch, change):
    wire = helper.array_wire_encode(np.zeros(2))
    wire["value"][1].update(change)
    def forbidden(*args, **kwargs):
        pytest.fail("malformed array reached allocation")
    monkeypatch.setattr(helper.np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        helper.array_wire_decode(wire)


@pytest.mark.parametrize("wire", (
    {"contract": "unknown", "value": ["scalar", 1]},
    {"contract": helper._WIRE_CONTRACT, "value": ["pickle", "x"]},
    {"contract": helper._WIRE_CONTRACT, "value": ["scalar", 1.0]},
    {"contract": helper._WIRE_CONTRACT, "value": ["map", [["a", ["scalar", 1]], ["a", ["scalar", 2]]]]},
    {"contract": helper._WIRE_CONTRACT, "value": ["map", [["", ["scalar", 1]]]]},
))
def test_decoder_refuses_unknown_contract_tags_and_duplicate_keys(wire):
    with pytest.raises(ValueError):
        helper.array_wire_decode(wire)


def test_four_actual_piece_schemas_two_rank_transport_and_auxiliary(monkeypatch):
    import pops._native_collectives as native
    from tests.python.support.collective_checks import collective_call
    payloads = [fixture_payload(0), fixture_payload(1)]
    events = []
    encode, decode = helper.array_wire_encode, helper.array_wire_decode
    def encoded(value):
        events.append("encode")
        return encode(value)
    def transported(world, value):
        events.append("allgather")
        # Production control-value codec is exercised for each source rank.
        return tuple(decode_value(encode_value(encode(payload))) for payload in payloads)
    def decoded(value):
        events.append("decode")
        return decode(value)
    monkeypatch.setattr(helper, "array_wire_encode", encoded)
    monkeypatch.setattr(helper, "array_wire_decode", decoded)
    monkeypatch.setattr(native, "allgather_value", transported)
    # world=None gives genuine fixture local convergence; the payload collective
    # seam alone is explicit host transport, not an MPI communicator substitute.
    assert helper.collective_call is collective_call
    result = helper._allgather_array_values(None, payloads[0])
    assert events == ["encode", "allgather", "decode", "decode"]
    for before, after in zip(payloads, result, strict=True):
        exact(before, after)
    events.clear()
    with pytest.raises(AssertionError, match="unsupported"):
        helper._allgather_array_values(None, object())
    assert events == ["encode"]


def test_reconstructed_archive_arrays_and_manifests_remain_exact():
    local = fixture_payload(0)
    local["pieces"].pop("empty")
    local.pop("auxiliary")
    geometry = dict(coverage=np.zeros((2, 3), dtype=bool), valid_cells=np.ones((2, 3), dtype=bool),
                    cell_volumes=np.ones((2, 3)), boxes=[(0, 0, 3, 2)], cell_shape=(3, 2))
    image = dict(clock=(0., 0), epoch=0, auxiliary=((b"opaque",),),
                 levels=[dict(level=0, origin=(.3, -.4), spacing=(1., 2.), geometry=geometry,
                              copies=(local,))])
    rebuilt = copy.copy(image)
    rebuilt["levels"] = [dict(image["levels"][0], copies=(helper.array_wire_decode(
        decode_value(encode_value(helper.array_wire_encode(local)))),))]
    exact(helper.arrays(image), helper.arrays(rebuilt))



def test_one_peer_encode_failure_votes_before_any_payload_transport(monkeypatch):
    import pops._native_collectives as native
    calls = []
    def failed_vote(world, value):
        calls.append(value)
        assert value is None, "entered payload transport despite peer encoder failure"
        return (None, ("TypeError", "rank-one unsupported array wire value", False))
    monkeypatch.setattr(native, "allgather_value", failed_vote)
    with pytest.raises(AssertionError, match="rank-one"):
        helper._allgather_array_values(object(), fixture_payload(0))
    assert calls == [None]


@pytest.mark.parametrize("word,float_dtype,patterns", (
    ("<u4", "<f4", (0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234)),
    ("<u8", "<f8", (0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                      0x7ff8000000001234, 0x7ff0000000001234)),
))
def test_float32_float64_nan_payload_and_signed_zero_bits(word, float_dtype, patterns):
    original = np.array(patterns, dtype=word).view(float_dtype)
    result = helper.array_wire_decode(decode_value(encode_value(helper.array_wire_encode(original))))
    exact(original, result)
