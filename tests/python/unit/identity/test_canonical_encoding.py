"""Golden and negative tests for the strict deterministic CBOR identity codec."""
from __future__ import annotations

import hashlib

import pytest

from pops.identity import canonical_bytes, canonical_sha256


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "f6"),
        (False, "f4"),
        (True, "f5"),
        (0, "00"),
        (23, "17"),
        (24, "1818"),
        (-1, "20"),
        (-24, "37"),
        (-25, "3818"),
        (b"PoPS", "44506f5053"),
        ("é", "62c3a9"),
    ],
)
def test_scalar_golden_bytes(value, expected):
    assert canonical_bytes(value).hex() == expected


def test_signed_int64_boundaries_and_overflow():
    assert canonical_bytes((1 << 63) - 1).hex() == "1b7fffffffffffffff"
    assert canonical_bytes(-(1 << 63)).hex() == "3b7fffffffffffffff"
    with pytest.raises(OverflowError, match="signed int64"):
        canonical_bytes(1 << 63)
    with pytest.raises(OverflowError, match="signed int64"):
        canonical_bytes(-(1 << 63) - 1)


def test_ordered_sequences_preserve_order_and_share_one_sequence_encoding():
    assert canonical_bytes([1, 2]) == canonical_bytes((1, 2))
    assert canonical_bytes([1, 2]) != canonical_bytes([2, 1])
    assert canonical_bytes([1, 2]).hex() == "820102"


def test_map_order_is_length_first_then_lexicographic_on_encoded_keys():
    left = {"é": 3, "aa": 2, "b": 1}
    right = {"b": 1, "aa": 2, "é": 3}
    encoded = canonical_bytes(left)
    assert encoded == canonical_bytes(right)
    # b has the shortest encoded key. aa and é tie in encoded length, then compare by bytes.
    assert encoded.hex() == "a36162016261610262c3a903"


def test_map_keys_are_strictly_strings():
    with pytest.raises(TypeError, match="map key.*string"):
        canonical_bytes({1: "one"})


def test_set_uses_tag_258_and_canonical_member_order():
    expected = bytes.fromhex("d9010283010203")
    assert canonical_bytes({3, 1, 2}) == expected
    assert canonical_bytes(frozenset((2, 3, 1))) == expected


def test_typed_cbor_prevents_the_old_tag_mapping_collision():
    scalar = canonical_bytes(1)
    tag_shaped_map = canonical_bytes({"$scalar": {"kind": "integer", "value": "1"}})
    assert scalar != tag_shaped_map
    assert scalar == b"\x01"
    assert tag_shaped_map.startswith(b"\xa1")


def test_unicode_is_encoded_as_exact_utf8():
    assert canonical_bytes("e\u0301") != canonical_bytes("é")
    assert canonical_bytes("東京").hex() == "66e69db1e4baac"
    with pytest.raises(ValueError, match="valid Unicode"):
        canonical_bytes("\ud800")


@pytest.mark.parametrize("value", [0.0, -0.0, float("inf"), float("-inf"), float("nan")])
def test_every_float_is_refused(value):
    with pytest.raises(TypeError, match=r"float\.hex\(\)"):
        canonical_bytes(value)


def test_opaque_values_and_cycles_are_refused():
    class Opaque:
        pass

    with pytest.raises(TypeError, match="opaque Opaque"):
        canonical_bytes(Opaque())

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="reference cycle"):
        canonical_bytes(cyclic)


def test_canonical_sha256_hashes_the_exact_canonical_bytes():
    value = {"components": ["rho", "rho_u"], "owner": "fluid"}
    expected = hashlib.sha256(canonical_bytes(value)).hexdigest()
    assert canonical_sha256(value) == expected
    assert len(expected) == 64
