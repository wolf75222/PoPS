"""Byte-for-byte parity of the Python and C++ identity implementations."""
from __future__ import annotations

import pytest

from pops import _pops
from pops.identity import canonical_bytes, canonical_sha256


VALUES = [
    None,
    False,
    True,
    0,
    23,
    24,
    -1,
    -24,
    -25,
    (1 << 63) - 1,
    -(1 << 63),
    b"\x00PoPS\xff",
    "é",
    "東京",
    [1, "two", None],
    (b"x", {"aa": 2, "b": 1, "é": 3}),
    {3, 1, 2},
    frozenset(("long", "x")),
    {"nested": [{"set": frozenset((1, 24, "a"))}], "ok": True},
]


@pytest.mark.parametrize("value", VALUES)
def test_native_encoder_and_sha_match_python(value):
    assert _pops._identity_canonical_bytes(value) == canonical_bytes(value)
    assert _pops._identity_sha256(value) == canonical_sha256(value)


@pytest.mark.parametrize("value", [1.0, object(), bytearray(b"x"), {1: "bad"}])
def test_native_and_python_reject_the_same_unsupported_vocabulary(value):
    with pytest.raises((TypeError, OverflowError, ValueError)):
        canonical_bytes(value)
    with pytest.raises((TypeError, OverflowError, ValueError)):
        _pops._identity_canonical_bytes(value)


@pytest.mark.parametrize("value", [1 << 63, -(1 << 63) - 1])
def test_native_and_python_reject_out_of_range_integers(value):
    with pytest.raises(OverflowError):
        canonical_bytes(value)
    with pytest.raises(OverflowError):
        _pops._identity_canonical_bytes(value)


def test_native_and_python_reject_cycles():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cycle"):
        canonical_bytes(value)
    with pytest.raises(ValueError, match="cycle"):
        _pops._identity_canonical_bytes(value)
