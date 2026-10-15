from fractions import Fraction
import struct
import pytest
from tests.python.support.relaxed_flux_receipts import exact_weight


def _image(numerator, denominator):
    def integer(value):
        return value.to_bytes(4*((value.bit_length()+31)//32), "little")
    n, d = integer(abs(numerator)), integer(denominator)
    return b"EW1"+bytes([numerator < 0])+struct.pack("<QQ", len(n)//4, len(d)//4)+n+d


@pytest.mark.parametrize("n,d", ((0, 1), (1, 3), (-17, 13), (2**130+1, 2**129)))
def test_independent_weight_reader_preserves_arbitrary_precision(n, d):
    assert exact_weight(_image(n, d)) == Fraction(n, d)


@pytest.mark.parametrize("payload", (_image(2, 4), _image(1, 0),
    _image(0, 2), _image(1, 3)+b"\0", b"EW1"))
def test_independent_weight_reader_rejects_noncanonical_or_truncated_images(payload):
    with pytest.raises(AssertionError):
        exact_weight(payload)
