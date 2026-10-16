"""Independent receipt inspection across the exact-rational wire boundary."""
from fractions import Fraction
import pytest
from tests.python.support.history_flux_receipts import decode_exact_weight


def _wire(numerator, denominator, *, sign=0):
    def words(value):
        return value.to_bytes(4*((value.bit_length()+31)//32), "little")
    n,d=words(numerator),words(denominator)
    return b"EW1"+bytes((sign,))+(len(n)//4).to_bytes(8,"little")+(len(d)//4).to_bytes(8,"little")+n+d


@pytest.mark.parametrize("numerator,denominator,sign", ((0,1,0),(1,3,0),(23,12,1),
    (2**2048+1,2**1074,0),(1,2**1074,1)))
def test_exact_weight_inspector_never_rounds_a_native_rational(numerator,denominator,sign):
    expected=Fraction(-numerator if sign else numerator,denominator)
    assert decode_exact_weight(_wire(numerator,denominator,sign=sign))==expected


@pytest.mark.parametrize("wire", (_wire(2,4),_wire(0,1,sign=1),_wire(1,0),
    _wire(1,3,sign=2),_wire(1,3)+b"\0",b"EW1",_wire(1,3)[:-1],
    b"EW1\0"+(2).to_bytes(8,"little")+(1).to_bytes(8,"little")+(1).to_bytes(8,"little")+(3).to_bytes(4,"little")))
def test_exact_weight_inspector_rejects_noncanonical_and_truncated_wires(wire):
    with pytest.raises(AssertionError): decode_exact_weight(wire)
