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


def _empty_face_ledger(monkeypatch):
    from tests.python.support import relaxed_flux_receipts as reader
    # Isolate the face-ledger framing from the preceding AMR topology codec.
    monkeypatch.setattr(reader, "accepted_tagging_hysteresis_span",
                        lambda image, dimension: (b"", 256))
    def text(value):
        return struct.pack("<Q", len(value)) + value
    image = b"POPSANDF" + struct.pack("<Q", 2) + bytes(240)
    image += text(b"artifact") + text(b"program") + struct.pack("<QQQ", 0, 0, 0)
    return reader, image


def test_forest_cursor_accepts_a_well_framed_empty_face_ledger(monkeypatch):
    reader, image = _empty_face_ledger(monkeypatch)
    faces, interfaces, cursor = reader.fragments(image, _with_cursor=True)
    assert faces == interfaces == [] and cursor == len(image)


def test_flux_qualification_still_requires_actual_faces(monkeypatch):
    reader, image = _empty_face_ledger(monkeypatch)
    for read in (reader.fragments, reader.interface_fragments):
        with pytest.raises(AssertionError):
            read(image)


def test_empty_face_ledger_cursor_rejects_truncated_interface_framing(monkeypatch):
    reader, image = _empty_face_ledger(monkeypatch)
    with pytest.raises(AssertionError):
        reader.fragments(image[:-1], _with_cursor=True)
