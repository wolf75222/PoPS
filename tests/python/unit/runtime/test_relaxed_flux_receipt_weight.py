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


def _qualified_face_image(version, weight, *, affine=0):
    """Independent wire witness for the conservative qualification's face reader."""
    def word(n):
        return struct.pack("<q", n)
    def real(n):
        return struct.pack("<d", n)
    def text(s):
        raw = s.encode()
        return word(len(raw)) + raw
    head = version + word(2) + word(0) + word(0) + word(0) + word(1)
    head += bytes(256-len(head))
    state = ("pops.handle.v1::case:qualified-amr-diffusion-case/block:heat/"
             "model_definition:qualified-amr-diffusion-model@pops.module%3Asha256%3A"
             + "a"*64 + "::state::U")
    row = text("heat") + text(state) + word(0) + word(1) + word(0) + word(0)
    row += word(0)*6 + word(0) + word(1) + real(0.)
    row += text("pops.program-flux-family.v1:sha256:" + "b"*64)
    row += text("pops.program-flux-expression.v1/provider/1/rhs/7") + word(1) + word(0)*2
    if version in (b"POPSANDE", b"POPSANDF"):
        encoded = _image(weight.numerator, weight.denominator)
        row += word(len(encoded)) + encoded
    else:
        row += word(weight.numerator) + word(weight.denominator)
    row += word(0) + word(1)*3 + real(.1) + real(.125)
    if version >= b"POPSANDC":
        row += word(affine) + real(0.)*4
    row += word(1) + real(3.5)
    return head + text("budget") + text("coupling") + word(1) + row + word(0)


def _qualification_reader(monkeypatch):
    from tests.python.integration.runtime import test_amr_transport_diffusion_qualification as reader
    # The prefix inspector has its own tests; isolate exact face framing here.
    monkeypatch.setattr(reader, "accepted_tagging_hysteresis_span",
                        lambda image, dimension: (b"", 256))
    return reader._accepted_face_fragments


@pytest.mark.parametrize("version", (b"POPSANDA", b"POPSANDB", b"POPSANDC",
                                     b"POPSANDD", b"POPSANDE", b"POPSANDF"))
def test_conservative_qualification_reads_exact_weight_and_metric(monkeypatch, version):
    weight = Fraction(2**130+1, 2**129) if version >= b"POPSANDE" else Fraction(1, 3)
    rows, digest = _qualification_reader(monkeypatch)(_qualified_face_image(version, weight))
    assert len(rows) == 1 and len(digest) == 64
    row = rows[0]
    assert row["weight"] == weight and row["levels"] == (0, 1)
    assert row["measure"] == .125 and row["flux"] == 3.5 and row["attempt"] == 1


@pytest.mark.parametrize("change", ("truncated", "noncanonical_affine"))
def test_conservative_qualification_rejects_invalid_face_payload(monkeypatch, change):
    payload = _qualified_face_image(b"POPSANDF", Fraction(1, 3),
                                    affine=int(change == "noncanonical_affine"))
    if change == "truncated":
        payload = payload[:-1]
    with pytest.raises(AssertionError):
        _qualification_reader(monkeypatch)(payload)
