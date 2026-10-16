"""Independent wire fixtures: no production codec is used to construct expected bytes."""
import struct
import pytest
from tests.python.support.history_field_metadata_receipts import unwrap_history_field_metadata


def _word(value):
    return (value % (1 << 64)).to_bytes(8, "little")


def _text(value):
    value = value.encode()
    return _word(len(value))+value


def _bits(value):
    return int.from_bytes(struct.pack("<d", value), "little")


def _wire(*, flux=b"", sample_kind=2, support_index=1, trailing=b"", program="exact.program"):
    support = (_text(program)+_text("old.fine.point")+_word(-1)+_word(2)
               +_word(7)+_word(11)+_text("problem/unknown"))
    metadata = (_word(1)+support+_word(1)+_text("Q.closed")+_text("old.origin")
                +_word(1)+_text("coarse:ring")+_word(2)
                +_word(support_index)+_word(1)+_word(1)+_word(1))
    sample = b"".join(_word(x) for x in
                      (sample_kind, _bits(.75), _bits(.25), 1, 0, 0, 0, 0))
    return (b"POPSHFM1"+_word(len(flux))+flux+_word(len(metadata))+metadata
            +_word(1)+_text("coarse:ring")+_word(2)+sample*2+trailing)


def test_no_flux_metadata_preserves_foreign_support_origin_and_aliases():
    flux, fields, samples = unwrap_history_field_metadata(_wire(), program="exact.program")
    assert flux == b""
    first, second = fields["coarse:ring"]
    assert first["support"] is second["support"]
    assert first["representation"] is second["representation"]
    assert first["support"]["program_owner"] == -1
    assert first["support"]["producer_level"] == 2
    assert first["support"]["origin_point"] == "old.fine.point"
    assert first["representation"] == dict(contract="Q.closed", origin="old.origin")
    assert samples["coarse:ring"][0]["start_bits"] == _bits(.75)


def test_flux_bytes_and_legacy_images_remain_verbatim():
    old = b"actual FLX bytes are not reinterpreted by the metadata reader"
    assert unwrap_history_field_metadata(old) == (old, {}, {})
    assert unwrap_history_field_metadata(_wire(flux=old))[0] == old
    metadata = _word(0)*3
    empty = b"POPSHFM1"+_word(0)+_word(len(metadata))+metadata+_word(0)
    assert unwrap_history_field_metadata(empty) == (b"", {}, {})


@pytest.mark.parametrize("alter", [
    lambda wire: wire[:-1], lambda wire: wire+b"x",
    lambda wire: _wire(support_index=2), lambda wire: _wire(sample_kind=0),
    lambda wire: _wire(program="foreign.program"),
])
def test_malformed_metadata_is_not_treated_as_an_empty_flux(alter):
    with pytest.raises(AssertionError):
        unwrap_history_field_metadata(alter(_wire()), program="exact.program")
