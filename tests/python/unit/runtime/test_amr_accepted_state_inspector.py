"""The MPI tagging witness must follow the versioned native history-slot layout."""
import struct

import pytest

from tests.python.support.amr_accepted_state import (
    accepted_tagging_hysteresis_span,
    replace_accepted_tagging_hysteresis,
)


def _image(version):
    """Populate every pre-tagging section, especially AND7's nonempty sample identities."""
    word = lambda value: struct.pack("<Q", value)
    frame = lambda value: word(len(value)) + value
    image = bytearray(f"POPSAND{version}".encode() + word(2) + frame(b"spatial"))
    image += word(9) + word(4)  # topology, materialization
    image += word(2) + struct.pack("<qqqqd", 0, 3, 0, 1, .25) * 2
    image += word(1) + frame(b"macro") + word(3)
    image += word(1) + frame(b"dense") + word(8)
    for identity in (b"state", b"space", b"clock", b"interpolation"):
        image += frame(identity)
    image += word(2) + word(1)
    image += word(2)
    for slot in (0, 1):
        image += frame(b"dense") + struct.pack("<qqdQq", 1, slot, .125, 1, 2)
        if version == 7:
            image += struct.pack("<QQQQ", 1, 0x3FD0000000000000, 0x3FC0000000000000, 3)
    image += word(1) + frame(b"remap") + struct.pack("<qqQQQQqqqddQ", 0, 1, 8, 3, 9, 4, 3, 1, 2, .25, .125, 1)
    image += frame(b"history-flux")
    image += word(1) + frame(b"temporal-provider") + word(9) + word(6) + word(2)
    image += word(2) + struct.pack("<qQqq", 1, 42, 1, 6) * 2
    tagging = b"POPSHYS2\x00persistent-tagging\x00"
    image += frame(tagging)
    # This inspector intentionally leaves accepted face/contract bytes to the native decoder.
    suffix = b"opaque flux budget, coupling, face families, interface and provenance"
    return bytes(image) + suffix, tagging, suffix


@pytest.mark.parametrize("version", (4, 5, 6, 7))
def test_tagging_extraction_and_replacement_preserve_other_authorities(version):
    image, tagging, suffix = _image(version)
    found, offset = accepted_tagging_hysteresis_span(image, dimension=2)
    assert found == tagging
    assert image[offset:] == tagging + suffix
    for replacement in (b"", b"replacement-longer-than-original-tagging-payload"):
        changed = replace_accepted_tagging_hysteresis(image, replacement, dimension=2)
        assert changed[:offset - 8] == image[:offset - 8]
        assert changed[offset:] == replacement + suffix
        assert accepted_tagging_hysteresis_span(changed, dimension=2) == (replacement, offset)


def test_truncated_prefix_unknown_version_and_wrong_dimension_are_refused():
    image, tagging, suffix = _image(7)
    # Every boundary, including the four newly encoded slot words, must fail closed.
    for length in range(len(image) - len(suffix)):
        with pytest.raises(AssertionError):
            accepted_tagging_hysteresis_span(image[:length], dimension=2)
    with pytest.raises(AssertionError, match="dimension"):
        accepted_tagging_hysteresis_span(image, dimension=1)
    with pytest.raises(AssertionError, match="v4/v5/v6/v7"):
        accepted_tagging_hysteresis_span(b"POPSAND8" + image[8:], dimension=2)
