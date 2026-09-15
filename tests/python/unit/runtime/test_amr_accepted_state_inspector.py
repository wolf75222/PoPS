"""The MPI tagging witness must follow the versioned native history-slot layout."""
import struct

import pytest

from tests.python.support.amr_accepted_state import (
    accepted_tagging_hysteresis_span,
    replace_accepted_tagging_hysteresis,
)


def _image(version, *, committed_attempt=37):
    """Keep the original AND4-7 prefixes and add AND8's independent attempt word."""
    word = lambda value: struct.pack("<Q", value)
    frame = lambda value: word(len(value)) + value
    image = bytearray(f"POPSAND{version}".encode() + word(2) + frame(b"spatial"))
    image += word(9) + word(4)  # topology, materialization
    if version == 8:
        image += word(committed_attempt)
    image += word(2) + struct.pack("<qqqqd", 0, 3, 0, 1, .25) * 2
    image += word(1) + frame(b"macro") + word(3)
    image += word(1) + frame(b"dense") + word(8)
    for identity in (b"state", b"space", b"clock", b"interpolation"):
        image += frame(identity)
    image += word(2) + word(1)
    image += word(2)
    for slot in (0, 1):
        image += frame(b"dense") + struct.pack("<qqdQq", 1, slot, .125, 1, 2)
        if version in (7, 8):
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


@pytest.mark.parametrize("version", (4, 5, 6, 7, 8))
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


@pytest.mark.parametrize("version", (7, 8))
def test_truncated_prefix_unknown_version_and_wrong_dimension_are_refused(version):
    image, tagging, suffix = _image(version)
    # Every boundary, including the attempt and four sample words, must fail closed.
    for length in range(len(image) - len(suffix)):
        with pytest.raises(AssertionError):
            accepted_tagging_hysteresis_span(image[:length], dimension=2)
    with pytest.raises(AssertionError, match="dimension"):
        accepted_tagging_hysteresis_span(image, dimension=1)
    with pytest.raises(AssertionError, match="v4/v5/v6/v7/v8"):
        accepted_tagging_hysteresis_span(b"POPSAND9" + image[8:], dimension=2)


@pytest.mark.parametrize("committed_attempt", (0, 42, 2**64 - 1))
def test_wire8_adds_exactly_one_attempt_word_and_preserves_it_on_tagging_replacement(committed_attempt):
    old, tagging, suffix = _image(7)
    current, _, _ = _image(8, committed_attempt=committed_attempt)
    attempt_offset = 5 * 8 + len(b"spatial")
    encoded_attempt = struct.pack("<Q", committed_attempt)
    assert current == (b"POPSAND8" + old[8:attempt_offset] + encoded_attempt + old[attempt_offset:])
    old_tagging, old_offset = accepted_tagging_hysteresis_span(old, dimension=2)
    actual_tagging, actual_offset = accepted_tagging_hysteresis_span(current, dimension=2)
    assert actual_tagging == old_tagging == tagging
    assert actual_offset == old_offset + 8
    replaced = replace_accepted_tagging_hysteresis(current, b"", dimension=2)
    assert replaced[attempt_offset:attempt_offset + 8] == encoded_attempt
    assert replaced[actual_offset:] == suffix


@pytest.mark.parametrize("committed_attempt", (0, 42, 2**64 - 1))
def test_wire8_empty_face_ledger_retains_its_independent_attempt_word(committed_attempt):
    # An independently packed complete envelope, with no histories, face entries or events.
    # This is a parser fixture, not evidence that native checkpoint admission was executed.
    word = lambda value: struct.pack("<Q", value)
    frame = lambda value: word(len(value)) + value
    spatial = b"empty-ledger"
    image = bytearray(b"POPSAND8" + word(2) + frame(spatial))
    image += word(0) + word(0) + word(committed_attempt)
    image += word(1) + struct.pack("<qqqqd", 0, 3, 0, 1, .25)
    image += word(0) * 5  # logical clocks, histories, slots, pending remaps, history flux
    image += word(0) + frame(b"pops.temporal-partition.global@1")
    image += word(0) + word(0) + word(1) + word(0)  # topology, sync tick, denominator, cells
    tagging = b"persistent-tagging"
    image += frame(tagging)
    suffix = frame(b"empty-flux-budget") + frame(b"no-coupling")
    suffix += word(0) * 5  # two face axes, interfaces, events, absent face provenance
    image += suffix
    image = bytes(image)
    found, offset = accepted_tagging_hysteresis_span(image, dimension=2)
    assert found == tagging
    assert image[offset:] == tagging + suffix
    changed = replace_accepted_tagging_hysteresis(image, b"replacement", dimension=2)
    attempt_offset = 5 * 8 + len(spatial)
    assert changed[attempt_offset:attempt_offset + 8] == word(committed_attempt)
    assert changed[offset:] == b"replacement" + suffix


def test_wire8_missing_misplaced_and_truncated_attempt_prefixes_refuse():
    old, _, _ = _image(7)
    current, _, _ = _image(8)
    attempt_offset = 5 * 8 + len(b"spatial")
    level_count_offset = attempt_offset + 8
    invalid = (
        b"POPSAND8" + old[8:],  # changed magic alone does not create committed authority
        (current[:attempt_offset] + current[level_count_offset:level_count_offset + 8]
         + current[attempt_offset:level_count_offset] + current[level_count_offset + 8:]),
        current[:level_count_offset] + struct.pack("<Q", 2**64 - 1) + current[level_count_offset + 8:],
        current[:attempt_offset + 7],
    )
    for image in invalid:
        with pytest.raises(AssertionError):
            accepted_tagging_hysteresis_span(image, dimension=2)
