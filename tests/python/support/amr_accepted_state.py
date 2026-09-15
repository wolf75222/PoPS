"""Inspect the tagging field in the native AMR accepted-state wire prefix.

This is a test inspector, not the authoritative checkpoint decoder. Native restore
still authenticates the complete image, including the untouched face-flux suffix.
The field order follows checkpoint_detail::write_state in amr_program_checkpoint.hpp.
"""
from __future__ import annotations


def accepted_tagging_hysteresis_span(encoded: bytes, *, dimension: int) -> tuple[bytes, int]:
    version = encoded[:8]
    if version not in (b"POPSAND4", b"POPSAND5", b"POPSAND6", b"POPSAND7", b"POPSAND8"):
        raise AssertionError("checkpoint does not contain exact-ranked accepted-state v4/v5/v6/v7/v8")
    cursor = 8

    def take(size: int, field: str) -> bytes:
        nonlocal cursor
        if size > len(encoded) - cursor:
            raise AssertionError(f"accepted-state {field} is truncated")
        value = encoded[cursor:cursor + size]
        cursor += size
        return value

    def word() -> int:
        return int.from_bytes(take(8, "word"), "little")

    def framed(field: str) -> bytes:
        return take(word(), field)

    if word() != dimension:
        raise AssertionError("accepted-state native dimension differs from the probe")
    framed("spatial contract")
    take(2 * 8, "topology epoch and materialization generation")
    if version == b"POPSAND8":
        # AND8 carries committed authority even when the accepted face ledger is empty.
        # Legacy inspection must not invent this field or upgrade a continuation image.
        take(8, "committed attempt")
    take(word() * 40, "level clocks")
    for _ in range(word()):
        framed("logical clock identity")
        take(8, "logical clock tick")
    for _ in range(word()):
        framed("history name")
        take(8, "history Program owner")
        for _identity in range(4):
            framed("history identity")
        take(2 * 8, "history depth and component count")
    for _ in range(word()):
        framed("history slot name")
        take(5 * 8, "history slot provenance")
        if version in (b"POPSAND7", b"POPSAND8"):
            # AND7/8 carry kind, start_bits, interval_bits and ordinal. AND4/5/6
            # have no sample identity; these are not interchangeable layouts.
            take(4 * 8, "history sample identity")
    for _ in range(word()):
        framed("pending history remap key")
        take(12 * 8, "pending history remap")
    framed("history flux payload")
    take(8, "cell temporal partition kind")
    framed("cell temporal provider identity")
    take(3 * 8, "cell temporal topology and synchronization")
    take(word() * 32, "cell temporal entries")
    size = word()
    offset = cursor
    return take(size, "persistent tagging payload"), offset


def replace_accepted_tagging_hysteresis(encoded: bytes, replacement: bytes, *, dimension: int) -> bytes:
    tagging, offset = accepted_tagging_hysteresis_span(encoded, dimension=dimension)
    return (encoded[:offset - 8] + len(replacement).to_bytes(8, "little") + replacement
            + encoded[offset + len(tagging):])
