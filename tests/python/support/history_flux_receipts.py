"""Independent readers of real native AMR history-flux receipts."""
from __future__ import annotations
from typing import Any
from fractions import Fraction
from math import gcd


def decode_exact_weight(payload: bytes) -> Fraction:
    """Independent canonical EW1 reader; no native codec or floating conversion."""
    assert len(payload) >= 20 and payload[:3] == b"EW1"
    sign = payload[3]
    assert sign in (0, 1)
    numerator_count = int.from_bytes(payload[4:12], "little")
    denominator_count = int.from_bytes(payload[12:20], "little")
    assert denominator_count > 0
    assert len(payload) == 20 + 4*(numerator_count+denominator_count)
    split = 20+4*numerator_count
    numerator = int.from_bytes(payload[20:split], "little")
    denominator = int.from_bytes(payload[split:], "little")
    assert denominator > 0 and payload[-4:] != bytes(4)
    if numerator_count:
        assert payload[split-4:split] != bytes(4)
    else:
        assert sign == 0 and denominator_count == 1 and denominator == 1
    assert gcd(numerator, denominator) == 1
    return Fraction(-numerator if sign else numerator, denominator)


def accepted_state_history_flux(payload: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Inspect the versioned POPSAND + POPSFLX4 envelope, retaining exact byte boundaries.

    This is deliberately a test-side reader: it proves the wire image contains actual retained
    expressions and also catches accidental offset drift before a checkpoint reaches native restore.
    """
    cursor = 0

    def require(count: int) -> None:
        assert 0 <= count <= len(payload) - cursor, "POPSAND payload is truncated"

    def read_u64() -> int:
        nonlocal cursor
        require(8)
        value = int.from_bytes(payload[cursor : cursor + 8], "little")
        cursor += 8
        return value

    def skip(count: int) -> None:
        nonlocal cursor
        require(count)
        cursor += count

    def read_string() -> str:
        size = read_u64()
        require(size)
        nonlocal_cursor = cursor
        skip(size)
        return payload[nonlocal_cursor : nonlocal_cursor + size].decode("utf-8")

    version = payload[:8]
    assert version in (b"POPSAND7", b"POPSAND8", b"POPSAND9", b"POPSANDA", b"POPSANDB", b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF")
    cursor = 8
    dimension = read_u64()
    assert 1 <= dimension <= 3
    skip(read_u64())  # spatial contract
    skip(16)  # topology epoch, materialization generation
    if version in (b"POPSAND8", b"POPSAND9", b"POPSANDA", b"POPSANDB", b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF"):
        read_u64()  # committed attempt, independent of face-ledger contents
    for _ in range(read_u64()):
        skip(40)
    for _ in range(read_u64()):
        skip(read_u64() + 8)
    for _ in range(read_u64()):
        read_string()
        skip(8)
        for _identity in range(4):
            read_string()
        skip(24 if version in (b"POPSANDA", b"POPSANDB", b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF") else 16)  # depth, components, then A's ghost depth
    for _ in range(read_u64()):
        read_string()
        skip(104 if version in (b"POPSANDB", b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF") else 72)  # five provenance, four identity, B digest
    for _ in range(read_u64()):
        read_string()
        skip(12 * 8)
        if version in (b"POPSAND9", b"POPSANDA", b"POPSANDB", b"POPSANDC", b"POPSANDD", b"POPSANDE", b"POPSANDF"):
            skip(6 * 8)
            skip(read_u64())
    history_flux_size = read_u64()
    history_flux_offset = cursor
    skip(history_flux_size)
    history_flux = payload[history_flux_offset:cursor]

    flux_cursor = 0

    def flux_u64() -> int:
        nonlocal flux_cursor
        assert flux_cursor + 8 <= len(history_flux), "history-flux size is truncated"
        value = int.from_bytes(history_flux[flux_cursor : flux_cursor + 8], "little")
        flux_cursor += 8
        return value

    def flux_i64() -> int:
        nonlocal flux_cursor
        assert flux_cursor + 8 <= len(history_flux), "history-flux scalar is truncated"
        value = int.from_bytes(history_flux[flux_cursor : flux_cursor + 8], "little", signed=True)
        flux_cursor += 8
        return value

    def flux_skip(count: int) -> None:
        nonlocal flux_cursor
        assert 0 <= count <= len(history_flux) - flux_cursor, "history-flux record is truncated"
        flux_cursor += count

    def flux_string() -> str:
        size = flux_u64()
        start = flux_cursor
        flux_skip(size)
        return history_flux[start : start + size].decode("utf-8")

    def snapshot_descriptor(depth: int = 0) -> dict[str, Any]:
        assert depth < 2, "history face projection exceeds this authored two-level profile"
        kind = flux_u64()
        assert kind in (1, 2), "every conservative history basis requires a complete source"
        identity = flux_string()
        source_identity = flux_string()
        level = flux_i64()
        components = flux_i64()
        assert identity and source_identity and 0 <= level < 2 and components >= 1
        flux_skip(dimension * 3 * 8)  # exact-ranked domain and spacing
        patches = flux_u64()
        assert patches > 0
        flux_skip(patches * dimension * 2 * 8)
        result = {"kind": kind, "identity": identity, "source_identity": source_identity,
                  "level": level, "components": components}
        if kind == 2:
            assert all(flux_i64() == 2 for _ in range(dimension))
            result["parent"] = snapshot_descriptor(depth + 1)
            assert result["parent"]["level"] + 1 == level
        return result

    rings: list[dict[str, Any]] = []
    if history_flux:
        magic = flux_u64()
        assert magic in (0x504F5053464C5833, 0x504F5053464C5834, 0x504F5053464C5835, 0x504F5053464C5836)
        wide = magic in (0x504F5053464C5835, 0x504F5053464C5836)
        snapshots = magic in (0x504F5053464C5834, 0x504F5053464C5836)
        for _ in range(flux_u64()):
            name = flux_string()
            slots = []
            for _ in range(flux_u64()):
                bases = []
                for _ in range(flux_u64()):
                    identity = flux_u64()
                    coefficients = flux_u64()
                    weights = []
                    for _coefficient in range(coefficients):
                        power = flux_i64()
                        if wide:
                            size = flux_u64(); start = flux_cursor
                            flux_skip(size)
                            weight = decode_exact_weight(history_flux[start:flux_cursor])
                        else:
                            numerator, denominator = flux_i64(), flux_i64()
                            assert denominator > 0 and gcd(numerator, denominator) == 1
                            weight = Fraction(numerator, denominator)
                        weights.append((power, weight))
                    runtime_block = flux_u64()
                    level = flux_i64()
                    flux_i64()  # RHS id
                    flux_skip(8)  # provider
                    assert flux_string()  # temporal family
                    # point: clock, tick, level/substep/stage, rational, dt/time, three identities
                    flux_string()
                    flux_skip(8 + 3 * 8 + 2 * 8 + 2 * 8)
                    flux_string()
                    flux_string()
                    flux_string()
                    flux_skip(80)  # two exact clock stamps
                    faces = []
                    for _ in range(flux_u64()):
                        flux_skip(8 + 8 + 2 * dimension * 8 + 8)
                        density_count = flux_u64()
                        density_offset = flux_cursor
                        flux_skip(density_count * 8)
                        faces.append((density_count, history_flux[density_offset:flux_cursor]))
                    assert flux_u64() == 0  # this fixture has no shared-interface providers
                    snapshot = snapshot_descriptor() if snapshots else None
                    if snapshot is not None:
                        assert all(count == snapshot["components"] for count, _ in faces)
                    bases.append({"identity": identity, "runtime_block": runtime_block,
                                  "level": level, "faces": faces, "snapshot": snapshot, "weights": weights})
                slots.append(bases)
            rings.append({"name": name, "slots": slots})
        assert flux_cursor == len(history_flux), "history-flux parser left trailing bytes"
    return rings, history_flux
