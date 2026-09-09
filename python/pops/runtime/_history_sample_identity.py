"""Canonical typed history sample ledger shared by Uniform and AMR checkpoints.

The envelope binds the logical ring name and level. Installed artifact/descriptor authority remains
with the enclosing checkpoint, and the native Program importer compares the complete slot ledger.
Absent legacy metadata is explicit unknown provenance, never a reconstructed publication.
"""

import math
import struct

PREFIX = "history_sample_identity_"
MAGIC = b"POPSHID1"


def identity_key(name, level):
    return PREFIX + name if level is None else "%s%s_level_%d" % (PREFIX, name, level)


def native_real_bytes():
    """Read actual C++ Real width; the report's precision label and static fallback are insufficient."""
    from pops import _pops

    provider = getattr(_pops, "runtime_environment_report", None)
    if not callable(provider):
        raise RuntimeError("history sample validation requires native Real width")
    report = provider()
    width = report.get("real_bytes") if isinstance(report, dict) else None
    if type(width) is not int or width not in (4, 8):
        raise RuntimeError("history sample validation has invalid native Real width")
    return width


def validate_identity_bytes(raw, name, level, depth, *, initialized=None, slot_dt=None):
    """Validate exact canonical bytes without native mutation or float conversion of identity bits."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError("history sample identity must be bytes")
    raw = bytes(raw)
    encoded_name = name.encode("utf-8")
    header = MAGIC + struct.pack("<Q", len(encoded_name)) + encoded_name
    header += struct.pack("<qQ", -1 if level is None else level, depth)
    if depth < 1 or len(raw) != len(header) + 32 * depth or raw[:len(header)] != header:
        raise ValueError("history sample identity has an invalid version, ring, level or depth")
    if slot_dt is not None and len(slot_dt) != depth:
        raise ValueError("history sample identity dt ledger differs from depth")
    real_bytes = None
    rows = struct.iter_unpack("<QQQQ", raw[len(header):])
    for slot, (kind, start_bits, interval_bits, ordinal) in enumerate(rows):
        if kind not in (0, 1, 2):
            raise ValueError("history sample identity has an invalid kind")
        if initialized is not None and ((kind == 1 and initialized) or (kind == 2 and not initialized)):
            raise ValueError("history sample identity differs from its initialized metadata")
        if kind != 2:
            if start_bits or interval_bits or ordinal:
                raise ValueError("history sample identity has noncanonical unknown/zero-start fields")
        else:
            start = struct.unpack("<d", struct.pack("<Q", start_bits))[0]
            interval = struct.unpack("<d", struct.pack("<Q", interval_bits))[0]
            if ordinal == 0 or not math.isfinite(start) or not math.isfinite(interval) or interval <= 0:
                raise ValueError("history sample identity has an invalid publication window")
            if slot_dt is not None:
                if real_bytes is None:
                    real_bytes = native_real_bytes()
                try:
                    native_interval = (struct.unpack("<f", struct.pack("<f", interval))[0]
                                       if real_bytes == 4 else interval)
                except OverflowError as error:
                    raise ValueError("history sample interval is outside native Real") from error
                if native_interval != slot_dt[slot]:
                    raise ValueError("history sample interval differs from its outgoing dt")
    return raw


def prepare_identity_payload(payload, name, level, depth, *, initialized=None, slot_dt=None):
    """Return exact bytes, or an explicit legacy-empty reset when the additive member is absent."""
    import numpy as np

    key = identity_key(name, level)
    if key not in payload:
        return b""
    values = np.asarray(payload[key])
    if values.dtype != np.dtype(np.uint8) or values.ndim != 1:
        raise TypeError("history sample identity must be a one-dimensional uint8 vector")
    return validate_identity_bytes(values.tobytes(), name, level, depth, initialized=initialized,
                                   slot_dt=slot_dt)
