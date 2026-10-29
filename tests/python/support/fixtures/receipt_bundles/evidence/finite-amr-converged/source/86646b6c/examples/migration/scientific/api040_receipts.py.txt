"""Lossless JSON presentation of scientific receipts and PoPS CBOR identities."""
from __future__ import annotations

import json


def _encode_wire_value(value):
    # Identity.to_data() deliberately returns a CBOR-ready 32-byte digest.
    # Preserve every byte with an explicit encoding instead of repr/default=str.
    if isinstance(value, bytes):
        return {"encoding": "hex", "bytes": value.hex()}
    raise TypeError("unsupported scientific receipt value: " + type(value).__name__)


def receipt_json(value, *, indent=2):
    """Encode finite JSON data, with a tagged representation for byte strings."""
    return json.dumps(value, indent=indent, default=_encode_wire_value, allow_nan=False)
