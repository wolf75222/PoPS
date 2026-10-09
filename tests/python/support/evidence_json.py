"""Lossless JSON projection for actual artifact metadata, never physical states."""
from collections.abc import Mapping
import json


ENCODING = "json-with-bytes-hex.v1"


def evidence_default(value):
    # Runtime bound snapshots use this exact tag for Identity digest bytes.
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported evidence metadata type: {type(value).__name__}")


def evidence_dumps(value):
    return json.dumps(value, indent=2, allow_nan=False, default=evidence_default) + "\n"
