"""Strict, lossless scientific JSON and saved-state receipt helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _wire(value):
    if isinstance(value, bytes):
        return {"encoding": "hex", "bytes": value.hex()}
    raise TypeError("unsupported receipt value: " + type(value).__name__)


def receipt_json(value):
    return json.dumps(value, indent=2, allow_nan=False, default=_wire)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
