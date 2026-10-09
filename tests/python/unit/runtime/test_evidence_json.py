"""Artifact metadata must retain exact binary identity bytes and reject coercion."""
import json
from types import MappingProxyType

import pytest

from tests.python.support.evidence_json import ENCODING, evidence_dumps


def test_binary_digest_projection_matches_bound_snapshot_encoding():
    digest = bytes(range(32))
    payload = MappingProxyType({"metadata_encoding": ENCODING,
                               "identity": MappingProxyType({"hexdigest": digest}),
                               "ordinary": "bytes_hex"})
    encoded = json.loads(evidence_dumps(payload))
    assert encoded["metadata_encoding"] == "json-with-bytes-hex.v1"
    assert set(encoded["identity"]["hexdigest"]) == {"bytes_hex"}
    assert bytes.fromhex(encoded["identity"]["hexdigest"]["bytes_hex"]) == digest
    assert encoded["ordinary"] == "bytes_hex"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_metadata_refuses(value):
    with pytest.raises(ValueError, match="Out of range float values"):
        evidence_dumps({"value": value})


@pytest.mark.parametrize("value", (object(), bytearray(b"abc"), memoryview(b"abc")))
def test_foreign_metadata_refuses_without_string_coercion(value):
    with pytest.raises(TypeError, match="unsupported evidence metadata type"):
        evidence_dumps({"value": value})
