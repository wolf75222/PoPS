"""Closed common envelope used by persisted Python manifests (ADC-657)."""
import pytest

from pops._manifest_protocol import (
    manifest_envelope,
    parse_manifest_envelope,
    strict_int,
    strict_json_loads,
)


def test_envelope_requires_exact_identity_and_payload_keys():
    value = manifest_envelope(kind="example", schema_version=1, payload={"name": "x"})
    assert parse_manifest_envelope(
        value, kind="example", schema_version=1, payload_keys={"name"}, where="example",
    ) == {"name": "x"}

    unknown = dict(value, extra=True)
    with pytest.raises(TypeError, match="unknown"):
        parse_manifest_envelope(
            unknown, kind="example", schema_version=1, payload_keys={"name"}, where="example",
        )
    missing = dict(value)
    missing.pop("protocol")
    with pytest.raises(TypeError, match="missing"):
        parse_manifest_envelope(
            missing, kind="example", schema_version=1, payload_keys={"name"}, where="example",
        )


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_strict_integer_never_coerces(value):
    with pytest.raises(TypeError, match="integer"):
        strict_int(value, where="version")


def test_json_refuses_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(ValueError, match="duplicate object key"):
        strict_json_loads('{"value":1,"value":2}')
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_loads('{"value":NaN}')
