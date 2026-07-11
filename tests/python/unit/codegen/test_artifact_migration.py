"""ADC-657: offline-only, deterministic external manifest migration."""
from __future__ import annotations

import json

import pytest

from pops.codegen.artifact_migration import migrate_external_brick_manifest


def _v2(**overrides):
    row = {
        "id": "third_party.hll", "category": "riemann",
        "requirements": "physical_flux,wave_speeds", "capabilities": "",
        "native_id": "third_party_hll", "supported_layouts": "uniform,amr",
        "supported_platforms": "cpu", "params": "", "options": "",
        "exported_symbols": "pops_brick_residual",
    }
    row.update(overrides)
    return {"schema_version": 2, "abi_key": "headers=abc|clang|20", "bricks": [row]}


def test_offline_migration_is_deterministic_idempotent_and_changes_identity(tmp_path):
    source = tmp_path / "legacy.json"
    first = tmp_path / "current-a.json"
    second = tmp_path / "current-b.json"
    source.write_text(json.dumps(_v2(), indent=2), encoding="utf-8")
    original = source.read_bytes()

    report = migrate_external_brick_manifest(source, first)
    repeat = migrate_external_brick_manifest(source, second)

    assert source.read_bytes() == original
    assert first.read_bytes() == second.read_bytes()
    assert report.artifact_identity == repeat.artifact_identity
    assert report.changed is True
    current_copy = tmp_path / "current-copy.json"
    current = migrate_external_brick_manifest(first, current_copy)
    assert current_copy.read_bytes() == first.read_bytes()
    assert current.changed is False


def test_offline_migration_requires_non_derivable_metadata(tmp_path):
    source = tmp_path / "legacy.json"
    destination = tmp_path / "current.json"
    doc = _v2()
    del doc["bricks"][0]["native_id"]
    source.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(ValueError, match="not derivable"):
        migrate_external_brick_manifest(source, destination)
    assert not destination.exists()

    migrate_external_brick_manifest(
        source,
        destination,
        metadata={"bricks": {"third_party.hll": {"native_id": "third_party_hll"}}},
    )
    assert json.loads(destination.read_text())["bricks"][0]["native_id"] == "third_party_hll"


def test_offline_migration_never_overwrites_legacy_source(tmp_path):
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_v2()), encoding="utf-8")
    with pytest.raises(ValueError, match="refuses to overwrite"):
        migrate_external_brick_manifest(source, source)

