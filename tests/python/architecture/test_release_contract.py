"""Source-only release/version contract gates (ADC-688)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _release_module():
    previous = {
        name: module for name, module in sys.modules.items()
        if name == "pops" or name.startswith("pops.")
    }
    package = types.ModuleType("pops")
    package.__path__ = [str(ROOT / "python" / "pops")]
    try:
        sys.modules["pops"] = package
        _load("pops._generated_release_contract",
              ROOT / "python" / "pops" / "_generated_release_contract.py")
        return _load("pops.release", ROOT / "python" / "pops" / "release.py")
    finally:
        for name in tuple(sys.modules):
            if name == "pops" or name.startswith("pops."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_generated_release_contract_is_current_and_preflight_passes_static_checks():
    commands = [[sys.executable, "scripts/generate_release_contract.py", "--check"]]
    final_contract = _load("_final_release_source_contract",
                           ROOT / "scripts" / "final_release_contract.py")
    # Adjacent worktrees intentionally land the canonical specification and the
    # final examples independently.  The release preflight itself is strict;
    # once that exact source set lands it is always part of this architecture
    # assertion.  The synthetic ADC-695 tests cover the source checker before
    # that integration point.
    if not final_contract.source_contract_errors(ROOT):
        commands.append([sys.executable, "scripts/release_preflight.py"])
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_final_and_release_preflights_verify_cpp_duration_catalogs_before_build():
    selector = (ROOT / "scripts" / "ci_select_tests.py").read_text(encoding="utf-8")
    final_gate = (ROOT / "scripts" / "run_final_gate.py").read_text(encoding="utf-8")
    release_preflight = (ROOT / "scripts" / "release_preflight.py").read_text(
        encoding="utf-8"
    )

    command = "verify-cpp-duration-catalogs"
    final_main = final_gate[final_gate.index("def main("):]
    assert command in selector
    assert final_main.index("_require_cpp_duration_catalogs()") < final_main.index(
        'recorder.run("official_build"'
    )
    assert command in release_preflight


def test_release_contract_versions_every_protocol_and_declares_exact_matrix():
    generated = _load("_release_contract_test",
                      ROOT / "python" / "pops" / "_generated_release_contract.py")
    source = json.loads((ROOT / "schemas" / "release_contract.v2.json").read_text())
    assert source["release_contract_schema_version"] == 2
    assert generated.PACKAGE_VERSION == "1.0.0"
    for name in (
        "public_api_version", "semantic_ir_version", "normalization_version",
        "component_catalog_schema_version", "component_manifest_schema_version",
        "component_registry_version", "capability_vocabulary_version", "native_abi_version",
        "component_interface_abi_version",
        "checkpoint_envelope_schema_version", "checkpoint_spatial_schema_version",
        "uniform_checkpoint_payload_version",
        "amr_checkpoint_payload_version",
    ):
        assert source[name] >= 1
    assert source["uniform_checkpoint_payload_version"] == 8
    assert source["amr_checkpoint_payload_version"] == 10
    assert source["checkpoint_spatial_schema_version"] == 1
    assert source["capability_vocabulary_version"] == 4
    assert generated.SUPPORTED_MATRIX["wheels"] == (
        {"arch": "arm64", "backend": "Kokkos Serial", "os": "macos", "python": "cp312"},
    )
    assert "CUDA wheel" in generated.SUPPORTED_MATRIX["not_promised"]


def test_release_contract_authenticates_component_catalog_digests():
    import copy
    import hashlib

    generated = _load(
        "_release_contract_component_digest_test",
        ROOT / "python" / "pops" / "_generated_release_contract.py",
    )
    catalog = json.loads((ROOT / "schemas" / "component_catalog.v2.json").read_text())
    canonical = json.dumps(
        catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    semantic = copy.deepcopy(catalog)
    for family in semantic["route_families"]:
        for route in family["routes"]:
            route.pop("limitations", None)
            route["metadata"].pop("summary", None)
    semantic_canonical = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert generated.COMPONENT_CATALOG_SHA256 == hashlib.sha256(canonical).hexdigest()
    assert generated.COMPONENT_CATALOG_SEMANTIC_SHA256 == hashlib.sha256(
        semantic_canonical
    ).hexdigest()
    release = _release_module().contract()
    assert release["component_catalog_sha256"] == generated.COMPONENT_CATALOG_SHA256
    assert (
        release["component_catalog_semantic_sha256"]
        == generated.COMPONENT_CATALOG_SEMANTIC_SHA256
    )


def test_release_generator_rejects_stale_component_catalog_digest(tmp_path):
    generator = _load(
        "_release_contract_stale_component_digest_test",
        ROOT / "scripts" / "generate_release_contract.py",
    )
    payload = json.loads((ROOT / "schemas" / "release_contract.v2.json").read_text())
    payload["component_catalog_sha256"] = "0" * 64
    source = tmp_path / "release_contract.v2.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    generator.SOURCE = source

    with pytest.raises(
        generator.ContractError,
        match="component_catalog_sha256 drifted from component_catalog.v2.json",
    ):
        generator._load()


def test_pre_one_compatibility_uses_minor_boundary_and_post_one_uses_major_boundary():
    release = _release_module()
    assert release.package_compatible(requested="0.3.0", available="0.3.9")
    assert not release.package_compatible(requested="0.3.0", available="0.4.0")
    assert not release.package_compatible(requested="0.3.2", available="0.3.1")
    assert release.package_compatible(requested="1.2.0", available="1.9.0")
    assert not release.package_compatible(requested="1.2.0", available="2.0.0")


def test_release_mode_cannot_run_without_tag_install_and_authenticated_evidence():
    result = subprocess.run(
        [sys.executable, "scripts/release_preflight.py", "--release"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert (
        "requires --tag, --installed, --dim, --evidence and --public-api-evidence"
        in result.stderr
    )
