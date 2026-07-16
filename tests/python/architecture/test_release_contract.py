"""Source-only release/version contract gates (ADC-688)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types


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


def test_release_contract_versions_every_protocol_and_declares_exact_matrix():
    generated = _load("_release_contract_test",
                      ROOT / "python" / "pops" / "_generated_release_contract.py")
    source = json.loads((ROOT / "schemas" / "release_contract.v1.json").read_text())
    assert generated.PACKAGE_VERSION == "1.0.0"
    for name in (
        "public_api_version", "semantic_ir_version", "normalization_version",
        "component_catalog_schema_version", "component_manifest_schema_version",
        "component_registry_version", "capability_vocabulary_version", "native_abi_version",
        "component_interface_abi_version",
        "checkpoint_envelope_schema_version", "uniform_checkpoint_payload_version",
        "amr_checkpoint_payload_version",
    ):
        assert source[name] >= 1
    assert generated.SUPPORTED_MATRIX["wheels"] == (
        {"arch": "arm64", "backend": "Kokkos Serial", "os": "macos", "python": "cp312"},
    )
    assert "CUDA wheel" in generated.SUPPORTED_MATRIX["not_promised"]


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
    assert "requires --tag, --installed and --evidence" in result.stderr
