"""Source-only integrity checks for the executable M3 AMR/multi-layout gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/m3_amr_multilayout.toml"
RUNNER = ROOT / "scripts/run_m3_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_m3_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m3_manifest_references_only_real_mandatory_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors, "M3 gate matrix is incomplete:\n  " + "\n  ".join(errors)
    assert len(data["check"]) == 26


def test_m3_final_gate_has_no_deferred_requirement():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == []
    assert {row["issue"] for row in data["check"]} == {
        "ADC-672", "ADC-673", "ADC-674", "ADC-675", "ADC-676", "ADC-677", "ADC-678",
    }


def test_m3_gate_requires_native_positive_and_refusal_proofs_for_every_issue(tmp_path):
    data = MANIFEST.read_text(encoding="utf-8")
    data = data.replace(
        'test_regex = "^test_mpi_system_layout_transfer_np2$"',
        'test_regex = "^test_mpi_system_layout_transfer_np2_removed$"',
        1,
    )
    data = data.replace(
        'polarity = "positive"\nkind = "ctest"\ntarget = "test_mpi_system_layout_transfer"',
        'polarity = "refusal"\nkind = "ctest"\ntarget = "test_mpi_system_layout_transfer"',
        1,
    )
    manifest = tmp_path / "m3.toml"
    manifest.write_text(data, encoding="utf-8")

    _, errors = _load_runner().validate_manifest(manifest)

    assert any("ADC-673 lacks a mandatory native positive proof" in error for error in errors)
