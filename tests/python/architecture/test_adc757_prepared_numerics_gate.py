"""Source-only integrity checks for the bounded ADC-757 numerical gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/adc757_prepared_numerics.toml"
RUNNER = ROOT / "scripts/run_adc757_prepared_numerics_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_adc757_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adc757_slice_references_exact_real_mandatory_native_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors, "ADC-757 slice matrix is invalid:\n  " + "\n  ".join(errors)
    assert len(data["check"]) == 14
    assert {row["requirement"] for row in data["check"]} == runner.EXPECTED_REQUIREMENTS
    assert runner.main(["--check-only"]) == 0


def test_adc757_slice_does_not_claim_full_mpi_gpu_or_runtime_closure():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == list(runner.EXPECTED_DEFERRED)
    assert "mpi_collective_execution" in data["deferred"]
    assert "gpu_backend_execution" in data["deferred"]
    assert "remaining_legacy_recovery_boundary_and_riemann_authority_deletion" in data["deferred"]
    assert "runtime_consumer_cutover_and_legacy_deletion" not in data["deferred"]
    assert "boundary_geometry_riemann_and_spatial_provider_families" not in data["deferred"]
    assert all(
        "mpi" not in row["target"].lower() and "gpu" not in row["target"].lower()
        for row in data["check"]
    )
    assert runner.main(["--check-only", "--closure"]) == 3


def test_adc757_manifest_refuses_missing_polarity_and_unknown_target(tmp_path):
    runner = _load_runner()
    source = MANIFEST.read_text(encoding="utf-8")

    missing_refusal = tmp_path / "missing_refusal.toml"
    missing_refusal.write_text(
        source.replace('polarity = "refusal"', 'polarity = "positive"', 1),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(missing_refusal)
    assert any("lacks refusal coverage" in error for error in errors)

    unknown_target = tmp_path / "unknown_target.toml"
    unknown_target.write_text(
        source.replace(
            'target = "test_newton_robustness"',
            'target = "test_missing_provider_proof"',
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(unknown_target)
    assert any("unknown CTest target" in error for error in errors)


def test_adc757_runner_refuses_a_declared_but_unbuilt_proof(monkeypatch, tmp_path):
    runner = _load_runner()

    def empty_ctest_listing(command, **kwargs):
        assert command[:2] == ["ctest", "--test-dir"]
        return SimpleNamespace(returncode=0, stdout="Total Tests: 0\n")

    monkeypatch.setattr(runner.subprocess, "run", empty_ctest_listing)
    with pytest.raises(RuntimeError, match="is not built"):
        runner._run_ctest(
            tmp_path,
            "test_prepared_numerics_gate",
            r"^PreparedNumericsGate\.ConvergedPreparedPathAllocatesNothingAndRollsBack$",
        )
