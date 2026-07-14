"""Source-only contract checks for the final release gate (ADC-695)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("final_release_contract", SCRIPTS / "final_release_contract.py")
gate = _load("_final_release_gate_test", SCRIPTS / "run_final_gate.py")


def _write_final_source_tree(root: Path) -> None:
    specification = root / contract.FINAL_SPECIFICATION
    specification.parent.mkdir(parents=True)
    specification.write_text("# Specification Technique Finale\n", encoding="utf-8")
    for example in contract.FINAL_EXAMPLES:
        path = root / example
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "--output-dir\n"
            + "\n".join(contract.REQUIRED_PROOF_MARKERS)
            + "\nif __name__ == \"__main__\":\n    pass\n",
            encoding="utf-8",
        )


def test_final_release_source_contract_accepts_exact_canonical_set(tmp_path):
    _write_final_source_tree(tmp_path)

    assert contract.source_contract_errors(tmp_path) == []


def test_final_release_source_contract_refuses_missing_and_extra_examples(tmp_path):
    _write_final_source_tree(tmp_path)
    (tmp_path / contract.FINAL_EXAMPLES[-1]).unlink()
    extra = tmp_path / "examples/final/temporary.py"
    extra.write_text("pass\n", encoding="utf-8")

    errors = contract.source_contract_errors(tmp_path)

    assert any("final examples must be exactly" in error for error in errors)


def test_final_release_source_contract_requires_executable_restart_output_proof(tmp_path):
    _write_final_source_tree(tmp_path)
    path = tmp_path / contract.FINAL_EXAMPLES[0]
    path.write_text("if __name__ == \"__main__\":\n    pass\n", encoding="utf-8")

    errors = contract.source_contract_errors(tmp_path)

    assert any("--output-dir" in error for error in errors)
    assert any("lacks final proof markers" in error for error in errors)


def test_required_junit_lane_rejects_skips_xfails_failures_and_empty_reports(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite tests="1"><testcase name="ok"/></testsuite>', encoding="utf-8")
    assert gate._junit_summary(report)["tests"] == 1

    for child in ('<skipped type="pytest.xfail"/>', '<failure/>', '<error/>'):
        report.write_text(
            '<testsuite tests="1"><testcase name="bad">%s</testcase></testsuite>' % child,
            encoding="utf-8",
        )
        with pytest.raises(gate.FinalGateError):
            gate._junit_summary(report)

    report.write_text('<testsuite tests="0"/>', encoding="utf-8")
    with pytest.raises(gate.FinalGateError):
        gate._junit_summary(report)


def test_required_python_lane_rejects_script_style_hidden_skips():
    gate._require_no_hidden_skip("42 tests passed")
    with pytest.raises(gate.FinalGateError, match="hidden skip"):
        gate._require_no_hidden_skip("skip (native engine unavailable)\n1 passed")


def test_final_gate_pins_native_headers_to_the_validated_checkout():
    command = gate._conda_command(["python", "-c", "import pops"])

    assert command[:2] == ["bash", "-lc"]
    assert "POPS_INCLUDE=" + str((ROOT / "include").resolve()) in command[2]


def test_artifact_reopen_requires_and_records_npz(tmp_path):
    (tmp_path / "state.h5").write_bytes(b"\x89HDF\r\n\x1a\ncontent")
    (tmp_path / "state.vtu").write_text("<VTKFile/>", encoding="utf-8")
    npz = tmp_path / "state.npz"
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("state.npy", b"payload")

    evidence, hdf5_paths, npz_paths = gate._reopen_outputs(
        tmp_path, example=Path("final.py"))

    assert set(evidence) == {"hdf5", "npz", "paraview"}
    assert hdf5_paths == (tmp_path / "state.h5",)
    assert npz_paths == (npz,)
    npz.unlink()
    with pytest.raises(gate.FinalGateError, match="HDF5, NPZ and ParaView"):
        gate._reopen_outputs(tmp_path, example=Path("final.py"))
