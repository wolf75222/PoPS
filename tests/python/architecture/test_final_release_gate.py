"""Source-only contract checks for the final release gate (ADC-695)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
preflight = _load("_release_preflight_test", SCRIPTS / "release_preflight.py")


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


@pytest.mark.parametrize("module", ("pops.ir", "pops._ir"))
def test_final_release_source_contract_refuses_internal_or_transitional_imports(
    tmp_path, module
):
    _write_final_source_tree(tmp_path)
    path = tmp_path / contract.FINAL_EXAMPLES[0]
    path.write_text(
        path.read_text(encoding="utf-8") + f"\nfrom {module} import ValueExpr\n",
        encoding="utf-8",
    )

    errors = contract.source_contract_errors(tmp_path)

    assert any("transitional/internal authoring names" in error for error in errors)


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


def test_final_gate_pins_one_conda_environment_and_native_headers(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("POPS_CONDA_EXE", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    executable = tmp_path / "conda"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(gate.shutil, "which", lambda name: str(executable))
    monkeypatch.setenv("POPS_ENV_NAME", "pops-proof")
    command = gate._conda_command(["python", "-c", "import pops"])

    assert command[:6] == [
        str(executable.resolve()), "run", "--no-capture-output", "-n", "pops-proof",
        "/usr/bin/env",
    ]
    assert "POPS_INCLUDE=" + str((ROOT / "include").resolve()) in command
    assert "POPS_REQUIRE_NATIVE_TESTS=1" in command
    assert command[-3:] == ["python", "-c", "import pops"]
    assert "bash" not in command


def test_final_gate_honours_explicit_conda_executable(monkeypatch, tmp_path):
    executable = tmp_path / "conda"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("POPS_CONDA_EXE", str(executable))
    monkeypatch.delenv("CONDA_EXE", raising=False)

    command = gate._conda_command(["python", "-V"])

    assert command[0] == str(executable.resolve())
    assert command[-2:] == ["python", "-V"]


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


def test_release_evidence_authenticates_the_exact_retained_wheel(tmp_path):
    wheel = tmp_path / "wheels" / "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "pops-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: PoPS\nVersion: 0.3.0\n",
        )
    gates = {
        "official_build": {
            "evidence": {
                "wheel": {
                    "path": str(wheel.relative_to(tmp_path)),
                    "sha256": gate._sha256(wheel),
                    "size": wheel.stat().st_size,
                },
            },
        },
    }
    release = type("ReleaseContract", (), {"PACKAGE_VERSION": "0.3.0"})

    preflight._wheel_evidence(tmp_path, gates, release)
    gates["official_build"]["evidence"]["wheel"]["size"] += 1
    with pytest.raises(preflight.PreflightError, match="size drifted"):
        preflight._wheel_evidence(tmp_path, gates, release)


def _write_public_api_evidence(tmp_path: Path) -> tuple[Path, dict, object]:
    package = tmp_path / "site-packages" / "pops"
    wheel_sha256 = "a" * 64
    typed_sha256 = "b" * 64
    public_sha256 = "c" * 64
    metadata_sha256 = "d" * 64
    payload = {
        "schema_version": preflight.PUBLIC_API_EVIDENCE_SCHEMA_VERSION,
        "producer": {
            "script": "scripts/prove_public_api_parity.py",
            "sha256": hashlib.sha256(
                (SCRIPTS / "prove_public_api_parity.py").read_bytes()
            ).hexdigest(),
        },
        "wheel_path": str(tmp_path / "pops.whl"),
        "wheel_sha256": wheel_sha256,
        "distribution": {
            "name": "PoPS",
            "version": "1.0.0",
            "metadata_sha256": metadata_sha256,
        },
        "typed_payload_files": 3,
        "typed_payload_sha256": typed_sha256,
        "public_api_sha256": public_sha256,
        "public_names": ["Model", "Program", "Case"],
        "pure_authoring": True,
        "qualified_handles": True,
        "py_typed": True,
        "installed": True,
        "installed_distribution": {
            "name": "PoPS",
            "version": "1.0.0",
            "metadata_sha256": metadata_sha256,
        },
        "installed_package": str(package),
        "installed_typed_payload_sha256": typed_sha256,
        "installed_public_api_sha256": public_sha256,
    }
    path = tmp_path / "public-api-evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    release_evidence = {
        "runtime": {"pops_file": str(package / "__init__.py")},
        "gates": {
            "official_build": {
                "evidence": {"wheel": {"sha256": wheel_sha256}},
            },
        },
    }
    release = type("ReleaseContract", (), {"PACKAGE_VERSION": "1.0.0"})
    return path, release_evidence, release


def test_release_preflight_binds_installed_public_api_to_wheel_and_runtime(tmp_path):
    evidence, release_evidence, release = _write_public_api_evidence(tmp_path)

    preflight._public_api_evidence(evidence, release_evidence, release)

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["wheel_sha256"] = "e" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="another wheel"):
        preflight._public_api_evidence(evidence, release_evidence, release)


def test_release_preflight_rejects_public_api_proven_on_another_install(tmp_path):
    evidence, release_evidence, release = _write_public_api_evidence(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["installed_package"] = str(tmp_path / "other" / "pops")
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(preflight.PreflightError, match="authenticated installed runtime"):
        preflight._public_api_evidence(evidence, release_evidence, release)


def test_tag_release_cannot_race_or_bypass_supported_matrix_wheel_and_final_gate():
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    wheels = (ROOT / ".github" / "workflows" / "wheels.yml").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    build = (ROOT / "scripts" / "build_python.sh").read_text()

    assert "uses: ./.github/workflows/ci.yml" in release
    assert "force_full: true" in release
    assert "uses: ./.github/workflows/wheels.yml" in release
    assert "needs: [full-source-matrix, wheel, validate]" in release
    assert "run_final_gate.py --wheel" in release
    assert "release_preflight.py" in release
    assert 'gh release create "$GITHUB_REF_NAME" wheelhouse/*.whl' in release
    assert "workflow_call:" in ci
    assert "FORCE_FULL: ${{ inputs.force_full || false }}" in ci
    assert "workflow_call:" in wheels
    assert "gh release upload" not in wheels
    assert "--wheel-dir" in build
    assert "--force-reinstall --no-deps" in build
