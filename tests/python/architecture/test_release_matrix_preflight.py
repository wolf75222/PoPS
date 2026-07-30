"""Source-only parity gate for the declared ADC-688 release matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "scripts" / "final_release_contract.py"
PROOF_SOURCES = (
    Path("CMakeLists.txt"),
    Path("schemas/release_contract.v1.json"),
    Path(".github/actions/setup-kokkos/action.yml"),
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/wheels.yml"),
    Path(".github/workflows/release.yml"),
)


def _load_contract():
    spec = importlib.util.spec_from_file_location("_release_matrix_contract_test", CONTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _copy_proof_sources(destination: Path) -> None:
    for relative in PROOF_SOURCES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())


def test_declared_release_matrix_has_exact_executable_workflow_proof():
    assert contract.release_matrix_source_errors(ROOT) == []


@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    (
        (
            Path(".github/workflows/wheels.yml"),
            "CIBW_BUILD: cp312-macosx_arm64",
            "CIBW_BUILD: cp313-macosx_arm64",
            "wheel lane",
        ),
        (
            Path(".github/workflows/ci.yml"),
            "-DPOPS_USE_MPI=ON",
            "-DPOPS_USE_MPI=OFF",
            "OpenMPI source lane",
        ),
        (
            Path(".github/workflows/release.yml"),
            "needs: [full-source-matrix, wheel, validate]",
            "needs: [wheel, validate]",
            "release publication dependency",
        ),
    ),
)
def test_release_matrix_preflight_refuses_workflow_drift(tmp_path, relative, old, new, message):
    _copy_proof_sources(tmp_path)
    path = tmp_path / relative
    source = path.read_text(encoding="utf-8")
    assert old in source
    path.write_text(source.replace(old, new), encoding="utf-8")

    errors = contract.release_matrix_source_errors(tmp_path)

    assert any(message in error for error in errors), errors
    with pytest.raises(ValueError, match=message):
        contract.require_release_matrix_source_contract(tmp_path)


def test_release_matrix_preflight_refuses_an_unimplemented_declared_lane(tmp_path):
    _copy_proof_sources(tmp_path)
    path = tmp_path / "schemas" / "release_contract.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["supported_matrix"]["wheels"].append(
        {
            "os": "linux",
            "arch": "x86_64",
            "python": "cp312",
            "backend": "Kokkos Serial",
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    errors = contract.release_matrix_source_errors(tmp_path)

    assert any("has no executable release proof" in error for error in errors)


def test_both_release_entrypoints_run_matrix_preflight_before_the_build():
    preflight = (ROOT / "scripts" / "release_preflight.py").read_text(encoding="utf-8")
    final_gate = (ROOT / "scripts" / "run_final_gate.py").read_text(encoding="utf-8")
    static_contract = preflight[
        preflight.index("def _static_contract(") : preflight.index("def _tag_contract(")
    ]
    final_main = final_gate[final_gate.index("def main(") :]

    assert "require_release_matrix_source_contract(ROOT)" in static_contract
    assert final_main.index("require_release_matrix_source_contract(ROOT)") < final_main.index(
        'recorder.run("official_build"'
    )
