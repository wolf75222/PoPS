"""CP-04 2-d oblique electrostatic eigenmode (in-memory oracle; no solver)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "oblique_wave"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
ADVANCE_TIME = 0.37


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_utility_oracle_oblique_wavevector():
    exact = _load_case_module("exact")
    kx, ky = exact.K_INTEGER
    assert kx != 0
    assert ky != 0
    assert (kx, ky) == (1, 2)
    wave = exact.physical_wavevector()
    np.testing.assert_allclose(wave, 2.0 * np.pi * np.array([kx, ky]))
    assert abs(wave[0]) > 0.0 and abs(wave[1]) > 0.0
    assert abs(wave[0]) != abs(wave[1])


def test_utility_oracle_oblique_poisson():
    exact = _load_case_module("exact")
    amplitudes = exact.complex_mode_amplitudes()
    wave = exact.physical_wavevector()
    ik_dot_e = 1.0j * np.dot(wave, amplitudes["E"])
    np.testing.assert_allclose(ik_dot_e, amplitudes["source"], rtol=0.0, atol=1.0e-14)
    x, y, _ = exact.uniform_cell_mesh(N_CELLS)
    for time in (0.0, ADVANCE_TIME, 1.25):
        fields = exact.exact_fields(x, y, time)
        residual = exact.poisson_residual(x, y, time)
        np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            fields["div_E"],
            fields["gauss_rhs"],
            rtol=0.0,
            atol=1.0e-12,
        )


def test_swapping_kx_ky_is_coordinate_permutation():
    exact = _load_case_module("exact")
    x, y, _ = exact.uniform_cell_mesh(N_CELLS)
    kx, ky = exact.K_INTEGER
    swapped = exact.exact_fields(x, y, ADVANCE_TIME, kx=ky, ky=kx)
    permuted = exact.exact_fields(y, x, ADVANCE_TIME, kx=kx, ky=ky)
    np.testing.assert_allclose(swapped["phi"], permuted["phi"], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(swapped["n_e"], permuted["n_e"], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(swapped["E_x"], permuted["E_y"], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(swapped["E_y"], permuted["E_x"], rtol=0.0, atol=1.0e-14)


def test_write_cp04_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_case_modules_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    tree = ast.parse(exact_text)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".", 1)[0])
    assert "pops" not in imported


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="CP-04",
        pops_native_dim=2,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=8,
        resources=CampaignResources(resolutions=(8,)),
        evidence_status="required",
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    assert "resolution" in result
    assert "result" in result


def test_report_fails_closed_without_native_output(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp04_report(tmp_path)
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert (
        loaded["coverage"]["cases_failed"] + loaded["coverage"]["cases_not_supported"]
        >= 1
    )
    reasons = " ".join(item["reason"] for item in loaded["failures"])
    notes = " ".join(loaded["coverage"].get("not_tested") or [])
    blob = (reasons + " " + notes).lower()
    assert (
        "native" in blob
        or "kokkos" in blob
        or "supported" in blob
        or "required" in blob
        or "not " in blob
        or "no " in blob
    )


def test_analyze_native_requires_native_field():
    analyze = _load_case_module("analyze")
    try:
        analyze.analyze_native({})
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError("analyze_native must refuse an empty mapping")


def test_analyze_native_computes_field_errors():
    analyze = _load_case_module("analyze")
    result = analyze.analyze_native(
        {
            "field": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "oracle": np.array([1.0, 2.0, 2.5], dtype=np.float64),
            "volumes": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        }
    )
    assert result["linf"] == 0.5
    assert result["l1"] > 0.0
    assert result["l2"] > 0.0


def test_write_report_stays_fail_closed_with_native_mapping(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp04_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
