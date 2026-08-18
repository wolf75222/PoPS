"""CP-06 ion-acoustic eigenmode (Boltzmann electrons, cold ions; no solver)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "ion_acoustic"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
CANONICAL_K = 2.0 * np.pi
WAVE_NUMBERS_OVER_2PI = (1, 2, 4, 8)
ADVANCE_TIME = 0.125


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


def test_dispersion_formula_at_canonical_wavenumbers():
    exact = _load_case_module("exact")
    c_s2 = float(exact.T_E) / float(exact.M_I)
    lambda_d2 = (
        float(exact.EPS0) * float(exact.T_E) / (float(exact.N0) * float(exact.E_CHARGE) ** 2)
    )
    np.testing.assert_allclose(exact.sound_speed_squared(), c_s2, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(exact.debye_length_squared(), lambda_d2, rtol=0.0, atol=1.0e-14)
    for cycles in WAVE_NUMBERS_OVER_2PI:
        wavenumber = 2.0 * np.pi * float(cycles)
        omega = exact.angular_frequency(wavenumber)
        expected = (wavenumber**2 * c_s2) / (1.0 + wavenumber**2 * lambda_d2)
        np.testing.assert_allclose(omega**2, expected, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            exact.dispersion_residual(wavenumber),
            0.0,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_eigenvector_identity():
    exact = _load_case_module("exact")
    assert exact.MODES == ("plus", "minus")
    for cycles in WAVE_NUMBERS_OVER_2PI:
        wavenumber = 2.0 * np.pi * float(cycles)
        matrix = exact.system_matrix(wavenumber)
        assert matrix.shape == (2, 2)
        omega = exact.angular_frequency(wavenumber)
        plus = exact.right_eigenvector("plus", wavenumber)
        np.testing.assert_allclose(
            matrix @ plus,
            -1.0j * omega * plus,
            rtol=0.0,
            atol=1.0e-14,
        )
        for mode in exact.MODES:
            eigenvalue = exact.eigenvalue(mode, wavenumber)
            vector = exact.right_eigenvector(mode, wavenumber)
            np.testing.assert_allclose(
                matrix @ vector,
                eigenvalue * vector,
                rtol=0.0,
                atol=1.0e-14,
            )


def test_time_advance_matches_closed_form():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(N_CELLS)
    background = np.asarray(exact.BACKGROUND, dtype=np.float64)
    amplitude = float(exact.EPS)
    omega = exact.angular_frequency(CANONICAL_K)
    for mode in exact.MODES:
        eigenvalue = exact.eigenvalue(mode, CANONICAL_K)
        vector = exact.right_eigenvector(mode, CANONICAL_K)
        evolved = exact.exact_state(
            centers,
            ADVANCE_TIME,
            mode=mode,
            k=CANONICAL_K,
            eps=amplitude,
        )
        phase = np.exp(1.0j * CANONICAL_K * centers + eigenvalue * ADVANCE_TIME)
        expected = background[:, None] + amplitude * np.real(vector[:, None] * phase[None, :])
        np.testing.assert_allclose(evolved, expected, rtol=0.0, atol=1.0e-14)
        if mode == "plus":
            traveling = np.exp(1.0j * CANONICAL_K * centers - 1.0j * omega * ADVANCE_TIME)
            np.testing.assert_allclose(phase, traveling, rtol=0.0, atol=1.0e-14)
        hat0 = amplitude * vector
        hat_t = exact.advance_fourier(hat0, ADVANCE_TIME, k=CANONICAL_K)
        np.testing.assert_allclose(
            hat_t,
            hat0 * np.exp(eigenvalue * ADVANCE_TIME),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_write_cp06_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_siblings_use_load_sibling_module():
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
    assert "from exact import" not in exact_text


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16, mode="plus")
    assert plan is not None


@pytest.mark.compiler
def test_compiler_run_native_returns_field_when_kokkos_present():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05, mode="plus"), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="CP-06",
        pops_native_dim=1,
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
    written = analyze.write_cp06_report(tmp_path)
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
    written = analyze.write_cp06_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
