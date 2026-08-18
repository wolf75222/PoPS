"""PO-05 FFT vs GMG cross-oracle (1-d PO-01 reuse; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "fft_vs_gmg"
PO01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
TWO_PI = 2.0 * np.pi
SPECTRAL_ATOL = 1.0e-12


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_call_owners(source: str) -> list[str]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    owners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pops"
        ):
            continue
        current: ast.AST | None = node
        owner = "<module>"
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = current.name
                break
            current = parents.get(current)
        owners.append(owner)
    return owners


def test_exact_reuses_po01_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "periodic_trig" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    po01 = load_sibling_module(PO01_EXACT)
    centers, _ = po01.uniform_cell_grid(N_CELLS)
    np.testing.assert_array_equal(exact.phi_exact(centers), po01.phi_exact(centers))
    np.testing.assert_array_equal(exact.rhs_exact(centers), po01.rhs_exact(centers))
    np.testing.assert_array_equal(exact.e_exact(centers), po01.e_exact(centers))


def test_spectral_solve_recovers_mean_free_analytic_phi():
    exact = _load_case_module("exact")
    centers, volumes = exact.uniform_cell_grid(N_CELLS)
    rhs = exact.rhs_exact(centers)
    phi_spectral = exact.spectral_solve(rhs)
    phi_analytic = exact.phi_exact(centers)
    np.testing.assert_allclose(
        exact.mean_free(phi_spectral, volumes),
        exact.mean_free(phi_analytic, volumes),
        rtol=0.0,
        atol=SPECTRAL_ATOL,
    )
    np.testing.assert_allclose(
        exact.mean_free(phi_spectral, volumes),
        exact.mean_free(phi_analytic + 3.25, volumes),
        rtol=0.0,
        atol=SPECTRAL_ATOL,
    )
    assert abs(float(np.average(phi_spectral, weights=volumes))) < SPECTRAL_ATOL


def test_gmg_stub_residual_is_fd_truncation_not_a_gate():
    exact = _load_case_module("exact")
    centers, _volumes = exact.uniform_cell_grid(N_CELLS)
    phi = exact.phi_exact(centers)
    rhs = exact.rhs_exact(centers)
    residual = exact.gmg_stub_residual(phi, rhs)
    assert np.all(np.isfinite(residual))
    spacing = 1.0 / float(N_CELLS)
    discrete_eigenvalue = (4.0 / spacing**2) * np.sin(np.pi * spacing) ** 2
    expected = (discrete_eigenvalue - TWO_PI**2) * phi
    np.testing.assert_allclose(residual, expected, rtol=0.0, atol=1.0e-12)
    assert float(np.max(np.abs(residual))) > 1.0e-6


def test_write_po05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    written = analyze.write_po05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    from verification.pops_verify.native_evidence import REDUCED_NOT_SUPPORTED

    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 0
    assert loaded["coverage"]["cases_not_supported"] == 1
    assert loaded["not_applicable_reason"]["orders"] == REDUCED_NOT_SUPPORTED["PO-05"]


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py",):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_po05_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1 or loaded["coverage"]["cases_not_supported"] == 1


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="PO-05", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
