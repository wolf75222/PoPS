"""PO-05 FFT vs GMG cross-oracle (1-d PO-01 reuse; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

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
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_po05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["poisson"]["potential_error"] < SPECTRAL_ATOL
    assert loaded["poisson"]["residual_l2"] > 1.0e-6
    assert loaded["failures"] == []
    assert loaded["coverage"]["cases_passed"] == 1


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
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
