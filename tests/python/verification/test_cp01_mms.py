"""CP-01 Euler–Poisson MMS (in-memory oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "mms"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width, width


def test_electron_density_is_positive():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(64)
    for time in (0.0, 0.25, 0.5, 1.0):
        fields = exact.fields_1d(x, time)
        assert np.all(fields["n_e"] > 0.0)
        assert np.all(fields["u_e"] > 0.0)
        assert np.all(fields["p_e"] > 0.0)


def test_poisson_identity_ni_minus_ne_equals_eps0_k2_phi_over_e():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(64)
    time = 0.37
    fields = exact.fields_1d(x, time)
    charge = exact.N_I - fields["n_e"]
    expected = (exact.EPS0 * exact.K**2 / exact.E_CHARGE) * fields["phi"]
    np.testing.assert_allclose(charge, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        fields["phi"],
        exact.A * np.cos(exact.K * x - exact.OMEGA * time),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_write_cp01_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["poisson"]["residual_l2"] == 0.0
    assert loaded["coupling"]["sign_ok"] is True


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text
