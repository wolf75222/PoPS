"""CP-03 warm Langmuir dispersion (in-memory oracle; no solver required)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_warm"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
OMEGA_PE = 1.0
C_E = 0.2
WAVE_NUMBERS_OVER_2PI = (1, 2, 4, 8)


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_dispersion_formula_at_canonical_wavenumbers():
    exact = _load_case_module("exact")
    for cycles in WAVE_NUMBERS_OVER_2PI:
        wavenumber = 2.0 * np.pi * float(cycles)
        omega = exact.angular_frequency(wavenumber)
        np.testing.assert_allclose(
            omega**2,
            OMEGA_PE**2 + C_E**2 * wavenumber**2,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_write_cp03_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("run_native"):
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


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


def test_isothermal_conserved_shape_matches_oracle():
    run = _load_case_module("run")
    conserved = run.initial_conserved(16, cycles=1)
    assert conserved.shape == (2, 16)
    fields = run.initial_fields(16, cycles=1)
    np.testing.assert_allclose(conserved[0], fields["n_e"])
    np.testing.assert_allclose(conserved[1], fields["n_e"] * fields["u_e"])


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05, cycles=1), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
