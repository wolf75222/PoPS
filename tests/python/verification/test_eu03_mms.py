"""EU-03 Euler MMS (in-memory oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "mms"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
TWO_PI = 2.0 * np.pi


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width, width


def test_density_and_pressure_are_positive():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(64)
    y = x
    for time in (0.0, 0.3, 1.0, 2.5):
        primitives = exact.primitives_1d(x, time)
        assert primitives.shape[0] == 3
        assert np.all(primitives[0] > 0.0)
        assert np.all(primitives[2] > 0.0)
        two_d = exact.primitives_2d(x, y, time)
        assert two_d.shape[0] == 4
        assert np.all(two_d[0] > 0.0)
        assert np.all(two_d[3] > 0.0)


def test_exact_time_shift_matches_spatial_shift():
    exact = _load_case_module("exact")
    x, _ = _cell_centers()
    time = 0.17
    evolved = exact.primitives_1d(x, time)
    shifted = exact.primitives_1d(x - time, 0.0)
    np.testing.assert_allclose(evolved, shifted, rtol=0.0, atol=1.0e-14)
    expected_rho = 2.0 + 0.1 * np.sin(TWO_PI * (x - time))
    expected_u = 0.3 + 0.1 * np.cos(TWO_PI * (x - time))
    expected_p = 1.0 + 0.05 * np.sin(TWO_PI * (x - time))
    np.testing.assert_allclose(evolved[0], expected_rho, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(evolved[1], expected_u, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(evolved[2], expected_p, rtol=0.0, atol=1.0e-14)


def test_write_eu03_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_eu03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


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
