"""AM-01 static coarse-fine wave (public 1-d AMR authoring plus leftover helpers)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "static_cf_wave"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")


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


def test_exact_translation_still_holds_via_tr01_load_sibling_module():
    assert TR01_EXACT.is_file()
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    centers, h_fine, h_coarse = exact.static_cf_centers()
    assert exact.INTERFACE_X == 0.5
    left = centers[centers < exact.INTERFACE_X]
    right = centers[centers > exact.INTERFACE_X]
    assert left.size > 0
    assert right.size > 0
    assert h_coarse > h_fine
    np.testing.assert_allclose(np.diff(left), h_coarse)
    np.testing.assert_allclose(np.diff(right), h_fine)
    np.testing.assert_allclose(
        exact.distance_to_interface(centers),
        np.abs(centers - 0.5),
    )
    np.testing.assert_allclose(
        exact.exact_sine(centers, 1.0, a=1.0, k=1.0),
        exact.exact_sine(centers, 0.0),
    )


def test_e_cf_is_defined():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert "interface_band_mask" in text
    assert "band_max_abs_error" in text
    exact_match = run.interface_bulk_errors(leftover=False)
    assert isinstance(exact_match["e_cf"], float)
    assert math.isfinite(exact_match["e_cf"])
    assert exact_match["e_cf"] == 0.0
    leftover = run.interface_bulk_errors()
    assert isinstance(leftover["e_cf"], float)
    assert math.isfinite(leftover["e_cf"])
    assert leftover["e_cf"] != 0.0


def test_e_cf_over_e_bulk_is_an_observation():
    run = _load_case_module("run")
    leftover = run.interface_bulk_errors()
    assert leftover["e_bulk"] != 0.0
    assert math.isfinite(leftover["e_bulk"])
    assert leftover["ratio"] is not None
    assert leftover["ratio"] != 0.0
    assert math.isfinite(leftover["ratio"])
    np.testing.assert_allclose(leftover["ratio"], leftover["e_cf"] / leftover["e_bulk"])


def test_write_am01_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["amr"]["interface_error"] is not None
    assert loaded["amr"]["bulk_error"] is not None
    assert math.isfinite(loaded["amr"]["interface_error"])
    assert math.isfinite(loaded["amr"]["bulk_error"])


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    case = run.build_case(8)
    plan = run.resolve_plan(n_coarse=8)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(run.run_native(8, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert field.size > 0
    assert field.shape == (8,)
    assert np.isfinite(field).all()
