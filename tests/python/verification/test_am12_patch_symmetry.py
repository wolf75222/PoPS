"""AM-12 patch-shape / reflection symmetry (public 1-d AMR plus leftover helpers)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "patch_symmetry"
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


def test_reflected_interface_mask_equals_spatially_reversed_original_mask():
    assert TR01_EXACT.is_file()
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in run_text
    assert "from exact import" not in run_text
    assert "interface_band_mask" in run_text
    assert "band_max_abs_error" in run_text
    assert exact.INTERFACE_X == 0.5
    original = run.patch_sample(reflected=False)
    reflected = run.patch_sample(reflected=True)
    np.testing.assert_allclose(
        reflected["centers"],
        exact.reflect_x(original["centers"])[::-1],
    )
    np.testing.assert_allclose(
        exact.distance_to_interface(reflected["centers"]),
        np.abs(reflected["centers"] - 0.5),
    )
    assert original["interface"].dtype == bool
    assert reflected["interface"].dtype == bool
    np.testing.assert_array_equal(reflected["interface"], original["interface"][::-1])
    left = original["centers"][original["centers"] < exact.INTERFACE_X]
    right = original["centers"][original["centers"] > exact.INTERFACE_X]
    assert left.size > 0
    assert right.size > 0
    assert original["h_coarse"] > original["h_fine"]
    np.testing.assert_allclose(np.diff(left), original["h_coarse"])
    np.testing.assert_allclose(np.diff(right), original["h_fine"])
    reflected_left = reflected["centers"][reflected["centers"] < exact.INTERFACE_X]
    reflected_right = reflected["centers"][reflected["centers"] > exact.INTERFACE_X]
    np.testing.assert_allclose(np.diff(reflected_left), reflected["h_fine"])
    np.testing.assert_allclose(np.diff(reflected_right), reflected["h_coarse"])


def test_e_cf_unchanged_under_reflection_of_leftover_and_oracle():
    run = _load_case_module("run")
    original = run.patch_sample()
    reflected = run.reflect_leftover_and_oracle(original)
    assert isinstance(original["e_cf"], float)
    assert math.isfinite(original["e_cf"])
    assert original["e_cf"] != 0.0
    assert math.isfinite(reflected["e_cf"])
    np.testing.assert_allclose(reflected["e_cf"], original["e_cf"])
    np.testing.assert_array_equal(reflected["interface"], original["interface"][::-1])
    np.testing.assert_allclose(reflected["field"], original["field"][::-1])
    np.testing.assert_allclose(reflected["oracle"], original["oracle"][::-1])
    exact_match = run.patch_sample(leftover=False)
    assert exact_match["e_cf"] == 0.0
    reflected_exact = run.reflect_leftover_and_oracle(exact_match)
    assert reflected_exact["e_cf"] == 0.0


def test_write_am12_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am12_report(tmp_path)
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
    assert loaded["amr"]["invariants_ok"] is True


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
        assert "from exact import" not in text


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
