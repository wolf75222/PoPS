"""TR-06 axis permutation / reflection (in-memory 2-d sine product; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "axis_permutation"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
T = 0.125
KX = 1.0
KY = 2.0
AX = 1.0
AY = 0.5


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


def test_exact_loads_tr01_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    tr01 = load_sibling_module(TR01_EXACT)
    x1d, _ = tr01.uniform_cell_centers(N_CELLS)
    np.testing.assert_array_equal(
        exact.exact_sine(x1d, T, a=AX, k=KX),
        tr01.exact_sine(x1d, T, a=AX, k=KX),
    )
    x, y = np.meshgrid(x1d, x1d, indexing="ij")
    sx = (tr01.exact_sine(x, T, a=AX, k=KX) - tr01.Q0) / tr01.EPS
    sy = (tr01.exact_sine(y, T, a=AY, k=KY) - tr01.Q0) / tr01.EPS
    np.testing.assert_array_equal(
        exact.exact_product(x, y, T, kx=KX, ky=KY, ax=AX, ay=AY),
        tr01.Q0 + tr01.EPS * sx * sy,
    )


def test_permutation_identity_linf_is_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    original, at_swapped, mapped, volumes = run.mapped_permutation_fields(
        n_cells=N_CELLS, t=T
    )
    assert original.shape == (N_CELLS, N_CELLS)
    assert at_swapped.shape == original.shape
    assert mapped.shape == original.shape
    assert not np.array_equal(original, at_swapped)
    np.testing.assert_array_equal(mapped, original)
    errors = reference_errors(mapped, original, volumes)
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    assert run.permutation_linf(n_cells=N_CELLS, t=T) == 0.0
    x, y, _, _ = exact.uniform_grid_2d(N_CELLS)
    swapped_x, swapped_y = exact.permute_xy(x, y)
    np.testing.assert_array_equal(swapped_x, y)
    np.testing.assert_array_equal(swapped_y, x)


def test_reflection_identity_linf_is_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    original, at_reflected, mapped, volumes = run.mapped_reflection_fields(
        n_cells=N_CELLS, t=T
    )
    assert original.shape == (N_CELLS, N_CELLS)
    assert at_reflected.shape == original.shape
    assert mapped.shape == original.shape
    assert not np.array_equal(original, at_reflected)
    np.testing.assert_array_equal(mapped, original)
    errors = reference_errors(mapped, original, volumes)
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    assert run.reflection_linf(n_cells=N_CELLS, t=T) == 0.0
    x, y, _, _ = exact.uniform_grid_2d(N_CELLS)
    reflected = exact.reflect_x(x)
    np.testing.assert_allclose(reflected + x, 1.0)


def test_write_tr06_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tr06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert 2 in loaded["native_dimensions"]


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
