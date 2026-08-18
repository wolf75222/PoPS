"""GE-05 polar axis regularity / volume conservation (in-memory; no live runtime)."""
from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "geometry" / "polar_axis"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
R_IN = 0.2
R_OUT = 1.0
N_R = 8
N_THETA = 16
AXIS_DR = 0.125
AXIS_DTHETA = 2.0 * math.pi / float(N_THETA)
VOLUME_ATOL = 1.0e-14
POLAR_RUNTIME_REFUSAL = "public polar System not active"


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


def _divisor_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Constant) and node.value == 0:
        names.add("0")
    for child in ast.iter_child_nodes(node):
        names.update(_divisor_names(child))
    return names


def test_annulus_volume_matches_analytic_area():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert "import pops" not in text
    assert exact.R_IN == R_IN
    assert exact.R_OUT == R_OUT
    expected = math.pi * (R_OUT * R_OUT - R_IN * R_IN)
    np.testing.assert_allclose(
        exact.annulus_area(R_IN, R_OUT),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    discrete = exact.annulus_volume(R_IN, R_OUT, N_R, N_THETA)
    np.testing.assert_allclose(discrete, expected, rtol=0.0, atol=VOLUME_ATOL)
    volumes = exact.polar_cell_volumes(R_IN, R_OUT, N_R, N_THETA)
    assert volumes.shape == (N_R, N_THETA)
    dr = (R_OUT - R_IN) / float(N_R)
    dtheta = 2.0 * math.pi / float(N_THETA)
    radii = R_IN + (np.arange(N_R, dtype=np.float64) + 0.5) * dr
    for index, radius in enumerate(radii):
        cell = exact.polar_cell_volume(float(radius), dr, dtheta)
        np.testing.assert_allclose(cell, radius * dr * dtheta, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(volumes[index], cell, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(float(np.sum(volumes)), expected, rtol=0.0, atol=VOLUME_ATOL)


def test_constant_state_integral_equals_annulus_area():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    area = exact.annulus_area(R_IN, R_OUT)
    np.testing.assert_allclose(
        exact.constant_state_integral(1.0, R_IN, R_OUT, N_R, N_THETA),
        area,
        rtol=0.0,
        atol=VOLUME_ATOL,
    )
    np.testing.assert_allclose(
        run.constant_state_integral(1.0, R_IN, R_OUT, N_R, N_THETA),
        area,
        rtol=0.0,
        atol=VOLUME_ATOL,
    )
    np.testing.assert_allclose(
        run.annulus_volume(R_IN, R_OUT, N_R, N_THETA),
        area,
        rtol=0.0,
        atol=VOLUME_ATOL,
    )


def test_axis_helper_does_not_divide_by_r_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    source = inspect.getsource(exact.axis_cell_volume)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
            names = _divisor_names(node.right)
            assert "r" not in names
            assert "0" not in names
    assert "/ 0" not in source
    assert "/0" not in source
    documented = 0.5 * AXIS_DR * AXIS_DR * AXIS_DTHETA
    volume = exact.axis_cell_volume(AXIS_DR, AXIS_DTHETA)
    assert math.isfinite(volume)
    np.testing.assert_allclose(volume, documented, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        run.axis_cell_volume(AXIS_DR, AXIS_DTHETA),
        documented,
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="r=0|axis|r <= 0"):
        exact.polar_cell_volume(0.0, AXIS_DR, AXIS_DTHETA)
    reason = run.refuse_public_polar_runtime()
    assert isinstance(reason, str)
    assert reason == POLAR_RUNTIME_REFUSAL
    with pytest.raises(run.NativeUnavailable, match=POLAR_RUNTIME_REFUSAL):
        run.run_native()


def test_write_ge05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_ge05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert "capability-gated polar runtime" in loaded["not_applicable_reason"]["orders"]


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
            assert "import pops" not in text
