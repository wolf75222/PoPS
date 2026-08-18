"""GE-04 same radial oracle on Cartesian vs polar (in-memory; no live runtime)."""
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

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "geometry" / "cartesian_polar_oracle"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RING_RADIUS = 0.5
SIGMA = 0.08
N_CELLS = 32
LINF_BOUND = 0.05
POLAR_RUNTIME_REFUSAL = "public polar System not active"


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _imports_pops(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "pops" or alias.name.startswith("pops.") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "pops" or node.module.startswith("pops."):
                return True
    return False


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


def test_both_samplings_peak_at_ring_radius():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert not _imports_pops(text)
    assert exact.SIGMA == SIGMA
    assert exact.RING_RADIUS == RING_RADIUS
    assert exact.N_CELLS == N_CELLS
    np.testing.assert_allclose(exact.phi_of_r(RING_RADIUS), 1.0, rtol=0.0, atol=0.0)
    assert float(exact.phi_of_r(RING_RADIUS)) > float(exact.phi_of_r(0.4))
    assert float(exact.phi_of_r(RING_RADIUS)) > float(exact.phi_of_r(0.6))
    cartesian_peak = exact.peak_radius_cartesian()
    polar_peak = exact.peak_radius_polar()
    assert math.isfinite(cartesian_peak)
    assert math.isfinite(polar_peak)
    assert abs(cartesian_peak - RING_RADIUS) <= 2.0 / float(N_CELLS)
    polar_spacing = float(exact.R_OUTER) / float(exact.N_R)
    assert abs(polar_peak - RING_RADIUS) <= polar_spacing


def test_interpolated_field_to_field_linf_is_small():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert run.LINF_BOUND == LINF_BOUND
    errors = run.field_to_field_errors()
    assert math.isfinite(errors.l1)
    assert math.isfinite(errors.l2)
    assert math.isfinite(errors.linf)
    assert errors.linf < LINF_BOUND


def test_polar_branch_is_capability_gated():
    payload = tomllib.loads((CASE_DIR / "case.toml").read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "capability-gated"
    assert payload["polar"]["branch"] == "capability-gated"
    assert "polar_system_runtime" in payload["requires"]
    run = _load_case_module("run")
    reason = run.refuse_public_polar_runtime()
    assert isinstance(reason, str)
    assert reason == POLAR_RUNTIME_REFUSAL
    with pytest.raises(run.NativeUnavailable, match=POLAR_RUNTIME_REFUSAL):
        run.run_native()


def test_write_ge04_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_ge04_report(tmp_path)
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
        assert "from exact import" not in text
