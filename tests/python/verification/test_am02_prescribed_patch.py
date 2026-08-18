"""AM-02 prescribed moving patch (public 1-d AMR authoring plus in-memory helpers)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "prescribed_patch"
TR02_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _load_tr02(name: str):
    return load_sibling_module(TR02_DIR / f"{name}.py")


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


def test_prescribed_patch_follows_x0_plus_a_t():
    run = _load_case_module("run")
    exact = _load_case_module("exact")
    tr02 = _load_tr02("exact")
    tr02_analyze = _load_tr02("analyze")
    x0 = tr02.X0
    speed = tr02.A
    time = 0.2
    n_cells = 256
    width = 1.0 / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    volumes = np.full(n_cells, width, dtype=np.float64)
    field = tr02.exact_gaussian(centers, time, x0=x0, a=speed)
    barycenter = tr02_analyze.pulse_barycenter(centers, field, volumes, tr02.Q0)
    expected = (x0 + speed * time) % 1.0
    center = run.prescribed_patch_center(time)
    assert exact.STRESS_CYCLES == 256
    assert center == pytest.approx(expected, abs=1.0e-6)
    assert center == pytest.approx(barycenter, abs=1.0e-6)
    assert exact.patch_center(time, x0=x0, a=speed) == pytest.approx(expected, abs=1.0e-14)


def test_256_cycle_stress_mass_drift_is_zero_for_exact():
    run = _load_case_module("run")
    exact = _load_case_module("exact")
    assert exact.STRESS_CYCLES == 256
    drift = run.stress_mass_drift(n_cycles=exact.STRESS_CYCLES)
    assert drift == pytest.approx(0.0, abs=1.0e-12)


def test_regrid_errors_are_two_scalars_with_manufactured_h2_jump():
    exact = _load_case_module("exact")
    coarse = 1.0 / 32.0
    fine = 1.0 / 64.0
    before_c, after_c = exact.manufactured_regrid_errors(coarse)
    before_f, after_f = exact.manufactured_regrid_errors(fine)
    assert isinstance(before_c, float) and isinstance(after_c, float)
    jump_c = after_c - before_c
    jump_f = after_f - before_f
    assert jump_c > 0.0
    assert jump_c / jump_f == pytest.approx((coarse / fine) ** 2, rel=1.0e-12)


def test_write_am02_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["amr"]["invariants_ok"] is True


def test_siblings_reuse_tr02_via_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "gaussian_pulse" in text


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
    plan = run.resolve_plan(n_cells=8)
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
