"""TM-07 field update every RK stage (in-memory contract plus public Case)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "rk_field_stages"
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


def test_exact_documents_ssprk_stage_counts_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    assert exact.SSPRK2_STAGES == 2
    assert exact.SSPRK3_STAGES == 3
    assert exact.stage_count("SSPRK2") == 2
    assert exact.stage_count("SSPRK3") == 3


def test_required_field_solves_equals_stage_count():
    exact = _load_case_module("exact")
    assert exact.required_field_solves(exact.SSPRK2_STAGES) == 2
    assert exact.required_field_solves(exact.SSPRK3_STAGES) == 3
    assert exact.required_field_solves(2) == 2
    assert exact.required_field_solves(3) == 3


def test_ssprk2_requires_two_field_solves_per_step():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert run.field_solves_per_step("SSPRK2") == 2


def test_ssprk3_requires_three_field_solves_per_step():
    run = _load_case_module("run")
    assert run.field_solves_per_step("SSPRK3") == 3


def test_frozen_field_is_negative_control_one_solve_per_step():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert exact.FROZEN_FIELD_SOLVES_PER_STEP == 1
    assert run.field_solves_per_step("SSPRK2", frozen_field=True) == 1
    assert run.field_solves_per_step("SSPRK3", frozen_field=True) == 1
    assert run.field_solves_per_step("SSPRK2", frozen_field=True) < run.field_solves_per_step(
        "SSPRK2"
    )
    assert run.field_solves_per_step("SSPRK3", frozen_field=True) < run.field_solves_per_step(
        "SSPRK3"
    )


def test_write_tm07_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tm07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "SSPRK2(" in text
    assert "fields=" in text
    case = run.build_case(16)
    plan = run.resolve_plan(16)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


@pytest.mark.compiler
def test_run_native_returns_finite_field_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(n_cells=16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
