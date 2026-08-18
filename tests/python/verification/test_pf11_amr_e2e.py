"""PF-11 dynamic AMR e2e (in-memory stand-in; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "amr_e2e"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_WARMUP = 2
N_STEPS = 50
REGRID_EVERY = 8
EXPECTED_REBUILDS = N_STEPS // REGRID_EVERY
ORDERS_REASON = "dynamic AMR e2e stand-in, not a timed PF run"


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


def test_rebuild_count_equals_measured_steps_over_regrid_interval():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert exact.N_WARMUP == N_WARMUP
    assert exact.N_STEPS == N_STEPS
    assert exact.REGRID_EVERY == REGRID_EVERY
    assert exact.expected_rebuilds() == EXPECTED_REBUILDS
    assert run.count_rebuilds() == EXPECTED_REBUILDS


def test_warmup_not_counted():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    sample = run.advance_fake_amr()
    assert sample["n_warmup"] == exact.N_WARMUP == N_WARMUP
    assert sample["n_steps"] == exact.N_STEPS == N_STEPS
    assert sample["warmup_rebuilds"] >= 1
    assert sample["rebuilds"] == exact.expected_rebuilds()
    assert sample["rebuilds"] == (
        sample["rebuilds_including_warmup"] - sample["warmup_rebuilds"]
    )
    for step in sample["rebuild_steps"]:
        assert step >= exact.N_WARMUP


def test_write_pf11_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf11_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON
    one_node = loaded["performance"]["one_node"]
    assert one_node["cells_per_second"] > 0.0
    assert "fake" in one_node["notes"].lower()


def test_run_native_points_at_official_benchmarks():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "benchmarks/manifest.toml" in text
    with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
        run.run_native()



def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
