"""PF-06 Euler–Poisson step stand-in plus optional CP-02 native timing."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "ep_step"
CP02_RUN = (
    REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "run.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
PIPELINE_SEGMENTS = ("halo", "hyperbolic", "charge", "poisson", "gradient", "source")
SEGMENT_NAMES = PIPELINE_SEGMENTS + ("total",)
ORDERS_REASON = "pipeline segment stand-in, not a timed PF run"


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


def test_seven_segments_present():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert tuple(exact.PIPELINE_SEGMENTS) == PIPELINE_SEGMENTS
    assert tuple(exact.SEGMENT_NAMES) == SEGMENT_NAMES
    assert len(exact.SEGMENT_NAMES) == 7
    result = run.run_ep_step()
    assert list(result["segments"]) == list(SEGMENT_NAMES)
    assert set(result["timings"]) == set(SEGMENT_NAMES)


def test_total_equals_sum_of_segments():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    result = run.run_ep_step()
    timings = result["timings"]
    for name in PIPELINE_SEGMENTS:
        assert timings[name] > 0.0
    expected = sum(float(timings[name]) for name in PIPELINE_SEGMENTS)
    assert timings["total"] == expected
    assert exact.pipeline_total(timings) == timings["total"]


def test_write_pf06_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_run_native_points_at_official_benchmarks():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "benchmarks/manifest.toml" in text
    with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
        run.run_native()



def test_run_native_points_at_official_benchmarks():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "benchmarks/manifest.toml" in text
    with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
        run.run_native()

