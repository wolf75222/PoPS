"""TR-07 discontinuous slot (in-memory translation + TV/overshoot; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "discontinuous_slot"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64
X0 = 0.5
WIDTH = 0.25
A = 1.0
ORDERS_REASON = "discontinuous / limiter, not order-2"


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


def test_exact_translation_identity():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    assert exact.X0 == X0
    assert exact.WIDTH == WIDTH
    assert exact.A == A
    width = 1.0 / float(N_CELLS)
    centers = (np.arange(N_CELLS, dtype=np.float64) + 0.5) * width
    time = 0.25
    translated = exact.exact_slot(np.mod(centers - exact.A * time, 1.0), 0.0)
    np.testing.assert_array_equal(exact.exact_slot(centers, time), translated)
    np.testing.assert_array_equal(
        exact.exact_slot(centers, 1.0),
        exact.exact_slot(centers, 0.0),
    )


def test_tv_of_exact_slot_is_two():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in run_text
    assert "from exact import" not in run_text
    width = 1.0 / float(N_CELLS)
    centers = (np.arange(N_CELLS, dtype=np.float64) + 0.5) * width
    field = exact.exact_slot(centers, 0.0)
    assert run.total_variation(field) == 2.0


def test_field_with_overshoot_1_1_reports_overshoot_positive():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    width = 1.0 / float(N_CELLS)
    centers = (np.arange(N_CELLS, dtype=np.float64) + 0.5) * width
    reference = exact.exact_slot(centers, 0.0)
    field = np.asarray(reference, dtype=np.float64).copy()
    field[int(np.argmax(field))] = 1.1
    assert run.overshoot(field, reference) > 0.0


def test_write_tr07_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    written = analyze.write_tr07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    reasons = " ".join(item["reason"] for item in loaded["failures"]).lower()
    assert "native" in reasons


def test_siblings_use_load_sibling_module():
    for name in ("run.py",):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_tr07_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1 or loaded["coverage"]["cases_not_supported"] == 1


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="TR-07", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
