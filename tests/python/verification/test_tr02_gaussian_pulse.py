"""TR-02 Gaussian pulse: oracle utilities plus fail-closed native evidence."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tr02_gaussian_pulse_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uniform_cells(n_cells: int = 128):
    width = 1.0 / int(n_cells)
    centers = (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width
    volumes = np.full(int(n_cells), width, dtype=np.float64)
    return centers, volumes


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_calls_outside_run_native(source: str) -> list[str]:
    tree = ast.parse(source)
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_native":
            allowed.update(id(child) for child in ast.walk(node))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if id(node) in allowed or not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "run":
            value = func.value
            if isinstance(value, ast.Name) and value.id == "pops":
                offenders.append("pops.run")
    return offenders


def test_translation_preserves_l2_of_exact_against_translated_exact():
    exact = _load_case_module("exact")
    x, volumes = _uniform_cells()
    time = 0.25
    speed = 1.0
    field = exact.exact_gaussian(x, time, a=speed)
    translated = exact.exact_gaussian((x - speed * time) % 1.0, 0.0, a=speed)
    errors = reference_errors(field, translated, volumes)
    assert errors.l2 == pytest.approx(0.0, abs=1.0e-14)


def test_barycenter_of_exact_pulse_translates_by_a_t():
    exact = _load_case_module("exact")
    analyze = _load_case_module("analyze")
    x, volumes = _uniform_cells(256)
    x0 = 0.37
    time = 0.2
    field = exact.exact_gaussian(x, time, q0=0.0, x0=x0, a=1.0)
    barycenter = analyze.pulse_barycenter(x, field, volumes, 0.0)
    assert barycenter == pytest.approx((x0 + time) % 1.0, abs=1.0e-6)


def test_pulse_mass_of_exact_field_is_independent_of_t():
    exact = _load_case_module("exact")
    analyze = _load_case_module("analyze")
    x, volumes = _uniform_cells(256)
    q0 = 0.0
    masses = [
        analyze.pulse_mass(exact.exact_gaussian(x, time, q0=q0) - q0, volumes)
        for time in (0.0, 0.2, 0.7)
    ]
    assert masses[0] == pytest.approx(masses[1], rel=1.0e-10, abs=1.0e-12)
    assert masses[0] == pytest.approx(masses[2], rel=1.0e-10, abs=1.0e-12)


def test_observed_order_utility_recovers_quadratic_spacing():
    spacings = np.asarray([1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0])
    orders = observed_order(spacings**2, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))


def test_resolve_plan_succeeds_without_native():
    run = _load_case_module("run")
    native_was_loaded = "pops._pops" in sys.modules
    resolved = run.resolve_plan(16)
    assert resolved is not None
    assert ("pops._pops" in sys.modules) is native_was_loaded


def test_report_cannot_pass_without_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_tr02_report(tmp_path)
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    assert loaded["orders"] == []
    reasons = " ".join(item["reason"] for item in loaded["failures"]).lower()
    assert "native" in reasons


def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_tr02_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings, "variable": "q"},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1 or loaded["coverage"]["cases_not_supported"] == 1


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="TR-02", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
    field = np.asarray(result["result"], dtype=np.float64)
    assert field.shape == (16,)
    assert np.isfinite(field).all()


def test_modules_do_not_hardcode_pops_run_outside_run_native():
    for name in CASE_MODULES:
        source = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_calls_outside_run_native(source) == []
        if name != "run.py":
            assert "pops.run(" not in source
        assert "from exact import" not in source
