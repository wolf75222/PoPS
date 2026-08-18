"""CP-09 linearized Debye screen (in-memory Helmholtz oracle; no solver)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "debye_screen"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
TWO_PI = 2.0 * np.pi
LAMBDA_D = 0.1
K = 1
POISSON_LIMIT_LAMBDA_D = 1.0e8


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS) -> np.ndarray:
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width


def test_analytic_helmholtz_identity():
    exact = _load_case_module("exact")
    x = _cell_centers(64)
    force = exact.f_exact(x)
    potential = exact.phi_exact(x)
    wave = TWO_PI * float(exact.K)
    screening = 1.0 / (float(exact.LAMBDA_D) ** 2)
    np.testing.assert_allclose(exact.LAMBDA_D, LAMBDA_D, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(exact.K, K, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(force, np.cos(wave * x), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        potential,
        force / (wave**2 + screening),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        (wave**2 + screening) * potential,
        force,
        rtol=0.0,
        atol=1.0e-14,
    )
    applied = exact.apply_helmholtz(potential, x)
    np.testing.assert_allclose(applied, force, rtol=0.0, atol=1.0e-12)


def test_poisson_limit_as_lambda_d_to_infinity():
    exact = _load_case_module("exact")
    x = _cell_centers()
    force = exact.f_exact(x)
    wave = TWO_PI * float(exact.K)
    poisson = force / (wave**2)
    np.testing.assert_allclose(
        exact.poisson_gain(),
        1.0 / (wave**2),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        exact.helmholtz_gain(lambda_d=np.inf),
        exact.poisson_gain(),
        rtol=0.0,
        atol=0.0,
    )
    large = exact.phi_exact(x, lambda_d=POISSON_LIMIT_LAMBDA_D)
    np.testing.assert_allclose(large, poisson, rtol=1.0e-10, atol=1.0e-14)
    assert exact.helmholtz_gain() < exact.poisson_gain()
    screened = exact.phi_exact(x)
    assert np.max(np.abs(screened - poisson)) > 1.0e-3


def test_write_cp09_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp09_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_siblings_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text
        assert "from exact import" not in text


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="CP-09",
        pops_native_dim=1,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=8,
        resources=CampaignResources(resolutions=(8,)),
        evidence_status="required",
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    assert "resolution" in result
    assert "result" in result


def test_report_fails_closed_without_native_output(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp09_report(tmp_path)
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert (
        loaded["coverage"]["cases_failed"] + loaded["coverage"]["cases_not_supported"]
        >= 1
    )
    reasons = " ".join(item["reason"] for item in loaded["failures"])
    notes = " ".join(loaded["coverage"].get("not_tested") or [])
    blob = (reasons + " " + notes).lower()
    assert (
        "native" in blob
        or "kokkos" in blob
        or "supported" in blob
        or "required" in blob
        or "not " in blob
        or "no " in blob
    )


def test_analyze_native_requires_native_field():
    analyze = _load_case_module("analyze")
    try:
        analyze.analyze_native({})
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError("analyze_native must refuse an empty mapping")


def test_analyze_native_computes_field_errors():
    analyze = _load_case_module("analyze")
    result = analyze.analyze_native(
        {
            "field": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "oracle": np.array([1.0, 2.0, 2.5], dtype=np.float64),
            "volumes": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        }
    )
    assert result["linf"] == 0.5
    assert result["l1"] > 0.0
    assert result["l2"] > 0.0


def test_write_report_stays_fail_closed_with_native_mapping(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp09_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
