"""TM-03 exact collision relaxation (in-memory ODE plus public Case resolve)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "collision_relax"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
DT = 1.0 / 64.0


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


def test_exact_exponential_relaxation_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" not in text or "from exact import" not in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    u0 = 3.0
    u_bar = 1.0
    nu = float(exact.NU)
    assert nu > 0.0
    np.testing.assert_allclose(exact.exact_relax(u0, 0.0, nu=nu, u_bar=u_bar), u0)
    half_life = math.log(2.0) / nu
    np.testing.assert_allclose(
        exact.exact_relax(u0, half_life, nu=nu, u_bar=u_bar),
        u_bar + 0.5 * (u0 - u_bar),
    )
    times = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float64)
    expected = u_bar + (u0 - u_bar) * np.exp(-nu * times)
    np.testing.assert_allclose(
        [exact.exact_relax(u0, float(t), nu=nu, u_bar=u_bar) for t in times],
        expected,
    )
    late = exact.exact_relax(u0, 40.0 / nu, nu=nu, u_bar=u_bar)
    np.testing.assert_allclose(late, u_bar, atol=1.0e-15)


def test_barycenter_moment_is_constant():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in run_text
    assert "from exact import" not in run_text
    centers, volumes = exact.uniform_cell_centers()
    u0 = exact.initial_field(centers)
    u_bar = exact.barycenter(u0, volumes)
    np.testing.assert_allclose(u_bar, 1.0)
    for time in (0.0, 0.1, 0.5, 1.0, 4.0):
        relaxed = exact.exact_relax(u0, time, nu=exact.NU, u_bar=u_bar)
        np.testing.assert_allclose(exact.barycenter(relaxed, volumes), u_bar)
        advanced = run.relax(u0, time, nu=exact.NU, volumes=volumes)
        np.testing.assert_allclose(exact.barycenter(advanced, volumes), u_bar)
        np.testing.assert_allclose(advanced, relaxed)


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(DT)
    plan = run.resolve_plan(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


def test_write_tm03_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tm03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert owners
            assert set(owners) <= {"run_native", "run_native_two_species"}
            assert "run_native" in owners
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_utility_python_two_species_relax_is_not_scientific_evidence():
    run = _load_case_module("run")
    q0 = 1.0
    q1 = 0.0
    stiffness = float(run.K)
    rho1 = float(run.RHO1)
    rho2 = float(run.RHO2)
    lam = stiffness * (1.0 / rho1 + 1.0 / rho2)
    np.testing.assert_allclose(lam, 2.0)
    np.testing.assert_allclose(run.collision_lambda(), lam)
    v0 = (rho1 * q0 + rho2 * q1) / (rho1 + rho2)
    w0 = q0 - q1
    for time in (0.0, 0.1, 0.5, 1.0, 4.0):
        q0_t, q1_t = run.relax_two_species(q0, q1, time)
        np.testing.assert_allclose(
            run.two_species_barycenter(q0_t, q1_t),
            v0,
        )
        np.testing.assert_allclose(q0_t - q1_t, w0 * math.exp(-lam * time))
    q0_u, q1_u = run.relax_two_species(2.0, -1.0, 0.5, k=3.0, rho1=2.0, rho2=4.0)
    mass = 6.0
    v_u = (2.0 * 2.0 + 4.0 * (-1.0)) / mass
    lam_u = 3.0 * (1.0 / 2.0 + 1.0 / 4.0)
    w_u = (2.0 - (-1.0)) * math.exp(-lam_u * 0.5)
    np.testing.assert_allclose(
        run.two_species_barycenter(q0_u, q1_u, rho1=2.0, rho2=4.0),
        v_u,
    )
    np.testing.assert_allclose(q0_u - q1_u, w_u)


def test_two_species_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_two_species(DT)
    plan = run.two_species_resolve(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert 'components=("q0", "q1")' in text
    assert 'components=("u", "v")' not in text


@pytest.mark.compiler
def test_compiler_run_native_returns_field_when_kokkos_present():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(DT, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.size == N_CELLS
    assert np.isfinite(field).all()


@pytest.mark.compiler
def test_run_native_two_species_returns_finite_field_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native_two_species(DT, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.size == 2 * int(run.TWO_SPECIES_N_CELLS)
    assert np.isfinite(field).all()


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="TM-03",
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
    written = analyze.write_tm03_report(tmp_path)
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
    written = analyze.write_tm03_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
