"""CP-08 uniform-E acceleration (in-memory oracle; no solver required)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "uniform_e_accel"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def _imports_pops_package(text: str) -> bool:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "pops" or alias.name.startswith("pops.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "pops" or node.module.startswith("pops."):
                return True
    return False


def test_velocity_is_linear_in_time():
    exact = _load("exact")
    charge, mass, e0, u0 = 2.0, 0.5, 3.0, -1.0
    times = np.linspace(0.0, 2.0, 5)
    expected = u0 + (charge / mass) * e0 * times
    computed = np.array(
        [
            exact.velocity(time, q=charge, mass=mass, e0=e0, u0=u0)
            for time in times
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(computed, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(computed, n=2), 0.0, rtol=0.0, atol=1.0e-14)


def test_opposite_charges_have_opposite_accelerations():
    exact = _load("exact")
    charge, mass, e0, u0, time = 1.5, 2.0, 0.8, 0.3, 0.4
    plus = exact.velocity(time, q=charge, mass=mass, e0=e0, u0=u0)
    minus = exact.velocity(time, q=-charge, mass=mass, e0=e0, u0=u0)
    np.testing.assert_allclose(plus - u0, -(minus - u0), rtol=0.0, atol=1.0e-14)
    plus_acc = exact.acceleration(q=charge, mass=mass, e0=e0)
    minus_acc = exact.acceleration(q=-charge, mass=mass, e0=e0)
    np.testing.assert_allclose(plus_acc, -minus_acc, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(plus_acc, (charge / mass) * e0, rtol=0.0, atol=1.0e-14)


def test_kinetic_energy_formula():
    exact = _load("exact")
    charge, mass, e0, u0, n0, time = 1.0, 2.0, 0.5, 0.25, 3.0, 0.8
    velocity = exact.velocity(time, q=charge, mass=mass, e0=e0, u0=u0)
    kinetic = exact.kinetic_energy_density(
        time, q=charge, mass=mass, e0=e0, u0=u0, n0=n0
    )
    np.testing.assert_allclose(
        kinetic, 0.5 * n0 * mass * velocity * velocity, rtol=0.0, atol=1.0e-14
    )


def test_density_is_uniform_and_unchanged():
    exact = _load("exact")
    centers = (np.arange(N_CELLS, dtype=np.float64) + 0.5) / float(N_CELLS)
    n0 = 1.7
    for time in (0.0, 0.3, 1.0, 5.0):
        density = exact.density(centers, time, n0=n0)
        assert density.shape == centers.shape
        np.testing.assert_allclose(density, n0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_array_equal(
        exact.density(centers, 0.0, n0=n0),
        exact.density(centers, 5.0, n0=n0),
    )


def test_write_cp08_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load("analyze")
    written = analyze.write_cp08_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [1]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "from exact import" not in text
        if name in ("run.py", "analyze.py"):
            assert "load_sibling_module" in text
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text
            assert not _imports_pops_package(text)


def test_resolve_plan_returns_without_authoring_pending():
    run = _load("run")
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    from tests.python.support.requirements import missing_compiler_requirement

    run = _load("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (3, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
