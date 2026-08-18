"""RB-08 Woodward–Colella double Mach reflection (geometry helpers only)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "double_mach"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_mesh(exact, n_x: int = N_CELLS, n_y: int = 8):
    lower_x, lower_y = (float(value) for value in exact.DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in exact.DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / float(n_x)
    width_y = (upper_y - lower_y) / float(n_y)
    x_centers = lower_x + (np.arange(n_x, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(n_y, dtype=np.float64) + 0.5) * width_y
    return np.meshgrid(x_centers, y_centers, indexing="xy")


def _independent_rankine_hugoniot(mach, gamma, rho0, u0, p0):
    sound = np.sqrt(gamma * p0 / rho0)
    density = rho0 * ((gamma + 1.0) * mach * mach) / ((gamma - 1.0) * mach * mach + 2.0)
    pressure = p0 * (2.0 * gamma * mach * mach - (gamma - 1.0)) / (gamma + 1.0)
    shock_speed = u0 + mach * sound
    velocity = shock_speed - (shock_speed - u0) * rho0 / density
    return density, pressure, velocity


def test_post_shock_matches_rankine_hugoniot_mach_10():
    exact = _load_case_module("exact")
    assert exact.MACH == 10.0
    assert exact.GAMMA == 1.4
    assert exact.RHO_PRE == 1.4
    assert exact.U_PRE == 0.0
    assert exact.P_PRE == 1.0
    rho_ref, p_ref, u_ref = _independent_rankine_hugoniot(10.0, 1.4, 1.4, 0.0, 1.0)
    jump = exact.rankine_hugoniot(mach=10.0, gamma=1.4, rho0=1.4, u0=0.0, p0=1.0)
    np.testing.assert_allclose(jump["rho"], rho_ref)
    np.testing.assert_allclose(jump["p"], p_ref)
    np.testing.assert_allclose(jump["u"], u_ref)
    post = exact.post_shock_state()
    np.testing.assert_allclose(post["rho"], rho_ref)
    np.testing.assert_allclose(post["p"], p_ref)
    np.testing.assert_allclose(np.hypot(post["u"], post["v"]), u_ref)


def test_shock_angle_is_30_degrees():
    exact = _load_case_module("exact")
    assert exact.SHOCK_ANGLE_DEG == 30.0
    assert exact.WEDGE_ANGLE_DEG == 30.0
    wall = 90.0 - float(exact.SHOCK_ANGLE_DEG)
    np.testing.assert_allclose(exact.SHOCK_WALL_ANGLE_DEG, wall)
    heights = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    expected = exact.X_FOOT + heights / np.tan(np.deg2rad(wall))
    np.testing.assert_allclose(exact.shock_front_x(heights), expected)
    np.testing.assert_allclose(exact.X_FOOT, 1.0 / 6.0)
    np.testing.assert_allclose(exact.T_END, 0.2)


def test_density_and_pressure_are_positive():
    exact = _load_case_module("exact")
    x, y = _cell_mesh(exact)
    primitives = exact.primitives(x, y, 0.0)
    assert np.all(np.asarray(primitives["rho"]) > 0.0)
    assert np.all(np.asarray(primitives["p"]) > 0.0)
    pre = exact.pre_shock_state()
    post = exact.post_shock_state()
    assert pre["rho"] > 0.0 and pre["p"] > 0.0
    assert post["rho"] > 0.0 and post["p"] > 0.0


def test_write_rb08_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_rb08_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert "DMR morphology / no closed form" in loaded["not_applicable_reason"]["orders"]


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


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
