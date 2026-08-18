"""EU-04 standing acoustic wave (in-memory oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.phase import phase_error
from verification.pops_verify.report import ARTIFACTS
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "standing_acoustic"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    volumes = np.full(n_cells, width, dtype=np.float64)
    return centers, volumes


def test_velocity_vanishes_at_reflecting_walls():
    exact = _load_case_module("exact")
    walls = np.array([0.0, 1.0], dtype=np.float64)
    for time in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        primitives = exact.primitives_1d(walls, time)
        np.testing.assert_allclose(primitives[1], 0.0, rtol=0.0, atol=0.0)


def test_acoustic_energy_has_period_two_over_c():
    exact = _load_case_module("exact")
    centers, volumes = _cell_centers()
    background = exact.background()
    speed = exact.acoustic_speed(background)
    period = 2.0 / speed
    np.testing.assert_allclose(exact.period(), period, rtol=0.0, atol=0.0)
    for time in (0.0, 0.13, 0.37, 0.91):
        energy = exact.acoustic_energy_density(centers, time)
        shifted = exact.acoustic_energy_density(centers, time + period)
        np.testing.assert_allclose(shifted, energy, rtol=0.0, atol=1.0e-14)
        total = exact.total_acoustic_energy(centers, volumes, time)
        total_shifted = exact.total_acoustic_energy(centers, volumes, time + period)
        np.testing.assert_allclose(total_shifted, total, rtol=0.0, atol=1.0e-14)
        state = exact.primitives_1d(centers, time)
        state_shifted = exact.primitives_1d(centers, time + period)
        np.testing.assert_allclose(state_shifted, state, rtol=0.0, atol=1.0e-14)


def test_density_and_pressure_are_in_phase():
    exact = _load_case_module("exact")
    centers, _ = _cell_centers()
    background = exact.background()
    speed = exact.acoustic_speed(background)
    for time in (0.0, 0.2, 0.7):
        density, _, pressure = exact.primitives_1d(centers, time)
        density_pert = density - background["rho"]
        pressure_pert = pressure - background["p"]
        np.testing.assert_allclose(
            pressure_pert, (speed * speed) * density_pert, rtol=0.0, atol=1.0e-14
        )
        if np.max(np.abs(density_pert)) > 0.0:
            np.testing.assert_allclose(
                phase_error(density_pert, pressure_pert), 0.0, atol=1.0e-12
            )
    times = np.linspace(0.0, 4.0 * exact.period(), 256, endpoint=False)
    density_t = exact.primitives_1d(0.0, times)[0] - background["rho"]
    pressure_t = exact.primitives_1d(0.0, times)[2] - background["p"]
    np.testing.assert_allclose(phase_error(density_t, pressure_t), 0.0, atol=1.0e-12)


def test_write_eu04_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_eu04_report(tmp_path)
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


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py",):
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

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_eu04_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 1
    expected = [float(value) for value in observed_order(linf, spacings)]
    observed = [row["observed_order"] for row in loaded["orders"]]
    np.testing.assert_allclose(observed, expected)
    assert not np.allclose(observed, np.full(len(observed), 2.0))


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="EU-04", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
