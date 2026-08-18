"""EU-02 isentropic vortex (2-d oracle; 1-d not applicable)."""
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
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "isentropic_vortex"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_mesh(exact, n_cells: int = N_CELLS):
    length = float(exact.PERIOD)
    width = length / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y


def test_density_and_pressure_are_positive():
    exact = _load("exact")
    x, y = _cell_mesh(exact)
    state = exact.exact_vortex(x, y, 0.0, u_inf=1.0, v_inf=0.0)
    assert np.all(np.asarray(state["rho"]) > 0.0)
    assert np.all(np.asarray(state["p"]) > 0.0)


def test_translation_by_one_zero_over_dt():
    exact = _load("exact")
    x, y = _cell_mesh(exact)
    dt = 0.5
    evolved = exact.exact_vortex(x, y, dt, u_inf=1.0, v_inf=0.0)
    shifted = exact.exact_vortex(x - dt, y, 0.0, u_inf=1.0, v_inf=0.0)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_allclose(
            evolved[key], shifted[key], rtol=0.0, atol=1.0e-14
        )


def test_entropy_function_is_constant_on_vortex():
    exact = _load("exact")
    x, y = _cell_mesh(exact)
    state = exact.exact_vortex(x, y, 0.25, u_inf=1.0, v_inf=0.0)
    entropy = exact.entropy_function(state["rho"], state["p"])
    background = exact.background()
    expected = exact.entropy_function(background["rho"], background["p"])
    np.testing.assert_allclose(entropy, expected, rtol=0.0, atol=1.0e-12)


def test_write_eu02_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load("analyze")
    written = analyze.write_eu02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    reasons = " ".join(item["reason"] for item in loaded["failures"]).lower()
    assert "native" in reasons


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
            assert "bind_public" in text
        else:
            assert "pops.run" not in text
        assert "from exact import" not in text


def test_pack_unpack_conserved_round_trip():
    run = _load("run")
    packed = run.pack_conserved(run.initial_conserved(N_CELLS))
    assert packed.shape == (4, N_CELLS, N_CELLS)
    assert packed.flags["C_CONTIGUOUS"]
    named = run.unpack_conserved(packed, N_CELLS)
    assert tuple(named) == run.COMPONENT_ORDER
    for name in run.COMPONENT_ORDER:
        np.testing.assert_array_equal(named[name], packed[run.COMPONENT_ORDER.index(name)])
    primitives = run.conserved_to_primitives(packed)
    assert np.all(primitives["rho"] > 0.0)
    assert np.all(primitives["p"] > 0.0)


def test_analyze_series_pass_shape_is_removed():
    analyze = _load("analyze")
    assert not hasattr(analyze, "analyze_series")


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load("run")
    missing = missing_compiler_requirement()
    try:
        conserved = run.run_native(16, t_end=0.05)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert set(conserved) == set(run.COMPONENT_ORDER)
    for field in conserved.values():
        array = np.asarray(field, dtype=np.float64)
        assert array.shape == (16, 16)
        assert np.isfinite(array).all()
    primitives = run.conserved_to_primitives(conserved)
    assert np.all(primitives["rho"] > 0.0)
    assert np.all(primitives["p"] > 0.0)


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_eu02_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1 or loaded["coverage"]["cases_not_supported"] == 1


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="EU-02", pops_native_dim=2, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
    packed = np.asarray(result["result"], dtype=np.float64)
    assert packed.shape == (4, 16, 16)
    assert np.isfinite(packed).all()


def test_analytic_center_and_vorticity_peak():
    exact = _load("exact")
    x, y = _cell_mesh(exact, 48)
    center = exact.analytic_center(0.5, u_inf=1.0, v_inf=0.0)
    np.testing.assert_allclose(center[0], 5.5, atol=1.0e-12)
    np.testing.assert_allclose(center[1], 5.0, atol=1.0e-12)
    vorticity = exact.exact_vorticity(x, y, 0.5, u_inf=1.0, v_inf=0.0)
    index = np.unravel_index(int(np.argmax(vorticity)), vorticity.shape)
    assert abs(float(x[index]) - center[0]) < 0.25
    assert abs(float(y[index]) - center[1]) < 0.25


def test_initial_conserved_averages_energy_not_converted_primitives():
    run = _load("run")
    exact = _load("exact")
    n_cells = 16
    conserved = run.average_conserved(n_cells, 0.0)
    primitives = run.average_primitives(n_cells, 0.0)
    converted = run.primitives_to_conserved(primitives)
    np.testing.assert_allclose(conserved["rho"], primitives["rho"])
    assert not np.allclose(conserved["E"], converted["E"])
    x, y, _ = run.cell_centers(n_cells)
    points = exact.exact_vortex(x, y, 0.0)["rho"]
    assert not np.allclose(conserved["rho"], points)


def test_evaluate_order_claim_refuses_cfl_as_spatial():
    analyze = _load("analyze")
    run = _load("run")
    fields = {n: run.pack_conserved(run.initial_conserved(n)) for n in (16, 32, 64, 128)}
    with pytest.raises(analyze.NativeSeriesError, match="global"):
        analyze.evaluate_order_claim(
            {
                "source": "native",
                "family": "spatial",
                "dt_scaling": "cfl",
                "resolutions": (16, 32, 64, 128),
                "fields": fields,
                "t_end": 0.0,
            }
        )


def test_evaluate_order_claim_refuses_collapsed_spatial_mesh():
    analyze = _load("analyze")
    run = _load("run")
    field = run.pack_conserved(run.initial_conserved(16))
    with pytest.raises(analyze.NativeSeriesError, match="distinct meshes"):
        analyze.evaluate_order_claim(
            {
                "source": "native",
                "family": "spatial",
                "dt_scaling": "h2",
                "resolutions": (16, 16, 16, 16),
                "fields": {16: field},
                "t_end": 1.0,
            }
        )


def test_spatial_driver_request_uses_leaf_resolution():
    from verification.machines.run_eu02_v15 import _job
    from verification.pops_verify.campaign import CampaignRequest
    import dataclasses

    job = _job("KokkosSerial", "off", 16, resolutions=(16, 32, 64, 128))
    leaf = dataclasses.replace(job, min_resolution=128)
    request = CampaignRequest.from_job(leaf)
    assert request.min_resolution == 128


def test_evaluate_order_claim_temporal_uses_dts_not_collapsed_n():
    analyze = _load("analyze")
    run = _load("run")
    base = run.pack_conserved(run.initial_conserved(16))
    dts = (0.08, 0.04, 0.02, 0.01)
    amps = (8.0e-3, 2.0e-3, 5.0e-4, 1.25e-4)
    runs = []
    for dt, amp in zip(dts, amps, strict=True):
        field = np.array(base, copy=True)
        field[0] += amp
        runs.append({"n_cells": 16, "field": field, "dt": dt})
    claim = analyze.evaluate_order_claim(
        {
            "source": "native",
            "family": "temporal",
            "dt_scaling": "fixed",
            "resolutions": (16, 16, 16, 16),
            "fields": {16: runs[-1]["field"]},
            "runs": runs,
            "dts": dts,
            "t_end": 0.0,
            "spatial_linf": 1.0e-10,
        }
    )
    assert claim["family"] == "temporal"
    assert claim["verdict"] in {"pass", "fail"}
    assert len(claim["orders"]) == 3
    assert claim["spacings"] == dts
    assert len(set(claim["linf"])) == 4


def test_evaluate_order_claim_three_resolutions_is_smoke():
    analyze = _load("analyze")
    run = _load("run")
    fields = {n: run.pack_conserved(run.initial_conserved(n)) for n in (16, 32, 64)}
    claim = analyze.evaluate_order_claim(
        {
            "source": "native",
            "family": "global",
            "dt_scaling": "cfl",
            "resolutions": (16, 32, 64),
            "fields": fields,
            "t_end": 0.0,
        }
    )
    assert claim["verdict"] == "smoke"
    assert claim["order_pass"] is False
    assert claim["orders"] == []


def test_oracle_producer_matches_conserved_shape():
    from verification.pops_verify.oracle_producers import produce_oracle

    result = np.zeros((4, 8, 8), dtype=np.float64)
    oracle = produce_oracle(
        "EU-02",
        {"job": {"min_resolution": 8}},
        result,
        {"final_time": 0.0},
    )
    assert oracle.shape == (4, 8, 8)
    assert np.all(oracle[0] > 0.0)


def test_vortex_center_from_exact_density():
    analyze = _load("analyze")
    exact = _load("exact")
    run = _load("run")
    n_cells = 32
    x, y, _ = run.cell_centers(n_cells)
    t = 0.25
    density = exact.exact_vortex(x, y, t)["rho"]
    analytic = exact.analytic_center(t)
    numerical = analyze.vortex_center_from_density(density, n_cells, expected=analytic)
    error = analyze.center_error(numerical, analytic)
    assert error["distance"] < 0.2


def test_conservation_zero_for_identical_states():
    analyze = _load("analyze")
    run = _load("run")
    conserved = run.initial_conserved(16)
    integrals = analyze.conservation_integrals(conserved, 16)
    drifts = analyze.conservation_drifts(integrals, integrals, n_updates=1)
    assert drifts["ok"]["mass"]
    assert drifts["ok"]["energy"]
    assert drifts["relative"]["mass"] == 0.0


def test_acceptance_reconstruction_is_public_weno5z():
    run = _load("run")
    assert run.ACCEPTANCE_RECONSTRUCTION == "weno5z"
    assert run.reconstruction_role("weno5z") == "acceptance"
    assert run.reconstruction_role("vanleer") == "tvd_fail_control"
    authored = run._author(8)
    assert authored.reconstruction == "weno5z"
    vanleer = run._author(8, reconstruction="vanleer")
    assert vanleer.reconstruction == "vanleer"
    weno = run._reconstruction_brick("weno5z")
    vanleer = run._reconstruction_brick("vanleer")
    assert weno.scheme == "weno5" and weno.name == "weno5z"
    assert vanleer.scheme == "vanleer"


def test_plot_artifact_stem_suffixes_resolution_and_time():
    plot = load_sibling_module(CASE_DIR / "plot_eu02.py")
    fine = plot.artifact_stem("triptych_rho", 128, 1.0)
    coarse = plot.artifact_stem("triptych_rho", 64, 1.0)
    assert fine == "triptych_rho_n128_t1"
    assert coarse == "triptych_rho_n064_t1"
    assert fine != coarse
    assert plot.TRAJECTORY_EQUAL_ASPECT is False
    assert plot.TRAJECTORY_XLABEL == "t"


def test_report_lists_radial_anisotropy_and_xy_symmetry_separately(tmp_path: Path):
    analyze = _load("analyze")
    run = _load("run")
    fields = {}
    for n_cells, amp in zip((16, 32, 64, 128), (8.0e-3, 2.0e-3, 5.0e-4, 1.25e-4)):
        field = np.array(run.pack_conserved(run.initial_conserved(n_cells)), copy=True)
        field[0] += amp
        fields[n_cells] = field
    extras = analyze.diagnose_resolution(fields[16], 16, 0.0)
    assert "radial_anisotropy" in extras["symmetry"]
    assert "xy_symmetry" in extras["symmetry"]
    analyze.write_native_campaign_report(
        tmp_path,
        {
            "source": "native",
            "family": "global",
            "dt_scaling": "cfl",
            "resolutions": (16, 32, 64, 128),
            "fields": fields,
            "t_end": 0.0,
        },
        extras,
    )
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "radial_anisotropy" in text
    assert "xy_symmetry" in text


def test_finest_visual_job_dir_and_phase8_payloads(tmp_path: Path):
    analyze = _load("analyze")
    run = _load("run")
    series = tmp_path / "series"
    (series / "n016").mkdir(parents=True)
    (series / "n128").mkdir()
    (series / "series.json").write_text(
        json.dumps({"case_id": "EU-02", "jobs": ["n016", "n128"]}) + "\n",
        encoding="utf-8",
    )
    assert analyze.finest_visual_job_dir(series) == series / "n128"
    fields = {}
    for n_cells, amp in zip((16, 32, 64, 128), (8.0e-3, 2.0e-3, 5.0e-4, 1.25e-4)):
        field = np.array(run.pack_conserved(run.initial_conserved(n_cells)), copy=True)
        field[0] += amp
        fields[n_cells] = field
    report_dir = tmp_path / "report"
    analyze.write_native_campaign_report(
        report_dir,
        {
            "source": "native",
            "family": "global",
            "dt_scaling": "cfl",
            "resolutions": (16, 32, 64, 128),
            "fields": fields,
            "t_end": 0.0,
            "bundle_path": str(series),
        },
    )
    finest = series / "n128"
    assert (finest / "status.json").is_file()
    assert (finest / "program.json").is_file()
    report_figure = json.loads(
        (finest / "analysis" / "visual_data" / "report_figure.json").read_text(encoding="utf-8")
    )
    assert report_figure["panels"]
    for panel in report_figure["panels"]:
        assert "x" in panel and "y" in panel
    assert (report_dir / "REPORT.md").is_file()


def test_temporal_isolation_requires_spatial_linf_ten_times_below_coarsest_dt():
    analyze = _load("analyze")
    assert analyze.temporal_is_isolated(1.0e-5, 2.0e-4) is True
    assert analyze.temporal_is_isolated(1.0e-4, 2.0e-4) is False
    run = _load("run")
    base = run.pack_conserved(run.initial_conserved(16))
    dts = (0.08, 0.04, 0.02, 0.01)
    amps = (8.0e-3, 2.0e-3, 5.0e-4, 1.25e-4)
    runs = []
    for dt, amp in zip(dts, amps, strict=True):
        field = np.array(base, copy=True)
        field[0] += amp
        runs.append({"n_cells": 16, "field": field, "dt": dt})
    with pytest.raises(analyze.NativeSeriesError, match="isolat"):
        analyze.evaluate_order_claim(
            {
                "source": "native",
                "family": "temporal",
                "dt_scaling": "fixed",
                "resolutions": (16, 16, 16, 16),
                "fields": {16: runs[-1]["field"]},
                "runs": runs,
                "dts": dts,
                "t_end": 1.0,
                "spatial_linf": 0.2,
            }
        )


def test_compare_smokes_records_truthful_leaves_ranks_threads():
    analyze = _load("analyze")
    field = np.ones((4, 4, 4), dtype=np.float64)
    shifted = field + 1.0e-12
    report = analyze.compare_smokes(
        {
            "label": "serial",
            "field": field,
            "leaf": "serial-leaf",
            "ranks": 1,
            "threads": 1,
            "space": "KokkosSerial",
            "mpi_mode": "off",
        },
        {
            "label": "openmp",
            "field": field,
            "leaf": "openmp-leaf",
            "ranks": 1,
            "threads": 4,
            "space": "KokkosOpenMP",
            "mpi_mode": "off",
        },
        {
            "label": "mpi",
            "field": shifted,
            "leaf": "mpi-leaf",
            "ranks": 2,
            "threads": 1,
            "space": "KokkosSerial",
            "mpi_mode": "on",
        },
    )
    assert report["serial"]["ranks"] == 1
    assert report["openmp"]["threads"] == 4
    assert report["mpi"]["ranks"] == 2
    assert report["serial"]["leaf"] == "serial-leaf"
    assert report["mpi"]["leaf"] == "mpi-leaf"
    assert report["serial"]["leaf"] != report["openmp"]["leaf"]
    assert report["serial_vs_openmp"]["bit_identical"] is True
    assert report["serial_vs_mpi"]["bit_identical"] is False
    assert "linf" in report["serial_vs_mpi"]
    assert "l2" in report["serial_vs_openmp"]
