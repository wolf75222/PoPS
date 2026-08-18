"""RED contract: typed NativeSeries, blocked/reduced writers, TM-01, authoring."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.metrics import collect_metrics
from verification.pops_verify.native_evidence import (
    BLOCKED_REQUIRED,
    REDUCED_NOT_SUPPORTED,
    NativeSeries,
    NativeSeriesError,
    campaign_run_fields,
    maybe_campaign_payload,
    metrics_from_native_series,
    provenance_from_native_series,
    report_from_native_series,
)
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_METRICS = REPO_ROOT / "schemas" / "verification_metrics.v1.json"
SCHEMA_PROVENANCE = REPO_ROOT / "schemas" / "verification_provenance.v1.json"
SCHEMA_REPORT = REPO_ROOT / "schemas" / "verification_report.v1.json"


def _load(case_dir: Path, name: str):
    path = case_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{case_dir.name}_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report_validator():
    schema = json.loads(SCHEMA_REPORT.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_arbitrary_dict_series_cannot_pass():
    summary = report_from_native_series(
        "TR-02",
        {"linf": [0.08, 0.03, 0.011], "spacings": [1 / 16, 1 / 32, 1 / 64]},
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    assert summary["coverage"]["cases_failed"] == 1


def test_empty_mapping_cannot_pass():
    summary = report_from_native_series(
        "EU-01",
        {},
        native_dimensions=[1],
        components=["euler"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_injected_linf_spacings_rejected_by_native_series():
    with pytest.raises(NativeSeriesError):
        NativeSeries.from_campaign_runs(
            "TR-02",
            [{"linf": [0.01], "spacings": [0.1], "result": [1.0]}],
        )


def test_unregistered_compile_identity_cannot_authenticate():
    fields = {key: 0 for key in RUN_FIELDS}
    fields.update(
        {
            "compiler": "c++",
            "build_type": "unknown",
            "precision": "float64",
            "kokkos_execution_space": "KokkosSerial",
            "mpi_enabled": False,
            "mpi_library": "none",
            "mpi_thread_level_requested": "none",
            "mpi_thread_level_provided": "none",
            "hdf5_collective_enabled": False,
            "mpi_ranks": 1,
            "omp_threads_per_rank": 1,
            "gpus": 0,
            "resolution": [16],
            "block_size": [16],
            "amr_total_levels": 1,
            "refinement_ratio": 2,
            "subcycling": False,
            "time_program": "SSPRK2",
            "cfl": 0.4,
            "final_time": 1.0,
            "result": np.ones(16),
            "compile_identity": "pops.compile:forged",
            "bind_identity": "pops.bind:forged",
        }
    )
    with pytest.raises(NativeSeriesError):
        NativeSeries.from_campaign_runs("TR-02", [fields])


def test_blocked_writers_reject_every_series(tmp_path: Path):
    cases = (
        ("transport", "single_vortex", "write_tr03_report", "TR-03"),
        ("euler", "mms", "write_eu03_report", "EU-03"),
        ("poisson", "dirichlet_mms", "write_po02_report", "PO-02"),
    )
    for family, slug, writer, case_id in cases:
        analyze = _load(REPO_ROOT / "verification" / "cases" / family / slug, "analyze")
        written = getattr(analyze, writer)(
            tmp_path / case_id,
            native_series={"linf": [1e-12], "spacings": [0.1, 0.05]},
        )
        assert written
        loaded = json.loads((tmp_path / case_id / "summary.json").read_text(encoding="utf-8"))
        _report_validator().validate(loaded)
        assert loaded["coverage"]["cases_passed"] == 0
        assert loaded["coverage"]["cases_failed"] == 1
        assert case_id in BLOCKED_REQUIRED


def test_blocked_run_native_raises_exact_reason():
    cases = (
        ("transport", "single_vortex", "TR-03", 2),
        ("euler", "mms", "EU-03", 1),
        ("poisson", "dirichlet_mms", "PO-02", 2),
    )
    for family, slug, case_id, dim in cases:
        run = _load(REPO_ROOT / "verification" / "cases" / family / slug, "run")
        request = CampaignRequest.from_job(
            CampaignJob(case_id=case_id, pops_native_dim=dim, min_resolution=16)
        )
        with pytest.raises(run.NativeUnavailable, match=BLOCKED_REQUIRED[case_id]):
            run.run_native(request=request)


def test_reduced_writers_cannot_pass_normative_ids(tmp_path: Path):
    cases = (
        ("poisson", "huang_greengard", "write_po04_report", "PO-04"),
        ("poisson", "fft_vs_gmg", "write_po05_report", "PO-05"),
        ("poisson", "elliptic_tolerance", "write_po07_report", "PO-07"),
    )
    for family, slug, writer, case_id in cases:
        analyze = _load(REPO_ROOT / "verification" / "cases" / family / slug, "analyze")
        getattr(analyze, writer)(tmp_path / case_id, native_series={"linf": [1e-9], "spacings": [0.1]})
        loaded = json.loads((tmp_path / case_id / "summary.json").read_text(encoding="utf-8"))
        _report_validator().validate(loaded)
        assert loaded["coverage"]["cases_passed"] == 0
        assert loaded["coverage"]["cases_not_supported"] == 1
        assert case_id in REDUCED_NOT_SUPPORTED


def test_reduced_run_native_raises_missing_requirement():
    cases = (
        ("poisson", "huang_greengard", "PO-04", 1),
        ("poisson", "fft_vs_gmg", "PO-05", 1),
        ("poisson", "elliptic_tolerance", "PO-07", 1),
    )
    for family, slug, case_id, dim in cases:
        run = _load(REPO_ROOT / "verification" / "cases" / family / slug, "run")
        request = CampaignRequest.from_job(
            CampaignJob(case_id=case_id, pops_native_dim=dim, min_resolution=16)
        )
        with pytest.raises(run.NativeUnavailable, match=REDUCED_NOT_SUPPORTED[case_id]):
            run.run_native(request=request)


def test_po_authoring_is_unified_not_split():
    for slug in ("neumann_nullspace", "huang_greengard", "fft_vs_gmg", "elliptic_tolerance"):
        run = _load(REPO_ROOT / "verification" / "cases" / "poisson" / slug, "run")
        case = run.build_case(16)
        plan = run.resolve_plan(16)
        assert plan is not None
        assert case._time is not None
        source = (REPO_ROOT / "verification" / "cases" / "poisson" / slug / "run.py").read_text()
        assert "author_periodic_poisson" in source
        tree = ast.parse(source)
        field_before_program = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_case":
                text = ast.get_source_segment(source, node) or ""
                assert "author_periodic_poisson" in text
                field_before_program = "Case(" in text and "author_periodic_poisson" not in text
        assert field_before_program is False


def test_elliptic_helper_does_not_import_tests_support():
    text = (
        REPO_ROOT / "verification" / "pops_verify" / "elliptic_stationary.py"
    ).read_text(encoding="utf-8")
    assert "tests.python.support" not in text
    assert "def attach_stationary_program" not in text


def test_tm01_refuses_grid_override_from_min_resolution():
    run = _load(REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal", "run")
    request = CampaignRequest.from_job(
        CampaignJob(case_id="TM-01", pops_native_dim=1, min_resolution=16)
    )
    with pytest.raises(run.NativeUnavailable, match="fixed N=64"):
        run.run_native(request=request)


def test_tm01_min_resolution_matching_fine_grid_does_not_change_n():
    run = _load(REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal", "run")
    assert run.N_CELLS == 64
    request = CampaignRequest.from_job(
        CampaignJob(case_id="TM-01", pops_native_dim=1, min_resolution=64)
    )
    try:
        result = run.run_native(dt=run.DT, t_end=0.01, request=request)
    except run.NativeUnavailable as exc:
        assert "fixed N=64" not in str(exc)
        return
    assert result["resolution"] == [64]


def test_eu01_uses_shared_campaign_helper_not_hardcoded_facts():
    source = (
        REPO_ROOT / "verification" / "cases" / "euler" / "linear_waves" / "run.py"
    ).read_text(encoding="utf-8")
    assert "build_type\": \"native-dsl\"" not in source
    assert "mpi_enabled\": False" not in source
    run = _load(REPO_ROOT / "verification" / "cases" / "euler" / "linear_waves", "run")
    assert "campaign_run_fields" not in dir(run) or run.campaign_run_fields is campaign_run_fields
    request = CampaignRequest.from_job(
        CampaignJob(case_id="EU-01", pops_native_dim=1, min_resolution=16)
    )
    fields = campaign_run_fields(
        request=request,
        n_cells=16,
        t_end=0.05,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
    assert fields["build_type"] in {"unknown", "Release", "Debug", "RelWithDebInfo"}
    missing = [key for key in RUN_FIELDS if key not in fields]
    assert missing == []


def test_campaign_run_fields_do_not_invent_mpi_thread_level():
    request = CampaignRequest.from_job(
        CampaignJob(case_id="TR-02", pops_native_dim=1, min_resolution=16, mpi_mode="on")
    )
    fields = campaign_run_fields(
        request=request,
        n_cells=16,
        t_end=1.0,
        time_program="SSPRK2",
        cfl=0.4,
    )
    assert fields["mpi_enabled"] is True
    assert fields["mpi_thread_level_requested"] == "unknown"
    assert fields["mpi_library"] == "unknown"


def test_metrics_and_provenance_from_untyped_dict_fail_closed():
    with pytest.raises(NativeSeriesError):
        metrics_from_native_series("TR-02", {"linf": [0.1]})
    with pytest.raises(NativeSeriesError):
        provenance_from_native_series("TR-02", {"result": [1.0]})


_SCHEMA_DIGESTS = {
    "component_catalog_digest": "0" * 64,
    "native_header_signature": "1" * 64,
    "native_variant_manifest_digest": "2" * 64,
}


def test_fail_closed_metrics_and_provenance_are_schema_valid():
    metrics = collect_metrics("TR-02", reason="no authenticated native series")
    Draft202012Validator(json.loads(SCHEMA_METRICS.read_text())).validate(metrics)
    request = CampaignRequest.from_job(
        CampaignJob(case_id="TR-02", pops_native_dim=1, min_resolution=16)
    )
    artifact, simulation = object(), object()
    record = maybe_campaign_payload(
        request,
        np.ones(16),
        artifact=artifact,
        simulation=simulation,
        n_cells=16,
        t_end=1.0,
        time_program="SSPRK2",
        cfl=0.4,
    )
    series = NativeSeries.from_campaign_runs(
        "TR-02",
        [record],
        error_fn=lambda run: 0.01,
        spacing_fn=lambda run: 1.0 / 16.0,
    )
    typed_metrics = metrics_from_native_series("TR-02", series)
    Draft202012Validator(json.loads(SCHEMA_METRICS.read_text())).validate(typed_metrics)
    provenance = provenance_from_native_series("TR-02", series, **_SCHEMA_DIGESTS)
    Draft202012Validator(json.loads(SCHEMA_PROVENANCE.read_text())).validate(provenance)


def test_tr02_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse", "exact")
    n_cells = 32
    width = 1.0 / n_cells
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    averages = analytic_cell_averages(lambda x: exact.exact_gaussian(x, 0.0), lo, hi)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    points = exact.exact_gaussian(centers, 0.0)
    assert not np.allclose(averages, points)
    field = run._initial_field(n_cells)
    np.testing.assert_allclose(np.ravel(field), np.ravel(averages))


def test_po01_rhs_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig", "exact")
    n_cells = 32
    width = 1.0 / n_cells
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    averages = analytic_cell_averages(exact.rhs_exact, lo, hi)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    assert not np.allclose(averages, exact.rhs_exact(centers))
    sample = run.build_rhs_and_oracle(n_cells)
    np.testing.assert_allclose(sample["rhs"], averages)


def test_eu02_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "euler" / "isentropic_vortex", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "euler" / "isentropic_vortex", "exact")
    n_cells = 16
    length = float(exact.PERIOD)
    width = length / n_cells
    axis_lo = np.arange(n_cells, dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing="xy")
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing="xy")
    lo = np.stack((x_lo, y_lo), axis=-1)
    hi = np.stack((x_hi, y_hi), axis=-1)

    def density(x, y):
        return exact.exact_vortex(x, y, 0.0, u_inf=1.0, v_inf=0.0)["rho"]

    averages = analytic_cell_averages(density, lo, hi)
    x, y, _ = run.cell_centers(n_cells)
    points = exact.exact_vortex(x, y, 0.0, u_inf=1.0, v_inf=0.0)["rho"]
    assert not np.allclose(averages, points)
    conserved = run.initial_conserved(n_cells)
    np.testing.assert_allclose(conserved["rho"], averages)


def test_manufactured_solve_is_quarantined_utility():
    source = (
        REPO_ROOT / "verification" / "cases" / "poisson" / "elliptic_tolerance" / "run.py"
    ).read_text(encoding="utf-8")
    analyze = (
        REPO_ROOT / "verification" / "cases" / "poisson" / "elliptic_tolerance" / "analyze.py"
    ).read_text(encoding="utf-8")
    assert "manufactured_solve" not in analyze
    assert "write_po07_report" in analyze
    run = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "elliptic_tolerance", "run")
    assert not hasattr(run, "manufactured_solve") or inspect.isfunction(run.manufactured_solve)


def test_threshold_below_min_cannot_pass_even_with_typed_empty_series():
    summary = report_from_native_series(
        "TR-02",
        None,
        native_dimensions=[1],
        components=["transport"],
        threshold=1.8,
    )
    assert summary["coverage"]["cases_passed"] == 0
