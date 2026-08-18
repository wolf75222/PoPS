"""RED contract: private NativeSeries factory, digest evidence, cell averages."""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.capabilities import authenticate_installed_artifact
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.metrics import collect_metrics
from verification.pops_verify import native_evidence as ne
from verification.pops_verify.native_evidence import (
    BLOCKED_REQUIRED,
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


def _write_leaf(tmp_path: Path, *, dimension: int = 1) -> Path:
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    root = tmp_path / "native"
    leaf = root / f"dim{dimension}" / f"_pops{suffix}"
    leaf.parent.mkdir(parents=True, exist_ok=True)
    payload = b"fake-exact-rank-leaf-smooth"
    leaf.write_bytes(payload)
    row = {
        "dimension": dimension,
        "path": f"dim{dimension}/_pops{suffix}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "version": "1.0.0",
        "abi_key": "abi-test",
        "has_mpi": False,
        "has_kokkos": True,
    }
    (root / "variants.json").write_text(
        json.dumps({"schema_version": 1, "variants": [row]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def _identity(tmp_path: Path, *, dimension: int = 1):
    return authenticate_installed_artifact(
        dimension=dimension,
        variants_root=_write_leaf(tmp_path, dimension=dimension),
        doctor_ok=False,
    )


def _request(case_id="TR-02", dim=1, n=16):
    return CampaignRequest.from_job(
        CampaignJob(case_id=case_id, pops_native_dim=dim, min_resolution=n)
    )


def _authenticated_record(
    tmp_path: Path,
    *,
    case_id="TR-02",
    result=None,
    oracle=None,
    n_cells=16,
    spacing=None,
    pair=None,
    t_end=1.0,
    time_program="SSPRK2",
    cfl=0.4,
    dimension=1,
):
    field = np.ones(n_cells, dtype=np.float64) if result is None else np.asarray(result)
    ref = np.zeros_like(field) if oracle is None else np.asarray(oracle)
    return maybe_campaign_payload(
        _request(case_id, dimension, n_cells),
        field,
        identity=_identity(tmp_path, dimension=dimension),
        case_id=case_id,
        oracle=ref,
        program_digest="a" * 64,
        resolved_case_digest="b" * 64,
        n_cells=n_cells,
        t_end=t_end,
        time_program=time_program,
        cfl=cfl,
        dimension=dimension,
        sample_spacing=spacing if spacing is not None else 1.0 / float(n_cells),
        pair=pair,
    )


def test_record_schema_documents_immutable_fields():
    required = {
        "case_id",
        "result_digest",
        "evidence_digest",
        "sample_spacing",
        "leaf_sha256",
        "native_header_signature",
        "native_variant_manifest_digest",
        "component_catalog_digest",
        "program_digest",
        "resolved_case_digest",
    }
    schema = getattr(ne, "RECORD_SCHEMA", ())
    assert required <= set(schema)
    assert "native_seal" not in schema
    assert "compile_identity" not in schema


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


def test_direct_nativeseries_constructor_cannot_pass():
    with pytest.raises(NativeSeriesError, match="cannot be constructed"):
        NativeSeries(case_id="TR-02", records=())
    assert not hasattr(NativeSeries, "from_campaign_runs")


def test_object_artifact_simulation_cannot_seal(tmp_path: Path):
    with pytest.raises(NativeSeriesError, match="artifact|identity|Authenticated"):
        maybe_campaign_payload(
            _request(),
            np.ones(16),
            artifact=object(),
            simulation=object(),
            n_cells=16,
            t_end=1.0,
            time_program="SSPRK2",
            cfl=0.4,
        )


def test_mutated_result_fails_revalidation(tmp_path: Path):
    record = _authenticated_record(tmp_path)
    record["result"] = np.asarray(record["result"]) * 2.0
    with pytest.raises(NativeSeriesError, match="digest|result"):
        ne.verify_campaign_run(record)
    summary = report_from_native_series(
        "TR-02",
        [record],
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_reused_evidence_digest_cannot_pass(tmp_path: Path):
    first = _authenticated_record(tmp_path, result=np.ones(16), oracle=np.zeros(16))
    second = _authenticated_record(
        tmp_path / "b", result=np.full(16, 2.0), oracle=np.zeros(16)
    )
    second["evidence_digest"] = first["evidence_digest"]
    with pytest.raises(NativeSeriesError):
        ne.verify_campaign_run(second)
    summary = report_from_native_series(
        "TR-02",
        [second],
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_caller_trusted_error_fn_is_not_a_public_api():
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    ).read_text(encoding="utf-8")
    assert "error_fn" not in source
    assert "_SEAL_KEY" not in source
    assert "_REGISTERED_BINDS" not in source
    assert "hmac" not in source


def test_cross_case_record_reuse_cannot_pass(tmp_path: Path):
    record = _authenticated_record(tmp_path, case_id="TR-02")
    summary = report_from_native_series(
        "EU-01",
        [record],
        native_dimensions=[1],
        components=["euler"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    assert summary["coverage"]["cases_failed"] == 1


def test_below_threshold_typed_series_cannot_pass(tmp_path: Path):
    records = []
    for n_cells, shift in ((16, 0.08), (32, 0.03), (64, 0.011)):
        oracle = np.zeros(n_cells)
        result = np.full(n_cells, shift)
        records.append(
            _authenticated_record(
                tmp_path / str(n_cells),
                result=result,
                oracle=oracle,
                n_cells=n_cells,
                spacing=1.0 / n_cells,
            )
        )
    summary = report_from_native_series(
        "TR-02",
        records,
        native_dimensions=[1],
        components=["transport"],
        threshold=1.8,
    )
    assert summary["coverage"]["cases_passed"] == 0
    reasons = " ".join(item["reason"] for item in summary["failures"])
    assert "threshold" in reasons.lower()


def test_persisted_record_revalidates_after_module_reload(tmp_path: Path):
    record = _authenticated_record(tmp_path)
    payload = {
        key: record[key]
        for key in getattr(ne, "RECORD_SCHEMA", ())
        if key in record and key not in {"result", "oracle", "pair_result"}
    }
    payload["result"] = np.asarray(record["result"]).tolist()
    payload["oracle"] = np.asarray(record["oracle"]).tolist()
    blob = tmp_path / "record.json"
    blob.write_text(json.dumps(payload), encoding="utf-8")
    restored = json.loads(blob.read_text(encoding="utf-8"))
    restored["result"] = np.asarray(restored["result"], dtype=np.float64)
    restored["oracle"] = np.asarray(restored["oracle"], dtype=np.float64)
    ne.verify_campaign_run(restored)
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    ).read_text(encoding="utf-8")
    assert "_REGISTERED_BINDS" not in source
    assert "hmac" not in source


def test_tr06_pair_requires_both_authenticated_results(tmp_path: Path):
    original = np.ones((8, 8))
    permuted = np.full((8, 8), 2.0)
    identity = _identity(tmp_path, dimension=2)
    incomplete = maybe_campaign_payload(
        _request("TR-06", 2, 8),
        original,
        identity=identity,
        case_id="TR-06",
        oracle=np.zeros((8, 8)),
        program_digest="a" * 64,
        resolved_case_digest="b" * 64,
        n_cells=8,
        t_end=1.0,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=2,
        sample_spacing=1.0 / 8.0,
    )
    with pytest.raises(NativeSeriesError, match="pair"):
        ne.verify_campaign_run(incomplete)
    paired = maybe_campaign_payload(
        _request("TR-06", 2, 8),
        original,
        identity=identity,
        case_id="TR-06",
        oracle=np.zeros((8, 8)),
        program_digest="a" * 64,
        resolved_case_digest="b" * 64,
        n_cells=8,
        t_end=1.0,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=2,
        sample_spacing=1.0 / 8.0,
        pair={
            "result": permuted,
            "identity": identity,
            "program_digest": "c" * 64,
            "resolved_case_digest": "d" * 64,
        },
    )
    ne.verify_campaign_run(paired)
    assert paired["pair_result_digest"] != paired["result_digest"]
    assert paired["pair_program_digest"] == "c" * 64


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


def test_reduced_writers_are_required_fail(tmp_path: Path):
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
        assert loaded["coverage"]["cases_failed"] == 1
        assert loaded["coverage"]["cases_not_supported"] == 0
        assert case_id in getattr(ne, "REDUCED_REQUIRED", {})


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
        with pytest.raises(
            run.NativeUnavailable, match=getattr(ne, "REDUCED_REQUIRED")[case_id]
        ):
            run.run_native(request=request)


def test_allow_empty_orders_is_removed():
    assert "allow_empty_orders" not in inspect.signature(report_from_native_series).parameters
    for family, slug in (
        ("euler", "isentropic_vortex"),
        ("time", "pure_temporal"),
    ):
        analyze = _load(REPO_ROOT / "verification" / "cases" / family / slug, "analyze")
        assert not hasattr(analyze, "analyze_series")


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
    assert run.campaign_run_fields.__module__ == "verification.pops_verify.native_evidence"
    fields = campaign_run_fields(
        request=_request("EU-01"),
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


def test_typed_metrics_and_provenance_are_schema_valid(tmp_path: Path):
    records = [
        _authenticated_record(
            tmp_path,
            result=np.full(16, 0.01),
            oracle=np.zeros(16),
        )
    ]
    metrics = collect_metrics("TR-02", reason="no authenticated native series")
    Draft202012Validator(json.loads(SCHEMA_METRICS.read_text())).validate(metrics)
    typed_metrics = metrics_from_native_series("TR-02", records)
    Draft202012Validator(json.loads(SCHEMA_METRICS.read_text())).validate(typed_metrics)
    provenance = provenance_from_native_series(
        "TR-02",
        records,
        component_catalog_digest="0" * 64,
        native_header_signature="1" * 64,
        native_variant_manifest_digest="2" * 64,
    )
    Draft202012Validator(json.loads(SCHEMA_PROVENANCE.read_text())).validate(provenance)


def _assert_1d_averages(run_fn, exact_fn, n_cells=32, length=1.0):
    width = length / n_cells
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    averages = analytic_cell_averages(exact_fn, lo, hi)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    points = exact_fn(centers)
    np.testing.assert_allclose(np.ravel(run_fn), np.ravel(averages))
    return not np.allclose(averages, points)


def test_tr02_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse", "exact")
    assert _assert_1d_averages(
        run._initial_field(32), lambda x: exact.exact_gaussian(x, 0.0)
    )


def test_po01_rhs_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig", "exact")
    assert _assert_1d_averages(run.build_rhs_and_oracle(32)["rhs"], exact.rhs_exact)


def test_tm01_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal", "exact")
    field = run._initial_field(64) if hasattr(run, "_initial_field") else None
    if field is None:
        pytest.fail("TM-01 run_native still point-samples exact_sine")
    _assert_1d_averages(field, lambda x: exact.exact_sine(x, 0.0), n_cells=64)


def test_tr06_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "transport" / "axis_permutation", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "transport" / "axis_permutation", "exact")
    n_cells = 16
    width = float(exact.PERIOD) / n_cells
    axis_lo = np.arange(n_cells, dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing="ij")
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing="ij")
    lo = np.stack((x_lo, y_lo), axis=-1)
    hi = np.stack((x_hi, y_hi), axis=-1)
    averages = analytic_cell_averages(lambda xx, yy: exact.exact_product(xx, yy, 0.0), lo, hi)
    x, y, _volumes, _axis = exact.uniform_grid_2d(n_cells)
    points = exact.exact_product(x, y, 0.0)
    assert not np.allclose(averages, points)
    np.testing.assert_allclose(run._initial_field(n_cells), averages)


def test_tr07_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "transport" / "discontinuous_slot", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "transport" / "discontinuous_slot", "exact")
    n_cells = 15
    centers, _ = exact.cell_centers(n_cells)
    width = float(centers[1] - centers[0])
    lo = centers - 0.5 * width
    hi = centers + 0.5 * width
    averages = analytic_cell_averages(lambda x: exact.exact_slot(x, 0.0), lo, hi)
    points = exact.exact_slot(centers, 0.0)
    assert not np.allclose(averages, points)
    np.testing.assert_allclose(np.ravel(run._initial_field(n_cells)), np.ravel(averages))


def test_eu01_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "euler" / "linear_waves", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "euler" / "linear_waves", "exact")
    n_cells = 32
    width = 1.0 / n_cells
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    def rho_exact(x):
        samples = np.asarray(x, dtype=np.float64)
        return exact.exact_mode(samples.reshape(-1), 0.0, mode="entropy")[0].reshape(
            samples.shape
        )

    averages = analytic_cell_averages(rho_exact, lo, hi)
    np.testing.assert_allclose(run.initial_primitives(n_cells)[0], averages)


def test_eu02_analysis_uses_cell_average_oracle():
    analyze = _load(REPO_ROOT / "verification" / "cases" / "euler" / "isentropic_vortex", "analyze")
    source = inspect.getsource(analyze.density_error)
    assert "analytic_cell_averages" in source


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


def test_eu04_initial_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "euler" / "standing_acoustic", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "euler" / "standing_acoustic", "exact")
    n_cells = 32
    width = 1.0 / n_cells
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    def rho_exact(x):
        samples = np.asarray(x, dtype=np.float64)
        return np.asarray(exact.primitives_1d(samples.reshape(-1), 0.0)[0]).reshape(
            samples.shape
        )

    averages = analytic_cell_averages(rho_exact, lo, hi)
    np.testing.assert_allclose(run.initial_primitives(n_cells)[0], averages)


def test_po03_rhs_uses_cell_averages_not_point_samples():
    run = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "neumann_nullspace", "run")
    exact = _load(REPO_ROOT / "verification" / "cases" / "poisson" / "neumann_nullspace", "exact")
    assert _assert_1d_averages(run.build_rhs_and_oracle(32)["rhs"], exact.rhs_exact)


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
