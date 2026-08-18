"""EvidenceBundle trust root: on-disk job directories, not in-memory hashes."""
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
from verification.pops_verify.capabilities import (
    AuthenticatedArtifact,
    authenticate_installed_artifact,
    sha256_file,
)
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError
from verification.pops_verify.evidence_contract import (
    EXTENSION_SLOTS,
    PARENT_INTEGRATION_PATCH,
    REQUIRED_JOB_FILES,
    emit_job_directory,
)
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
from verification.pops_verify.provenance import RUN_FIELDS, collect_provenance

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


def _request(case_id="TR-02", dim=1, n=16):
    return CampaignRequest.from_job(
        CampaignJob(case_id=case_id, pops_native_dim=dim, min_resolution=n)
    )


def _run_fields(*, n_cells=16, t_end=1.0, space="KokkosSerial", mpi_on=False, dim=1):
    return campaign_run_fields(
        request=CampaignRequest.from_job(
            CampaignJob(
                case_id="TR-02",
                pops_native_dim=dim,
                min_resolution=n_cells,
                execution_space=space,
                mpi_mode="on" if mpi_on else "off",
            )
        ),
        n_cells=n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=dim,
    )


def _tr02_oracle(n_cells: int, t_end: float = 1.0):
    exact = _load(REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse", "exact")
    width = 1.0 / int(n_cells)
    lo = np.arange(n_cells, dtype=np.float64) * width
    hi = lo + width
    return analytic_cell_averages(lambda x: exact.exact_gaussian(x, t_end), lo, hi)


def _emit_job(
    job_dir: Path,
    variants_root: Path,
    *,
    case_id="TR-02",
    n_cells=16,
    t_end=1.0,
    result=None,
    dimension=1,
    space="KokkosSerial",
    mpi_mode="off",
    repository_sha=None,
    pair_result=None,
    program_bytes=b"program-bytes",
    pair_program_bytes=b"pair-program-bytes",
):
    identity = authenticate_installed_artifact(
        dimension=dimension,
        variants_root=variants_root,
        doctor_ok=False,
    )
    field = (
        np.asarray(result, dtype=np.float64)
        if result is not None
        else np.ones(n_cells, dtype=np.float64)
    )
    resolved = {
        "case": {"id": case_id},
        "job": {
            "case_id": case_id,
            "pops_native_dim": dimension,
            "suite": "pr",
            "execution_space": space,
            "mpi_mode": mpi_mode,
            "min_resolution": n_cells,
            "evidence_status": "required",
            "resources": {
                "nodes": 1,
                "mpi_ranks": 1,
                "omp_threads": 1,
                "resolutions": [n_cells],
            },
        },
        "status": "pass",
        "reason": None,
    }
    fields = _run_fields(
        n_cells=n_cells,
        t_end=t_end,
        space=space,
        mpi_on=mpi_mode == "on",
        dim=dimension,
    )
    provenance = collect_provenance(
        case_id,
        pops_native_dim=dimension,
        dimension=dimension,
        nodes=1,
        pops_version="test",
        doctor_ok=False,
        component_catalog_digest=identity.component_catalog_digest,
        native_header_signature=identity.native_header_signature,
        native_variant_manifest_digest=identity.native_variant_manifest_digest,
        **fields,
    )
    if repository_sha is not None:
        provenance["repository_sha"] = repository_sha
    metrics = collect_metrics(case_id, reason="contract fixture")
    emit_job_directory(
        job_dir,
        resolved_case=resolved,
        provenance=provenance,
        metrics=metrics,
        result=field,
        program_bytes=program_bytes,
        native_artifact={
            "path": str(identity.path),
            "sha256": identity.sha256,
            "dimension": identity.dimension,
            "variants_root": str(variants_root),
        },
        pair_result=pair_result,
        pair_program_bytes=pair_program_bytes if pair_result is not None else None,
    )
    return job_dir


def _emit_series(
    tmp_path: Path,
    *,
    case_id="TR-02",
    shifts=None,
    n_cells_list=(16, 32, 64),
    t_end=1.0,
    variants_root=None,
):
    root = variants_root or _write_leaf(tmp_path)
    series_dir = tmp_path / "series"
    jobs = []
    for n_cells in n_cells_list:
        job_name = f"n{n_cells:03d}"
        jobs.append(job_name)
        if shifts is None:
            h = 1.0 / float(n_cells)
            result = _tr02_oracle(n_cells, t_end) + 0.25 * h * h
        else:
            result = _tr02_oracle(n_cells, t_end) + float(shifts[n_cells])
        _emit_job(
            series_dir / job_name,
            root,
            case_id=case_id,
            n_cells=n_cells,
            t_end=t_end,
            result=result,
        )
    (series_dir / "series.json").write_text(
        json.dumps({"case_id": case_id, "jobs": jobs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return series_dir, root


def test_record_schema_documents_immutable_fields_and_extension_slots():
    required = {
        "case_id",
        "result_digest",
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
    assert "coupling_digest" in schema
    assert "amr_mask_digest" in schema
    assert set(EXTENSION_SLOTS) >= {"coupling", "amr_mask"}


def test_contract_lists_required_job_files_and_parent_patch():
    for name in (
        "resolved_case.json",
        "resolved_case.sha256",
        "provenance.json",
        "metrics.json",
        "result.npy",
        "result.sha256",
        "program.bin",
        "program.sha256",
        "native_artifact.json",
        "producer_manifest.json",
        "producer_manifest.sha256",
    ):
        assert name in REQUIRED_JOB_FILES
    assert "result.npy" in PARENT_INTEGRATION_PATCH
    assert "program.bin" in PARENT_INTEGRATION_PATCH
    assert "native_artifact.json" in PARENT_INTEGRATION_PATCH
    assert "scripts/run_verification.py" in PARENT_INTEGRATION_PATCH


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


def test_in_memory_sequence_cannot_pass():
    summary = report_from_native_series(
        "TR-02",
        [{"result": np.ones(16), "result_digest": "a" * 64}],
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    assert summary["coverage"]["cases_failed"] == 1


def test_direct_nativeseries_constructor_cannot_pass():
    with pytest.raises(NativeSeriesError, match="cannot be constructed"):
        NativeSeries(case_id="TR-02", records=())
    assert not hasattr(NativeSeries, "from_campaign_runs")
    assert not hasattr(ne, "_series_from_records")


def test_object_artifact_is_program_bytes_blocker():
    with pytest.raises(NativeSeriesError, match="program bytes|so_path|CompiledSimulationArtifact"):
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
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    ).read_text(encoding="utf-8")
    assert "repr(artifact" not in source
    assert "_standin" not in source


def test_in_memory_payload_cannot_pass():
    summary = report_from_native_series(
        "TR-02",
        [{"result": np.ones(16), "program_bytes": b"x"}],
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_caller_oracle_and_error_fn_are_rejected():
    with pytest.raises(NativeSeriesError, match="oracle"):
        maybe_campaign_payload(
            _request(),
            np.ones(16),
            oracle=np.zeros(16),
            n_cells=16,
            t_end=1.0,
            time_program="SSPRK2",
            cfl=0.4,
        )
    with pytest.raises(NativeSeriesError, match="error_fn"):
        maybe_campaign_payload(
            _request(),
            np.ones(16),
            error_fn=lambda *_: 0.0,
            n_cells=16,
            t_end=1.0,
            time_program="SSPRK2",
            cfl=0.4,
        )
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    ).read_text(encoding="utf-8")
    assert "error_fn" not in source or "caller oracle/error_fn" in source
    assert "_SEAL_KEY" not in source
    assert "_REGISTERED_BINDS" not in source
    assert "hmac" not in source
    assert "_standin_identity" not in source
    assert "repr(artifact" not in source


def test_hand_built_authenticated_artifact_is_never_accepted(tmp_path: Path):
    artifact = AuthenticatedArtifact(
        dimension=1,
        path=tmp_path / "missing-leaf",
        sha256="a" * 64,
        version="1.0.0",
        abi_key="abi",
        has_mpi=False,
        has_kokkos=True,
        hdf5_collective=False,
        doctor_ok=False,
        native_variant_manifest_digest="b" * 64,
        native_header_signature="c" * 64,
        component_catalog_digest="d" * 64,
    )
    with pytest.raises(EvidenceError, match="path|artifact|directory"):
        EvidenceBundle(artifact)
    summary = report_from_native_series(
        "TR-02",
        artifact,
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    forged = object.__new__(EvidenceBundle)
    summary = report_from_native_series(
        "TR-02",
        forged,
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_missing_leaf_path_is_rejected(tmp_path: Path):
    series_dir, root = _emit_series(tmp_path)
    leaf = next(root.rglob("_pops*"))
    leaf.unlink()
    with pytest.raises(EvidenceError, match="leaf|absent|missing"):
        EvidenceBundle(series_dir)
    summary = report_from_native_series(
        "TR-02",
        series_dir,
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_forged_result_digest_is_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    digest = series_dir / "n016" / "result.sha256"
    digest.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="result"):
        EvidenceBundle(series_dir)


def test_modified_result_bytes_are_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    path = series_dir / "n016" / "result.npy"
    array = np.load(path)
    np.save(path, array * 2.0)
    with pytest.raises(EvidenceError, match="result"):
        EvidenceBundle(series_dir)


def test_forged_native_artifact_digest_is_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    path = series_dir / "n016" / "native_artifact.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sha256"] = "0" * 64
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="leaf|digest|artifact"):
        EvidenceBundle(series_dir)


def test_wrong_repository_sha_is_rejected(tmp_path: Path):
    root = _write_leaf(tmp_path)
    job = tmp_path / "series" / "n016"
    _emit_job(job, root, repository_sha="deadbeef" * 8)
    (tmp_path / "series" / "series.json").write_text(
        json.dumps({"case_id": "TR-02", "jobs": ["n016"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="repository"):
        EvidenceBundle(tmp_path / "series")


def test_wrong_case_and_dimension_are_rejected(tmp_path: Path):
    series_dir, root = _emit_series(tmp_path, case_id="TR-02")
    summary = report_from_native_series(
        "EU-01",
        series_dir,
        native_dimensions=[1],
        components=["euler"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    job = series_dir / "n016"
    resolved = json.loads((job / "resolved_case.json").read_text(encoding="utf-8"))
    resolved["job"]["pops_native_dim"] = 2
    (job / "resolved_case.json").write_text(json.dumps(resolved) + "\n", encoding="utf-8")
    (job / "resolved_case.sha256").write_text(
        sha256_file(job / "resolved_case.json") + "\n", encoding="utf-8"
    )
    with pytest.raises(EvidenceError, match="dimension|dim"):
        EvidenceBundle(series_dir)


def test_copied_record_cannot_change_case(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path, case_id="TR-02")
    document = json.loads((series_dir / "series.json").read_text(encoding="utf-8"))
    document["case_id"] = "EU-01"
    (series_dir / "series.json").write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="case"):
        EvidenceBundle(series_dir)


def test_absent_oracle_producer_is_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path, case_id="XX-99")
    with pytest.raises(EvidenceError, match="oracle producer"):
        EvidenceBundle(series_dir)


def test_fake_program_and_resolved_bytes_are_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    (series_dir / "n016" / "program.bin").write_bytes(b"forged-program")
    with pytest.raises(EvidenceError, match="program"):
        EvidenceBundle(series_dir)
    series_dir, _root = _emit_series(tmp_path / "b")
    resolved = series_dir / "n016" / "resolved_case.json"
    document = json.loads(resolved.read_text(encoding="utf-8"))
    document["job"]["suite"] = "forged"
    resolved.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="resolved"):
        EvidenceBundle(series_dir)


def test_tr06_job_without_pair_files_is_rejected(tmp_path: Path):
    root = _write_leaf(tmp_path, dimension=2)
    job = tmp_path / "job"
    _emit_job(
        job,
        root,
        case_id="TR-06",
        n_cells=8,
        dimension=2,
        result=np.ones((8, 8)),
    )
    with pytest.raises(EvidenceError, match="pair"):
        EvidenceBundle(job)


def test_private_variants_root_cannot_authenticate(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    with pytest.raises(EvidenceError, match="leaf|installed|variants"):
        EvidenceBundle(series_dir)
    summary = report_from_native_series(
        "TR-02",
        series_dir,
        native_dimensions=[1],
        components=["transport"],
        threshold=1.8,
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_report_never_accepts_prebuilt_bundle(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    forged = object.__new__(EvidenceBundle)
    try:
        object.__setattr__(forged, "_trusted", True)
        object.__setattr__(forged, "case_id", "TR-02")
        object.__setattr__(forged, "derived_linf", (0.01, 0.0025, 0.000625))
        object.__setattr__(forged, "derived_spacings", (1 / 16, 1 / 32, 1 / 64))
    except Exception:
        pass
    summary = report_from_native_series(
        "TR-02",
        forged,
        native_dimensions=[1],
        components=["transport"],
        threshold=1.8,
    )
    assert summary["coverage"]["cases_passed"] == 0
    assert "_trusted" not in getattr(EvidenceBundle, "__slots__", ())
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    ).read_text(encoding="utf-8")
    assert "_trusted" not in source
    with pytest.raises((EvidenceError, AttributeError, TypeError)):
        EvidenceBundle(series_dir).derived_linf = (0.0, 0.0)


def test_series_job_name_rejects_escape_and_absolute(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    (series_dir / "series.json").write_text(
        json.dumps({"case_id": "TR-02", "jobs": ["../n016"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="job name|relative|escape"):
        EvidenceBundle(series_dir)
    (series_dir / "series.json").write_text(
        json.dumps({"case_id": "TR-02", "jobs": [str(series_dir / "n016")]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="job name|relative|escape"):
        EvidenceBundle(series_dir)


def test_symlinked_series_job_or_result_is_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    target = tmp_path / "outside_result.npy"
    np.save(target, np.ones(16))
    result = series_dir / "n016" / "result.npy"
    result.unlink()
    result.symlink_to(target)
    with pytest.raises(EvidenceError, match="symlink"):
        EvidenceBundle(series_dir)


def test_pickle_and_unbounded_npy_are_rejected(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    path = series_dir / "n016" / "result.npy"
    np.save(path, np.array([{"forged": True}], dtype=object))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (series_dir / "n016" / "result.sha256").write_text(digest + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="pickle|dtype|shape|result"):
        EvidenceBundle(series_dir)
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "evidence_bundle.py"
    ).read_text(encoding="utf-8")
    assert "allow_pickle=False" in source


def test_tm01_spacing_is_actual_dt_not_one_over_n():
    from verification.pops_verify import evidence_bundle as eb

    spacing = getattr(eb, "sample_spacing", None)
    assert spacing is not None
    exact = _load(REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal", "exact")
    resolved = {"job": {"case_id": "TM-01", "dt": exact.DT, "min_resolution": 64}}
    provenance = {"resolution": [64], "cfl": float(exact.DT) * 64.0, "final_time": 0.01}
    assert spacing("TM-01", provenance, resolved, np.zeros(64)) == pytest.approx(exact.DT)
    assert spacing("TM-01", provenance, resolved, np.zeros(64)) != pytest.approx(1.0 / 64.0)
    dts = []
    for dt in exact.DT_SERIES:
        dts.append(
            spacing(
                "TM-01",
                provenance,
                {"job": {"case_id": "TM-01", "dt": dt, "min_resolution": 64}},
                np.zeros(64),
            )
        )
    assert dts == list(exact.DT_SERIES)
    assert len(set(np.round(dts, 16))) == len(exact.DT_SERIES)


def test_tr06_pair_must_be_distinct_and_analyzed(tmp_path: Path):
    root = _write_leaf(tmp_path, dimension=2)
    job = tmp_path / "job"
    field = np.ones((8, 8))
    _emit_job(
        job,
        root,
        case_id="TR-06",
        n_cells=8,
        dimension=2,
        result=field,
        pair_result=np.full((8, 8), 2.0),
        program_bytes=b"program-bytes",
        pair_program_bytes=b"program-bytes",
    )
    with pytest.raises(EvidenceError, match="pair"):
        EvidenceBundle(job)
    _emit_job(
        tmp_path / "garbage",
        root,
        case_id="TR-06",
        n_cells=8,
        dimension=2,
        result=field,
        pair_result=np.full((8, 8), 99.0),
        program_bytes=b"program-A",
        pair_program_bytes=b"program-B",
    )
    with pytest.raises(EvidenceError, match="pair|oracle|permutation"):
        EvidenceBundle(tmp_path / "garbage")


def test_positive_fixture_never_passes_on_private_leaf(tmp_path: Path):
    from verification.pops_verify.capabilities import CapabilityError, resolve_variants_root

    series_dir, _root = _emit_series(tmp_path)
    summary = report_from_native_series(
        "TR-02",
        series_dir,
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    try:
        resolve_variants_root(None)
    except CapabilityError:
        return
    # Installed leaf may exist; a private-root series still cannot pass.
    assert summary["coverage"]["cases_failed"] == 1


def test_below_threshold_path_cannot_pass(tmp_path: Path):
    series_dir, _root = _emit_series(
        tmp_path,
        shifts={16: 0.08, 32: 0.03, 64: 0.011},
    )
    summary = report_from_native_series(
        "TR-02",
        series_dir,
        native_dimensions=[1],
        components=["transport"],
        threshold=1.8,
    )
    assert summary["coverage"]["cases_passed"] == 0


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
    series_dir, _root = _emit_series(tmp_path)
    metrics = collect_metrics("TR-02", reason="no authenticated native series")
    Draft202012Validator(json.loads(SCHEMA_METRICS.read_text())).validate(metrics)
    with pytest.raises(NativeSeriesError):
        metrics_from_native_series("TR-02", series_dir)
    with pytest.raises(NativeSeriesError):
        provenance_from_native_series("TR-02", {"result": [1.0]})
    with pytest.raises(NativeSeriesError):
        metrics_from_native_series("TR-02", object.__new__(EvidenceBundle))


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


def test_producer_sources_include_cell_averages_helpers_and_git_blobs():
    from verification.pops_verify import oracle_producers as op

    files = {str(path.relative_to(REPO_ROOT)) for path in op.producer_source_files("TR-02")}
    assert "verification/pops_verify/cell_averages.py" in files
    assert "verification/pops_verify/oracle_producers.py" in files
    assert "verification/cases/transport/gaussian_pulse/exact.py" in files
    tm01 = {str(path.relative_to(REPO_ROOT)) for path in op.producer_source_files("TM-01")}
    assert "verification/cases/transport/advection_sine/exact.py" in tm01
    tr06 = {str(path.relative_to(REPO_ROOT)) for path in op.producer_source_files("TR-06")}
    assert "verification/cases/transport/advection_sine/exact.py" in tr06
    manifest = op.hash_producer_files("TR-02")
    cell = manifest["verification/pops_verify/cell_averages.py"]
    assert isinstance(cell, dict)
    assert "git_blob" in cell
    assert "head_sha256" in cell
    assert "sha256" in cell
    verify = getattr(op, "verify_committed_producers", None)
    assert verify is not None
    assert hasattr(op, "dirty_producer_paths")


def test_rewritten_producer_manifest_cannot_pass(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    path = series_dir / "n016" / "producer_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["forged"] = True
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="producer|manifest"):
        EvidenceBundle(series_dir)


def test_rewritten_cell_averages_manifest_cannot_pass(tmp_path: Path):
    series_dir, _root = _emit_series(tmp_path)
    path = series_dir / "n016" / "producer_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    key = "verification/pops_verify/cell_averages.py"
    assert key in document
    if isinstance(document[key], dict):
        document[key]["sha256"] = "0" * 64
    else:
        document[key] = "0" * 64
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    sidecar = series_dir / "n016" / "producer_manifest.sha256"
    if sidecar.is_file():
        sidecar.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    with pytest.raises(EvidenceError, match="producer|cell_averages|manifest"):
        EvidenceBundle(series_dir)


def test_dirty_cell_averages_cannot_pass_scientific_analysis(monkeypatch):
    from verification.pops_verify import oracle_producers as op

    def dirty(_case_id):
        return ["verification/pops_verify/cell_averages.py"]

    monkeypatch.setattr(op, "dirty_producer_paths", dirty)
    with pytest.raises(Exception, match="cell_averages|dirty|producer"):
        op.verify_committed_producers("TR-02")
    summary = report_from_native_series(
        "TR-02",
        None,
        native_dimensions=[1],
        components=["transport"],
    )
    assert summary["coverage"]["cases_passed"] == 0
    reasons = " ".join(item["reason"] for item in summary["failures"])
    assert "producer" in reasons.lower() or "cell_averages" in reasons.lower() or "oracle" in reasons.lower() or "native" in reasons.lower()
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "evidence_bundle.py"
    ).read_text(encoding="utf-8")
    assert "_repository_dirty() and" not in source


def test_payload_evidence_keys_stripped_and_can_emit_tm01_job(tmp_path: Path):
    from verification.pops_verify.native_evidence import (
        emission_from_payload,
        run_fields_from_payload,
    )
    from verification.pops_verify.capabilities import sha256_file

    program = tmp_path / "compiled.so"
    program.write_bytes(b"compiled-program-image")
    exact = _load(REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal", "exact")
    payload = {
        **campaign_run_fields(
            request=_request("TM-01", 1, 64),
            n_cells=64,
            t_end=0.01,
            time_program="SSPRK2",
            cfl=float(exact.DT) * 64.0,
            dimension=1,
        ),
        "case_id": "TM-01",
        "result": np.zeros(64, dtype=np.float64),
        "program_bytes": program.read_bytes(),
        "dt": float(exact.DT),
    }
    fields = run_fields_from_payload(payload)
    assert "program_bytes" not in fields
    assert "result" not in fields
    assert "dt" not in fields
    missing = [key for key in RUN_FIELDS if key not in fields]
    assert missing == []
    extra = emission_from_payload(payload)
    assert extra["program_bytes"] == program.read_bytes()
    assert extra["dt"] == pytest.approx(exact.DT)
    identity = authenticate_installed_artifact(
        dimension=1, variants_root=_write_leaf(tmp_path), doctor_ok=False
    )
    job_dir = tmp_path / "tm01-job"
    emit_job_directory(
        job_dir,
        resolved_case={
            "case": {"id": "TM-01"},
            "job": {
                "case_id": "TM-01",
                "pops_native_dim": 1,
                "suite": "pr",
                "execution_space": "KokkosSerial",
                "mpi_mode": "off",
                "min_resolution": 64,
                "dt": extra["dt"],
                "evidence_status": "required",
                "resources": {
                    "nodes": 1,
                    "mpi_ranks": 1,
                    "omp_threads": 1,
                    "resolutions": [64],
                },
            },
        },
        provenance=collect_provenance(
            "TM-01",
            pops_native_dim=1,
            dimension=1,
            nodes=1,
            pops_version="test",
            doctor_ok=False,
            component_catalog_digest=identity.component_catalog_digest,
            native_header_signature=identity.native_header_signature,
            native_variant_manifest_digest=identity.native_variant_manifest_digest,
            **fields,
        ),
        metrics=collect_metrics("TM-01", reason="contract fixture"),
        result=extra["result"],
        program_bytes=extra["program_bytes"],
        native_artifact={
            "path": str(identity.path),
            "sha256": identity.sha256,
            "dimension": 1,
        },
    )
    resolved = json.loads((job_dir / "resolved_case.json").read_text(encoding="utf-8"))
    assert resolved["job"]["dt"] == pytest.approx(exact.DT)
    assert (job_dir / "program.bin").read_bytes() == program.read_bytes()
    assert sha256_file(job_dir / "program.bin") != hashlib.sha256(repr(object()).encode()).hexdigest()
    summary = report_from_native_series(
        "TM-01",
        job_dir,
        native_dimensions=[1],
        components=["time"],
    )
    assert summary["coverage"]["cases_passed"] == 0


def test_emit_and_load_reject_dirty_or_head_mismatched_producers(tmp_path, monkeypatch):
    from verification.pops_verify import evidence_contract as ec
    from verification.pops_verify import oracle_producers as op
    from verification.pops_verify.oracle_producers import OracleProducerError

    def boom(case_id):
        raise OracleProducerError("dirty producer files: ['verification/pops_verify/cell_averages.py']")

    monkeypatch.setattr(op, "verify_committed_producers", boom)
    if hasattr(ec, "verify_committed_producers"):
        monkeypatch.setattr(ec, "verify_committed_producers", boom)
    with pytest.raises((EvidenceError, OracleProducerError, Exception), match="dirty|producer"):
        _emit_series(tmp_path / "emit")

    monkeypatch.undo()
    series_dir, _root = _emit_series(tmp_path / "load")
    live = op.hash_producer_files("TR-02")
    dirty = json.loads(json.dumps(live))
    key = "verification/pops_verify/cell_averages.py"
    dirty[key]["sha256"] = "a" * 64
    path = series_dir / "n016" / "producer_manifest.json"
    path.write_text(json.dumps(dirty, indent=2) + "\n", encoding="utf-8")
    (series_dir / "n016" / "producer_manifest.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(op, "hash_producer_files", lambda case_id: dirty)
    with pytest.raises(EvidenceError, match="HEAD|dirty|producer"):
        EvidenceBundle(series_dir)
    source = (
        REPO_ROOT / "verification" / "pops_verify" / "native_evidence.py"
    ).read_text(encoding="utf-8")
    assert "verify_committed_producers(" not in source


def test_program_so_path_rejects_symlink_and_leaf_alias(tmp_path: Path):
    from verification.pops_verify.native_evidence import NativeSeriesError, read_stable_program_file

    real = tmp_path / "prog.so"
    real.write_bytes(b"compiled-program-image")
    link = tmp_path / "prog-link.so"
    link.symlink_to(real)
    with pytest.raises((EvidenceError, NativeSeriesError), match="symlink|so_path"):
        read_stable_program_file(link)
    leaf_root = _write_leaf(tmp_path / "leaf")
    identity = authenticate_installed_artifact(
        dimension=1, variants_root=leaf_root, doctor_ok=False
    )
    with pytest.raises((EvidenceError, NativeSeriesError), match="leaf|distinct"):
        read_stable_program_file(identity.path, installed_leaf=identity.path)
    data = read_stable_program_file(real, installed_leaf=identity.path)
    assert data == b"compiled-program-image"


def test_program_digest_must_not_equal_leaf_digest(tmp_path: Path):
    leaf_payload = b"fake-exact-rank-leaf-smooth"
    series_dir, root = _emit_series(tmp_path)
    job = series_dir / "n016"
    identity = authenticate_installed_artifact(
        dimension=1, variants_root=root, doctor_ok=False
    )
    (job / "program.bin").write_bytes(leaf_payload)
    (job / "program.sha256").write_text(identity.sha256 + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="program digest must not equal leaf"):
        EvidenceBundle(series_dir)


def test_tr06_pair_program_must_not_equal_leaf_digest(tmp_path: Path):
    root = _write_leaf(tmp_path, dimension=2)
    identity = authenticate_installed_artifact(
        dimension=2, variants_root=root, doctor_ok=False
    )
    exact = _load(REPO_ROOT / "verification" / "cases" / "transport" / "axis_permutation", "exact")
    n = 8
    width = float(exact.PERIOD) / n
    axis_lo = np.arange(n, dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing="ij")
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing="ij")
    lo = np.stack((x_lo, y_lo), axis=-1)
    hi = np.stack((x_hi, y_hi), axis=-1)
    original = analytic_cell_averages(lambda x, y: exact.exact_product(x, y, 1.0), lo, hi)
    paired = analytic_cell_averages(lambda x, y: exact.exact_product(y, x, 1.0), lo, hi)
    job = tmp_path / "job"
    _emit_job(
        job,
        root,
        case_id="TR-06",
        n_cells=8,
        dimension=2,
        result=original,
        pair_result=paired,
        program_bytes=b"program-A",
        pair_program_bytes=identity.path.read_bytes(),
    )
    with pytest.raises(EvidenceError, match="pair program digest must not equal leaf"):
        EvidenceBundle(job)
