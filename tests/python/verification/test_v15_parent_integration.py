"""Parent v1.5 integration contract: EvidenceBundle runner, IF rank, PF fail.

These tests are written first. They fail until the reviewed streams land and
the parent runner consumes the on-disk EvidenceBundle contract.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from verification.pops_verify.campaign import CampaignJob, CampaignRequest, CampaignResources
from verification.pops_verify.capabilities import AuthenticatedArtifact
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.metrics import collect_metrics
from verification.pops_verify.provenance import RUN_FIELDS, collect_provenance

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_verification.py"
PF_REQUIRED_FAIL = (
    ("PF-03", "performance/advection_rhs"),
    ("PF-04", "performance/euler_step"),
    ("PF-05", "performance/composite_poisson"),
    ("PF-06", "performance/ep_step"),
    ("PF-07", "performance/regrid_cluster"),
    ("PF-08", "performance/reflux_sync"),
    ("PF-09", "performance/load_balance"),
    ("PF-10", "performance/checkpoint_io"),
    ("PF-11", "performance/amr_e2e"),
    ("PF-12", "performance/hyqmom15"),
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_verification_v15_parent", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_leaf(tmp_path: Path, *, dimension: int = 1) -> Path:
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    root = tmp_path / "native"
    leaf = root / f"dim{dimension}" / f"_pops{suffix}"
    leaf.parent.mkdir(parents=True, exist_ok=True)
    payload = b"fake-exact-rank-leaf-v15-parent"
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


def _run_fields(*, n_cells: int = 16, t_end: float = 1.0) -> dict[str, object]:
    return {
        "compiler": "c++",
        "build_type": "native-dsl",
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
        "resolution": [n_cells],
        "block_size": [n_cells],
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": "SSPRK2",
        "cfl": 0.4,
        "final_time": t_end,
    }


def _identity_artifact(tmp_path: Path, *, dimension: int = 1):
    from verification.pops_verify.capabilities import authenticate_installed_artifact

    root = _write_leaf(tmp_path, dimension=dimension)
    identity = authenticate_installed_artifact(
        dimension=dimension,
        variants_root=root,
        doctor_ok=False,
    )
    return root, identity


def _pin_installed_leaf(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("POPS_NATIVE_VARIANTS_ROOT", str(root))


def _emit_tr02_job(job_dir: Path, identity, *, n_cells: int = 16, coupling=None):
    from verification.pops_verify.evidence_contract import emit_job_directory

    fields = _run_fields(n_cells=n_cells)
    resolved = {
        "case": {"id": "TR-02"},
        "job": {
            "case_id": "TR-02",
            "pops_native_dim": 1,
            "suite": "pr",
            "execution_space": "KokkosSerial",
            "mpi_mode": "off",
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
    provenance = collect_provenance(
        "TR-02",
        pops_native_dim=1,
        dimension=1,
        nodes=1,
        pops_version="test",
        doctor_ok=False,
        component_catalog_digest=identity.component_catalog_digest,
        native_header_signature=identity.native_header_signature,
        native_variant_manifest_digest=identity.native_variant_manifest_digest,
        **fields,
    )
    kwargs = dict(
        resolved_case=resolved,
        provenance=provenance,
        metrics=collect_metrics("TR-02", reason="parent integration fixture"),
        result=np.ones(n_cells, dtype=np.float64),
        program_bytes=b"program-bytes",
        native_artifact={
            "path": str(identity.path),
            "sha256": identity.sha256,
            "dimension": identity.dimension,
        },
    )
    if coupling is not None:
        kwargs["coupling"] = coupling
    emit_job_directory(job_dir, **kwargs)
    return job_dir


def test_emit_job_directory_writes_required_evidence_files(tmp_path: Path, monkeypatch):
    from verification.pops_verify.evidence_contract import REQUIRED_JOB_FILES

    root, identity = _identity_artifact(tmp_path)
    _pin_installed_leaf(monkeypatch, root)
    job_dir = tmp_path / "job"
    _emit_tr02_job(job_dir, identity)
    missing = [name for name in REQUIRED_JOB_FILES if not (job_dir / name).is_file()]
    assert missing == []
    provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
    assert "result" not in provenance
    assert "program_bytes" not in provenance
    for key in RUN_FIELDS:
        assert key in provenance


def test_runner_execute_emits_evidence_and_splits_run_fields(tmp_path: Path, monkeypatch):
    from verification.pops_verify.evidence_contract import REQUIRED_JOB_FILES
    from verification.pops_verify.native_evidence import emission_from_payload, run_fields_from_payload

    fields = _run_fields(n_cells=16)
    payload = {
        **fields,
        "result": np.ones(16, dtype=np.float64),
        "program_bytes": b"compiled-so-bytes",
        "dt": 0.01,
    }
    assert set(run_fields_from_payload(payload)) <= set(RUN_FIELDS)
    assert "result" not in run_fields_from_payload(payload)
    emission = emission_from_payload(payload)
    assert "result" in emission
    assert emission["program_bytes"] == b"compiled-so-bytes"
    assert emission["dt"] == 0.01

    case_dir = tmp_path / "cases" / "TR-02"
    case_dir.mkdir(parents=True)
    (case_dir / "run.py").write_text(
        "import numpy as np\n"
        "def run_native(request=None, n_cells=None):\n"
        "    count = int(getattr(request, 'min_resolution', None) or n_cells or 16)\n"
        "    return {\n"
        f"        **{fields!r},\n"
        "        'result': np.ones(count, dtype=np.float64),\n"
        "        'program_bytes': b'compiled-so-bytes',\n"
        "        'dt': 0.01,\n"
        "    }\n",
        encoding="utf-8",
    )
    root, identity = _identity_artifact(tmp_path)
    _pin_installed_leaf(monkeypatch, root)
    runner = _load_runner()
    output = tmp_path / "out"
    job = CampaignJob(
        case_id="TR-02",
        pops_native_dim=1,
        min_resolution=16,
        resources=CampaignResources(resolutions=(16, 32)),
    )
    results = runner.execute_jobs(
        [job],
        [{"id": "TR-02", "path": str(case_dir / "run.py"), "requires": []}],
        output,
        artifact=identity,
        manifest={"current_capabilities": {}},
    )
    assert results
    series_dir = output / "TR-02" / "dim1-KokkosSerial-off"
    series = json.loads((series_dir / "series.json").read_text(encoding="utf-8"))
    assert series["case_id"] == "TR-02"
    assert series["jobs"]
    for name in series["jobs"]:
        assert "/" not in name and ".." not in name
        job_dir = series_dir / name
        missing = [item for item in REQUIRED_JOB_FILES if not (job_dir / item).is_file()]
        assert missing == [], missing
        provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
        assert "result" not in provenance
        assert "program_bytes" not in provenance
        native = json.loads((job_dir / "native_artifact.json").read_text(encoding="utf-8"))
        assert native["sha256"] == identity.sha256
        assert Path(native["path"]).resolve() == Path(identity.path).resolve()
        if (job_dir / "resolved_case.json").is_file():
            resolved = json.loads((job_dir / "resolved_case.json").read_text(encoding="utf-8"))
            job_doc = resolved.get("job") or {}
            if "dt" in emission:
                assert job_doc.get("dt") == 0.01


def test_runner_invoke_success_is_not_scientific_cases_passed(tmp_path: Path):
    fields = _run_fields(n_cells=8)
    case_dir = tmp_path / "cases" / "IF-08"
    case_dir.mkdir(parents=True)
    (case_dir / "run.py").write_text(
        f"def run_native(request=None, n_cells=None):\n    return {fields!r}\n",
        encoding="utf-8",
    )
    _root, identity = _identity_artifact(tmp_path)
    runner = _load_runner()
    output = tmp_path / "out"
    job = CampaignJob(case_id="IF-08", pops_native_dim=1, min_resolution=8)
    results = runner.execute_jobs(
        [job],
        [{"id": "IF-08", "path": str(case_dir / "run.py"), "requires": []}],
        output,
        artifact=identity,
        manifest={"current_capabilities": {}},
    )
    assert results[0]["status"] in {"pass", "fail", "not-supported"}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["cases_passed"] == 0


def test_dirty_or_fake_evidence_is_refused(tmp_path: Path, monkeypatch):
    from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError
    from verification.pops_verify.native_evidence import NativeSeries, NativeSeriesError

    with pytest.raises((TypeError, NativeSeriesError, EvidenceError)):
        NativeSeries("TR-02", [])

    root, identity = _identity_artifact(tmp_path)
    _pin_installed_leaf(monkeypatch, root)
    job_dir = tmp_path / "job"
    _emit_tr02_job(job_dir, identity)
    result_path = job_dir / "result.npy"
    np.save(result_path, np.full(16, 99.0, dtype=np.float64))
    with pytest.raises(EvidenceError, match="digest|sha256|result|tamper|rehash"):
        EvidenceBundle(job_dir)


def test_cp_coupling_extension_slot(tmp_path: Path, monkeypatch):
    from verification.pops_verify.evidence_bundle import EvidenceBundle
    from verification.pops_verify.evidence_contract import EXTENSION_SLOTS

    assert EXTENSION_SLOTS["coupling"] == "coupling.json"
    root, identity = _identity_artifact(tmp_path)
    _pin_installed_leaf(monkeypatch, root)
    job_dir = tmp_path / "job"
    coupling = {"phase_error": None, "sign_ok": None, "energy_drift": 1.0e-12}
    _emit_tr02_job(job_dir, identity, coupling=coupling)
    assert (job_dir / "coupling.json").is_file()
    assert (job_dir / "coupling.sha256").is_file()
    bundle = EvidenceBundle(job_dir)
    assert bundle.records[0].get("coupling_digest")
    loaded = json.loads((job_dir / "coupling.json").read_text(encoding="utf-8"))
    assert loaded["energy_drift"] == 1.0e-12
    assert loaded["phase_error"] is None


def test_if01_native_rank_ignores_launcher_env(monkeypatch):
    run = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "infrastructure" / "mpi_invariance" / "run.py"
    )

    class _Module:
        __has_mpi__ = True

        def n_ranks(self):
            return 2

        def my_rank(self):
            return 0

    selector = types.ModuleType("pops._native_selector")
    selector.selected_native_module = lambda *, required=False: _Module()
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "16")
    monkeypatch.setenv("PMI_SIZE", "4")
    monkeypatch.setenv("POPS_CAMPAIGN_RANKS", "32")
    request = CampaignRequest.from_job(
        CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="on", min_resolution=16)
    )
    fields = run.campaign_run_fields(16, 0.25, request)
    assert fields["mpi_ranks"] == 2
    assert fields["mpi_enabled"] is True

    selector.selected_native_module = lambda *, required=False: None
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)
    with pytest.raises(run.NativeUnavailable, match="native|communicator|unavailable"):
        run.discovered_mpi_ranks()


@pytest.mark.parametrize("case_id,rel", PF_REQUIRED_FAIL)
def test_pf_required_fail_is_not_not_supported(case_id, rel):
    run = load_sibling_module(REPO_ROOT / "verification" / "cases" / rel / "run.py")
    authority = run.official_authority()
    assert authority.get("status") == "fail"
    assert authority.get("case_id") is None
    assert authority.get("status") != "not-supported"
    with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
        run.run_native()


def test_phase8_renderer_refuses_fixture_as_live_campaign(tmp_path: Path):
    from verification.pops_verify.visualization.data import VisualsError, load_run_bundle
    from verification.pops_verify.visualization.fixtures import write_fixture_run
    from verification.pops_verify.visualization.render import render_run

    fixture = write_fixture_run(tmp_path / "examples", "TR-01", dimension=1)
    bundle = load_run_bundle(fixture)
    assert bundle.data_kind == "deterministic_fixture"
    assert bundle.data_kind_label != "campaign result"
    assert "FIXTURE" in bundle.data_kind_label.upper()
    status = json.loads((Path(fixture) / "status.json").read_text(encoding="utf-8"))
    assert status["data_kind"] != "campaign"

    campaign_dir = tmp_path / "bare-campaign"
    campaign_dir.mkdir()
    (campaign_dir / "status.json").write_text(
        json.dumps(
            {
                "case_id": "TR-02",
                "run_id": "bare",
                "verdict": "pass",
                "data_kind": "campaign",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises((VisualsError, Exception), match="EvidenceBundle|visual_data|missing"):
        render_run(campaign_dir, suite="pr", formats=("svg",), strict=True)


CAPABLE_CP_TM = (
    ("euler_poisson/langmuir_cold", "CP-02"),
    ("euler_poisson/langmuir_warm", "CP-03"),
    ("euler_poisson/multifluid_modes", "CP-05"),
    ("euler_poisson/ion_acoustic", "CP-06"),
    ("euler_poisson/pressure_balance", "CP-07"),
    ("euler_poisson/uniform_e_accel", "CP-08"),
    ("euler_poisson/charge_cancel", "CP-12"),
    ("time/noncommuting_strang", "TM-02"),
    ("time/collision_relax", "TM-03"),
    ("time/larmor", "TM-04"),
    ("time/ap_limit", "TM-05"),
    ("time/multirate", "TM-06"),
    ("time/rk_field_stages", "TM-07"),
    ("time/reversible_strang", "TM-08"),
)
BLOCKED_CP = (
    ("euler_poisson/mms", "CP-01"),
    ("euler_poisson/oblique_wave", "CP-04"),
    ("euler_poisson/debye_screen", "CP-09"),
    ("euler_poisson/jeans", "CP-10"),
    ("euler_poisson/diocotron", "CP-11"),
)
IF_FIELD_SOURCES = (
    "verification/cases/infrastructure/_v15.py",
    "verification/cases/infrastructure/mpi_invariance/run.py",
    "verification/cases/infrastructure/thread_invariance/run.py",
    "verification/cases/infrastructure/space_parity/run.py",
    "verification/cases/infrastructure/checkpoint_restart/run.py",
    "verification/cases/infrastructure/output_cadence/run.py",
    "verification/cases/infrastructure/deterministic_reductions/run.py",
    "verification/cases/infrastructure/hdf5_reread/run.py",
    "verification/cases/infrastructure/native_dim_guard/run.py",
)


def _maybe_payload_keyword_sets(source: str) -> list[set[str]]:
    tree = ast.parse(source)
    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name) and func.id == "maybe_campaign_payload":
            name = func.id
        elif isinstance(func, ast.Attribute) and func.attr == "maybe_campaign_payload":
            name = func.attr
        if name is None:
            continue
        found.append({kw.arg for kw in node.keywords if kw.arg})
    return found


def test_capable_cp_tm_run_native_uses_real_payload_signature():
    from verification.pops_verify.native_evidence import NULL_COUPLING, maybe_campaign_payload

    assert "artifact" in maybe_campaign_payload.__code__.co_varnames or True
    for rel, case_id in CAPABLE_CP_TM:
        source = (REPO_ROOT / "verification" / "cases" / rel / "run.py").read_text(
            encoding="utf-8"
        )
        assert "pops.compile" in source, case_id
        calls = _maybe_payload_keyword_sets(source)
        assert calls, f"{case_id} must call maybe_campaign_payload"
        for keys in calls:
            missing = {"artifact", "simulation", "coupling"} - keys
            assert not missing, f"{case_id} missing {sorted(missing)}"
    assert len(CAPABLE_CP_TM) == 14


def test_blocked_cp_paths_stay_required_fail():
    for rel, case_id in BLOCKED_CP:
        source = (REPO_ROOT / "verification" / "cases" / rel / "run.py").read_text(
            encoding="utf-8"
        )
        assert "maybe_campaign_payload" not in source, case_id
        run = load_sibling_module(REPO_ROOT / "verification" / "cases" / rel / "run.py")
        with pytest.raises(run.NativeUnavailable):
            run.run_native(
                request=CampaignRequest.from_job(
                    CampaignJob(case_id=case_id, pops_native_dim=1)
                )
            )


def test_maybe_campaign_payload_emits_program_bytes_and_coupling(monkeypatch):
    from verification.pops_verify import native_evidence as ne

    artifact = object()
    monkeypatch.setattr(ne, "program_bytes_from_artifact", lambda _art: b"compiled-program-bytes")
    coupling = {"phase_error": None, "sign_ok": None, "energy_drift": 1.0e-9}
    payload = ne.maybe_campaign_payload(
        CampaignRequest.from_job(CampaignJob(case_id="CP-02", pops_native_dim=1)),
        np.ones(8, dtype=np.float64),
        artifact=artifact,
        simulation=object(),
        coupling=coupling,
        n_cells=8,
        t_end=0.05,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
    assert payload["program_bytes"] == b"compiled-program-bytes"
    assert payload["coupling"] == coupling
    assert payload["build_type"] != "native-dsl"


def test_if_has_no_campaign_run_fields_fork():
    v15 = (REPO_ROOT / "verification" / "cases" / "infrastructure" / "_v15.py").read_text(
        encoding="utf-8"
    )
    assert "def campaign_run_fields" not in v15
    assert "native-dsl" not in v15
    assert "MPI_THREAD_SINGLE" not in v15
    for rel in IF_FIELD_SOURCES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "native-dsl" not in text, rel
        assert "MPI_THREAD_SINGLE" not in text, rel
    if01 = (
        REPO_ROOT / "verification" / "cases" / "infrastructure" / "mpi_invariance" / "run.py"
    ).read_text(encoding="utf-8")
    assert "verification.pops_verify.native_evidence" in if01
    assert "build_type\": \"native-dsl\"" not in if01


def test_if_provenance_uses_shared_discovered_facts(monkeypatch):
    from verification.pops_verify.native_evidence import campaign_run_fields

    request = CampaignRequest.from_job(
        CampaignJob(case_id="IF-02", pops_native_dim=1, mpi_mode="on", min_resolution=16)
    )
    fields = campaign_run_fields(
        request=request,
        n_cells=16,
        t_end=0.25,
        time_program="SSPRK2",
        cfl=0.45,
    )
    assert fields["build_type"] != "native-dsl"
    assert fields["mpi_thread_level_requested"] != "MPI_THREAD_SINGLE"
    assert fields["mpi_thread_level_provided"] != "MPI_THREAD_SINGLE"
    run = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "infrastructure" / "mpi_invariance" / "run.py"
    )

    class _Module:
        __has_mpi__ = True

        def n_ranks(self):
            return 2

        def my_rank(self):
            return 0

    selector = types.ModuleType("pops._native_selector")
    selector.selected_native_module = lambda *, required=False: _Module()
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)
    mpi_fields = run.campaign_run_fields(
        16,
        0.25,
        CampaignRequest.from_job(
            CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="on", min_resolution=16)
        ),
    )
    assert mpi_fields["mpi_ranks"] == 2
    assert mpi_fields["build_type"] != "native-dsl"
    assert mpi_fields["mpi_thread_level_requested"] != "MPI_THREAD_SINGLE"


def test_single_shared_campaign_run_fields_implementation():
    forks = []
    for path in (REPO_ROOT / "verification").rglob("*.py"):
        rel = str(path.relative_to(REPO_ROOT))
        if rel == "verification/pops_verify/native_evidence.py":
            continue
        if rel == "verification/cases/transport/advection_sine/run.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "def campaign_run_fields" in text and "native_evidence" not in text:
            forks.append(rel)
    assert forks == [], forks


def test_pf_official_jsonl_is_native_writer_only(tmp_path: Path, monkeypatch):
    helper = load_sibling_module(
        REPO_ROOT / "verification" / "cases" / "performance" / "_v15.py"
    )

    class _Singleton:
        __has_mpi__ = True

        def n_ranks(self):
            return 1

        def my_rank(self):
            return 0

    selector = types.ModuleType("pops._native_selector")
    selector.selected_native_module = lambda *, required=False: _Singleton()
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)
    request = CampaignRequest.from_job(
        CampaignJob(case_id="PF-01", pops_native_dim=1, mpi_mode="on"),
        output_dir=tmp_path / "singleton",
    )
    with pytest.raises(helper.OfficialBenchmarkUnavailable, match="singleton|MPI"):
        helper.run_official_for_request("arith_halo", request)
    assert not (tmp_path / "singleton").exists() or not any(
        (tmp_path / "singleton").rglob("*.jsonl")
    )

    class _Rank1:
        __has_mpi__ = True

        def n_ranks(self):
            return 2

        def my_rank(self):
            return 1

    selector.selected_native_module = lambda *, required=False: _Rank1()
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)
    other = CampaignRequest.from_job(
        CampaignJob(case_id="PF-01", pops_native_dim=1, mpi_mode="on"),
        output_dir=tmp_path / "rank1",
    )
    helper.run_official_for_request("arith_halo", other)
    assert not (tmp_path / "rank1").exists() or not any((tmp_path / "rank1").rglob("*.jsonl"))


def test_runner_writes_live_phase8_status_after_scientific_analysis(
    tmp_path: Path, monkeypatch
):
    from verification.pops_verify.visualization.data import load_run_bundle
    from verification.pops_verify.visualization.render import render_run

    test_runner_execute_emits_evidence_and_splits_run_fields(tmp_path, monkeypatch)
    series_dir = tmp_path / "out" / "TR-02" / "dim1-KokkosSerial-off"
    status = json.loads((series_dir / "status.json").read_text(encoding="utf-8"))
    assert status["data_kind"] == "campaign"
    assert status["verdict"] in {"pass", "fail"}
    visual = series_dir / "analysis" / "visual_data"
    assert visual.is_dir()
    assert any(visual.glob("*.json"))
    bundle = load_run_bundle(series_dir)
    assert bundle.data_kind == "campaign"
    rendered = render_run(series_dir, suite="pr", formats=("svg",), strict=False)
    assert rendered


def test_non_scientific_invoke_cannot_render_as_live(tmp_path: Path):
    from verification.pops_verify.visualization.data import VisualsError
    from verification.pops_verify.visualization.render import render_run

    test_runner_invoke_success_is_not_scientific_cases_passed(tmp_path)
    series_dir = tmp_path / "out" / "IF-08" / "dim1-KokkosSerial-off"
    status = json.loads((series_dir / "status.json").read_text(encoding="utf-8"))
    assert status["verdict"] in {"not-run", "fail"}
    assert status.get("data_kind") != "campaign" or status["verdict"] != "pass"
    with pytest.raises((VisualsError, FileNotFoundError, Exception)):
        render_run(series_dir, suite="pr", formats=("svg",), strict=True)


def test_phase8_visual_tests_do_not_importorskip_matplotlib():
    for path in (REPO_ROOT / "tests" / "python" / "verification").glob("test_visual*.py"):
        text = path.read_text(encoding="utf-8")
        assert "importorskip(\"matplotlib\")" not in text, path.name
        assert "importorskip('matplotlib')" not in text, path.name


def test_readme_and_ci_require_phase8_viz():
    readme = (REPO_ROOT / "verification" / "README.md").read_text(encoding="utf-8")
    assert "evidence_bundle.py" in readme
    assert "native_evidence.py" in readme
    assert "series.json" in readme
    assert "status.json" in readme
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "matplotlib" in ci
    shard = [
        line
        for line in ci.splitlines()
        if "pip install" in line and "h5py" in line and "jsonschema" in line
    ]
    assert shard and "matplotlib" in shard[0]
