"""Parent integration contracts for approved CP-02 1-d (evidence SHA e78e310ad).

These tests are written first. They fail until the reviewed CP-02 case, core,
and Phase 8 patches land on the parent EvidenceBundle / native-writer APIs.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from verification.pops_verify.case_authoring import load_sibling_module

REPO_ROOT = Path(__file__).resolve().parents[3]
CP02_RUN = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "run.py"
CP02_ANALYZE = (
    REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "analyze.py"
)
EVIDENCE_SHA = "e78e310ad7ce60cf4c4bdc3a91db650ec2b4372c"


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


def test_cp02_payload_uses_weno5z_and_approved_helper_api():
    source = CP02_RUN.read_text(encoding="utf-8")
    run = load_sibling_module(CP02_RUN)
    exact = load_sibling_module(CP02_RUN.parent / "exact.py")
    assert getattr(run, "DEFAULT_RECONSTRUCTION", None) == "weno5z"
    assert "WENO5Z" in source
    assert "reconstruction.MUSCL(limiter=limiters.VanLeer())" not in source.split(
        "def _reconstruction_brick"
    )[0]
    assert hasattr(exact, "Q_E")
    assert hasattr(exact, "PHASE_CFL")
    assert hasattr(run, "stamp_grid_clips_dt")
    assert hasattr(run, "sample_times_for_family")
    assert hasattr(run, "run_temporal_campaign")
    assert hasattr(run, "energy_baseline")
    assert hasattr(run, "cell_average_fields")
    calls = _maybe_payload_keyword_sets(source)
    assert calls, "CP-02 must call maybe_campaign_payload"
    for keys in calls:
        missing = {"artifact", "simulation", "coupling"} - keys
        assert not missing, f"CP-02 missing approved helper keywords {sorted(missing)}"
    stamps = run.sample_times_for_family("temporal", 2.0 * 3.141592653589793)
    assert stamps.size == 1
    assert run.stamp_grid_clips_dt(run.default_sample_times(2.0 * 3.141592653589793), 0.4)
    assert not run.stamp_grid_clips_dt(stamps, 0.4)


def test_cp02_oracle_producer_and_parent_tr01_eu02_stay():
    from verification.pops_verify.oracle_producers import CASE_SOURCES, PRODUCERS

    assert "CP-02" in PRODUCERS
    assert CASE_SOURCES["CP-02"] == ("euler_poisson", "langmuir_cold")
    assert "TR-01" in PRODUCERS
    assert CASE_SOURCES["TR-01"] == ("transport", "advection_sine")
    assert "EU-02" in PRODUCERS
    assert CASE_SOURCES["EU-02"] == ("euler", "isentropic_vortex")


def test_compile_cache_lock_and_last_axis_slab_helpers():
    from pops.codegen.compile_provenance import load_or_publish_cached_artifact
    from pops.runtime._runtime_mesh_lowering import last_axis_slab_boxes

    assert callable(load_or_publish_cached_artifact)
    assert last_axis_slab_boxes((16,), 1) == (((0,), (16,)),)
    assert last_axis_slab_boxes((16,), 2) == (((0,), (8,)), ((8,), (16,)))
    assert last_axis_slab_boxes((8, 16), 2) == (((0, 0), (8, 8)), ((0, 8), (8, 16)))
    with pytest.raises(ValueError):
        last_axis_slab_boxes((16,), 3)
    facade = (REPO_ROOT / "python" / "pops" / "physics" / "_facade_compile.py").read_text(
        encoding="utf-8"
    )
    assert "load_or_publish_cached_artifact" in facade


def test_generated_weno_halo_and_finalize_rethrow_sources():
    block = (
        REPO_ROOT
        / "include"
        / "pops"
        / "runtime"
        / "builders"
        / "compiled"
        / "generated_system_block.hpp"
    ).read_text(encoding="utf-8")
    assert "fill_generated_state_boundary" in block
    assert "HaloExchange" in block
    assert "duplicate_world_collectively" in block
    finalize = REPO_ROOT / "include" / "pops" / "runtime" / "native_package_finalize.hpp"
    assert finalize.is_file()
    text = finalize.read_text(encoding="utf-8")
    assert "rethrow_native_package_finalize_failure" in text
    halo = REPO_ROOT / "tests" / "cpp" / "integration" / "mpi" / "test_mpi_generated_weno_halo.cpp"
    assert halo.is_file()
    inner = (
        REPO_ROOT
        / "tests"
        / "cpp"
        / "unit"
        / "runtime"
        / "test_native_package_finalize_inner_exception.cpp"
    )
    assert inner.is_file()


def test_cmake_weno_halo_links_runtime_system():
    cmake = (REPO_ROOT / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")
    sources = (REPO_ROOT / "tests" / "cpp" / "test_sources.cmake").read_text(encoding="utf-8")
    assert "test_mpi_generated_weno_halo" in cmake
    assert "test_mpi_generated_weno_halo" in sources
    assert "POPS_MPI_RANKS_test_mpi_generated_weno_halo 1 2" in cmake
    assert 'elseif(_test STREQUAL "test_mpi_generated_weno_halo")' in cmake
    match = re.search(
        r'elseif\(_test STREQUAL "test_mpi_generated_weno_halo"\)\s*'
        r"list\(APPEND _extra_libs[^\n]*pops_runtime_system",
        cmake,
    )
    assert match, "test_mpi_generated_weno_halo must EXTRA_LIBS pops_runtime_system"


def test_phase8_figure_verdict_is_last_two_linf_pairs():
    from verification.pops_verify.visualization.plots import derive_figure_verdict

    passing = {
        "kind": "spatial_convergence",
        "units": {"x": "1/h", "y": "error"},
        "series": [
            {
                "name": "L2",
                "x": [16, 32, 64, 128],
                "y": [4.80e-7, 9.91e-8, 1.30e-8, 1.65e-9],
                "unit": "1",
            },
            {
                "name": "Linf",
                "x": [16, 32, 64, 128],
                "y": [6.676e-7, 1.396e-7, 1.831e-8, 2.336e-9],
                "unit": "1",
            },
        ],
        "reference_slopes": [{"name": "order 2", "order": 2.0, "anchor": [16, 1.0e-3]}],
    }
    assert derive_figure_verdict(passing) == "pass"

    mean_two_but_last_pairs_fail = {
        "kind": "spatial_convergence",
        "units": {"x": "1/h", "y": "error"},
        "series": [
            {
                "name": "L2",
                "x": [16, 32, 64, 128],
                "y": [1.6e-2, 4.0e-3, 2.0e-3, 1.0e-3],
                "unit": "1",
            },
            {
                "name": "Linf",
                "x": [16, 32, 64, 128],
                "y": [1.6e-2, 4.0e-3, 2.0e-3, 1.4e-3],
                "unit": "1",
            },
        ],
        "reference_slopes": [{"order": 2, "anchor": [16, 1.6e-2]}],
    }
    assert derive_figure_verdict(mean_two_but_last_pairs_fail) == "fail"


def test_cp02_2d_catalog_is_extended_parent_tr01_eu02_unchanged():
    from verification.pops_verify.visualization.catalog import (
        catalog_entry,
        visual_contract_for,
    )

    contract = visual_contract_for("CP-02")
    assert contract["dimensions"]["1d"]["status"] == "required"
    assert contract["dimensions"]["2d"]["status"] == "extended"
    assert contract["dimensions"]["3d"]["status"] == "extended"
    justification = contract["dimensions"]["2d"]["justification"]
    assert "1-d cold Langmuir" in justification
    assert "extended" in justification
    entry = catalog_entry("CP-02")
    assert entry.dimension_codes == ("R", "E", "E")
    assert catalog_entry("TR-01").dimension_codes == ("R", "R", "R")
    assert catalog_entry("EU-02").dimension_codes == ("N/A", "R", "E")


def test_temporal_final_job_resolves_from_series_json_dt_star(tmp_path: Path):
    analyze = load_sibling_module(CP02_ANALYZE)
    series = tmp_path / "temporal"
    series.mkdir()
    jobs = ["dt0", "dt1", "dt2", "dt3"]
    (series / "series.json").write_text(
        json.dumps({"case_id": "CP-02", "jobs": jobs}) + "\n",
        encoding="utf-8",
    )
    for name in jobs:
        (series / name).mkdir()
        (series / name / "resolved_case.json").write_text(
            json.dumps({"case": {"id": "CP-02"}, "job": {"dt": 0.1}, "family": "temporal"})
            + "\n",
            encoding="utf-8",
        )
    assert not (series / "n256").exists()
    resolved = analyze.final_campaign_job(series)
    assert Path(resolved) == series / "dt3"
    source = CP02_ANALYZE.read_text(encoding="utf-8")
    assert "final_campaign_job" in source
    assert 'f"n{campaign' not in source


def test_parent_runner_mpi_writer_and_live_semantics_preserved():
    runner = (REPO_ROOT / "scripts" / "run_verification.py").read_text(encoding="utf-8")
    assert "def is_campaign_writer_rank" in runner
    assert "is_native_writer_rank" in runner
    live = REPO_ROOT / "verification" / "pops_verify" / "visualization" / "live.py"
    assert live.is_file()
    live_src = live.read_text(encoding="utf-8")
    assert "np.load" not in live_src
    assert "result.npy" not in live_src
    from verification.pops_verify.native_evidence import campaign_run_fields, maybe_campaign_payload

    assert "comparison" in campaign_run_fields.__code__.co_varnames
    assert callable(maybe_campaign_payload)
    assert (REPO_ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "plot_eu02.py").is_file()
    assert (REPO_ROOT / "verification" / "machines" / "run_eu02_v15.py").is_file()
    mpi_world = (REPO_ROOT / "verification" / "pops_verify" / "mpi_world.py").read_text(
        encoding="utf-8"
    )
    assert "def is_native_writer_rank" in mpi_world
    assert "native_world_size" in mpi_world


def test_accepted_romeo_evidence_sha_is_e78e310ad():
    assert EVIDENCE_SHA.startswith("e78e310ad")
    report = REPO_ROOT / ".superpowers" / "sdd" / "v15-case-cp02-1d-report.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        assert "e78e310ad" in text
