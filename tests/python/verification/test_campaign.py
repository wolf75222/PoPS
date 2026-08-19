"""Single-dimension campaign jobs and exact POPS_NATIVE_DIM matching."""
from __future__ import annotations

import pytest

from verification.pops_verify.campaign import (
    CampaignError,
    CampaignJob,
    expand_jobs,
    resolve_artifact_dim,
)

CASE = {"id": "CP-02", "native_dimensions": [1, 2]}


def test_expand_jobs_without_artifact_dim_emits_requested_intersection():
    jobs = expand_jobs([CASE], [1, 2], artifact_dim=None)
    assert jobs == [
        CampaignJob(case_id="CP-02", pops_native_dim=1),
        CampaignJob(case_id="CP-02", pops_native_dim=2),
    ]


def test_expand_jobs_matching_artifact_dim_emits_one_job():
    jobs = expand_jobs([CASE], [1], artifact_dim=1)
    assert jobs == [CampaignJob(case_id="CP-02", pops_native_dim=1)]


def test_expand_jobs_refuses_requested_dim_that_differs_from_artifact():
    with pytest.raises(CampaignError, match="POPS_NATIVE_DIM") as exc_info:
        expand_jobs([CASE], [1, 2], artifact_dim=1)
    assert "fallback" in str(exc_info.value).lower()


def test_expand_jobs_empty_cases_yields_no_jobs():
    assert expand_jobs([], [1, 2], artifact_dim=None) == []


def test_expand_jobs_stable_order_is_case_id_then_dim():
    cases = [
        {"id": "CP-02", "native_dimensions": [1, 2]},
        {"id": "CP-01", "native_dimensions": [2]},
    ]
    jobs = expand_jobs(cases, [2, 1], artifact_dim=None)
    assert [(job.case_id, job.pops_native_dim) for job in jobs] == [
        ("CP-01", 2),
        ("CP-02", 1),
        ("CP-02", 2),
    ]


def test_resolve_artifact_dim_cli_overrides_env():
    assert resolve_artifact_dim(cli_value=2, environ={"POPS_NATIVE_DIM": "1"}) == 2
    assert resolve_artifact_dim(cli_value=None, environ={"POPS_NATIVE_DIM": "1"}) == 1
    assert resolve_artifact_dim(cli_value=None, environ={}) is None


RICH_IF01 = {
    "id": "IF-01",
    "native_dimensions": [1],
    "execution_spaces": ["KokkosSerial"],
    "mpi_modes": ["on"],
    "evidence_status": "required",
    "requires": ["mpi"],
    "resources": {
        "pr": {
            "nodes": 1,
            "mpi_ranks": 2,
            "omp_threads": 1,
            "resolutions": [32, 64],
        }
    },
}


def test_expand_jobs_parameterizes_suite_space_mpi_resources_and_evidence():
    from verification.pops_verify.campaign import CampaignRequest

    jobs = expand_jobs(
        [RICH_IF01],
        [1],
        artifact_dim=None,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="on",
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert job.case_id == "IF-01"
    assert job.pops_native_dim == 1
    assert job.suite == "pr"
    assert job.execution_space == "KokkosSerial"
    assert job.mpi_mode == "on"
    assert job.min_resolution == 32
    assert job.evidence_status == "required"
    assert job.resources.nodes == 1
    assert job.resources.mpi_ranks == 2
    assert job.resources.resolutions == (32, 64)
    request = CampaignRequest.from_job(job)
    assert request.mpi_mode == "on"
    assert request.min_resolution == 32


IN_MEMORY_HELPERS = frozenset({"PH-00", "NO-01"})


def test_catalogued_cases_expose_run_native_except_in_memory_helpers():
    """Required and capability-gated cases have run_native; PH-00/NO-01 stay in-memory."""
    import tomllib
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    manifest = tomllib.loads((repo / "verification" / "manifest.toml").read_text(encoding="utf-8"))
    missing: list[str] = []
    unexpected_helpers: list[str] = []
    for case in manifest["case"]:
        source = (repo / case["path"]).read_text(encoding="utf-8")
        has_runner = "def run_native" in source
        if case["id"] in IN_MEMORY_HELPERS:
            if has_runner:
                unexpected_helpers.append(case["id"])
            continue
        if not has_runner:
            missing.append(case["id"])
    assert unexpected_helpers == []
    assert missing == []
