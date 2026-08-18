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
