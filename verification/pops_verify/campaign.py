"""Expand a verification campaign into single-dimension jobs.

One native module compiles exactly one spatial dimension. When an artifact
dimension is known, every planned job must match it; there is no fallback to
another native extension.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

ALLOWED_DIMENSIONS = (1, 2, 3)
MAX_NODES_LIMIT = 2
ALLOWED_EXECUTION_SPACES = ("KokkosSerial", "KokkosOpenMP", "KokkosCuda")
ALLOWED_MPI_MODES = ("off", "on")
ALLOWED_STATUSES = ("pass", "fail", "not-supported", "not-run")


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CampaignResources:
    """Suite-scoped nodes, ranks, threads, and resolutions from the manifest."""

    nodes: int = 1
    mpi_ranks: int = 1
    omp_threads: int = 1
    resolutions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignJob:
    """One planned run of a case at a single native spatial dimension."""

    case_id: str
    pops_native_dim: int
    suite: str = "pr"
    execution_space: str = "KokkosSerial"
    mpi_mode: str = "off"
    min_resolution: int | None = None
    resources: CampaignResources = CampaignResources()
    evidence_status: str = "required"


@dataclass(frozen=True, slots=True)
class CampaignRequest:
    """Manifest parameters passed to a case ``run_native``."""

    case_id: str
    pops_native_dim: int
    suite: str
    execution_space: str
    mpi_mode: str
    min_resolution: int | None
    resources: CampaignResources
    evidence_status: str
    output_dir: Path | None = None

    @classmethod
    def from_job(
        cls,
        job: CampaignJob,
        output_dir: Path | None = None,
    ) -> CampaignRequest:
        return cls(
            case_id=job.case_id,
            pops_native_dim=job.pops_native_dim,
            suite=job.suite,
            execution_space=job.execution_space,
            mpi_mode=job.mpi_mode,
            min_resolution=job.min_resolution,
            resources=job.resources,
            evidence_status=job.evidence_status,
            output_dir=output_dir,
        )


def job_to_dict(job: CampaignJob) -> dict[str, Any]:
    return {
        "case_id": job.case_id,
        "pops_native_dim": job.pops_native_dim,
        "suite": job.suite,
        "execution_space": job.execution_space,
        "mpi_mode": job.mpi_mode,
        "min_resolution": job.min_resolution,
        "evidence_status": job.evidence_status,
        "resources": {
            "nodes": job.resources.nodes,
            "mpi_ranks": job.resources.mpi_ranks,
            "omp_threads": job.resources.omp_threads,
            "resolutions": list(job.resources.resolutions),
        },
    }


def resolve_artifact_dim(
    cli_value: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Return the known artifact dim, if any.

    ``--pops-native-dim`` overrides ``POPS_NATIVE_DIM``. Unset means local
    planning without a native build.
    """
    if cli_value is not None:
        if cli_value not in ALLOWED_DIMENSIONS:
            raise CampaignError(
                f"invalid --pops-native-dim {cli_value} (expected 1, 2, or 3)"
            )
        return cli_value
    env = os.environ if environ is None else environ
    raw = env.get("POPS_NATIVE_DIM")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise CampaignError(
            f"invalid POPS_NATIVE_DIM {raw!r} (expected 1, 2, or 3)"
        ) from exc
    if value not in ALLOWED_DIMENSIONS:
        raise CampaignError(
            f"invalid POPS_NATIVE_DIM {raw!r} (expected 1, 2, or 3)"
        )
    return value


def _nodes_value(raw: Any) -> int:
    if isinstance(raw, list):
        return int(raw[0]) if raw else 1
    if raw is None:
        return 1
    return int(raw)


def resources_for(case: Mapping[str, Any], suite: str) -> CampaignResources:
    tables = case.get("resources") or {}
    raw = tables.get(suite) or tables.get("pr") or {}
    resolutions = tuple(int(item) for item in (raw.get("resolutions") or ()))
    return CampaignResources(
        nodes=_nodes_value(raw.get("nodes")),
        mpi_ranks=int(raw.get("mpi_ranks") or 1),
        omp_threads=int(raw.get("omp_threads") or 1),
        resolutions=resolutions,
    )


def expand_jobs(
    cases: list[dict],
    dimensions: list[int],
    artifact_dim: int | None = None,
    *,
    suite: str = "pr",
    execution_space: str | None = None,
    mpi_mode: str | None = None,
) -> list[CampaignJob]:
    """Expand selected cases × requested dimensions into single-dimension jobs.

    A job is emitted only when the dimension is in the case's
    ``native_dimensions``. If ``artifact_dim`` is set, every requested
    dimension must equal it; mismatched requests are refused rather than
    dropped.
    """
    if artifact_dim is not None:
        if artifact_dim not in ALLOWED_DIMENSIONS:
            raise CampaignError(
                f"invalid POPS_NATIVE_DIM {artifact_dim} (expected 1, 2, or 3)"
            )
        if any(dim != artifact_dim for dim in dimensions):
            requested = ",".join(str(dim) for dim in dimensions)
            raise CampaignError(
                f"requested dimensions {requested} do not match "
                f"POPS_NATIVE_DIM={artifact_dim}; "
                "no fallback to another native extension"
            )
    if execution_space is not None and execution_space not in ALLOWED_EXECUTION_SPACES:
        raise CampaignError(
            f"invalid execution space {execution_space!r} "
            f"(expected {', '.join(ALLOWED_EXECUTION_SPACES)})"
        )
    if mpi_mode is not None and mpi_mode not in ALLOWED_MPI_MODES:
        raise CampaignError(
            f"invalid mpi mode {mpi_mode!r} (expected {', '.join(ALLOWED_MPI_MODES)})"
        )

    jobs: list[CampaignJob] = []
    requested = set(dimensions)
    for case in cases:
        native = set(case.get("native_dimensions", []))
        spaces = list(case.get("execution_spaces") or ["KokkosSerial"])
        modes = list(case.get("mpi_modes") or ["off"])
        if execution_space is not None and execution_space not in spaces:
            continue
        if mpi_mode is not None and mpi_mode not in modes:
            continue
        space = execution_space or spaces[0]
        mode = mpi_mode or modes[0]
        resources = resources_for(case, suite)
        min_resolution = resources.resolutions[0] if resources.resolutions else None
        for dim in ALLOWED_DIMENSIONS:
            if dim in requested and dim in native:
                jobs.append(
                    CampaignJob(
                        case_id=case["id"],
                        pops_native_dim=dim,
                        suite=suite,
                        execution_space=space,
                        mpi_mode=mode,
                        min_resolution=min_resolution,
                        resources=resources,
                        evidence_status=str(case.get("evidence_status") or "required"),
                    )
                )
    jobs.sort(key=lambda job: (job.case_id, job.pops_native_dim))
    return jobs
