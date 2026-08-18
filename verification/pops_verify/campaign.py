"""Expand a verification campaign into single-dimension jobs.

One native module compiles exactly one spatial dimension. When an artifact
dimension is known, every planned job must match it; there is no fallback to
another native extension.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

ALLOWED_DIMENSIONS = (1, 2, 3)
MAX_NODES_LIMIT = 2


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CampaignJob:
    """One planned run of a case at a single native spatial dimension."""

    case_id: str
    pops_native_dim: int


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


def expand_jobs(
    cases: list[dict],
    dimensions: list[int],
    artifact_dim: int | None = None,
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

    jobs: list[CampaignJob] = []
    requested = set(dimensions)
    for case in cases:
        native = set(case.get("native_dimensions", []))
        for dim in ALLOWED_DIMENSIONS:
            if dim in requested and dim in native:
                jobs.append(
                    CampaignJob(case_id=case["id"], pops_native_dim=dim)
                )
    jobs.sort(key=lambda job: (job.case_id, job.pops_native_dim))
    return jobs
