"""IF-08 planner guard. Refuse a mismatched dim before any fake/native run.

Reuses expand_jobs / resolve_artifact_dim. Optional ``run_doctor`` wraps
``pops.doctor()``. A Dim2 case under ``POPS_NATIVE_DIM=1`` raises
``NativeUnavailable`` before a fake or GE-03 native run.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from verification.pops_verify.campaign import expand_jobs, resolve_artifact_dim
from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_GE03_RUN = (
    Path(__file__).resolve().parents[2] / "geometry" / "radial_acoustic" / "run.py"
)

_FAKE_RUNS: list[list] = []


class NativeUnavailable(RuntimeError):
    """Raised when a requested native dim cannot be served by this artifact."""


def reset_fake_runs() -> None:
    """Clear the in-memory fake-run log (tests and analyze)."""
    _FAKE_RUNS.clear()


def fake_run_count() -> int:
    """Return how many fake TR-01 / Dim2 runs started after a successful plan."""
    return len(_FAKE_RUNS)


def plan_tr01_jobs(
    requested_dimensions,
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Expand TR-01 at the resolved artifact dim, or refuse a mismatch."""
    resolved = resolve_artifact_dim(cli_value=artifact_dim, environ=environ)
    return expand_jobs(
        [_exact.tr01_case()],
        list(requested_dimensions),
        artifact_dim=resolved,
    )


def fake_run_tr01(
    requested_dimensions,
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Plan first; start the fake run only after the dim guard accepts."""
    jobs = plan_tr01_jobs(
        requested_dimensions, artifact_dim=artifact_dim, environ=environ
    )
    _FAKE_RUNS.append(list(jobs))
    return {"jobs": jobs, "ran": True}


def require_native_dim(
    required_dim: int,
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Refuse before any fake/native run when the artifact dim does not match."""
    resolved = resolve_artifact_dim(cli_value=artifact_dim, environ=environ)
    if resolved != int(required_dim):
        raise NativeUnavailable(
            f"POPS_NATIVE_DIM={resolved!r} does not match required dim "
            f"{int(required_dim)}; no fallback to another native extension"
        )
    return resolved


def present_dim2_case(
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Accept a Dim2 case only when the artifact is dim 2."""
    resolved = require_native_dim(
        _exact.DIM2_REQUIRED, artifact_dim=artifact_dim, environ=environ
    )
    return expand_jobs(
        [_exact.dim2_case()],
        [_exact.DIM2_REQUIRED],
        artifact_dim=resolved,
    )


def fake_run_dim2(
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Plan the Dim2 case first; never increment the fake-run log on mismatch."""
    jobs = present_dim2_case(artifact_dim=artifact_dim, environ=environ)
    _FAKE_RUNS.append(list(jobs))
    return {"jobs": jobs, "ran": True}


def run_doctor():
    """Wrap ``pops.doctor()``. Raises NativeUnavailable without a live doctor."""
    import pops
    from pops.runtime.doctor import doctor as _runtime_doctor

    doctor = pops.doctor if hasattr(pops, "doctor") else _runtime_doctor
    try:
        return doctor(verbose=False)
    except NativeUnavailable:
        raise
    except Exception as exc:
        raise NativeUnavailable(f"pops.doctor unavailable: {exc}") from exc


def run_native(n_cells: int = 8, t_end: float = 0.01, request=None):
    """Campaign entry: dim-1 jobs run TR-01; dim-2 jobs run the GE-03 path.

    Without ``request``, keep the historical Dim2 refuse path so existing
    IF-08 unit tests still exercise ``require_native_dim(2)``.
    """
    if request is not None:
        if request.min_resolution is not None:
            n_cells = int(request.min_resolution)
        required = int(request.pops_native_dim)
        if required == _exact.MATCHING_DIM:
            return run_native_dim1(n_cells, t_end=t_end)
        if required != _exact.DIM2_REQUIRED:
            raise NativeUnavailable(
                f"IF-08 has no campaign path for pops_native_dim={required}"
            )
    require_native_dim(_exact.DIM2_REQUIRED)
    ge03 = load_sibling_module(_GE03_RUN)
    try:
        return ge03.run_native(n_cells, t_end=t_end)
    except NativeUnavailable:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "NativeUnavailable":
            raise NativeUnavailable(str(exc)) from exc
        raise


def present_dim1_case(
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Accept TR-01 only when the artifact is dim 1."""
    resolved = require_native_dim(
        _exact.MATCHING_DIM, artifact_dim=artifact_dim, environ=environ
    )
    return expand_jobs(
        [_exact.tr01_case()],
        [_exact.MATCHING_DIM],
        artifact_dim=resolved,
    )


def run_native_dim1(n_cells: int = 16, t_end: float = 0.1):
    """Run TR-01 only after the artifact dim is 1. Dim2 artifacts refuse."""
    require_native_dim(_exact.MATCHING_DIM)
    tr01 = load_sibling_module(
        Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
    )
    try:
        return tr01.run_native(n_cells, t_end=t_end)
    except NativeUnavailable:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "NativeUnavailable":
            raise NativeUnavailable(str(exc)) from exc
        raise
