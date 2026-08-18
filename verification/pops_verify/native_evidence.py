"""Authenticated native-evidence contract for the v1.5 smooth-case set.

A report can pass only from a ``NativeSeries`` built from sealed
``run_native`` campaign payloads. Arbitrary dicts, injected L∞/spacings,
and reduced/blocked IDs never yield ``cases_passed=1``.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verification.pops_verify.convergence import observed_order
from verification.pops_verify.metrics import collect_metrics
from verification.pops_verify.provenance import RUN_FIELDS, collect_provenance
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSING_NATIVE_SERIES = "no native result series"
_SEAL_KEY = b"pops.native-series.v1"
_REGISTERED_BINDS: dict[str, str] = {}

BLOCKED_REQUIRED = {
    "TR-03": "public time-dependent incompressible swirl velocity is not supported",
    "EU-03": "public MMS source injection is not supported",
    "PO-02": "public non-homogeneous Dirichlet data is not attached to the Case",
}
REDUCED_NOT_SUPPORTED = {
    "PO-04": "public composite AMR Poisson is not supported for PO-04",
    "PO-05": (
        "public GeometricMG on uniform Systems lowers to CartesianCG; "
        "FFT-vs-GMG is not supported"
    ),
    "PO-07": "public FFT has no tolerance sweep",
}

NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
}
NULL_POISSON = {
    "potential_error": None,
    "field_error": None,
    "residual_l2": None,
}
NULL_COUPLING = {
    "phase_error": None,
    "sign_ok": None,
    "energy_drift": None,
}
NULL_PARALLEL = {
    "ranks_ok": None,
    "threads_ok": None,
    "gpu_ok": None,
}
NULL_PERFORMANCE = {
    "one_node": None,
    "two_node": None,
}


class NativeSeriesError(ValueError):
    """Raised when a caller tries to forge or inject native evidence."""


def repository_sha(repo: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else "unknown"


def resolution_from_request(request, default: int) -> int:
    if request is not None and getattr(request, "min_resolution", None) is not None:
        return int(request.min_resolution)
    return int(default)


def require_native_dim(request, required: int, case_id: str) -> None:
    if request is None:
        return
    got = int(request.pops_native_dim)
    if got != int(required):
        raise RuntimeError(
            f"{case_id} requires pops_native_dim={required} (got {got}); no fallback"
        )


def register_native_identities(artifact: Any, simulation: Any) -> tuple[str, str]:
    """Record compile/bind identities minted after a real pops.compile/bind."""
    compile_id = f"pops.compile:{id(artifact)}:{type(artifact).__name__}"
    bind_id = f"pops.bind:{id(simulation)}:{type(simulation).__name__}"
    _REGISTERED_BINDS[compile_id] = bind_id
    return compile_id, bind_id


def _seal_material(payload: Mapping[str, Any]) -> bytes:
    return "|".join(
        (
            str(payload.get("compile_identity", "")),
            str(payload.get("bind_identity", "")),
            str(id(payload.get("result"))),
        )
    ).encode("utf-8")


def _seal(payload: Mapping[str, Any]) -> str:
    return hmac.new(_SEAL_KEY, _seal_material(payload), hashlib.sha256).hexdigest()


def campaign_run_fields(
    *,
    request,
    n_cells: int,
    t_end: float,
    time_program: str,
    cfl: float,
    dimension: int = 1,
    **overrides: Any,
) -> dict[str, Any]:
    """Truthful campaign facts from the request and environment. No invented MPI/build."""
    space = getattr(request, "execution_space", None) or "unknown"
    mpi_mode = getattr(request, "mpi_mode", "off")
    mpi_on = mpi_mode == "on"
    resources = getattr(request, "resources", None)
    count = int(n_cells)
    dim = int(dimension)
    fields: dict[str, Any] = {
        "compiler": os.environ.get("CXX") or "unknown",
        "build_type": os.environ.get("CMAKE_BUILD_TYPE") or "unknown",
        "precision": "float64",
        "kokkos_execution_space": space,
        "mpi_enabled": bool(mpi_on),
        "mpi_library": "unknown" if mpi_on else "none",
        "mpi_thread_level_requested": "unknown" if mpi_on else "none",
        "mpi_thread_level_provided": "unknown" if mpi_on else "none",
        "hdf5_collective_enabled": False,
        "mpi_ranks": int(getattr(resources, "mpi_ranks", None) or 1),
        "omp_threads_per_rank": int(getattr(resources, "omp_threads", None) or 1),
        "gpus": 0,
        "resolution": [count] * dim,
        "block_size": [count] * dim,
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": time_program,
        "cfl": float(cfl),
        "final_time": float(t_end),
    }
    fields.update(overrides)
    return fields


def maybe_campaign_payload(
    request,
    field,
    *,
    artifact=None,
    simulation=None,
    **fields_kwargs,
):
    """Return ``field`` unless a campaign request is present; then seal RUN_FIELDS."""
    if request is None:
        return field
    if artifact is None or simulation is None:
        raise NativeSeriesError("campaign payload requires compile/bind identities")
    payload = campaign_run_fields(request=request, **fields_kwargs)
    compile_id, bind_id = register_native_identities(artifact, simulation)
    payload["result"] = field
    payload["compile_identity"] = compile_id
    payload["bind_identity"] = bind_id
    payload["native_seal"] = _seal(payload)
    return payload


def verify_campaign_run(run: Mapping[str, Any]) -> None:
    if not isinstance(run, Mapping):
        raise NativeSeriesError("campaign run must be a mapping")
    if "linf" in run or "spacings" in run:
        raise NativeSeriesError("injected linf/spacings are not native evidence")
    compile_id = run.get("compile_identity")
    bind_id = run.get("bind_identity")
    if not compile_id or _REGISTERED_BINDS.get(str(compile_id)) != bind_id:
        raise NativeSeriesError("unregistered compile/bind identity")
    if run.get("native_seal") != _seal(run):
        raise NativeSeriesError("invalid native seal")
    missing = [key for key in RUN_FIELDS if key not in run]
    if missing:
        raise NativeSeriesError(f"missing RUN_FIELDS: {missing}")
    if "result" not in run:
        raise NativeSeriesError("missing result")


@dataclass(frozen=True)
class NativeSeries:
    """Authenticated multi-run record. Construct only via ``from_campaign_runs``."""

    case_id: str
    records: tuple[Mapping[str, Any], ...]
    kind: str = "spatial"
    variable: str = "q"
    threshold: float = 1.8
    derived_linf: tuple[float, ...] = ()
    derived_spacings: tuple[float, ...] = ()

    @classmethod
    def from_campaign_runs(
        cls,
        case_id: str,
        runs: Sequence[Mapping[str, Any]],
        *,
        error_fn: Callable[[Mapping[str, Any]], float] | None = None,
        spacing_fn: Callable[[Mapping[str, Any]], float] | None = None,
        kind: str = "spatial",
        variable: str = "q",
        threshold: float = 1.8,
    ) -> NativeSeries:
        if not runs:
            raise NativeSeriesError("empty campaign run list")
        records = []
        for run in runs:
            verify_campaign_run(run)
            records.append(dict(run))
        linf = (
            tuple(float(error_fn(run)) for run in records) if error_fn is not None else ()
        )
        spacings = (
            tuple(float(spacing_fn(run)) for run in records)
            if spacing_fn is not None
            else ()
        )
        return cls(
            case_id=str(case_id),
            records=tuple(records),
            kind=kind,
            variable=variable,
            threshold=float(threshold),
            derived_linf=linf,
            derived_spacings=spacings,
        )


def orders_from_native_errors(
    case_id: str,
    errors,
    spacings,
    *,
    kind: str = "spatial",
    variable: str = "q",
    threshold: float = 1.8,
) -> list[dict[str, Any]]:
    observed = observed_order(errors, spacings)
    return [
        {
            "case_id": case_id,
            "kind": kind,
            "variable": variable,
            "observed_order": float(value),
            "threshold": float(threshold),
        }
        for value in observed
    ]


def _base_reasons(case_id: str) -> dict[str, str]:
    return {
        "amr.*": f"AMR not evidenced for {case_id} without a native AMR series",
        "poisson.*": f"Poisson not evidenced for {case_id} without a native elliptic series",
        "coupling.*": f"coupling not evidenced for {case_id}",
        "parallel_invariance.*": f"parallel invariance not evidenced for {case_id}",
        "performance.one_node": f"performance not measured for {case_id}",
        "performance.two_node": f"performance not measured for {case_id}",
    }


def _coverage(*, passed: int, failed: int, not_supported: int, components: list[str]):
    return {
        "components": list(components),
        "cases_planned": 1,
        "cases_run": 1,
        "cases_passed": int(passed),
        "cases_failed": int(failed),
        "cases_not_supported": int(not_supported),
        "not_tested": [],
    }


def fail_closed_summary(
    case_id: str,
    reason: str,
    *,
    native_dimensions: list[int],
    components: list[str],
    extra_reasons: dict[str, str] | None = None,
    suite: str = "pr",
    poisson: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons = _base_reasons(case_id)
    reasons["orders"] = reason
    if extra_reasons:
        reasons.update(extra_reasons)
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": repository_sha(),
        "suite": suite,
        "max_nodes": 2,
        "native_dimensions": list(native_dimensions),
        "execution_spaces": ["KokkosSerial"],
        "coverage": _coverage(passed=0, failed=1, not_supported=0, components=components),
        "failures": [
            {
                "case_id": case_id,
                "reason": reason,
                "metrics_ref": "",
                "provenance_ref": "",
            }
        ],
        "orders": [],
        "amr": dict(NULL_AMR),
        "poisson": dict(poisson or NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": reasons,
        "artifacts": dict(ARTIFACTS),
    }


def not_supported_summary(
    case_id: str,
    reason: str,
    *,
    native_dimensions: list[int],
    components: list[str],
    extra_reasons: dict[str, str] | None = None,
    suite: str = "pr",
) -> dict[str, Any]:
    reasons = _base_reasons(case_id)
    reasons["orders"] = reason
    if extra_reasons:
        reasons.update(extra_reasons)
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": repository_sha(),
        "suite": suite,
        "max_nodes": 2,
        "native_dimensions": list(native_dimensions),
        "execution_spaces": ["KokkosSerial"],
        "coverage": _coverage(passed=0, failed=0, not_supported=1, components=components),
        "failures": [],
        "orders": [],
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": reasons,
        "artifacts": dict(ARTIFACTS),
    }


def passed_summary(
    case_id: str,
    *,
    native_dimensions: list[int],
    components: list[str],
    orders: list[dict[str, Any]],
    extra_reasons: dict[str, str] | None = None,
    poisson: dict[str, Any] | None = None,
    suite: str = "pr",
) -> dict[str, Any]:
    reasons = _base_reasons(case_id)
    if not orders:
        reasons["orders"] = (
            f"native series recorded without an order campaign for {case_id}"
        )
    if extra_reasons:
        reasons.update(extra_reasons)
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": repository_sha(),
        "suite": suite,
        "max_nodes": 2,
        "native_dimensions": list(native_dimensions),
        "execution_spaces": ["KokkosSerial"],
        "coverage": _coverage(passed=1, failed=0, not_supported=0, components=components),
        "failures": [],
        "orders": list(orders),
        "amr": dict(NULL_AMR),
        "poisson": dict(poisson or NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": reasons,
        "artifacts": dict(ARTIFACTS),
    }


def report_from_native_series(
    case_id: str,
    native_series: NativeSeries | Mapping[str, Any] | None,
    *,
    native_dimensions: list[int],
    components: list[str],
    kind: str = "spatial",
    variable: str = "q",
    threshold: float = 1.8,
    extra_reasons: dict[str, str] | None = None,
    allow_empty_orders: bool = False,
) -> dict[str, Any]:
    if case_id in BLOCKED_REQUIRED:
        return fail_closed_summary(
            case_id,
            BLOCKED_REQUIRED[case_id],
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if case_id in REDUCED_NOT_SUPPORTED:
        return not_supported_summary(
            case_id,
            REDUCED_NOT_SUPPORTED[case_id],
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if native_series is None or not isinstance(native_series, NativeSeries):
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if native_series.case_id != case_id:
        return fail_closed_summary(
            case_id,
            f"series case_id {native_series.case_id!r} does not match {case_id}",
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    orders: list[dict[str, Any]] = []
    if native_series.derived_linf and native_series.derived_spacings:
        orders = orders_from_native_errors(
            case_id,
            native_series.derived_linf,
            native_series.derived_spacings,
            kind=str(native_series.kind or kind),
            variable=str(native_series.variable or variable),
            threshold=float(native_series.threshold or threshold),
        )
        limit = float(native_series.threshold or threshold)
        if any(float(row["observed_order"]) < limit for row in orders):
            return fail_closed_summary(
                case_id,
                "observed order below threshold",
                native_dimensions=native_dimensions,
                components=components,
                extra_reasons=extra_reasons,
            )
    elif not allow_empty_orders:
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    return passed_summary(
        case_id,
        native_dimensions=native_dimensions,
        components=components,
        orders=orders,
        extra_reasons=extra_reasons,
    )


def metrics_from_native_series(case_id: str, series: NativeSeries) -> dict[str, Any]:
    if not isinstance(series, NativeSeries):
        raise NativeSeriesError("metrics require an authenticated NativeSeries")
    return collect_metrics(
        case_id,
        reason="authenticated native series; unused fields remain null",
    )


def provenance_from_native_series(
    case_id: str,
    series: NativeSeries,
    **provenance_kwargs: Any,
) -> dict[str, Any]:
    if not isinstance(series, NativeSeries):
        raise NativeSeriesError("provenance require an authenticated NativeSeries")
    record = series.records[0]
    dim = len(record.get("resolution") or [1])
    kwargs = dict(provenance_kwargs)
    kwargs.setdefault("doctor_ok", False)
    return collect_provenance(
        case_id,
        pops_native_dim=dim,
        dimension=dim,
        nodes=1,
        run={key: record[key] for key in RUN_FIELDS if key in record},
        **kwargs,
    )
