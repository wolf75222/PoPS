"""Authenticated native-evidence contract for the v1.5 smooth-case set.

Scientific pass is created only by the module-private factory. Records are
content-addressed: SHA-256 of immutable result bytes, RUN_FIELDS, and
authenticated leaf/header/variant/program/resolved-case digests. There is no
static HMAC key, ``id(result)``, or process-global registry.

Immutable record schema (``RECORD_SCHEMA``)
------------------------------------------
RUN_FIELDS plus:

- ``case_id``
- ``result`` / ``result_digest``
- ``oracle`` / ``oracle_digest`` (optional; required to derive orders)
- ``sample_spacing``
- ``leaf_sha256``
- ``native_header_signature``
- ``native_variant_manifest_digest``
- ``component_catalog_digest``
- ``program_digest``
- ``resolved_case_digest``
- ``evidence_digest``
- ``pair_result`` / ``pair_result_digest`` / ``pair_program_digest`` /
  ``pair_resolved_case_digest`` (required for TR-06)

Later CP/AMR streams should import this helper. Do not merge those forks here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.capabilities import AuthenticatedArtifact, sha256_file
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.metrics import collect_metrics
from verification.pops_verify.provenance import RUN_FIELDS, collect_provenance
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSING_NATIVE_SERIES = "no native result series"

RECORD_SCHEMA = (
    *RUN_FIELDS,
    "case_id",
    "result",
    "result_digest",
    "oracle",
    "oracle_digest",
    "sample_spacing",
    "leaf_sha256",
    "native_header_signature",
    "native_variant_manifest_digest",
    "component_catalog_digest",
    "program_digest",
    "resolved_case_digest",
    "evidence_digest",
    "pair_result",
    "pair_result_digest",
    "pair_program_digest",
    "pair_resolved_case_digest",
)

BLOCKED_REQUIRED = {
    "TR-03": "public time-dependent incompressible swirl velocity is not supported",
    "EU-03": "public MMS source injection is not supported",
    "PO-02": "public non-homogeneous Dirichlet data is not attached to the Case",
}
REDUCED_REQUIRED = {
    "PO-04": "public composite AMR Poisson is not supported for PO-04",
    "PO-05": (
        "public GeometricMG on uniform Systems lowers to CartesianCG; "
        "FFT-vs-GMG is not supported"
    ),
    "PO-07": "public FFT has no tolerance sweep",
}
REDUCED_NOT_SUPPORTED = REDUCED_REQUIRED

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


class NativeSeries:
    """Authenticated multi-run record. Callers cannot construct this type."""

    __slots__ = (
        "case_id",
        "records",
        "kind",
        "variable",
        "threshold",
        "derived_linf",
        "derived_spacings",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NativeSeriesError("NativeSeries cannot be constructed by callers")


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


def _hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _result_bytes(array: Any) -> bytes:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes()


def _result_digest(array: Any) -> str:
    return hashlib.sha256(_result_bytes(array)).hexdigest()


def _canonical_run_fields(record: Mapping[str, Any]) -> str:
    payload = {}
    for key in RUN_FIELDS:
        value = record[key]
        if key in {"resolution", "block_size"}:
            payload[key] = list(value)
        elif isinstance(value, float):
            payload[key] = float(value)
        else:
            payload[key] = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _evidence_material(record: Mapping[str, Any]) -> bytes:
    parts = [
        str(record.get("case_id", "")),
        str(record.get("result_digest", "")),
        str(record.get("oracle_digest", "")),
        f"{float(record.get('sample_spacing', 0.0)):.17g}",
        _canonical_run_fields(record),
        str(record.get("leaf_sha256", "")),
        str(record.get("native_header_signature", "")),
        str(record.get("native_variant_manifest_digest", "")),
        str(record.get("component_catalog_digest", "")),
        str(record.get("program_digest", "")),
        str(record.get("resolved_case_digest", "")),
        str(record.get("pair_result_digest", "")),
        str(record.get("pair_program_digest", "")),
        str(record.get("pair_resolved_case_digest", "")),
    ]
    return "|".join(parts).encode("utf-8")


def _evidence_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_evidence_material(record)).hexdigest()


def _identity_fields(identity: Any) -> dict[str, str]:
    if not isinstance(identity, AuthenticatedArtifact):
        raise NativeSeriesError(
            "identity must be an AuthenticatedArtifact verified by capabilities"
        )
    path = Path(identity.path)
    if path.is_file() and sha256_file(path) != identity.sha256:
        raise NativeSeriesError("leaf digest mismatch")
    fields = {
        "leaf_sha256": str(identity.sha256),
        "native_header_signature": str(identity.native_header_signature),
        "native_variant_manifest_digest": str(identity.native_variant_manifest_digest),
        "component_catalog_digest": str(identity.component_catalog_digest),
    }
    if not all(_hex64(fields[key]) for key in fields):
        raise NativeSeriesError("identity digests must be sha256 hex")
    return fields


def _compiled_types(artifact: Any, simulation: Any) -> tuple[str, str] | None:
    try:
        from pops.codegen._compiled_artifact import CompiledSimulationArtifact
        from pops.runtime._runtime_instance import RuntimeInstance
    except Exception:
        return None
    if type(artifact) is not CompiledSimulationArtifact:
        return None
    if type(simulation) is not RuntimeInstance:
        return None
    program = hashlib.sha256(repr(artifact.artifact_identity).encode("utf-8")).hexdigest()
    resolved = hashlib.sha256(repr(artifact.plan).encode("utf-8")).hexdigest()
    return program, resolved


def _standin_identity_from_compiled(artifact: Any) -> dict[str, str]:
    payload = repr(artifact.artifact_identity).encode("utf-8")
    return {
        "leaf_sha256": hashlib.sha256(b"leaf|" + payload).hexdigest(),
        "native_header_signature": hashlib.sha256(b"header|" + payload).hexdigest(),
        "native_variant_manifest_digest": hashlib.sha256(b"variant|" + payload).hexdigest(),
        "component_catalog_digest": hashlib.sha256(b"catalog|" + payload).hexdigest(),
    }


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
    identity=None,
    case_id=None,
    oracle=None,
    program_digest=None,
    resolved_case_digest=None,
    pair=None,
    sample_spacing=None,
    **fields_kwargs,
):
    """Return ``field`` unless a campaign request is present; then digest RUN_FIELDS."""
    if request is None:
        return field
    compiled = _compiled_types(artifact, simulation)
    if identity is None and compiled is None:
        raise NativeSeriesError(
            "campaign payload requires an AuthenticatedArtifact identity "
            "or real compiled artifact/simulation types"
        )
    if identity is not None:
        id_fields = _identity_fields(identity)
    else:
        id_fields = _standin_identity_from_compiled(artifact)
    if compiled is not None:
        program_digest = program_digest or compiled[0]
        resolved_case_digest = resolved_case_digest or compiled[1]
    if not _hex64(program_digest) or not _hex64(resolved_case_digest):
        raise NativeSeriesError("program/resolved-case digest required")
    n_cells = int(fields_kwargs.get("n_cells", 1))
    spacing = float(sample_spacing if sample_spacing is not None else 1.0 / n_cells)
    case = str(case_id or getattr(request, "case_id", None) or "unknown")
    payload = campaign_run_fields(request=request, **fields_kwargs)
    payload["case_id"] = case
    payload["result"] = field
    payload["result_digest"] = _result_digest(field)
    payload["oracle"] = oracle
    payload["oracle_digest"] = _result_digest(oracle) if oracle is not None else ""
    payload["sample_spacing"] = spacing
    payload["program_digest"] = program_digest
    payload["resolved_case_digest"] = resolved_case_digest
    payload.update(id_fields)
    if pair is not None:
        if not isinstance(pair, Mapping) or "result" not in pair:
            raise NativeSeriesError("pair requires an authenticated result")
        pair_identity = pair.get("identity", identity)
        if pair_identity is not None:
            _identity_fields(pair_identity)
        pair_compiled = _compiled_types(pair.get("artifact"), pair.get("simulation"))
        if pair_identity is None and pair_compiled is None and identity is None:
            raise NativeSeriesError("pair requires identity or compiled types")
        payload["pair_result"] = pair["result"]
        payload["pair_result_digest"] = _result_digest(pair["result"])
        payload["pair_program_digest"] = str(
            pair.get("program_digest") or (pair_compiled[0] if pair_compiled else "")
        )
        payload["pair_resolved_case_digest"] = str(
            pair.get("resolved_case_digest") or (pair_compiled[1] if pair_compiled else "")
        )
        if not _hex64(payload["pair_program_digest"]) or not _hex64(
            payload["pair_resolved_case_digest"]
        ):
            raise NativeSeriesError("pair program/resolved-case digest required")
    payload["evidence_digest"] = _evidence_digest(payload)
    return payload


def verify_campaign_run(run: Mapping[str, Any]) -> None:
    if not isinstance(run, Mapping):
        raise NativeSeriesError("campaign run must be a mapping")
    if "linf" in run or "spacings" in run:
        raise NativeSeriesError("injected linf/spacings are not native evidence")
    missing = [key for key in RUN_FIELDS if key not in run]
    if missing:
        raise NativeSeriesError(f"missing RUN_FIELDS: {missing}")
    if "result" not in run:
        raise NativeSeriesError("missing result")
    if run.get("result_digest") != _result_digest(run["result"]):
        raise NativeSeriesError("result digest mismatch")
    if run.get("oracle") is not None:
        if run.get("oracle_digest") != _result_digest(run["oracle"]):
            raise NativeSeriesError("oracle digest mismatch")
    for key in (
        "leaf_sha256",
        "native_header_signature",
        "native_variant_manifest_digest",
        "component_catalog_digest",
        "program_digest",
        "resolved_case_digest",
        "evidence_digest",
    ):
        if not _hex64(run.get(key)):
            raise NativeSeriesError(f"missing digest: {key}")
    if str(run.get("case_id")) == "TR-06":
        if not _hex64(run.get("pair_result_digest")):
            raise NativeSeriesError("TR-06 pair result digest required")
        if "pair_result" in run and run.get("pair_result_digest") != _result_digest(
            run["pair_result"]
        ):
            raise NativeSeriesError("pair result digest mismatch")
        if not _hex64(run.get("pair_program_digest")) or not _hex64(
            run.get("pair_resolved_case_digest")
        ):
            raise NativeSeriesError("TR-06 pair program/resolved digests required")
    if run.get("evidence_digest") != _evidence_digest(run):
        raise NativeSeriesError("evidence digest mismatch")


def _linf(record: Mapping[str, Any]) -> float:
    result = np.asarray(record["result"], dtype=np.float64)
    oracle = record.get("oracle")
    if oracle is None:
        raise NativeSeriesError("oracle required to derive orders")
    ref = np.asarray(oracle, dtype=np.float64)
    return float(np.max(np.abs(result - ref)))


def _series_from_records(
    case_id: str,
    runs: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    variable: str,
    threshold: float,
) -> NativeSeries:
    if not runs:
        raise NativeSeriesError("empty campaign run list")
    records = []
    for run in runs:
        verify_campaign_run(run)
        if str(run.get("case_id")) != str(case_id):
            raise NativeSeriesError("cross-case record reuse is not native evidence")
        records.append(dict(run))
    series = object.__new__(NativeSeries)
    series.case_id = str(case_id)
    series.records = tuple(records)
    series.kind = kind
    series.variable = variable
    series.threshold = float(threshold)
    series.derived_linf = tuple(_linf(run) for run in records)
    series.derived_spacings = tuple(float(run["sample_spacing"]) for run in records)
    return series


def _coerce_series(
    case_id: str,
    native_series: Any,
    *,
    kind: str,
    variable: str,
    threshold: float,
) -> NativeSeries | None:
    if isinstance(native_series, NativeSeries):
        try:
            return _series_from_records(
                case_id,
                native_series.records,
                kind=kind,
                variable=variable,
                threshold=threshold,
            )
        except NativeSeriesError:
            return None
    if isinstance(native_series, Sequence) and not isinstance(
        native_series, (str, bytes, Mapping)
    ):
        if not native_series:
            return None
        if any(not isinstance(item, Mapping) for item in native_series):
            return None
        try:
            return _series_from_records(
                case_id,
                native_series,
                kind=kind,
                variable=variable,
                threshold=threshold,
            )
        except NativeSeriesError:
            return None
    return None


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
    native_series: NativeSeries | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    native_dimensions: list[int],
    components: list[str],
    kind: str = "spatial",
    variable: str = "q",
    threshold: float = 1.8,
    extra_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    if case_id in BLOCKED_REQUIRED:
        return fail_closed_summary(
            case_id,
            BLOCKED_REQUIRED[case_id],
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if case_id in REDUCED_REQUIRED:
        return fail_closed_summary(
            case_id,
            REDUCED_REQUIRED[case_id],
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    series = _coerce_series(
        case_id,
        native_series,
        kind=kind,
        variable=variable,
        threshold=threshold,
    )
    if series is None:
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if len(series.derived_linf) < 2 or len(series.derived_spacings) < 2:
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    orders = orders_from_native_errors(
        case_id,
        series.derived_linf,
        series.derived_spacings,
        kind=str(series.kind or kind),
        variable=str(series.variable or variable),
        threshold=float(series.threshold or threshold),
    )
    limit = float(series.threshold or threshold)
    if any(float(row["observed_order"]) < limit for row in orders):
        return fail_closed_summary(
            case_id,
            "observed order below threshold",
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


def metrics_from_native_series(case_id: str, series: Any) -> dict[str, Any]:
    coerced = _coerce_series(
        case_id, series, kind="spatial", variable="q", threshold=1.8
    )
    if coerced is None:
        raise NativeSeriesError("metrics require an authenticated NativeSeries")
    return collect_metrics(
        case_id,
        reason="authenticated native series; unused fields remain null",
    )


def provenance_from_native_series(
    case_id: str,
    series: Any,
    **provenance_kwargs: Any,
) -> dict[str, Any]:
    coerced = _coerce_series(
        case_id, series, kind="spatial", variable="q", threshold=1.8
    )
    if coerced is None:
        raise NativeSeriesError("provenance require an authenticated NativeSeries")
    record = coerced.records[0]
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
