"""Fail-closed reports over an EvidenceBundle trust root.

``run_native`` returns a raw truthful payload only. Scientific pass is minted
only after ``EvidenceBundle`` loads a completed runner job directory.
"""
from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.capabilities import (
    authenticate_installed_artifact,
    resolve_variants_root,
)
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError
from verification.pops_verify.phase import frequency_error, numerical_frequency, phase_error
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.metrics import collect_metrics
from verification.pops_verify.provenance import RUN_FIELDS, collect_provenance
from verification.pops_verify.report import ARTIFACTS

PROGRAM_BYTES_BLOCKER = (
    "public CompiledSimulationArtifact.so_path program bytes are required; "
    "object stand-ins are not accepted"
)
EVIDENCE_PAYLOAD_KEYS = (
    "result",
    "pair_result",
    "program_bytes",
    "pair_program_bytes",
    "dt",
    "case_id",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSING_NATIVE_SERIES = "no native result series"
NO_NATIVE_REASON = (
    "no native Kokkos output; exact-vs-exact and synthetic identities "
    "are not scientific evidence"
)

RECORD_SCHEMA = (
    *RUN_FIELDS,
    "case_id",
    "result",
    "result_digest",
    "sample_spacing",
    "leaf_sha256",
    "native_header_signature",
    "native_variant_manifest_digest",
    "component_catalog_digest",
    "program_digest",
    "resolved_case_digest",
    "pair_result",
    "pair_result_digest",
    "pair_program_digest",
    "pair_resolved_case_digest",
    "coupling_digest",
    "amr_mask_digest",
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


def apply_campaign_request(
    n_cells,
    request,
    *,
    case_id: str,
    allowed_dims,
    unavailable,
):
    """Honor CampaignRequest resolution and refuse unknown native dims."""
    count = int(n_cells)
    if request is None:
        return count
    dim = int(request.pops_native_dim)
    if dim not in tuple(int(item) for item in allowed_dims):
        raise unavailable(f"{case_id} has no campaign path for pops_native_dim={dim}")
    return resolution_from_request(request, count)


def require_bind_request(request, unavailable, case_id: str) -> str:
    """Return mpi_mode to pass to bind_public, or refuse an unauthenticated space."""
    if request is None:
        return "off"
    space = getattr(request, "execution_space", None) or "KokkosSerial"
    if space not in ("KokkosSerial", "KokkosOpenMP", "KokkosCuda"):
        raise unavailable(f"{case_id} has no authenticated execution space {space!r}")
    mpi_mode = getattr(request, "mpi_mode", None) or "off"
    if mpi_mode not in ("on", "off"):
        raise unavailable(f"{case_id} mpi_mode must be on or off, got {mpi_mode!r}")
    return str(mpi_mode)


def evidence_status_of(request=None, *, evidence_status: str | None = None) -> str:
    return str(
        evidence_status
        or getattr(request, "evidence_status", None)
        or "required"
    )


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


def _installed_leaf_paths() -> list[Path]:
    try:
        root = resolve_variants_root(None)
    except Exception:
        return []
    leaves: list[Path] = []
    for dim in (1, 2, 3):
        try:
            authenticated = authenticate_installed_artifact(
                dimension=dim,
                variants_root=root,
                doctor_ok=False,
            )
            leaves.append(Path(authenticated.path).resolve())
        except Exception:
            continue
    return leaves


def read_stable_program_file(
    path: str | Path,
    *,
    installed_leaf: str | Path | None = None,
) -> bytes:
    """lstat, reject symlink ``so_path``, pin inode/size, require leaf-distinct path."""
    target = Path(path)
    try:
        before = target.lstat()
    except OSError as exc:
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER) from exc
    if stat.S_ISLNK(before.st_mode):
        raise NativeSeriesError("symlinked so_path is not accepted")
    if not stat.S_ISREG(before.st_mode):
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER)
    resolved = target.resolve()
    try:
        resolved_info = resolved.lstat()
    except OSError as exc:
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER) from exc
    if stat.S_ISLNK(resolved_info.st_mode) or not stat.S_ISREG(resolved_info.st_mode):
        raise NativeSeriesError("symlinked so_path is not accepted")
    leaves = (
        [Path(installed_leaf).resolve()]
        if installed_leaf is not None
        else _installed_leaf_paths()
    )
    if resolved in leaves:
        raise NativeSeriesError("so_path must be distinct from the installed exact-rank leaf")
    pin = (resolved_info.st_dev, resolved_info.st_ino, resolved_info.st_size)
    if (before.st_dev, before.st_ino, before.st_size) != pin:
        raise NativeSeriesError("so_path resolve is not the lstat'd file")
    data = resolved.read_bytes()
    try:
        after = resolved.lstat()
    except OSError as exc:
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER) from exc
    if (after.st_dev, after.st_ino, after.st_size) != pin:
        raise NativeSeriesError("so_path changed during read")
    if stat.S_ISLNK(after.st_mode):
        raise NativeSeriesError("symlinked so_path is not accepted")
    if not data:
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER)
    return data


def program_bytes_from_artifact(artifact: Any) -> bytes:
    """Read compiled program bytes from the public artifact ``so_path``."""
    try:
        from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    except Exception as exc:
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER) from exc
    if type(artifact) is not CompiledSimulationArtifact:
        raise NativeSeriesError(PROGRAM_BYTES_BLOCKER)
    path = Path(str(getattr(artifact, "so_path", "") or ""))
    return read_stable_program_file(path)


def run_fields_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy RUN_FIELDS only. Binary/result keys are never provenance."""
    return {key: payload[key] for key in RUN_FIELDS if key in payload}


def emission_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keys handed to ``emit_job_directory``, never to ``collect_provenance``."""
    extra: dict[str, Any] = {
        "result": payload["result"],
        "program_bytes": payload["program_bytes"],
    }
    if "pair_result" in payload:
        extra["pair_result"] = payload["pair_result"]
        extra["pair_program_bytes"] = payload["pair_program_bytes"]
    if "dt" in payload:
        extra["dt"] = float(payload["dt"])
    if "coupling" in payload:
        extra["coupling"] = payload["coupling"]
    return extra


def maybe_campaign_payload(
    request,
    field,
    *,
    oracle=None,
    error_fn=None,
    pair=None,
    dt=None,
    **fields_kwargs,
):
    """Return ``field`` unless a campaign request is present; then raw RUN_FIELDS.

    Never mints ``NativeSeries``, identity digests, or a scientific pass.
    """
    if oracle is not None:
        raise NativeSeriesError("caller oracle/error_fn is not accepted")
    if error_fn is not None:
        raise NativeSeriesError("caller oracle/error_fn is not accepted")
    if request is None:
        return field
    artifact = fields_kwargs.get("artifact")
    for key in (
        "artifact",
        "simulation",
        "identity",
        "program_digest",
        "resolved_case_digest",
        "case_id",
        "sample_spacing",
        "dt",
    ):
        fields_kwargs.pop(key, None)
    payload = campaign_run_fields(request=request, **fields_kwargs)
    payload["case_id"] = getattr(request, "case_id", None)
    payload["result"] = field
    if artifact is not None:
        payload["program_bytes"] = program_bytes_from_artifact(artifact)
    step = dt if dt is not None else None
    if step is not None:
        payload["dt"] = float(step)
    if pair is not None:
        if not isinstance(pair, Mapping) or "result" not in pair:
            raise NativeSeriesError("pair requires a result mapping")
        payload["pair_result"] = pair["result"]
        payload["pair_program_bytes"] = program_bytes_from_artifact(pair.get("artifact"))
    return payload


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


def _series_orders(mapping) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(mapping, dict):
        raise TypeError("native series mapping must be a dict")
    errors = mapping.get("linf")
    spacings = mapping.get("dts") or mapping.get("spacings")
    if errors is None or spacings is None:
        return (), ()
    observed = observed_order(errors, spacings)
    return tuple(float(item) for item in errors), tuple(float(item) for item in observed)


def native_diagnostics(native) -> dict[str, Any]:
    """Compute field/frequency/energy/order from native arrays only.

    Injected ``orders`` are ignored. Frequency is not a phase error.
    Accepts ``result`` (EvidenceBundle payload) or ``field``.
    """
    if not isinstance(native, dict):
        raise TypeError("native mapping must be a dict")
    field = native.get("result", native.get("field"))
    has_lie = native.get("lie") is not None
    has_probe = native.get("probe") is not None
    if field is None and not has_lie and not has_probe:
        raise ValueError("native mapping must include field")
    out: dict[str, Any] = {}
    if has_lie:
        lie_linf, lie_orders = _series_orders(native["lie"])
        strang_linf, strang_orders = _series_orders(native.get("strang") or {})
        out["lie_linf"] = lie_linf
        out["strang_linf"] = strang_linf
        out["lie_orders"] = lie_orders
        out["strang_orders"] = strang_orders
    if field is not None:
        array = np.asarray(field, dtype=np.float64)
        if array.size == 0:
            raise ValueError("empty native field")
        volumes = native.get("volumes")
        oracle = native.get("oracle")
        if volumes is None or oracle is None:
            out["linf"] = float(np.max(np.abs(array)))
            out["size"] = int(array.size)
        else:
            errors = reference_errors(
                array,
                np.asarray(oracle, dtype=np.float64),
                np.asarray(volumes, dtype=np.float64),
            )
            out["l1"] = float(errors.l1)
            out["l2"] = float(errors.l2)
            out["linf"] = float(errors.linf)
            out["field_error"] = float(errors.linf)
    if has_probe:
        times = native.get("times")
        if times is None:
            raise ValueError("probe requires times")
        omega_ref = native.get("omega_ref")
        omega_num = numerical_frequency(times, native["probe"], method="fft")
        out["omega_num"] = float(omega_num)
        if omega_ref is not None:
            out["frequency_error"] = float(frequency_error(omega_num, omega_ref))
        if native.get("oracle_probe") is not None:
            out["phase_error"] = float(phase_error(native["probe"], native["oracle_probe"]))
    energy = native.get("energy")
    energy_ref = native.get("energy_ref")
    if energy is not None and energy_ref is not None:
        out["energy_drift"] = float(energy) - float(energy_ref)
    elif native.get("energy_initial") is not None and native.get("energy_final") is not None:
        initial = float(native["energy_initial"])
        final = float(native["energy_final"])
        out["energy_drift"] = (final - initial) / initial if initial != 0.0 else final - initial
    if native.get("potential_error") is not None:
        out["potential_error"] = float(native["potential_error"])
    if native.get("residual_l2") is not None:
        out["residual_l2"] = float(native["residual_l2"])
    return out


def native_report_sections(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Map native diagnostics onto report blocks. Frequency is not phase."""
    poisson = dict(NULL_POISSON)
    coupling = dict(NULL_COUPLING)
    extra_reasons: dict[str, str] = {}
    if diagnostics.get("potential_error") is not None:
        poisson["potential_error"] = float(diagnostics["potential_error"])
    if diagnostics.get("field_error") is not None:
        poisson["field_error"] = float(diagnostics["field_error"])
    if diagnostics.get("residual_l2") is not None:
        poisson["residual_l2"] = float(diagnostics["residual_l2"])
    if diagnostics.get("phase_error") is not None:
        coupling["phase_error"] = float(diagnostics["phase_error"])
    elif diagnostics.get("frequency_error") is not None:
        extra_reasons["coupling.phase_error"] = (
            "frequency_error is not coupling.phase_error; phase left null"
        )
    if diagnostics.get("energy_drift") is not None:
        coupling["energy_drift"] = float(diagnostics["energy_drift"])
    return {"poisson": poisson, "coupling": coupling, "extra_reasons": extra_reasons}


def order_rows(case_id: str, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Temporal order rows derived from native error series only."""
    rows: list[dict[str, Any]] = []
    if diagnostics.get("lie_orders"):
        rows.extend(
            orders_from_native_errors(
                case_id,
                diagnostics["lie_linf"],
                diagnostics.get("lie_dts")
                or tuple(2.0 ** -index for index in range(len(diagnostics["lie_linf"]))),
                kind="temporal",
                variable="lie",
                threshold=0.8,
            )
            if diagnostics.get("lie_linf")
            and len(diagnostics["lie_linf"]) == len(diagnostics["lie_orders"]) + 1
            else [
                {
                    "case_id": case_id,
                    "kind": "temporal",
                    "variable": "lie",
                    "observed_order": float(value),
                    "threshold": 0.8,
                }
                for value in diagnostics["lie_orders"]
            ]
        )
    if diagnostics.get("strang_orders"):
        rows.extend(
            {
                "case_id": case_id,
                "kind": "temporal",
                "variable": "strang",
                "observed_order": float(value),
                "threshold": 1.8,
            }
            for value in diagnostics["strang_orders"]
        )
    return rows


def fail_closed_report(
    *,
    case_id: str,
    component: str,
    native_dims: list[int],
    reason: str = NO_NATIVE_REASON,
    suite: str = "pr",
    request=None,
    evidence_status: str | None = None,
    orders: list | None = None,
    poisson: dict | None = None,
    coupling: dict | None = None,
    extra_reasons: dict | None = None,
) -> dict:
    """Schema-valid report driven by CampaignRequest.evidence_status."""
    return fail_closed_summary(
        case_id,
        reason,
        native_dimensions=list(native_dims),
        components=[component],
        extra_reasons=extra_reasons,
        suite=suite,
        poisson=poisson,
        coupling=coupling,
        orders=orders,
        request=request,
        evidence_status=evidence_status,
    )


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
    coupling: dict[str, Any] | None = None,
    orders: list | None = None,
    request=None,
    evidence_status: str | None = None,
) -> dict[str, Any]:
    status_name = evidence_status_of(request, evidence_status=evidence_status)
    not_supported = 1 if status_name == "capability-gated" else 0
    failed = 0 if not_supported else 1
    reasons = _base_reasons(case_id)
    reasons["orders"] = reason
    if extra_reasons:
        reasons.update(extra_reasons)
    if orders:
        reasons.pop("orders", None)
    failures = []
    if failed:
        failures.append(
            {
                "case_id": case_id,
                "reason": reason,
                "metrics_ref": "",
                "provenance_ref": "",
            }
        )
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": repository_sha(),
        "suite": suite,
        "max_nodes": 2,
        "native_dimensions": list(native_dimensions),
        "execution_spaces": ["KokkosSerial"],
        "coverage": _coverage(
            passed=0, failed=failed, not_supported=not_supported, components=components
        ),
        "failures": failures,
        "orders": list(orders or []),
        "amr": dict(NULL_AMR),
        "poisson": dict(poisson or NULL_POISSON),
        "coupling": dict(coupling or NULL_COUPLING),
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


def _as_bundle(evidence: Any) -> EvidenceBundle | None:
    if isinstance(evidence, EvidenceBundle):
        return None
    if isinstance(evidence, (str, Path)):
        try:
            return EvidenceBundle(evidence)
        except EvidenceError:
            return None
    return None


def report_from_native_series(
    case_id: str,
    native_series: str | Path | None,
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
    bundle = _as_bundle(native_series)
    if bundle is None or getattr(bundle, "case_id", None) != str(case_id):
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if len(bundle.derived_linf) < 2 or len(bundle.derived_spacings) < 2:
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    try:
        orders = orders_from_native_errors(
            case_id,
            bundle.derived_linf,
            bundle.derived_spacings,
            kind=kind,
            variable=variable,
            threshold=threshold,
        )
    except ValueError:
        return fail_closed_summary(
            case_id,
            MISSING_NATIVE_SERIES,
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if any(float(row["observed_order"]) < float(threshold) for row in orders):
        return fail_closed_summary(
            case_id,
            "observed order below threshold",
            native_dimensions=native_dimensions,
            components=components,
            extra_reasons=extra_reasons,
        )
    if case_id == "TR-06":
        if len(bundle.derived_pair_linf) < 2:
            return fail_closed_summary(
                case_id,
                "TR-06 pair result was not analyzed",
                native_dimensions=native_dimensions,
                components=components,
                extra_reasons=extra_reasons,
            )
        try:
            pair_orders = orders_from_native_errors(
                case_id,
                bundle.derived_pair_linf,
                bundle.derived_spacings,
                kind=kind,
                variable=f"{variable}_pair",
                threshold=threshold,
            )
        except ValueError:
            return fail_closed_summary(
                case_id,
                "TR-06 pair permutation discrepancy is not an order series",
                native_dimensions=native_dimensions,
                components=components,
                extra_reasons=extra_reasons,
            )
        if any(float(row["observed_order"]) < float(threshold) for row in pair_orders):
            return fail_closed_summary(
                case_id,
                "TR-06 pair observed order below threshold",
                native_dimensions=native_dimensions,
                components=components,
                extra_reasons=extra_reasons,
            )
        orders = list(orders) + list(pair_orders)
    return passed_summary(
        case_id,
        native_dimensions=native_dimensions,
        components=components,
        orders=orders,
        extra_reasons=extra_reasons,
    )


def metrics_from_native_series(case_id: str, series: Any) -> dict[str, Any]:
    bundle = _as_bundle(series)
    if bundle is None or getattr(bundle, "case_id", None) != str(case_id):
        raise NativeSeriesError("metrics require a validated EvidenceBundle")
    return collect_metrics(
        case_id,
        reason="authenticated evidence bundle; unused fields remain null",
    )


def provenance_from_native_series(
    case_id: str,
    series: Any,
    **provenance_kwargs: Any,
) -> dict[str, Any]:
    bundle = _as_bundle(series)
    if bundle is None or getattr(bundle, "case_id", None) != str(case_id):
        raise NativeSeriesError("provenance require a validated EvidenceBundle")
    record = bundle.records[0]
    stored = record.get("provenance")
    if isinstance(stored, Mapping) and stored.get("case_id") == case_id:
        return dict(stored)
    dim = len(record.get("resolution") or record["provenance"].get("resolution") or [1])
    kwargs = dict(provenance_kwargs)
    kwargs.setdefault("doctor_ok", False)
    run = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else record
    return collect_provenance(
        case_id,
        pops_native_dim=dim,
        dimension=dim,
        nodes=1,
        run={key: run[key] for key in RUN_FIELDS if key in run},
        **kwargs,
    )
