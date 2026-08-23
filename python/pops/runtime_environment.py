"""pops.runtime_environment -- explicit native runtime environment capabilities.

This module is metadata-only at import time. It centralizes the current native runtime facts:
an artifact-selected compile-time spatial dimension, hierarchy-selected AMR transition ratios,
double precision, and no custom communicator route.
When the compiled extension is available, :func:`runtime_environment_report` delegates to the
C++ report; otherwise it returns the same conservative static facts with unknown lifecycle fields
and zero active Kokkos concurrency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pops.params.use_sites import ParamUse, resolve_param_use

# The declared native-core facts live in the dependency-free leaf pops._native_facts (so the
# runtime-fenced layers can read them); this module re-exports them as the public spelling.
from pops._native_facts import (  # noqa: F401  (re-export)
    NATIVE_COMMUNICATOR,
    NATIVE_MAX_RUNTIME_PARAMS,
    NATIVE_PRECISION,
    NATIVE_REAL_BYTES,
    NATIVE_SUPPORTED_DIMENSIONS,
)


def native_dimension() -> int | None:
    """Return the exact dimension baked into the loaded extension, or ``None`` without one."""
    from pops._native_selector import selected_native_dimension

    return selected_native_dimension()


class RuntimeCapabilityError(ValueError):
    """Unsupported runtime capability request with the structured report attached."""

    def __init__(self, message: str, *, field: str, requested: Any, report: Any = None) -> None:
        super().__init__(message)
        self.field = field
        self.requested = requested
        self.report = dict(report) if report is not None else runtime_environment_report()

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "requested": self.requested,
            "message": str(self),
            "runtime_environment": dict(self.report),
        }


def _static_report() -> dict:
    return {
        "dimension": native_dimension(),
        "amr_refinement_ratio": None,
        "amr_refinement_ratio_selection": "hierarchy_exact_rank",
        "amr_refinement_ratio_rank": native_dimension(),
        "precision": NATIVE_PRECISION,
        "real_bytes": NATIVE_REAL_BYTES,
        "max_runtime_params": NATIVE_MAX_RUNTIME_PARAMS,
        "supports_single_precision": False,
        "supports_mixed_precision": False,
        "has_kokkos": None,
        "kokkos_initialized": None,
        "kokkos_finalized": None,
        "kokkos_initialized_by_pops": None,
        "kokkos_atexit_finalize_registered": None,
        "kokkos_backend": "unknown",
        "kokkos_device": "unknown",
        "kokkos_shared_space": "unknown",
        "field_memory_space": "unknown",
        "kokkos_stream": "unknown",
        "kokkos_stream_synchronous": False,
        "kokkos_concurrency": 0,
        "kokkos_ownership": "unknown",
        "kokkos_lifecycle": "unknown until _pops.runtime_environment_report() is available",
        "gpu_device_ordinal": -1,
        "gpu_uuid": "",
        "gpu_uuid_method": "none",
        "gpu_uuid_diagnostic": "native runtime environment report unavailable",
        "mpi_compiled": None,
        "mpi_active": None,
        "mpi_rank": 0,
        "mpi_ranks": 1,
        "communicator": "unknown",
        "supports_custom_communicator": False,
        "allocator_mode": "unknown",
        "comm_allocator_mode": "unknown",
        "allocator_lifetime": "unknown until _pops.runtime_environment_report() is available",
    }


def runtime_environment_report() -> dict:
    """Return runtime facts for reports and validators.

    The preferred source is ``_pops.runtime_environment_report()``. The fallback is static and
    conservative: it never claims custom communicators, an unauthenticated dimension,
    a fabricated process-global AMR ratio, non-double
    precision support, or an active Kokkos execution-space concurrency.
    """
    from pops._native_selector import selected_native_module

    native = selected_native_module(required=False)
    if native is None:
        return _static_report()
    fn = getattr(native, "runtime_environment_report", None)
    if fn is not None:
        # A present native module is authoritative.  Import/ABI/runtime failures must remain
        # visible; silently reporting a serial/unknown fallback would let resolve authenticate a
        # topology different from the loaded binary.
        return dict(fn())
    return _static_report()


def compiled_runtime_facts(*, supports_mpi: Any = None) -> dict:
    """Runtime facts for inert compiled-artifact reports.

    ``supports_mpi`` is the artifact's own MPI capability when known. ``None`` keeps the
    communicator unknown rather than fabricating MPI support.
    """
    facts = _static_report()
    if supports_mpi is True:
        facts["communicator"] = NATIVE_COMMUNICATOR
    elif supports_mpi is False:
        facts["communicator"] = "serial"
    else:
        facts["communicator"] = "unknown"
    facts["mpi_compiled"] = supports_mpi
    return facts


def validate_dimension(value: Any, *, where: str = "runtime") -> int:
    """Require the requested dimension to equal the compiled artifact specialization."""
    value = resolve_param_use(value, ParamUse.ABI, where="%s(dimension=)" % where)
    if type(value) is not int or value not in NATIVE_SUPPORTED_DIMENSIONS:
        raise RuntimeCapabilityError(
            "%s: dimension must be exactly 1, 2, or 3" % where, field="dimension", requested=value
        )
    dim = value
    active = native_dimension()
    if active is None:
        raise RuntimeCapabilityError(
            "%s: no compiled PoPS artifact authenticates the requested dimension" % where,
            field="dimension",
            requested=dim,
        )
    if dim != active:
        raise RuntimeCapabilityError(
            "%s: dimension=%d differs from the loaded compile-time specialization dimension=%d"
            % (where, dim, active),
            field="dimension",
            requested=dim,
        )
    return dim


def validate_amr_refinement_ratio(value: Any, *, where: str = "AMR") -> tuple[int, ...]:
    """Authenticate one isotropic or exact-rank anisotropic AMR transition ratio.

    A scalar denotes an isotropic refinement of at least two on every native axis.  A sequence
    is an exact native-rank ratio: each axis must remain positive and at least one axis must
    refine.  Selection belongs to hierarchy construction; this validator deliberately does not
    compare the result with a process-global native constant.
    """
    value = resolve_param_use(value, ParamUse.AMR_HIERARCHY, where="%s(refinement_ratio=)" % where)
    dimension = native_dimension()
    if type(dimension) is not int or dimension not in NATIVE_SUPPORTED_DIMENSIONS:
        raise RuntimeCapabilityError(
            "%s: no compiled PoPS artifact authenticates the AMR ratio rank" % where,
            field="amr_refinement_ratio",
            requested=value,
        )
    if type(value) is int:
        if value < 2:
            raise RuntimeCapabilityError(
                "%s: isotropic AMR refinement_ratio must be an integer >= 2" % where,
                field="amr_refinement_ratio",
                requested=value,
            )
        return (value,) * dimension
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RuntimeCapabilityError(
            "%s: AMR refinement_ratio must be an integer or a length-%d sequence"
            % (where, dimension),
            field="amr_refinement_ratio",
            requested=value,
        )
    ratio = tuple(value)
    if len(ratio) != dimension:
        raise RuntimeCapabilityError(
            "%s: AMR refinement_ratio sequence must have exactly %d axes (got %d)"
            % (where, dimension, len(ratio)),
            field="amr_refinement_ratio",
            requested=value,
        )
    if any(type(axis) is not int or axis < 1 for axis in ratio):
        raise RuntimeCapabilityError(
            "%s: AMR refinement_ratio axes must be plain integers >= 1" % where,
            field="amr_refinement_ratio",
            requested=value,
        )
    if not any(axis > 1 for axis in ratio):
        raise RuntimeCapabilityError(
            "%s: AMR refinement_ratio must refine at least one axis" % where,
            field="amr_refinement_ratio",
            requested=value,
        )
    return ratio


def validate_precision(value: Any, *, where: str = "runtime") -> str:
    """Reject precision policies that the hardcoded C++ ``Real=double`` core cannot honor."""
    value = resolve_param_use(value, ParamUse.ABI, where="%s(precision=)" % where)
    precision = str(value).lower()
    aliases = {"double", "float64", "real64"}
    if precision not in aliases:
        raise RuntimeCapabilityError(
            "%s: precision=%r is unsupported; native PoPS is Real=double only "
            "(single/mixed precision has no C++ policy route)." % (where, value),
            field="precision",
            requested=value,
        )
    return NATIVE_PRECISION


def validate_communicator(value: Any, *, where: str = "runtime") -> str:
    """Reject custom communicator requests until the native MPI seam supports them."""
    value = resolve_param_use(value, ParamUse.ABI, where="%s(communicator=)" % where)
    comm = str(value)
    if comm in ("serial", "none"):
        return "serial"
    if comm in (NATIVE_COMMUNICATOR, "world"):
        report = runtime_environment_report()
        if report.get("communicator") == NATIVE_COMMUNICATOR:
            return NATIVE_COMMUNICATOR
    raise RuntimeCapabilityError(
        "%s: communicator=%r is unsupported; native PoPS exposes only %s when MPI is compiled, "
        "or serial otherwise. Custom MPI communicators are not a native route yet."
        % (where, value, NATIVE_COMMUNICATOR),
        field="communicator",
        requested=value,
    )


def validate_runtime_environment(
    *,
    dimension: Any = None,
    amr_refinement_ratio: Any = None,
    precision: Any = None,
    communicator: Any = None,
    where: str = "runtime",
) -> dict:
    """Validate all explicit runtime environment requests supplied by a caller."""
    out: dict = {}
    if dimension is not None:
        out["dimension"] = validate_dimension(dimension, where=where)
    if amr_refinement_ratio is not None:
        out["amr_refinement_ratio"] = validate_amr_refinement_ratio(
            amr_refinement_ratio, where=where
        )
    if precision is not None:
        out["precision"] = validate_precision(precision, where=where)
    if communicator is not None:
        out["communicator"] = validate_communicator(communicator, where=where)
    return out


__all__ = [
    "NATIVE_SUPPORTED_DIMENSIONS",
    "native_dimension",
    "NATIVE_PRECISION",
    "NATIVE_REAL_BYTES",
    "NATIVE_COMMUNICATOR",
    "runtime_environment_report",
    "compiled_runtime_facts",
    "validate_dimension",
    "validate_amr_refinement_ratio",
    "validate_precision",
    "validate_communicator",
    "validate_runtime_environment",
    "RuntimeCapabilityError",
]
