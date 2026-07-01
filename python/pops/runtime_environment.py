"""pops.runtime_environment -- explicit native runtime environment capabilities.

This module is metadata-only at import time. It centralizes the current native runtime facts:
2D mesh core, AMR refinement ratio 2, double precision, and no custom communicator route.
When the compiled extension is available, :func:`runtime_environment_report` delegates to the
C++ report; otherwise it returns the same conservative static facts with unknown lifecycle fields.
"""

NATIVE_DIMENSION = 2
NATIVE_AMR_REFINEMENT_RATIO = 2
NATIVE_PRECISION = "double"
NATIVE_REAL_BYTES = 8
NATIVE_COMMUNICATOR = "MPI_COMM_WORLD"


def _static_report():
    return {
        "dimension": NATIVE_DIMENSION,
        "amr_refinement_ratio": NATIVE_AMR_REFINEMENT_RATIO,
        "precision": NATIVE_PRECISION,
        "real_bytes": NATIVE_REAL_BYTES,
        "supports_single_precision": False,
        "supports_mixed_precision": False,
        "has_kokkos": None,
        "kokkos_initialized": None,
        "kokkos_finalized": None,
        "kokkos_initialized_by_pops": None,
        "kokkos_atexit_finalize_registered": None,
        "kokkos_backend": "unknown",
        "kokkos_ownership": "unknown",
        "kokkos_lifecycle": "unknown until _pops.runtime_environment_report() is available",
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


def runtime_environment_report():
    """Return runtime facts for reports and validators.

    The preferred source is ``_pops.runtime_environment_report()``. The fallback is static and
    conservative: it never claims custom communicators, non-2D, non-ratio-2 AMR, or non-double
    precision support.
    """
    try:
        from pops import _pops  # noqa: PLC0415 -- optional runtime extension
        fn = getattr(_pops, "runtime_environment_report", None)
        if fn is not None:
            return dict(fn())
    except Exception:
        pass
    return _static_report()


def compiled_runtime_facts(*, supports_mpi=None):
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


def validate_dimension(value, *, where="runtime"):
    """Reject any requested dimension other than the native 2D core."""
    dim = int(value)
    if dim != NATIVE_DIMENSION:
        raise ValueError(
            "%s: dimension=%d is unsupported; native PoPS is dimension=%d only "
            "(Box2D/Fab2D/Geometry/Euler/Lorentz/EB/AMR kernels are 2D)."
            % (where, dim, NATIVE_DIMENSION))
    return dim


def validate_amr_refinement_ratio(value, *, where="AMR"):
    """Reject any requested AMR refinement ratio other than 2."""
    ratio = int(value)
    if ratio != NATIVE_AMR_REFINEMENT_RATIO:
        raise ValueError(
            "%s: AMR refinement ratio %d is unsupported; native AMR supports ratio %d only "
            "(hierarchy, patch ranges, reflux and subcycling are ratio-2 kernels)."
            % (where, ratio, NATIVE_AMR_REFINEMENT_RATIO))
    return ratio


def validate_precision(value, *, where="runtime"):
    """Reject precision policies that the hardcoded C++ ``Real=double`` core cannot honor."""
    precision = str(value).lower()
    aliases = {"double", "float64", "real64"}
    if precision not in aliases:
        raise ValueError(
            "%s: precision=%r is unsupported; native PoPS is Real=double only "
            "(single/mixed precision has no C++ policy route)." % (where, value))
    return NATIVE_PRECISION


def validate_communicator(value, *, where="runtime"):
    """Reject custom communicator requests until the native MPI seam supports them."""
    comm = str(value)
    if comm in ("serial", "none"):
        return "serial"
    if comm in (NATIVE_COMMUNICATOR, "world"):
        report = runtime_environment_report()
        if report.get("communicator") == NATIVE_COMMUNICATOR:
            return NATIVE_COMMUNICATOR
    raise ValueError(
        "%s: communicator=%r is unsupported; native PoPS exposes only %s when MPI is compiled, "
        "or serial otherwise. Custom MPI communicators are not a native route yet."
        % (where, value, NATIVE_COMMUNICATOR))


def validate_runtime_environment(*, dimension=None, amr_refinement_ratio=None,
                                 precision=None, communicator=None, where="runtime"):
    """Validate all explicit runtime environment requests supplied by a caller."""
    out = {}
    if dimension is not None:
        out["dimension"] = validate_dimension(dimension, where=where)
    if amr_refinement_ratio is not None:
        out["amr_refinement_ratio"] = validate_amr_refinement_ratio(
            amr_refinement_ratio, where=where)
    if precision is not None:
        out["precision"] = validate_precision(precision, where=where)
    if communicator is not None:
        out["communicator"] = validate_communicator(communicator, where=where)
    return out


__all__ = [
    "NATIVE_DIMENSION", "NATIVE_AMR_REFINEMENT_RATIO", "NATIVE_PRECISION",
    "NATIVE_REAL_BYTES", "NATIVE_COMMUNICATOR", "runtime_environment_report",
    "compiled_runtime_facts", "validate_dimension", "validate_amr_refinement_ratio",
    "validate_precision", "validate_communicator", "validate_runtime_environment",
]
