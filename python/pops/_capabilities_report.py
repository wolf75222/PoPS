"""Native capability report and route-row builders (ADC-619 split).

The native side of the capability layer: :class:`NativeCapabilityReport` (the
Python value object for ``_pops.capability_report()``), the lazy ``_pops`` bridges
(``_module_capabilities`` / ``_native_capability_report_from_extension``), the flat
route-row builders (``_support_rows`` / ``_inventory_rows`` / ``_row``) and the
public ``native_capability_report`` / ``native_capability_matrix`` entry points.
Split out of ``_capabilities`` for the 500-line cap; ``pops._capabilities``
re-exports every name here. ``_pops`` is imported LAZILY so this module stays
importable without a compiled extension.
"""

import json

from pops._capabilities_common import (
    CapabilityRouteMatrix,
    CapabilityRouteRow,
    _flag_value,
    _status_from_flag,
    _unsupported_error,
)


def _module_capabilities(target="module"):
    """The C++ authoritative capability dict (``_pops.module_capabilities()``) or ``None``.

    Lazily imports ``_pops`` (top-level then ``pops._pops``, mirroring the codegen toolchain) so the
    module import graph stays acyclic and the catalog walk works with no compiled module present.
    Returns ``None`` when ``_pops`` is unavailable or predates ``module_capabilities`` (old build):
    the descriptor walk then proceeds WITHOUT the cross-check rather than failing -- a missing native
    source is a graceful degradation, never a fabricated capability.
    """
    try:
        import _pops as mod  # noqa: PLC0415 -- lazy: keeps this module's import graph acyclic
    except Exception:
        try:
            from pops import _pops as mod  # noqa: PLC0415
        except Exception:
            return None
    fn = getattr(mod, "module_capabilities", None)
    if fn is None:  # an _pops built before this work: no authoritative source to cross-check against.
        return None
    try:
        return dict(fn(target))
    except Exception:
        return None


def _native_capability_report_from_extension(target="module"):
    """Return ``_pops.capability_report(target)`` as :class:`NativeCapabilityReport`, or ``None``."""
    try:
        import _pops as mod  # noqa: PLC0415 -- lazy: keeps this module importable without _pops
    except Exception:
        try:
            from pops import _pops as mod  # noqa: PLC0415
        except Exception:
            return None
    fn = getattr(mod, "capability_report", None)
    if fn is None:
        return None
    try:
        return NativeCapabilityReport.from_dict(dict(fn(target)))
    except Exception:
        return None


def native_capability_report(target="module", *, flags=None, source=None):
    """Return the structured native capability report (ADC-591).

    With a current ``_pops`` build, values come from C++ ``capability_report(target)``. ``flags`` is
    the manifest fallback path for already-compiled artifacts: it builds the same stable envelope from
    the manifest support flags and the Python inventory rows, without loading or recompiling the
    artifact.
    """
    if flags is None:
        report = _native_capability_report_from_extension(target)
        if report is not None:
            return report
        flags = _module_capabilities(target)
        source = source or ("native" if flags is not None else "unknown")
    else:
        source = source or "manifest"
    rows = _support_rows(flags, source) + _inventory_rows(flags, source)
    caps = dict(flags or {})
    return NativeCapabilityReport(
        schema_version=0, abi_version=int(caps.get("abi_version", 0) or 0), target=target,
        abi_key=None, platform="unknown", capabilities=caps, runtime={}, routes=rows)


def _feature_layout(feature):
    if feature == "supports_uniform":
        return "uniform"
    if feature == "supports_amr":
        return "amr"
    return "uniform|amr"


def _feature_backend(feature):
    if feature in ("supports_stride", "supports_amr"):
        return "production"
    return "module"


def _feature_platform(feature):
    if feature in ("supports_mpi", "supports_custom_communicator"):
        return "mpi"
    if feature == "supports_gpu":
        return "gpu"
    return "host"


def _flag_error_message(feature):
    requests = {
        "supports_amr": ("layout=AMR", "layout=Uniform or backend='production' target='amr_system'",
                         "use layout=Uniform or compile with backend='production' target='amr_system'"),
        "supports_mpi": ("platform=MPI", "serial/OpenMP build", "rebuild _pops with POPS_USE_MPI=ON"),
        "supports_gpu": ("platform=GPU", "host CPU platform", "use KokkosOpenMP/KokkosSerial or a CUDA/HIP build"),
        "supports_stride": ("strided cell access", "backend='production'",
                            "compile with backend='production'"),
        "supports_partial_imex_mask": ("partial IMEX mask", "full source implicit / split routes",
                                       "use IMEX/IMEXRK/Split without partial masks"),
        "supports_custom_communicator": ("communicator != MPI_COMM_WORLD",
                                         "MPI_COMM_WORLD or serial",
                                         "run on MPI_COMM_WORLD until ParallelContext lands"),
    }
    requested, available, alternative = requests.get(
        feature, (feature, "no route in this build", None))
    return _unsupported_error(requested=requested, available=available, alternative=alternative)


class NativeCapabilityReport:
    """Versioned structured native capability report (ADC-591).

    This is the Python value object for ``_pops.capability_report()``. Pretty route matrices and
    legacy ``module_capabilities()`` dicts are projections of this object. ``routes`` is a list of
    :class:`CapabilityRouteRow` instances, each carrying a status and reason directly, so tests and
    validators do not parse formatted strings.
    """

    def __init__(self, *, schema_version, abi_version, target, abi_key, platform,
                 capabilities, runtime, routes):
        self.schema_version = int(schema_version)
        self.abi_version = int(abi_version)
        self.target = target
        self.abi_key = abi_key
        self.platform = platform
        self.capabilities = dict(capabilities or {})
        self.runtime = dict(runtime or {})
        self.routes = list(routes)

    @classmethod
    def from_dict(cls, payload):
        routes = [_route_from_native_dict(row) for row in payload.get("routes", [])]
        return cls(
            schema_version=payload.get("schema_version", 0),
            abi_version=payload.get("abi_version", payload.get("capabilities", {}).get("abi_version", 0)),
            target=payload.get("target", "module"),
            abi_key=payload.get("abi_key"),
            platform=payload.get("platform"),
            capabilities=payload.get("capabilities", {}),
            runtime=payload.get("runtime", {}),
            routes=routes)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "abi_version": self.abi_version,
            "target": self.target,
            "abi_key": self.abi_key,
            "platform": self.platform,
            "capabilities": dict(self.capabilities),
            "runtime": dict(self.runtime),
            "routes": [row.to_dict() for row in self.routes],
        }

    def to_json(self, path=None, *, indent=2):
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def route(self, feature):
        for row in self.routes:
            if row.feature == feature:
                return row
        raise KeyError(feature)

    def __repr__(self):
        return ("NativeCapabilityReport(schema=%r, abi=%r, target=%r, routes=%d)"
                % (self.schema_version, self.abi_version, self.target, len(self.routes)))

    def __str__(self):
        lines = ["native capability report (schema=%s, abi=%s, target=%s)"
                 % (self.schema_version, self.abi_version, self.target)]
        lines.append("  platform : %s" % self.platform)
        lines.append("  abi_key  : %s" % ((self.abi_key or "")[:12] or "none"))
        lines.append("  runtime  : dimension=%s amr_refinement_ratio=%s precision=%s communicator=%s"
                     % (self.runtime.get("dimension"), self.runtime.get("amr_refinement_ratio"),
                        self.runtime.get("precision"), self.runtime.get("communicator")))
        lines.append("  routes   : %d structured row(s)" % len(self.routes))
        for row in self.routes:
            if row.status != "available":
                lines.append("    %-34s %-11s %s" % (row.feature, row.status, row.limitation))
        return "\n".join(lines)


def _route_from_native_dict(raw):
    status = raw.get("status", "unknown")
    requested = raw.get("requested") or raw.get("feature")
    available_route = raw.get("available_route") or "no native route"
    alternative = raw.get("alternative") or None
    limitation = raw.get("reason") or raw.get("limitation") or ""
    error = raw.get("error_message") or ""
    if status == "unavailable" and not error:
        error = _unsupported_error(
            requested=requested, available=available_route, alternative=alternative)
    return CapabilityRouteRow(
        raw.get("feature") or raw.get("route_id"),
        layout=raw.get("layout", "any"),
        backend=raw.get("backend", "any"),
        platform=raw.get("platform", "host"),
        mpi=raw.get("mpi", False),
        gpu=raw.get("gpu", False),
        status=status,
        limitation=limitation,
        error_message=error,
        source=raw.get("source", "native"),
        axis=raw.get("axis"),
        available_route=raw.get("available_route", ""),
        alternative=raw.get("alternative", ""))


def _row(feature, *, layout="any", backend="any", platform="host", flags=None,
         flag=None, mpi=False, gpu=False, limitation="", requested=None,
         available_route=None, alternative=None, source="native", status=None):
    if status is None:
        status = _status_from_flag(flags, flag) if flag else "available"
    err = ""
    if status == "unavailable":
        err = _unsupported_error(
            requested=requested or feature,
            available=available_route or "no native route",
            alternative=alternative)
    return CapabilityRouteRow(
        feature, layout=layout, backend=backend, platform=platform, mpi=mpi, gpu=gpu,
        status=status, limitation=limitation, error_message=err, source=source,
        available_route=available_route or "", alternative=alternative or "")


def _support_rows(flags, source):
    return [
        _row("supports_uniform", layout="uniform", backend="module", platform="host",
             flags=flags, flag="supports_uniform", limitation="single-level Uniform layout",
             requested="layout=Uniform", available_route="layout=Uniform", source=source),
        _row("supports_amr", layout="amr", backend="production", platform="host",
             flags=flags, flag="supports_amr",
             limitation="native AMR envelope: max_levels<=2, ratio=2",
             requested="layout=AMR", available_route="backend='production' target='amr_system'",
             alternative="use Uniform or AMR(max_levels<=2, ratio=2)", source=source),
        _row("supports_mpi", layout="uniform|amr", backend="production", platform="mpi",
             flags=flags, flag="supports_mpi", mpi=bool(_flag_value(flags, "supports_mpi")),
             limitation="MPI is available only when _pops is built with POPS_USE_MPI=ON",
             requested="platform=MPI", available_route="serial/OpenMP build",
             alternative="rebuild with -DPOPS_USE_MPI=ON", source=source),
        _row("supports_gpu", layout="uniform|amr", backend="production", platform="gpu",
             flags=flags, flag="supports_gpu", gpu=bool(_flag_value(flags, "supports_gpu")),
             limitation="GPU is available only for a Kokkos CUDA/HIP device build",
             requested="platform=GPU", available_route="host CPU platform",
             alternative="use KokkosOpenMP/KokkosSerial or a CUDA/HIP build", source=source),
        _row("supports_stride", layout="uniform|amr", backend="production", platform="host",
             flags=flags, flag="supports_stride",
             limitation="real cell stride is carried only by the production/native route",
             requested="strided cell access", available_route="backend='production'",
             alternative="compile with backend='production'", source=source),
        _row("supports_named_fields", layout="uniform|amr", backend="production", platform="host",
             flags=flags, flag="supports_named_fields",
             limitation="named aux-field transport", requested="named aux fields",
             available_route="native named-field transport", source=source),
        _row("supports_partial_imex_mask", layout="uniform|amr", backend="production",
             platform="host", flags=flags, flag="supports_partial_imex_mask",
             limitation="no C++ route backs a partial IMEX mask",
             requested="partial IMEX mask", available_route="full source implicit / split routes",
             alternative="use IMEX/IMEXRK/Split without partial masks", source=source),
        _row("supports_custom_communicator", layout="uniform|amr", backend="none",
             platform="mpi", flags=flags, flag="supports_custom_communicator",
             limitation="no C++ route accepts a caller-provided MPI_Comm",
             requested="communicator != MPI_COMM_WORLD",
             available_route="MPI_COMM_WORLD or serial",
             alternative="run on MPI_COMM_WORLD until ParallelContext lands", source=source),
    ]


def _inventory_rows(flags, source):
    mpi = bool(_flag_value(flags, "supports_mpi"))
    gpu = bool(_flag_value(flags, "supports_gpu"))
    return [
        _row("layout:Uniform", layout="uniform", backend="module", platform="host",
             mpi=mpi, gpu=gpu, limitation="2D single-level Cartesian/Polar layout", source=source),
        _row("layout:AMR", layout="amr", backend="production", platform="host",
             flags=flags, flag="supports_amr", mpi=mpi, gpu=gpu,
             limitation="max_levels<=2 and ratio=2; unsupported ratios/levels validate before bind",
             requested="AMR(max_levels>2 or ratio!=2)", available_route="AMR(max_levels<=2, ratio=2)",
             alternative="use Uniform or the native AMR envelope", source=source),
        _row("spatial:finite_volume", layout="uniform|amr", backend="production|aot|prototype",
             platform="host", mpi=mpi, gpu=gpu,
             limitation="2D finite-volume route; prototype backend is host-only", source=source),
        _row("riemann:rusanov", layout="uniform|amr", backend="production|aot|prototype",
             platform="host", mpi=mpi, gpu=gpu,
             limitation="requires model max_wave_speed", source=source),
        _row("riemann:hll", layout="uniform|amr", backend="production|aot", platform="host",
             mpi=mpi, gpu=gpu, limitation="requires physical_flux and wave_speeds", source=source),
        _row("riemann:hllc", layout="uniform|amr", backend="production|aot", platform="host",
             mpi=mpi, gpu=gpu,
             limitation="requires Euler/HLLC model capabilities; polar route is unavailable",
             source=source),
        _row("riemann:roe", layout="uniform|amr", backend="production|aot", platform="host",
             mpi=mpi, gpu=gpu,
             limitation="requires Roe dissipation capability; polar route is unavailable",
             source=source),
        _row("reconstruction:firstorder", layout="uniform|amr", backend="production|aot|prototype",
             limitation="ghost_depth=1", source=source),
        _row("reconstruction:muscl", layout="uniform|amr", backend="production|aot|prototype",
             limitation="ghost_depth=2; native limiters minmod/vanleer", source=source),
        _row("reconstruction:weno5", layout="uniform|amr", backend="production|aot",
             limitation="ghost_depth=3; high-order route is native", source=source),
        _row("limiter:mc", layout="uniform|amr", backend="none", status="unavailable",
             limitation="catalogued but no native C++ limiter symbol exists",
             requested="limiter=MC()", available_route="Minmod() or VanLeer()",
             alternative="use pops.numerics.reconstruction.limiters.Minmod()", source=source),
        _row("limiter:superbee", layout="uniform|amr", backend="none", status="unavailable",
             limitation="catalogued but no native C++ limiter symbol exists",
             requested="limiter=Superbee()", available_route="Minmod() or VanLeer()",
             alternative="use pops.numerics.reconstruction.limiters.VanLeer()", source=source),
        _row("elliptic:geometric_mg", layout="uniform|amr", backend="production", platform="host",
             mpi=mpi, gpu=gpu, limitation="native multigrid route; supports variable epsilon",
             source=source),
        _row("elliptic:fft", layout="uniform", backend="production", platform="host", mpi=mpi,
             gpu=gpu, limitation="periodic, constant coefficient, power-of-two uniform grid only",
             source=source),
        _row("elliptic:fft_direct_dft_fallback", layout="uniform", backend="production",
             platform="host", mpi=mpi, gpu=gpu, status="partial",
             limitation=("non-power-of-two Nx/Ny remain correct by falling back to direct O(n^2) "
                         "DFT; fallback_diagnostics_report exposes the policy and count"),
             source=source),
        _row("elliptic:mg_fac_defaults", layout="uniform|amr", backend="production",
             platform="host", mpi=mpi, gpu=gpu, status="partial",
             limitation=("geometric MG/FAC defaults and debug diagnostics are still header-local; "
                         "central SolverDefaults/logger follow-up is required"),
             source=source),
        _row("elliptic:fft_amr", layout="amr", backend="none", status="unavailable",
             limitation="FFT requires a single uniform periodic mesh, not AMR",
             requested="solver=FFT() with layout=AMR", available_route="GeometricMG() on AMR",
             alternative="use pops.solvers.elliptic.GeometricMG()", source=source),
        _row("mesh:2d_storage_arithmetic", layout="uniform|amr", backend="production",
             platform="host", mpi=mpi, gpu=gpu, status="partial",
             limitation=("native mesh/storage/arithmetic primitives are Box2D/Fab2D/MultiFab 2D; "
                         "Dim!=2 is rejected by validate_dimension() before runtime"),
             source=source),
        _row("amr:refinement_ratio", layout="amr", backend="production", platform="host",
             mpi=mpi, gpu=gpu, status="partial",
             limitation=("AMR hierarchy, patch ranges, reflux and subcycling are ratio=2 only; "
                         "validate_amr_refinement_ratio() rejects other ratios"),
             source=source),
        _row("parallel:mpi_world_communicator", layout="uniform|amr", backend="production",
             platform="mpi", mpi=mpi, status="partial",
             limitation=("MPI collectives use MPI_COMM_WORLD; a caller-provided communicator is not "
                         "a supported native route yet"),
             source=source),
        _row("parallel:custom_communicator", layout="uniform|amr", backend="none",
             platform="mpi", mpi=mpi, status="unavailable",
             limitation="no native route accepts a caller-provided MPI_Comm",
             requested="communicator != MPI_COMM_WORLD",
             available_route="MPI_COMM_WORLD or serial",
             alternative="run on MPI_COMM_WORLD until ParallelContext lands", source=source),
        _row("precision:single_or_mixed", layout="uniform|amr", backend="none",
             platform="host", status="unavailable",
             limitation="pops::Real is hardcoded to double; no PrecisionPolicy route exists",
             requested="precision=single or precision=mixed",
             available_route="precision=double",
             alternative="use double precision or implement a native PrecisionPolicy", source=source),
        _row("runtime:kokkos_lifecycle", layout="uniform|amr", backend="production",
             platform="host|gpu", mpi=mpi, gpu=gpu, status="partial",
             limitation=("Kokkos is lazily initialized by PoPS on first allocation/kernel unless "
                         "the caller already initialized it; runtime_environment_report() exposes "
                         "ownership and initialized/finalized state"),
             source=source),
        _row("runtime:allocator_lifetime", layout="uniform|amr", backend="production",
             platform="host|gpu", mpi=mpi, gpu=gpu, status="partial",
             limitation=("Kokkos builds use a process-lifetime ManagedArena; blocks are released "
                         "by a Kokkos finalize hook and the arena tables intentionally survive "
                         "process teardown"),
             source=source),
        _row("krylov:cg_bicgstab_gmres_richardson", layout="uniform|amr", backend="production",
             platform="host", mpi=mpi, gpu=gpu,
             limitation="matrix-free Krylov over native MultiFab primitives", source=source),
        _row("schur:condensed_source", layout="uniform|amr", backend="production",
             platform="host", mpi=mpi, gpu=gpu, status="partial",
             limitation=("Schur condensation/source kernels are specialised to 2D plus Bz/Lorentz "
                         "coupling; no generic 3D route"),
             source=source),
        _row("program_context:system", layout="uniform", backend="production", platform="host",
             mpi=mpi, gpu=gpu, limitation="compiled ProgramContext install on System",
             source=source),
        _row("program_context:amr", layout="amr", backend="production", platform="host",
             flags=flags, flag="supports_amr", mpi=mpi, gpu=gpu,
             limitation="AMR program install requires target='amr_system'", source=source),
        _row("runtime:native_loader_legacy_metadata", layout="uniform|amr",
             backend="aot|dynamic|prototype", platform="host", mpi=mpi, gpu=gpu, status="partial",
             limitation=("old native modules without metadata fall back to u0.. names, empty roles, "
                         "legacy default gamma and host prototype copies"),
             source=source),
        _row("output:npz_vtk_hdf5", layout="uniform|amr", backend="runtime", platform="host",
             mpi=mpi, limitation="runtime output writers; AMR VTK is coarse + patch metadata",
             source=source),
        _row("output:plotfile_uniform", layout="uniform", backend="none", status="unavailable",
             limitation="Plotfile is an AMR per-level format; Uniform System has no writer",
             requested="OutputPolicy(format=Plotfile()) on Uniform",
             available_route="HDF5() or npz on Uniform",
             alternative="use HDF5() or bind an AMR output route", source=source),
        _row("checkpoint:system_v1", layout="uniform", backend="runtime", platform="host",
             mpi=mpi, limitation="npz rank-0 gather checkpoint/restart v1", source=source),
        _row("checkpoint:parallel_hdf5", layout="uniform|amr", backend="none",
             platform="mpi", status="unavailable",
             limitation="parallel HDF5 checkpoint is not a native checkpoint route",
             requested="checkpoint(parallel=True)",
             available_route="checkpoint(parallel=False) or write(format='hdf5', parallel=True)",
             alternative="use checkpoint(parallel=False)", source=source),
        _row("checkpoint:amr_dynamic_regrid", layout="amr", backend="none", status="unavailable",
             limitation="bit-identical AMR checkpoint requires a frozen hierarchy (regrid_every=0)",
             requested="AMR checkpoint with dynamic regrid",
             available_route="AMR checkpoint with regrid_every=0",
             alternative="use AmrSystem.write for visualization or freeze regrid", source=source),
    ]


def native_capability_matrix(*, owner="module", layout="module", target="module",
                             flags=None, source=None):
    """Return the ADC-549 native route matrix.

    ``flags`` can be supplied by a compiled artifact manifest. When absent, the built module's
    C++ ``module_capabilities(target)`` is used. The returned rows always expose:
    feature, layout, backend, platform, MPI, GPU, status, limitation and error_message.
    """
    report = native_capability_report(target, flags=flags, source=source)
    return CapabilityRouteMatrix(
        owner, layout, report.routes, schema_version=report.schema_version,
        abi_version=report.abi_version, target=report.target, abi_key=report.abi_key,
        platform=report.platform)
