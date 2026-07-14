#pragma once

/// @file
/// @brief Authoritative STATIC capability facts of the built _pops module (Spec 5 sec.13.12 /
///        sec.13.12.1, criteria #36/#37).
///
/// MOTIVATION. Python's internal descriptor report walks the inert catalog;
/// that walk is "Python-derived, not authoritative" (Spec 5 sec.13.12). The transport capabilities a
/// module actually provides -- which backend it was compiled with, whether MPI / GPU is real, whether
/// the route carries a stride, named aux fields, a partial IMEX mask -- are decided by the C++ build,
/// not by Python. This header sources those facts from the SAME compile-time tokens the module attrs
/// already expose (``POPS_HAS_KOKKOS`` / ``POPS_HAS_MPI`` in init_core.cpp) so the Python read side can
/// cross-check its descriptor walk against the C++ truth and FAIL LOUD on a disagreement.
///
/// HONESTY (non-negotiable, Spec 5 sec.13.12). A capability is reported TRUE only when a C++ path backs
/// it. ``supports_partial_imex_mask`` is FALSE: a tree-wide grep finds NO partial-IMEX-mask code path,
/// so claiming it would be a lie. ``supports_gpu`` is TRUE only under Kokkos AND a real device backend
/// token (CUDA/HIP); a Kokkos-Serial / OpenMP CPU build reports FALSE. ``supports_stride`` is
/// route-dependent (the installed production package carries a stride while the route-agnostic module
/// report does not describe one), so the facts are queried per @p target.
///
/// This is a pure free-function / POD header: no System state, no out-of-line definition (kept out of
/// System::Impl on purpose, so lightweight C++ runtime mocks can share it).

#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/generated_component_catalog.hpp>
#include <pops/runtime/config/generated_release_contract.hpp>
#include <pops/runtime/runtime_environment.hpp>

#include <string>
#include <vector>

namespace pops {

/// Discrete, monotonic ABI revision of the module capability contract. Bump when the SHAPE of
/// ModuleCapabilities (its fields / their meaning) changes, so a per-artifact manifest baked into an
/// older .so (pops_compiled_manifest) can be told apart from a newer module at load time. Distinct from
/// the textual pops::abi_key() (compiler / std / header signature): that detects a toolchain ABI break,
/// this versions the capability *vocabulary*.
inline constexpr int kAbiVersion = 3;
static_assert(kAbiVersion == release_contract::kReleaseNativeAbiVersion,
              "native ABI and generated release contract drifted");

/// Version of the structured capability-report schema exposed by native_capability_report().
/// This versions the report envelope independently from the capability vocabulary above.
inline constexpr int kCapabilityReportSchemaVersion = 1;

/// The lowering route whose static capabilities are queried. ``kProduction`` describes the sole
/// compiled-package route; ``kModule`` reports route-agnostic facts and therefore leaves stride false.
enum class CapabilityTarget { kModule, kProduction };

/// The STATIC transport capabilities the built _pops module provides (Spec 5 sec.13.12). A small POD of
/// booleans + the ABI version, sourced from compile-time tokens only -- it allocates nothing, touches no
/// System, runs no kernel.
struct ModuleCapabilities {
  int abi_version;                  ///< pops::kAbiVersion (this build's capability-contract revision).
  bool supports_uniform;            ///< single-level uniform grid (always available).
  bool supports_amr;                ///< adaptive mesh refinement runtime (AmrSystem; always built in).
  bool supports_mpi;                ///< real MPI transport (POPS_HAS_MPI); false on a serial module.
  bool supports_gpu;               ///< real GPU device backend (Kokkos AND a CUDA/HIP token).
  bool supports_stride;             ///< the route carries a cell stride (production: yes; aot: no).
  bool supports_named_fields;       ///< named aux-field transport (named_aux, aux_field; always built).
  bool supports_partial_imex_mask;  ///< partial IMEX mask -- FALSE: no C++ path backs it (do not lie).
};

/// One native route/capability row in the structured report. ``status`` is one of
/// ``available``, ``partial`` or ``unavailable``. ``reason`` is the machine-readable limitation text
/// pretty printers render later; callers must not parse a formatted report string to recover it.
struct CapabilityRouteReport {
  std::string route_id;
  std::string feature;
  std::string layout;
  std::string backend;
  std::string platform;
  bool mpi = false;
  bool gpu = false;
  std::string status = "unknown";
  std::string reason;
  std::string requested;
  std::string available_route;
  std::string alternative;
  std::string source = "native";
};

/// Versioned native capability report. The legacy module_capabilities() bool dict is retained as a
/// compatibility projection of this object; new inspection paths should consume this report.
struct NativeCapabilityReport {
  int schema_version = kCapabilityReportSchemaVersion;
  int abi_version = kAbiVersion;
  std::string target = "module";
  std::string abi_key;
  std::string platform = "host";
  ModuleCapabilities capabilities{};
  RuntimeEnvironmentReport runtime{};
  std::vector<CapabilityRouteReport> routes;
};

namespace detail {

/// True iff this translation unit is compiled for a real GPU device backend. Conservative and honest:
/// a Kokkos build is necessary but NOT sufficient (Kokkos-Serial / OpenMP is a CPU build). __CUDACC__ /
/// __HIPCC__ are the device-compiler tokens (cf. core/foundation/types.hpp); absent them we report
/// false rather than fabricate GPU support from the mere presence of Kokkos.
inline constexpr bool kHasGpuBackend =
#if defined(POPS_HAS_KOKKOS) && (defined(__CUDACC__) || defined(__HIPCC__))
    true;
#else
    false;
#endif

inline constexpr bool kHasMpi =
#if defined(POPS_HAS_MPI)
    true;
#else
    false;
#endif

}  // namespace detail

/// The module's STATIC capability facts for a given lowering route @p target (Spec 5 sec.13.12 / #36).
///
/// All values come from compile-time tokens, never a Python computation:
///   - ``abi_version`` = pops::kAbiVersion;
///   - ``supports_uniform`` / ``supports_amr`` = true (both runtimes are built into _pops);
///   - ``supports_mpi`` = POPS_HAS_MPI;
///   - ``supports_gpu`` = POPS_HAS_KOKKOS AND a device token (else false, conservatively honest);
///   - ``supports_stride`` = true for the production package and false for the route-agnostic
///     ``kModule`` query;
///   - ``supports_named_fields`` = true (the named-aux transport exists, kAuxNamedBase / aux_field);
///   - ``supports_partial_imex_mask`` = false (NO C++ path backs it -- reporting true would be a lie).
inline ModuleCapabilities module_capabilities(CapabilityTarget target = CapabilityTarget::kModule) {
  ModuleCapabilities caps{};
  caps.abi_version = kAbiVersion;
  caps.supports_uniform = true;
  caps.supports_amr = true;
  caps.supports_mpi = detail::kHasMpi;
  caps.supports_gpu = detail::kHasGpuBackend;
  caps.supports_stride = (target == CapabilityTarget::kProduction);
  caps.supports_named_fields = true;
  caps.supports_partial_imex_mask = false;
  return caps;
}

inline const char* capability_target_name(CapabilityTarget target) {
  switch (target) {
    case CapabilityTarget::kProduction:
      return "production";
    case CapabilityTarget::kModule:
    default:
      return "module";
  }
}

inline CapabilityRouteReport capability_route(std::string feature, std::string status,
                                              std::string reason,
                                              std::string layout = kLayoutRouteTokensCsv,
                                              std::string backend = "production",
                                              std::string platform = "host", bool mpi = false,
                                              bool gpu = false, std::string requested = {},
                                              std::string available_route = {},
                                              std::string alternative = {}) {
  CapabilityRouteReport row{};
  row.route_id = feature;
  row.feature = std::move(feature);
  row.layout = std::move(layout);
  row.backend = std::move(backend);
  row.platform = std::move(platform);
  row.mpi = mpi;
  row.gpu = gpu;
  row.status = std::move(status);
  row.reason = std::move(reason);
  row.requested = std::move(requested);
  row.available_route = std::move(available_route);
  row.alternative = std::move(alternative);
  return row;
}

inline std::string status_from_bool(bool supported) {
  return supported ? "available" : "unavailable";
}

inline std::vector<CapabilityRouteReport> native_capability_routes(
    const ModuleCapabilities& caps, const RuntimeEnvironmentReport& env) {
  const bool mpi = caps.supports_mpi;
  const bool gpu = caps.supports_gpu;
  const std::string amr_note =
      "hierarchy depth is resource-policy controlled; native ratio=2";
  return {
      capability_route("supports_uniform", status_from_bool(caps.supports_uniform),
                       "single-level Uniform layout", "uniform", "module", "host", mpi, gpu,
                       "layout=Uniform", "layout=Uniform"),
      capability_route("supports_amr", status_from_bool(caps.supports_amr), amr_note, "amr",
                       "production", "host", mpi, gpu, "layout=AMR",
                       "backend='production' target='amr_system'",
                       "use Uniform or AMR(ratio=2)"),
      capability_route("supports_mpi", status_from_bool(caps.supports_mpi),
                       "MPI transport is compiled only when POPS_USE_MPI=ON", kLayoutRouteTokensCsv,
                       "production", "mpi", mpi, gpu, "platform=MPI", "serial/OpenMP build",
                       "rebuild with -DPOPS_USE_MPI=ON"),
      capability_route("supports_gpu", status_from_bool(caps.supports_gpu),
                       "GPU device backend requires a Kokkos CUDA/HIP build", kLayoutRouteTokensCsv,
                       "production", "gpu", mpi, gpu, "platform=GPU", "host CPU platform",
                       "use KokkosOpenMP/KokkosSerial or a CUDA/HIP build"),
      capability_route("supports_stride", status_from_bool(caps.supports_stride),
                       "real cell stride is carried only by the production/native route",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu, "strided cell access",
                       "backend='production'", "compile with backend='production'"),
      capability_route("supports_named_fields", status_from_bool(caps.supports_named_fields),
                       "named aux-field transport", kLayoutRouteTokensCsv, "production", "host", mpi,
                       gpu, "named aux fields", "native named-field transport"),
      capability_route("supports_partial_imex_mask",
                       status_from_bool(caps.supports_partial_imex_mask),
                       "no C++ route backs a partial IMEX mask", kLayoutRouteTokensCsv, "production",
                       "host", mpi, gpu, "partial IMEX mask",
                       "full source implicit / split routes",
                       "use IMEX/IMEXRK/Split without partial masks"),
      capability_route("supports_custom_communicator",
                       status_from_bool(env.supports_custom_communicator),
                       "no C++ route accepts a caller-provided MPI_Comm", kLayoutRouteTokensCsv, "none",
                       "mpi", mpi, gpu, "communicator != MPI_COMM_WORLD",
                       "MPI_COMM_WORLD or serial",
                       "run on MPI_COMM_WORLD until ParallelContext lands"),
      capability_route("layout:Uniform", "available", "2D single-level Cartesian/Polar layout",
                       "uniform", "module", "host", mpi, gpu),
      capability_route("layout:AMR", status_from_bool(caps.supports_amr),
                       "resource-policy-controlled depth and native ratio=2", "amr",
                       "production", "host", mpi, gpu, "AMR(ratio!=2)", "AMR(ratio=2)",
                       "use Uniform or the native AMR envelope"),
      capability_route("spatial:finite_volume", "available",
                       "2D finite-volume production route", kLayoutRouteTokensCsv,
                       "production", "host", mpi, gpu),
      capability_route("riemann:rusanov", "available", "requires model max_wave_speed",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("riemann:hll", "available", "requires physical_flux and wave_speeds",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("riemann:hllc", "available",
                       "requires Euler/HLLC model capabilities; polar route is unavailable",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("riemann:roe", "available",
                       "requires Roe dissipation capability; polar route is unavailable",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("reconstruction:firstorder", "available", "ghost_depth=1",
                       kLayoutRouteTokensCsv,
                       "production", "host", mpi, gpu),
      capability_route("reconstruction:muscl", "available",
                       "ghost_depth=2; native limiters minmod/vanleer", kLayoutRouteTokensCsv,
                       "production", "host", mpi, gpu),
      capability_route("reconstruction:weno5", "available", "ghost_depth=3; high-order route is native",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("limiter:mc", "unavailable",
                       "catalogued but no native C++ limiter symbol exists", kLayoutRouteTokensCsv,
                       "none",
                       "host", mpi, gpu, "limiter=MC()", "Minmod() or VanLeer()",
                       "use pops.numerics.reconstruction.limiters.Minmod()"),
      capability_route("limiter:superbee", "unavailable",
                       "catalogued but no native C++ limiter symbol exists", kLayoutRouteTokensCsv,
                       "none",
                       "host", mpi, gpu, "limiter=Superbee()", "Minmod() or VanLeer()",
                       "use pops.numerics.reconstruction.limiters.VanLeer()"),
      capability_route("elliptic:geometric_mg", "available",
                       "native multigrid route; supports variable epsilon", kLayoutRouteTokensCsv,
                       "production", "host", mpi, gpu),
      capability_route("elliptic:fft", "available",
                       "periodic, constant coefficient, power-of-two uniform grid only", "uniform",
                       "production", "host", mpi, gpu),
      capability_route("elliptic:fft_direct_dft_fallback", "partial",
                       "non-power-of-two Nx/Ny remain correct by direct O(n^2) DFT; "
                       "fallback_diagnostics_report exposes the policy and count",
                       "uniform", "production", "host", mpi, gpu),
      capability_route("elliptic:mg_fac_defaults", "partial",
                       "geometric MG/FAC defaults and debug diagnostics are still header-local; "
                       "central SolverDefaults/logger follow-up is required",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("elliptic:fft_amr", "unavailable",
                       "FFT requires a single uniform periodic mesh, not AMR", "amr", "none",
                       "host", mpi, gpu, "solver=FFT() with layout=AMR", "GeometricMG() on AMR",
                       "use pops.solvers.elliptic.GeometricMG()"),
      capability_route("mesh:2d_storage_arithmetic", "partial",
                       "native mesh/storage/arithmetic primitives are Box2D/Fab2D/MultiFab 2D; "
                       "Dim!=2 is rejected before runtime",
                       kLayoutRouteTokensCsv, "production", "host", mpi, gpu),
      capability_route("amr:refinement_ratio", "partial",
                       "AMR hierarchy, patch ranges, reflux and subcycling are ratio=2 only; "
                       "validate_amr_refinement_ratio() rejects other ratios",
                       "amr", "production", "host", mpi, gpu),
      capability_route("parallel:mpi_world_communicator", "partial",
                       "MPI collectives use MPI_COMM_WORLD; a caller-provided communicator is not "
                       "a supported native route yet",
                       kLayoutRouteTokensCsv, "production", "mpi", mpi, gpu),
      capability_route("parallel:custom_communicator", "unavailable",
                       "no native route accepts a caller-provided MPI_Comm", kLayoutRouteTokensCsv,
                       "none",
                       "mpi", mpi, gpu, "communicator != MPI_COMM_WORLD", "MPI_COMM_WORLD or serial",
                       "run on MPI_COMM_WORLD until ParallelContext lands"),
      capability_route("precision:single_or_mixed", "unavailable",
                       "pops::Real is hardcoded to double; no PrecisionPolicy route exists",
                       kLayoutRouteTokensCsv, "none", "host", mpi, gpu,
                       "precision=single or precision=mixed", "precision=double",
                       "use double precision or implement a native PrecisionPolicy"),
      capability_route("runtime:kokkos_lifecycle", "partial",
                       "Kokkos is lazily initialized by PoPS on first allocation/kernel unless "
                       "the caller already initialized it",
                       kLayoutRouteTokensCsv, "production", "host|gpu", mpi, gpu),
      capability_route("runtime:allocator_lifetime", "partial",
                       "Kokkos builds use a process-lifetime ManagedArena; blocks are released "
                       "by a Kokkos finalize hook",
                       kLayoutRouteTokensCsv, "production", "host|gpu", mpi, gpu),
      capability_route("krylov:cg_bicgstab_gmres_richardson", "available",
                       "matrix-free Krylov over native MultiFab primitives", kLayoutRouteTokensCsv,
                       "production", "host", mpi, gpu),
      capability_route("program:hierarchy_scoped_solve", "partial",
                       "Program.solve and its provider protocol are physics-independent; AMR "
                       "hierarchy lowering currently supports one top-level linear solve with "
                       "CompositeTensorFAC()",
                       "uniform|amr", "production", "host", mpi, gpu),
      capability_route("program_context:system", "available",
                       "compiled ProgramContext install on System", "uniform", "production", "host",
                       mpi, gpu),
      capability_route("program_context:amr", status_from_bool(caps.supports_amr),
                       "AMR program install requires target='amr_system'", "amr", "production",
                       "host", mpi, gpu),
      capability_route("output:npz_vtk_hdf5", "available",
                       "runtime output writers; AMR VTK is coarse + patch metadata",
                       kLayoutRouteTokensCsv,
                       "runtime", "host", mpi, gpu),
      capability_route("output:plotfile_uniform", "unavailable",
                       "Plotfile is an AMR per-level format; Uniform System has no writer",
                       "uniform", "none", "host", mpi, gpu,
                       "OutputPolicy(format=Plotfile()) on Uniform", "HDF5() or npz on Uniform",
                       "use HDF5() or bind an AMR output route"),
      capability_route("checkpoint:system_v1", "available",
                       "npz rank-0 gather checkpoint/restart v1", "uniform", "runtime", "host", mpi,
                       gpu),
      capability_route("checkpoint:parallel_hdf5", "unavailable",
                       "parallel HDF5 checkpoint is not a native checkpoint route",
                       kLayoutRouteTokensCsv,
                       "none", "mpi", mpi, gpu, "checkpoint(parallel=True)",
                       "checkpoint(parallel=False) or write(format='hdf5', parallel=True)",
                       "use checkpoint(parallel=False)"),
      capability_route("checkpoint:amr_dynamic_regrid", "unavailable",
                       "bit-identical AMR checkpoint requires a frozen hierarchy", "amr", "none",
                       "host", mpi, gpu, "AMR checkpoint with dynamic regrid",
                       "AMR checkpoint with regrid_every=0",
                       "use AmrSystem.write for visualization or freeze regrid"),
  };
}

inline NativeCapabilityReport native_capability_report(
    CapabilityTarget target = CapabilityTarget::kModule) {
  NativeCapabilityReport report{};
  report.schema_version = kCapabilityReportSchemaVersion;
  report.abi_version = kAbiVersion;
  report.target = capability_target_name(target);
  report.abi_key = detail::abi_key_string();
  report.capabilities = module_capabilities(target);
  report.runtime = runtime_environment_report();
  if (report.runtime.mpi_compiled) {
    report.platform = "mpi";
  } else if (report.runtime.has_kokkos) {
    report.platform = "kokkos";
  } else {
    report.platform = "serial";
  }
  report.routes = native_capability_routes(report.capabilities, report.runtime);
  return report;
}

}  // namespace pops
