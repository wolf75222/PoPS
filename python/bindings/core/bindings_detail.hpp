#pragma once
// Shared surface for the split pybind11 bindings of `_pops` (ADC-365). bindings.cpp is the thin
// PYBIND11_MODULE that calls init_core / init_system / init_amr; each lives in its own TU so the
// py::class_/.def template instantiations compile in parallel (better incremental, lower peak pybind
// memory per TU). This header carries the common includes, the small array/POD helpers (moved verbatim
// from the old monolithic bindings.cpp), and the init_* declarations.

#include <pybind11/functional.h>  // std::function<double()> <- Python callable (add_dt_bound)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/core/foundation/kokkos_env.hpp>  // Kokkos_Core under POPS_HAS_KOKKOS (kokkos_is_initialized)
#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/parallel/comm.hpp>  // pops::my_rank / n_ranks: rank-0 guard of the multi-rank IO facade
#include <pops/runtime/dynamic/abi_key.hpp>  // pops::abi_key: ABI key exposed to the DSL ("production" path)
#include <pops/runtime/config/runtime_params.hpp>          // kMaxRuntimeParams (ADC-618 hard_limit)
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>  // DFT-fallback counter (ADC-618 diagnostic)
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/system.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>  // std::tuple: argument of AmrSystem.set_hierarchy (patch_boxes boxes) (ADC-65)
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pops;

// field (ny*nx row-major, j slow / i fast) -> numpy array (ny, nx) (copy). We size the buffer
// with BOTH real extents of the index domain (rows = ny, cols = nx): square n x n in Cartesian
// (UNCHANGED), but nr x ntheta in polar where nr != ntheta. A square reshape (n, n) would allocate nx^2
// slots for ny*nx values -> memcpy overflows the numpy buffer (heap overflow, crash at teardown). We
// CHECK buffer size == source size before the memcpy (explicit guard).
inline py::array_t<double> to_2d(const std::vector<double>& v, int rows, int cols) {
  py::array_t<double> a({rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("pops (bindings): field size (" + std::to_string(v.size()) +
                             ") != rows*cols (" + std::to_string(rows) + "*" +
                             std::to_string(cols) + "); inconsistent 2D reshape");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
// state (ncomp*ny*nx, component-major order, j slow / i fast) -> numpy array (ncomp, ny, nx).
// Same guard as to_2d: rows = ny, cols = nx (square in Cartesian, nr x ntheta in polar).
inline py::array_t<double> to_3d(const std::vector<double>& v, int ncomp, int rows, int cols) {
  py::array_t<double> a({ncomp, rows, cols});
  if (static_cast<std::size_t>(a.size()) != v.size())
    throw std::runtime_error("pops (bindings): state size (" + std::to_string(v.size()) +
                             ") != ncomp*rows*cols (" + std::to_string(ncomp) + "*" +
                             std::to_string(rows) + "*" + std::to_string(cols) +
                             "); inconsistent 3D reshape");
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}
inline py::tuple output_pieces_to_python(const std::vector<OutputPiece>& pieces) {
  py::tuple result(pieces.size());
  for (std::size_t index = 0; index < pieces.size(); ++index) {
    const OutputPiece& piece = pieces[index];
    const int nx = piece.box.ihi - piece.box.ilo + 1;
    const int ny = piece.box.jhi - piece.box.jlo + 1;
    if (piece.ncomp < 1 || nx < 1 || ny < 1 ||
        piece.values.size() != static_cast<std::size_t>(piece.ncomp) *
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx))
      throw std::runtime_error("native output piece has an inconsistent compact shape");
    py::dict row;
    row["lower"] = py::make_tuple(piece.box.jlo, piece.box.ilo);
    row["upper"] = py::make_tuple(piece.box.jhi + 1, piece.box.ihi + 1);
    row["values"] = to_3d(piece.values, piece.ncomp, ny, nx);
    row["global_box_index"] = piece.global_box_index;
    row["owner_rank"] = piece.owner_rank;
    row["replicated"] = piece.replicated;
    result[index] = std::move(row);
  }
  return result;
}
inline std::vector<double> flat(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

// ADC-214: the Python SURFACE keeps the newton_fail_policy kwarg as a STRING ("none"/"warn"/"throw");
// the NewtonOptions POD carries an integer (NewtonOptions::kFail*). This conversion table therefore
// lives in the bindings (where the flat kwargs are assembled into a POD), with the SAME explicit error
// message as before this work. @p where names the calling method in the message.
inline int newton_fail_policy_from_string(const std::string& policy, const char* where) {
  if (policy == "none")
    return NewtonOptions::kFailNone;
  if (policy == "warn")
    return NewtonOptions::kFailWarn;
  if (policy == "throw")
    return NewtonOptions::kFailThrow;
  throw std::runtime_error(std::string(where) +
                           ": newton_fail_policy 'none'|'warn'|'throw' (got '" + policy + "')");
}

inline py::dict profile_snapshot_to_dict(
    const pops::runtime::program::Profiler::Snapshot& snapshot) {
  py::list scopes;
  for (const auto& scope : snapshot.scopes) {
    py::dict row;
    row["name"] = scope.name;
    row["count"] = scope.count;
    row["total_s"] = scope.total_s;
    row["mean_s"] = scope.mean_s;
    row["min_s"] = scope.min_s;
    row["max_s"] = scope.max_s;
    scopes.append(row);
  }
  py::list counters;
  for (const auto& counter : snapshot.counters) {
    py::dict row;
    row["name"] = counter.name;
    row["value"] = counter.value;
    counters.append(row);
  }
  py::dict out;
  out["schema_version"] = snapshot.schema_version;
  out["enabled"] = snapshot.enabled;
  out["total_s"] = snapshot.total_s;
  out["scopes"] = scopes;
  out["counters"] = counters;
  return out;
}

inline py::dict numerical_defaults_report_to_dict() {
  py::dict newton;
  newton["max_iters"] = kNewtonDefaultMaxIters;
  newton["rel_tol"] = static_cast<double>(kNewtonDefaultRelTol);
  newton["abs_tol"] = static_cast<double>(kNewtonDefaultAbsTol);
  newton["fd_eps"] = static_cast<double>(kNewtonDefaultFdEps);
  newton["damping"] = static_cast<double>(kNewtonDefaultDamping);
  newton["fail_policy"] = newton_fail_policy_name(kNewtonDefaultFailPolicy);
  newton["finite_abs_limit"] = static_cast<double>(kNewtonFiniteAbsLimit);

  py::dict krylov;
  krylov["rel_tol"] = static_cast<double>(kKrylovDefaultRelTol);
  krylov["tensor_max_iters"] = kTensorKrylovDefaultMaxIters;
  krylov["schur_cartesian_max_iters"] = kSchurKrylovCartesianMaxIters;
  krylov["schur_polar_max_iters"] = kSchurKrylovPolarMaxIters;
  krylov["breakdown_tiny"] = static_cast<double>(kKrylovBreakdownTiny);

  py::dict mg;
  mg["rel_tol"] = static_cast<double>(kMGDefaultRelTol);
  mg["max_cycles"] = kMGDefaultMaxCycles;
  mg["abs_tol"] = static_cast<double>(kMGDefaultAbsTol);
  mg["min_coarse"] = kMGDefaultMinCoarse;
  mg["pre_smooth"] = kMGDefaultPreSmooth;
  mg["post_smooth"] = kMGDefaultPostSmooth;
  mg["bottom_sweeps"] = kMGDefaultBottomSweeps;
  mg["coarse_threshold"] = kMGDefaultCoarseThreshold;  // ADC-644: total-cell coarsening ceiling.

  py::dict fac;
  fac["max_iters"] = kFACDefaultMaxIters;
  fac["fine_sweeps"] = kFACDefaultFineSweeps;
  fac["rel_tol"] = static_cast<double>(kFACDefaultRelTol);
  fac["abs_tol"] = static_cast<double>(kFACDefaultAbsTol);
  fac["coarse_rel_tol"] = static_cast<double>(kFACInitialCoarseRelTol);
  fac["coarse_abs_tol"] = static_cast<double>(kFACInitialCoarseAbsTol);
  fac["coarse_cycles"] = kFACInitialCoarseMaxCycles;

  py::dict fft;
  fft["spectral_default"] = kFFTDefaultSpectral;
  fft["zero_mean_gauge"] = kFFTZeroMeanGauge;
  fft["direct_dft_fallback"] = kFFTDirectDftFallback;

  py::dict eb;
  eb["cut_fraction_floor"] = static_cast<double>(kEbCutFractionFloor);
  eb["face_open_eps"] = static_cast<double>(kEbFaceOpenEps);  // ADC-615/618
  eb["kappa_min"] = static_cast<double>(kEbKappaMin);

  py::dict weno;
  weno["epsilon"] = static_cast<double>(kWenoEpsilon);

  py::dict performance;
  performance["cfl_speed_floor"] = static_cast<double>(kCflSpeedFloor);
  performance["adaptive_no_evolving_block_sentinel"] =
      static_cast<double>(kAdaptiveNoEvolvingBlockSentinel);

  py::dict amr;
  amr["max_levels"] = kAmrDefaultMaxLevels;
  amr["refinement_ratio"] = kAmrRefRatio;
  amr["refinement_disabled_threshold"] = static_cast<double>(kAmrRefinementDisabledThreshold);
  amr["phi_refinement_disabled_threshold"] =
      static_cast<double>(kAmrPhiRefinementDisabledThreshold);

  py::dict physical;
  physical["preset"] = "legacy_native_brick_defaults";
  physical["B0"] = static_cast<double>(kPhysicalDefaultB0);
  physical["gamma"] = static_cast<double>(kPhysicalDefaultGamma);
  physical["fluid_state_cs2"] = static_cast<double>(kPhysicalDefaultFluidStateCs2);
  physical["native_brick_isothermal_cs2"] =
      static_cast<double>(kPhysicalDefaultNativeIsothermalCs2);
  physical["vacuum_floor"] = static_cast<double>(kPhysicalDefaultVacuumFloor);
  physical["qom"] = static_cast<double>(kPhysicalDefaultQOverM);
  physical["charge_q"] = static_cast<double>(kPhysicalDefaultChargeQ);
  physical["alpha"] = static_cast<double>(kPhysicalDefaultAlpha);
  physical["n0"] = static_cast<double>(kPhysicalDefaultBackgroundN0);
  physical["gravity_sign"] = static_cast<double>(kPhysicalDefaultGravitySign);
  physical["four_pi_G"] = static_cast<double>(kPhysicalDefaultFourPiG);
  physical["gravity_rho0"] = static_cast<double>(kPhysicalDefaultGravityRho0);
  physical["cs2_note"] =
      "FluidState defaults to 0.5 while the raw native IsothermalFlux brick defaults to 1.0.";

  // ADC-618: hard limits + diagnostics. kMaxRuntimeParams is a fixed-size device carrier bound
  // (native_loader fails fast above it); the DFT-fallback counter records each time the FFT Poisson
  // falls back to the O(n^2) direct DFT on a non-power-of-two grid.
  py::dict runtime;
  runtime["max_runtime_params"] = kMaxRuntimeParams;

  py::dict diagnostics;
  diagnostics["fft_direct_dft_fallback_count"] =
      static_cast<int>(poisson_fft_direct_dft_fallback_count());

  // ADC-618: the CLASSIFICATION fence. EVERY user-visible inline constexpr numeric constant of
  // numerical_defaults.hpp / types.hpp / runtime_params.hpp appears here with an explicit class:
  //   public_knob     -- configurable end to end (a typed descriptor / setter reaches the native use);
  //   internal_default -- a fixed default not (yet) user-configurable, but inspectable;
  //   diagnostic_only  -- a counter / instrumented fact, not a tuning knob;
  //   hard_limit       -- a fixed cap enforced fail-fast (changing it needs a header rebuild).
  // The source-scanning architecture test (tests/python/architecture/test_numeric_constant_fence.py)
  // asserts no constant is missing from this map -> a new user-visible constant cannot ship unclassified.
  py::dict classification;
  auto klass = [&classification](const char* name, const char* cls) { classification[name] = cls; };
  klass("kNewtonFailNone", "internal_default");
  klass("kNewtonFailWarn", "internal_default");
  klass("kNewtonFailThrow", "internal_default");
  klass("kNewtonDefaultMaxIters", "public_knob");
  klass("kNewtonDefaultRelTol", "public_knob");
  klass("kNewtonDefaultAbsTol", "public_knob");
  klass("kNewtonDefaultFdEps", "public_knob");
  klass("kNewtonDefaultDamping", "public_knob");
  klass("kNewtonDefaultFailPolicy", "public_knob");
  klass("kNewtonFiniteAbsLimit", "internal_default");
  klass("kKrylovDefaultRelTol", "public_knob");
  klass("kTensorKrylovDefaultMaxIters", "internal_default");
  klass("kSchurKrylovCartesianMaxIters", "public_knob");
  klass("kSchurKrylovPolarMaxIters", "public_knob");
  klass("kKrylovBreakdownTiny", "internal_default");
  klass("kMGDefaultRelTol", "public_knob");
  klass("kMGDefaultMaxCycles", "public_knob");
  klass("kMGDefaultAbsTol", "public_knob");
  klass("kMGDefaultMinCoarse", "public_knob");
  klass("kMGDefaultPreSmooth", "public_knob");
  klass("kMGDefaultPostSmooth", "public_knob");
  klass("kMGDefaultBottomSweeps", "public_knob");
  klass("kMGDefaultCoarseThreshold", "public_knob");
  klass("kFACDefaultMaxIters", "public_knob");
  klass("kFACDefaultFineSweeps", "public_knob");
  klass("kFACDefaultRelTol", "public_knob");
  klass("kFACDefaultAbsTol", "public_knob");
  klass("kFACInitialCoarseRelTol", "public_knob");
  klass("kFACInitialCoarseAbsTol", "public_knob");
  klass("kFACInitialCoarseMaxCycles", "public_knob");
  klass("kFFTDefaultSpectral", "public_knob");
  klass("kFFTZeroMeanGauge", "internal_default");
  klass("kFFTDirectDftFallback", "diagnostic_only");
  klass("kEbCutFractionFloor", "public_knob");
  klass("kWenoEpsilon", "public_knob");  // ADC-645: WENO5(epsilon=) is wired end to end
  klass("kEbFaceOpenEps", "public_knob");
  klass("kEbKappaMin", "public_knob");
  klass("kAmrDefaultMaxLevels", "internal_default");
  klass("kAmrRefinementDisabledThreshold", "internal_default");
  klass("kAmrPhiRefinementDisabledThreshold", "internal_default");
  klass("kAdaptiveNoEvolvingBlockSentinel", "diagnostic_only");
  klass("kAmrClusterMinEfficiency", "public_knob");
  klass("kAmrClusterMinBoxSize", "public_knob");
  klass("kAmrClusterMaxBoxSize", "public_knob");
  klass("kAmrDriftSpeedFloor", "internal_default");
  klass("kPhysicalDefaultB0", "public_knob");
  klass("kPhysicalDefaultGamma", "public_knob");
  klass("kPhysicalDefaultFluidStateCs2", "public_knob");
  klass("kPhysicalDefaultNativeIsothermalCs2", "internal_default");
  klass("kPhysicalDefaultVacuumFloor", "public_knob");
  klass("kPhysicalDefaultQOverM", "public_knob");
  klass("kPhysicalDefaultChargeQ", "public_knob");
  klass("kPhysicalDefaultAlpha", "public_knob");
  klass("kPhysicalDefaultBackgroundN0", "public_knob");
  klass("kPhysicalDefaultGravitySign", "public_knob");
  klass("kPhysicalDefaultFourPiG", "public_knob");
  klass("kPhysicalDefaultGravityRho0", "public_knob");
  klass("kCflSpeedFloor", "public_knob");  // ADC-645: step_cfl(speed_floor=) is wired end to end
  klass("kMaxRuntimeParams", "hard_limit");

  py::dict out;
  out["schema_version"] = 1;
  out["source"] = "pops.runtime.numerical_defaults";
  out["newton"] = newton;
  out["krylov"] = krylov;
  out["mg"] = mg;
  out["fac"] = fac;
  out["fft"] = fft;
  out["eb"] = eb;
  out["weno"] = weno;
  out["performance"] = performance;
  out["amr"] = amr;
  out["physical"] = physical;
  out["runtime"] = runtime;
  out["diagnostics"] = diagnostics;
  out["classification"] = classification;
  return out;
}

inline py::dict fallback_diagnostics_report_to_dict(const FallbackDiagnosticsReport& report) {
  py::list entries;
  std::size_t total_count = 0;
  for (const FallbackDiagnosticEntry& entry : report.entries) {
    py::dict row;
    row["key"] = entry.key;
    row["route"] = entry.route;
    row["cause"] = entry.cause;
    row["policy"] = entry.policy;
    row["default_action"] = entry.default_action;
    row["impact"] = entry.impact;
    row["frequency"] = entry.frequency;
    row["count"] = entry.count;
    row["explicit_opt_in"] = entry.explicit_opt_in;
    row["performance_degraded"] = entry.performance_degraded;
    row["semantics_changed"] = entry.semantics_changed;
    total_count += entry.count;
    entries.append(row);
  }
  py::dict out;
  out["schema_version"] = report.schema_version;
  out["source"] = report.source;
  out["entries"] = entries;
  out["total_count"] = total_count;
  return out;
}

inline py::dict effective_newton_options_to_dict(const EffectiveNewtonOptions& n) {
  py::dict d;
  d["max_iters"] = n.max_iters;
  d["rel_tol"] = n.rel_tol;
  d["abs_tol"] = n.abs_tol;
  d["fd_eps"] = n.fd_eps;
  d["damping"] = n.damping;
  d["fail_policy"] = n.fail_policy;
  d["diagnostics"] = n.diagnostics;
  d["non_default"] = n.non_default;
  return d;
}

inline py::dict effective_block_options_to_dict(const EffectiveBlockOptions& b) {
  py::dict physical;
  physical["gamma"] = b.gamma;
  physical["B0"] = b.B0;
  physical["cs2"] = b.cs2;
  physical["vacuum_floor"] = b.vacuum_floor;
  physical["qom"] = b.qom;
  physical["q"] = b.q;
  physical["alpha"] = b.alpha;
  physical["n0"] = b.n0;
  physical["sign"] = b.sign;
  physical["four_pi_G"] = b.four_pi_G;
  physical["rho0"] = b.rho0;

  py::dict d;
  d["name"] = b.name;
  d["route"] = b.route;
  d["compiled"] = b.compiled;
  d["transport"] = b.transport;
  d["source"] = b.source;
  d["elliptic"] = b.elliptic;
  d["limiter"] = b.limiter;
  d["riemann"] = b.riemann;
  d["recon"] = b.recon;
  d["time"] = b.time;
  d["time_method"] = b.time_method;
  d["imex"] = b.imex;
  d["substeps"] = b.substeps;
  d["stride"] = b.stride;
  d["evolve"] = b.evolve;
  d["ncomp"] = b.ncomp;
  d["n_ghost"] = b.n_ghost;
  d["conservative_vars"] = py::cast(b.conservative_vars);
  d["primitive_vars"] = py::cast(b.primitive_vars);
  d["implicit_vars"] = py::cast(b.implicit_vars);
  d["implicit_roles"] = py::cast(b.implicit_roles);
  d["newton"] = effective_newton_options_to_dict(b.newton);
  d["positivity_floor"] = b.positivity_floor;
  d["wave_speed_cache"] = b.wave_speed_cache;
  d["physical"] = physical;
  return d;
}

inline py::dict effective_poisson_options_to_dict(const EffectivePoissonOptions& p) {
  py::dict d;
  d["rhs"] = p.rhs;
  d["solver"] = p.solver;
  d["bc"] = p.bc;
  d["wall"] = p.wall;
  d["wall_radius"] = p.wall_radius;
  d["epsilon"] = p.epsilon;
  d["rel_tol"] = p.rel_tol;  // ADC-613: effective GeometricMG V-cycle knobs
  d["abs_tol"] = p.abs_tol;
  d["max_cycles"] = p.max_cycles;
  d["min_coarse"] = p.min_coarse;
  d["pre_smooth"] = p.pre_smooth;
  d["post_smooth"] = p.post_smooth;
  d["bottom_sweeps"] = p.bottom_sweeps;
  d["coarse_threshold"] = p.coarse_threshold;  // ADC-644: total-cell coarsening ceiling.
  d["smoother"] = p.smoother;
  d["coarse"] = p.coarse;
  d["has_epsilon_field"] = p.has_epsilon_field;
  d["has_anisotropic_epsilon"] = p.has_anisotropic_epsilon;
  d["has_reaction_field"] = p.has_reaction_field;
  return d;
}

inline py::dict effective_eb_options_to_dict(const EffectiveEbOptions& e) {
  py::dict d;
  d["enabled"] = e.enabled;
  d["geometry_mode"] = e.geometry_mode;
  d["kappa_min"] = e.kappa_min;
  d["face_open_eps"] = e.face_open_eps;
  d["cut_theta_min"] = e.cut_theta_min;
  return d;
}

inline py::dict effective_refinement_options_to_dict(const EffectiveRefinementOptions& r) {
  py::dict d;
  d["threshold"] = r.threshold;
  d["disabled"] = r.disabled;
  d["disabled_policy"] = r.disabled_policy;
  d["variable"] = r.variable;
  d["role"] = r.role;
  d["phi_grad_threshold"] = r.phi_grad_threshold;
  d["phi_refinement_enabled"] = r.phi_refinement_enabled;
  // ADC-616: effective Berger-Rigoutsos clustering params.
  d["cluster_min_efficiency"] = r.cluster_min_efficiency;
  d["cluster_min_box_size"] = r.cluster_min_box_size;
  d["cluster_max_box_size"] = r.cluster_max_box_size;
  return d;
}

inline py::dict effective_options_report_to_dict(const EffectiveOptionsReport& report) {
  py::list blocks;
  for (const auto& b : report.blocks)
    blocks.append(effective_block_options_to_dict(b));
  py::dict d;
  d["schema_version"] = report.schema_version;
  d["runtime"] = report.runtime;
  d["defaults"] = numerical_defaults_report_to_dict();
  d["blocks"] = blocks;
  d["poisson"] = effective_poisson_options_to_dict(report.poisson);
  d["eb"] = effective_eb_options_to_dict(report.eb);  // ADC-615
  if (report.has_amr)
    d["amr"] = effective_refinement_options_to_dict(report.amr_refinement);
  else
    d["amr"] = py::none();
  return d;
}

// Per-area binding registration, each defined in its own TU (init_core.cpp / init_system.cpp /
// init_amr.cpp). bindings.cpp calls them in this order: init_core registers SystemConfig / ModelSpec
// (used by System / AmrSystem signatures) before init_system / init_amr run.
void init_core(py::module_& m);
void init_identity(py::module_& m);
void init_component_loader(py::module_& m);
void init_parallel_hdf5(py::module_& m);
void init_system(py::module_& m);
void init_amr(py::module_& m);
