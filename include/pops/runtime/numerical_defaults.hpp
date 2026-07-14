#pragma once

#include <pops/core/foundation/types.hpp>

#include <string>
#include <vector>

/// @file
/// @brief Central numerical/physical defaults and small PODs used by inspection reports.
///
/// These values are the user-visible defaults of the native runtime. Keeping them in one header
/// avoids silent drift between pybind defaults, C++ facade defaults, solver fallbacks and reports.
/// The constants intentionally preserve historical values; this header makes them inspectable.

namespace pops {

// Newton / IMEX source solve.
inline constexpr int kNewtonFailNone = 0;
inline constexpr int kNewtonFailWarn = 1;
inline constexpr int kNewtonFailThrow = 2;
inline constexpr int kNewtonDefaultMaxIters = 2;
inline constexpr Real kNewtonDefaultRelTol = Real(0);
inline constexpr Real kNewtonDefaultAbsTol = Real(0);
inline constexpr Real kNewtonDefaultFdEps = Real(1e-7);
inline constexpr Real kNewtonDefaultDamping = Real(1);
inline constexpr int kNewtonDefaultFailPolicy = kNewtonFailNone;
inline constexpr Real kNewtonFiniteAbsLimit = Real(1e300);

// Krylov family defaults.
inline constexpr Real kKrylovDefaultRelTol = Real(1e-10);
inline constexpr int kTensorKrylovDefaultMaxIters = 200;
inline constexpr int kSchurKrylovCartesianMaxIters = 400;
inline constexpr int kSchurKrylovPolarMaxIters = 600;
inline constexpr Real kKrylovBreakdownTiny = Real(1e-300);

// Geometric multigrid defaults.
inline constexpr Real kMGDefaultRelTol = Real(1e-8);
inline constexpr int kMGDefaultMaxCycles = 50;
inline constexpr Real kMGDefaultAbsTol = Real(0);
inline constexpr int kMGDefaultMinCoarse = 2;
inline constexpr int kMGDefaultPreSmooth = 2;
inline constexpr int kMGDefaultPostSmooth = 2;
inline constexpr int kMGDefaultBottomSweeps = 50;
// ADC-644: the total-cell coarsening ceiling. Coarsening STOPS once a level's total unknown count
// (nx*ny) is at or below this; distinct from kMGDefaultMinCoarse (a per-axis floor). Default sentinel
// 0 = disabled (only min_coarse governs) -> the historical V-cycle hierarchy, bit-identical.
inline constexpr int kMGDefaultCoarseThreshold = 0;

// Composite FAC defaults.
inline constexpr int kFACDefaultMaxIters = 30;
inline constexpr int kFACDefaultFineSweeps = 400;
inline constexpr Real kFACDefaultRelTol = Real(1e-9);
inline constexpr Real kFACDefaultAbsTol = Real(0);
inline constexpr Real kFACInitialCoarseRelTol = Real(1e-12);
inline constexpr Real kFACInitialCoarseAbsTol = Real(0);
inline constexpr int kFACInitialCoarseMaxCycles = 100;

// FFT Poisson route facts.
inline constexpr bool kFFTDefaultSpectral = false;
inline constexpr bool kFFTZeroMeanGauge = true;
inline constexpr bool kFFTDirectDftFallback = true;

// FV / EB numerics.
inline constexpr Real kEbCutFractionFloor = Real(1e-3);
inline constexpr Real kWenoEpsilon = Real(1e-40);
// EB face-open / small-cell thresholds (ADC-615/643). SINGLE SOURCE: the numerics EB operator
// (numerics/spatial/embedded_boundary/operator.hpp) consumes these via cut_fraction.hpp, exactly
// like kEbCutFractionFloor above, so the report, the typed pops.numerics.CutCell descriptor and the
// FV kernels share one default.
inline constexpr Real kEbFaceOpenEps = Real(1e-6);
inline constexpr Real kEbKappaMin = Real(1e-2);

// AMR / runtime policy defaults.
inline constexpr int kAmrDefaultMaxLevels = 2;
inline constexpr Real kAmrRefinementDisabledThreshold = Real(1e30);
inline constexpr Real kAmrPhiRefinementDisabledThreshold = Real(0);
inline constexpr Real kAdaptiveNoEvolvingBlockSentinel = Real(1e30);

// ADC-616/643: Berger-Rigoutsos clustering defaults -- SINGLE SOURCE shared with pops::ClusterParams
// (amr/tagging/cluster.hpp) and EffectiveRefinementOptions below.
inline constexpr double kAmrClusterMinEfficiency = 0.7;
inline constexpr int kAmrClusterMinBoxSize = 1;
inline constexpr int kAmrClusterMaxBoxSize = 32;

// Physical defaults carried by the native brick facades.
inline constexpr Real kPhysicalDefaultB0 = Real(1);
inline constexpr Real kPhysicalDefaultGamma = Real(1.4);
inline constexpr Real kPhysicalDefaultFluidStateCs2 = Real(0.5);
inline constexpr Real kPhysicalDefaultNativeIsothermalCs2 = Real(1);
inline constexpr Real kPhysicalDefaultVacuumFloor = Real(0);
inline constexpr Real kPhysicalDefaultQOverM = Real(1);
inline constexpr Real kPhysicalDefaultChargeQ = Real(1);
inline constexpr Real kPhysicalDefaultAlpha = Real(1);
inline constexpr Real kPhysicalDefaultBackgroundN0 = Real(0);
inline constexpr Real kPhysicalDefaultGravitySign = Real(1);
inline constexpr Real kPhysicalDefaultFourPiG = Real(1);
inline constexpr Real kPhysicalDefaultGravityRho0 = Real(1);

inline const char* newton_fail_policy_name(int policy) {
  if (policy == kNewtonFailNone)
    return "none";
  if (policy == kNewtonFailWarn)
    return "warn";
  if (policy == kNewtonFailThrow)
    return "throw";
  return "invalid";
}

/// The embedded-boundary / cut-cell thresholds (ADC-615): the volume-fraction small-cell floor, the
/// closed-face aperture threshold, and the cut-fraction clamp shared with the elliptic wall. Defaults
/// are the kEb* constants, so a default-constructed EbThresholds reproduces today's EB scheme AND the
/// elliptic Shortley-Weller wall bit-for-bit (the cut_theta_min is passed to BOTH consumers).
struct EbThresholds {
  Real kappa_min = kEbKappaMin;              ///< small-cell volume-fraction floor.
  Real face_open_eps = kEbFaceOpenEps;       ///< aperture below which a face is closed.
  Real cut_theta_min = kEbCutFractionFloor;  ///< cut-fraction clamp (shared elliptic + EB).
};

/// The composite FAC Poisson knobs (ADC-614): the AMR composite elliptic solver's outer iteration
/// budget, per-fine-patch SOR sweeps, mixed composite-residual tolerance, the internal coarse-level
/// GeometricMG tolerance/cycles, and the verbose diagnostics flag. The outer solve stops at
/// max(rel_tol * ||R(0)||inf, abs_tol), where R(0) is the exact composite affine forcing.
struct CompositeFacOptions {
  int max_iters = kFACDefaultMaxIters;               ///< FAC two-way iterations.
  int fine_sweeps = kFACDefaultFineSweeps;           ///< SOR sweeps per fine-patch solve.
  Real rel_tol = kFACDefaultRelTol;                  ///< relative composite-residual tolerance.
  Real abs_tol = kFACDefaultAbsTol;                  ///< absolute composite-residual floor.
  Real coarse_rel_tol = kFACInitialCoarseRelTol;     ///< internal coarse GeometricMG rel_tol.
  Real coarse_abs_tol = kFACInitialCoarseAbsTol;     ///< internal coarse GeometricMG abs_tol.
  int coarse_cycles = kFACInitialCoarseMaxCycles;    ///< internal coarse GeometricMG max_cycles.
  bool verbose = false;                              ///< record the per-iteration residual trace.
};

struct EffectiveNewtonOptions {
  int max_iters = kNewtonDefaultMaxIters;
  double rel_tol = static_cast<double>(kNewtonDefaultRelTol);
  double abs_tol = static_cast<double>(kNewtonDefaultAbsTol);
  double fd_eps = static_cast<double>(kNewtonDefaultFdEps);
  double damping = static_cast<double>(kNewtonDefaultDamping);
  std::string fail_policy = "none";
  bool diagnostics = false;
  bool non_default = false;
};

struct EffectiveBlockOptions {
  std::string name;
  std::string route;
  bool compiled = false;
  std::string transport;
  std::string source;
  std::string elliptic;
  std::string limiter;
  std::string riemann;
  std::string recon;
  std::string time;
  std::string time_method;
  bool imex = false;
  int substeps = 1;
  int stride = 1;
  bool evolve = true;
  int ncomp = 0;
  int n_ghost = 2;
  std::vector<std::string> conservative_vars;
  std::vector<std::string> primitive_vars;
  std::vector<std::string> implicit_vars;
  std::vector<std::string> implicit_roles;
  EffectiveNewtonOptions newton;
  double positivity_floor = 0.0;
  bool wave_speed_cache = false;
  double gamma = static_cast<double>(kPhysicalDefaultGamma);
  double B0 = static_cast<double>(kPhysicalDefaultB0);
  double cs2 = static_cast<double>(kPhysicalDefaultFluidStateCs2);
  double vacuum_floor = static_cast<double>(kPhysicalDefaultVacuumFloor);
  double qom = static_cast<double>(kPhysicalDefaultQOverM);
  double q = static_cast<double>(kPhysicalDefaultChargeQ);
  double alpha = static_cast<double>(kPhysicalDefaultAlpha);
  double n0 = static_cast<double>(kPhysicalDefaultBackgroundN0);
  double sign = static_cast<double>(kPhysicalDefaultGravitySign);
  double four_pi_G = static_cast<double>(kPhysicalDefaultFourPiG);
  double rho0 = static_cast<double>(kPhysicalDefaultGravityRho0);
};

struct EffectivePoissonOptions {
  std::string rhs = "charge_density";
  std::string solver = "geometric_mg";
  std::string bc = "auto";
  std::string wall = "none";
  double wall_radius = 0.0;
  double epsilon = 1.0;
  // GeometricMG V-cycle knobs (ADC-613): defaults are the kMG* constants, so an unconfigured System
  // reports (and runs) the historical V-cycle. Populated from the resolved GeometricMgOptions.
  double rel_tol = static_cast<double>(kMGDefaultRelTol);
  double abs_tol = static_cast<double>(kMGDefaultAbsTol);
  int max_cycles = kMGDefaultMaxCycles;
  int min_coarse = kMGDefaultMinCoarse;
  int pre_smooth = kMGDefaultPreSmooth;
  int post_smooth = kMGDefaultPostSmooth;
  int bottom_sweeps = kMGDefaultBottomSweeps;
  int coarse_threshold = kMGDefaultCoarseThreshold;  ///< ADC-644: total-cell coarsening ceiling (0 = off).
  std::string smoother = "red_black_gauss_seidel";
  std::string coarse = "direct_small_grid";
  bool has_epsilon_field = false;
  bool has_anisotropic_epsilon = false;
  bool has_reaction_field = false;
};

struct EffectiveRefinementOptions {
  double threshold = static_cast<double>(kAmrRefinementDisabledThreshold);
  bool disabled = true;
  std::string disabled_policy = "threshold >= amr.refinement_disabled_threshold";
  std::string variable;
  std::string role;
  double phi_grad_threshold = static_cast<double>(kAmrPhiRefinementDisabledThreshold);
  bool phi_refinement_enabled = false;
  // ADC-616: effective Berger-Rigoutsos clustering params (default {0.7, 1, 32} unless overridden).
  double cluster_min_efficiency = kAmrClusterMinEfficiency;
  int cluster_min_box_size = kAmrClusterMinBoxSize;
  int cluster_max_box_size = kAmrClusterMaxBoxSize;
};

/// The EFFECTIVE embedded-boundary thresholds (ADC-615): default or overridden. enabled=false when
/// no cut-cell domain is configured (the fields then carry the kEb* defaults, inert). geometry_mode
/// mirrors the disc transport routing ("none" / "staircase" / "cutcell").
struct EffectiveEbOptions {
  bool enabled = false;
  std::string geometry_mode = "none";
  double kappa_min = static_cast<double>(kEbKappaMin);
  double face_open_eps = static_cast<double>(kEbFaceOpenEps);
  double cut_theta_min = static_cast<double>(kEbCutFractionFloor);
};

struct EffectiveOptionsReport {
  int schema_version = 1;
  std::string runtime;
  std::vector<EffectiveBlockOptions> blocks;
  EffectivePoissonOptions poisson;
  bool has_amr = false;
  EffectiveRefinementOptions amr_refinement;
  EffectiveEbOptions eb;  ///< ADC-615: effective cut-cell / EB thresholds.
};

}  // namespace pops
