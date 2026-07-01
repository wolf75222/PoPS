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

// Composite FAC defaults.
inline constexpr int kFACDefaultMaxIters = 30;
inline constexpr int kFACDefaultFineSweeps = 400;
inline constexpr Real kFACDefaultTol = Real(1e-9);
inline constexpr Real kFACInitialCoarseRelTol = Real(1e-12);
inline constexpr int kFACInitialCoarseMaxCycles = 100;

// FFT Poisson route facts.
inline constexpr bool kFFTDefaultSpectral = false;
inline constexpr bool kFFTZeroMeanGauge = true;
inline constexpr bool kFFTDirectDftFallback = true;

// FV / EB numerics.
inline constexpr Real kEbCutFractionFloor = Real(1e-3);
inline constexpr Real kWenoEpsilon = Real(1e-40);

// AMR / runtime policy defaults.
inline constexpr int kAmrDefaultMaxLevels = 2;
inline constexpr Real kAmrRefinementDisabledThreshold = Real(1e30);
inline constexpr Real kAmrPhiRefinementDisabledThreshold = Real(0);
inline constexpr Real kAdaptiveNoEvolvingBlockSentinel = Real(1e30);

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
  double abs_tol = static_cast<double>(kMGDefaultAbsTol);
  bool has_epsilon_field = false;
  bool has_anisotropic_epsilon = false;
  bool has_reaction_field = false;
};

struct EffectiveSourceStageOptions {
  std::string block;
  std::string kind;
  std::string geometry;
  double theta = 0.5;
  double alpha = static_cast<double>(kPhysicalDefaultAlpha);
  double requested_krylov_tol = 0.0;
  int requested_krylov_max_iters = 0;
  double effective_krylov_tol = static_cast<double>(kKrylovDefaultRelTol);
  int effective_krylov_max_iters = kSchurKrylovCartesianMaxIters;
  std::string density;
  std::string momentum_x;
  std::string momentum_y;
  std::string energy;
  int bz_aux_component = -1;
};

struct EffectiveRefinementOptions {
  double threshold = static_cast<double>(kAmrRefinementDisabledThreshold);
  bool disabled = true;
  std::string disabled_policy = "threshold >= amr.refinement_disabled_threshold";
  std::string variable;
  std::string role;
  double phi_grad_threshold = static_cast<double>(kAmrPhiRefinementDisabledThreshold);
  bool phi_refinement_enabled = false;
};

struct EffectiveOptionsReport {
  int schema_version = 1;
  std::string runtime;
  std::vector<EffectiveBlockOptions> blocks;
  EffectivePoissonOptions poisson;
  std::vector<EffectiveSourceStageOptions> source_stages;
  std::string time_scheme = "lie";
  std::string gauss_policy = "restart";
  bool has_amr = false;
  EffectiveRefinementOptions amr_refinement;
};

}  // namespace pops
