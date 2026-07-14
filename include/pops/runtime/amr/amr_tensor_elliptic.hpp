#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <pops/amr/hierarchy/refinement_ratio.hpp>  // kAmrRefRatio (ratio 2)
#include <pops/core/foundation/kokkos_env.hpp>       // device_fence
#include <pops/mesh/layout/box_array.hpp>            // BoxArray / Box2D
#include <pops/mesh/storage/mf_arith.hpp>            // pops::lincomb (device-clean copy / negate)
#include <pops/mesh/storage/multifab.hpp>            // MultiFab / DistributionMapping
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>  // CompositeFacPoisson (composite FAC elliptic)
#include <pops/numerics/elliptic/linear/solve_report.hpp>       // SolveReport
#include <pops/parallel/comm.hpp>                    // pops::n_ranks (MPI-multilevel refusal)
#include <pops/runtime/amr/amr_runtime.hpp>          // AmrRuntime (the engine this helper reads)

/// @file
/// @brief AmrTensorElliptic -- the composite tensor-coefficient elliptic driver a compiled
///        condensed-implicit time Program routes to on a REFINED AMR hierarchy (ADC-633 / ADC-637).
///
/// The compiled condensed-implicit Program lowers, per AMR level, to inline block-inverse assembly
/// kernels (no coupling/schur call, ADC-637). On a FLAT hierarchy those run the emitted matrix-free
/// BiCGStab on level 0 -- bit-identical to the uniform Program. On a REFINED hierarchy (>= one fine
/// patch) the single-level matrix-free solve cannot address the fine levels, so the tensor elliptic is
/// solved COMPOSITELY: this helper owns per-level tensor-coefficient buffers (eps_x / eps_y / a_xy /
/// a_yx), a per-level right-hand side and a per-level potential, plus a lazily built, box-cached
/// pops::CompositeFacPoisson (two-way, variable coefficient + cross terms). The emitted assembly ops
/// write THROUGH AmrProgramContext::assembly_target into these level-shaped buffers (the level-0-bound
/// emitted scratch is unusable on a fine level); solve_composite() copies them into the FAC's per-level
/// fields, solves, and publishes each level's potential for the emitted reconstruction to READ through
/// AmrProgramContext::assembly_source.
///
/// GENERIC LAYER (owner directive, ADC-637): this driver names ONLY mathematical objects -- tensor
/// coefficients, right-hand side, potential, composite FAC. No B_z / Lorentz / electrostatic / Schur
/// vocabulary: the physics is authored in the DSL and emitted inline; this helper just co-distributes
/// the level buffers and drives the composite solve.
///
/// SCOPE. Inherited verbatim from pops::CompositeFacPoisson (ADC-636 generalized envelope): N levels,
/// 1..N disjoint fine patches (nested, ratio 2), replicated mono-box coarse. MPI multilevel is refused
/// precisely (mono-rank only) -- the composite path is a mono-rank driver here. Beyond that the FAC
/// ctor refuses (non-nested / misaligned patches) with a precise message; no silent partial solve.
/// CompositeFacPoisson currently owns one diagonal coefficient, so this solver accepts only
/// eps_x == eps_y and returns a typed capability failure for a genuinely anisotropic diagonal rather
/// than silently dropping eps_y. Cross terms a_xy/a_yx remain supported.
/// Unsupported MPI multilevel execution returns a typed capability-failure report, so the authored
/// SolveOutcome action decides whether to reject the attempt or fail the run. Single block (the AMR
/// Program v1 block scope); theta<1 composes through the gathered per-level phi^n history guess.

namespace pops {
namespace runtime {
namespace program {

namespace detail {
/// FAC reports a composite infinity norm while solve_linear exposes a relative tolerance.  Match the
/// established Krylov zero-RHS convention exactly: every non-zero RHS, including ||rhs|| < 1, keeps
/// its own scale; only the homogeneous RHS substitutes one to avoid division by zero.
inline Real tensor_fac_relative_scale(Real rhs_norm) {
  return rhs_norm > Real(0) ? rhs_norm : Real(1);
}

/// Native FAC controls. CompositeTensorFAC owns the outer tolerance and iteration budget; those are
/// joined with these optional overrides only at the native direct-solve boundary.
struct TensorFacControls {
  std::optional<int> fine_sweeps;
  std::optional<Real> coarse_rel_tol;
  std::optional<int> coarse_cycles;
  std::optional<bool> verbose;
};

inline void validate_tensor_fac_controls(const TensorFacControls& options) {
  if (options.fine_sweeps && *options.fine_sweeps <= 0)
    throw std::invalid_argument("CompositeTensorFAC fine_sweeps must be positive");
  if (options.coarse_rel_tol &&
      (!std::isfinite(static_cast<double>(*options.coarse_rel_tol)) ||
       *options.coarse_rel_tol <= Real(0) || *options.coarse_rel_tol >= Real(1)))
    throw std::invalid_argument("CompositeTensorFAC coarse_rel_tol must be finite and in (0, 1)");
  if (options.coarse_cycles && *options.coarse_cycles <= 0)
    throw std::invalid_argument("CompositeTensorFAC coarse_cycles must be positive");
}

inline TensorFacControls tensor_fac_controls(int fine_sweeps, Real coarse_rel_tol,
                                             int coarse_cycles, int verbose) {
  if (fine_sweeps < 0 || coarse_rel_tol < Real(0) || coarse_cycles < 0)
    throw std::invalid_argument(
        "CompositeTensorFAC wire options use zero for native default or a positive override");
  if (verbose < -1 || verbose > 1)
    throw std::invalid_argument(
        "CompositeTensorFAC verbose wire option must be -1 (native default), 0, or 1");
  TensorFacControls options{
      fine_sweeps == 0 ? std::nullopt : std::optional<int>(fine_sweeps),
      coarse_rel_tol == Real(0) ? std::nullopt : std::optional<Real>(coarse_rel_tol),
      coarse_cycles == 0 ? std::nullopt : std::optional<int>(coarse_cycles),
      verbose == -1 ? std::nullopt : std::optional<bool>(verbose == 1)};
  validate_tensor_fac_controls(options);
  return options;
}

inline CompositeFacOptions tensor_fac_options(const TensorFacControls& controls, Real tol,
                                              int max_iter) {
  validate_tensor_fac_controls(controls);
  if (!std::isfinite(static_cast<double>(tol)) || tol <= Real(0))
    throw std::invalid_argument("CompositeTensorFAC Program solver tolerance must be finite and positive");
  if (max_iter <= 0)
    throw std::invalid_argument("CompositeTensorFAC Program solver max_iter must be positive");
  // CompositeFacOptions is the single native source of truth for omitted FAC knobs. The
  // direct solver controls are always overwritten, and only explicitly present controls
  // override the canonical native defaults.
  CompositeFacOptions options;
  options.max_iters = max_iter;
  options.tol = tol;
  if (controls.fine_sweeps)
    options.fine_sweeps = *controls.fine_sweeps;
  if (controls.coarse_rel_tol)
    options.coarse_rel_tol = *controls.coarse_rel_tol;
  if (controls.coarse_cycles)
    options.coarse_cycles = *controls.coarse_cycles;
  if (controls.verbose)
    options.verbose = *controls.verbose;
  return options;
}
}  // namespace detail

/// Per-level tensor-coefficient buffers + a cached composite FAC solve, for one AMR block's condensed
/// tensor elliptic on a refined hierarchy. Owned by AmrProgramContext (one per installed Program on the
/// refined path); rebuilt lazily when the fine tiling changes. Indexed by AMR level (0 = coarse).
class AmrTensorElliptic {
 public:
  /// @p eng: the AMR engine (levels / geom / bc); @p block: the exact AMR system block index;
  /// @p ncomp: the authenticated operator component count. The native tensor route is scalar.
  AmrTensorElliptic(AmrRuntime* eng, int block, int ncomp)
      : eng_(eng), block_(block), ncomp_(ncomp) {
    if (ncomp_ != 1)
      throw std::invalid_argument("AmrTensorElliptic requires exactly one component");
  }

  /// Install the hierarchy-solver controls. Zero marks omitted numeric knobs and -1 marks omitted
  /// verbosity; those values resolve from CompositeFacOptions, the sole native defaults authority.
  void configure_composite_tensor_fac(int fine_sweeps, Real coarse_rel_tol, int coarse_cycles,
                                      int verbose) {
    fac_controls_ = detail::tensor_fac_controls(
        fine_sweeps, coarse_rel_tol, coarse_cycles, verbose);
  }

  /// Join native controls with CompositeTensorFAC tolerance / iteration budget. Public on this
  /// internal driver so the bridge is inspectable and unit-testable without constructing AMR storage.
  CompositeFacOptions composite_fac_options(Real tol, int max_iter) const {
    if (!fac_controls_)
      throw std::logic_error(
          "AmrTensorElliptic requires configure_composite_tensor_fac before a composite solve");
    return detail::tensor_fac_options(*fac_controls_, tol, max_iter);
  }

  /// True iff there is >= one populated fine level (level 1 carries >= one patch for this block). The
  /// AmrProgramContext gates the flat (matrix-free BiCGStab) vs composite (FAC) branch on this.
  bool has_fine_patches() const {
    if (eng_->nlev() < 2)
      return false;
    return eng_->level_state(static_cast<std::size_t>(block_), 1).box_array().size() > 0;
  }

  /// The level-shaped WRITE target for an assembly field of @p role at level @p k. The emitted
  /// assembly kernel reaches it via AmrProgramContext::assembly_target so its per-cell write lands in
  /// the composite buffer instead of the level-0-bound emitted scratch. Roles map to the AssemblyFieldRole
  /// enum in coeff_elliptic_ops.hpp (eps_x / eps_y / a_xy / a_yx / rhs / flux).
  MultiFab& target(int role, int k) {
    ensure_level_buffers(k);
    LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
    switch (role) {
      case 0: return lb.eps_x;   // kEpsX
      case 1: return lb.eps_y;   // kEpsY
      case 2: return lb.a_xy;    // kAxy
      case 3: return lb.a_yx;    // kAyx
      case 4: return lb.rhs;     // kRhs
      case 5: return lb.flux;    // kFlux (transient explicit-flux scratch)
      default:
        throw std::runtime_error("AmrTensorElliptic::target: unknown AssemblyFieldRole wire id " +
                                 std::to_string(role));
    }
  }

  /// The published composite potential of level @p k (filled by solve_composite): the emitted
  /// reconstruction reads it as phi^{n+theta} on that level (via AmrProgramContext::assembly_source).
  MultiFab& phi(int k) {
    ensure_level_buffers(k);
    return levels_[static_cast<std::size_t>(k)].phi;
  }

  /// Stage the current level's explicit solve initial guess.  Gathering this separately from the
  /// published solution is load-bearing: a rejected/non-converged FAC attempt must not publish its
  /// partial iterate, and a retry must start from the authored guess rather than leaked solver state.
  /// @p guess == nullptr is the declared zero initial guess.
  void stage_initial_guess(int k, const MultiFab* guess) {
    ensure_level_buffers(k);
    MultiFab& staged = levels_[static_cast<std::size_t>(k)].initial_guess;
    if (guess)
      copy0(staged, *guess);
    else
      staged.set_val(Real(0));
  }

  /// Solve the composite tensor elliptic across the whole nested tower: build/reuse the FAC on the fine
  /// tilings, copy the per-level coefficient / RHS buffers into the FAC's level fields, enable variable
  /// coefficient + cross terms + two-way, solve, then publish each level's potential into phi(k). REUSES
  /// pops::CompositeFacPoisson wholesale. This is a direct solver with a structurally authenticated
  /// tensor operator; MPI multilevel and unequal diagonal tensor coefficients report capability failure
  /// precisely (mono-rank) rather than publishing a partial value.
  SolveReport solve_composite(Real tol, int max_iter) {
    const int L = eng_->nlev();
    if (L < 2)
      return SolveReport::capability_failure();
    if (pops::n_ranks() != 1)
      return SolveReport::capability_failure();
    for (int k = 0; k < L; ++k)
      ensure_level_buffers(k);

    // CompositeFacPoisson exposes one diagonal coefficient. The built-in tensor operator route
    // requires equal diagonal entries, but the generic Program protocol can author a full tensor.
    // Reject that unsupported solver/operator pair explicitly instead of solving a different
    // operator by ignoring eps_y.
    for (int k = 0; k < L; ++k) {
      const LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
      MultiFab diagonal_delta(lb.eps_x.box_array(), lb.eps_x.dmap(), 1, 0);
      pops::lincomb(diagonal_delta, Real(1), lb.eps_x, Real(-1), lb.eps_y);
      if (pops::norm_inf(diagonal_delta) != Real(0))
        return SolveReport::capability_failure();
    }

    // The fine tilings (levels 1..L-1) key the FAC build; rebuild only when a tiling changes.
    std::vector<BoxArray> level_boxes;
    for (int k = 1; k < L; ++k)
      level_boxes.push_back(eng_->level_state(static_cast<std::size_t>(block_), k).box_array());
    ensure_fac(level_boxes);

    fac_->use_variable_coefficient(true);
    fac_->use_cross_terms(true);
    fac_->set_two_way(true);
    for (int k = 0; k < L; ++k) {
      LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
      // The tensor coefficient A = [[eps_x, a_xy], [a_yx, eps_y]] per level. Equality of the two
      // diagonal entries was checked above because this FAC solver currently stores one diagonal.
      copy0(fac_->eps_level(k), lb.eps_x);
      copy0(fac_->a_xy_level(k), lb.a_xy);
      copy0(fac_->a_yx_level(k), lb.a_yx);
      // the emitted condensed_rhs builds -Lap phi^n - g div(F): the matrix-free operator sign is
      // -div(A grad); the FAC solves div(eps grad phi) = f, so f = -rhs (the sign convention #126).
      negate_into(fac_->rhs_level(k), lb.rhs);
      // Do not inherit a partial FAC iterate from a rejected attempt.  Every attempt starts from the
      // per-level guess gathered from the Program (zero, or the carried phi^n history).
      copy0(fac_->phi_level(k), lb.initial_guess);
    }

    // solve_linear's public tolerance is relative, whereas FAC consumes
    // an absolute composite infinity-norm tolerance.  Use the same max norm FAC reports, across the
    // whole tower, so the adapter does not silently reinterpret a relative tolerance as an absolute
    // one. FAC and the generic Krylov path use different native norms (composite infinity vs
    // global L2), but share the same relative convention. rhs_norm == 0 is explicit: scale == 1, so a
    // homogeneous problem receives a finite absolute tolerance and reports its absolute residual.
    Real rhs_norm = 0;
    for (int k = 0; k < L; ++k)
      rhs_norm = std::max(rhs_norm, pops::norm_inf(fac_->rhs_level(k)));
    const Real scale = detail::tensor_fac_relative_scale(rhs_norm);
    const Real absolute_tol = tol * scale;

    const CompositeFacOptions options = composite_fac_options(absolute_tol, max_iter);
    fac_->set_options(options);
    const Real residual = fac_->solve(options.max_iters, options.fine_sweeps, options.tol);

    SolveReport report;
    report.rel_residual = residual / scale;
    if (std::isfinite(static_cast<double>(report.rel_residual)) && report.rel_residual <= tol)
      report.mark_solved();
    else
      report.mark_failed(std::isfinite(static_cast<double>(report.rel_residual))
                             ? SolveStatus::kIterationLimit
                             : SolveStatus::kInvalidEvaluation);
    if (!report.solved_value_available())
      return report;
    // Publication is atomic with respect to solve success: reconstruction cannot observe a partial
    // iterate, and the final SolveOutcome/StepTransaction contract can roll back later phases without
    // a failed solve having exposed a value.
    for (int k = 0; k < L; ++k)
      copy0(levels_[static_cast<std::size_t>(k)].phi, fac_->phi_level(k));
    return report;
  }

 private:
  struct LevelBuffers {
    MultiFab eps_x, eps_y, a_xy, a_yx;  ///< tensor coefficient A = [[eps_x, a_xy], [a_yx, eps_y]]
    MultiFab rhs;                        ///< condensed right-hand side (-Lap phi^n - g div F)
    MultiFab flux;                       ///< transient explicit-flux scratch (2-comp, if the body uses it)
    MultiFab initial_guess;              ///< gathered per-level initial guess for the next solve attempt
    MultiFab phi;                        ///< published composite potential of this level
    bool built = false;
  };

  /// Allocate level @p k's buffers on that level's grid (co-distributed with its state), once. eps /
  /// coefficient / phi carry 1 ghost (the operator face mean + the centered gradient); rhs 0 ghost.
  void ensure_level_buffers(int k) {
    if (k >= static_cast<int>(levels_.size()))
      levels_.resize(static_cast<std::size_t>(k) + 1);
    const MultiFab& U = eng_->level_state(static_cast<std::size_t>(block_), k);
    const BoxArray ba = U.box_array();
    const DistributionMapping dm = U.dmap();
    LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
    // Regrid may replace a level with a different patch tiling while retaining the level index.  The
    // old built flag alone would then route assembly into stale storage.  Multi-level execution is
    // mono-rank here, so the BoxArray is the complete distribution identity that can change.
    if (lb.built && lb.phi.box_array().boxes() == ba.boxes())
      return;
    lb = LevelBuffers{};
    lb.eps_x = MultiFab(ba, dm, 1, 1);
    lb.eps_y = MultiFab(ba, dm, 1, 1);
    lb.a_xy = MultiFab(ba, dm, 1, 1);
    lb.a_yx = MultiFab(ba, dm, 1, 1);
    lb.rhs = MultiFab(ba, dm, 1, 0);
    lb.flux = MultiFab(ba, dm, 2, 1);
    lb.initial_guess = MultiFab(ba, dm, 1, 1);
    lb.phi = MultiFab(ba, dm, 1, 1);
    lb.eps_x.set_val(Real(0));
    lb.eps_y.set_val(Real(0));
    lb.a_xy.set_val(Real(0));
    lb.a_yx.set_val(Real(0));
    lb.rhs.set_val(Real(0));
    lb.flux.set_val(Real(0));
    lb.initial_guess.set_val(Real(0));
    lb.phi.set_val(Real(0));
    lb.built = true;
  }

  /// Build (or rebuild on a fine-tiling change) the composite FAC over ALL fine levels -- the verbatim
  /// ensure_fac idiom of the native source stepper (compare per-level boxes + order, rebuild only on a
  /// change). A single fine level uses the 2-level ctor (bit-identical), deeper towers the N-level ctor;
  /// the FAC ctor refuses ratio != 2 / non-nested / misaligned patches, precisely.
  void ensure_fac(const std::vector<BoxArray>& level_boxes) {
    std::vector<std::vector<Box2D>> key;
    key.reserve(level_boxes.size());
    for (const BoxArray& ba : level_boxes)
      key.push_back(ba.boxes());
    if (fac_ && fac_level_boxes_ == key)
      return;
    const Geometry geom_c = eng_->level_geom(0);
    const BoxArray coarse_ba =
        eng_->level_state(static_cast<std::size_t>(block_), 0).box_array();
    if (level_boxes.size() == 1)
      fac_ = std::make_unique<CompositeFacPoisson>(geom_c, coarse_ba, eng_->poisson_bc(),
                                                   level_boxes[0], kAmrRefRatio);
    else
      fac_ = std::make_unique<CompositeFacPoisson>(geom_c, coarse_ba, eng_->poisson_bc(), level_boxes,
                                                   kAmrRefRatio);
    fac_level_boxes_ = std::move(key);
  }

  static void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    pops::lincomb(dst, Real(1), src, Real(0), src);  // dst <- src (comp 0), device-clean
  }
  static void negate_into(MultiFab& dst, const MultiFab& src) {
    device_fence();
    pops::lincomb(dst, Real(-1), src, Real(0), src);  // dst <- -src
  }

  AmrRuntime* eng_;
  int block_;
  int ncomp_;
  std::vector<LevelBuffers> levels_;
  std::unique_ptr<CompositeFacPoisson> fac_;
  std::vector<std::vector<Box2D>> fac_level_boxes_;
  std::optional<detail::TensorFacControls> fac_controls_;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
