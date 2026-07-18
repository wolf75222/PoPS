#pragma once

/// @file
/// @brief GeometricMG: in-house geometric multigrid (V-cycle) for the elliptic operator, Gauss-Seidel
///        smoother and bottom solve. Models the EllipticSolver and LinearSolver concepts.
///
/// Layer: `include/pops/numerics/elliptic/mg`.
/// Role: solve L(phi) = f by a classic V-cycle (pre-smoothing, residual restriction via average_down
/// onto a twice-coarser grid, recursive solve of the correction with homogeneous BCs,
/// prolongation via interpolate, post-smoothing; at the coarsest level, long smoothing = bottom solve).
/// The hierarchy is obtained by coarsening the domain by 2 down to a minimal size; restriction and
/// prolongation reuse the AMR transfer operators. This is the ONLY type that carries the operator
/// role (EllipticOperator: accessors op_eps()/op_kappa()/... + bc() + geom()), reused by the
/// Krylov solver for a matvec consistent with the MG residual.
/// Contract: solve(rel_tol, max_cycles, abs_tol=0) returns the number of cycles; its mixed stopping
/// criterion is ||r||inf <= max(rel_tol * ||R(0)||inf, abs_tol). A unit denominator is used only to
/// report a finite relative residual when ||R(0)||inf=0; it never relaxes the stopping criterion.
/// solve() with no argument takes the default tolerance (1e-8, 50 cycles). phi is kept between calls
/// (warm start), so an unchanged already-converged system exits with zero cycles. solve_robust hardens
/// the smoothing ONLY in case of true divergence at the embedded boundary (otherwise bit-identical).
///
/// Invariants:
/// - coarsening stops if a box does not coarsen CLEANLY (refine(coarsen(b)) != b): avoids
///   a degenerate coarse BoxArray (duplicate 1x1 boxes) where average_down would read out of bounds (MPI bug);
/// - current_residual() does a MANDATORY all_reduce_max (distributed multi-box coarse): otherwise the
///   stopping criterion fires at different iterations per rank -> MPI desynchronization;
/// - FieldDistribution::Replicated: level replicated on all ranks (per-fab V-cycle without
///   communication); in serial bit-for-bit identical to round-robin;
/// - cut_cell: order-2 Shortley-Weller weights at the embedded boundary (vs staircase); cut_cell=false
///   bit-identical to the historical stencil;
/// - device kernels are NAMED FUNCTORS (recipe #93/#64): extended lambda forbidden cross-TU under nvcc.

#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/numerics/elliptic/eb/cut_fraction.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/interface/spatial_provider.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/field_replica_consensus.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <bit>
#include <chrono>  // last_bottom_seconds(): self-time the coarsest (bottom) GS solve (Spec 5, ADC-479)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>  // getenv
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

namespace detail {
// Copy component 0 of a fine field (discretized eps/eps_y/kappa) onto the MG fine level.
// NAMED FUNCTOR (not an POPS_HD lambda): same device-clean recipe as the rest (#93). Identical
// body -> bit-identical. Inert on the constant-eps path, exercised as soon as a field is wired.
struct CopyComp0Kernel {
  Array4 d;
  ConstArray4 s;
  POPS_HD void operator()(int i, int j) const { d(i, j) = s(i, j, 0); }
};
}  // namespace detail

inline BCRec homogeneous(const BCRec& b) {
  BCRec h = b;
  h.xlo_val = h.xhi_val = h.ylo_val = h.yhi_val = 0;
  return h;
}

class GeometricMG {
 public:
  // active(x, y): optional "active cell" predicate (interior of the conductor).
  // Empty => everything active (no embedded wall).
  // distribution: a replicated field owns one complete level on every rank (dmap = my_rank()
  // everywhere); a distributed field uses the supplied distribution or the default round-robin.
  // Replicated ranks solve the same coarse Poisson redundantly, without communication inside the
  // per-fab V-cycle. Any provider may select this explicit distribution contract. In serial the two
  // distribution modes have identical storage but remain distinct semantic contracts.
  //
  // cut_cell + levelset: ORDER-2 embedded boundary (Shortley-Weller) instead of
  // the staircase. levelset(x, y) is a level-set function (< 0 inside, sign of
  // the boundary); for the conducting circle, levelset = hypot(x - cx, y - cy) - Rwall.
  // Each active cell receives 5 coefficients computed from the distances to the
  // boundary (cut fraction theta per direction). active is then deduced from the sign of
  // levelset if it is not provided. cut_cell=false => historical staircase stencil (bit-identical).
  //
  // V-cycle parameters (proven defaults):
  //   min_coarse (default 2): minimal size of a grid dimension below which we STOP
  //                           coarsening. Coarsening grows the domain by 2 as long as nx/2 and ny/2
  //                           stay >= min_coarse (and the boxes coarsen cleanly); the
  //                           coarsest grid (the bottom) thus keeps >= min_coarse cells per axis.
  //   nu1 (default 2): number of PRE-smoothing Gauss-Seidel sweeps (before descending to the
  //                           coarse grid), at each non-bottom level.
  //   nu2 (default 2): number of POST-smoothing Gauss-Seidel sweeps (after ascending and adding
  //                           the prolonged correction), at each non-bottom level.
  //   nbottom (default 50): number of Gauss-Seidel sweeps at the coarsest level (bottom solve);
  //                           this long smoothing stands in for an exact solve on the small bottom grid.
  // (solve_robust LOCALLY doubles nu1/nu2 if the embedded boundary makes the cycle diverge, then restores them.)
  GeometricMG(const Geometry& geom, const BoxArray& ba, const BCRec& bc,
              ActiveRegionProvider2D active = {},
              FieldDistribution distribution = FieldDistribution::Distributed,
              int min_coarse = kMGDefaultMinCoarse, int nu1 = kMGDefaultPreSmooth,
              int nu2 = kMGDefaultPostSmooth, int nbottom = kMGDefaultBottomSweeps,
              bool cut_cell = false, LevelSetProvider2D levelset = {},
              Real cut_theta_min = kEbCutFractionFloor,
              int coarse_threshold = kMGDefaultCoarseThreshold,
              const DistributionMapping* finest_distribution = nullptr)
      : bc_(bc),
        active_(std::move(active)),
        min_coarse_(min_coarse),
        nu1_(nu1),
        nu2_(nu2),
        nbottom_(nbottom),
        coarse_threshold_(
            coarse_threshold),          // ADC-644: total-cell coarsening ceiling (0 = disabled)
        cut_theta_min_(cut_theta_min),  // ADC-615: cut-fraction clamp shared with the EB transport
        distribution_(distribution),
        cut_cell_(cut_cell),
        levelset_(std::move(levelset)) {
    if (!field_distribution_is_valid(distribution_))
      throw std::invalid_argument("GeometricMG received invalid field distribution");
    if (is_replicated())
      replica_validation_data_.assign(detail::field_replica_consensus_storage_size(), char{0});
    if (is_replicated() && finest_distribution) {
      const int rank = my_rank();
      for (const int owner : finest_distribution->ranks())
        if (owner != rank)
          throw std::invalid_argument(
              "replicated GeometricMG must own every finest-level box on every rank");
    }
    bc_.dx = geom.dx();
    bc_.dy = geom.dy();
    if (cut_cell_ && levelset_ && !active_)
      active_ = active_region_from_level_set(levelset_);
    add_level(geom, ba, finest_distribution);
    while (true) {
      const Geometry g = lev_.back().geom;
      // ADC-644: an explicit total-cell coarsening ceiling. STOP coarsening once the current level's
      // total unknown count (nx*ny) is at or below coarse_threshold_ (a direct-small-grid stand-in:
      // the nbottom Gauss-Seidel bottom solve then runs on this level). Distinct from min_coarse (a
      // PER-AXIS lower bound); when both are active coarsening halts at whichever is reached first.
      // Sentinel 0 = disabled (only min_coarse governs) -> the guard is inert, hierarchy unchanged.
      if (coarse_threshold_ > 0 && g.domain.nx() * g.domain.ny() <= coarse_threshold_)
        break;
      if (g.domain.nx() % 2 || g.domain.ny() % 2)
        break;
      if (g.domain.nx() / 2 < min_coarse || g.domain.ny() / 2 < min_coarse)
        break;
      // Stop if a box of the current level does not coarsen CLEANLY: on a MULTI-BOX domain
      // (max_grid_size < n), the boxes shrink by 2 at each level and
      // end up at 1 cell; coarsen(ba, 2) would then make SEVERAL distinct fine
      // boxes fall onto the SAME coarse cell -> DEGENERATE coarse BoxArray (duplicate boxes
      // covering the same cell). average_down reads an r x r block per coarse cell
      // (F(r*I+a, r*J+b)): for a fine fab of 1 cell (0 ghost) three of the four reads
      // fall OUT of the buffer bounds (negative indices), i.e. into uninitialized memory.
      // In serial the heap is stable (deterministic read), but on the MPI path the heap is
      // shuffled and the read becomes ERRATIC (pointwise deviation up to blow-up). So we keep
      // the current level as the coarsest grid. refine(coarsen(b)) == b characterizes
      // exactly the boxes that are aligned AND of even size (exact coarsening, no duplicate or
      // overflow); mono-box and non-degenerate multi-box never cross this break ->
      // hierarchy (and result) STRICTLY unchanged on those cases.
      const BoxArray& cur = lev_.back().ba;
      bool coarsenable = true;
      for (int i = 0; i < cur.size(); ++i)
        if (!(cur[i].coarsen(2).refine(2) == cur[i])) {
          coarsenable = false;
          break;
        }
      if (!coarsenable)
        break;
      Geometry gc{g.domain.coarsen(2), g.xlo, g.xhi, g.ylo, g.yhi};
      const DistributionMapping coarse_distribution = lev_.back().dm;
      add_level(gc, coarsen(lev_.back().ba, 2), &coarse_distribution);
    }
    // V-cycle buffers (corr/cfine) allocated ONCE for each NON-bottom level. cfine adopts the
    // exact layout that average_down/interpolate would have allocated internally: coarsen(L.ba, 2) on the
    // FINE dmap (L.dm), 0 ghost. It is REUSED for restriction (average_down(L.res, C.rhs)) AND
    // prolongation (interpolate(C.phi, L.corr)) of the same level (uses disjoint in time -> a single
    // buffer suffices). The bottom does not need them (early return from vcycle_rec) and its coarsen would
    // be degenerate (the very reason coarsening stops) -> not allocated.
    for (int l = 0; l + 1 < static_cast<int>(lev_.size()); ++l) {
      lev_[l].corr = MultiFab(lev_[l].ba, lev_[l].dm, 1, 0);
      lev_[l].cfine = MultiFab(coarsen(lev_[l].ba, 2), lev_[l].dm, 1, 0);
    }
    if (active_) {
      // each level evaluates its own mask from the physical circle
      for (auto& L : lev_) {
        L.mask = MultiFab(L.ba, L.dm, 1, 0);
        for (int li = 0; li < L.mask.local_size(); ++li) {
          Array4 m = L.mask.fab(li).array();
          const Geometry& g = L.geom;
          const Box2D b = L.mask.box(li);
          // host initialization (std::function predicate not device-callable);
          // writes unified memory before any kernel.
          for (int j = b.lo[1]; j <= b.hi[1]; ++j)
            for (int i = b.lo[0]; i <= b.hi[0]; ++i)
              m(i, j) = active_(g.x_cell(i), g.y_cell(j)) ? Real(1) : Real(0);
        }
      }
    }
    if (cut_cell_ && levelset_) {
      // Shortley-Weller coefficients per active cell, computed per level from
      // the level-set cut fractions (linear crossing). w_diag grows near the
      // boundary (cut cell) but the system STAYS diagonally dominant (GS converges):
      // we only clamp theta at 1e-3 to avoid division by 0, without degrading order 2
      // (a wider clamp, e.g. 0.05, shifts the worst cut cells and breaks the order).
      for (auto& L : lev_) {
        L.coef = MultiFab(L.ba, L.dm, 5, 0);
        const Geometry& g = L.geom;
        const Real dx = g.dx(), dy = g.dy();
        for (int li = 0; li < L.coef.local_size(); ++li) {
          Array4 c = L.coef.fab(li).array();
          const ConstArray4 m = L.mask.fab(li).const_array();
          const Box2D b = L.coef.box(li);
          // SHARED face-crossing primitive (cut_fraction.hpp): SAME aperture geometry
          // as the future EB transport. detail::cut_fraction reproduces verbatim the old 'cut'
          // lambda (cut_distance, same branches and same 1e-3 clamp) and detail::shortley_weller the
          // formula for the 5 weights -> coef BIT-IDENTICAL to the inline assembly before the refactor.
          const auto& ls = levelset_;
          for (int j = b.lo[1]; j <= b.hi[1]; ++j)
            for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
              if (m(i, j) == Real(0)) {  // conductor: coef unused (cell skipped)
                for (int k = 0; k < 5; ++k)
                  c(i, j, k) = 0;
                continue;
              }
              const detail::CutFraction cf =
                  detail::cut_fraction(ls, g.x_cell(i), g.y_cell(j), dx, dy, cut_theta_min_);
              const detail::ShortleyWellerWeights w = detail::shortley_weller(cf);
              c(i, j, 0) = w.w_xm;    // w_xm on p(i-1)
              c(i, j, 1) = w.w_xp;    // w_xp on p(i+1)
              c(i, j, 2) = w.w_ym;    // w_ym on p(i,j-1)
              c(i, j, 3) = w.w_yp;    // w_yp on p(i,j+1)
              c(i, j, 4) = w.w_diag;  // w_diag
            }
        }
      }
    }
    prepared_operator_contract_ = make_materialized_elliptic_operator_contract(
        operator_identity(), this->geom(), bc_, active_, distribution_, rhs(), phi(),
        construction_options_contract(min_coarse_, nu1_, nu2_, nbottom_, cut_cell_, levelset_,
                                      cut_theta_min_, coarse_threshold_));
  }

  /// Build the finest level on an already materialized distribution. This is the prepared
  /// preconditioner route: a custom but valid box distribution must remain co-distributed with the
  /// Krylov vectors instead of being silently replaced by a new round-robin mapping.
  GeometricMG(const Geometry& geom, const BoxArray& ba, const DistributionMapping& mapping,
              const BCRec& bc, ActiveRegionProvider2D active = {},
              int min_coarse = kMGDefaultMinCoarse, int nu1 = kMGDefaultPreSmooth,
              int nu2 = kMGDefaultPostSmooth, int nbottom = kMGDefaultBottomSweeps,
              int coarse_threshold = kMGDefaultCoarseThreshold,
              FieldDistribution field_distribution = FieldDistribution::Distributed)
      : GeometricMG(geom, ba, bc, std::move(active), field_distribution, min_coarse, nu1, nu2,
                    nbottom, /*cut_cell=*/false, {}, kEbCutFractionFloor, coarse_threshold,
                    &mapping) {}

  /// Backend-neutral factory entry point. The representation contract and the exact ownership map
  /// are independent: replicated/distributed controls collective semantics, while @p mapping is the
  /// already materialized layout authority and is never reconstructed by the solver.
  GeometricMG(const Geometry& geom, const BoxArray& ba, const DistributionMapping& mapping,
              const BCRec& bc, ActiveRegionProvider2D active, FieldDistribution field_distribution)
      : GeometricMG(geom, ba, mapping, bc, std::move(active), kMGDefaultMinCoarse,
                    kMGDefaultPreSmooth, kMGDefaultPostSmooth, kMGDefaultBottomSweeps,
                    kMGDefaultCoarseThreshold, field_distribution) {}

  MultiFab& phi() { return lev_[0].phi; }
  MultiFab& rhs() { return lev_[0].rhs; }
  const Geometry& geom() const { return lev_[0].geom; }
  int num_levels() const { return static_cast<int>(lev_.size()); }
  /// Exact number of persistent hierarchy MultiFab slots owned by this prepared engine. This is a
  /// storage-shape query, not an allocation-event counter; it remains stable until rematerializing
  /// the hierarchy.
  [[nodiscard]] std::size_t persistent_field_count() const noexcept {
    constexpr std::size_t fields_per_level = 15u;
    return lev_.size() * fields_per_level;
  }
  FieldDistribution field_distribution() const noexcept { return distribution_; }
  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.elliptic.geometric-mg", 1};
  }
  static EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request, int min_coarse = kMGDefaultMinCoarse,
      int nu1 = kMGDefaultPreSmooth, int nu2 = kMGDefaultPostSmooth,
      int nbottom = kMGDefaultBottomSweeps, bool cut_cell = false, LevelSetProvider2D levelset = {},
      Real cut_theta_min = kEbCutFractionFloor, int coarse_threshold = kMGDefaultCoarseThreshold) {
    EllipticBuildRequest effective = request;
    if (cut_cell && levelset && !effective.active)
      effective.active = active_region_from_level_set(levelset);
    return make_expected_elliptic_operator_contract(
        operator_identity(), effective,
        construction_options_contract(min_coarse, nu1, nu2, nbottom, cut_cell, levelset,
                                      cut_theta_min, coarse_threshold));
  }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return prepared_operator_contract_;
  }

  const RuntimeDiagnosticsReport& diagnostics_report() const { return diagnostics_; }
  void reset_diagnostics() { diagnostics_.clear(); }

  // --- PER-SOLVE PROFILING STATS (Spec 5 sec.13.11.1, ADC-479 criteria 42/43) -------------------
  // Cached by the most recent solve(rel_tol, max_cycles, abs_tol) call (the no-argument concept-level
  // solve() funnels through it). The System reads these back at the field_solve seam to populate the
  // elliptic-solver native counters WITHOUT threading a profiler into the deep numerics: chrono-only
  // here, no profiler / Kokkos dependency. Additive accessors -- no existing path reads them, the
  // default behavior is unchanged.
  //   last_cycles():         V-cycles performed by the last solve (the value solve() returns).
  //   last_residual():       final residual (infinity norm) reached by the last solve.
  //   last_bottom_seconds(): wall-clock self-time of the coarsest-grid (bottom) Gauss-Seidel solves
  //                          summed over the V-cycles of the last solve (steady_clock; host serial /
  //                          per-rank; on a device backend a fence would be needed for an exact bottom
  //                          time, deferred -- the counter stays an honest host-side measurement).
  int last_cycles() const { return last_cycles_; }
  Real last_residual() const { return last_residual_; }
  double last_bottom_seconds() const { return last_bottom_seconds_; }

  // Activates VARIABLE permittivity eps(x): the operator goes from lap(phi)=f to
  // div(eps grad phi)=f. eps is a CELL-CENTERED field, evaluated by the
  // analytic function provided on EACH level of the hierarchy (like the mask
  // and the cut-cell coefficients), then its ghosts are filled. Evaluating eps level
  // by level (rather than restricting from the fine level) gives the EXACT permittivity
  // at each coarse resolution, which preserves order 2. Call once
  // after construction, before solve. DO NOT call => uniform eps (historical path).
  void set_epsilon(ScalarFieldProvider2D eps_fn) {
    if (!eps_fn)
      throw std::invalid_argument("GeometricMG epsilon provider must not be empty");
    // 1 ghost (box-boundary neighbors read), ghosts filled (do_fill).
    sample_per_level(&MGLevel::eps, eps_fn, 1, true, eps_bc());
    eps_provider_contract_ = eps_fn.collective_contract();
    has_eps_ = true;
  }

  // Overload taking an ALREADY-discretized eps field (1-component MultiFab, defined
  // on the finest level grid). It is copied onto the fine level then
  // RESTRICTED (average_down, 2x2 average) to the coarse levels, and its ghosts
  // are filled at each level. Use it when eps comes from a per-cell field
  // (not from an analytic formula): this is the entry point for System wiring.
  void set_epsilon(const MultiFab& eps_fine) {
    // copy on the fine + restriction to the coarse; 1 ghost, ghosts filled at each level.
    restrict_and_fill(&MGLevel::eps, eps_fine, 1, true, eps_bc());
    eps_provider_contract_ = "pops.scalar-field.prepared-multifab@1";
    has_eps_ = true;
  }

  // Activates ANISOTROPIC permittivity: the operator goes from div(eps grad phi) (scalar
  // eps) to div(diag(eps_x, eps_y) grad phi). Faces NORMAL TO X use eps_x,
  // faces NORMAL TO Y use eps_y. eps_x is wired like the isotropic eps (sets
  // the internal eps field, x faces) and eps_y a SECOND field (y faces). Same conventions
  // as set_epsilon: CELL-CENTERED field, evaluated PER LEVEL (exact coarse permittivity,
  // order 2 preserved) then ghosts filled. Use case: anisotropic medium/mesh.
  // Giving eps_x_fn == eps_y_fn gives back the isotropic operator eps=eps_x. Composable with
  // set_reaction (kappa). Call once after construction, before solve.
  void set_epsilon_anisotropic(ScalarFieldProvider2D eps_x_fn, ScalarFieldProvider2D eps_y_fn) {
    if (!eps_x_fn || !eps_y_fn)
      throw std::invalid_argument("GeometricMG anisotropic epsilon providers must not be empty");
    set_epsilon(std::move(eps_x_fn));  // x faces: reuse the isotropic eps wiring
    // y faces: second eps_y field, same convention (1 ghost, ghosts filled).
    sample_per_level(&MGLevel::eps_y, eps_y_fn, 1, true, eps_bc());
    eps_y_provider_contract_ = eps_y_fn.collective_contract();
    has_eps_y_ = true;
  }

  // Overload taking two ALREADY-discretized fields (finest level grid), copied
  // onto the fine level then RESTRICTED (average_down) to the coarse and ghosts filled,
  // exactly like set_epsilon(const MultiFab&). Entry point for per-field wiring
  // (e.g. from System). eps_x carries the x faces, eps_y the y faces.
  void set_epsilon_anisotropic(const MultiFab& eps_x_fine, const MultiFab& eps_y_fine) {
    set_epsilon(eps_x_fine);  // x faces: reuse the isotropic eps wiring (+ restriction)
    // y faces: second eps_y field, copy + restriction (1 ghost, ghosts filled at each level).
    restrict_and_fill(&MGLevel::eps_y, eps_y_fine, 1, true, eps_bc());
    eps_y_provider_contract_ = "pops.scalar-field.prepared-multifab@1";
    has_eps_y_ = true;
  }

  // Activates the REACTION term kappa(x): the operator goes from div(eps grad phi) = f to
  // div(eps grad phi) - kappa phi = f (SCREENED Poisson / Helmholtz; kappa = 1/lambda_D^2 for
  // Debye screening). kappa >= 0 makes the operator more diagonally dominant (the multigrid
  // converges at least as well). It is a PHYSICAL coefficient (unit 1/length^2), DIAGONAL:
  // read at (i,j) only (no neighbor), so 0 ghost; restricted by average on the coarse
  // levels (same physical value sampled). DO NOT call => kappa = 0 (Poisson, historical
  // path strictly unchanged). Composable with set_epsilon (eps(x) and kappa(x) together).
  // ADC-251: 0 ghost / no fill_ghosts is DELIBERATE (a reaction term is zeroth-order: kappa is never
  // read at a neighbor, so its ghosts cannot be needed); filling them would be dead work. The
  // invariant is locked by the VARYING-kappa MMS in tests/test_screened_poisson.cpp (cases D/E),
  // which a future stencil reading kappa on its unfilled ghosts would break.
  void set_reaction(ScalarFieldProvider2D kappa_fn) {
    if (!kappa_fn)
      throw std::invalid_argument("GeometricMG reaction provider must not be empty");
    // kappa: DIAGONAL, read at (i,j) only -> 0 ghost and do_fill=false (NO fill_ghosts, historical).
    // ebc is then unused (BCRec{} never read).
    sample_per_level(&MGLevel::kappa, kappa_fn, 0, false, BCRec{});
    kappa_provider_contract_ = kappa_fn.collective_contract();
    has_kappa_ = true;
  }

  // Overload: ALREADY-discretized kappa field (1-component MultiFab, fine grid), copied onto the
  // fine level then RESTRICTED (average_down) to the coarse. Entry point for System wiring
  // (a per-cell kappa field).
  void set_reaction(const MultiFab& kappa_fine) {
    // kappa: DIAGONAL -> 0 ghost and do_fill=false (NO fill_ghosts, neither fine nor coarse, historical).
    restrict_and_fill(&MGLevel::kappa, kappa_fine, 0, false, BCRec{});
    kappa_provider_contract_ = "pops.scalar-field.prepared-multifab@1";
    has_kappa_ = true;
  }

  // Activates the OFF-DIAGONAL COEFFICIENTS of the FULL tensor A = [[eps_x, Axy], [Ayx, eps_y]]:
  // the operator goes from div(diag(eps_x, eps_y) grad phi) to div(A grad phi), adding the CROSS
  // fluxes d_x(Axy d_y phi) + d_y(Ayx d_x phi) (cf. poisson_operator.hpp). A may be NON
  // symmetric (Axy != Ayx). Same conventions as set_epsilon: CELL-CENTERED fields, evaluated PER
  // LEVEL (exact coarse coefficient) then ghosts filled (the face average reads the neighbor at
  // i+-1 / j+-1). Composable with set_epsilon[_anisotropic] and set_reaction. Call once after
  // construction, before solve. DO NOT call => DIAGONAL block (current path bit-identical).
  // WARNING: for strongly non-symmetric A the 5-point GS V-cycle (smoother of the DIAGONAL
  // block, EXPLICIT cross terms) may NOT converge; a Krylov would then be required.
  void set_cross_terms(ScalarFieldProvider2D a_xy_fn, ScalarFieldProvider2D a_yx_fn) {
    if (!a_xy_fn || !a_yx_fn)
      throw std::invalid_argument("GeometricMG cross-term providers must not be empty");
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) {
      L.a_xy = MultiFab(L.ba, L.dm, 1, 1);  // 1 ghost: the face average reads the boundary neighbor
      L.a_yx = MultiFab(L.ba, L.dm, 1, 1);
      const Geometry& g = L.geom;
      for (int li = 0; li < L.a_xy.local_size(); ++li) {
        Array4 fxy = L.a_xy.fab(li).array();
        Array4 fyx = L.a_yx.fab(li).array();
        const Box2D b = L.a_xy.box(li);
        // host initialization (std::function not device-callable); unified memory before kernel
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
            const Real x = g.x_cell(i), y = g.y_cell(j);
            fxy(i, j) = a_xy_fn(x, y);
            fyx(i, j) = a_yx_fn(x, y);
          }
      }
      fill_ghosts(L.a_xy, g.domain, ebc);
      fill_ghosts(L.a_yx, g.domain, ebc);
    }
    a_xy_provider_contract_ = a_xy_fn.collective_contract();
    a_yx_provider_contract_ = a_yx_fn.collective_contract();
    has_cross_ = true;
  }

  // Overload taking two ALREADY-discretized fields (finest level grid), copied onto the
  // fine level then RESTRICTED (average_down) to the coarse and ghosts filled, exactly like
  // set_epsilon_anisotropic(const MultiFab&, const MultiFab&). Entry point for PER-CELL cross
  // terms (e.g. A = I + c rho B^{-1} from Schur condensation, where rho varies in space, so
  // a_xy/a_yx are not analytic formulas but fields). The cross coefficients only
  // serve the residual / the FULL matvec (the GS smoother stays 5-point, diagonal block); their
  // restriction to the coarse therefore only serves a possible MG residual on the full operator (the
  // Krylov preconditioner is wired WITHOUT cross terms -> symmetric part). DO NOT call
  // => DIAGONAL block (current path bit-identical).
  void set_cross_terms(const MultiFab& a_xy_fine, const MultiFab& a_yx_fine) {
    const BCRec ebc = eps_bc();
    for (auto& L : lev_) {
      L.a_xy = MultiFab(L.ba, L.dm, 1, 1);
      L.a_yx = MultiFab(L.ba, L.dm, 1, 1);
    }
    for (int li = 0; li < lev_[0].a_xy.local_size(); ++li) {
      Array4 fxy = lev_[0].a_xy.fab(li).array();
      Array4 fyx = lev_[0].a_yx.fab(li).array();
      const ConstArray4 sxy = a_xy_fine.fab(li).const_array();
      const ConstArray4 syx = a_yx_fine.fab(li).const_array();
      const Box2D b = lev_[0].a_xy.box(li);
      for_each_cell(b, detail::CopyComp0Kernel{fxy, sxy});
      for_each_cell(b, detail::CopyComp0Kernel{fyx, syx});
    }
    fill_ghosts(lev_[0].a_xy, lev_[0].geom.domain, ebc);
    fill_ghosts(lev_[0].a_yx, lev_[0].geom.domain, ebc);
    for (int l = 1; l < num_levels(); ++l) {
      average_down(lev_[l - 1].a_xy, lev_[l].a_xy, 2);
      average_down(lev_[l - 1].a_yx, lev_[l].a_yx, 2);
      fill_ghosts(lev_[l].a_xy, lev_[l].geom.domain, ebc);
      fill_ghosts(lev_[l].a_yx, lev_[l].geom.domain, ebc);
    }
    a_xy_provider_contract_ = "pops.scalar-field.prepared-multifab@1";
    a_yx_provider_contract_ = "pops.scalar-field.prepared-multifab@1";
    has_cross_ = true;
  }

  void vcycle() { vcycle_rec(0, bc_); }
  void vcycle(const ExecutionLane& lane) { vcycle_rec(0, bc_, &lane); }
  // ROMEO-ONLY (deferred): a Kokkos::Profiling::pushRegion("mg:vcycle")/popRegion() pair (with a
  // Kokkos::fence() before popRegion) around this V-cycle would let Nsight attribute the GPU time on
  // ROMEO. It is intentionally NOT added here: it needs a Kokkos include in a header the profiling
  // design keeps Kokkos-free (chrono only, last_bottom_seconds()), and the host build (Serial-only
  // conda Kokkos) gains nothing. Add it at the System/ProgramContext seam if Nsight attribution is
  // wanted, not in this numerics header.

  // V-cycles until the residual is under the mixed floor (or max_cycles). Returns the number
  // of cycles performed. phi is kept between calls (warm start).
  //
  // MIXED relative/absolute stopping criterion:
  //   ||residual||inf <= max(rel_tol * ||R(0)||inf, abs_tol)
  // abs_tol is an ABSOLUTE floor on the residual norm (SAME units as current_residual(),
  // so scaled to the problem by the caller who knows it: no magic constant is baked in here).
  // R(0) is the exact affine discrete forcing, including inhomogeneous physical/generated boundary
  // lifting. Its norm is independent of the incoming warm start: repeated solves do not demand
  // another rel_tol factor from an already-converged iterate.
  int solve(Real rel_tol, int max_cycles, Real abs_tol = Real(0)) {
    require_collective_solve_contract_(rel_tol, max_cycles, abs_tol, "GeometricMG::solve");
    const bool fallible_linear_boundary =
        has_boundary_kernel_ && !boundary_kernel_.observes_iteration;
    if (fallible_linear_boundary)
      boundary_failure_.reset();
    auto finish = [&](int cycles, Real relative_residual, SolveStatus status, SolveAction action) {
      if (fallible_linear_boundary && boundary_failure_.synchronize_across_ranks())
        throw std::runtime_error("field boundary evaluation failed at face " +
                                 std::to_string(boundary_failure_.face) + " cell (" +
                                 std::to_string(boundary_failure_.i) + "," +
                                 std::to_string(boundary_failure_.j) + ")");
      require_exact_published_phi_("GeometricMG::solve");
      publish_last_solve_report_collectively_(cycles, relative_residual, status, action,
                                              "GeometricMG::solve");
      return cycles;
    };
    auto invalid_evaluation = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      return finish(cycles, std::numeric_limits<Real>::infinity(), SolveStatus::kInvalidEvaluation,
                    SolveAction::kRejectAttempt);
    };
    trace_mark("solve: before initial current_residual");
    last_bottom_seconds_ = 0.0;  // reset the per-solve bottom self-time (accumulated by vcycle_rec)
    double initial_norms[2] = {static_cast<double>(forcing_residual_local()),
                               static_cast<double>(current_residual_local())};
    all_reduce_max_inplace(initial_norms, 2);
    const Real forcing_norm = static_cast<Real>(initial_norms[0]);
    const Real r0 = static_cast<Real>(initial_norms[1]);
    if (!std::isfinite(static_cast<double>(forcing_norm)) ||
        !std::isfinite(static_cast<double>(r0)))
      return invalid_evaluation(0, r0);
    const Real report_denom = forcing_norm > Real(0) ? forcing_norm : Real(1);
    const Real stop = (rel_tol * forcing_norm > abs_tol) ? rel_tol * forcing_norm : abs_tol;
    trace_mark("solve: after initial current_residual");
    if (r0 <= stop) {
      last_cycles_ = 0;
      last_residual_ = r0;
      return finish(0, r0 / report_denom, SolveStatus::kSolved, SolveAction::kNone);
    }
    for (int c = 1; c <= max_cycles; ++c) {
      trace_mark("solve: before vcycle");
      vcycle();
      trace_mark("solve: after vcycle");
      const Real r = current_residual();
      if (!std::isfinite(static_cast<double>(r)))
        return invalid_evaluation(c, r);
      if (r <= stop) {
        last_cycles_ = c;
        last_residual_ = r;
        return finish(c, r / report_denom, SolveStatus::kSolved, SolveAction::kNone);
      }
    }
    last_cycles_ = max_cycles;
    last_residual_ = current_residual();
    if (!std::isfinite(static_cast<double>(last_residual_)))
      return invalid_evaluation(max_cycles, last_residual_);
    return finish(max_cycles, last_residual_ / report_denom, SolveStatus::kIterationLimit,
                  SolveAction::kRejectAttempt);
  }

  // EllipticSolver concept interface: solve() with no argument (default
  // tolerance) and residual() (alias of current_residual). Lets couplers
  // depend on the concept, not on GeometricMG directly. Propagates abs_tol_ (absolute
  // floor, default 0 -> historical relative criterion unchanged) to the mixed criterion.
  void solve() {
    require_collective_configuration_(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_,
                                      "GeometricMG::solve()");
    if (has_boundary_kernel_ && boundary_kernel_.observes_iteration) {
      require_collective_solve_inputs_("GeometricMG::solve()");
      if (!has_field_newton_options_)
        throw std::runtime_error(
            "iterate-dependent field boundary requires an installed nonlinear outer solver");
      SolveReport nonlinear_report = solve_boundary_newton(field_newton_options_);
      last_solve_report_ = std::move(nonlinear_report);
      if (!last_solve_report_.solved())
        throw std::runtime_error(std::string("field Newton solve failed: ") +
                                 last_solve_report_.status_name());
      require_exact_published_phi_("GeometricMG::solve()");
      return;
    }
    solve(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_);
  }
  Real residual() { return current_residual(); }

  // ABSOLUTE floor on the residual used by the no-argument solve() (the EllipticSolver
  // concept path, taken by the couplers / the runtime). Same units as residual().
  // Default 0: the criterion is purely relative to ||R(0)||. Setting it > 0 adds an
  // application-scaled absolute floor for near-zero affine forcing.
  void set_abs_tol(Real abs_tol) {
    if (abs_tol < Real(0) || !std::isfinite(static_cast<double>(abs_tol)))
      throw std::invalid_argument("GeometricMG::set_abs_tol requires finite abs_tol >= 0");
    abs_tol_ = abs_tol;
  }
  Real abs_tol() const { return abs_tol_; }

  // HARDENED solve for the embedded boundary at high resolution. On a fine grid, the geometric
  // V-cycle sometimes diverges near the conducting wall: coarsening is
  // NON-Galerkin and the circle mask is re-evaluated per level, so the coarse
  // correction becomes inconsistent with the fine boundary and the nu1=nu2=2 smoothing no longer
  // dominates it (cycle spectral radius > 1). The potential then diverges on each call (the
  // warm start propagates the divergence from one step to the next), hence a nan in the field at high
  // resolution (see docs/HERO_RUN_AMR.md). The divergence is ERRATIC in resolution
  // (it depends on the alignment of the circle on the grid hierarchy).
  //
  // Strategy, BIT-IDENTICAL when the solver already converges (or stalls):
  //   1. standard cycle at the current smoothing: EXACTLY the body of solve(rel_tol,
  //      max_cycles, abs_tol), so identical to the already-stable runs;
  //   2. ONLY if the final residual EXCEEDS the initial residual (true divergence,
  //      ratio > 1; not a mere stagnation ratio < 1, which we keep as-is to
  //      stay bit-identical): we harden the smoothing LOCALLY to the solve (nu doubled,
  //      nu1_/nu2_ restored on return, the next steps restart at nominal smoothing) and
  //      RESTART COLD (phi=0, the warm start was carrying the diverged state), until convergence
  //      or nu saturation. More smoothing makes the V-cycle contractive (GS dominates the
  //      inconsistent coarse correction): cf. sweep, nu=2 diverges at nc=640, nu>=4
  //      converges. Any run stable today did NOT diverge (divergence -> nan -> not
  //      recorded), so phase 2 never fires for them: bit-identical.
  int solve_robust(Real rel_tol, int max_cycles, Real abs_tol = Real(0)) {
    require_collective_solve_contract_(rel_tol, max_cycles, abs_tol, "GeometricMG::solve_robust");
    last_bottom_seconds_ = 0.0;
    const bool fallible_linear_boundary =
        has_boundary_kernel_ && !boundary_kernel_.observes_iteration;
    if (fallible_linear_boundary)
      boundary_failure_.reset();
    auto finish = [&](int cycles, Real relative_residual, SolveStatus status, SolveAction action) {
      if (fallible_linear_boundary && boundary_failure_.synchronize_across_ranks())
        throw std::runtime_error("field boundary evaluation failed at face " +
                                 std::to_string(boundary_failure_.face) + " cell (" +
                                 std::to_string(boundary_failure_.i) + "," +
                                 std::to_string(boundary_failure_.j) + ")");
      require_exact_published_phi_("GeometricMG::solve_robust");
      publish_last_solve_report_collectively_(cycles, relative_residual, status, action,
                                              "GeometricMG::solve_robust");
      return cycles;
    };
    auto invalid_evaluation = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      return finish(cycles, std::numeric_limits<Real>::infinity(), SolveStatus::kInvalidEvaluation,
                    SolveAction::kRejectAttempt);
    };
    double initial_norms[2] = {static_cast<double>(forcing_residual_local()),
                               static_cast<double>(current_residual_local())};
    all_reduce_max_inplace(initial_norms, 2);
    const Real forcing_norm = static_cast<Real>(initial_norms[0]);
    const Real r0 = static_cast<Real>(initial_norms[1]);
    if (!std::isfinite(static_cast<double>(forcing_norm)) ||
        !std::isfinite(static_cast<double>(r0)))
      return invalid_evaluation(0, r0);
    const Real report_denom = forcing_norm > Real(0) ? forcing_norm : Real(1);
    const Real stop = (rel_tol * forcing_norm > abs_tol) ? rel_tol * forcing_norm : abs_tol;
    auto solved = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      return finish(cycles, residual / report_denom, SolveStatus::kSolved, SolveAction::kNone);
    };
    auto iteration_limit = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      return finish(cycles, residual / report_denom, SolveStatus::kIterationLimit,
                    SolveAction::kRejectAttempt);
    };
    if (r0 <= stop)
      return solved(0, r0);
    int total = 0;
    for (int c = 1; c <= max_cycles; ++c) {  // phase 1: EXACTLY the body of solve()
      vcycle();
      ++total;
      const Real residual = current_residual();
      if (!std::isfinite(static_cast<double>(residual)))
        return invalid_evaluation(total, residual);
      if (residual <= stop)
        return solved(total, residual);  // -> bit-identical to recorded runs
    }
    const Real nominal_residual = current_residual();
    if (!std::isfinite(static_cast<double>(nominal_residual)))
      return invalid_evaluation(total, nominal_residual);
    if (nominal_residual <= r0)
      return iteration_limit(total, nominal_residual);  // stagnation (not divergence): keep as-is
    // phase 2: V-cycle divergence at the embedded boundary. Smoothing hardening LOCAL to the solve
    // (nu1_/nu2_ saved then RESTORED before each return): no permanent ratchet on the hot
    // path, the overhead is paid ONLY by the solve that diverges; the next solves restart at
    // nominal smoothing (reproducibility preserved, cost independent of history). Cold restart
    // (phi=0, the warm start was carrying the diverged state). More smoothing makes the cycle contractive.
    const int nu1_save = nu1_, nu2_save = nu2_;
    Real hardened_residual = nominal_residual;
    while (nu1_ < 64 || nu2_ < 64) {
      if (nu1_ < 64)
        nu1_ *= 2;
      if (nu2_ < 64)
        nu2_ *= 2;
      lev_[0].phi.set_val(Real(0));
      for (int c = 1; c <= max_cycles; ++c) {
        vcycle();
        ++total;
        const Real residual = current_residual();
        hardened_residual = residual;
        if (!std::isfinite(static_cast<double>(residual))) {
          nu1_ = nu1_save;
          nu2_ = nu2_save;
          return invalid_evaluation(total, residual);
        }
        if (residual <= stop) {
          nu1_ = nu1_save;
          nu2_ = nu2_save;
          return solved(total, residual);
        }
      }
    }
    nu1_ = nu1_save;
    nu2_ = nu2_save;
    // Best effort at maximal smoothing: a finite unconverged residual is an iteration limit,
    // never an invalid evaluation.
    return iteration_limit(total, hardened_residual);
  }

  // Current residual (infinity norm) at the finest level. all_reduce_max MANDATORY for
  // a DISTRIBUTED MULTI-BOX coarse: without it, norm_inf returns the LOCAL max (different per rank),
  // so the V-cycle stopping criterion fires at different iterations depending on the rank
  // -> different number of V-cycles (and fill_boundary calls) -> desynchronization of the
  // MPI fluxes (MPI_ERR_TRUNCATE). Idempotent under replication (local max = global on each rank) and
  // identity in serial -> bit-identical to the historical behavior.
  Real current_residual() { return all_reduce_max(current_residual_local()); }

  // ACCESS to the FINE-level (level 0) operator coefficient pointers and to the BC. Expose
  // EXACTLY what current_residual() passes to poisson_residual: an external caller (the Krylov
  // solver, which uses apply_laplacian as the matvec and needs a matvec CONSISTENT with the
  // MG residual) thus reuses the same operator, without duplicating the eps/kappa/Axy field wiring.
  // nullptr when the corresponding term is inactive (cf. the internal *_ptr). Additive: no existing
  // path calls them, the default behavior is unchanged.
  const MultiFab* op_mask() { return mask_ptr(0); }
  const MultiFab* op_coef() { return coef_ptr(0); }
  const MultiFab* op_eps() { return eps_ptr(0); }
  const MultiFab* op_kappa() { return kappa_ptr(0); }
  const MultiFab* op_eps_y() { return eps_y_ptr(0); }
  const MultiFab* op_a_xy() { return a_xy_ptr(0); }
  const MultiFab* op_a_yx() { return a_yx_ptr(0); }
  const BCRec& bc() const { return bc_; }
  const BoxArray& box_array() const { return lev_[0].ba; }
  const DistributionMapping& dmap() const { return lev_[0].dm; }

  /// Attach one already-resolved generated boundary kernel.  The direct function pointers and
  /// dependency-buffer table are copied into the solver once; V-cycle cell kernels never perform a
  /// registry lookup.  Omitting this call preserves the zero-overhead BCRec path.
  void set_boundary_kernel(const CompiledFieldBoundaryKernel& kernel,
                           const FieldBoundaryExecutionContext& context) {
    std::optional<CompiledFieldBoundaryKernel> staged_kernel;
    FieldBoundaryExecutionContext staged_context{};
    MultiFab staged_boundary_view;
    MultiFab staged_direction_view;
    std::vector<char, comm_allocator<char>> staged_replica_validation;
    run_collective_materialization_stage_("generated boundary configuration", [&] {
      kernel.validate();
      staged_kernel.emplace(kernel);
      staged_context = context;
      staged_context.failure = &boundary_failure_;
      staged_replica_validation = replica_validation_data_;
      bool needs_replica_storage = is_replicated();
      if (context.state_distributions != nullptr)
        for (int index = 0; index < context.state_count; ++index)
          needs_replica_storage = needs_replica_storage || context.state_distributions[index] ==
                                                               FieldDistribution::Replicated;
      if (context.field_distributions != nullptr)
        for (int index = 0; index < context.field_count; ++index)
          needs_replica_storage = needs_replica_storage || context.field_distributions[index] ==
                                                               FieldDistribution::Replicated;
      if (needs_replica_storage && staged_replica_validation.empty())
        staged_replica_validation.assign(detail::field_replica_consensus_storage_size(), char{0});
      staged_boundary_view = MultiFab(lev_[0].ba, lev_[0].dm, 1, 1);
      staged_direction_view = MultiFab(lev_[0].ba, lev_[0].dm, 1, 1);
    });
    // Commit only for the exact-contract check. Every previous value is retained by move in a
    // no-throw rollback record; a rank-divergent staged context therefore cannot destroy a valid
    // prepared configuration on the ranks that happened to accept it locally.
    CompiledFieldBoundaryKernel previous_kernel = std::move(boundary_kernel_);
    const FieldBoundaryExecutionContext previous_context = boundary_context_;
    auto previous_replica_validation = std::move(replica_validation_data_);
    MultiFab previous_boundary_view = std::move(lev_[0].boundary_view);
    MultiFab previous_direction_view = std::move(lev_[0].direction_view);
    std::unique_ptr<BoundaryNewtonCache> previous_cache = std::move(boundary_newton_cache_.value);
    const bool previous_has_boundary_kernel = has_boundary_kernel_;

    boundary_kernel_ = std::move(*staged_kernel);
    boundary_context_ = staged_context;
    replica_validation_data_ = std::move(staged_replica_validation);
    lev_[0].boundary_view = std::move(staged_boundary_view);
    lev_[0].direction_view = std::move(staged_direction_view);
    has_boundary_kernel_ = true;
    try {
      require_collective_configuration_(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_,
                                        "GeometricMG::set_boundary_kernel");
    } catch (...) {
      boundary_kernel_ = std::move(previous_kernel);
      boundary_context_ = previous_context;
      replica_validation_data_ = std::move(previous_replica_validation);
      lev_[0].boundary_view = std::move(previous_boundary_view);
      lev_[0].direction_view = std::move(previous_direction_view);
      boundary_newton_cache_.value = std::move(previous_cache);
      has_boundary_kernel_ = previous_has_boundary_kernel;
      throw;
    }
    if (has_field_newton_options_ && boundary_kernel_.observes_iteration &&
        boundary_kernel_.jvp != nullptr) {
      try {
        prepare_boundary_newton_cache_(field_newton_options_);
      } catch (...) {
        // The new configuration passed exact consensus and remains retryable, but no stale cache
        // from the previous kernel may authorize a solve after failed preparation.
        boundary_newton_cache_.value.reset();
        throw;
      }
    }
  }

  void clear_boundary_kernel() {
    // A prepared cache owns duplicated MPI communicators. Enter one exact world-level gate before
    // destroying it so MPI_Comm_free is reached by every owner in the same lifecycle order.
    require_collective_configuration_(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_,
                                      "GeometricMG::clear_boundary_kernel");
    clear_boundary_kernel_local_();
  }

  void set_boundary_context(const FieldBoundaryExecutionContext& context) {
    if (!has_boundary_kernel_)
      throw std::runtime_error("GeometricMG boundary context installed without a compiled kernel");
    FieldBoundaryExecutionContext staged_context{};
    std::vector<char, comm_allocator<char>> staged_replica_validation;
    run_collective_materialization_stage_("generated boundary context refresh", [&] {
      staged_context = context;
      staged_context.failure = &boundary_failure_;
      staged_replica_validation = replica_validation_data_;
      bool needs_replica_storage = is_replicated();
      if (context.state_distributions != nullptr)
        for (int index = 0; index < context.state_count; ++index)
          needs_replica_storage = needs_replica_storage || context.state_distributions[index] ==
                                                               FieldDistribution::Replicated;
      if (context.field_distributions != nullptr)
        for (int index = 0; index < context.field_count; ++index)
          needs_replica_storage = needs_replica_storage || context.field_distributions[index] ==
                                                               FieldDistribution::Replicated;
      if (needs_replica_storage && staged_replica_validation.empty())
        staged_replica_validation.assign(detail::field_replica_consensus_storage_size(), char{0});
    });
    const FieldBoundaryExecutionContext previous_context = boundary_context_;
    auto previous_replica_validation = std::move(replica_validation_data_);
    boundary_context_ = staged_context;
    replica_validation_data_ = std::move(staged_replica_validation);
    try {
      require_collective_configuration_(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_,
                                        "GeometricMG::set_boundary_context");
    } catch (...) {
      boundary_context_ = previous_context;
      replica_validation_data_ = std::move(previous_replica_validation);
      throw;
    }
    if (has_field_newton_options_ && boundary_kernel_.observes_iteration &&
        boundary_kernel_.jvp != nullptr) {
      try {
        prepare_boundary_newton_cache_(field_newton_options_);
      } catch (...) {
        boundary_newton_cache_.value.reset();
        throw;
      }
    }
  }

  void set_field_newton_options(const FieldNewtonOptions& options) {
    prepare_boundary_newton(options);
  }

  /// Materialize every persistent Newton/Krylov resource while configuration is still in its
  /// collective control phase.  A later solve may refresh values and snapshots, but it never
  /// allocates the cache, resolves a Krylov provider, or duplicates an MPI communicator.
  void prepare_boundary_newton(const FieldNewtonOptions& options) {
    require_collective_newton_options_(options, "GeometricMG::prepare_boundary_newton");

    field_newton_options_ = options;
    has_field_newton_options_ = true;
    require_collective_configuration_(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_,
                                      "GeometricMG::prepare_boundary_newton");
    if (!has_boundary_kernel_ || !boundary_kernel_.observes_iteration ||
        boundary_kernel_.jvp == nullptr)
      return;
    try {
      prepare_boundary_newton_cache_(options);
    } catch (...) {
      // A failed refresh may have invalidated an existing prepared problem snapshot. Remove its
      // authorization collectively so solve cannot continue with a stale cache; a later explicit
      // prepare retries the complete materialization protocol.
      boundary_newton_cache_.value.reset();
      throw;
    }
  }

  const SolveReport& last_solve_report() const { return last_solve_report_; }

  /// Monotonic identity of the active Newton cache. It changes only after a successful rebuild for
  /// a different field layout or GMRES restart.
  std::uint64_t boundary_newton_cache_generation() const;

  /// Number of persistent MultiFab slots owned by the active Newton cache (scratch, prepared
  /// operator/preconditioner state, and GMRES workspace). Zero before the first cached solve.
  std::size_t boundary_newton_cache_allocation_count() const;

  SolveReport solve_boundary_newton(const FieldNewtonOptions& options) {
    require_collective_configuration_(kMGDefaultRelTol, kMGDefaultMaxCycles, abs_tol_,
                                      "GeometricMG::solve_boundary_newton");
    require_collective_solve_inputs_("GeometricMG::solve_boundary_newton");
    if (!has_boundary_kernel_ || !boundary_kernel_.observes_iteration ||
        boundary_kernel_.jvp == nullptr)
      return make_solve_report_collectively_(0, Real(0), SolveStatus::kCapabilityFailure,
                                             SolveAction::kFailRun,
                                             "GeometricMG::solve_boundary_newton");
    require_prepared_boundary_newton_(options);
    return solve_boundary_newton_cached(options);
  }

  void apply_jvp(const MultiFab& iterate, const MultiFab& direction, MultiFab& output) {
    apply_laplacian_jvp(iterate, direction, lev_[0].geom, bc_, output, boundary_kernel_ptr(0),
                        boundary_context_ptr(0),
                        has_boundary_kernel_ ? &lev_[0].direction_view : nullptr, coef_ptr(0),
                        eps_ptr(0), kappa_ptr(0), eps_y_ptr(0), a_xy_ptr(0), a_yx_ptr(0));
  }

 private:
  void clear_boundary_kernel_local_() {
    has_boundary_kernel_ = false;
    boundary_kernel_ = {};
    boundary_context_ = {};
    boundary_newton_cache_.value.reset();
  }

  static void require_collective_newton_options_(const FieldNewtonOptions& options,
                                                 const char* where) {
    long validation_failure_local = 0;
    try {
      validate_field_newton_options(options);
    } catch (...) {
      validation_failure_local = 1;
    }
    if (all_reduce_max(validation_failure_local) != 0)
      throw std::invalid_argument(std::string(where) +
                                  ": invalid options on at least one communicator rank");

    std::string contract;
    long contract_failure_local = 0;
    const auto append = [&](const auto& value) {
      if (contract_failure_local != 0)
        return;
      try {
        detail::append_exact_contract_value(contract, value);
      } catch (...) {
        contract_failure_local = 1;
      }
    };
    append(options.tolerance);
    append(options.max_iterations);
    append(options.linear_tolerance);
    append(options.linear_max_iterations);
    append(options.restart);
    append(options.armijo);
    append(options.minimum_step);
    if (all_reduce_max(contract_failure_local) != 0)
      throw std::runtime_error(std::string(where) +
                               ": option contract failed to materialize on at least one "
                               "communicator rank");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("geometric-mg-newton-options"), std::string_view(contract)}}))
      throw std::invalid_argument(std::string(where) +
                                  ": options differ between communicator ranks");
  }

  template <class Action>
  static void run_collective_materialization_stage_(const char* stage, Action&& action) {
    long failure_local = 0;
    try {
      std::forward<Action>(action)();
    } catch (...) {
      failure_local = 1;
    }
    if (all_reduce_max(failure_local) != 0)
      throw std::runtime_error(std::string("GeometricMG ") + stage +
                               " failed on at least one communicator rank");
  }

  static SolveReport make_solve_report_collectively_(int iterations, Real relative_residual,
                                                     SolveStatus status, SolveAction action,
                                                     const char* where) {
    std::optional<SolveReport> staged;
    run_collective_materialization_stage_("solve-report materialization", [&] {
      staged.emplace();
      staged->iters = iterations;
      staged->rel_residual = relative_residual;
      if (status == SolveStatus::kSolved)
        staged->mark_solved();
      else
        staged->mark_failed(status, action);
    });
    if (!staged)
      throw std::logic_error(std::string(where) + ": missing collectively materialized report");
    static_assert(std::is_nothrow_move_constructible_v<SolveReport>);
    return std::move(*staged);
  }

  void publish_last_solve_report_collectively_(int iterations, Real relative_residual,
                                               SolveStatus status, SolveAction action,
                                               const char* where) {
    SolveReport staged =
        make_solve_report_collectively_(iterations, relative_residual, status, action, where);
    // std::allocator-backed std::string move assignment is non-throwing. Publish only after the
    // exact field contract and the report-allocation gate have completed on every rank.
    static_assert(std::is_nothrow_move_assignable_v<SolveReport>);
    last_solve_report_ = std::move(staged);
  }

  void require_collective_configuration_(Real rel_tol, int max_cycles, Real abs_tol,
                                         const char* where) const {
    bool invalid = !(rel_tol > Real(0)) || !std::isfinite(static_cast<double>(rel_tol)) ||
                   max_cycles < 1 || abs_tol < Real(0) ||
                   !std::isfinite(static_cast<double>(abs_tol)) ||
                   !field_distribution_is_valid(distribution_);
    if (has_boundary_kernel_) {
      invalid =
          invalid || boundary_context_.state_count < 0 || boundary_context_.field_count < 0 ||
          boundary_context_.parameter_count < 0 ||
          (boundary_context_.state_count > 0 && boundary_context_.states == nullptr) ||
          (boundary_context_.state_count > 0 && boundary_context_.state_distributions == nullptr) ||
          (boundary_context_.field_count > 0 && boundary_context_.fields == nullptr) ||
          (boundary_context_.field_count > 0 && boundary_context_.field_distributions == nullptr) ||
          (boundary_context_.parameter_count > 0 && boundary_context_.parameters == nullptr) ||
          (boundary_context_.parameters != nullptr &&
           static_cast<std::size_t>(boundary_context_.parameter_count) >
               boundary_context_.parameters->size());
      if (!invalid) {
        for (int index = 0; index < boundary_context_.state_count; ++index)
          invalid = invalid || boundary_context_.states[index] == nullptr ||
                    !field_distribution_is_valid(boundary_context_.state_distributions[index]);
        for (int index = 0; index < boundary_context_.field_count; ++index)
          invalid = invalid || boundary_context_.fields[index] == nullptr ||
                    !field_distribution_is_valid(boundary_context_.field_distributions[index]);
      }
    }
    if (all_reduce_max(invalid ? 1L : 0L) != 0)
      throw std::invalid_argument(std::string(where) +
                                  ": invalid controls or generated-boundary context");

    std::string contract;
    long contract_failure_local = 0;
    const auto append = [&](const auto& value) {
      if (contract_failure_local != 0)
        return;
      try {
        detail::append_exact_contract_value(contract, value);
      } catch (...) {
        contract_failure_local = 1;
      }
    };
    const auto append_text = [&](std::string_view value) {
      append(static_cast<std::uint64_t>(value.size()));
      if (contract_failure_local != 0)
        return;
      try {
        contract.append(value.data(), value.size());
      } catch (...) {
        contract_failure_local = 1;
      }
    };
    append(rel_tol);
    append(max_cycles);
    append(abs_tol);
    append(static_cast<std::uint8_t>(distribution_));
    append(nu1_);
    append(nu2_);
    append(nbottom_);
    append(coarse_threshold_);
    append(cut_theta_min_);
    append_text(active_.collective_contract());
    append_text(levelset_.collective_contract());
    append(cut_cell_);
    append(has_eps_);
    append(has_eps_y_);
    append(has_kappa_);
    append(has_cross_);
    append_text(eps_provider_contract_);
    append_text(eps_y_provider_contract_);
    append_text(kappa_provider_contract_);
    append_text(a_xy_provider_contract_);
    append_text(a_yx_provider_contract_);
    append(has_boundary_kernel_);
    append(has_field_newton_options_);
    append(static_cast<std::uint64_t>(lev_.size()));

    append(static_cast<std::uint8_t>(bc_.xlo));
    append(static_cast<std::uint8_t>(bc_.xhi));
    append(static_cast<std::uint8_t>(bc_.ylo));
    append(static_cast<std::uint8_t>(bc_.yhi));
    append(bc_.xlo_val);
    append(bc_.xhi_val);
    append(bc_.ylo_val);
    append(bc_.yhi_val);
    append(bc_.xlo_alpha);
    append(bc_.xlo_beta);
    append(bc_.xhi_alpha);
    append(bc_.xhi_beta);
    append(bc_.ylo_alpha);
    append(bc_.ylo_beta);
    append(bc_.yhi_alpha);
    append(bc_.yhi_beta);
    append(bc_.dx);
    append(bc_.dy);
    for (const MGLevel& level : lev_) {
      append(level.geom.domain.lo[0]);
      append(level.geom.domain.lo[1]);
      append(level.geom.domain.hi[0]);
      append(level.geom.domain.hi[1]);
      append(level.geom.xlo);
      append(level.geom.xhi);
      append(level.geom.ylo);
      append(level.geom.yhi);
    }

    if (has_boundary_kernel_) {
      append_text(boundary_kernel_.identity);
      append_text(boundary_kernel_.residual_identity);
      append_text(boundary_kernel_.jvp_identity);
      append(boundary_kernel_.observes_iteration);
      append(boundary_context_.point.time);
      append(boundary_context_.point.dt);
      append(boundary_context_.point.clock_slot);
      append(boundary_context_.point.partition_slot);
      append(boundary_context_.point.stage_slot);
      append(boundary_context_.point.step);
      append(boundary_context_.point.substep);
      append(boundary_context_.point.iteration);
      append(boundary_context_.state_count);
      append(boundary_context_.field_count);
      append(boundary_context_.parameter_count);
      for (int index = 0; index < boundary_context_.state_count; ++index)
        append(static_cast<std::uint8_t>(boundary_context_.state_distributions[index]));
      for (int index = 0; index < boundary_context_.field_count; ++index)
        append(static_cast<std::uint8_t>(boundary_context_.field_distributions[index]));
      for (int index = 0; index < boundary_context_.parameter_count; ++index)
        append((*boundary_context_.parameters)[static_cast<std::size_t>(index)]);
    }
    if (has_field_newton_options_) {
      append(field_newton_options_.tolerance);
      append(field_newton_options_.max_iterations);
      append(field_newton_options_.linear_tolerance);
      append(field_newton_options_.linear_max_iterations);
      append(field_newton_options_.restart);
      append(field_newton_options_.armijo);
      append(field_newton_options_.minimum_step);
    }

    if (all_reduce_max(contract_failure_local) != 0)
      throw std::runtime_error(std::string(where) +
                               ": collective contract materialization failed on at least one "
                               "communicator rank");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("geometric-mg-collective-contract"), std::string_view(contract)}}))
      throw std::invalid_argument(std::string(where) +
                                  ": collective contract differs between communicator ranks");
  }

  void require_collective_solve_inputs_(const char* where) {
    // Materialize every dynamic diagnostic label before the first field collective. A rank-local
    // allocation failure must be published while all ranks are still at the same control gate;
    // building `level N`/`state N` strings between field collectives can otherwise strand peers.
    std::vector<std::string> locations;
    long location_failure_local = 0;
    try {
      const auto add_location = [&](std::string role) {
        locations.emplace_back(std::string(where) + ": " + std::move(role));
      };
      add_location("finest phi");
      add_location("finest rhs");
      for (std::size_t level_index = 0; level_index < lev_.size(); ++level_index) {
        const std::string prefix = "level " + std::to_string(level_index) + " ";
        if (active_)
          add_location(prefix + "mask");
        if (cut_cell_)
          add_location(prefix + "cut-cell coefficients");
        if (has_eps_)
          add_location(prefix + "epsilon-x");
        if (has_eps_y_)
          add_location(prefix + "epsilon-y");
        if (has_kappa_)
          add_location(prefix + "reaction coefficient");
        if (has_cross_) {
          add_location(prefix + "a-xy");
          add_location(prefix + "a-yx");
        }
      }
      if (has_boundary_kernel_) {
        for (int index = 0; index < boundary_context_.state_count; ++index)
          add_location("state " + std::to_string(index));
        for (int index = 0; index < boundary_context_.field_count; ++index)
          add_location("field " + std::to_string(index));
      }
    } catch (...) {
      location_failure_local = 1;
    }
    if (all_reduce_max(location_failure_local) != 0)
      throw std::runtime_error(std::string(where) +
                               ": solve-input labels failed to materialize on at least one "
                               "communicator rank");
    std::size_t location_index = 0;
    bool label_plan_invalid = false;
    const auto next_location = [&]() noexcept {
      if (location_index >= locations.size()) {
        label_plan_invalid = true;
        return "GeometricMG invalid solve-input label plan";
      }
      return locations[location_index++].c_str();
    };
    const auto require_field = [&](const MultiFab& field, FieldDistribution field_distribution,
                                   const char* location) {
      detail::require_collective_field_distribution_layout(field, field_distribution, location);
      if (field_distribution == FieldDistribution::Replicated)
        detail::require_exact_replicated_field_values_prevalidated(
            field, replica_validation_data_.data(), replica_validation_data_.size(), location);
    };
    const auto require_boundary_dependency =
        [&](const MultiFab& field, FieldDistribution field_distribution, const char* location) {
          require_field(field, field_distribution, location);
          bool materializes_local_solver_patches =
              field.box_array().boxes() == lev_[0].phi.box_array().boxes();
          for (int local = 0; materializes_local_solver_patches && local < lev_[0].phi.local_size();
               ++local) {
            const int global = lev_[0].phi.global_index(local);
            const int dependency_local = field.local_index_of(global);
            materializes_local_solver_patches =
                dependency_local >= 0 && field.box(dependency_local) == lev_[0].phi.box(local);
          }
          if (all_reduce_max(materializes_local_solver_patches ? 0L : 1L) != 0)
            throw std::invalid_argument(std::string(location) +
                                        " does not materialize every local patch required by "
                                        "the solved field");
        };

    require_field(lev_[0].phi, distribution_, next_location());
    require_field(lev_[0].rhs, distribution_, next_location());
    for (std::size_t level_index = 0; level_index < lev_.size(); ++level_index) {
      const MGLevel& level = lev_[level_index];
      if (active_)
        require_field(level.mask, distribution_, next_location());
      if (cut_cell_)
        require_field(level.coef, distribution_, next_location());
      if (has_eps_)
        require_field(level.eps, distribution_, next_location());
      if (has_eps_y_)
        require_field(level.eps_y, distribution_, next_location());
      if (has_kappa_)
        require_field(level.kappa, distribution_, next_location());
      if (has_cross_) {
        require_field(level.a_xy, distribution_, next_location());
        require_field(level.a_yx, distribution_, next_location());
      }
    }
    if (has_boundary_kernel_) {
      for (int index = 0; index < boundary_context_.state_count; ++index)
        require_boundary_dependency(*boundary_context_.states[index],
                                    boundary_context_.state_distributions[index], next_location());
      for (int index = 0; index < boundary_context_.field_count; ++index)
        require_boundary_dependency(*boundary_context_.fields[index],
                                    boundary_context_.field_distributions[index], next_location());
    }
    if (label_plan_invalid || location_index != locations.size())
      throw std::logic_error("GeometricMG solve-input label plan was not consumed exactly");
  }

  void require_collective_solve_contract_(Real rel_tol, int max_cycles, Real abs_tol,
                                          const char* where) {
    require_collective_configuration_(rel_tol, max_cycles, abs_tol, where);
    require_collective_solve_inputs_(where);
  }

  void require_exact_published_phi_(const char* where) {
    if (!is_replicated())
      return;
    detail::require_exact_replicated_field_values_prevalidated(
        lev_[0].phi, replica_validation_data_.data(), replica_validation_data_.size(), where);
  }

  Real evaluate_residual_local(MultiFab& iterate) {
    auto& L = lev_[0];
    poisson_residual(iterate, L.rhs, L.geom, bc_, L.res, mask_ptr(0), coef_ptr(0), eps_ptr(0),
                     kappa_ptr(0), eps_y_ptr(0), a_xy_ptr(0), a_yx_ptr(0), boundary_kernel_ptr(0),
                     boundary_context_ptr(0), has_boundary_kernel_ ? &L.boundary_view : nullptr);
    return norm_inf(L.res);
  }

  // Exact affine forcing R(0), evaluated with the same discretized operator, masks, coefficients,
  // generated boundary kernel and logical context as the warm-start residual. zero_probe is a
  // persistent level-0 iterate, allocated and initialized once by add_level(); L.res is reused as
  // the output, so this adds one residual stencil per solve but no allocation or MPI collective.
  Real forcing_residual_local() {
    trace_mark("forcing_residual: before poisson_residual");
    const Real result = evaluate_residual_local(lev_[0].zero_probe);
    trace_mark("forcing_residual: after norm_inf");
    return result;
  }

  Real current_residual_local() {
    trace_mark("current_residual: before poisson_residual");
    const Real result = evaluate_residual_local(lev_[0].phi);
    trace_mark("current_residual: after norm_inf");
    return result;
  }

  struct MGLevel {
    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    MultiFab phi, rhs, res, mask, coef, eps, kappa, eps_y, a_xy, a_yx;
    // REUSED V-cycle buffers, allocated once by the constructor for the NON-bottom levels:
    // corr = prolonged correction (level layout); cfine = "fine coarsened" grid shared by the
    // restriction (average_down) and the prolongation (interpolate) of the level. The bottom leaves them empty
    // (vcycle_rec returns before touching them, and its coarsen would be degenerate).
    MultiFab corr, cfine;
    MultiFab boundary_view, direction_view;
    MultiFab zero_probe;  // persistent level-0 zero iterate for the exact affine forcing R(0)
  };

  /// Stable-address storage for the nonlinear boundary solve. The prepared callbacks capture this
  /// cache (and reach the owning GeometricMG through owner_), so the object is deliberately neither
  /// copied nor moved after construction. A compatible solve reuses every field and callback.
  struct BoundaryNewtonCache {
    static MultiFab allocate_like(const MultiFab& prototype, int ghosts) {
      MultiFab field(prototype.box_array(), prototype.dmap(), prototype.ncomp(), ghosts);
      if (ghosts == prototype.n_grow())
        field.share_halo_cache_from(prototype);
      return field;
    }

    static OperatorFingerprint make_linear_topology(
        GeometricMG& owner, const MultiFab& prototype,
        const PreparedVectorDistribution& vector_distribution) {
      OperatorFingerprint topology = detail::layout_fingerprint(prototype, vector_distribution);
      detail::fingerprint_geometry(topology, owner.lev_[0].geom);
      detail::fingerprint_boundary(topology, owner.bc_);
      return topology;
    }

    static std::uint32_t operator_shape_bits(const GeometricMG& owner) noexcept {
      return (owner.active_ ? UINT32_C(1) : UINT32_C(0)) |
             (owner.cut_cell_ ? UINT32_C(1) << 1u : UINT32_C(0)) |
             (owner.has_eps_ ? UINT32_C(1) << 2u : UINT32_C(0)) |
             (owner.has_eps_y_ ? UINT32_C(1) << 3u : UINT32_C(0)) |
             (owner.has_kappa_ ? UINT32_C(1) << 4u : UINT32_C(0)) |
             (owner.has_cross_ ? UINT32_C(1) << 5u : UINT32_C(0));
    }

    static std::unique_ptr<GeometricMG> clone_workspace_engine(BoundaryNewtonCache& source) {
      auto engine = std::make_unique<GeometricMG>(*source.owner_);
      engine->boundary_newton_cache_.value.reset();
      engine->boundary_newton_cache_.generation = 0;
      return engine;
    }

    static void detach_workspace_communication_caches(GeometricMG& engine) {
      // MultiFab value copies intentionally share communication caches. A prepared execution
      // session owns its native scratch and may run concurrently with another session, so none of
      // the reusable communication buffers may remain shared with the source solver.
      for (MGLevel& level : engine.lev_) {
        level.phi.detach_communication_caches();
        level.rhs.detach_communication_caches();
        level.res.detach_communication_caches();
        level.mask.detach_communication_caches();
        level.coef.detach_communication_caches();
        level.eps.detach_communication_caches();
        level.kappa.detach_communication_caches();
        level.eps_y.detach_communication_caches();
        level.a_xy.detach_communication_caches();
        level.a_yx.detach_communication_caches();
        level.corr.detach_communication_caches();
        level.cfine.detach_communication_caches();
        level.boundary_view.detach_communication_caches();
        level.direction_view.detach_communication_caches();
        level.zero_probe.detach_communication_caches();
      }
    }

    struct WorkspacePreconditionerState {
      WorkspacePreconditionerState(BoundaryNewtonCache& source, const ExecutionLane& lane)
          : source_(&source), engine_(clone_workspace_engine(source)), lane_(&lane) {
        // Construction is the materialization boundary: an allocation failure is reported before
        // any rank can enter prepare/apply. The V-cycle is the homogeneous preconditioner and must
        // never execute the nonlinear generated boundary closure.
        // The copied engine deliberately has no derived cache. Session construction is already
        // inside the owning prepared problem's collective factory gate, so use the non-public
        // local reset rather than nesting the public WORLD lifecycle protocol.
        engine_->clear_boundary_kernel_local_();
        detach_workspace_communication_caches(*engine_);
      }

      void prepare() {
        require_materialized();
        // The session storage is persistent, but coefficient values are not assumed immutable
        // across prepared logical points. Refresh them into the already allocated hierarchy before
        // each bind; no provider, MultiFab, or communication cache is created here.
        GeometricMG& source_engine = *source_->owner_;
        if (engine_->lev_.size() != source_engine.lev_.size())
          throw std::logic_error(
              "boundary Newton preconditioner hierarchy changed after cache preparation");
        for (std::size_t level = 0; level < engine_->lev_.size(); ++level) {
          MGLevel& target = engine_->lev_[level];
          const MGLevel& source = source_engine.lev_[level];
          if (source_engine.active_)
            lincomb(target.mask, Real(1), source.mask, Real(0), source.mask);
          if (source_engine.cut_cell_)
            lincomb(target.coef, Real(1), source.coef, Real(0), source.coef);
          if (source_engine.has_eps_)
            lincomb(target.eps, Real(1), source.eps, Real(0), source.eps);
          if (source_engine.has_kappa_)
            lincomb(target.kappa, Real(1), source.kappa, Real(0), source.kappa);
          if (source_engine.has_eps_y_)
            lincomb(target.eps_y, Real(1), source.eps_y, Real(0), source.eps_y);
          if (source_engine.has_cross_) {
            lincomb(target.a_xy, Real(1), source.a_xy, Real(0), source.a_xy);
            lincomb(target.a_yx, Real(1), source.a_yx, Real(0), source.a_yx);
          }
        }
      }

      void apply(MultiFab& out, const MultiFab& in) {
        require_materialized();
        MGLevel& level = engine_->lev_[0];
        PureFieldAlgebra::zero_valid(level.phi);
        lincomb(level.rhs, Real(1), in, Real(0), in);
        engine_->vcycle(*lane_);
        lincomb(out, Real(1), level.phi, Real(0), level.phi);
      }

      [[nodiscard]] std::size_t allocation_count() const noexcept {
        return engine_ ? engine_->persistent_field_count() : 0u;
      }

     private:
      void require_materialized() const {
        if (!engine_)
          throw std::logic_error(
              "boundary Newton preconditioner session has no materialized engine");
      }

      BoundaryNewtonCache* source_ = nullptr;
      std::unique_ptr<GeometricMG> engine_{};
      const ExecutionLane* lane_ = nullptr;
    };

    struct WorkspaceOperatorState {
      WorkspaceOperatorState(BoundaryNewtonCache& source, const ExecutionLane& lane)
          : source_(&source),
            geometry_(source.owner_->lev_[0].geom),
            boundary_(source.owner_->bc_),
            boundary_kernel_(source.owner_->boundary_kernel_),
            boundary_context_(source.owner_->boundary_context_),
            accepted_(source.accepted_),
            direction_view_(source.owner_->lev_[0].direction_view),
            coef_(source.owner_->lev_[0].coef),
            eps_(source.owner_->lev_[0].eps),
            kappa_(source.owner_->lev_[0].kappa),
            eps_y_(source.owner_->lev_[0].eps_y),
            a_xy_(source.owner_->lev_[0].a_xy),
            a_yx_(source.owner_->lev_[0].a_yx),
            has_coef_(source.owner_->cut_cell_),
            has_eps_(source.owner_->has_eps_),
            has_kappa_(source.owner_->has_kappa_),
            has_eps_y_(source.owner_->has_eps_y_),
            has_cross_(source.owner_->has_cross_),
            lane_(&lane) {
        // Freeze only the level-0 data the JVP reads; cloning the full multigrid hierarchy here
        // would multiply every Krylov workspace's memory for fields the operator never touches.
        // The failure sink and direction view are session-private mutable state.
        if (!source.owner_->has_boundary_kernel_ || boundary_kernel_.jvp == nullptr)
          throw std::logic_error(
              "boundary Newton operator session requires a prepared boundary JVP");
        boundary_context_.failure = &boundary_failure_;
        accepted_.detach_communication_caches();
        direction_view_.detach_communication_caches();
        coef_.detach_communication_caches();
        eps_.detach_communication_caches();
        kappa_.detach_communication_caches();
        eps_y_.detach_communication_caches();
        a_xy_.detach_communication_caches();
        a_yx_.detach_communication_caches();
      }

      void prepare() {
        // `accepted_` is the current Newton point, not a construction-time constant. The same
        // prepared session is deliberately reused across Newton iterations and time stages, so
        // refresh every mutable numerical input into its private, already allocated storage.
        GeometricMG& owner = *source_->owner_;
        geometry_ = owner.lev_[0].geom;
        boundary_ = owner.bc_;
        boundary_context_ = owner.boundary_context_;
        boundary_context_.failure = &boundary_failure_;
        lincomb(accepted_, Real(1), source_->accepted_, Real(0), source_->accepted_);
        if (owner.cut_cell_)
          lincomb(coef_, Real(1), owner.lev_[0].coef, Real(0), owner.lev_[0].coef);
        if (owner.has_eps_)
          lincomb(eps_, Real(1), owner.lev_[0].eps, Real(0), owner.lev_[0].eps);
        if (owner.has_kappa_)
          lincomb(kappa_, Real(1), owner.lev_[0].kappa, Real(0), owner.lev_[0].kappa);
        if (owner.has_eps_y_)
          lincomb(eps_y_, Real(1), owner.lev_[0].eps_y, Real(0), owner.lev_[0].eps_y);
        if (owner.has_cross_) {
          lincomb(a_xy_, Real(1), owner.lev_[0].a_xy, Real(0), owner.lev_[0].a_xy);
          lincomb(a_yx_, Real(1), owner.lev_[0].a_yx, Real(0), owner.lev_[0].a_yx);
        }
      }

      void apply(MultiFab& out, const MultiFab& in) {
        boundary_failure_.reset();
        copy_field_valid(in, direction_view_);
        fill_ghosts(direction_view_, geometry_.domain, homogeneous_field_bc(boundary_), *lane_);
        for (int face = 0; face < 4; ++face)
          boundary_kernel_.prepare_jvp_view(face, accepted_, in, direction_view_, geometry_,
                                            boundary_context_);
        apply_laplacian(direction_view_, geometry_, out, has_coef_ ? &coef_ : nullptr,
                        has_eps_ ? &eps_ : nullptr, has_kappa_ ? &kappa_ : nullptr,
                        has_eps_y_ ? &eps_y_ : nullptr, has_cross_ ? &a_xy_ : nullptr,
                        has_cross_ ? &a_yx_ : nullptr);
        for (int face = 0; face < 4; ++face)
          boundary_kernel_.apply_jvp(face, accepted_, in, out, geometry_, boundary_context_);
        if (synchronize_failure())
          PureFieldAlgebra::fill_valid(out, std::numeric_limits<Real>::quiet_NaN());
      }

      [[nodiscard]] std::size_t allocation_count() const noexcept { return 8u; }

     private:
      bool synchronize_failure() {
        const bool local_failed = boundary_failure_.failed();
        const long failure_count = all_reduce_sum(local_failed ? 1L : 0L, *lane_);
        if (failure_count == 0) {
          boundary_failure_.reset();
          return false;
        }

        const int rank = lane_->rank();
        const int owner = static_cast<int>(
            all_reduce_min(static_cast<double>(local_failed ? rank : lane_->size()), *lane_));
        const bool publish = local_failed && rank == owner;
        boundary_failure_.code = static_cast<int>(
            all_reduce_sum(publish ? static_cast<long>(boundary_failure_.code) : 0L, *lane_));
        boundary_failure_.face = static_cast<int>(
            all_reduce_sum(publish ? static_cast<long>(boundary_failure_.face) : 0L, *lane_));
        boundary_failure_.i = static_cast<int>(
            all_reduce_sum(publish ? static_cast<long>(boundary_failure_.i) : 0L, *lane_));
        boundary_failure_.j = static_cast<int>(
            all_reduce_sum(publish ? static_cast<long>(boundary_failure_.j) : 0L, *lane_));
        boundary_failure_.value = static_cast<Real>(
            all_reduce_sum(publish ? static_cast<double>(boundary_failure_.value) : 0.0, *lane_));
        return true;
      }

      BoundaryNewtonCache* source_ = nullptr;
      Geometry geometry_;
      BCRec boundary_;
      CompiledFieldBoundaryKernel boundary_kernel_{};
      FieldBoundaryFailure boundary_failure_{};
      FieldBoundaryExecutionContext boundary_context_{};
      MultiFab accepted_;
      MultiFab direction_view_;
      MultiFab coef_;
      MultiFab eps_;
      MultiFab kappa_;
      MultiFab eps_y_;
      MultiFab a_xy_;
      MultiFab a_yx_;
      bool has_coef_ = false;
      bool has_eps_ = false;
      bool has_kappa_ = false;
      bool has_eps_y_ = false;
      bool has_cross_ = false;
      const ExecutionLane* lane_ = nullptr;
    };

    BoundaryNewtonCache(GeometricMG& owner, int restart)
        : owner_(&owner),
          vector_distribution_(PreparedVectorDistribution(owner.distribution_)),
          operator_shape_bits_(operator_shape_bits(owner)),
          restart_(restart) {}

    /// Build in a canonical collective order. No constructor that duplicates a communicator is
    /// entered until every rank has successfully materialized all of its rank-local arguments.
    /// Conversely, once a duplicated lane exists, its owning constructor provides the next common
    /// failure gate. This prevents a rank-local bad_alloc from stranding peers in MPI_Comm_dup.
    static std::unique_ptr<BoundaryNewtonCache> materialize(GeometricMG& owner, int restart) {
      std::unique_ptr<BoundaryNewtonCache> candidate;
      GeometricMG::run_collective_materialization_stage_("boundary Newton cache shell", [&] {
        candidate = std::make_unique<BoundaryNewtonCache>(owner, restart);
      });
      BoundaryNewtonCache& cache = *candidate;

      GeometricMG::run_collective_materialization_stage_(
          "boundary Newton layout fingerprints", [&] {
            cache.phi_layout_ =
                detail::layout_fingerprint(owner.lev_[0].phi, cache.vector_distribution_);
            cache.rhs_layout_ =
                detail::layout_fingerprint(owner.lev_[0].rhs, cache.vector_distribution_);
          });
      const auto allocate_field = [&](const char* stage, MultiFab& target,
                                      const MultiFab& prototype, int ghosts) {
        GeometricMG::run_collective_materialization_stage_(
            stage, [&] { target = allocate_like(prototype, ghosts); });
      };
      allocate_field("boundary Newton published snapshot", cache.published_snapshot_,
                     owner.lev_[0].phi, owner.lev_[0].phi.n_grow());
      allocate_field("boundary Newton accepted iterate", cache.accepted_, owner.lev_[0].phi,
                     owner.lev_[0].phi.n_grow());
      allocate_field("boundary Newton trial iterate", cache.trial_, owner.lev_[0].phi,
                     owner.lev_[0].phi.n_grow());
      allocate_field("boundary Newton residual", cache.residual_, owner.lev_[0].rhs, 0);
      allocate_field("boundary Newton trial residual", cache.trial_residual_, owner.lev_[0].rhs, 0);
      allocate_field("boundary Newton correction", cache.delta_, owner.lev_[0].phi,
                     owner.lev_[0].phi.n_grow());
      allocate_field("boundary Newton rhs snapshot", cache.rhs_snapshot_, owner.lev_[0].rhs,
                     owner.lev_[0].rhs.n_grow());

      GeometricMG::run_collective_materialization_stage_("boundary Newton topology", [&] {
        cache.footprint_ = {cache.delta_.ncomp(), cache.delta_.n_grow(), true};
        cache.linear_topology_ =
            make_linear_topology(owner, cache.delta_, cache.vector_distribution_);
      });

      std::string operator_contract;
      std::string preconditioner_contract;
      std::optional<PreparedAffineOperatorProvider> operator_provider;
      std::optional<PreparedLinearPreconditioner> preconditioner;
      std::optional<PreparedNullspacePolicy> nullspace_policy;
      std::optional<OperatorSnapshotProbe> snapshot_probe;
      std::optional<PreparedVectorDistribution> problem_distribution;
      std::string problem_lane_identity;
      GeometricMG::run_collective_materialization_stage_(
          "boundary Newton prepared-problem arguments", [&] {
            operator_contract = std::string(owner.prepared_operator_contract().exact_fingerprint());
            preconditioner_contract = operator_contract;
            BoundaryNewtonCache* stable_cache = &cache;
            operator_provider.emplace(PreparedAffineOperatorProvider::trusted_extension(
                {"pops.elliptic.geometric-mg.boundary-newton-operator", 1},
                std::move(operator_contract), [stable_cache](const ExecutionLane& lane) {
                  auto state = std::make_shared<WorkspaceOperatorState>(*stable_cache, lane);
                  return PreparedAffineOperatorSessionCallbacks{
                      [state]() { state->prepare(); },
                      [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
                      [state] { return state->allocation_count(); }};
                }));
            auto preconditioner_provider = PreparedLinearPreconditionerProvider::trusted_extension(
                {"pops.elliptic.geometric-mg.boundary-newton-preconditioner", 1},
                std::move(preconditioner_contract), [stable_cache](const ExecutionLane& lane) {
                  auto state = std::make_shared<WorkspacePreconditionerState>(*stable_cache, lane);
                  return PreparedLinearPreconditionerSessionCallbacks{
                      [state]() { state->prepare(); },
                      [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
                      [state] { return state->allocation_count(); }};
                });
            preconditioner.emplace(cache.delta_, std::move(preconditioner_provider),
                                   cache.vector_distribution_);
            nullspace_policy.emplace(PreparedNullspacePolicy::nonsingular());
            snapshot_probe.emplace(
                [stable_cache]() { return stable_cache->probe_linear_snapshot(); });
            problem_distribution.emplace(cache.vector_distribution_);
            problem_lane_identity = "pops.geometric-mg.boundary-newton.problem";
          });

      // std::optional owns the object storage, so no rank-local outer allocation can occur before
      // PreparedAffineLinearProblem enters its collective lane-duplication constructor.
      cache.linear_problem_.emplace(
          ExecutionCommunicator::world(), std::move(problem_lane_identity), cache.delta_,
          std::move(*operator_provider), std::move(*preconditioner),
          LinearOperatorProperties::general(), cache.footprint_, std::move(*nullspace_policy),
          std::move(*snapshot_probe), PreparedResourceFn{}, std::move(*problem_distribution));

      std::optional<PreparedKrylovMethod> workspace_method;
      std::optional<PreparedVectorDistribution> workspace_distribution;
      std::string workspace_lane_identity;
      GeometricMG::run_collective_materialization_stage_("boundary Newton GMRES arguments", [&] {
        cache.linear_method_ = gmres_krylov_method(restart);
        workspace_method.emplace(cache.linear_method_);
        workspace_distribution.emplace(cache.vector_distribution_);
        workspace_lane_identity = "pops.geometric-mg.boundary-newton.workspace";
      });
      cache.linear_workspace_.emplace(
          ExecutionCommunicator::world(), std::move(workspace_lane_identity), cache.delta_,
          std::move(*workspace_method), cache.footprint_, std::move(*workspace_distribution));
      return candidate;
    }

    BoundaryNewtonCache(const BoundaryNewtonCache&) = delete;
    BoundaryNewtonCache& operator=(const BoundaryNewtonCache&) = delete;
    BoundaryNewtonCache(BoundaryNewtonCache&&) = delete;
    BoundaryNewtonCache& operator=(BoundaryNewtonCache&&) = delete;

    bool compatible(const MultiFab& phi, const MultiFab& rhs, int restart) const {
      return restart_ == restart && operator_shape_bits_ == operator_shape_bits(*owner_) &&
             phi_layout_ == detail::layout_fingerprint(phi, vector_distribution_) &&
             rhs_layout_ == detail::layout_fingerprint(rhs, vector_distribution_);
    }

    std::size_t allocation_count() const {
      // Seven solve scratches plus the two fields owned by each prepared wrapper. The remaining
      // count is the method/restart-dependent persistent GMRES workspace.
      constexpr std::size_t fixed_fields = 7u + 2u + 2u;
      return fixed_fields + (linear_workspace_ ? linear_workspace_->allocation_count() : 0u);
    }

    [[nodiscard]] int restart() const noexcept { return restart_; }

    PreparedAffineLinearProblem& linear_problem() {
      if (!linear_problem_)
        throw std::logic_error("boundary Newton cache has no prepared affine problem");
      return *linear_problem_;
    }

    KrylovWorkspace& linear_workspace() {
      if (!linear_workspace_)
        throw std::logic_error("boundary Newton cache has no prepared Krylov workspace");
      return *linear_workspace_;
    }

    const PreparedKrylovMethod& linear_method() const {
      if (!linear_method_)
        throw std::logic_error("boundary Newton cache has no prepared Krylov method");
      return linear_method_;
    }

    void begin_solve() {
      auto& L = owner_->lev_[0];
      lincomb(published_snapshot_, Real(1), L.phi, Real(0), L.phi);
      lincomb(accepted_, Real(1), published_snapshot_, Real(0), published_snapshot_);
      lincomb(rhs_snapshot_, Real(1), L.rhs, Real(0), L.rhs);
    }

    void advance_linear_revision() {
      if (linear_revision_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("GeometricMG boundary Newton snapshot revision overflow");
      ++linear_revision_;
    }

    void restore_published() {
      auto& L = owner_->lev_[0];
      lincomb(L.phi, Real(1), published_snapshot_, Real(0), published_snapshot_);
      lincomb(L.rhs, Real(1), rhs_snapshot_, Real(0), rhs_snapshot_);
    }

    Real evaluate(MultiFab& iterate, MultiFab& output) {
      auto& owner = *owner_;
      auto& L = owner.lev_[0];
      owner.boundary_failure_.reset();
      poisson_residual(iterate, rhs_snapshot_, L.geom, owner.bc_, output, owner.mask_ptr(0),
                       owner.coef_ptr(0), owner.eps_ptr(0), owner.kappa_ptr(0), owner.eps_y_ptr(0),
                       owner.a_xy_ptr(0), owner.a_yx_ptr(0), &owner.boundary_kernel_,
                       &owner.boundary_context_, &L.boundary_view);
      if (owner.boundary_failure_.synchronize_across_ranks())
        return std::numeric_limits<Real>::quiet_NaN();
      return all_reduce_max(norm_inf(output));
    }

    OperatorEvaluationSnapshot probe_linear_snapshot() const {
      const auto& owner = *owner_;
      OperatorFingerprint resources = detail::fingerprint_seed();
      detail::fingerprint_mix(resources, static_cast<std::uint64_t>(owner.has_boundary_kernel_));
      detail::fingerprint_mix(resources,
                              static_cast<std::uint64_t>(owner.boundary_context_.point.iteration));
      detail::fingerprint_mix(resources, static_cast<std::uint64_t>(static_cast<std::int64_t>(
                                             owner.boundary_context_.point.clock_slot)));
      detail::fingerprint_mix(resources, static_cast<std::uint64_t>(static_cast<std::int64_t>(
                                             owner.boundary_context_.point.partition_slot)));
      detail::fingerprint_mix(resources, static_cast<std::uint64_t>(static_cast<std::int64_t>(
                                             owner.boundary_context_.point.stage_slot)));
      detail::fingerprint_mix(resources, static_cast<std::uint64_t>(static_cast<std::int64_t>(
                                             owner.boundary_context_.point.substep)));
      detail::fingerprint_mix(resources,
                              std::bit_cast<std::uint64_t>(owner.boundary_context_.point.time));
      detail::fingerprint_mix(resources,
                              std::bit_cast<std::uint64_t>(owner.boundary_context_.point.dt));
      return OperatorEvaluationSnapshot{
          {0x504f50534e455754ull, 0x4f4e4a41434f4249ull, 0x414e505245504152ull,
           0x45444b52594c4f56ull},
          linear_revision_,
          static_cast<std::int64_t>(owner.boundary_context_.point.step),
          0,
          1,
          std::bit_cast<std::uint64_t>(owner.boundary_context_.point.dt),
          std::bit_cast<std::uint64_t>(owner.boundary_context_.point.time),
          UINT64_C(1),
          linear_topology_,
          resources};
    }

    GeometricMG* owner_;
    PreparedVectorDistribution vector_distribution_;
    OperatorFingerprint phi_layout_{};
    OperatorFingerprint rhs_layout_{};
    std::uint32_t operator_shape_bits_ = 0;
    int restart_ = 0;
    std::uint64_t linear_revision_ = 0;
    MultiFab published_snapshot_;
    MultiFab accepted_;
    MultiFab trial_;
    MultiFab residual_;
    MultiFab trial_residual_;
    MultiFab delta_;
    MultiFab rhs_snapshot_;
    KrylovFootprint footprint_{};
    OperatorFingerprint linear_topology_{};
    std::optional<PreparedAffineLinearProblem> linear_problem_;
    std::optional<KrylovWorkspace> linear_workspace_;
    PreparedKrylovMethod linear_method_{};
  };

  /// Prepared callbacks must never retain a previous object's address. Copying or moving this slot
  /// therefore drops only the derived cache; the destination rebuilds it during explicit
  /// preparation. Once value is non-null, destruction/reset owns MPI_Comm_free and is consequently
  /// a collective lifecycle operation: all ranks that materialized the parent GeometricMG must
  /// copy, move, assign, clear, or destroy it in the same canonical order.
  struct BoundaryNewtonCacheSlot {
    BoundaryNewtonCacheSlot() = default;
    BoundaryNewtonCacheSlot(const BoundaryNewtonCacheSlot&) noexcept {}
    BoundaryNewtonCacheSlot& operator=(const BoundaryNewtonCacheSlot&) noexcept {
      value.reset();
      generation = 0;
      return *this;
    }
    BoundaryNewtonCacheSlot(BoundaryNewtonCacheSlot&& other) noexcept {
      other.value.reset();
      other.generation = 0;
    }
    BoundaryNewtonCacheSlot& operator=(BoundaryNewtonCacheSlot&& other) noexcept {
      if (this != &other) {
        value.reset();
        generation = 0;
        other.value.reset();
        other.generation = 0;
      }
      return *this;
    }

    std::unique_ptr<BoundaryNewtonCache> value;
    std::uint64_t generation = 0;
  };

  void prime_boundary_newton_cache_(BoundaryNewtonCache& cache) {
    OperatorEvaluationSnapshot snapshot{};
    run_collective_materialization_stage_("boundary Newton value snapshot", [&] {
      cache.begin_solve();
      cache.advance_linear_revision();
      snapshot = cache.probe_linear_snapshot();
    });
    cache.linear_problem().prepare(snapshot);
    cache.linear_workspace().bind(cache.linear_problem());
  }

  void prepare_boundary_newton_cache_(const FieldNewtonOptions& options) {
    require_collective_solve_inputs_("GeometricMG::prepare_boundary_newton");
    auto& slot = boundary_newton_cache_;
    auto& level = lev_[0];

    bool has_cache = static_cast<bool>(slot.value);
    if (all_reduce_min(has_cache ? 1L : 0L) != all_reduce_max(has_cache ? 1L : 0L))
      throw std::logic_error(
          "GeometricMG boundary Newton cache presence differs between communicator ranks");

    bool compatible = false;
    long compatibility_failure_local = 0;
    if (has_cache) {
      try {
        compatible = slot.value->compatible(level.phi, level.rhs, options.restart);
      } catch (...) {
        compatibility_failure_local = 1;
      }
    }
    if (all_reduce_max(compatibility_failure_local) != 0)
      throw std::runtime_error(
          "GeometricMG boundary Newton cache compatibility failed on at least one communicator "
          "rank");
    if (all_reduce_min(compatible ? 1L : 0L) != all_reduce_max(compatible ? 1L : 0L))
      throw std::logic_error(
          "GeometricMG boundary Newton cache compatibility differs between communicator ranks");

    if (compatible) {
      prime_boundary_newton_cache_(*slot.value);
      return;
    }

    const long generation_failure_local =
        slot.generation == std::numeric_limits<std::uint64_t>::max() ? 1L : 0L;
    if (all_reduce_max(generation_failure_local) != 0)
      throw std::overflow_error("GeometricMG boundary Newton cache generation overflow");
    auto replacement = BoundaryNewtonCache::materialize(*this, options.restart);
    prime_boundary_newton_cache_(*replacement);
    slot.value = std::move(replacement);
    ++slot.generation;
  }

  void require_prepared_boundary_newton_(const FieldNewtonOptions& options) {
    require_collective_newton_options_(options, "GeometricMG::solve_boundary_newton");
    const bool restart_matches =
        has_field_newton_options_ && options.restart == field_newton_options_.restart;
    if (all_reduce_max(restart_matches ? 0L : 1L) != 0)
      throw std::logic_error(
          "GeometricMG boundary Newton restart changed after preparation; call "
          "prepare_boundary_newton first");

    bool compatible = false;
    long compatibility_failure_local = 0;
    if (boundary_newton_cache_.value) {
      try {
        compatible =
            boundary_newton_cache_.value->compatible(lev_[0].phi, lev_[0].rhs, options.restart);
      } catch (...) {
        compatibility_failure_local = 1;
      }
    }
    if (all_reduce_max(compatibility_failure_local) != 0)
      throw std::runtime_error(
          "GeometricMG prepared boundary Newton layout check failed on at least one communicator "
          "rank");
    if (all_reduce_max(compatible ? 0L : 1L) != 0)
      throw std::logic_error(
          "GeometricMG boundary Newton cache is absent or stale; call prepare_boundary_newton "
          "after changing its layout or restart");
  }

  SolveReport solve_boundary_newton_cached(const FieldNewtonOptions& options) {
    auto& L = lev_[0];
    BoundaryNewtonCache& cache = *boundary_newton_cache_.value;
    std::optional<KrylovControls> linear_controls;
    run_collective_materialization_stage_("boundary Newton per-call Krylov controls", [&] {
      linear_controls.emplace(KrylovControls{cache.linear_method(), options.linear_tolerance,
                                             Real(0), options.linear_max_iterations});
    });
    run_collective_materialization_stage_("boundary Newton solve snapshot",
                                          [&] { cache.begin_solve(); });

    // A direct repeated solve must start from the same well-defined logical Newton point; retaining
    // the previous solve's last iteration would make the initial residual depend on call history.
    boundary_context_.point.iteration = 0;
    Real r0 = cache.evaluate(cache.accepted_, cache.residual_);
    if (!std::isfinite(static_cast<double>(r0))) {
      run_collective_materialization_stage_("boundary Newton failed-state restore",
                                            [&] { cache.restore_published(); });
      return make_solve_report_collectively_(
          0, std::numeric_limits<Real>::infinity(), SolveStatus::kInvalidEvaluation,
          SolveAction::kRejectAttempt, "GeometricMG::solve_boundary_newton");
    }
    const Real base = r0 > Real(0) ? r0 : Real(1);
    if (r0 == Real(0)) {
      run_collective_materialization_stage_("boundary Newton solved-state publication", [&] {
        lincomb(L.phi, Real(1), cache.accepted_, Real(0), cache.accepted_);
      });
      return make_solve_report_collectively_(0, r0 / base, SolveStatus::kSolved, SolveAction::kNone,
                                             "GeometricMG::solve_boundary_newton");
    }

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
      boundary_context_.point.iteration = iteration;
      PureFieldAlgebra::zero_valid(cache.delta_);
      // The prepared problem outlives one solve. Give every linearisation a cache-lifetime-unique
      // identity even when two calls use the same external logical time/stage.
      OperatorEvaluationSnapshot linear_snapshot{};
      run_collective_materialization_stage_("boundary Newton linear snapshot", [&] {
        cache.advance_linear_revision();
        linear_snapshot = cache.probe_linear_snapshot();
      });
      std::optional<SolveReport> linear;
      try {
        cache.linear_problem().prepare(linear_snapshot);
        cache.linear_workspace().bind(cache.linear_problem());
        linear.emplace(solve_prepared_affine(cache.linear_problem(), cache.linear_workspace(),
                                             cache.delta_, cache.residual_, *linear_controls));
      } catch (...) {
        run_collective_materialization_stage_("boundary Newton exceptional-state restore",
                                              [&] { cache.restore_published(); });
        throw;
      }
      if (!linear->solved()) {
        run_collective_materialization_stage_("boundary Newton rejected-state restore",
                                              [&] { cache.restore_published(); });
        linear->action = SolveAction::kRejectAttempt;
        return std::move(*linear);
      }

      Real step = Real(1);
      Real trial_norm = std::numeric_limits<Real>::infinity();
      bool accepted_step = false;
      while (step >= options.minimum_step) {
        lincomb(cache.trial_, Real(1), cache.accepted_, step, cache.delta_);
        trial_norm = cache.evaluate(cache.trial_, cache.trial_residual_);
        if (std::isfinite(static_cast<double>(trial_norm)) &&
            trial_norm <= (Real(1) - options.armijo * step) * r0) {
          accepted_step = true;
          break;
        }
        step *= Real(0.5);
      }
      if (!accepted_step) {
        run_collective_materialization_stage_("boundary Newton line-search restore",
                                              [&] { cache.restore_published(); });
        return make_solve_report_collectively_(
            iteration + 1, r0 / base, SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt,
            "GeometricMG::solve_boundary_newton");
      }
      lincomb(cache.accepted_, Real(1), cache.trial_, Real(0), cache.trial_);
      lincomb(cache.residual_, Real(1), cache.trial_residual_, Real(0), cache.trial_residual_);
      r0 = trial_norm;
      if (r0 <= options.tolerance * base) {
        run_collective_materialization_stage_("boundary Newton converged-state publication", [&] {
          lincomb(L.phi, Real(1), cache.accepted_, Real(0), cache.accepted_);
          lincomb(L.rhs, Real(1), cache.rhs_snapshot_, Real(0), cache.rhs_snapshot_);
        });
        return make_solve_report_collectively_(iteration + 1, r0 / base, SolveStatus::kSolved,
                                               SolveAction::kNone,
                                               "GeometricMG::solve_boundary_newton");
      }
    }
    run_collective_materialization_stage_("boundary Newton iteration-limit restore",
                                          [&] { cache.restore_published(); });
    return make_solve_report_collectively_(
        options.max_iterations, r0 / base, SolveStatus::kIterationLimit,
        SolveAction::kRejectAttempt, "GeometricMG::solve_boundary_newton");
  }

  const MultiFab* mask_ptr(int l) { return active_ ? &lev_[l].mask : nullptr; }
  const MultiFab* coef_ptr(int l) { return cut_cell_ ? &lev_[l].coef : nullptr; }
  const MultiFab* eps_ptr(int l) { return has_eps_ ? &lev_[l].eps : nullptr; }
  const MultiFab* kappa_ptr(int l) { return has_kappa_ ? &lev_[l].kappa : nullptr; }
  // eps_y absent => nullptr => isotropic operator (eps_y = eps_x) unchanged.
  const MultiFab* eps_y_ptr(int l) { return has_eps_y_ ? &lev_[l].eps_y : nullptr; }
  // cross terms absent => nullptr => DIAGONAL block (current path unchanged).
  const MultiFab* a_xy_ptr(int l) { return has_cross_ ? &lev_[l].a_xy : nullptr; }
  const MultiFab* a_yx_ptr(int l) { return has_cross_ ? &lev_[l].a_yx : nullptr; }
  const CompiledFieldBoundaryKernel* boundary_kernel_ptr(int l) {
    // Coarse MG levels solve a homogeneous correction.  The exact nonlinear closure/JVP is applied
    // on the materialized operator level; coarse levels remain an internal preconditioner.
    return has_boundary_kernel_ && l == 0 ? &boundary_kernel_ : nullptr;
  }
  const FieldBoundaryExecutionContext* boundary_context_ptr(int l) {
    return has_boundary_kernel_ && l == 0 ? &boundary_context_ : nullptr;
  }

  void trace_mark(const char* marker) noexcept {
    if (std::getenv("POPS_TRACE_SOLVE_FIELDS") == nullptr)
      return;
    (void)diagnostics_.try_record("elliptic.mg.trace", "GeometricMG", "trace", marker);
  }

  // BC used to fill the eps field ghosts: we keep the periodic but
  // replace every physical boundary (Dirichlet or outflow of phi) by a
  // zero-gradient extrapolation (eps_ghost = interior eps), which gives a
  // face permittivity = eps at the boundary (face on the domain contour).
  BCRec eps_bc() const {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc_.xlo);
    b.xhi = fo(bc_.xhi);
    b.ylo = fo(bc_.ylo);
    b.yhi = fo(bc_.yhi);
    return b;
  }

  bool is_replicated() const noexcept { return distribution_ == FieldDistribution::Replicated; }

  void add_level(const Geometry& g, const BoxArray& ba,
                 const DistributionMapping* distribution = nullptr) {
    if (distribution && distribution->size() != ba.size())
      throw std::invalid_argument("GeometricMG distribution size disagrees with BoxArray");
    DistributionMapping dm =
        distribution
            ? *distribution
            : (is_replicated() ? DistributionMapping(std::vector<int>(ba.size(), my_rank()))
                               : DistributionMapping(ba.size(), n_ranks()));
    lev_.push_back(MGLevel{g, ba, dm, MultiFab(ba, dm, 1, 1), MultiFab(ba, dm, 1, 0),
                           MultiFab(ba, dm, 1, 0), MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{},
                           MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{}, MultiFab{},
                           MultiFab{}, MultiFab{}});
    if (lev_.size() == 1) {
      lev_[0].zero_probe = MultiFab(ba, dm, 1, 1);
      lev_[0].zero_probe.set_val(Real(0));
    }
  }

  // FACTORIZATION (operator coefficient wiring, COMMON part): a scalar field
  // (eps, eps_y, kappa, ...) designated by a pointer-to-MGLevel-member MGLevel::*, either SAMPLED
  // PER LEVEL from an analytic function (sample_per_level), or COPIED onto the fine level
  // then RESTRICTED (average_down) to the coarse (restrict_and_fill). Both preserve EXACTLY
  // the original inline bodies, including the DIFFERENCES between coefficients:
  //   - nghost: 1 for eps/eps_y (face neighbors read), 0 for kappa (diagonal, read at (i,j) only);
  //   - do_fill: eps/eps_y fill their ghosts (fill_ghosts); kappa DOES NOT FILL THEM
  //     (0 ghost, HISTORICAL omission kept unchanged -- NO fill_ghosts added here).

  // Host PER-LEVEL sampling of a field from fn (std::function not device-callable): allocates
  // MultiFab(L.ba, L.dm, 1, nghost) at each level, writes f(x_cell, y_cell) at the center, then ghosts
  // (fill_ghosts with ebc) ONLY if do_fill. Body extracted word-for-word from set_epsilon(fn) etc.
  void sample_per_level(MultiFab MGLevel::* field, const ScalarFieldProvider2D& fn, int nghost,
                        bool do_fill, const BCRec& ebc) {
    for (auto& L : lev_) {
      MultiFab& F = L.*field;
      F = MultiFab(L.ba, L.dm, 1, nghost);
      const Geometry& g = L.geom;
      for (int li = 0; li < F.local_size(); ++li) {
        Array4 e = F.fab(li).array();
        const Box2D b = F.box(li);
        // host initialization (std::function not device-callable)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            e(i, j) = fn(g.x_cell(i), g.y_cell(j));
      }
      if (do_fill)
        fill_ghosts(F, g.domain, ebc);
    }
  }

  // Copy comp 0 of the fine field (already discretized) onto the fine level then RESTRICTION (average_down,
  // 2x2 average) to the coarse: allocates MultiFab(L.ba, L.dm, 1, nghost) at each level, ghosts
  // (fill_ghosts with ebc) of the fine level THEN of each coarse level after its average, ONLY
  // if do_fill. Body extracted word-for-word from set_epsilon(const MultiFab&) / set_reaction(const MultiFab&).
  void restrict_and_fill(MultiFab MGLevel::* field, const MultiFab& fine, int nghost, bool do_fill,
                         const BCRec& ebc) {
    for (auto& L : lev_)
      L.*field = MultiFab(L.ba, L.dm, 1, nghost);
    for (int li = 0; li < (lev_[0].*field).local_size(); ++li) {
      Array4 e = (lev_[0].*field).fab(li).array();
      const ConstArray4 s = fine.fab(li).const_array();
      const Box2D b = (lev_[0].*field).box(li);
      for_each_cell(b, detail::CopyComp0Kernel{e, s});
    }
    if (do_fill)
      fill_ghosts(lev_[0].*field, lev_[0].geom.domain, ebc);
    for (int l = 1; l < num_levels(); ++l) {
      average_down(lev_[l - 1].*field, lev_[l].*field, 2);
      if (do_fill)
        fill_ghosts(lev_[l].*field, lev_[l].geom.domain, ebc);
    }
  }

  static void gs_smooth_on_lane(MultiFab& phi, const MultiFab& rhs, const Geometry& geometry,
                                const BCRec& boundary, int sweeps, const MultiFab* mask,
                                const MultiFab* coef, const MultiFab* eps, const MultiFab* kappa,
                                const MultiFab* eps_y,
                                const CompiledFieldBoundaryKernel* boundary_kernel,
                                const FieldBoundaryExecutionContext* boundary_context,
                                MultiFab* boundary_view, const ExecutionLane& lane) {
    const auto prepare_view = [&]() -> MultiFab& {
      if (boundary_kernel == nullptr) {
        fill_ghosts(phi, geometry.domain, boundary, lane);
        return phi;
      }
      if (boundary_context == nullptr || boundary_view == nullptr)
        throw std::runtime_error(
            "compiled field boundary smoother is missing its context or persistent work view");
      copy_field_boundary_band(phi, *boundary_view);
      fill_ghosts(*boundary_view, geometry.domain, boundary, lane);
      for (int face = 0; face < 4; ++face)
        boundary_kernel->prepare_residual_view(face, phi, *boundary_view, geometry,
                                               *boundary_context);
      return *boundary_view;
    };
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      MultiFab& red_view = prepare_view();
      detail::gs_color(phi, rhs, geometry, 0, mask, coef, eps, kappa, eps_y, &red_view);
      MultiFab& black_view = prepare_view();
      detail::gs_color(phi, rhs, geometry, 1, mask, coef, eps, kappa, eps_y, &black_view);
    }
  }

  static void poisson_residual_on_lane(MultiFab& phi, const MultiFab& rhs, const Geometry& geometry,
                                       const BCRec& boundary, MultiFab& residual,
                                       const MultiFab* mask, const MultiFab* coef,
                                       const MultiFab* eps, const MultiFab* kappa,
                                       const MultiFab* eps_y, const MultiFab* a_xy,
                                       const MultiFab* a_yx,
                                       const CompiledFieldBoundaryKernel* boundary_kernel,
                                       const FieldBoundaryExecutionContext* boundary_context,
                                       MultiFab* boundary_view, const ExecutionLane& lane) {
    MultiFab* operator_view = &phi;
    if (boundary_kernel == nullptr) {
      fill_ghosts(phi, geometry.domain, boundary, lane);
    } else {
      if (boundary_context == nullptr || boundary_view == nullptr)
        throw std::runtime_error(
            "compiled field boundary residual is missing its context or persistent operator view");
      copy_field_valid(phi, *boundary_view);
      fill_ghosts(*boundary_view, geometry.domain, boundary, lane);
      for (int face = 0; face < 4; ++face)
        boundary_kernel->prepare_residual_view(face, phi, *boundary_view, geometry,
                                               *boundary_context);
      operator_view = boundary_view;
    }
    const Real idx2 = Real(1) / (geometry.dx() * geometry.dx());
    const Real idy2 = Real(1) / (geometry.dy() * geometry.dy());
    const Real idx = Real(1) / geometry.dx();
    const Real idy = Real(1) / geometry.dy();
    for (int local = 0; local < operator_view->local_size(); ++local) {
      const ConstArray4 p = operator_view->fab(local).const_array();
      const ConstArray4 f = rhs.fab(local).const_array();
      Array4 r = residual.fab(local).array();
      const Box2D valid = residual.box(local);
      const bool has_mask = mask != nullptr;
      const ConstArray4 mask_values = has_mask ? mask->fab(local).const_array() : ConstArray4{};
      const bool has_coef = coef != nullptr;
      const ConstArray4 coef_values = has_coef ? coef->fab(local).const_array() : ConstArray4{};
      const bool has_eps = eps != nullptr;
      const ConstArray4 eps_values = has_eps ? eps->fab(local).const_array() : ConstArray4{};
      const ConstArray4 eps_y_values =
          has_eps && eps_y != nullptr ? eps_y->fab(local).const_array() : eps_values;
      const bool has_kappa = kappa != nullptr;
      const ConstArray4 kappa_values = has_kappa ? kappa->fab(local).const_array() : ConstArray4{};
      const bool has_a_xy = a_xy != nullptr;
      const bool has_a_yx = a_yx != nullptr;
      const ConstArray4 a_xy_values = has_a_xy ? a_xy->fab(local).const_array() : ConstArray4{};
      const ConstArray4 a_yx_values = has_a_yx ? a_yx->fab(local).const_array() : ConstArray4{};
      for_each_cell(valid, detail::PoissonResidualKernel{p,           f,
                                                         r,           idx2,
                                                         idy2,        idx,
                                                         idy,         has_mask,
                                                         mask_values, has_coef,
                                                         coef_values, has_eps,
                                                         eps_values,  eps_y_values,
                                                         has_kappa,   kappa_values,
                                                         has_a_xy,    has_a_yx,
                                                         a_xy_values, a_yx_values});
    }
    if (boundary_kernel != nullptr)
      for (int face = 0; face < 4; ++face)
        boundary_kernel->add_residual(face, phi, residual, geometry, *boundary_context);
  }

  void vcycle_rec(int l, const BCRec& bc, const ExecutionLane* lane = nullptr) {
    MGLevel& L = lev_[l];
    BCRec level_bc = bc;
    level_bc.dx = L.geom.dx();
    level_bc.dy = L.geom.dy();
    const MultiFab* mk = mask_ptr(l);
    const MultiFab* ck = coef_ptr(l);
    const MultiFab* ep = eps_ptr(l);
    const MultiFab* kp = kappa_ptr(l);
    const MultiFab* ey = eps_y_ptr(l);  // nullptr => isotropic (eps_y = eps_x)
    const MultiFab* axy = a_xy_ptr(l);  // nullptr => diagonal block (no cross flux)
    const MultiFab* ayx = a_yx_ptr(l);
    // NB: gs_smooth stays 5-POINT (diagonal block). The cross terms are EXPLICIT: only the
    // residual (poisson_residual) carries them. The GS smoother touches only the diagonal -> its diag stays
    // dominant (kappa>=0, eps>0); the cross coupling is relegated to the residual, per the header
    // convention. For symmetric-positive-definite A the V-cycle stays contractive; for strongly non-symmetric
    // A, it may diverge (cf. set_cross_terms, reported observation).
    if (l == 0)
      trace_mark("vcycle_rec(0): before gs_smooth(nu1) [first GS kernel]");
    const auto smooth = [&](int sweeps) {
      if (lane != nullptr)
        gs_smooth_on_lane(L.phi, L.rhs, L.geom, level_bc, sweeps, mk, ck, ep, kp, ey,
                          boundary_kernel_ptr(l), boundary_context_ptr(l),
                          boundary_kernel_ptr(l) ? &L.boundary_view : nullptr, *lane);
      else
        gs_smooth(L.phi, L.rhs, L.geom, level_bc, sweeps, mk, ck, ep, kp, ey,
                  boundary_kernel_ptr(l), boundary_context_ptr(l),
                  boundary_kernel_ptr(l) ? &L.boundary_view : nullptr);
    };
    smooth(nu1_);
    if (l == 0)
      trace_mark("vcycle_rec(0): after gs_smooth(nu1)");

    if (l + 1 == static_cast<int>(lev_.size())) {
      // BOTTOM solve = long Gauss-Seidel smoothing on the coarsest grid. Self-time it (chrono only,
      // no profiler dependency here) and accumulate into the per-solve last_bottom_seconds_ (reset at
      // the top of solve()): the System reads it back to attribute the coarsest-grid cost (Spec 5
      // sec.13.11.1, ADC-479). Host serial / per-rank; the device-fence for an exact GPU bottom time is
      // deferred (counter stays an honest host-side measurement).
      const auto bottom_t0 = std::chrono::steady_clock::now();
      smooth(nbottom_);  // bottom solve
      const auto bottom_t1 = std::chrono::steady_clock::now();
      last_bottom_seconds_ += std::chrono::duration<double>(bottom_t1 - bottom_t0).count();
      if (mk)
        zero_conductor(L.phi, L.mask);
      return;
    }

    if (lane != nullptr)
      poisson_residual_on_lane(L.phi, L.rhs, L.geom, level_bc, L.res, mk, ck, ep, kp, ey, axy, ayx,
                               boundary_kernel_ptr(l), boundary_context_ptr(l),
                               boundary_kernel_ptr(l) ? &L.boundary_view : nullptr, *lane);
    else
      poisson_residual(L.phi, L.rhs, L.geom, level_bc, L.res, mk, ck, ep, kp, ey, axy, ayx,
                       boundary_kernel_ptr(l), boundary_context_ptr(l),
                       boundary_kernel_ptr(l) ? &L.boundary_view : nullptr);
    if (l == 0)
      trace_mark("vcycle_rec(0): after poisson_residual");
    MGLevel& C = lev_[l + 1];
    if (lane != nullptr)
      average_down(L.res, C.rhs, 2, L.cfine, *lane);
    else
      average_down(L.res, C.rhs, 2, L.cfine);  // residual restriction (cfine buffer reused)
    if (l == 0)
      trace_mark("vcycle_rec(0): after average_down");
    C.phi.set_val(0.0);
    vcycle_rec(l + 1, homogeneous(level_bc), lane);
    if (l == 0)
      trace_mark("vcycle_rec(0): after coarse recursion");

    if (lane != nullptr)
      interpolate(C.phi, L.corr, 2, L.cfine, *lane);
    else
      interpolate(C.phi, L.corr, 2,
                  L.cfine);  // correction prolongation (corr/cfine buffers reused)
    if (l == 0)
      trace_mark("vcycle_rec(0): after interpolate");
    saxpy(L.phi, Real(1), L.corr);
    if (l == 0)
      trace_mark("vcycle_rec(0): after saxpy");
    if (mk)
      zero_conductor(L.phi, L.mask);  // re-pin the conductor
    smooth(nu2_);
    if (l == 0)
      trace_mark("vcycle_rec(0): after gs_smooth(nu2)");
  }

  static std::string construction_options_contract(int min_coarse, int nu1, int nu2, int nbottom,
                                                   bool cut_cell,
                                                   const LevelSetProvider2D& levelset,
                                                   Real cut_theta_min, int coarse_threshold) {
    ExactContractBuilder contract;
    contract.text("pops.elliptic.geometric-mg.options")
        .scalar(std::uint32_t{1})
        .scalar(min_coarse)
        .scalar(nu1)
        .scalar(nu2)
        .scalar(nbottom)
        .scalar(cut_cell)
        .optional_collective_contract(levelset)
        .scalar(cut_theta_min)
        .scalar(coarse_threshold);
    return std::move(contract).release();
  }

  BCRec bc_;
  ActiveRegionProvider2D active_;
  int min_coarse_, nu1_, nu2_, nbottom_;
  int coarse_threshold_ =
      kMGDefaultCoarseThreshold;              ///< ADC-644: total-cell coarsening ceiling (0 = off).
  Real cut_theta_min_ = kEbCutFractionFloor;  ///< ADC-615: cut-fraction clamp (default 1e-3).
  FieldDistribution distribution_ = FieldDistribution::Distributed;
  bool cut_cell_ = false;
  bool has_eps_ = false;
  bool has_eps_y_ = false;
  bool has_kappa_ = false;
  bool has_cross_ = false;  // off-diagonal Axy/Ayx coefficients (FULL tensor) active
  std::string eps_provider_contract_;
  std::string eps_y_provider_contract_;
  std::string kappa_provider_contract_;
  std::string a_xy_provider_contract_;
  std::string a_yx_provider_contract_;
  bool has_boundary_kernel_ = false;
  CompiledFieldBoundaryKernel boundary_kernel_{};
  FieldBoundaryFailure boundary_failure_{};
  FieldNewtonOptions field_newton_options_{};
  bool has_field_newton_options_ = false;
  SolveReport last_solve_report_{};
  FieldBoundaryExecutionContext boundary_context_{};
  Real abs_tol_ =
      kMGDefaultAbsTol;  // absolute floor of the no-argument solve() (0 = relative criterion only)
  // PER-SOLVE PROFILING STATS (read back at the System field_solve seam, ADC-479 criteria 42/43).
  // last_cycles_/last_residual_ are set by solve(); last_bottom_seconds_ is reset at the top of solve()
  // and accumulated by vcycle_rec's bottom branch. 0 until the first solve (no cycle recorded yet).
  int last_cycles_ = 0;
  Real last_residual_ = Real(0);
  double last_bottom_seconds_ = 0.0;
  RuntimeDiagnosticsReport diagnostics_ =
      make_runtime_diagnostics_report("pops.numerics.elliptic.geometric_mg");
  LevelSetProvider2D levelset_;
  std::vector<MGLevel> lev_;
  std::vector<char, comm_allocator<char>> replica_validation_data_;
  BoundaryNewtonCacheSlot boundary_newton_cache_;
  EllipticOperatorContract prepared_operator_contract_;
};

inline std::uint64_t GeometricMG::boundary_newton_cache_generation() const {
  return boundary_newton_cache_.generation;
}

inline std::size_t GeometricMG::boundary_newton_cache_allocation_count() const {
  return boundary_newton_cache_.value ? boundary_newton_cache_.value->allocation_count() : 0u;
}

}  // namespace pops
