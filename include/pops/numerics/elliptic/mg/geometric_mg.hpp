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
/// - replicated: level replicated on all ranks (per-fab V-cycle without communication), as expected by
///   the AMR coupler; in serial bit-for-bit identical to round-robin;
/// - cut_cell: order-2 Shortley-Weller weights at the embedded boundary (vs staircase); cut_cell=false
///   bit-identical to the historical stencil;
/// - device kernels are NAMED FUNCTORS (recipe #93/#64): extended lambda forbidden cross-TU under nvcc.

#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/numerics/elliptic/eb/cut_fraction.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/parallel/comm.hpp>

#include <bit>
#include <chrono>   // last_bottom_seconds(): self-time the coarsest (bottom) GS solve (Spec 5, ADC-479)
#include <cmath>
#include <cstdlib>  // getenv
#include <functional>
#include <limits>
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
  // replicated: if true, each level (mono-box covering the domain) is REPLICATED on
  // all ranks (dmap = my_rank() everywhere) instead of the default round-robin. Each rank
  // then solves the SAME coarse Poisson redundantly, WITHOUT communication (per-fab V-cycle,
  // fill_boundary on a box covering the domain is local, and current_residual reduced by
  // norm_inf = all_reduce_MAX, idempotent under replication). This is what the AMR coupler
  // expects (level 0 replicated). In serial my_rank()=0 -> bit-for-bit identical to round-robin.
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
              std::function<bool(Real, Real)> active = {}, bool replicated = false,
              int min_coarse = kMGDefaultMinCoarse, int nu1 = kMGDefaultPreSmooth,
              int nu2 = kMGDefaultPostSmooth, int nbottom = kMGDefaultBottomSweeps,
              bool cut_cell = false, std::function<Real(Real, Real)> levelset = {},
              Real cut_theta_min = kEbCutFractionFloor,
              int coarse_threshold = kMGDefaultCoarseThreshold)
      : bc_(bc),
        active_(std::move(active)),
        nu1_(nu1),
        nu2_(nu2),
        nbottom_(nbottom),
        coarse_threshold_(
            coarse_threshold),          // ADC-644: total-cell coarsening ceiling (0 = disabled)
        cut_theta_min_(cut_theta_min),  // ADC-615: cut-fraction clamp shared with the EB transport
        replicated_(replicated),
        cut_cell_(cut_cell),
        levelset_(std::move(levelset)) {
    bc_.dx = geom.dx();
    bc_.dy = geom.dy();
    if (cut_cell_ && levelset_ && !active_)
      active_ = [ls = levelset_](Real x, Real y) { return ls(x, y) < Real(0); };
    add_level(geom, ba);
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
      add_level(gc, coarsen(lev_.back().ba, 2));
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
  }

  MultiFab& phi() { return lev_[0].phi; }
  MultiFab& rhs() { return lev_[0].rhs; }
  const Geometry& geom() const { return lev_[0].geom; }
  int num_levels() const { return static_cast<int>(lev_.size()); }

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
  void set_epsilon(std::function<Real(Real, Real)> eps_fn) {
    // 1 ghost (box-boundary neighbors read), ghosts filled (do_fill).
    sample_per_level(&MGLevel::eps, eps_fn, 1, true, eps_bc());
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
  void set_epsilon_anisotropic(std::function<Real(Real, Real)> eps_x_fn,
                               std::function<Real(Real, Real)> eps_y_fn) {
    set_epsilon(std::move(eps_x_fn));  // x faces: reuse the isotropic eps wiring
    // y faces: second eps_y field, same convention (1 ghost, ghosts filled).
    sample_per_level(&MGLevel::eps_y, eps_y_fn, 1, true, eps_bc());
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
  void set_reaction(std::function<Real(Real, Real)> kappa_fn) {
    // kappa: DIAGONAL, read at (i,j) only -> 0 ghost and do_fill=false (NO fill_ghosts, historical).
    // ebc is then unused (BCRec{} never read).
    sample_per_level(&MGLevel::kappa, kappa_fn, 0, false, BCRec{});
    has_kappa_ = true;
  }

  // Overload: ALREADY-discretized kappa field (1-component MultiFab, fine grid), copied onto the
  // fine level then RESTRICTED (average_down) to the coarse. Entry point for System wiring
  // (a per-cell kappa field).
  void set_reaction(const MultiFab& kappa_fine) {
    // kappa: DIAGONAL -> 0 ghost and do_fill=false (NO fill_ghosts, neither fine nor coarse, historical).
    restrict_and_fill(&MGLevel::kappa, kappa_fine, 0, false, BCRec{});
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
  void set_cross_terms(std::function<Real(Real, Real)> a_xy_fn,
                       std::function<Real(Real, Real)> a_yx_fn) {
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
    has_cross_ = true;
  }

  void vcycle() { vcycle_rec(0, bc_); }
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
    if (!(rel_tol > Real(0)) || !std::isfinite(static_cast<double>(rel_tol)) || max_cycles < 1 ||
        abs_tol < Real(0) || !std::isfinite(static_cast<double>(abs_tol)))
      throw std::invalid_argument(
          "GeometricMG::solve requires finite rel_tol > 0, max_cycles >= 1 and "
          "finite abs_tol >= 0");
    last_solve_report_ = {};
    const bool fallible_linear_boundary =
        has_boundary_kernel_ && !boundary_kernel_.observes_iteration;
    if (fallible_linear_boundary)
      boundary_failure_.reset();
    auto finish = [&](int cycles) {
      if (fallible_linear_boundary && boundary_failure_.synchronize_across_ranks())
        throw std::runtime_error("field boundary evaluation failed at face " +
                                 std::to_string(boundary_failure_.face) + " cell (" +
                                 std::to_string(boundary_failure_.i) + "," +
                                 std::to_string(boundary_failure_.j) + ")");
      return cycles;
    };
    auto invalid_evaluation = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      last_solve_report_.iters = cycles;
      last_solve_report_.rel_residual = std::numeric_limits<Real>::infinity();
      last_solve_report_.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt);
      return finish(cycles);
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
      last_solve_report_.iters = 0;
      last_solve_report_.rel_residual = r0 / report_denom;
      last_solve_report_.mark_solved();
      return finish(0);
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
        last_solve_report_.iters = c;
        last_solve_report_.rel_residual = r / report_denom;
        last_solve_report_.mark_solved();
        return finish(c);
      }
    }
    last_cycles_ = max_cycles;
    last_residual_ = current_residual();
    if (!std::isfinite(static_cast<double>(last_residual_)))
      return invalid_evaluation(max_cycles, last_residual_);
    last_solve_report_.iters = max_cycles;
    last_solve_report_.rel_residual = last_residual_ / report_denom;
    last_solve_report_.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt);
    return finish(max_cycles);
  }

  // EllipticSolver concept interface: solve() with no argument (default
  // tolerance) and residual() (alias of current_residual). Lets couplers
  // depend on the concept, not on GeometricMG directly. Propagates abs_tol_ (absolute
  // floor, default 0 -> historical relative criterion unchanged) to the mixed criterion.
  void solve() {
    if (has_boundary_kernel_ && boundary_kernel_.observes_iteration) {
      if (!has_field_newton_options_)
        throw std::runtime_error(
            "iterate-dependent field boundary requires an installed nonlinear outer solver");
      last_solve_report_ = solve_boundary_newton(field_newton_options_);
      if (!last_solve_report_.solved())
        throw std::runtime_error(std::string("field Newton solve failed: ") +
                                 last_solve_report_.status_name());
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
    if (!(rel_tol > Real(0)) || !std::isfinite(static_cast<double>(rel_tol)) || max_cycles < 1 ||
        abs_tol < Real(0) || !std::isfinite(static_cast<double>(abs_tol)))
      throw std::invalid_argument(
          "GeometricMG::solve_robust requires finite rel_tol > 0, max_cycles >= 1 and "
          "finite abs_tol >= 0");
    last_solve_report_ = {};
    last_bottom_seconds_ = 0.0;
    auto invalid_evaluation = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      last_solve_report_.iters = cycles;
      last_solve_report_.rel_residual = std::numeric_limits<Real>::infinity();
      last_solve_report_.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt);
      return cycles;
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
      last_solve_report_.iters = cycles;
      last_solve_report_.rel_residual = residual / report_denom;
      last_solve_report_.mark_solved();
      return cycles;
    };
    auto iteration_limit = [&](int cycles, Real residual) {
      last_cycles_ = cycles;
      last_residual_ = residual;
      last_solve_report_.iters = cycles;
      last_solve_report_.rel_residual = residual / report_denom;
      last_solve_report_.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt);
      return cycles;
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
    kernel.validate();
    boundary_kernel_ = kernel;
    boundary_context_ = context;
    boundary_context_.failure = &boundary_failure_;
    has_boundary_kernel_ = true;
    lev_[0].boundary_view = MultiFab(lev_[0].ba, lev_[0].dm, 1, 1);
    lev_[0].direction_view = MultiFab(lev_[0].ba, lev_[0].dm, 1, 1);
  }

  void clear_boundary_kernel() {
    has_boundary_kernel_ = false;
    boundary_kernel_ = {};
    boundary_context_ = {};
  }

  void set_boundary_context(const FieldBoundaryExecutionContext& context) {
    if (!has_boundary_kernel_)
      throw std::runtime_error("GeometricMG boundary context installed without a compiled kernel");
    boundary_context_ = context;
    boundary_context_.failure = &boundary_failure_;
  }

  void set_field_newton_options(const FieldNewtonOptions& options) {
    validate_field_newton_options(options);
    field_newton_options_ = options;
    has_field_newton_options_ = true;
  }

  const SolveReport& last_solve_report() const { return last_solve_report_; }

  SolveReport solve_boundary_newton(const FieldNewtonOptions& options) {
    if (!has_boundary_kernel_ || !boundary_kernel_.observes_iteration ||
        boundary_kernel_.jvp == nullptr)
      return SolveReport::capability_failure();
    set_field_newton_options(options);
    auto& L = lev_[0];
    MultiFab published_snapshot(L.ba, L.dm, 1, L.phi.n_grow());
    MultiFab accepted(L.ba, L.dm, 1, L.phi.n_grow());
    MultiFab trial(L.ba, L.dm, 1, L.phi.n_grow());
    MultiFab residual(L.ba, L.dm, 1, 0);
    MultiFab trial_residual(L.ba, L.dm, 1, 0);
    MultiFab delta(L.ba, L.dm, 1, L.phi.n_grow());
    MultiFab rhs_snapshot(L.ba, L.dm, 1, L.rhs.n_grow());
    lincomb(published_snapshot, Real(1), L.phi, Real(0), L.phi);
    lincomb(accepted, Real(1), published_snapshot, Real(0), published_snapshot);
    lincomb(rhs_snapshot, Real(1), L.rhs, Real(0), L.rhs);

    auto restore_published = [&]() {
      lincomb(L.phi, Real(1), published_snapshot, Real(0), published_snapshot);
      lincomb(L.rhs, Real(1), rhs_snapshot, Real(0), rhs_snapshot);
    };

    auto evaluate = [&](MultiFab& iterate, MultiFab& output) -> Real {
      boundary_failure_.reset();
      poisson_residual(iterate, rhs_snapshot, L.geom, bc_, output, mask_ptr(0), coef_ptr(0),
                       eps_ptr(0), kappa_ptr(0), eps_y_ptr(0), a_xy_ptr(0), a_yx_ptr(0),
                       &boundary_kernel_, &boundary_context_, &L.boundary_view);
      if (boundary_failure_.synchronize_across_ranks())
        return std::numeric_limits<Real>::quiet_NaN();
      return all_reduce_max(norm_inf(output));
    };

    SolveReport report;
    Real r0 = evaluate(accepted, residual);
    if (!std::isfinite(static_cast<double>(r0))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt);
      restore_published();
      return report;
    }
    const Real base = r0 > Real(0) ? r0 : Real(1);
    if (r0 == Real(0)) {
      report.rel_residual = r0 / base;
      report.mark_solved();
      lincomb(L.phi, Real(1), accepted, Real(0), accepted);
      return report;
    }

    ApplyFn jacobian = [&](MultiFab& out, const MultiFab& in) {
      boundary_failure_.reset();
      apply_jvp(accepted, in, out);
      if (boundary_failure_.synchronize_across_ranks())
        out.set_val(std::numeric_limits<Real>::quiet_NaN());
    };
    ApplyFn preconditioner = [&](MultiFab& out, const MultiFab& in) {
      struct RestorePreconditionerState {
        bool& boundary_enabled;
        bool saved_boundary_enabled;
        MultiFab& phi;
        MultiFab& rhs;
        const MultiFab& saved_phi;
        const MultiFab& saved_rhs;
        ~RestorePreconditionerState() {
          lincomb(phi, Real(1), saved_phi, Real(0), saved_phi);
          lincomb(rhs, Real(1), saved_rhs, Real(0), saved_rhs);
          boundary_enabled = saved_boundary_enabled;
        }
      } restore{has_boundary_kernel_, has_boundary_kernel_, L.phi, L.rhs, accepted,
                rhs_snapshot};
      has_boundary_kernel_ = false;
      L.phi.set_val(Real(0));
      lincomb(L.rhs, Real(1), in, Real(0), in);
      vcycle();
      lincomb(out, Real(1), L.phi, Real(0), L.phi);
    };
    PreparedResourceFn prepare_linear_preconditioner = [&]() {
      delta.set_val(Real(0));
      preconditioner(delta, delta);
      delta.set_val(Real(0));
    };
    KrylovFootprint footprint{1, delta.n_grow(), options.restart, true};
    std::uint64_t linear_revision = 0;
    OperatorFingerprint linear_topology = detail::layout_fingerprint(delta);
    detail::fingerprint_geometry(linear_topology, L.geom);
    detail::fingerprint_boundary(linear_topology, bc_);
    auto probe_linear_snapshot = [&]() {
      OperatorFingerprint resources = detail::fingerprint_seed();
      detail::fingerprint_mix(resources, static_cast<std::uint64_t>(has_boundary_kernel_));
      detail::fingerprint_mix(
          resources, static_cast<std::uint64_t>(boundary_context_.point.iteration));
      detail::fingerprint_mix(resources,
                              std::bit_cast<std::uint64_t>(boundary_context_.point.time));
      detail::fingerprint_mix(resources,
                              std::bit_cast<std::uint64_t>(boundary_context_.point.dt));
      return OperatorEvaluationSnapshot{
          {0x504f50534e455754ull, 0x4f4e4a41434f4249ull,
           0x414e505245504152ull, 0x45444b52594c4f56ull},
          linear_revision,
          static_cast<std::int64_t>(boundary_context_.point.step),
          static_cast<std::int64_t>(boundary_context_.point.stage_slot),
          1,
          std::bit_cast<std::uint64_t>(boundary_context_.point.dt),
          std::bit_cast<std::uint64_t>(boundary_context_.point.time),
          UINT64_C(1),
          linear_topology,
          resources};
    };
    PreparedAffineLinearProblem linear_problem(
        delta, jacobian,
        PreparedLinearPreconditioner(preconditioner, prepare_linear_preconditioner),
        LinearOperatorProperties::general(), footprint,
        probe_linear_snapshot);
    KrylovWorkspace linear_workspace(delta, KrylovMethod::kGmres, footprint);
    const KrylovControls linear_controls{KrylovMethod::kGmres,
                                         options.linear_tolerance,
                                         Real(0),
                                         options.linear_max_iterations,
                                         options.restart,
                                         Real(1)};

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
      boundary_context_.point.iteration = iteration;
      delta.set_val(Real(0));
      linear_revision = static_cast<std::uint64_t>(iteration + 1);
      const OperatorEvaluationSnapshot linear_snapshot = probe_linear_snapshot();
      SolveReport linear;
      try {
        linear_problem.prepare(linear_snapshot);
        linear_workspace.bind(linear_problem);
        linear = solve_prepared_affine(
            linear_problem, linear_workspace, delta, residual, linear_controls);
      } catch (...) {
        restore_published();
        throw;
      }
      if (!linear.solved()) {
        report = linear;
        report.action = SolveAction::kRejectAttempt;
        restore_published();
        return report;
      }

      Real step = Real(1);
      Real trial_norm = std::numeric_limits<Real>::infinity();
      bool accepted_step = false;
      while (step >= options.minimum_step) {
        lincomb(trial, Real(1), accepted, step, delta);
        trial_norm = evaluate(trial, trial_residual);
        if (std::isfinite(static_cast<double>(trial_norm)) &&
            trial_norm <= (Real(1) - options.armijo * step) * r0) {
          accepted_step = true;
          break;
        }
        step *= Real(0.5);
      }
      if (!accepted_step) {
        report.iters = iteration + 1;
        report.rel_residual = r0 / base;
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kRejectAttempt);
        restore_published();
        return report;
      }
      lincomb(accepted, Real(1), trial, Real(0), trial);
      lincomb(residual, Real(1), trial_residual, Real(0), trial_residual);
      r0 = trial_norm;
      report.iters = iteration + 1;
      report.rel_residual = r0 / base;
      if (r0 <= options.tolerance * base) {
        report.mark_solved();
        lincomb(L.phi, Real(1), accepted, Real(0), accepted);
        lincomb(L.rhs, Real(1), rhs_snapshot, Real(0), rhs_snapshot);
        return report;
      }
    }
    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt);
    restore_published();
    return report;
  }

  void apply_jvp(const MultiFab& iterate, const MultiFab& direction, MultiFab& output) {
    apply_laplacian_jvp(iterate, direction, lev_[0].geom, bc_, output, boundary_kernel_ptr(0),
                        boundary_context_ptr(0),
                        has_boundary_kernel_ ? &lev_[0].direction_view : nullptr, coef_ptr(0),
                        eps_ptr(0), kappa_ptr(0), eps_y_ptr(0), a_xy_ptr(0), a_yx_ptr(0));
  }

 private:
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

  void trace_mark(const char* marker) {
    if (std::getenv("POPS_TRACE_SOLVE_FIELDS") == nullptr)
      return;
    diagnostics_.record("elliptic.mg.trace", "GeometricMG", "trace", marker);
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

  void add_level(const Geometry& g, const BoxArray& ba) {
    DistributionMapping dm = replicated_
                                 ? DistributionMapping(std::vector<int>(ba.size(), my_rank()))
                                 : DistributionMapping(ba.size(), n_ranks());
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
  void sample_per_level(MultiFab MGLevel::* field, const std::function<Real(Real, Real)>& fn,
                        int nghost, bool do_fill, const BCRec& ebc) {
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

  void vcycle_rec(int l, const BCRec& bc) {
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
    gs_smooth(L.phi, L.rhs, L.geom, level_bc, nu1_, mk, ck, ep, kp, ey, boundary_kernel_ptr(l),
              boundary_context_ptr(l), boundary_kernel_ptr(l) ? &L.boundary_view : nullptr);
    if (l == 0)
      trace_mark("vcycle_rec(0): after gs_smooth(nu1)");

    if (l + 1 == static_cast<int>(lev_.size())) {
      // BOTTOM solve = long Gauss-Seidel smoothing on the coarsest grid. Self-time it (chrono only,
      // no profiler dependency here) and accumulate into the per-solve last_bottom_seconds_ (reset at
      // the top of solve()): the System reads it back to attribute the coarsest-grid cost (Spec 5
      // sec.13.11.1, ADC-479). Host serial / per-rank; the device-fence for an exact GPU bottom time is
      // deferred (counter stays an honest host-side measurement).
      const auto bottom_t0 = std::chrono::steady_clock::now();
      gs_smooth(L.phi, L.rhs, L.geom, level_bc, nbottom_, mk, ck, ep, kp, ey,
                boundary_kernel_ptr(l), boundary_context_ptr(l),
                boundary_kernel_ptr(l) ? &L.boundary_view : nullptr);  // bottom solve
      const auto bottom_t1 = std::chrono::steady_clock::now();
      last_bottom_seconds_ += std::chrono::duration<double>(bottom_t1 - bottom_t0).count();
      if (mk)
        zero_conductor(L.phi, L.mask);
      return;
    }

    poisson_residual(L.phi, L.rhs, L.geom, level_bc, L.res, mk, ck, ep, kp, ey, axy, ayx,
                     boundary_kernel_ptr(l), boundary_context_ptr(l),
                     boundary_kernel_ptr(l) ? &L.boundary_view : nullptr);
    if (l == 0)
      trace_mark("vcycle_rec(0): after poisson_residual");
    MGLevel& C = lev_[l + 1];
    average_down(L.res, C.rhs, 2, L.cfine);  // residual restriction (cfine buffer reused)
    if (l == 0)
      trace_mark("vcycle_rec(0): after average_down");
    C.phi.set_val(0.0);
    vcycle_rec(l + 1, homogeneous(level_bc));
    if (l == 0)
      trace_mark("vcycle_rec(0): after coarse recursion");

    interpolate(C.phi, L.corr, 2, L.cfine);  // correction prolongation (corr/cfine buffers reused)
    if (l == 0)
      trace_mark("vcycle_rec(0): after interpolate");
    saxpy(L.phi, Real(1), L.corr);
    if (l == 0)
      trace_mark("vcycle_rec(0): after saxpy");
    if (mk)
      zero_conductor(L.phi, L.mask);  // re-pin the conductor
    gs_smooth(L.phi, L.rhs, L.geom, level_bc, nu2_, mk, ck, ep, kp, ey, boundary_kernel_ptr(l),
              boundary_context_ptr(l), boundary_kernel_ptr(l) ? &L.boundary_view : nullptr);
    if (l == 0)
      trace_mark("vcycle_rec(0): after gs_smooth(nu2)");
  }

  BCRec bc_;
  std::function<bool(Real, Real)> active_;
  int nu1_, nu2_, nbottom_;
  int coarse_threshold_ =
      kMGDefaultCoarseThreshold;              ///< ADC-644: total-cell coarsening ceiling (0 = off).
  Real cut_theta_min_ = kEbCutFractionFloor;  ///< ADC-615: cut-fraction clamp (default 1e-3).
  bool replicated_ = false;
  bool cut_cell_ = false;
  bool has_eps_ = false;
  bool has_eps_y_ = false;
  bool has_kappa_ = false;
  bool has_cross_ = false;  // off-diagonal Axy/Ayx coefficients (FULL tensor) active
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
  std::function<Real(Real, Real)> levelset_;
  std::vector<MGLevel> lev_;
};

}  // namespace pops
