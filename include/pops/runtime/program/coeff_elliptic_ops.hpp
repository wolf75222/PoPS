#pragma once

#include <optional>
#include <stdexcept>

#include <pops/core/foundation/types.hpp>      // Real
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic / physical halo exchange)
#include <pops/mesh/storage/multifab.hpp>       // MultiFab
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>  // GeometricMG (the wired V-cycle, reused as a precond)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/runtime/context/grid_context.hpp>                // GridContext (System aux seam)
#include <pops/runtime/numerical_defaults.hpp>  // kMGDefault* (V-cycle shape defaults)

/// @file
/// @brief Schur-free tensor-coefficient elliptic infrastructure a compiled time Program lowers to.
///
/// The generic condensed-implicit route (ADC-637) authors the elliptic operator's per-cell tensor
/// coefficient A = [[eps_x, a_xy], [a_yx, eps_y]] in the DSL and applies it matrix-free. These ops carry
/// NO scheme vocabulary -- only mathematical objects (a coefficient tensor, a V-cycle preconditioner) --
/// so they live in the runtime/program layer, not under coupling/schur. They were re-homed VERBATIM from
/// the retired condensed-Schur Program brick, a pure module move with zero numerical change:
/// @c apply_laplacian_coeff is a thin ctx wrapper over pops::apply_laplacian's tensor-coefficient path,
/// and @c GeometricMgPreconditioner is a generic geometric-multigrid V-cycle reused as a Krylov
/// preconditioner. Each is a TEMPLATE on the runtime facade type @c Ctx, reaching the runtime through its
/// PUBLIC seam accessors -- it reimplements nothing. A compiled geometric-MG Poisson Program (with NO
/// condensed op) pulls this header for the preconditioner alone.

namespace pops {
namespace runtime {
namespace program {

// The AssemblyFieldRole enum (kEpsX..kPhi, the write/read-redirection wire ids) lives on the always-
// included facade program_context.hpp so every generated .so sees it, whether or not it pulls THIS header
// (a condensed-only Program includes block_inverse.hpp, not this). AmrTensorElliptic::target switches
// on the same ints.

/// out = div(A grad in), A = [[eps_x, a_xy], [a_yx, eps_y]] -- the coefficiented matrix-free matvec of a
/// tensor elliptic operator. Fills @p in's ghosts (transport BC) then forwards to the SAME
/// pops::apply_laplacian coefficient path the native GeometricMG operator uses (eps / cross pointers),
/// component 0 (the scalar potential). @p in is non-const because the ghost fill writes its halos. A
/// condensed operator L(phi) = -div(A grad phi) = -out forms it as ``ctx.apply_laplacian_coeff(out, in,
/// ...); out *= -1`` via the affine algebra. The coefficient fields carry 1 ghost each.
template <class Ctx>
inline void apply_laplacian_coeff(const Ctx& ctx, MultiFab& out, MultiFab& in,
                                  const MultiFab& eps_x, const MultiFab& eps_y,
                                  const MultiFab& a_xy, const MultiFab& a_yx) {
  ctx.count_kernel();
  const GridContext gc = ctx.grid_context();
  fill_ghosts(in, gc.geom.domain, gc.bc);
  apply_laplacian(in, gc.geom, out, /*coef=*/nullptr, /*eps=*/&eps_x, /*kappa=*/nullptr,
                  /*eps_y=*/&eps_y, /*a_xy=*/&a_xy, /*a_yx=*/&a_yx);
}

/// A geometric-multigrid V-cycle reused as a Krylov preconditioner (ADC-516). Owns the CACHED
/// GeometricMG prepared before iteration and reused across every Krylov iteration / step. Kept off the
/// generic facade so it carries no MG state; the codegen allocates ONE persistent instance (alloc-once,
/// like the matrix-free scratch) and captures it into the preconditioner ApplyFn lambda alongside the
/// context.
struct GeometricMgPreconditioner {
  /// ADC-644: the V-cycle SHAPE of the preconditioner map. A Krylov preconditioner must stay a FIXED
  /// linear map, so the configurable knobs are the V-cycle shape (pre/post/bottom sweeps, coarsest-grid
  /// floor) and how many composed fixed V-cycles form the map (n_vcycles). The DEFAULT ctor reproduces
  /// the historical single-V-cycle preconditioner bit-for-bit (nu1=nu2=2, nbottom=50, min_coarse=2, one
  /// vcycle -- the same emplace args and loop count as before ADC-644).
  GeometricMgPreconditioner(int nu1 = kMGDefaultPreSmooth, int nu2 = kMGDefaultPostSmooth,
                            int nbottom = kMGDefaultBottomSweeps,
                            int min_coarse = kMGDefaultMinCoarse, int n_vcycles = 1)
      : nu1_(nu1), nu2_(nu2), nbottom_(nbottom), min_coarse_(min_coarse), n_vcycles_(n_vcycles) {}

  /// Build all hierarchy/storage state before the first Krylov iteration. Re-preparing the same
  /// vector space is a no-op; changing topology requires a new preconditioner object.
  template <class Ctx>
  void prepare(const Ctx& ctx, const MultiFab& prototype) {
    if (prototype.ncomp() != 1)
      throw std::invalid_argument(
          "GeometricMgPreconditioner supports exactly one component; a multi-component "
          "operator requires a genuinely block-aware preconditioner");
    if (ctx.is_polar_geometry())
      throw std::invalid_argument(
          "GeometricMgPreconditioner is Cartesian-only; polar operators require an explicit "
          "metric-aware prepared preconditioner");
    if (mg) {
      if (!PureFieldAlgebra::same_vector_space(mg->phi(), prototype) ||
          mg->phi().n_grow() != prototype.n_grow())
        throw std::logic_error("GeometricMgPreconditioner topology changed after preparation");
      return;
    }
    const GridContext gc = ctx.grid_context();
    mg.emplace(gc.geom, prototype.box_array(), gc.bc, std::function<bool(Real, Real)>{},
               /*replicated=*/false, min_coarse_, nu1_, nu2_, nbottom_);
    // Materialize halo schedules, MPI buffer capacities and every lazy V-cycle resource now. The
    // zero probe is mathematically neutral and happens once, before a Krylov iteration can begin.
    PureFieldAlgebra::zero_valid(mg->rhs());
    PureFieldAlgebra::zero_valid(mg->phi());
    mg->vcycle();
    PureFieldAlgebra::zero_valid(mg->rhs());
    PureFieldAlgebra::zero_valid(mg->phi());
  }

  /// out <- M^{-1}(in): ONE geometric-multigrid V-cycle of the bare 5-point Laplacian, used as a
  /// matrix-free Krylov PRECONDITIONER (the ``preconditioner=preconditioners.GeometricMG()`` route of
  /// P.solve_linear for GMRES / BiCGStab, ADC-516). It REUSES the already-wired pops::GeometricMG (the
  /// same V-cycle the field solve runs) -- no new numerical kernel: set the level-0 rhs to @p in, start
  /// from phi = 0, run a SINGLE @c vcycle(), copy the result into @p out.
  ///
  /// EXACTLY ONE V-cycle from a ZERO guess is mandatory: a preconditioner must be a FIXED linear map
  /// M^{-1} (the same operator on every Krylov apply) for GMRES / BiCGStab to converge. Iterating to a
  /// tolerance (``solve()``) would make the trip count -- hence the map -- depend on the input vector, a
  /// VARIABLE (nonlinear) preconditioner that breaks the Krylov recurrences. The V-cycle of the bare
  /// Laplacian is symmetric-positive and history-free, so one cycle from phi=0 is a valid stationary
  /// M^{-1} approximating L^{-1}.
  ///
  /// The GeometricMG instance is built ONCE by prepare() on the System mesh and CACHED in @c mg,
  /// co-distributed with the
  /// Krylov scratch so its level-0 phi/rhs pair @p in / @p out by local fab index. @p in is the Krylov
  /// vector (logically read-only); @p out is fully overwritten. The matvec budget is decided C++-side
  /// inside the Krylov loop, so this apply is invisible to the IR.
  template <class Ctx>
  void apply(const Ctx& ctx, MultiFab& out, const MultiFab& in) {
    ctx.count_kernel();
    if (!mg)
      throw std::logic_error("GeometricMgPreconditioner::apply called before prepare");
    GeometricMG& m = *mg;
    // rhs <- in (the vector to precondition); phi <- 0 (a fixed-linear cycle starts cold).
    PureFieldAlgebra::copy(m.rhs(), in);
    PureFieldAlgebra::zero_valid(m.phi());
    // n_vcycles_ composed V-cycles (default 1): still a FIXED linear map M^{-1}. phi carries forward
    // across the loop so N cycles compose the same stationary iteration.
    for (int i = 0; i < n_vcycles_; ++i)
      m.vcycle();
    PureFieldAlgebra::copy(out, m.phi());
  }

  int nu1_ = kMGDefaultPreSmooth;         ///< ADC-644: pre-smoothing sweeps (V-cycle shape).
  int nu2_ = kMGDefaultPostSmooth;        ///< ADC-644: post-smoothing sweeps.
  int nbottom_ = kMGDefaultBottomSweeps;  ///< ADC-644: coarsest-grid (bottom) sweeps.
  int min_coarse_ = kMGDefaultMinCoarse;  ///< ADC-644: per-axis coarsening floor.
  int n_vcycles_ = 1;                  ///< ADC-644: composed fixed V-cycles forming the map.
  std::optional<GeometricMG> mg;  ///< the cached V-cycle (built explicitly by prepare)
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
