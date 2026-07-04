#pragma once

#include <optional>

#include <pops/core/foundation/types.hpp>                   // Real
#include <pops/coupling/schur/core/schur_condensation.hpp>  // pops::detail::NegateKernel (-Lap phi^n)
#include <pops/coupling/schur/program/schur_program_kernels.hpp>  // the aux-aware Schur functors + coeff_bc
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic / physical halo exchange)
#include <pops/mesh/execution/for_each.hpp>  // for_each_cell (per-cell coeff / reconstruct kernels)
#include <pops/mesh/geometry/geometry.hpp>   // Geometry (mesh metric)
#include <pops/mesh/storage/fab2d.hpp>       // Array4 / ConstArray4
#include <pops/mesh/storage/multifab.hpp>    // MultiFab
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>  // GeometricMG (the wired V-cycle, reused as a precond)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/runtime/context/grid_context.hpp>                // GridContext (System aux seam)
#include <pops/runtime/program/program_context.hpp>  // ProgramContext (the generic runtime facade)

/// @file
/// @brief The native condensed-Schur / Lorentz operator a compiled time Program lowers to (ADC-587).
///
/// These free functions were extracted VERBATIM from the ProgramContext methods
/// ``assemble_schur_coeffs`` / ``apply_laplacian_coeff`` / ``schur_explicit_flux`` /
/// ``assemble_schur_rhs`` / ``schur_reconstruct`` / ``schur_energy`` and
/// ``geometric_mg_precond_apply`` (epic ADC-399 / ADC-421 / ADC-427 / ADC-516). The kernel bodies are
/// byte-identical to the pre-split versions -- a pure module move, zero numerical change. Splitting
/// them out of ``program_context.hpp`` keeps the generic runtime facade free of any
/// Schur/Lorentz/electrostatic token, so a Schur-free Program's generated .so no longer transitively
/// includes ``coupling/schur/**`` / ``geometric_mg.hpp`` / ``lorentz_eliminator.hpp``. The codegen
/// emits calls to these functions ONLY when the Program's IR carries a Schur op.
///
/// Each function is a TEMPLATE on the runtime facade type ``Ctx`` (ADC-633) and reaches the runtime
/// through its PUBLIC seam accessors (``grid_context`` / ``alloc_scalar_field`` / ``lincomb`` /
/// ``count_kernel`` / ``schur_target``) -- it REIMPLEMENTS NOTHING, exactly as the methods did when
/// they lived on ProgramContext. Instantiating ``Ctx = ProgramContext`` reproduces the pre-template
/// bodies BYTE-FOR-BYTE (the ``schur_target`` hook is the identity on the uniform System), so a uniform
/// Program's generated .so and trajectory are unchanged. Instantiating ``Ctx = AmrProgramContext`` runs
/// the SAME assembly per AMR level (the ctx's ``grid_context()`` returns the current level's geom / aux
/// / bc, and ``schur_target`` redirects the coefficient / RHS writes to per-level composite buffers on a
/// refined hierarchy). No arithmetic is duplicated: the kernels, BC policy and stencils are shared.

namespace pops {
namespace coupling {
namespace schur {
namespace program {

using pops::runtime::program::ProgramContext;

/// Schur field-role ids for the ``ctx.assembly_target(field, role)`` write-redirection hook (ADC-633).
/// The uniform ProgramContext ignores the role (identity, byte-preserving); the AMR ProgramContext uses
/// it to route each assembled field to the matching per-level composite buffer on a refined hierarchy.
enum SchurTargetRole {
  kSchurEpsX = 0,  ///< diagonal coefficient eps_x (A_op(0,0))
  kSchurEpsY = 1,  ///< diagonal coefficient eps_y (A_op(1,1))
  kSchurAxy = 2,   ///< cross coefficient a_xy (A_op(0,1))
  kSchurAyx = 3,   ///< cross coefficient a_yx (A_op(1,0))
  kSchurRhs = 4,   ///< condensed right-hand side -Lap phi^n - g div(F)
  kSchurFlux = 5,  ///< explicit flux F = B^{-1}(mx, my)
  kSchurPhi = 6,   ///< solved potential phi^{n+theta} (READ role for schur_reconstruct / schur_source)
};

/// @name Anisotropic Schur condensation (epic ADC-399 / ADC-421)
/// The full condensed-Schur operator is L_schur(phi) = -div((I + c*rho*B^{-1}) grad phi), a tensor
/// elliptic operator whose per-cell coefficient varies with rho and B_z. These primitives let a
/// compiled Program ASSEMBLE that coefficient tensor (from the live state + B_z aux) and APPLY it
/// matrix-free, REUSING the native Schur kernels (coupling/schur/core/schur_condensation.hpp) and
/// pops::apply_laplacian's coefficient path -- no stencil / elimination reimplementation. The native
/// pops::CondensedSchur source stepper is untouched.
/// @{

/// Assemble the tensor coefficient A_op = I + c*rho*B^{-1} of the condensed-Schur operator per cell:
/// eps_x = 1 + c*rho*binv_11, eps_y = 1 + c*rho*binv_22, a_xy = c*rho*binv_12, a_yx = c*rho*binv_21,
/// with B^{-1} the closed 2x2 LorentzEliminator(th_dt, 1, B_z). @p state carries rho at component
/// @p c_rho; B_z is read from the System aux at component @p c_bz. The four coefficient fields are
/// filled over the valid cells (the SAME detail::SchurOperatorCoeffKernel the native builder uses)
/// and their ghosts extended by zero-gradient (Foextrap, periodic preserved) -- the eps_bc the
/// GeometricMG / native assembly use, so the face mean at the boundary is consistent. @p c =
/// theta^2 dt^2 alpha, @p th_dt = theta*dt. Assembled ONCE per step (rho / B_z frozen in the source),
/// then reused across every Krylov iteration of the matrix-free phi solve.
template <class Ctx>
inline void assemble_schur_coeffs(const Ctx& ctx, MultiFab& eps_x_in, MultiFab& eps_y_in,
                                  MultiFab& a_xy_in, MultiFab& a_yx_in, const MultiFab& state, Real c,
                                  Real th_dt, int c_rho, int c_bz) {
  ctx.count_kernel();
  const GridContext gc = ctx.grid_context();
  const MultiFab& aux = *gc.aux;
  // schur_target is the identity on a uniform System (byte-for-byte the pre-template body); on a refined
  // AMR hierarchy it redirects the per-cell write into that level's composite coefficient buffer.
  MultiFab& eps_x = ctx.assembly_target(eps_x_in, kSchurEpsX);
  MultiFab& eps_y = ctx.assembly_target(eps_y_in, kSchurEpsY);
  MultiFab& a_xy = ctx.assembly_target(a_xy_in, kSchurAxy);
  MultiFab& a_yx = ctx.assembly_target(a_yx_in, kSchurAyx);
  for (int li = 0; li < eps_x.local_size(); ++li) {
    const ConstArray4 s = state.fab(li).const_array();
    const ConstArray4 b = aux.fab(li).const_array();
    for_each_cell(eps_x.box(li),
                  detail::SchurOperatorCoeffKernelC{s, b, eps_x.fab(li).array(),
                                                    eps_y.fab(li).array(), a_xy.fab(li).array(),
                                                    a_yx.fab(li).array(), c, th_dt, c_rho, c_bz});
  }
  const BCRec ebc = schur_coeff_bc(gc.bc);
  fill_ghosts(eps_x, gc.geom.domain, ebc);
  fill_ghosts(eps_y, gc.geom.domain, ebc);
  fill_ghosts(a_xy, gc.geom.domain, ebc);
  fill_ghosts(a_yx, gc.geom.domain, ebc);
}

/// out = div(A grad in), A = [[eps_x, a_xy], [a_yx, eps_y]] -- the coefficiented matrix-free matvec
/// of the condensed-Schur operator. Fills @p in's ghosts (transport BC) then forwards to the SAME
/// pops::apply_laplacian coefficient path the native GeometricMG operator uses (eps / cross pointers),
/// component 0 (the scalar potential). @p in is non-const because the ghost fill writes its halos.
/// The condensed operator is L_schur(phi) = -div(A grad phi) = -out, so a matrix-free apply forms
/// it as ``ctx.apply_laplacian_coeff(out, in, ...); out *= -1`` via the affine algebra. The
/// coefficient fields are the ones assemble_schur_coeffs filled (1 ghost each).
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

/// out = B^{-1} (mx, my) per cell -- the EXPLICIT condensed-Schur flux F = rho*B^{-1}*v^n (= B^{-1}
/// applied to the momentum, avoiding the divide by rho). @p out has >= 2 components (Fx in comp 0,
/// Fy in comp 1, the layout ctx.divergence reads). @p state carries mx / my at @p c_mx / @p c_my;
/// B_z from the aux at @p c_bz; @p th_dt = theta*dt (w = th_dt*B_z). Reuses the native
/// detail::SchurExplicitFluxKernel. The condensed RHS is then -Lap phi^n - theta*dt*alpha*div(F),
/// assembled with ctx.laplacian + ctx.divergence + the affine algebra.
template <class Ctx>
inline void schur_explicit_flux(const Ctx& ctx, MultiFab& out_in, const MultiFab& state, Real th_dt,
                                int c_mx, int c_my, int c_bz) {
  ctx.count_kernel();
  const GridContext gc = ctx.grid_context();
  const MultiFab& aux = *gc.aux;
  MultiFab& out = ctx.assembly_target(out_in, kSchurFlux);  // identity on uniform; per-level on AMR
  for (int li = 0; li < out.local_size(); ++li) {
    const ConstArray4 s = state.fab(li).const_array();
    const ConstArray4 b = aux.fab(li).const_array();
    Array4 o = out.fab(li).array();
    for_each_cell(out.box(li), detail::SchurExplicitFluxKernelC{s, b, o, th_dt, c_mx, c_my, c_bz});
  }
  const BCRec ebc = schur_coeff_bc(gc.bc);
  fill_ghosts(out, gc.geom.domain, ebc);
}

/// rhs = -Lap(phi_n) - g*div(F), F = B^{-1}(mx, my) -- the FUSED condensed-Schur right-hand side
/// (the native ElectrostaticLorentzCondensation::assemble_rhs, reading B_z from the aux at @p c_bz).
/// @p phi_n is phi^n (its ghosts are filled here for the Laplacian); @p state carries mx / my at
/// @p c_mx / @p c_my; @p th_dt = theta*dt; @p g = theta*dt*alpha. @p rhs is a 1-component scalar field.
/// Internal Lap / flux buffers are allocated on @p rhs's layout (transient, like the native assembler).
/// Mirrors native assemble_rhs step-for-step (bare apply_laplacian + NegateKernel + the explicit flux
/// + SchurRhsAssembleKernel), so the top-level RHS assembly is a SINGLE op (no scalar-field affine
/// combine at IR level): the same fused -Lap - g*div(F) the native source stepper assembles.
template <class Ctx>
inline void assemble_schur_rhs(const Ctx& ctx, MultiFab& rhs_in, MultiFab& phi_n,
                               const MultiFab& state, Real th_dt, Real g, int c_mx, int c_my,
                               int c_bz) {
  ctx.count_kernel();
  const GridContext gc = ctx.grid_context();
  const MultiFab& aux = *gc.aux;
  // Redirect first: on a refined AMR hierarchy the RHS (and its transient Lap / flux scratch, sized
  // from its layout) must live on the current level, not the level-0-bound field. Identity on uniform.
  MultiFab& rhs = ctx.assembly_target(rhs_in, kSchurRhs);
  const BoxArray& ba = rhs.box_array();
  const DistributionMapping& dm = rhs.dmap();
  // 1) -Lap phi^n (bare 5-point Laplacian of the warm-started potential, negated).
  fill_ghosts(phi_n, gc.geom.domain, gc.bc);
  MultiFab lap(ba, dm, 1, 0);
  apply_laplacian(phi_n, gc.geom, lap);
  MultiFab neg_lap(ba, dm, 1, 0);
  for (int li = 0; li < neg_lap.local_size(); ++li)
    for_each_cell(neg_lap.box(li),
                  pops::detail::NegateKernel{lap.fab(li).const_array(), neg_lap.fab(li).array()});
  // 2) explicit flux F = B^{-1}(mx, my) at the center (1 ghost for the centered divergence).
  MultiFab fx(ba, dm, 2, 1);
  for (int li = 0; li < state.local_size(); ++li) {
    const ConstArray4 s = state.fab(li).const_array();
    const ConstArray4 b = aux.fab(li).const_array();
    for_each_cell(fx.box(li), detail::SchurExplicitFluxKernelC{s, b, fx.fab(li).array(), th_dt,
                                                               c_mx, c_my, c_bz});
  }
  const BCRec ebc = schur_coeff_bc(gc.bc);
  fill_ghosts(fx, gc.geom.domain, ebc);
  // 3) rhs = -Lap phi^n - g*div(F) (centered FV divergence; Fx in comp 0, Fy in comp 1 of fx).
  const Real half_idx = Real(1) / (Real(2) * gc.geom.dx());
  const Real half_idy = Real(1) / (Real(2) * gc.geom.dy());
  for (int li = 0; li < rhs.local_size(); ++li)
    for_each_cell(rhs.box(li), detail::SchurRhsAssembleKernelC{
                                   neg_lap.fab(li).const_array(), fx.fab(li).const_array(),
                                   rhs.fab(li).array(), g, half_idx, half_idy});
}

/// Reconstruct v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi^{n+theta}) and write mom = rho^n*v into
/// @p state in place (rho frozen). @p phi is phi^{n+theta} (its ghosts are filled here for the
/// centered gradient); B_z from the aux at @p c_bz; @p th_dt = theta*dt. v^n is read from the state
/// (mx/my / rho), the same closed B^{-1} (LorentzEliminator) the native reconstruction uses. The
/// final n+1 extrapolation (factor 1/theta) is left to the caller's affine algebra.
template <class Ctx>
inline void schur_reconstruct(const Ctx& ctx, MultiFab& state, MultiFab& phi, Real th_dt, int c_rho,
                              int c_mx, int c_my, int c_bz) {
  ctx.count_kernel();
  const GridContext gc = ctx.grid_context();
  const MultiFab& aux = *gc.aux;
  // On a refined AMR hierarchy the emitted (level-0-bound) solution field cannot hold a fine level's
  // potential; schur_source redirects the READ to the level's published composite phi. Identity on the
  // uniform System and the flat AMR branch (returns the passed field), so the reconstruction is
  // byte-for-byte unchanged there.
  MultiFab& phi_lvl = ctx.assembly_source(phi, kSchurPhi);
  fill_ghosts(phi_lvl, gc.geom.domain, gc.bc);
  const Real half_idx = Real(1) / (Real(2) * gc.geom.dx());
  const Real half_idy = Real(1) / (Real(2) * gc.geom.dy());
  for (int li = 0; li < state.local_size(); ++li) {
    const ConstArray4 ph = phi_lvl.fab(li).const_array();
    const ConstArray4 b = aux.fab(li).const_array();
    Array4 st = state.fab(li).array();
    for_each_cell(state.box(li),
                  detail::SchurReconstructKernelC{ph, b, st, th_dt, half_idx, half_idy, c_rho, c_mx,
                                                  c_my, c_bz});
  }
}

/// Condensed-Schur kinetic-energy increment (ADC-427): E^{n+1} = E^n + (1/2)*rho*(|v^{n+1}|^2 -
/// |v^n|^2) IN PLACE on @p state, reading v^{n+1} from @p state (after the velocity update +
/// extrapolation) and v^n from @p state_old (U^n). rho is frozen (read from @p state). Reuses the
/// native energy formula (detail::SchurEnergyKernel). Applied only when the model carries an energy
/// component (the macro passes c_E only for an energy block).
template <class Ctx>
inline void schur_energy(const Ctx& ctx, MultiFab& state, const MultiFab& state_old, int c_rho,
                         int c_mx, int c_my, int c_E) {
  ctx.count_kernel();
  for (int li = 0; li < state.local_size(); ++li) {
    Array4 st = state.fab(li).array();
    const ConstArray4 so = state_old.fab(li).const_array();
    for_each_cell(state.box(li), detail::SchurEnergyKernelC{st, so, c_rho, c_mx, c_my, c_E});
  }
}
/// @}

/// A geometric-multigrid V-cycle reused as a Krylov preconditioner (ADC-516). Owns the CACHED
/// GeometricMG the apply builds once and reuses across every Krylov iteration / step. Extracted from
/// ProgramContext's mutable ``mg_precond_`` member so the generic facade carries no MG state; the
/// codegen allocates ONE persistent instance (alloc-once, like the matrix-free scratch) and captures
/// it into the preconditioner ApplyFn lambda alongside the ProgramContext.
struct GeometricMgPreconditioner {
  /// ADC-644: the V-cycle SHAPE of the preconditioner map. A Krylov preconditioner must stay a FIXED
  /// linear map, so the configurable knobs are the V-cycle shape (pre/post/bottom sweeps, coarsest-grid
  /// floor) and how many composed fixed V-cycles form the map (n_vcycles). The DEFAULT ctor reproduces
  /// the historical single-V-cycle preconditioner bit-for-bit (nu1=nu2=2, nbottom=50, min_coarse=2, one
  /// vcycle -- the same emplace args and loop count as before ADC-644).
  GeometricMgPreconditioner(int nu1 = kMGDefaultPreSmooth, int nu2 = kMGDefaultPostSmooth,
                            int nbottom = kMGDefaultBottomSweeps, int min_coarse = kMGDefaultMinCoarse,
                            int n_vcycles = 1)
      : nu1_(nu1), nu2_(nu2), nbottom_(nbottom), min_coarse_(min_coarse), n_vcycles_(n_vcycles) {}

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
  /// The GeometricMG instance is built ONCE (lazily, on the first call) on the System mesh (geometry +
  /// block-0 BoxArray/DistributionMapping + transport BC) and CACHED in @c mg, co-distributed with the
  /// Krylov scratch so its level-0 phi/rhs pair @p in / @p out by local fab index. @p in is the Krylov
  /// vector (logically read-only); @p out is fully overwritten. The matvec budget is decided C++-side
  /// inside the Krylov loop, so this apply is invisible to the IR.
  template <class Ctx>
  void apply(const Ctx& ctx, MultiFab& out, const MultiFab& in) {
    ctx.count_kernel();
    if (!mg) {
      // Build once, on the System mesh: a scratch scalar field exposes block 0's BoxArray /
      // DistributionMapping (the same the Krylov solve allocates its r/p/Ap from), and grid_context()
      // gives the geometry + transport BC. The default V-cycle parameters (nu1=nu2=2, nbottom=50) match
      // the field-solve GeometricMG.
      const GridContext gc = ctx.grid_context();
      const MultiFab tmpl = ctx.alloc_scalar_field(1, 1);
      // ADC-644: build the V-cycle with the configured shape knobs. The defaults reproduce the
      // pre-644 emplace (min_coarse=2, nu1=nu2=2, nbottom=50), so a default-constructed
      // GeometricMgPreconditioner is bit-identical.
      mg.emplace(gc.geom, tmpl.box_array(), gc.bc, std::function<bool(Real, Real)>{},
                 /*replicated=*/false, min_coarse_, nu1_, nu2_, nbottom_);
    }
    GeometricMG& m = *mg;
    // rhs <- in (the vector to precondition); phi <- 0 (a fixed-linear cycle starts cold).
    ctx.lincomb(m.rhs(), Real(1), in, Real(0), in);
    m.phi().set_val(Real(0));
    // n_vcycles_ composed V-cycles (default 1): still a FIXED linear map M^{-1}. phi carries forward
    // across the loop so N cycles compose the same stationary iteration.
    for (int i = 0; i < n_vcycles_; ++i)
      m.vcycle();
    ctx.lincomb(out, Real(1), m.phi(), Real(0), out);  // out <- phi
  }

  int nu1_ = kMGDefaultPreSmooth;      ///< ADC-644: pre-smoothing sweeps (V-cycle shape).
  int nu2_ = kMGDefaultPostSmooth;     ///< ADC-644: post-smoothing sweeps.
  int nbottom_ = kMGDefaultBottomSweeps;  ///< ADC-644: coarsest-grid (bottom) sweeps.
  int min_coarse_ = kMGDefaultMinCoarse;  ///< ADC-644: per-axis coarsening floor.
  int n_vcycles_ = 1;                  ///< ADC-644: composed fixed V-cycles forming the map.
  std::optional<GeometricMG> mg;  ///< the cached V-cycle (built lazily on the first apply)
};

}  // namespace program
}  // namespace schur
}  // namespace coupling
}  // namespace pops
