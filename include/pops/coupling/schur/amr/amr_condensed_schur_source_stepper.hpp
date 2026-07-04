#pragma once

#include <pops/coupling/schur/source/condensed_schur_source_stepper.hpp>  // CondensedSchurSourceStepper (#126) + detail kernels
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/coupling/schur/core/schur_condensation.hpp>  // ElectrostaticLorentzCondensation (assemble per level)
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>  // CompositeFacPoisson (composite FAC elliptic solve)
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>   // mf_average_down_mb (fine -> coarse cascade)
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>  // AmrLevelMP (multi-patch hierarchy)

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief AmrCondensedSchurSourceStepper: AMR counterpart of the Schur-condensed SOURCE stage
///        (CondensedSchurSourceStepper, #126), carried over a HIERARCHY of levels (AmrLevelMP) rather
///        than over a uniform grid. This is the GLOBAL electrostatic/Lorentz source stage of the
///        "amr-schur" path -- the refined equivalent of the uniform path
///          System(...).add_equation(time=Strang(hyperbolic=Explicit(ssprk3),
///                                                source=CondensedSchur(theta, alpha)))
///        and NOT a local cell-by-cell source (cf. the local IMEX backward_euler_source of the
///        amr-imex path, which is NOT the quantitatively-validated reference source treatment;
///        cf. docs/HOFFART_FIDELITY.md).
///
/// STRATEGY (option A, mirror of the existing AMR Poisson compute_aux/solve_fields). The AMR elliptic
/// solver of this code solves Poisson on the COARSE LEVEL then injects grad phi to the fine levels (the
/// fine patches refine TRANSPORT, not the elliptic solve). The condensed source stage follows the
/// SAME approach: it assembles and solves the condensed operator A_op = I + theta^2 dt^2 alpha rho B^{-1}
/// on the coarse level (by COMPOSING the uniform stage #126, bit-for-bit), then -- for a multi-level
/// hierarchy -- injects grad phi^{n+theta} to the fine levels and reconstructs the velocities there, ending with
/// the fine -> coarse cascade (average_down) which restores the consistency of the covered coarse cells
/// (invariant #169). A spatially constant state (mono-level) degenerates EXACTLY into the uniform stage:
/// this is the parity criterion (Step 2).
///
/// SCOPE (ADC-636, generalized envelope). The MONO-LEVEL path is complete and bit-identical to the
/// uniform stage #126. The MULTI-LEVEL path is IMPLEMENTED as a COMPOSITE condensed source stage: the
/// tensor Schur elliptic is solved by the composite FAC over the WHOLE nested tower, velocity
/// reconstruction PER LEVEL, then the fine->coarse average_down cascade (cf. step_multilevel). It
/// inherits the lifted FAC envelope: an arbitrary nested hierarchy (N levels, adjacent fine patches,
/// MPI with a replicated coarse + distributed fine). ONE patch, 2 levels, mono-rank degenerates
/// EXACTLY into Phase 3c (bit-identical). Only ratio != 2 (ADC-602) and overlapping / non-nested /
/// misaligned patches are refused, precisely, by the FAC ctor.
///
/// LIFE CYCLE / DEVICE / MPI. Built ONCE on the COARSE layout (BoxArray + Geometry + Poisson BC);
/// all buffers of the coarse uniform stage are allocated at construction and reused
/// by step(). The coarse Krylov solve is COLLECTIVE (dot/all_reduce over all ranks, including
/// empty ones) -- like the uniform stage: no deadlock. theta/dt may change between calls.

namespace pops {

/// Schur-condensed SOURCE stage over an AMR hierarchy. GENERIC over any fluid block that exposes
/// the Density / MomentumX / MomentumY roles (+ optional Energy), exactly like the uniform stage.
class AmrCondensedSchurSourceStepper {
 public:
  /// @p vars: descriptor of the fluid block (MUST expose Density / MomentumX / MomentumY; Energy
  ///            optional). Validated HERE (host) by the ctor of the coarse uniform stage.
  /// @p coarse_geom: geometry of the COARSE LEVEL (cartesian).
  /// @p coarse_ba: decomposition of the coarse level (replicated mono-box or distributed multi-box).
  /// @p bcPhi: BC of the potential phi (same as the coarse Poisson).
  /// @p alpha: electrostatic coupling constant.
  /// @p n_precond_vcycles: N MG V-cycles per application of the BiCGStab preconditioner (1 or 2).
  AmrCondensedSchurSourceStepper(const VariableSet& vars, const Geometry& coarse_geom,
                                 const BoxArray& coarse_ba, const BCRec& bcPhi, Real alpha,
                                 int n_precond_vcycles = 1)
      : AmrCondensedSchurSourceStepper(
            vars, vars.index_of(VariableRole::Density), vars.index_of(VariableRole::MomentumX),
            vars.index_of(VariableRole::MomentumY), vars.index_of(VariableRole::Energy),
            coarse_geom, coarse_ba, bcPhi, alpha, n_precond_vcycles) {}

  /// EXPLICIT-COMPONENT variant (audit wave 3, parity with the System steppers): roles
  /// carried by the ABI instead of being resolved canonically. The canonical ctor DELEGATES here.
  AmrCondensedSchurSourceStepper(const VariableSet& vars, int c_rho, int c_mx, int c_my, int c_E,
                                 const Geometry& coarse_geom, const BoxArray& coarse_ba,
                                 const BCRec& bcPhi, Real alpha, int n_precond_vcycles = 1)
      : vars_(vars),
        coarse_geom_(coarse_geom),
        coarse_ba_(coarse_ba),
        bcPhi_(bcPhi),
        alpha_(alpha),
        c_rho_(c_rho),
        c_mx_(c_mx),
        c_my_(c_my),
        c_E_(c_E),
        coarse_(vars, c_rho, c_mx, c_my, c_E, coarse_geom, coarse_ba, bcPhi, alpha,
                n_precond_vcycles) {}

  /// Tolerance / budget of the COARSE stage Krylov solve (delegated to the uniform stage #126;
  /// historical defaults 1e-10 / 400). The COMPOSITE multi-level solve (FAC) is configured
  /// separately by set_fac_options.
  void set_krylov(Real tol, int max_iters) { coarse_.set_krylov(tol, max_iters); }

  /// ADC-614: install the composite-FAC knobs (outer iters / fine sweeps / composite tol / internal
  /// coarse GeometricMG rel_tol+cycles / verbose) applied to the composite solver when the multi-level
  /// path builds it (ensure_fac). Defaults are the kFAC* constants -> bit-identical composite solve.
  void set_fac_options(const CompositeFacOptions& o) {
    fac_options_ = o;
    if (fac_)
      fac_->set_options(o);  // already-built solver picks up the new knobs on the next solve
  }

  /// true if the model carries an Energy role (energy update active in the coarse stage).
  bool has_energy() const { return coarse_.energy_comp() >= 0; }

  /// Condensed SOURCE stage, IN-PLACE on the hierarchy @p levels and the coarse potential @p coarse_phi.
  ///   @p levels: multi-patch hierarchy; levels[0] = COARSE (level 0), levels[k>=1] = FINE
  ///                  (ratio 2). The conservative state of each level is levels[k].U (rho FROZEN,
  ///                  mom/E updated; same convention as the uniform stage).
  ///   @p coarse_phi: potential of the coarse level. INPUT phi^n (warm start of the solve); OUTPUT
  ///                  phi^{n+1}. Same object as the coarse Poisson (mg_.phi() of the coupler) on the facade side.
  ///   @p coarse_bz: B_z field of the coarse level (aux channel), component @p c_bz read at the center.
  ///   @p theta / @p dt: theta-scheme (theta in (0, 1]); dt = effective step (stride factor included
  ///                  by the caller, like s.advance / run_source_stage of the uniform path).
  void step(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi, const MultiFab& coarse_bz,
            int c_bz, Real theta, Real dt) {
    if (levels.empty())
      return;
    // A fine level EFFECTIVELY POPULATED (>= one patch) signals a multi-level hierarchy. NB: the
    // compiled path (build_amr_compiled) ALWAYS allocates a seed fine level, EMPTY after regrid when
    // no refinement is requested (refine_threshold disabled) -> levels.size() is 2 but the
    // hierarchy is EFFECTIVELY mono-level. So we gate on the NUMBER OF fine PATCHES, not on
    // levels.size(), to avoid refusing the mono-level case with an allocated but empty fine level.
    int n_fine_patches = 0;
    for (std::size_t k = 1; k < levels.size(); ++k)
      n_fine_patches += static_cast<int>(levels[k].U.box_array().size());
    if (n_fine_patches == 0) {
      // MONO-LEVEL (no fine patch): COMPLETE uniform stage on the coarse level (assemble + solve +
      // reconstruction + extrapolation + energy + ghosts), bit-for-bit identical to #126.
      coarse_.step(levels[0].U, coarse_phi, coarse_bz, c_bz, theta, dt);
      return;
    }
    // MULTI-LEVEL: COMPOSITE condensed source stage -- the fine patches REALLY refine the elliptic
    // (tensor Schur operator solved by FAC over the whole tower), then velocity reconstruction PER
    // LEVEL and the fine->coarse average_down cascade. ADC-636 lifted the FAC envelope, so the
    // hierarchy may be an arbitrary NESTED tower (N levels, adjacent fine patches, MPI: replicated
    // coarse + distributed fine). Only ratio != 2 (ADC-602) and overlapping/non-nested/misaligned
    // patches are refused, precisely, by the FAC ctor. step_multilevel loops over the levels.
    step_multilevel(levels, coarse_phi, coarse_bz, c_bz, theta, dt);
  }

  /// Diagnostic of the last coarse stage solve (BiCGStab iterations, relative residual, convergence).
  const KrylovResult& last_solve() const { return coarse_.last_solve(); }

  int density_comp() const { return coarse_.density_comp(); }
  int momentum_x_comp() const { return coarse_.momentum_x_comp(); }
  int momentum_y_comp() const { return coarse_.momentum_y_comp(); }
  int energy_comp() const { return coarse_.energy_comp(); }

 private:
  /// COMPOSITE N-level condensed source stage (ADC-636). Builds the composite elliptic over the whole
  /// nested tower (levels[1..L-1] patches), assembles the Schur condensed operator (A = I + c rho
  /// B^{-1}, full tensor) + the condensed RHS PER LEVEL (ElectrostaticLorentzCondensation), solves the
  /// COMPOSITE elliptic (CompositeFacPoisson: every level refines the elliptic), reconstructs the
  /// velocity PER LEVEL (v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})), extrapolates phi/v to
  /// the full step, updates the energy, then cascades fine -> coarse (average_down, covered cells). The
  /// coarse (level 0) stays replicated; the fine levels are distributed (MPI). Only ratio != 2 and
  /// overlapping/non-nested/misaligned patches are refused, precisely, at the FAC ctor.
  void step_multilevel(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi,
                       const MultiFab& coarse_bz, int c_bz, Real theta, Real dt) {
    const int L = static_cast<int>(levels.size());
    // Fine-level tilings (levels[1..L-1]); build/rebuild the composite FAC on the whole tower.
    std::vector<BoxArray> level_boxes;
    for (int k = 1; k < L; ++k)
      level_boxes.push_back(levels[k].U.box_array());
    ensure_fac(level_boxes);
    ElectrostaticLorentzCondensation builder(vars_, alpha_, theta, dt);

    // Per-level geometry, B_z, phi^n, v^n. bz/phi/v are needed by the reconstruction after the solve.
    std::vector<Geometry> geom(L);
    std::vector<MultiFab> bz(L), phi_n(L), vx_n(L), vy_n(L);
    for (int k = 0; k < L; ++k) {
      geom[k] = (k == 0) ? coarse_geom_ : coarse_geom_.refine(1 << k);
      const BoxArray ba = levels[k].U.box_array();
      const DistributionMapping dm = levels[k].U.dmap();
      bz[k] = MultiFab(ba, dm, 1, 1);
      phi_n[k] = MultiFab(ba, dm, 1, 1);
      vx_n[k] = MultiFab(ba, dm, 1, 0);
      vy_n[k] = MultiFab(ba, dm, 1, 0);
      if (k == 0) {
        copy_comp(bz[0], coarse_bz, c_bz);
        device_fence();
        fill_ghosts(bz[0], geom[0].domain, coeff_bc(bcPhi_));
        copy0(phi_n[0], coarse_phi);
      } else {
        bilerp_coarse_to_fine(bz[k], bz[k - 1]);  // fine B_z from the parent (B0 uniform -> exact)
        copy0(phi_n[k], *levels[k].aux);          // injected phi^n aux (comp 0)
      }
      extract_v(levels[k].U, vx_n[k], vy_n[k]);
    }

    // Operator + condensed RHS assembly PER LEVEL, into the composite solver's per-level fields.
    // eps_x == eps_y for the Schur (A_xx = A_yy = 1 + c rho/det): eps_x -> the composite eps, eps_y ->
    // a discarded scratch. f_composite = -rhs_schur (sign convention #126).
    fac_->use_variable_coefficient(true);
    fac_->use_cross_terms(true);
    for (int k = 0; k < L; ++k) {
      const BoxArray ba = levels[k].U.box_array();
      const DistributionMapping dm = levels[k].U.dmap();
      MultiFab eps_y(ba, dm, 1, 1), rhs(ba, dm, 1, 0), pn(ba, dm, 1, 1);
      builder.assemble_operator(levels[k].U, bz[k], geom[k], bcPhi_, fac_->eps_level(k), eps_y,
                                fac_->a_xy_level(k), fac_->a_yx_level(k));
      copy0(pn, phi_n[k]);
      builder.assemble_rhs(pn, levels[k].U, bz[k], geom[k], bcPhi_, rhs);
      negate_into(fac_->rhs_level(k), rhs);
    }

    // COMPOSITE SOLVE: phi^{n+theta} per level (every level refines the elliptic).
    fac_->solve();

    // Velocity reconstruction + phi/v extrapolation + energy, PER LEVEL. Only the coarse level fills
    // its physical phi ghosts; the finer levels keep the C/F ghosts the composite solve set.
    for (int k = 0; k < L; ++k)
      reconstruct_level(levels[k].U, fac_->phi_level(k), phi_n[k], bz[k], vx_n[k], vy_n[k], geom[k],
                        theta, dt, /*fill_phi_ghosts=*/k == 0);

    // coarse phi^{n+1} (extrapolated in place into fac_->phi_level(0)) -> published into coarse_phi.
    copy0(coarse_phi, fac_->phi_level(0));

    // fine -> coarse cascade (finest to coarsest): each covered parent cell = 2x2 average of the child
    // (invariant #169). At L == 2 this is the single average_down of the historical path.
    device_fence();
    for (int k = L - 1; k >= 1; --k)
      mf_average_down_mb(levels[k].U, levels[k - 1].U);
    device_fence();
    fill_ghosts(coarse_phi, geom[0].domain, bcPhi_);
  }

  /// Reconstructs v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}) (CENTERED grad), writes mom = rho v;
  /// extrapolates phi and v from the theta-stage to the full step (f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n)); updates
  /// the energy (if the role is present). @p fill_phi_ghosts: fill the physical ghosts of phi (coarse)
  /// ; false for the fine level (the C-F ghosts are already set by the composite solve -- do NOT overwrite them).
  void reconstruct_level(MultiFab& state, MultiFab& phi_nt, const MultiFab& phi_n,
                         const MultiFab& bz, const MultiFab& vx_n, const MultiFab& vy_n,
                         const Geometry& geom, Real theta, Real dt, bool fill_phi_ghosts) {
    const Real th_dt = theta * dt, inv_theta = Real(1) / theta;
    const Real half_idx = Real(1) / (Real(2) * geom.dx());
    const Real half_idy = Real(1) / (Real(2) * geom.dy());
    device_fence();
    if (fill_phi_ghosts)
      fill_ghosts(phi_nt, geom.domain, bcPhi_);
    device_fence();
    MultiFab vx_t(state.box_array(), state.dmap(), 1, 0),
        vy_t(state.box_array(), state.dmap(), 1, 0);
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(
          state.box(li),
          detail::SchurReconstructKernel{
              phi_nt.fab(li).const_array(), vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
              bz.fab(li).const_array(), state.fab(li).array(), vx_t.fab(li).array(),
              vy_t.fab(li).array(), th_dt, half_idx, half_idy, c_rho_, c_mx_, c_my_});
    for (int li = 0; li < phi_nt.local_size(); ++li)
      for_each_cell(phi_nt.box(li),
                    detail::SchurExtrapolateScalarKernel{phi_n.fab(li).const_array(),
                                                         phi_nt.fab(li).array(), inv_theta});
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li), detail::SchurExtrapolateVelocityKernel{
                                       vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
                                       vx_t.fab(li).array(), vy_t.fab(li).array(),
                                       state.fab(li).array(), inv_theta, c_rho_, c_mx_, c_my_});
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li), detail::SchurEnergyKernel{
                                         vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
                                         vx_t.fab(li).const_array(), vy_t.fab(li).const_array(),
                                         state.fab(li).array(), c_rho_, c_E_});
    device_fence();
    fill_ghosts(state, geom.domain, coeff_bc(bcPhi_));
  }

  /// Builds (or rebuilds if the tower changes) the composite elliptic solver over ALL fine levels
  /// (ADC-636). The rebuild key is the FULL per-level box set (boxes AND order at every level), so a
  /// regrid that changes any level's tiling rebuilds; an unchanged tower reuses the FAC. A
  /// single-patch-level tower uses the 2-level ctor (bit-identical); deeper towers use the N-level
  /// ctor. The FAC ctor refuses ratio != 2 and overlapping/non-nested/misaligned patches, precisely.
  void ensure_fac(const std::vector<BoxArray>& level_boxes) {
    std::vector<std::vector<Box2D>> key;
    for (const BoxArray& ba : level_boxes)
      key.push_back(ba.boxes());
    if (fac_ && fac_level_boxes_ == key)
      return;
    if (level_boxes.size() == 1)
      fac_ = std::make_unique<CompositeFacPoisson>(coarse_geom_, coarse_ba_, bcPhi_, level_boxes[0],
                                                   kAmrRefRatio);
    else
      fac_ = std::make_unique<CompositeFacPoisson>(coarse_geom_, coarse_ba_, bcPhi_, level_boxes,
                                                   kAmrRefRatio);
    fac_->set_options(fac_options_);  // ADC-614: apply the installed FAC knobs (default = kFAC*).
    fac_level_boxes_ = std::move(key);
  }

  /// BC of the coefficients (eps/B_z) and of the published state: periodic preserved, physical edge zero-gradient.
  static BCRec coeff_bc(const BCRec& b) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec c;
    c.xlo = fo(b.xlo);
    c.xhi = fo(b.xhi);
    c.ylo = fo(b.ylo);
    c.yhi = fo(b.yhi);
    return c;
  }

  void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyComp0Kernel{dst.fab(li).array(), src.fab(li).const_array()});
  }
  void copy_comp(MultiFab& dst, const MultiFab& src, int c) {  // dst comp0 <- src comp c
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyBzKernel{src.fab(li).const_array(), dst.fab(li).array(), c});
  }
  void negate_into(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::NegateKernel{src.fab(li).const_array(), dst.fab(li).array()});
  }
  void extract_v(const MultiFab& state, MultiFab& vx, MultiFab& vy) {
    device_fence();
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::ExtractVelocityKernel{state.fab(li).const_array(), vx.fab(li).array(),
                                                  vy.fab(li).array(), c_rho_, c_mx_, c_my_});
  }
  /// Fills valid + ghosts of EACH fine patch @p fine by bilerp of the coarse field @p coarse (B_z,
  /// etc.). Coarse mono-box replicated; we loop over the local fine patches (multi-patch).
  void bilerp_coarse_to_fine(MultiFab& fine, const MultiFab& coarse) {
    device_fence();
    const ConstArray4 C = coarse.fab(0).const_array();
    const int ng = fine.n_grow();
    for (int li = 0; li < fine.local_size(); ++li) {
      Array4 F = fine.fab(li).array();
      const Box2D vb = fine.box(li);
      for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
        for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i)
          F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, kAmrRefRatio);
    }
  }

  VariableSet vars_;
  Geometry coarse_geom_;
  BoxArray coarse_ba_;
  BCRec bcPhi_;
  Real alpha_;
  int c_rho_, c_mx_, c_my_, c_E_;
  /// Uniform condensed source stage carried over the COARSE LEVEL (MONO-LEVEL path, parity #126).
  CondensedSchurSourceStepper coarse_;
  /// Composite elliptic solver (MULTI-LEVEL path), built lazily on the fine patches.
  std::unique_ptr<CompositeFacPoisson> fac_;
  /// ADC-614: FAC knobs applied to the composite solver at build; defaults = kFAC* (bit-identical).
  CompositeFacOptions fac_options_;
  /// Per-level tiling (boxes + order at each fine level) of the last built FAC: detects a tower change
  /// (ADC-636: the rebuild key spans the whole tower, not just level 1).
  std::vector<std::vector<Box2D>> fac_level_boxes_;
};

}  // namespace pops
