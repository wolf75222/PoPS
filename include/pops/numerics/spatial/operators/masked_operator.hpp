/// @file
/// @brief Domain-mask-aware Cartesian residual (conservative active sub-domain, OPT-IN).
///
/// CONTRACT: the mask-aware variant of assemble_rhs. SEPARATE entry point: the default path
/// (System::step) stays strictly bit-identical as long as it does not call this overload.
///   - assemble_rhs_masked<Limiter,NumericalFlux>: residual restricted to a 0/1 cell-centered mask.
///
/// Convention: mask(i,j) >= 0.5 -> ACTIVE. A face is OPEN only if BOTH adjacent cells are active;
/// otherwise the normal flux is set to ZERO (FV wall), so the mass over the active sub-domain is
/// conserved to machine precision. Reconstruction and the positivity role come from face_flux.hpp /
/// positivity.hpp.

#pragma once

#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/primitives/finite.hpp>
#include <pops/numerics/spatial/primitives/face_flux.hpp>  // reconstruct_pp, require_reconstruction_ghosts
#include <pops/numerics/spatial/primitives/positivity.hpp>    // detail::positivity_comp
#include <pops/numerics/spatial/primitives/state_access.hpp>  // load_state, load_aux

#include <type_traits>

namespace pops {

// ============================================================================
// DOMAIN MASK (T2 effort, conservative, OPT-IN -- default path untouched)
// ============================================================================
// The mask makes the FV transport aware of an ACTIVE sub-domain (e.g. a bounded disk-shaped region).
// Convention: mask(i, j) >= 0.5 -> ACTIVE cell, otherwise INACTIVE. A face is OPEN (normal flux
// computed) if BOTH adjacent cells are active; it is CLOSED (normal flux set to ZERO) if at least
// one is inactive. Zeroing the normal flux at active/inactive faces makes the step CONSERVATIVE
// over the active sub-domain: no mass crosses the boundary, so the total mass over the active cells
// is conserved to machine precision (telescoping internal fluxes, zero boundary fluxes). This is the
// FV counterpart of the conducting wall (which only acts on the elliptic part).
//
// The residual is written ONLY on the active cells; an inactive cell keeps its residual at 0
// (the caller does not advance it). This header does NOT wire this path into System::step: it
// provides the mask-aware brick, exercised directly by the tests and, eventually, behind the
// active-sub-domain opt-in.

namespace detail {
/// Activity indicator of a cell from a 0/1 cell-centered mask (>= 0.5 -> active).
POPS_HD inline bool mask_active(const ConstArray4& mask, int i, int j) {
  return mask(i, j, 0) >= Real(0.5);
}

/// Device-safe description of Cartesian boundary faces owned by a prepared shared interface.
/// The ordinary masked operator receives the default value (no omission).  A multi-block Program
/// supplies the exact domain and four booleans resolved by its boundary plan, so the local residual
/// leaves those faces empty before the unique pair flux is scattered by the interface scheduler.
/// This is geometry-agnostic: it describes face ownership, not the shape of the embedded boundary.
struct BoundaryFaceOmission {
  Box2D domain{};
  bool xlo = false;
  bool xhi = false;
  bool ylo = false;
  bool yhi = false;

  POPS_HD bool omit(int axis, int side, int i, int j) const {
    if (axis == 0)
      return side < 0 ? (xlo && i == domain.lo[0]) : (xhi && i == domain.hi[0]);
    return side < 0 ? (ylo && j == domain.lo[1]) : (yhi && j == domain.hi[1]);
  }
};

/// AssembleRhsMaskedKernel: variant of AssembleRhsKernel AWARE of a domain mask.
///
/// Inactive cell -> residual 0 (not advanced by the caller). Active cell -> R = -div Fhat + S,
/// BUT the normal flux of a face whose neighbor cell is INACTIVE is set to ZERO (FV wall:
/// zero normal flux at the active/inactive boundary) -> mass conservation over the active
/// sub-domain. Named functor (same device contract as AssembleRhsKernel). POPS_HD.
///
/// Diffusive models are rejected by the block capability preflight until a conservative EB
/// diffusive-flux provider exists.  This kernel therefore owns hyperbolic transport plus source only.
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsMaskedKernel {
  Model model;
  ConstArray4 u, ax, mask;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  BoundaryFaceOmission omission{};
  POPS_HD void operator()(int i, int j) const {
    if (!mask_active(mask, i,
                     j)) {  // cell outside the active sub-domain: zero residual, not advanced
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) = Real(0);
      return;
    }
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);

    // Reconstruct only after the face is proven open.  This ordering is part of the EB policy:
    // inactive storage may contain arbitrary sentinels and must never enter a primitive conversion,
    // limiter or numerical flux merely because its face is later zeroed.
    const FaceContext xface = FaceContext::axis_aligned(0);
    typename Model::State Fxm{}, Fxp{};
    if (!omission.omit(0, -1, i, j) && mask_active(mask, i - 1, j)) {
      const auto Lxm =
          reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
      const auto Rxm =
          reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
      const auto evaluation =
          evaluate_numerical_flux_at(nflux, model, Lxm, ax, i - 1, j, Rxm, ax, i, j, xface);
      Fxm = apply_face_measure(evaluation.checked_density(), xface).value;
    }
    if (!omission.omit(0, +1, i, j) && mask_active(mask, i + 1, j)) {
      const auto Lxp =
          reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
      const auto Rxp =
          reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
      const auto evaluation =
          evaluate_numerical_flux_at(nflux, model, Lxp, ax, i, j, Rxp, ax, i + 1, j, xface);
      Fxp = apply_face_measure(evaluation.checked_density(), xface).value;
    }

    // y faces
    const FaceContext yface = FaceContext::axis_aligned(1);
    typename Model::State Fym{}, Fyp{};
    if (!omission.omit(1, -1, i, j) && mask_active(mask, i, j - 1)) {
      const auto Lym =
          reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
      const auto Rym =
          reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
      const auto evaluation =
          evaluate_numerical_flux_at(nflux, model, Lym, ax, i, j - 1, Rym, ax, i, j, yface);
      Fym = apply_face_measure(evaluation.checked_density(), yface).value;
    }
    if (!omission.omit(1, +1, i, j) && mask_active(mask, i, j + 1)) {
      const auto Lyp =
          reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
      const auto Ryp =
          reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
      const auto evaluation =
          evaluate_numerical_flux_at(nflux, model, Lyp, ax, i, j, Ryp, ax, i, j + 1, yface);
      Fyp = apply_face_measure(evaluation.checked_density(), yface).value;
    }

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;
  }
};

template <class Limiter, class NumericalFlux, class Model>
void assemble_rhs_masked_impl(const Model& model, const MultiFab& U, const MultiFab& aux,
                              const MultiFab& mask, const Geometry& geom, MultiFab& R,
                              bool recon_prim, Real pos_floor, Real weno_eps,
                              BoundaryFaceOmission omission) {
  require_reconstruction_ghosts<Limiter>(U);
  const Real dx = geom.dx(), dy = geom.dy();
  Limiter lim{};
  if constexpr (std::is_same_v<Limiter, Weno5>)
    lim.eps = weno_eps;
  const NumericalFlux nflux{};
  const int pos_comp = positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 mk = mask.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(
        v, AssembleRhsMaskedKernel<Limiter, NumericalFlux, Model>{
               model, u, ax, mk, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp, omission});
  }
  reject_nonfinite_finite_volume_data("assemble_rhs_masked", R);
}
}  // namespace detail

/// assemble_rhs_masked<Limiter,NumericalFlux>: residual R = -div Fhat + S RESTRICTED to a 0/1
/// cell-centered domain mask (OPT-IN, T2 effort). On an inactive cell R = 0 (not advanced); on an
/// active cell, the normal flux of a face whose neighbor is inactive is set to zero (FV wall).
/// Result: the mass over the active sub-domain is CONSERVED to machine precision (no flux crosses
/// the boundary) -- property validated by the active-sub-domain mass-conservation test.
///
/// @p mask must have the SAME layout as @p U (same BoxArray / DistributionMapping) and carry at
/// least 1 ghost (reading the neighbors i-1/i+1/j-1/j+1 up to the edge). This entry point is
/// SEPARATE from assemble_rhs: the default path (System::step) stays strictly bit-identical as long
/// as it does NOT call this overload.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_masked(const Model& model, const MultiFab& U, const MultiFab& aux,
                         const MultiFab& mask, const Geometry& geom, MultiFab& R,
                         bool recon_prim = false, Real pos_floor = Real(0),
                         Real weno_eps = kWenoEpsilon) {
  detail::assemble_rhs_masked_impl<Limiter, NumericalFlux>(model, U, aux, mask, geom, R, recon_prim,
                                                           pos_floor, weno_eps, {});
}

}  // namespace pops
