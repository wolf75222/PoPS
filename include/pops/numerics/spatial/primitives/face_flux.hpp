/// @file
/// @brief Face-state reconstruction and face fluxes of the Cartesian spatial operator.
///
/// CONTRACT: everything that lives AT a face, before the divergence.
///   - reconstruct<>: face value from the MUSCL or WENO5 stencil (POPS_HD).
///   - reconstruct_pp<>: reconstruct + Zhang-Shu positivity limiter (positivity.hpp).
///   - require_reconstruction_ghosts<>: structural entry guard (state ghosts >= stencil).
///   - xface_box / yface_box: face boxes normal to x / y for a cell box.
///   - compute_face_fluxes<>: face fluxes (the brick required by the AMR reflux).
///
/// reconstruct_pp is THE single reconstruction entry point that every assembly kernel calls
/// (cartesian_operator.hpp, masked_operator.hpp), so it bundles the limiter here next to the
/// reconstruction it limits. Depends on state_access.hpp and positivity.hpp.

#pragma once

#include <pops/core/model/physical_model.hpp>  // HasPrimitiveVars: optional primitive reconstruction
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/primitives/positivity.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <stdexcept>  // require_reconstruction_ghosts: state without the stencil width -> clear error

namespace pops {

namespace detail {

/// Device-callable view of one conservative component along the oriented reconstruction line.
/// The policy chooses every integer offset; this adapter alone translates it into mesh indices.
/// It is a trivially copyable kernel value and performs no allocation or host/device transfer.
struct ConservativeStencilSampler {
  ConstArray4 values;
  int i;
  int j;
  int direction;
  int orientation;
  int component;

  POPS_HD Real operator()(int offset) const {
    const int displacement = orientation * offset;
    return direction == 0 ? values(i + displacement, j, component)
                          : values(i, j + displacement, component);
  }
};

/// Fixed-capacity, kernel-stack cache for primitive states.  Capacity comes from the storage
/// contract, while the policy's independent envelope controls which offsets are materialized.
/// Each requested state is converted exactly once, not once per component.
template <class Model, class Reconstruction>
struct PrimitiveStencilCache {
  static_assert(stencil_envelope_fits_storage<Reconstruction>,
                "sampled reconstruction offsets exceed the declared ghost-storage capacity");
  static constexpr int radius = Reconstruction::n_ghost - 1;
  static constexpr int capacity = radius > 0 ? 2 * radius + 1 : 1;
  typename Model::Prim values[capacity]{};

  POPS_HD typename Model::Prim& at(int offset) { return values[offset + radius]; }
};

template <class Model, class Reconstruction>
struct CachedPrimitiveComponentSampler {
  const PrimitiveStencilCache<Model, Reconstruction>* cache;
  int component;

  POPS_HD Real operator()(int offset) const {
    return cache->values[offset + PrimitiveStencilCache<Model, Reconstruction>::radius][component];
  }
};

}  // namespace detail

/// reconstruct<Model,Limiter>: face value at (i,j) extrapolated in direction dir.
///
/// sgn = +1 -> +dir face of (i,j); sgn = -1 -> -dir face. Reconstructs in PRIMITIVE
/// variables if prim == true AND if Model exposes HasPrimitiveVars (positivity of rho and p
/// for Euler); otherwise in conservative variables. The returned state is ALWAYS conservative.
/// The reconstruction policy selects its pointwise algorithm through one explicit protocol;
/// n_ghost is used only to validate the storage envelope.
/// INVARIANT: POINTWISE function, does NOT loop over the grid. POPS_HD.
template <class Model, class Limiter>
POPS_HD inline typename Model::State reconstruct(const Model& model, const ConstArray4& u, int i,
                                                 int j, int dir, Real sgn, const Limiter& lim,
                                                 bool prim) {
  static_assert(
      ReconstructionPolicy<Limiter>,
      "a reconstruction policy must declare positive formal_order/n_ghost metadata and implement "
      "exactly one pointwise protocol");
  static_assert(stencil_envelope_fits_storage<Limiter>,
                "sampled reconstruction offsets exceed Limiter::n_ghost storage capacity");
  if constexpr (HasPrimitiveVars<Model> && !CellValueReconstruction<Limiter>) {
    if (prim) {  // convert the stencil U->P, limit on P, convert back P->U
      using Prim = typename Model::Prim;
      Prim Pf{};
      if constexpr (SlopeReconstruction<Limiter>) {
        const Prim P0 = model.to_primitive(load_state<Model>(u, i, j));
        const Prim Pm =
            model.to_primitive(load_state<Model>(u, dir == 0 ? i - 1 : i, dir == 0 ? j : j - 1));
        const Prim Pp =
            model.to_primitive(load_state<Model>(u, dir == 0 ? i + 1 : i, dir == 0 ? j : j + 1));
        for (int c = 0; c < Model::n_vars; ++c)
          Pf[c] =
              P0[c] + sgn * Real(0.5) * lim.limited_slope(P0[c] - Pm[c], Pp[c] - P0[c]);
      } else if constexpr (StencilReconstruction<Limiter>) {
        const int orientation = (sgn > Real(0)) ? 1 : -1;
        detail::PrimitiveStencilCache<Model, Limiter> cache{};
        for (int offset = Limiter::stencil_min_offset; offset <= Limiter::stencil_max_offset;
             ++offset) {
          const int displacement = orientation * offset;
          const auto state = load_state<Model>(u, dir == 0 ? i + displacement : i,
                                               dir == 0 ? j : j + displacement);
          cache.at(offset) = model.to_primitive(state);
        }
        for (int c = 0; c < Model::n_vars; ++c) {
          const detail::CachedPrimitiveComponentSampler<Model, Limiter> sample{&cache, c};
          Pf[c] = lim.stencil_face_value(sample);
        }
      }
      return model.to_conservative(Pf);
    }
  }
  (void)model;
  (void)prim;
  typename Model::State s = load_state<Model>(u, i, j);
  if constexpr (CellValueReconstruction<Limiter>) {
    for (int c = 0; c < Model::n_vars; ++c)
      s[c] = lim.cell_face_value(s[c]);
  } else if constexpr (SlopeReconstruction<Limiter>) {
    // MUSCL: per-component limited slope (order 2).
    for (int c = 0; c < Model::n_vars; ++c) {
      const Real am = (dir == 0) ? u(i, j, c) - u(i - 1, j, c) : u(i, j, c) - u(i, j - 1, c);
      const Real ap = (dir == 0) ? u(i + 1, j, c) - u(i, j, c) : u(i, j + 1, c) - u(i, j, c);
      s[c] += sgn * Real(0.5) * lim.limited_slope(am, ap);
    }
  } else if constexpr (StencilReconstruction<Limiter>) {
    // Generic sampled reconstruction (sgn<0 reverses the sampler orientation).  The policy, not
    // the operator, chooses how many values to read and at which offsets.
    const int orientation = (sgn > Real(0)) ? 1 : -1;
    for (int c = 0; c < Model::n_vars; ++c) {
      const detail::ConservativeStencilSampler sample{u, i, j, dir, orientation, c};
      s[c] = lim.stencil_face_value(sample);
    }
  }
  return s;
}

/// reconstruct_pp: reconstruct + zhang_shu_scale positivity limiter on the returned state.
///
/// (i, j) is the SOURCE cell of the reconstruction: it is to ITS average that the face state is
/// brought back. pos_floor <= 0 -> strictly identical to reconstruct (short-circuit). POPS_HD.
template <class Model, class Limiter>
POPS_HD inline typename Model::State reconstruct_pp(const Model& model, const ConstArray4& u, int i,
                                                    int j, int dir, Real sgn, const Limiter& lim,
                                                    bool prim, Real pos_floor, int pos_comp) {
  typename Model::State s = reconstruct<Model>(model, u, i, j, dir, sgn, lim, prim);
  zhang_shu_scale<Model>(s, u, i, j, pos_floor, pos_comp);
  return s;
}

namespace detail {
/// require_reconstruction_ghosts<Limiter>: STRUCTURAL ENTRY GUARD of the FV spatial operators.
/// A limiter's reconstruction stencil reads up to Limiter::n_ghost cells BEYOND the valid box: we
/// reconstruct the NEIGHBOR cells i+-1 of each valid cell, which reads i+-2 for a 2-ghost MUSCL
/// (Minmod / VanLeer) and i+-3 for WENO5. If the state does not carry this ghost width, the read
/// runs off the Fab buffer (heap-buffer-overflow, silent UB: negative linear index). We REQUIRE the
/// contract at entry -- CLEAR error rather than an out-of-bounds read -- exactly the rule already
/// applied to ALLOCATION (Limiter::n_ghost) on the AMR side and block_builder (cf. python/system.cpp
/// and PR #22). aux / mask are only read at i+-1 (1 ghost), strictly smaller width: it is the STATE
/// ghosts that size the stencil.
template <class Limiter>
inline void require_reconstruction_ghosts(const MultiFab& U) {
  static_assert(
      ReconstructionPolicy<Limiter>,
      "a reconstruction policy must declare positive formal_order/n_ghost metadata and implement "
      "exactly one pointwise protocol");
  static_assert(stencil_envelope_fits_storage<Limiter>,
                "sampled reconstruction offsets exceed Limiter::n_ghost storage capacity");
  if (U.n_grow() < Limiter::n_ghost)
    throw std::runtime_error(
        "spatial operator: the state must carry at least Limiter::n_ghost ghost layers "
        "(the reconstruction stencil reads i+-Limiter::n_ghost at the edge of the valid box); "
        "allocate the state MultiFab with this number of ghosts.");
}
}  // namespace detail

/// xface_box / yface_box: face boxes normal to x (resp. y) associated with a cell box.
///
/// xface_box(v): nx+1 x ny (i in [lo..hi+1], j in [lo..hi]).
/// yface_box(v): nx x ny+1 (i in [lo..hi], j in [lo..hi+1]).
/// Used to size the MultiFab Fx, Fy received by compute_face_fluxes.
inline Box2D xface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0] + 1, v.hi[1]}};
}
inline Box2D yface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0], v.hi[1] + 1}};
}

namespace detail {
/// FaceFluxXKernel: device kernel for the flux at the radial x face (between i-1 and i).
///
/// Reconstructs the L (cell i-1, +x face) and R (cell i, -x face) states, computes the
/// numerical flux, writes into fx(i,j). Adds the Fickian flux if DiffusiveModel.
/// Named functor (device-clean cross-TU). POPS_HD.
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;
  Real dx;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  FluxEvaluationRecorder failures;
  POPS_HD void operator()(int i, int j, std::uint64_t& failure) const {
    const auto L =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext face = FaceContext::axis_aligned(0);
    const auto evaluation =
        evaluate_numerical_flux_at(nflux, model, L, ax, i - 1, j, Rr, ax, i, j, face);
    failures.record(evaluation, failure);
    const auto F = apply_face_measure(evaluation.checked_density(), face).value;
    for (int c = 0; c < Model::n_vars; ++c)
      fx(i, j, c) = F[c];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int c = 0; c < Model::n_vars; ++c)
        fx(i, j, c) += -nu * (u(i, j, c) - u(i - 1, j, c)) / dx;
    }
    if (evaluation.succeeded())
      for (int c = 0; c < Model::n_vars; ++c)
        failures.record_nonfinite(fx(i, j, c), failure);
  }
};
/// FaceFluxYKernel: device kernel for the flux at the y face (between j-1 and j).
///
/// Analogue of FaceFluxXKernel in the j direction. Named functor. POPS_HD.
template <class Limiter, class NumericalFlux, class Model>
struct FaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  Real dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  FluxEvaluationRecorder failures;
  POPS_HD void operator()(int i, int j, std::uint64_t& failure) const {
    const auto L =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext face = FaceContext::axis_aligned(1);
    const auto evaluation =
        evaluate_numerical_flux_at(nflux, model, L, ax, i, j - 1, Rr, ax, i, j, face);
    failures.record(evaluation, failure);
    const auto F = apply_face_measure(evaluation.checked_density(), face).value;
    for (int c = 0; c < Model::n_vars; ++c)
      fy(i, j, c) = F[c];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int c = 0; c < Model::n_vars; ++c)
        fy(i, j, c) += -nu * (u(i, j, c) - u(i, j - 1, c)) / dy;
    }
    if (evaluation.succeeded())
      for (int c = 0; c < Model::n_vars; ++c)
        failures.record_nonfinite(fy(i, j, c), failure);
  }
};
}  // namespace detail

/// compute_face_fluxes<Limiter,NumericalFlux>: writes the face fluxes BEFORE divergence.
///
/// Fx(i,j) = flux at the face between (i-1,j) and (i,j), i in [lo..hi+1].
/// Fy(i,j) = flux between (i,j-1) and (i,j), j in [lo..hi+1].
/// Brick required by the AMR reflux: assemble_rhs computes -div F directly and discards the face
/// fluxes, but the reflux must see them to correct the coarse-fine interfaces.
/// For a DiffusiveModel, the Fickian flux F_diff = -nu (u_R-u_L)/h is added (its divergence
/// reproduces EXACTLY +nu Lap(u) of assemble_rhs, and stays visible to the reflux).
/// dx=0, dy=0 by default: not read for a non-diffusive model (hyperbolic bit-identical).
//
// compute_face_fluxes: writes the numerical fluxes at the FACES (Fx at faces normal to x,
// Fy at y), BEFORE divergence. This is the brick the AMR reflux needs (it accumulates the
// fine fluxes and subtracts the coarse flux at the coarse-fine interfaces; assemble_rhs
// itself computes -div F directly and discards the face fluxes).
//
// Conventions: Fx(i,j) = flux at the face between cells (i-1,j) and (i,j), i in [lo..hi+1].
// Fy(i,j) = flux between (i,j-1) and (i,j), j in [lo..hi+1]. Same reconstruction (Limiter)
// and numerical flux (NumericalFlux) as assemble_rhs, so
//   r(i,j) = S - (Fx(i+1,j)-Fx(i,j))/dx - (Fy(i,j+1)-Fy(i,j))/dy
// gives back EXACTLY the assemble_rhs residual. Fx, Fy sized by the caller (xface_box/yface_box
// boxes, ncomp = Model::n_vars, 0 ghost). Device-callable.
//
// DIFFUSION on AMR (milestone 4): for a DiffusiveModel, we add the FACE Fickian flux
// F_diff = -nu (u_R - u_L)/h (centered gradient at the face, cell values). Its divergence
// -(Fx(i+1)-Fx(i))/dx gives back EXACTLY +nu Lap(u) of assemble_rhs, but treated as a FLUX:
// the AMR reflux therefore sees it, and the diffusion stays conservative at the coarse-fine
// interfaces (otherwise a direct Laplacian would be ignored by the reflux). dx/dy = step of
// the LEVEL (passed by the caller; 0 by default, not read for a non-diffusive model -> the
// hyperbolic path is strictly bit-identical).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void compute_face_fluxes(const Model& model, const MultiFab& U, const MultiFab& aux, MultiFab& Fx,
                         MultiFab& Fy, Real dx = 0, Real dy = 0, bool recon_prim = false,
                         Real pos_floor = Real(0), Real weno_eps = kWenoEpsilon) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  Limiter lim = configured_reconstruction<Limiter>(weno_eps);
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  FluxEvaluationTracker failures{process_world_flux_collective};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = U.box(li);
    failures.merge(reduce_max_uint64_cell(
        xface_box(v), detail::FaceFluxXKernel<Limiter, NumericalFlux, Model>{
                          model, u, ax, fx, dx, lim, nflux, recon_prim, pos_floor, pos_comp,
                          failures.recorder()}));
    failures.merge(reduce_max_uint64_cell(
        yface_box(v), detail::FaceFluxYKernel<Limiter, NumericalFlux, Model>{
                          model, u, ax, fy, dy, lim, nflux, recon_prim, pos_floor, pos_comp,
                          failures.recorder()}));
  }
  failures.throw_if_failed("compute_face_fluxes");
}

}  // namespace pops
