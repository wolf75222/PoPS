/// @file
/// @brief Cartesian residual R = -div Fhat + S over the cells of a level (method of lines).
///
/// CONTRACT: the "PDE -> ODE system" arrow. The time integrator (time/) only knows R; it is
/// unaware of the geometry and the reconstruction scheme.
///   - assemble_rhs<Limiter,NumericalFlux>: main entry point; residual + optional Fickian term.
///   - assemble_rhs_hll_cached<Limiter>: OPT-IN HLL path with exact reconstructed-trace signal
///     speeds cached once per face; BIT-IDENTICAL to assemble_rhs<Limiter, HLLFlux>.
///
/// Reconstruction (reconstruct_pp) and the structural ghost guard come from face_flux.hpp; the
/// positivity role from positivity.hpp.

#pragma once

#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/primitives/finite.hpp>
#include <pops/numerics/spatial/primitives/face_flux.hpp>  // reconstruct_pp, require_reconstruction_ghosts
#include <pops/numerics/spatial/primitives/positivity.hpp>    // detail::positivity_comp
#include <pops/numerics/spatial/primitives/state_access.hpp>  // load_state, load_aux, DiffusiveModel

#include <stdexcept>
#include <type_traits>
namespace pops {

namespace detail {
inline bool wave_speed_cache_matches(const MultiFab& cache, const MultiFab& state) {
  if (cache.ncomp() != 4 || cache.n_grow() < 1 ||
      cache.box_array().size() != state.box_array().size() ||
      cache.dmap().ranks() != state.dmap().ranks())
    return false;
  for (int box = 0; box < state.box_array().size(); ++box)
    if (cache.box_array()[box] != state.box_array()[box])
      return false;
  return true;
}

/// AssembleRhsKernel<Limiter,NumericalFlux,Model>: device kernel of the central residual of
/// assemble_rhs.
///
/// Computes R(i,j) = S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy (+ Fickian term if DiffusiveModel).
/// Named functor: key point of the AOT native parity (add_compiled_model via external TU).
/// Body bit-identical to the former lambda. POPS_HD.
//
// nvcc does not reliably emit the device kernel of a Model-template extended lambda first
// instantiated from an EXTERNAL TU through the std::function / host-lambda nesting of block_builder:
// the test passes on Serial and under compute-sanitizer but segfaults at runtime on Cuda (Heisenbug).
// A device-callable class does not have these instantiation-context restrictions. Body IDENTICAL to
// the former lambda -> residual BIT-IDENTICAL to add_block on CPU (and, targeted, on device).
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  FluxEvaluationRecorder failures;
  POPS_HD void operator()(int i, int j, std::uint64_t& failure) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);

    // x faces: reconstruction of the states on either side of each face
    const auto Lxm =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp =
        reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp =
        reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext xface = FaceContext::axis_aligned(0);
    const auto evaluation_xm =
        evaluate_numerical_flux_at(nflux, model, Lxm, ax, i - 1, j, Rxm, ax, i, j, xface);
    const auto evaluation_xp =
        evaluate_numerical_flux_at(nflux, model, Lxp, ax, i, j, Rxp, ax, i + 1, j, xface);
    failures.record(evaluation_xm, failure);
    failures.record(evaluation_xp, failure);
    const auto Fxm = apply_face_measure(evaluation_xm.checked_density(), xface).value;
    const auto Fxp = apply_face_measure(evaluation_xp.checked_density(), xface).value;

    // y faces
    const auto Lym =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp =
        reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp =
        reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext yface = FaceContext::axis_aligned(1);
    const auto evaluation_ym =
        evaluate_numerical_flux_at(nflux, model, Lym, ax, i, j - 1, Rym, ax, i, j, yface);
    const auto evaluation_yp =
        evaluate_numerical_flux_at(nflux, model, Lyp, ax, i, j, Ryp, ax, i, j + 1, yface);
    failures.record(evaluation_ym, failure);
    failures.record(evaluation_yp, failure);
    const auto Fym = apply_face_measure(evaluation_ym.checked_density(), yface).value;
    const auto Fyp = apply_face_measure(evaluation_yp.checked_density(), yface).value;

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;

    // Parabolic (Fickian) term: +nu Lap(U), 5-point centered differences.
    // Guarded by DiffusiveModel: no effect (nor codegen) for a non-diffusive model.
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      const Real idx2 = Real(1) / (dx * dx), idy2 = Real(1) / (dy * dy);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) += nu * ((u(i + 1, j, c) - 2 * u(i, j, c) + u(i - 1, j, c)) * idx2 +
                            (u(i, j + 1, c) - 2 * u(i, j, c) + u(i, j - 1, c)) * idy2);
    }
    if (evaluation_xm.succeeded() && evaluation_xp.succeeded() && evaluation_ym.succeeded() &&
        evaluation_yp.succeeded())
      for (int c = 0; c < Model::n_vars; ++c)
        failures.record_nonfinite(r(i, j, c), failure);
  }
};
}  // namespace detail

/// assemble_rhs<Limiter,NumericalFlux>: residual R = -div Fhat + S over all boxes.
///
/// Main entry point of the Cartesian spatial operator. The limiter (reconstruction) AND the
/// numerical flux are template parameters chosen at compile time (default: NoSlope + RusanovFlux).
/// recon_prim = true enables reconstruction in primitive variables if the model exposes
/// HasPrimitiveVars. For the diffusive term, see DiffusiveModel.
/// INVARIANT: the operator does not modify U, aux -- it only writes R. No ghost fill.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux, const Geometry& geom,
                  MultiFab& R, bool recon_prim = false, Real pos_floor = Real(0),
                  Real weno_eps = kWenoEpsilon) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const Real dx = geom.dx(), dy = geom.dy();
  // ADC-645: the per-block WENO-Z regulariser (only Weno5 carries an eps member; the default value
  // IS kWenoEpsilon, so every existing call site is bit-identical).
  Limiter lim = configured_reconstruction<Limiter>(weno_eps);
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  FluxEvaluationTracker failures{process_world_flux_collective};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    failures.merge(reduce_max_uint64_cell(
        v, detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>{
               model, u, ax, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp,
               failures.recorder()}));
  }
  failures.throw_if_failed("assemble_rhs");
}

namespace detail {
/// Exact HLL signal speeds for the x-normal face indexed by (i,j).  The cache stores the same
/// reconstructed traces, provider samples and canonical face orientation consumed by HLLFlux; it
/// therefore remains exact for first-order, MUSCL and WENO reconstructions.
template <class Limiter, class Model>
struct HllFaceSpeedXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 ws;
  Limiter lim;
  bool recon_prim;
  Real pos_floor;
  int pos_comp;

  POPS_HD void operator()(int i, int j) const {
    const auto left =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto right =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext face = FaceContext::axis_aligned(0);
    const PhysicalFluxView<Model> physical{model};
    Real lower, upper;
    hll_speeds(physical, make_face_trace_at<Model>(left, ax, i - 1, j),
               make_face_trace_at<Model>(right, ax, i, j), face, lower, upper);
    ws(i, j, 0) = lower;
    ws(i, j, 1) = upper;
  }
};

/// Exact HLL signal speeds for the y-normal face indexed by (i,j).  Components 2/3 are disjoint
/// from the x-face lanes, so both face families share the existing four-component scratch.
template <class Limiter, class Model>
struct HllFaceSpeedYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 ws;
  Limiter lim;
  bool recon_prim;
  Real pos_floor;
  int pos_comp;

  POPS_HD void operator()(int i, int j) const {
    const auto left =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto right =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext face = FaceContext::axis_aligned(1);
    const PhysicalFluxView<Model> physical{model};
    Real lower, upper;
    hll_speeds(physical, make_face_trace_at<Model>(left, ax, i, j - 1),
               make_face_trace_at<Model>(right, ax, i, j), face, lower, upper);
    ws(i, j, 2) = lower;
    ws(i, j, 3) = upper;
  }
};

template <class Limiter, class Model>
inline void fill_hll_face_speed_cache(const Model& model, const MultiFab& U, const MultiFab& aux,
                                      MultiFab& cache, const Limiter& limiter, bool recon_prim,
                                      Real pos_floor, int pos_comp) {
  for (int local = 0; local < U.local_size(); ++local) {
    const ConstArray4 state = U.fab(local).const_array();
    const ConstArray4 providers = aux.fab(local).const_array();
    Array4 speeds = cache.fab(local).array();
    for_each_cell(xface_box(U.box(local)),
                  HllFaceSpeedXKernel<Limiter, Model>{model, state, providers, speeds, limiter,
                                                      recon_prim, pos_floor, pos_comp});
    for_each_cell(yface_box(U.box(local)),
                  HllFaceSpeedYKernel<Limiter, Model>{model, state, providers, speeds, limiter,
                                                      recon_prim, pos_floor, pos_comp});
  }
}

/// AssembleRhsHllCachedKernel: residual R = -div Fhat + S for HLL with exact per-face signal speeds
/// pre-computed from the reconstructed traces. Reconstruction and numerical flux are identical to
/// AssembleRhsKernel<.., HLLFlux>; only duplicate calls to model.wave_speeds are removed.
template <class Limiter, class Model>
struct AssembleRhsHllCachedKernel {
  Model model;
  ConstArray4 u, ax, ws;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< Density role component (resolved by the host caller)
  FluxEvaluationRecorder failures;
  POPS_HD void operator()(int i, int j, std::uint64_t& failure) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);

    // x faces: reconstruction of the states on both sides of each face
    const auto Lxm =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp =
        reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp =
        reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const Real sLxm = ws(i, j, 0), sRxm = ws(i, j, 1);
    const Real sLxp = ws(i + 1, j, 0), sRxp = ws(i + 1, j, 1);
    const FaceContext xface = FaceContext::axis_aligned(0);
    const PhysicalFluxView<Model> physical{model};
    const auto evaluation_xm =
        hll_flux_with_speeds(physical, make_face_trace_at<Model>(Lxm, ax, i - 1, j),
                             make_face_trace_at<Model>(Rxm, ax, i, j), xface, sLxm, sRxm);
    const auto evaluation_xp =
        hll_flux_with_speeds(physical, make_face_trace_at<Model>(Lxp, ax, i, j),
                             make_face_trace_at<Model>(Rxp, ax, i + 1, j), xface, sLxp, sRxp);
    failures.record(evaluation_xm, failure);
    failures.record(evaluation_xp, failure);
    const auto Fxm = apply_face_measure(evaluation_xm.checked_density(), xface).value;
    const auto Fxp = apply_face_measure(evaluation_xp.checked_density(), xface).value;

    // y faces (components 2/3 hold the exact interval of the indexed y-normal face)
    const auto Lym =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp =
        reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp =
        reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const Real sLym = ws(i, j, 2), sRym = ws(i, j, 3);
    const Real sLyp = ws(i, j + 1, 2), sRyp = ws(i, j + 1, 3);
    const FaceContext yface = FaceContext::axis_aligned(1);
    const auto evaluation_ym =
        hll_flux_with_speeds(physical, make_face_trace_at<Model>(Lym, ax, i, j - 1),
                             make_face_trace_at<Model>(Rym, ax, i, j), yface, sLym, sRym);
    const auto evaluation_yp =
        hll_flux_with_speeds(physical, make_face_trace_at<Model>(Lyp, ax, i, j),
                             make_face_trace_at<Model>(Ryp, ax, i, j + 1), yface, sLyp, sRyp);
    failures.record(evaluation_ym, failure);
    failures.record(evaluation_yp, failure);
    const auto Fym = apply_face_measure(evaluation_ym.checked_density(), yface).value;
    const auto Fyp = apply_face_measure(evaluation_yp.checked_density(), yface).value;

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;

    // Parabolic (Fickian) term: identical to AssembleRhsKernel, guarded by DiffusiveModel.
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      const Real idx2 = Real(1) / (dx * dx), idy2 = Real(1) / (dy * dy);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) += nu * ((u(i + 1, j, c) - 2 * u(i, j, c) + u(i - 1, j, c)) * idx2 +
                            (u(i, j + 1, c) - 2 * u(i, j, c) + u(i, j - 1, c)) * idy2);
    }
    if (evaluation_xm.succeeded() && evaluation_xp.succeeded() && evaluation_ym.succeeded() &&
        evaluation_yp.succeeded())
      for (int c = 0; c < Model::n_vars; ++c)
        failures.record_nonfinite(r(i, j, c), failure);
  }
};

template <class Limiter, class Model>
struct FaceFluxHllCachedXKernel {
  Model model;
  ConstArray4 u, ax, ws;
  Array4 flux;
  Real dx;
  Limiter lim;
  bool recon_prim;
  Real pos_floor;
  int pos_comp;
  FluxEvaluationRecorder failures;

  POPS_HD void operator()(int i, int j, std::uint64_t& failure) const {
    const auto left =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto right =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const Real speed_left = ws(i, j, 0), speed_right = ws(i, j, 1);
    const FaceContext face = FaceContext::axis_aligned(0);
    const PhysicalFluxView<Model> physical{model};
    const auto evaluation =
        hll_flux_with_speeds(physical, make_face_trace_at<Model>(left, ax, i - 1, j),
                             make_face_trace_at<Model>(right, ax, i, j), face, speed_left,
                             speed_right);
    failures.record(evaluation, failure);
    const auto value = apply_face_measure(evaluation.checked_density(), face).value;
    for (int component = 0; component < Model::n_vars; ++component)
      flux(i, j, component) = value[component];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int component = 0; component < Model::n_vars; ++component)
        flux(i, j, component) +=
            -nu * (u(i, j, component) - u(i - 1, j, component)) / dx;
    }
    if (evaluation.succeeded())
      for (int component = 0; component < Model::n_vars; ++component)
        failures.record_nonfinite(flux(i, j, component), failure);
  }
};

template <class Limiter, class Model>
struct FaceFluxHllCachedYKernel {
  Model model;
  ConstArray4 u, ax, ws;
  Array4 flux;
  Real dy;
  Limiter lim;
  bool recon_prim;
  Real pos_floor;
  int pos_comp;
  FluxEvaluationRecorder failures;

  POPS_HD void operator()(int i, int j, std::uint64_t& failure) const {
    const auto left =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto right =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const Real speed_left = ws(i, j, 2), speed_right = ws(i, j, 3);
    const FaceContext face = FaceContext::axis_aligned(1);
    const PhysicalFluxView<Model> physical{model};
    const auto evaluation =
        hll_flux_with_speeds(physical, make_face_trace_at<Model>(left, ax, i, j - 1),
                             make_face_trace_at<Model>(right, ax, i, j), face, speed_left,
                             speed_right);
    failures.record(evaluation, failure);
    const auto value = apply_face_measure(evaluation.checked_density(), face).value;
    for (int component = 0; component < Model::n_vars; ++component)
      flux(i, j, component) = value[component];
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      for (int component = 0; component < Model::n_vars; ++component)
        flux(i, j, component) +=
            -nu * (u(i, j, component) - u(i, j - 1, component)) / dy;
    }
    if (evaluation.succeeded())
      for (int component = 0; component < Model::n_vars; ++component)
        failures.record_nonfinite(flux(i, j, component), failure);
  }
};
}  // namespace detail

/// assemble_rhs_hll_cached<Limiter>: residual R = -div Fhat + S at the HLL flux, with exact signal
/// speeds pre-computed once for every reconstructed face trace pair (OPT-IN). The residual then
/// consumes those intervals without recalling model.wave_speeds for shared faces.
/// @p cache must have the layout of @p U, 4 components, >= 1 ghost (re-allocated by the caller).
/// BIT-IDENTICAL to assemble_rhs<Limiter, HLLFlux> for every supported Limiter. The model MUST expose
/// wave_speeds (guaranteed by the HLL dispatch).
template <class Limiter = NoSlope, class Model>
void assemble_rhs_hll_cached(const Model& model, const MultiFab& U, const MultiFab& aux,
                             const Geometry& geom, MultiFab& R, MultiFab& cache,
                             bool recon_prim = false, Real pos_floor = Real(0),
                             Real weno_eps = kWenoEpsilon) {
  detail::require_reconstruction_ghosts<Limiter>(U);
  if (!detail::wave_speed_cache_matches(cache, U))
    throw std::invalid_argument(
        "assemble_rhs_hll_cached requires an exact four-component face-speed cache");
  const Real dx = geom.dx(), dy = geom.dy();
  Limiter lim = configured_reconstruction<Limiter>(weno_eps);
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  detail::fill_hll_face_speed_cache(model, U, aux, cache, lim, recon_prim, pos_floor, pos_comp);
  FluxEvaluationTracker failures{process_world_flux_collective};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 ws = cache.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    failures.merge(reduce_max_uint64_cell(
        v, detail::AssembleRhsHllCachedKernel<Limiter, Model>{
               model, u, ax, ws, r, dx, dy, lim, recon_prim, pos_floor, pos_comp,
               failures.recorder()}));
  }
  failures.throw_if_failed("assemble_rhs_hll_cached");
}

/// Materialise the exact cached-HLL face flux used by AMR reflux and prepared-interface omission.
/// Keeping this twin next to assemble_rhs_hll_cached prevents a cached residual from being paired
/// with an independently evaluated, uncached conservation register.
template <class Limiter = NoSlope, class Model>
void compute_face_fluxes_hll_cached(const Model& model, const MultiFab& U, const MultiFab& aux,
                                    MultiFab& Fx, MultiFab& Fy, MultiFab& cache, Real dx = Real(0),
                                    Real dy = Real(0), bool recon_prim = false,
                                    Real pos_floor = Real(0), Real weno_eps = kWenoEpsilon) {
  detail::require_reconstruction_ghosts<Limiter>(U);
  if (!detail::wave_speed_cache_matches(cache, U))
    cache = MultiFab(U.box_array(), U.dmap(), 4, 1);
  Limiter limiter = configured_reconstruction<Limiter>(weno_eps);
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  detail::fill_hll_face_speed_cache(model, U, aux, cache, limiter, recon_prim, pos_floor, pos_comp);
  FluxEvaluationTracker failures{process_world_flux_collective};
  for (int local = 0; local < U.local_size(); ++local) {
    const ConstArray4 state = U.fab(local).const_array();
    const ConstArray4 providers = aux.fab(local).const_array();
    const ConstArray4 speeds = cache.fab(local).const_array();
    failures.merge(reduce_max_uint64_cell(
        xface_box(U.box(local)), detail::FaceFluxHllCachedXKernel<Limiter, Model>{
                                      model, state, providers, speeds, Fx.fab(local).array(), dx,
                                      limiter, recon_prim, pos_floor, pos_comp,
                                      failures.recorder()}));
    failures.merge(reduce_max_uint64_cell(
        yface_box(U.box(local)), detail::FaceFluxHllCachedYKernel<Limiter, Model>{
                                      model, state, providers, speeds, Fy.fab(local).array(), dy,
                                      limiter, recon_prim, pos_floor, pos_comp,
                                      failures.recorder()}));
  }
  failures.throw_if_failed("compute_face_fluxes_hll_cached");
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void compute_face_fluxes_with_optional_hll_cache(
    const Model& model, const MultiFab& U, const MultiFab& aux, MultiFab& Fx, MultiFab& Fy,
    MultiFab* wave_speed_cache, Real dx = Real(0), Real dy = Real(0), bool recon_prim = false,
    Real pos_floor = Real(0), Real weno_eps = kWenoEpsilon) {
  if constexpr (std::is_same_v<NumericalFlux, HLLFlux>) {
    if (wave_speed_cache != nullptr) {
      compute_face_fluxes_hll_cached<Limiter>(model, U, aux, Fx, Fy, *wave_speed_cache, dx, dy,
                                              recon_prim, pos_floor, weno_eps);
      return;
    }
  }
  compute_face_fluxes<Limiter, NumericalFlux>(model, U, aux, Fx, Fy, dx, dy, recon_prim,
                                              pos_floor, weno_eps);
}

}  // namespace pops
