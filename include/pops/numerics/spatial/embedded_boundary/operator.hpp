#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/eb/cut_fraction.hpp>  // detail::cut_fraction, CutFraction (PR1)
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/primitives/finite.hpp>
#include <pops/numerics/spatial/embedded_boundary/domain.hpp>  // detail::DiscDomain (level-set domain; numerics, not runtime)
#include <pops/numerics/spatial_operator.hpp>  // reconstruct<>, load_state/load_aux, *_face_box (REUSED verbatim)

#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// @brief Centre-sampled embedded-boundary transport: R = -div F + S on the active cells of a
///        signed level set, with a locally estimated and clamped inverse volume factor.
///
/// HONEST GEOMETRIC CONTRACT. Cell activity is `level_set(cell_center) < 0`. A Cartesian face is
/// open exactly when both adjacent cell centres are active and closed otherwise, so the current face
/// measure is binary (0 or 1), not a continuous geometric aperture. For an active cell, `kappa` is a
/// five-sample algebraic estimate built from the centre and four cardinal neighbours, then clamped by
/// `kappa_min`; it is not an exact cut-cell area. Consequently this operator does not claim a generic
/// second-order embedded-boundary discretisation. It is a conservative no-penetration transport
/// policy over the centre-sampled active set with an approximate small-cell volume correction.
///
/// CONSERVATIVE EB FORM (cell (i, j), volume kappa dx dy):
///   kappa_eff dx dy d_t U = - [ alpha_xp Fx_{i+1} - alpha_xm Fx_i ] dy
///                           - [ alpha_yp Fy_{j+1} - alpha_ym Fy_i ] dx
///                       - alpha_wall |wall| F_wall                       (immersed WALL term)
///                       + kappa dx dy S
/// that is, after dividing by kappa dx dy (with kappa CLAMPED, cf. small-cell stability):
///   R = S - (1/kappa) [ (alpha_xp Fx_{i+1} - alpha_xm Fx_i) / dx
///                     + (alpha_yp Fy_{j+1} - alpha_ym Fy_i) / dy ]
///         - (1/kappa) (alpha_wall |wall| / (dx dy)) F_wall
/// The immersed WALL flux is a NO-PENETRATION flux (zero-normal-flux): F_wall = 0. It is the
/// FV counterpart of the conducting wall (the elliptic side applies Dirichlet; the transport applies a solid
/// wall at the SAME geometric boundary). The wall term is therefore IDENTICALLY ZERO; it is written
/// explicitly (and kept at zero) so the contract stays readable and so that a future nonzero wall flux
/// (injection, slip) has its single attachment point.
///
/// MASS CONSERVATION. On two neighboring active cells, the shared binary face measure is one on
/// both sides, so its numerical flux telescopes in the weighted sum using the same `kappa_eff` as the
/// residual. A face whose neighbor cell is INACTIVE is CLOSED (alpha_f = 0)
/// AND the wall flux is zero -> no mass crosses the immersed boundary. The total mass over
/// the active cells is therefore closed to transport to floating-point accuracy.
///
/// SMALL-CELL STABILITY (small-cell problem). The 1/kappa factor amplifies the residual when kappa
/// becomes small on the r0/r1 shear layer; at a FIXED time step dt calibrated on full cells, an
/// unbounded amplification blows up (Inf/NaN) the explicit step on a strongly cut cell. Two STACKED
/// complementary guards:
///   1. The PR1 primitive floor: cut_distance clamps each half-face at theta >= 1e-3 (anti-division
///      guard INHERITED from the elliptic wall). kappa = product of half-face averages can therefore
///      never equal EXACTLY 0 -> no strict division by zero. BUT the induced lower bound is
///      ~ (1e-3)^2 / 4 ~ 2.5e-7: 1/kappa can reach ~4e6, which is enough to overflow the fixed step.
///   2. RETAINED SCHEME = VOLUME CLAMP (this header): kappa_eff = max(kappa, kappa_min), kappa_min by
///      default 1e-2 (1% of the full volume). A SCHEME-LEVEL guard, INDEPENDENT of the elliptic floor:
///      it bounds the 1/kappa amplification to 1/kappa_min = 100, a value CALIBRATED so the fixed explicit
///      step stays stable whatever the degree of cut. This is the simplest and most robust implicit
///      "volume merging" for a FIXED step, at the cost of a slight LOCAL NON-conservation on the
///      most-cut cells (effective volume > real volume).
/// The GLOBAL mass stays conserved TO MACHINE PRECISION because the clamp acts only on the DENOMINATOR (volume),
/// NOT on the face fluxes (numerator): the telescoping sum of fluxes is unchanged (the discrete mass
/// consistent with the scheme uses the SAME kappa_eff, cf. conservation test). Documented alternative
/// (outside PR2): flux redistribution (AMReX-EB's flux redistribution) spreads the excess
/// divergence of small cells onto the full neighbors -> exact LOCAL conservation but a non-local stencil;
/// the clamp suffices for the target (calibrated fixed step, smooth MMS, global conservation).
///
/// NAMED FUNCTORS (and not extended lambdas), like spatial_operator.hpp / _polar.hpp (#64/#97):
/// robust device emission when the Model-template kernel is instantiated cross-TU. The RECONSTRUCTION and the
/// numerical FLUX are REUSED verbatim from the Cartesian operator (reconstruct<>, RusanovFlux).
/// The level set is passed BY VALUE (POPS_HD callable, e.g. captured detail::DiscDomain): device-safe.
///
/// INVARIANT: this header is PURELY ADDITIVE and OPT-IN. The Cartesian operator (assemble_rhs) and the
/// T2 mask path (assemble_rhs_masked) stay STRICTLY UNTOUCHED; a run without an embedded boundary is
/// bit-identical. assemble_rhs_eb is called only on explicit opt-in (geometry = cutcell).

namespace pops {

namespace detail {

// EB thresholds kEbFaceOpenEps / kEbKappaMin / kEbCutFractionFloor come from
// numerical_defaults.hpp (via cut_fraction.hpp) -- single source, ADC-643.

/// Device-safe adapter wrapping a DiscDomain as the Real(Real, Real) callable expected by cut_fraction
/// and the EB operator. Since ADC-327 DiscDomain is itself callable (operator() forwards to level_set),
/// this is now a thin compat shim kept for the existing call sites. NAMED FUNCTOR (captures the
/// DiscDomain BY VALUE: three doubles, device-safe), not an extended lambda. operator() forwards
/// EXACTLY DiscDomain::level_set -> same cut geometry as the elliptic wall (bit consistency).
struct DiscLevelSet {
  DiscDomain disc;
  POPS_HD Real operator()(Real x, Real y) const { return disc.level_set(x, y); }
};

/// Builds the disc level set callable from a DiscDomain (sugar: disc_level_set(d)).
POPS_HD inline DiscLevelSet disc_level_set(const DiscDomain& d) {
  return DiscLevelSet{d};
}

/// Activity indicator (cell centre in the authored domain, ls < 0) from a callable level set.
template <class LevelSet>
POPS_HD inline bool eb_cell_active(const LevelSet& ls, Real xc, Real yc) {
  return ls(xc, yc) < Real(0);
}

/// Aperture of ONE face between the active cell (xc, yc) and its neighbor at (xn, yn), step h.
///
/// FV EB convention:
///   - INACTIVE neighbor (ls(xn,yn) >= 0): the face touches the immersed wall -> CLOSED (alpha = 0,
///     no-penetration). This is the generalization of the T2 0/1 gate: the inactive side closes the face.
///   - ACTIVE neighbor (ln < 0): the aperture reuses VERBATIM the shared primitive cut_distance
///     (hence bit-consistent with the elliptic wall), alpha = cut_distance(lc, ln, h) / h. But for an
///     active neighbor cut_distance takes the "interior neighbor" branch and returns h -> alpha = 1 EXACTLY,
///     far and near the edge alike: the shared face of two active cells is always FULL. The face
///     apertures are therefore BINARY {0, 1}; it is the VOLUME FRACTION kappa in (0, 1], and
///     not the aperture of internal faces, that carries the cut geometry (cf. NOTE alpha_f of @file).
/// SYMMETRY (key to conservation): cut_distance(lc, ln, h) depends only on (lc, ln); the shared
/// face seen from cell i (center lc, neighbor ln) and seen from cell i+1 (center ln, neighbor lc)
/// gives the SAME aperture as soon as both are active (ln < 0: the "interior neighbor" branch returns
/// h on both sides -> alpha = 1). The internal active/active boundary is therefore treated SYMMETRICALLY.
POPS_HD inline Real eb_face_aperture(Real lc, Real ln, Real h) {
  if (ln >= Real(0))
    return Real(0);                    // inactive neighbor: closed face (wall, no-penetration)
  return cut_distance(lc, ln, h) / h;  // active neighbor: linear aperture (== elliptic wall)
}

/// Geometry metrics evaluated from a device-callable level set. This preserves the low-level EB API;
/// the production System path uses PreparedEbMetricsView below and never enters this evaluator.
template <class LevelSet>
struct CallableEbMetrics {
  LevelSet level_set;
  Geometry geometry;
  Real dx, dy, kappa_min, cut_theta_min;

  POPS_HD Real value(int i, int j) const {
    return level_set(geometry.x_cell(i), geometry.y_cell(j));
  }
  POPS_HD bool active(int i, int j) const { return value(i, j) < Real(0); }
  POPS_HD Real x_face_aperture(int i, int j) const {
    const Real left = value(i - 1, j), right = value(i, j);
    if (left < Real(0) && right < Real(0))
      return eb_face_aperture(left, right, dx);
    if (right < Real(0))
      return eb_face_aperture(right, left, dx);
    if (left < Real(0))
      return eb_face_aperture(left, right, dx);
    return Real(0);
  }
  POPS_HD Real y_face_aperture(int i, int j) const {
    const Real lower = value(i, j - 1), upper = value(i, j);
    if (lower < Real(0) && upper < Real(0))
      return eb_face_aperture(lower, upper, dy);
    if (upper < Real(0))
      return eb_face_aperture(upper, lower, dy);
    if (lower < Real(0))
      return eb_face_aperture(lower, upper, dy);
    return Real(0);
  }
  POPS_HD Real inverse_kappa(int i, int j) const {
    const Real center = value(i, j);
    const CutFraction fraction = cut_fraction_from_samples(
        center, value(i - 1, j), value(i + 1, j), value(i, j - 1), value(i, j + 1), dx,
        dy, cut_theta_min);
    const Real effective = fraction.kappa > kappa_min ? fraction.kappa : kappa_min;
    return Real(1) / effective;
  }
};

/// Static per-patch metrics prepared once when System installs its level set. The face aperture in
/// the current no-penetration scheme is exactly one only between two active centers and zero
/// otherwise; inverse_kappa carries the full signed-level-set cut geometry computed at installation.
struct PreparedEbMetricsView {
  ConstArray4 active_mask;
  ConstArray4 inverse_volume_fraction;

  POPS_HD bool active(int i, int j) const { return active_mask(i, j, 0) > Real(0); }
  POPS_HD Real x_face_aperture(int i, int j) const {
    return active(i - 1, j) && active(i, j) ? Real(1) : Real(0);
  }
  POPS_HD Real y_face_aperture(int i, int j) const {
    return active(i, j - 1) && active(i, j) ? Real(1) : Real(0);
  }
  POPS_HD Real inverse_kappa(int i, int j) const {
    return inverse_volume_fraction(i, j, 0);
  }
};

/// FACE FLUX kernel for x (dir 0) of the EB transport: numerical flux at the face between (i-1, j) and
/// (i, j), WEIGHTED by the aperture alpha_x of that face. We store alpha_x * Fx so the EB divergence
/// is a simple difference (like the r weighting of the polar operator). NAMED FUNCTOR (device-clean).
///
/// The aperture of face i is computed FROM THE SIDE of the active cell: if (i, j) is active, we take
/// the aperture seen from (i, j) toward its neighbor (i-1); if (i, j) is inactive but (i-1, j) is active, we
/// take the aperture seen from (i-1) toward (i). If BOTH are inactive, the face is outside the active domain
/// (alpha = 0). This symmetry guarantees the uniqueness of alpha on the shared face (conservation).
template <class Limiter, class NumericalFlux, class Model, class GeometryMetrics>
struct EbFaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;  // output: alpha_x * Fx at the face between i-1 and i (ncomp components)
  GeometryMetrics metrics;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  Real face_open_eps = kEbFaceOpenEps;  ///< ADC-615: closed-face aperture threshold (default 1e-6).
  POPS_HD void operator()(int i, int j) const {
    const Real alpha = metrics.x_face_aperture(i, j);
    if (alpha < face_open_eps) {  // closed face (immersed wall): zero normal flux
      for (int c = 0; c < Model::n_vars; ++c)
        fx(i, j, c) = Real(0);
      return;
    }
    const auto L =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext face = FaceContext::axis_aligned(0, alpha);
    const auto evaluation =
        evaluate_numerical_flux_at(nflux, model, L, ax, i - 1, j, Rr, ax, i, j, face);
    const auto F = apply_face_measure(evaluation.checked_density(), face).value;
    for (int c = 0; c < Model::n_vars; ++c)
      fx(i, j, c) = F[c];
  }
};

/// FACE FLUX kernel for y (dir 1) of the EB transport: analogue of EbFaceFluxXKernel in j. Stores
/// alpha_y * Fy at the face between (i, j-1) and (i, j). NAMED FUNCTOR (device-clean).
template <class Limiter, class NumericalFlux, class Model, class GeometryMetrics>
struct EbFaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  GeometryMetrics metrics;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  Real face_open_eps = kEbFaceOpenEps;  ///< ADC-615: closed-face aperture threshold (default 1e-6).
  POPS_HD void operator()(int i, int j) const {
    const Real alpha = metrics.y_face_aperture(i, j);
    if (alpha < face_open_eps) {
      for (int c = 0; c < Model::n_vars; ++c)
        fy(i, j, c) = Real(0);
      return;
    }
    const auto L =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const FaceContext face = FaceContext::axis_aligned(1, alpha);
    const auto evaluation =
        evaluate_numerical_flux_at(nflux, model, L, ax, i, j - 1, Rr, ax, i, j, face);
    const auto F = apply_face_measure(evaluation.checked_density(), face).value;
    for (int c = 0; c < Model::n_vars; ++c)
      fy(i, j, c) = F[c];
  }
};

/// Kernel assembling the EB residual at cell (i, j):
///   INACTIVE cell -> residual 0 (not advanced, like T2);
///   ACTIVE cell -> R = S - (1/kappa_eff) [ (fx_{i+1} - fx_i)/dx + (fy_{j+1} - fy_i)/dy ] - wall_term.
/// fx/fy ALREADY contain alpha_f * F (produced by the face kernels). kappa_eff = max(kappa, min)
/// (small-cell clamp). The immersed WALL term is a no-penetration F_wall = 0 -> zero, written
/// explicitly (kept at zero) as a single attachment point. NAMED FUNCTOR (device-clean).
template <class Model, class GeometryMetrics>
struct EbAssembleRhsKernel {
  Model model;
  ConstArray4 u, ax, fx, fy;  // state, aux, x flux weighted by alpha, y flux weighted by alpha
  Array4 r;                   // output: residual
  Real dx, dy;
  GeometryMetrics metrics;
  POPS_HD void operator()(int i, int j) const {
    if (!metrics.active(i, j)) {
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) = Real(0);
      return;
    }
    const Real inv_kappa = metrics.inverse_kappa(i, j);

    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const auto S = model.source(load_state<Model>(u, i, j), Ac);

    // Immersed WALL flux (no-penetration): F_wall = 0 -> zero term. Single attachment point for a
    // future nonzero wall flux. Stays at 0 in PR2 (solid wall, like the elliptic Dirichlet wall).
    constexpr Real wall_flux = Real(0);

    for (int c = 0; c < Model::n_vars; ++c) {
      const Real div_x = (fx(i + 1, j, c) - fx(i, j, c)) / dx;  // discrete d_x(alpha Fx)
      const Real div_y = (fy(i, j + 1, c) - fy(i, j, c)) / dy;  // discrete d_y(alpha Fy)
      // TERM-BY-TERM accumulation (and not inv_kappa*(div_x + div_y)): when kappa_eff = 1 and all
      // alpha = 1 (case WITHOUT cut), inv_kappa = 1 and each inv_kappa*div_* = div_* by IEEE identity
      // (x*1.0 == x), so r = S - div_x - div_y - 0 reproduces BIT FOR BIT the Cartesian operator
      // (S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy), term by term. A grouping (div_x + div_y) would break
      // floating-point associativity and the bit-identity of the default path.
      r(i, j, c) = S[c] - inv_kappa * div_x - inv_kappa * div_y - inv_kappa * wall_flux;
    }
  }
};

template <class LevelSet>
struct CallableEbMetricsProvider {
  CallableEbMetrics<LevelSet> metrics;
  CallableEbMetrics<LevelSet> local(int) const { return metrics; }
};

struct PreparedEbMetricsProvider {
  const MultiFab* active_mask;
  const MultiFab* inverse_volume_fraction;
  PreparedEbMetricsView local(int local_index) const {
    return PreparedEbMetricsView{active_mask->fab(local_index).const_array(),
                                 inverse_volume_fraction->fab(local_index).const_array()};
  }
};

struct KeepEbFaceFluxes {
  void operator()(MultiFab&, MultiFab&) const {}
};

/// The one EB operator implementation. Metric providers differ only in setup-time ownership:
/// low-level callers evaluate a callable level set, while System supplies static device views that
/// were prepared transactionally at geometry installation.
template <class Limiter, class NumericalFlux, class Model, class MetricsProvider,
          class FaceFluxTransform = KeepEbFaceFluxes>
void assemble_rhs_eb_with_metrics(const Model& model, const MultiFab& U, const MultiFab& aux,
                                  const MetricsProvider& provider, const Geometry& geom,
                                  MultiFab& R, bool recon_prim, Real pos_floor,
                                  Real face_open_eps, Real weno_eps,
                                  FaceFluxTransform transform = {}) {
  require_reconstruction_ghosts<Limiter>(U);
  const Real dx = geom.dx(), dy = geom.dy();
  Limiter lim{};
  if constexpr (std::is_same_v<Limiter, Weno5>)
    lim.eps = weno_eps;
  const NumericalFlux nflux{};
  const int pos_comp = positivity_comp<Model>(pos_floor);
  std::vector<Box2D> xfaces, yfaces;
  xfaces.reserve(U.box_array().size());
  yfaces.reserve(U.box_array().size());
  for (const Box2D& b : U.box_array().boxes()) {
    xfaces.push_back(xface_box(b));
    yfaces.push_back(yface_box(b));
  }
  MultiFab Fx(BoxArray(std::move(xfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab Fy(BoxArray(std::move(yfaces)), U.dmap(), Model::n_vars, 0);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = R.box(li);
    auto metrics = provider.local(li);
    for_each_cell(
        xface_box(v),
        EbFaceFluxXKernel<Limiter, NumericalFlux, Model, decltype(metrics)>{
            model, u, ax, fx, metrics, lim, nflux, recon_prim, pos_floor, pos_comp,
            face_open_eps});
    for_each_cell(
        yface_box(v),
        EbFaceFluxYKernel<Limiter, NumericalFlux, Model, decltype(metrics)>{
            model, u, ax, fy, metrics, lim, nflux, recon_prim, pos_floor, pos_comp,
            face_open_eps});
  }
  // A shared-interface Program owns selected outer faces in its pair scheduler.  The transform is a
  // zero-cost no-op for ordinary callers and a small host-side face filter for that prepared route;
  // the EB flux and metric kernels remain exactly the same compiled Kokkos kernels.
  transform(Fx, Fy);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 fx = Fx.fab(li).const_array();
    const ConstArray4 fy = Fy.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    auto metrics = provider.local(li);
    for_each_cell(v, EbAssembleRhsKernel<Model, decltype(metrics)>{
                         model, u, ax, fx, fy, r, dx, dy, metrics});
  }
  reject_nonfinite_finite_volume_data("assemble_rhs_eb", R);
}

}  // namespace detail

/// assemble_rhs_eb<Limiter, NumericalFlux>: centre-sampled embedded-boundary residual. Faces use a
/// binary active/active gate; active cells use the clamped five-sample `kappa` estimate documented
/// above. No flux crosses an active/inactive face. This API accepts any device-callable level set and
/// contains no shape-specific transport branch.
///
/// @tparam Limiter        reconstruction (NoSlope / Minmod / VanLeer / Weno5), like the Cartesian operator.
/// @tparam NumericalFlux  flux policy (RusanovFlux by default).
/// @param  ls             POPS_HD callable level set (e.g. detail::DiscDomain): ls < 0 inside.
/// @param  kappa_min      volume fraction floor (small-cell clamp), default kEbKappaMin.
/// @param  face_open_eps  aperture below which a face is CLOSED (immersed wall), default
///                        kEbFaceOpenEps (ADC-615). @param cut_theta_min cut-fraction clamp shared
///                        with the elliptic wall, default kEbCutFractionFloor (ADC-615). The three
///                        thresholds flow from the same typed CutCell descriptor.
///
/// IMPLEMENTATION in TWO PASSES (structure REUSED from the polar operator): pass 1 computes the
/// FACE fluxes gated by the binary alpha_f into temporary MultiFabs; pass 2 differences and divides
/// by kappa_eff. The low-level callable route evaluates the one signed level set for both metrics.
///
/// BOUNDARY CONDITIONS: the caller fills the ghosts (fill_ghosts) before the call, as for
/// assemble_rhs. Faces touching an inactive cell are CLOSED by the kernel (immersed wall),
/// so a closed embedded boundary shields its active interior from the outer box boundary.
///
/// INVARIANT: SEPARATE entry point; the default path (assemble_rhs) stays bit-identical as long
/// as it does NOT call this overload.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model, class LevelSet>
void assemble_rhs_eb(const Model& model, const MultiFab& U, const MultiFab& aux, const LevelSet& ls,
                     const Geometry& geom, MultiFab& R, bool recon_prim = false,
                     Real kappa_min = kEbKappaMin, Real pos_floor = Real(0),
                     Real face_open_eps = kEbFaceOpenEps,
                     Real cut_theta_min = kEbCutFractionFloor,
                     Real weno_eps = kWenoEpsilon) {
  const Real dx = geom.dx(), dy = geom.dy();
  const detail::CallableEbMetricsProvider<LevelSet> provider{
      detail::CallableEbMetrics<LevelSet>{ls, geom, dx, dy, kappa_min, cut_theta_min}};
  detail::assemble_rhs_eb_with_metrics<Limiter, NumericalFlux>(
      model, U, aux, provider, geom, R, recon_prim, pos_floor, face_open_eps, weno_eps);
}

/// Runtime-prepared EB operator. @p active_mask and @p inverse_volume_fraction are immutable,
/// System-owned, cell-centred fields with one ghost layer, sampled and validated once at geometry
/// installation. The time-step hot path performs no analytic-program interpretation.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_eb_prepared(const Model& model, const MultiFab& U, const MultiFab& aux,
                              const MultiFab& active_mask,
                              const MultiFab& inverse_volume_fraction, const Geometry& geom,
                              MultiFab& R, bool recon_prim = false, Real pos_floor = Real(0),
                              Real face_open_eps = kEbFaceOpenEps,
                              Real weno_eps = kWenoEpsilon) {
  assert(active_mask.ncomp() == 1 && active_mask.n_grow() >= 1);
  assert(inverse_volume_fraction.ncomp() == 1);
  assert(active_mask.local_size() == U.local_size());
  assert(inverse_volume_fraction.local_size() == U.local_size());
  const detail::PreparedEbMetricsProvider provider{&active_mask,
                                                   &inverse_volume_fraction};
  detail::assemble_rhs_eb_with_metrics<Limiter, NumericalFlux>(
      model, U, aux, provider, geom, R, recon_prim, pos_floor, face_open_eps, weno_eps);
}

}  // namespace pops
