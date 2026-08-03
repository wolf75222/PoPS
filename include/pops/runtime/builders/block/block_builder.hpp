#pragma once

#include <pops/core/foundation/cold.hpp>  // POPS_COLD_FN: COLD block-builder no-optimize attribute (ADC-337)
#include <pops/core/foundation/types.hpp>
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/execution/for_each.hpp>  // for_each_cell (projection ponctuelle post-pas, ADC-177)
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>  // assemble_rhs_eb (cut-cell EB) + detail::DiscLevelSet (T5-PR2)
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/runtime/builders/scheme_dispatch.hpp>  // dispatch_limiter: ONE limiter-route dispatch generator (ADC-640)
#include <pops/runtime/config/dispatch_tags.hpp>  // UNIQUE registry of tags (validate_limiter/riemann, limiter_n_ghost)
#include <pops/runtime/context/grid_context.hpp>  // GridContext + BlockClosures (shared lightweight header)
#include <pops/runtime/numerical_defaults.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <memory>  // std::shared_ptr (shared scratch of the HLL wave speed cache, opt-in)
#include <stdexcept>
#include <string>
#include <type_traits>  // std::is_same_v (cache engages only for the HLL flux)
#include <utility>
#include <vector>

/// @file
/// @brief Builds the spatial closures of a block (residual + Poisson contribution) from a
///        COMPILED model (CompositeModel) and a grid context.
///
/// This code used to live in System::Impl; it is extracted into a header so that the SAME template
/// path (assemble_rhs<Limiter, Flux>, inlinable and device-ready) is instantiable from an EXTERNAL
/// TRANSLATION UNIT. It is the brick that lets a DSL-generated model be compiled AOT (ahead-of-time)
/// and then plugged into the System via the PRODUCTION path (HLLC/Roe flux, order 2, GPU), no longer
/// only via the virtual host path of the dynamic block.
///
/// The System remains the sole owner of the mesh and the aux; GridContext only carries immutable
/// copies of them (domain, BC, geometry) and a non-owning POINTER to the aux (stable address,
/// lifetime longer than the block).

namespace pops {

// GridContext and BlockClosures: defined in pops/runtime/grid_context.hpp (lightweight header, also
// included by system.hpp to expose grid_context() / install_block() without pulling in the numerics).

namespace detail {
inline bool embedded_boundary_active(const GridContext& context) {
  return context.embedded_boundary_set != nullptr && *context.embedded_boundary_set &&
         context.geometry_mode != nullptr && *context.geometry_mode != GeometryMode::None;
}

inline bool cutcell_geometry_active(const GridContext& context) {
  return embedded_boundary_active(context) && *context.geometry_mode == GeometryMode::CutCell;
}

inline void require_geometry_aware_boundary_provider(const GridContext& context,
                                                     const char* operation) {
  if (embedded_boundary_active(context) && context.boundary_plan &&
      context.boundary_plan->has_component_boundaries())
    throw std::runtime_error(
        std::string(operation) +
        ": embedded-boundary transport cannot execute a native boundary component because "
        "that provider has no active-cell or cut-cell metric contract");
}

struct ZeroPreparedBoundaryFace {
  Array4 flux;
  int axis = 0;
  int coordinate = 0;
  int components = 0;
  POPS_HD void operator()(int i, int j) const {
    if ((axis == 0 ? i : j) != coordinate)
      return;
    for (int component = 0; component < components; ++component)
      flux(i, j, component) = Real(0);
  }
};

inline void zero_prepared_boundary_fluxes(MultiFab& fx, MultiFab& fy, const GridContext& context) {
  if (!context.boundary_plan || (!context.boundary_plan->has_omitted_faces() &&
                                 !context.boundary_plan->has_zero_flux_faces()))
    return;
  for (int local = 0; local < fx.local_size(); ++local) {
    const Box2D faces = fx.box(local);
    if (context.boundary_plan->omits_face(0, -1) || context.boundary_plan->zeroes_face(0, -1))
      for_each_cell(
          faces, ZeroPreparedBoundaryFace{fx.fab(local).array(), 0, context.dom.lo[0], fx.ncomp()});
    if (context.boundary_plan->omits_face(0, 1) || context.boundary_plan->zeroes_face(0, 1))
      for_each_cell(faces, ZeroPreparedBoundaryFace{fx.fab(local).array(), 0, context.dom.hi[0] + 1,
                                                    fx.ncomp()});
  }
  for (int local = 0; local < fy.local_size(); ++local) {
    const Box2D faces = fy.box(local);
    if (context.boundary_plan->omits_face(1, -1) || context.boundary_plan->zeroes_face(1, -1))
      for_each_cell(
          faces, ZeroPreparedBoundaryFace{fy.fab(local).array(), 1, context.dom.lo[1], fy.ncomp()});
    if (context.boundary_plan->omits_face(1, 1) || context.boundary_plan->zeroes_face(1, 1))
      for_each_cell(faces, ZeroPreparedBoundaryFace{fy.fab(local).array(), 1, context.dom.hi[1] + 1,
                                                    fy.ncomp()});
  }
}

inline BoundaryFaceOmission prepared_boundary_face_omission(const GridContext& context) {
  BoundaryFaceOmission omission;
  omission.domain = context.dom;
  if (context.boundary_plan) {
    omission.xlo =
        context.boundary_plan->omits_face(0, -1) || context.boundary_plan->zeroes_face(0, -1);
    omission.xhi =
        context.boundary_plan->omits_face(0, +1) || context.boundary_plan->zeroes_face(0, +1);
    omission.ylo =
        context.boundary_plan->omits_face(1, -1) || context.boundary_plan->zeroes_face(1, -1);
    omission.yhi =
        context.boundary_plan->omits_face(1, +1) || context.boundary_plan->zeroes_face(1, +1);
  }
  return omission;
}

struct PreparedBoundaryFluxFilter {
  const GridContext* context = nullptr;
  void operator()(MultiFab& fx, MultiFab& fy) const {
    if (context != nullptr)
      zero_prepared_boundary_fluxes(fx, fy, *context);
  }
};

template <class Limiter, class Flux, class Model>
inline void assemble_rhs_without_prepared_interfaces(
    const Model& model, MultiFab& state, const GridContext& context, MultiFab& residual,
    bool reconstruct_primitive, Real positivity_floor, Real weno_epsilon = kWenoEpsilon,
    const std::shared_ptr<MultiFab>& ws_cache = {},
    const runtime::multiblock::BoundaryEvaluationPoint* point = nullptr,
    const PreparedGridBoundarySession* boundary = nullptr) {
  std::vector<Box2D> xboxes;
  std::vector<Box2D> yboxes;
  xboxes.reserve(static_cast<std::size_t>(state.box_array().size()));
  yboxes.reserve(static_cast<std::size_t>(state.box_array().size()));
  for (int box = 0; box < state.box_array().size(); ++box) {
    xboxes.push_back(xface_box(state.box_array()[box]));
    yboxes.push_back(yface_box(state.box_array()[box]));
  }
  MultiFab fx(BoxArray(std::move(xboxes)), state.dmap(), state.ncomp(), 0);
  MultiFab fy(BoxArray(std::move(yboxes)), state.dmap(), state.ncomp(), 0);
  if constexpr (std::is_same_v<Flux, HLLFlux>) {
    if (ws_cache) {
      compute_face_fluxes_hll_cached<Limiter>(
          model, state, *context.aux, fx, fy, *ws_cache, context.geom.dx(), context.geom.dy(),
          reconstruct_primitive, positivity_floor, weno_epsilon);
    } else {
      compute_face_fluxes<Limiter, Flux>(model, state, *context.aux, fx, fy, context.geom.dx(),
                                         context.geom.dy(), reconstruct_primitive, positivity_floor,
                                         weno_epsilon);
    }
  } else {
    compute_face_fluxes<Limiter, Flux>(model, state, *context.aux, fx, fy, context.geom.dx(),
                                       context.geom.dy(), reconstruct_primitive, positivity_floor,
                                       weno_epsilon);
  }
  if (context.boundary_plan && context.boundary_plan->has_flux_transformations()) {
    if (point == nullptr)
      throw std::logic_error(
          "post-Riemann boundary flux transformation requires a BoundaryEvaluationPoint");
    if (boundary != nullptr)
      transform_grid_boundary_fluxes(state, fx, fy, *boundary, *point);
    else
      transform_grid_boundary_fluxes(state, fx, fy, context, *point);
  }
  zero_prepared_boundary_fluxes(fx, fy, context);
  mf_eval_rhs(model, state, *context.aux, fx, fy, context.geom.dx(), context.geom.dy(), residual);
}

/// Residual functor -div F + S (fill_ghosts then assemble_rhs), passed TO THE TimeStepper as RhsEval.
/// NAMED FUNCTOR (not a lambda): this is what take_step receives and what triggers the instantiation
/// of assemble_rhs<Limiter, Flux> (and its device AssembleRhsKernel). First-instantiated from an
/// EXTERNAL TU (add_compiled_model), a lambda here makes nvcc choke on emitting the nested device
/// kernel (Heisenbug: OK Serial + compute-sanitizer, segfault at Cuda run time). A class has a stable
/// instantiation context -> robust device codegen. Body identical to the former lambda -> residual
/// bit-identical to add_block on CPU (and, intended, on device).
template <class Limiter, class Flux, class Model>
struct BlockRhsEval {
  Model model;
  const GridContext* ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  /// Exact reconstructed-face signal-speed scratch (HLL cache, opt-in). nullptr (default) keeps the
  /// uncached path. Non-null ONLY for the HLL flux (cf. build_block): four lanes store the lower and
  /// upper speeds of each x/y face, so model.wave_speeds is always present there.
  std::shared_ptr<MultiFab> ws_cache;
  Real weno_eps =
      kWenoEpsilon;  ///< ADC-645: WENO-Z regulariser (default = historical, bit-identical)
  void operator()(MultiFab& U, MultiFab& R) const {
    if (ctx->boundary_plan && ctx->boundary_plan->has_flux_transformations())
      throw std::logic_error(
          "post-Riemann boundary flux transformation requires a BoundaryEvaluationPoint");
    if (ctx->boundary_plan && ctx->boundary_plan->has_omitted_faces())
      throw std::logic_error(
          "prepared shared-interface flux requires BoundaryEvaluationPoint group authority");
    fill_grid_ghosts(U, *ctx);
    eval_core_filled(U, R);
  }

  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  MultiFab& R) const {
    eval_core(point, U, R);
    add_grid_boundary_residual(U, R, *ctx, point);
  }

  /// Point-qualified transport/source core.  It includes ghost producers and prepared shared-face
  /// omission, but deliberately excludes additive FieldBoundary residuals so an implicit operator
  /// can compose their exact JVP without finite-differencing them twice.
  void eval_core(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                 MultiFab& R) const {
    fill_grid_ghosts(U, *ctx, point);
    eval_core_filled(U, R, &point, nullptr);
  }

  void eval_core(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                 MultiFab& R, const PreparedGridBoundarySession& boundary) const {
    fill_grid_ghosts(U, boundary, point);
    eval_core_filled(U, R, &point, &boundary);
  }

 private:
  void eval_core_filled(MultiFab& U, MultiFab& R,
                        const runtime::multiblock::BoundaryEvaluationPoint* point = nullptr,
                        const PreparedGridBoundarySession* boundary = nullptr) const {
    if (ctx->boundary_plan &&
        (ctx->boundary_plan->has_omitted_faces() || ctx->boundary_plan->has_zero_flux_faces() ||
         ctx->boundary_plan->has_flux_transformations())) {
      assemble_rhs_without_prepared_interfaces<Limiter, Flux>(
          model, U, *ctx, R, recon_prim, pos_floor, weno_eps, ws_cache, point, boundary);
      return;
    }
    if constexpr (std::is_same_v<Flux, HLLFlux>) {
      if (ws_cache) {
        if (!detail::wave_speed_cache_matches(*ws_cache, U))
          *ws_cache = MultiFab(U.box_array(), U.dmap(), 4, 1);
        assemble_rhs_hll_cached<Limiter>(model, U, *ctx->aux, ctx->geom, R, *ws_cache, recon_prim,
                                         pos_floor, weno_eps);
        return;
      }
    }
    assemble_rhs<Limiter, Flux>(model, U, *ctx->aux, ctx->geom, R, recon_prim, pos_floor, weno_eps);
  }
};

template <class Limiter, class Flux, class Model>
struct RhsCoreInto {
  Model model;
  GridContext ctx;
  bool recon_prim;
  Real pos_floor = Real(0);
  std::shared_ptr<MultiFab> ws_cache;
  Real weno_eps = kWenoEpsilon;
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  MultiFab& R) const {
    BlockRhsEval<Limiter, Flux, Model>{model, &ctx, recon_prim, pos_floor, ws_cache, weno_eps}
        .eval_core(point, U, R);
  }
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  MultiFab& R, const PreparedGridBoundarySession& boundary) const {
    BlockRhsEval<Limiter, Flux, Model>{model, &ctx, recon_prim, pos_floor, ws_cache, weno_eps}
        .eval_core(point, U, R, boundary);
  }
};

struct BoundaryResidualInto {
  GridContext ctx;
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  MultiFab& R) const {
    add_grid_boundary_residual(U, R, ctx, point);
  }
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  MultiFab& R, const PreparedGridBoundarySession& boundary) const {
    add_grid_boundary_residual(U, R, boundary, point);
  }
};

struct BoundaryJvpInto {
  GridContext ctx;
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  const MultiFab& V, MultiFab& J) const {
    apply_grid_boundary_jvp(U, V, J, ctx, point);
  }
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  const MultiFab& V, MultiFab& J,
                  const PreparedGridBoundarySession& boundary) const {
    apply_grid_boundary_jvp(U, V, J, boundary, point);
  }
};

/// Frozen residual (fill_ghosts + assemble_rhs) installed as the block's rhs_into.
/// Functor of the dt_hotspot diagnostic (ADC-182): dominant cell of the block CFL.
/// HOST (the internal reductions are device); named, like MaxSpeed.
template <class Model>
struct HotspotFn {
  Model m;
  GridContext ctx;
  void operator()(const MultiFab& U, Real& w, int& i, int& j) const {
    if (cutcell_geometry_active(ctx))
      max_wave_speed_hotspot_mf(m, U, *ctx.aux, *ctx.domain_mask, *ctx.eb_inverse_volume_fraction,
                                ctx.dom.nx(), w, i, j);
    else if (embedded_boundary_active(ctx))
      max_wave_speed_hotspot_mf(m, U, *ctx.aux, *ctx.domain_mask, ctx.dom.nx(), w, i, j);
    else
      max_wave_speed_hotspot_mf(m, U, *ctx.aux, ctx.dom.nx(), w, i, j);
  }
};

/// Kernel device de la PROJECTION PONCTUELLE post-pas (ADC-177) :
/// U(i, j) <- m.project(U(i, j), aux(i, j)). FONCTEUR NOMME (meme contrat device que BlockRhsEval).
/// Lecture et ecriture sur le MEME fab : acces strictement ponctuel (aucun voisin lu), donc aucune
/// dependance inter-cellule -- parallelisable sans tampon.
template <class Model>
struct ProjectCellKernel {
  Model m;
  Array4 u;        // ecriture (etat du bloc)
  ConstArray4 uc;  // lecture (meme fab, vue const)
  ConstArray4 a;   // aux du System (phi, grad phi, champs extra)
  POPS_HD void operator()(int i, int j) const {
    const typename Model::State p =
        m.project(load_state<Model>(uc, i, j), load_aux<aux_comps<Model>()>(a, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = p[c];
  }
};

template <class Model>
struct ProjectActiveCellKernel {
  Model m;
  Array4 u;
  ConstArray4 uc;
  ConstArray4 a;
  ConstArray4 mask;
  POPS_HD void operator()(int i, int j) const {
    if (!mask_active(mask, i, j))
      return;
    const typename Model::State p =
        m.project(load_state<Model>(uc, i, j), load_aux<aux_comps<Model>()>(a, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = p[c];
  }
};

/// Foncteur HOTE de la projection ponctuelle : for_each_cell du kernel sur les cellules VALIDES de
/// chaque fab local. Les GHOSTS ne sont pas projetes : tout consommateur de ghosts (residu de
/// transport) refait fill_ghosts en tete d'evaluation (cf. BlockRhsEval), donc l'etat fantome est
/// reconstruit du valide projete au pas suivant -- aucun fill_boundary necessaire ici.
template <class Model>
struct PointwiseProject {
  Model m;
  GridContext ctx;
  void operator()(MultiFab& U) const {
    for (int li = 0; li < U.local_size(); ++li)
      for_each_cell(U.box(li),
                    ProjectCellKernel<Model>{m, U.fab(li).array(), U.fab(li).const_array(),
                                             ctx.aux->fab(li).const_array()});
  }
};

/// Embedded-boundary projection.  The prepared domain mask is a geometry-provider output rather
/// than a shape-specific contract; both Staircase and CutCell use it to preserve inactive storage.
template <class Model>
struct PointwiseProjectMasked {
  Model m;
  GridContext ctx;
  const MultiFab* mask;
  void operator()(MultiFab& U) const {
    for (int li = 0; li < U.local_size(); ++li)
      for_each_cell(U.box(li), ProjectActiveCellKernel<Model>{
                                   m, U.fab(li).array(), U.fab(li).const_array(),
                                   ctx.aux->fab(li).const_array(), mask->fab(li).const_array()});
  }
};

template <class Limiter, class Flux, class Model>
struct RhsInto {
  Model m;
  GridContext ctx;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  std::shared_ptr<MultiFab> ws_cache;  ///< HLL wave speed cache (opt-in); nullptr -> per-face path
  Real weno_eps = kWenoEpsilon;        ///< ADC-645: WENO-Z regulariser (default = historical)
  void operator()(MultiFab& U, MultiFab& R) const {
    // Delegates to BlockRhsEval (fill_ghosts + assemble_rhs OR cached path): single source of the residual.
    BlockRhsEval<Limiter, Flux, Model>{m, &ctx, recon_prim, pos_floor, ws_cache, weno_eps}(U, R);
  }
  void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                  MultiFab& R) const {
    BlockRhsEval<Limiter, Flux, Model>{m, &ctx, recon_prim, pos_floor, ws_cache, weno_eps}(point, U,
                                                                                           R);
  }
};

/// SOURCE-ONLY residual kernel R(i,j) <- m.source(U(i,j), aux(i,j)): the EXACT source term of
/// AssembleRhsKernel (cf. cartesian_operator.hpp: r = S - div Fhat), with NO flux / reconstruction /
/// numerical-flux dispatch (ADC-430). NAMED FUNCTOR (same device contract as AssembleRhsKernel /
/// ProjectCellKernel), so the device codegen is robust across the AOT TU boundary. Reads the cell's own
/// state + aux only (POINTWISE, no neighbor): no ghosts required. POPS_HD.
template <class Model>
struct SourceOnlyKernel {
  Model m;
  ConstArray4 u;  // block state (read)
  ConstArray4 a;  // System aux (phi, grad phi, extra fields)
  Array4 r;       // residual (write)
  POPS_HD void operator()(int i, int j) const {
    const auto S = m.source(load_state<Model>(u, i, j), load_aux<aux_comps<Model>()>(a, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c];
  }
};

template <class Model>
struct SourceOnlyActiveKernel {
  Model m;
  ConstArray4 u;
  ConstArray4 a;
  ConstArray4 mask;
  Array4 r;
  POPS_HD void operator()(int i, int j) const {
    if (!mask_active(mask, i, j)) {
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) = Real(0);
      return;
    }
    const auto S = m.source(load_state<Model>(u, i, j), load_aux<aux_comps<Model>()>(a, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c];
  }
};

/// SOURCE-ONLY residual R <- S(U, aux) installed as the block's source_only closure (ADC-430). The exact
/// MIRROR of RhsInto on SourceFreeModel (which is flux without source): SourceInto is source without flux.
/// Together they split the rhs_into residual -div F + S into its two halves. Bit-identical to the source
/// term assemble_rhs adds (SAME m.source, SAME load_state / load_aux), but with no numerical-flux
/// dispatch -- so it is flux-template agnostic (a zero-flux MODEL adapter could not zero HLL/Roe, which
/// recombine via wave_speeds, but skipping the flux entirely always gives exactly S). Cell-local: NO
/// fill_ghosts (the source reads the valid cell only, unlike the flux divergence). HOST loop over the
/// valid cells of each local fab (the kernel is device).
template <class Model>
struct SourceInto {
  Model m;
  GridContext ctx;
  void operator()(MultiFab& U, MultiFab& R) const {
    for (int li = 0; li < U.local_size(); ++li)
      for_each_cell(R.box(li),
                    SourceOnlyKernel<Model>{m, U.fab(li).const_array(),
                                            ctx.aux->fab(li).const_array(), R.fab(li).array()});
  }
};

/// Embedded-boundary source-only residual.  This is deliberately a mask policy, not a Disc or CSG
/// branch.  The CutCell transport operator applies inverse volume fraction only to flux divergence,
/// so its source half is identical to Staircase on the same prepared active-cell set.
template <class Model>
struct SourceIntoMasked {
  Model m;
  GridContext ctx;
  const MultiFab* mask;
  void operator()(MultiFab& U, MultiFab& R) const {
    for (int li = 0; li < U.local_size(); ++li)
      for_each_cell(R.box(li), SourceOnlyActiveKernel<Model>{
                                   m, U.fab(li).const_array(), ctx.aux->fab(li).const_array(),
                                   mask->fab(li).const_array(), R.fab(li).array()});
  }
};

// ============================================================================
// EMBEDDED-BOUNDARY ROUTING: generic level-set residual evaluators and their advances.
// ============================================================================
// The transport residual of a block goes through BlockRhsEval (assemble_rhs, full cartesian). The two
// evaluators below substitute an EB operator for assemble_rhs, reading the System geometry by
// pointer (stable address of an Impl member) at step time -- so block and geometry authoring order
// is indifferent. Named functors keep the same device contract as BlockRhsEval.

/// MASKED transport residual (Staircase mode): fill_ghosts then assemble_rhs_masked on the
/// cell-centered 0/1 mask of the System (read via @c mask, pointer to Impl::domain_mask_, stable
/// address). The mask has the SAME layout as U (same ba/dm, 1 ghost). Inactive cell -> residual 0;
/// face toward an inactive cell -> zero normal flux (FV wall). The flux / reconstruction are REUSED
/// verbatim.
template <class Limiter, class Flux, class Model>
struct BlockRhsEvalMasked {
  Model model;
  GridContext ctx;
  const MultiFab* mask;  // Impl::domain_mask_ (NOT owned; stable address)
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  Real weno_eps = kWenoEpsilon;
  void operator()(MultiFab& U, MultiFab& R) const {
    require_geometry_aware_boundary_provider(ctx, "masked transport residual");
    fill_grid_ghosts(U, ctx);
    const BoundaryFaceOmission omission = prepared_boundary_face_omission(ctx);
    assemble_rhs_masked_impl<Limiter, Flux>(model, U, *ctx.aux, *mask, ctx.geom, R, recon_prim,
                                            pos_floor, weno_eps, omission);
  }

  /// Program core after its point-qualified host protocol has produced ghosts. Kept as the only
  /// extra Model/Limiter/Flux instantiation; point/prepared/boundary composition is out of line.
  void eval_program_core(MultiFab& U, MultiFab& R) const {
    const BoundaryFaceOmission omission = prepared_boundary_face_omission(ctx);
    assemble_rhs_masked_impl<Limiter, Flux>(model, U, *ctx.aux, *mask, ctx.geom, R, recon_prim,
                                            pos_floor, weno_eps, omission);
  }
};

/// CUT-CELL / EB transport residual (CutCell mode): fill ghosts then use the static mask and inverse
/// volume fraction prepared once from the signed level set at System installation. Stable owner
/// pointers make block/geometry authoring order irrelevant. No expression interpreter,
/// shape-specific branch, std::function, or Python callback enters the RHS hot path.
template <class Limiter, class Flux, class Model>
struct BlockRhsEvalEb {
  Model model;
  GridContext ctx;
  const MultiFab* inverse_volume_fraction;  // NOT owned; stable System owner
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  Real weno_eps = kWenoEpsilon;
  void operator()(MultiFab& U, MultiFab& R) const {
    require_geometry_aware_boundary_provider(ctx, "cut-cell transport residual");
    fill_grid_ghosts(U, ctx);
    const Real face_open_eps =
        ctx.eb_thresholds ? ctx.eb_thresholds->face_open_eps : ctx.eb_face_open_eps;
    const PreparedEbMetricsProvider provider{ctx.domain_mask, inverse_volume_fraction};
    assemble_rhs_eb_with_metrics<Limiter, Flux>(model, U, *ctx.aux, provider, ctx.geom, R,
                                                recon_prim, pos_floor, face_open_eps, weno_eps,
                                                PreparedBoundaryFluxFilter{&ctx});
  }

  /// Program core after its point-qualified host protocol has produced ghosts. See the staircase
  /// twin above; geometry metrics remain statically compiled into this native callback.
  void eval_program_core(MultiFab& U, MultiFab& R) const {
    const Real face_open_eps =
        ctx.eb_thresholds ? ctx.eb_thresholds->face_open_eps : ctx.eb_face_open_eps;
    const PreparedEbMetricsProvider provider{ctx.domain_mask, inverse_volume_fraction};
    assemble_rhs_eb_with_metrics<Limiter, Flux>(model, U, *ctx.aux, provider, ctx.geom, R,
                                                recon_prim, pos_floor, face_open_eps, weno_eps,
                                                PreparedBoundaryFluxFilter{&ctx});
  }
};

/// The only model-dependent adaptor required by the point-qualified Program protocol. Its erased
/// signature is consumed by make_geometry_residual_closures(), whose host-only point/prepared/core
/// wrappers are non-template and therefore compiled once per TU rather than once per transport leaf.
template <class Eval>
struct GeometryProgramCore {
  Eval eval;
  void operator()(MultiFab& U, MultiFab& R) const { eval.eval_program_core(U, R); }
};

}  // namespace detail

/// Spatial closures for a frozen scheme (Limiter x Flux). Time integration, substeps and implicit
/// solves belong exclusively to the installed ProgramGraph; this builder materializes only residual,
/// boundary, projection and diagnostic primitives.
template <class Limiter, class Flux, class Model>
POPS_COLD_FN BlockClosures build_block(const Model& m, const GridContext& ctx, bool recon_prim,
                                       Real pos_floor = Real(0), bool wave_speed_cache = false,
                                       Real weno_eps = kWenoEpsilon) {
  const MultiFab* domain_mask = ctx.domain_mask;
  const MultiFab* eb_inverse_volume_fraction = ctx.eb_inverse_volume_fraction;
  // The per-block WENO-Z regulariser belongs to the reconstruction policy, independently of domain
  // geometry. Thread the same value through full Cartesian, masked and cut-cell residuals so changing
  // geometry never changes or silently discards the authored numerical scheme.
  BlockClosures bc;
  // The current EB operators own only first-order hyperbolic transport. Higher-order
  // reconstructions can cross the inactive set, and DiffusiveModel needs a conservative embedded
  // diffusive flux that is not implemented here. Advertise only capabilities that are physically
  // executable; System validates this bitset before publishing a geometry.
  constexpr bool supports_embedded_boundary =
      supports_embedded_boundary_reconstruction_v<Limiter> && !DiffusiveModel<Model>;
  if constexpr (supports_embedded_boundary)
    bc.supported_geometry_modes = kAllGeometrySupport;
  // SHARED scratch of the HLL wave speed cache (opt-in): a single MultiFab for the residual family
  // (never called concurrently by one Program stage). nullptr when the option is OFF -> BlockRhsEval
  // keeps the per-face path (bit-identical). Allocated at the real layout on the first call
  // (cf. BlockRhsEval).
  std::shared_ptr<MultiFab> ws_cache =
      wave_speed_cache ? std::make_shared<MultiFab>() : std::shared_ptr<MultiFab>{};
  bc.rhs_into =
      detail::RhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache, weno_eps};
  bc.rhs_at_point =
      detail::RhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache, weno_eps};
  // FLUX-ONLY residual R <- -div F(U) (ADC-425): the SAME RhsInto path on SourceFreeModel<Model> (the
  // canonical zero-source adapter the IMEX explicit half-step already uses, state_access.hpp), so the
  // flux / ghost / geometry / positivity handling is bit-identical to rhs_into -- only the model's
  // default/composite source is dropped. A compiled time Program's hyperbolic stage reads it so a
  // Lie/Strang split assembles "flux but no source" without the default source leaking in (spec
  // criterion 17). NO HLL cache: the full and core residuals share ws_cache (never concurrent), but a
  // flux-only RHS can interleave with them, so it keeps the per-face path. The
  // residual is identical either way with limiter='none'; the HLL wave-speed cache -- rejected on the
  // aot/production backends compiled Programs use -- is the only path where cached cell-center speeds
  // differ from the per-face reconstruction, so for a compiled Program the cache is a perf scratch,
  // not a numerics change.
  bc.rhs_flux_only = detail::RhsInto<Limiter, Flux, SourceFreeModel<Model>>{
      SourceFreeModel<Model>{m}, ctx, recon_prim, pos_floor, nullptr, weno_eps};
  bc.rhs_flux_only_at_point = detail::RhsInto<Limiter, Flux, SourceFreeModel<Model>>{
      SourceFreeModel<Model>{m}, ctx, recon_prim, pos_floor, nullptr, weno_eps};
  bc.rhs_core_at_point =
      detail::RhsCoreInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache, weno_eps};
  bc.rhs_flux_only_core_at_point = detail::RhsCoreInto<Limiter, Flux, SourceFreeModel<Model>>{
      SourceFreeModel<Model>{m}, ctx, recon_prim, pos_floor, nullptr, weno_eps};
  bc.boundary_residual_at_point = detail::BoundaryResidualInto{ctx};
  bc.boundary_jvp_at_point = detail::BoundaryJvpInto{ctx};
  bc.rhs_core_at_point_prepared =
      detail::RhsCoreInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache, weno_eps};
  bc.rhs_flux_only_core_at_point_prepared =
      detail::RhsCoreInto<Limiter, Flux, SourceFreeModel<Model>>{
          SourceFreeModel<Model>{m}, ctx, recon_prim, pos_floor, nullptr, weno_eps};
  bc.boundary_residual_at_point_prepared = detail::BoundaryResidualInto{ctx};
  bc.boundary_jvp_at_point_prepared = detail::BoundaryJvpInto{ctx};

  // The compiled Program uses the same geometry-aware residual family as the legacy time
  // integrator.  Each family freezes the exact limiter, Riemann flux, reconstruction, positivity
  // floor and WENO regulariser selected above; only its metric provider differs.  No shape name or
  // analytic expression is inspected in this hot path.
  if (supports_embedded_boundary && domain_mask) {
    using FullEval = detail::BlockRhsEvalMasked<Limiter, Flux, Model>;
    using FluxEval = detail::BlockRhsEvalMasked<Limiter, Flux, SourceFreeModel<Model>>;
    const FullEval full_eval{m, ctx, domain_mask, recon_prim, pos_floor, weno_eps};
    const FluxEval flux_eval{
        SourceFreeModel<Model>{m}, ctx, domain_mask, recon_prim, pos_floor, weno_eps};
    bc.staircase_residuals = make_geometry_residual_closures(
        ctx, detail::GeometryProgramCore<FullEval>{full_eval},
        detail::GeometryProgramCore<FluxEval>{flux_eval}, "masked Program residual");
  }
  if (supports_embedded_boundary && eb_inverse_volume_fraction) {
    using FullEval = detail::BlockRhsEvalEb<Limiter, Flux, Model>;
    using FluxEval = detail::BlockRhsEvalEb<Limiter, Flux, SourceFreeModel<Model>>;
    const FullEval full_eval{m, ctx, eb_inverse_volume_fraction, recon_prim, pos_floor, weno_eps};
    const FluxEval flux_eval{SourceFreeModel<Model>{m},
                             ctx,
                             eb_inverse_volume_fraction,
                             recon_prim,
                             pos_floor,
                             weno_eps};
    bc.cutcell_residuals = make_geometry_residual_closures(
        ctx, detail::GeometryProgramCore<FullEval>{full_eval},
        detail::GeometryProgramCore<FluxEval>{flux_eval}, "cut-cell Program residual");
  }
  if (ctx.boundary_plan && ctx.boundary_plan->has_omitted_faces()) {
    bc.rhs_without_prepared_interfaces =
        detail::RhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, pos_floor, ws_cache, weno_eps};
    bc.rhs_flux_only_without_prepared_interfaces =
        detail::RhsInto<Limiter, Flux, SourceFreeModel<Model>>{
            SourceFreeModel<Model>{m}, ctx, recon_prim, pos_floor, nullptr, weno_eps};
  }
  // SOURCE-ONLY residual R <- S(U, aux) (ADC-430): the exact MIRROR of rhs_flux_only. SourceInto
  // evaluates m.source per cell (the SAME source term assemble_rhs / rhs_into add) with no numerical-flux
  // dispatch, so it is bit-identical to the source half of rhs_into and flux-template agnostic (a
  // zero-flux model adapter could not zero HLL/Roe, which recombine via wave_speeds). A compiled time
  // Program's source stage reads it so a Lie/Strang split assembles "the default source but no flux"
  // without the -div F base leaking in (spec: rhs flux=False is source-only). No Limiter/Flux: the
  // source is cell-local, independent of the spatial scheme.
  bc.source_only = detail::SourceInto<Model>{m, ctx};
  if (domain_mask)
    bc.source_only_masked = detail::SourceIntoMasked<Model>{m, ctx, domain_mask};
  bc.hotspot =
      detail::HotspotFn<Model>{m, ctx};  // dt_hotspot diagnostic (ADC-182), off the hot path
  // PROJECTION PONCTUELLE Program (ADC-177) : fabriquee SEULEMENT si le modele declare le trait
  // (HasPointwiseProjection, cf. core/physical_model.hpp) ; vide sinon. Partagee par add_block ET
  // add_compiled_model (les deux
  // passent par make_block) : un .so 'production' la transporte donc nativement.
  if constexpr (HasPointwiseProjection<Model>) {
    bc.project = detail::PointwiseProject<Model>{m, ctx};
    if (domain_mask)
      bc.project_masked = detail::PointwiseProjectMasked<Model>{m, ctx, domain_mask};
  }
  return bc;
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> compiled closures. HLLC / Roe are
/// guarded only by their exact physical-provider capabilities (otherwise an explicit error).
/// "weno5" = WENO5-Z reconstruction (order 5, 5-point stencil, 3 ghosts); spatial_operator routes
/// through the policy's explicit stencil protocol (the caller allocates its declared ghost radius,
/// cf. block_n_ghost).
// Per-flux limiter ladders, split out of make_block (ADC-335) so each flux's build_block leaves can be
// instantiated in their OWN translation unit (python/system_compressible_<flux>.cpp). Each body is the
// VERBATIM content of make_block's old `if (riem == "<flux>")` branch (same capability if-constexpr,
// same limiter ladder, same throws) -> bit-identical. make_block (below) is now a thin riem dispatcher
// that calls these; it stays the entry point for the non-subdivided callers (exb/isothermal seams, the
// .so/AOT loader path). The flux string is implied by which helper is called -> validation moves to the
// make_block dispatcher (kept) and, for the per-flux seam path, to the caller (System).
template <class Model>
POPS_COLD_FN BlockClosures make_block_rusanov(const Model& m, const std::string& lim,
                                              const GridContext& ctx, bool recon_prim,
                                              Real pos_floor, Real weno_eps = kWenoEpsilon) {
  return dispatch_limiter(parse_limiter_route(lim, "System"), "System", [&](auto tag) {
    using L = typename decltype(tag)::type;
    return build_block<L, RusanovFlux>(m, ctx, recon_prim, pos_floor,
                                       /*wave_speed_cache=*/false, weno_eps);
  });
}

template <class Model>
POPS_COLD_FN BlockClosures make_block_hll(const Model& m, const std::string& lim,
                                          const GridContext& ctx, bool recon_prim, Real pos_floor,
                                          bool wave_speed_cache, Real weno_eps = kWenoEpsilon) {
  // HLL (Harten-Lax-van Leer, 2 waves): less diffusive than Rusanov (dissipation ~ signed |sR-sL|
  // instead of symmetric 2*max|v|), but does NOT require pressure (unlike HLLC/Roe) -- only SIGNED
  // wave speeds model.wave_speeds. Available as soon as a model exposes its signed eigenvalues (the
  // DSL emits wave_speeds as soon as a primitive 'p' is declared, even cold isothermal p=0 -> c=0 ->
  // HLL degenerates to upwind, still less diffusive than Rusanov at the contact). Does NOT REQUIRE
  // n_vars==4 nor a pressure: usable by a 3-var isothermal model (rho, m_x, m_y) exposing signed
  // wave speeds but no pressure, where hllc/roe are rejected. Gated on the presence of wave_speeds
  // (otherwise a CLEAR error, not a compilation failure for a scalar model without a signed wave,
  // e.g. ExB transport).
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    // wave_speed_cache (opt-in) forwarded ONLY here: the wave speed cache only engages for the HLL
    // flux (BlockRhsEval guarded by Flux == HLLFlux). rusanov/hllc/roe ignore it.
    return dispatch_limiter(parse_limiter_route(lim, "System"), "System", [&](auto tag) {
      using L = typename decltype(tag)::type;
      return build_block<L, HLLFlux>(m, ctx, recon_prim, pos_floor, wave_speed_cache, weno_eps);
    });
  } else {
    throw std::runtime_error(
        "System: flux 'hll' requires signed wave speeds "
        "(model.wave_speeds: declare a primitive 'p' / eigenvalues); "
        "this transport -> 'rusanov'");
  }
}

template <class Model>
POPS_COLD_FN BlockClosures make_block_hllc(const Model& m, const std::string& lim,
                                           const GridContext& ctx, bool recon_prim, Real pos_floor,
                                           Real weno_eps = kWenoEpsilon) {
  // HLLC is capability-only: the model supplies the contact closure and star-state construction.
  // Euler reaches this exact path because its physical provider satisfies HasHLLCStructure.
  if constexpr (HasHLLCStructure<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "System"), "System", [&](auto tag) {
      using L = typename decltype(tag)::type;
      return build_block<L, HLLCFlux>(m, ctx, recon_prim, pos_floor,
                                      /*wave_speed_cache=*/false, weno_eps);
    });
  } else {
    throw std::runtime_error(
        "System: flux 'hllc' requires the model's HLLC capability "
        "(HasHLLCStructure: pressure + wave_speeds + contact_speed + hllc_star_state); "
        "install an explicit contact/star-state provider; this transport -> 'hll'/'rusanov'");
  }
}

template <class Model>
POPS_COLD_FN BlockClosures make_block_roe(const Model& m, const std::string& lim,
                                          const GridContext& ctx, bool recon_prim, Real pos_floor,
                                          Real weno_eps = kWenoEpsilon) {
  // Roe is capability-only: the physical provider supplies the complete d = |A_roe| dU action.
  // Euler and non-Euler models therefore consume the same numerical-flux implementation.
  if constexpr (HasRoeDissipation<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "System"), "System", [&](auto tag) {
      using L = typename decltype(tag)::type;
      return build_block<L, RoeFlux>(m, ctx, recon_prim, pos_floor,
                                     /*wave_speed_cache=*/false, weno_eps);
    });
  } else {
    throw std::runtime_error(
        "System: flux 'roe' requires the model's Roe capability "
        "(HasRoeDissipation: roe_dissipation d = |A_roe| dU); install an explicit analytic "
        "or Jacobian-derived provider; this transport -> 'hll'/'rusanov'");
  }
}

template <class Model>
POPS_COLD_FN BlockClosures make_block(const Model& m, const std::string& lim,
                                      const std::string& riem, const GridContext& ctx,
                                      bool recon_prim, Real pos_floor = Real(0),
                                      bool wave_speed_cache = false, Real weno_eps = kWenoEpsilon) {
  // CENTRALIZED VALIDATION (registry dispatch_tags.hpp) BEFORE the dispatch: same tag acceptances /
  // rejections as before, identical messages (validate_* keeps the historical wording). The flux
  // dispatch now forwards to the per-flux helpers above (each holds the unchanged capability
  // `if constexpr` guard + limiter ladder); the final throw stays a registry/dispatch-inconsistency
  // guard (unreachable after validate_riemann).
  validate_riemann(riem, /*polar=*/false, "System");
  validate_limiter(lim, "System");
  // Parse the validated tag ONCE into the typed RiemannRouteId (ADC-641). Each public provider owns
  // exactly one leaf; the default is a defense-in-depth registry/dispatch guard.
  switch (parse_riemann_route(riem, "System")) {
    case RiemannRouteId::kRusanov:
      return make_block_rusanov(m, lim, ctx, recon_prim, pos_floor, weno_eps);
    case RiemannRouteId::kHll:
      return make_block_hll(m, lim, ctx, recon_prim, pos_floor, wave_speed_cache, weno_eps);
    case RiemannRouteId::kHllc:
      return make_block_hllc(m, lim, ctx, recon_prim, pos_floor, weno_eps);
    case RiemannRouteId::kRoe:
      return make_block_roe(m, lim, ctx, recon_prim, pos_floor, weno_eps);
  }
  throw_registry_dispatch_mismatch("System", "flux", riem);
}

/// Number of ghosts required by the spatial scheme @p lim (single source: Limiter::n_ghost). Used for
/// the allocation of a block state MultiFab, so that the wide WENO5 stencil (5 points, 3 ghosts) does
/// not read out of bounds -- cf. AmrSystem allocates with Limiter::n_ghost (PR #22). Default 2 (MUSCL)
/// for an unknown limiter: that is the historical allocation, hence bit-identical.
inline int block_n_ghost(const std::string& lim) {
  // SINGLE source: limiter_n_ghost(lim) (registry dispatch_tags.hpp). The default 2 (MUSCL) for an
  // unknown limiter is carried by the registry -> same historical allocation, bit-identical. The
  // static_asserts below (this TU sees BOTH the registry AND the types) guarantee that the kLimiters
  // table never drifts from the real::n_ghost constants.
  static_assert(limiter_n_ghost_ct("none") == NoSlope::n_ghost, "kLimiters[none].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("minmod") == Minmod::n_ghost,
                "kLimiters[minmod].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("vanleer") == VanLeer::n_ghost,
                "kLimiters[vanleer].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("weno5") == Weno5::n_ghost, "kLimiters[weno5].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("mc") == MC::n_ghost, "kLimiters[mc].n_ghost drifted");
  static_assert(limiter_n_ghost_ct("superbee") == Superbee::n_ghost,
                "kLimiters[superbee].n_ghost drifted");
  return limiter_n_ghost(lim);
}

namespace detail {
/// Block max wave speed functor (max_wave_speed_mf, reduction over the seam). NAMED FUNCTOR:
/// max_wave_speed_mf instantiates MaxWaveSpeedKernel (already a device functor); wrapping it in a
/// named class rather than a lambda preserves the cross-TU instantiation context under nvcc.
template <class Model>
struct MaxSpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const {
    if (cutcell_geometry_active(ctx))
      return max_wave_speed_mf(m, U, *ctx.aux, *ctx.domain_mask, *ctx.eb_inverse_volume_fraction);
    if (embedded_boundary_active(ctx))
      return max_wave_speed_mf(m, U, *ctx.aux, *ctx.domain_mask);
    return max_wave_speed_mf(m, U, *ctx.aux);
  }
};

/// Block max STABILITY speed functor (HasStabilitySpeed trait): replaces MaxSpeed in the CFL when the
/// model declares stability_speed (the Riemann solvers keep max_wave_speed).
template <class Model>
struct MaxStabilitySpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const {
    if (cutcell_geometry_active(ctx))
      return max_stability_speed_mf(m, U, *ctx.aux, *ctx.domain_mask,
                                    *ctx.eb_inverse_volume_fraction);
    if (embedded_boundary_active(ctx))
      return max_stability_speed_mf(m, U, *ctx.aux, *ctx.domain_mask);
    return max_stability_speed_mf(m, U, *ctx.aux);
  }
};

/// Block max source frequency functor (HasSourceFrequency trait, bound dt <= cfl/mu without h).
template <class Model>
struct MaxSourceFreq {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const {
    return embedded_boundary_active(ctx) ? max_source_frequency_mf(m, U, *ctx.aux, *ctx.domain_mask)
                                         : max_source_frequency_mf(m, U, *ctx.aux);
  }
};

/// Block min admissible step functor (HasStabilityDt trait; 0 = no cell constrains it).
template <class Model>
struct MinStabilityDt {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const {
    return embedded_boundary_active(ctx) ? min_stability_dt_mf(m, U, *ctx.aux, *ctx.domain_mask)
                                         : min_stability_dt_mf(m, U, *ctx.aux);
  }
};

/// Poisson contribution functor: rhs += elliptic_rhs(U), executed by the configured Kokkos backend.
template <class Model>
struct PoissonRhs {
  Model m;
  void operator()(const MultiFab& U, MultiFab& rhs) const { add_model_elliptic_rhs(m, U, rhs); }
};
}  // namespace detail

/// Closure of the speed used by the block CFL step. If the model declares the OPTIONAL stability_speed
/// trait (HasStabilitySpeed), THAT is what drives the CFL (stability lambda*); otherwise STRICT
/// fallback on max_wave_speed (historical behavior, bit-identical). The Riemann solvers always read
/// max_wave_speed: this choice only changes the step policy.
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed(const Model& m, const GridContext& ctx) {
  if constexpr (HasStabilitySpeed<Model>)
    return detail::MaxStabilitySpeed<Model>{m, ctx};
  else
    return detail::MaxSpeed<Model>{m, ctx};
}

/// Closure of the block max source frequency (bound dt <= cfl * substeps / (stride * mu)). EMPTY (null
/// std::function) if the model does not declare the trait -> the Program CFL service ignores it
/// (historical behavior).
template <class Model>
std::function<Real(const MultiFab&)> make_source_frequency(const Model& m, const GridContext& ctx) {
  if constexpr (HasSourceFrequency<Model>)
    return detail::MaxSourceFreq<Model>{m, ctx};
  else
    return {};
}

/// Closure of the block min admissible step (bound dt <= stability_dt * substeps / stride, WITHOUT
/// cfl). EMPTY if the model does not declare the trait -> ignored by the Program CFL service
/// (historical).
template <class Model>
std::function<Real(const MultiFab&)> make_stability_dt(const Model& m, const GridContext& ctx) {
  if constexpr (HasStabilityDt<Model>)
    return detail::MinStabilityDt<Model>{m, ctx};
  else
    return {};
}

/// Block contribution to the Poisson right-hand side: rhs += elliptic_rhs(U) on Kokkos.
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs(const Model& m) {
  return detail::PoissonRhs<Model>{m};
}

namespace detail {
template <int N, class Forward, class Recovery>
auto make_recovery_validated_forward_conversion(Forward forward, Recovery recovery) {
  return [forward = std::move(forward), recovery = std::move(recovery)](const double* in,
                                                                        double* out) {
    double candidate[N] = {};
    double recovered[N] = {};
    forward(in, candidate);
    for (int component = 0; component < N; ++component)
      if (!std::isfinite(candidate[component]))
        throw std::runtime_error(
            "primitive-to-conservative conversion produced a non-finite candidate");
    const RecoveryReport report = recovery(candidate, recovered);
    if (!report.publication_permitted())
      throw std::runtime_error(
          "primitive-to-conservative conversion produced a candidate rejected by prepared "
          "variable recovery");
    for (int component = 0; component < N; ++component)
      out[component] = candidate[component];
  };
}
}  // namespace detail

namespace detail {

template <class Model>
concept HasCharacteristicNoInflow = requires(
    const Model model, const typename Model::State interior, const typename Model::State reference,
    int axis, int side, typename Model::State& ghost) {
  { model.characteristic_no_inflow(interior, reference, axis, side, ghost) } -> std::same_as<bool>;
};

template <class Model>
struct CharacteristicNoInflowPreflightKernel {
  Model model;
  ConstArray4 state;
  typename Model::State reference;
  int axis = 0;
  int side = -1;
  int boundary = 0;

  POPS_HD Real operator()(int i, int j) const {
    const int source_i = axis == 0 ? (side < 0 ? 2 * boundary - i - 1 : 2 * boundary - i + 1) : i;
    const int source_j = axis == 1 ? (side < 0 ? 2 * boundary - j - 1 : 2 * boundary - j + 1) : j;
    const typename Model::State interior = load_state<Model>(state, source_i, source_j);
    typename Model::State ghost{};
    if (!model.characteristic_no_inflow(interior, reference, axis, side, ghost))
      return Real(1);
    for (int component = 0; component < Model::n_vars; ++component)
      if (!std::isfinite(ghost[component]))
        return Real(1);
    return Real(0);
  }
};

template <class Model>
struct CharacteristicNoInflowCommitKernel {
  Model model;
  Array4 state;
  ConstArray4 source;
  typename Model::State reference;
  int axis = 0;
  int side = -1;
  int boundary = 0;

  POPS_HD void operator()(int i, int j) const {
    const int source_i = axis == 0 ? (side < 0 ? 2 * boundary - i - 1 : 2 * boundary - i + 1) : i;
    const int source_j = axis == 1 ? (side < 0 ? 2 * boundary - j - 1 : 2 * boundary - j + 1) : j;
    const typename Model::State interior = load_state<Model>(source, source_i, source_j);
    typename Model::State ghost{};
    const bool accepted = model.characteristic_no_inflow(interior, reference, axis, side, ghost);
    for (int component = 0; component < Model::n_vars; ++component)
      state(i, j, component) = accepted ? ghost[component] : std::numeric_limits<Real>::quiet_NaN();
  }
};

template <class Visitor>
void for_each_characteristic_no_inflow_region(const PreparedHyperbolicBoundary<2>& boundary,
                                              const MultiFab& state, const Box2D& domain,
                                              Visitor&& visitor) {
  const int depth = state.n_grow();
  for (int local = 0; local < state.local_size(); ++local) {
    const Box2D valid = state.box(local);
    int tangential_lo = valid.lo[1] - depth;
    int tangential_hi = valid.hi[1] + depth;
    if (boundary.face(1, -1).law != HyperbolicBoundaryLaw::Periodic)
      tangential_lo = std::max(tangential_lo, domain.lo[1]);
    if (boundary.face(1, 1).law != HyperbolicBoundaryLaw::Periodic)
      tangential_hi = std::min(tangential_hi, domain.hi[1]);
    if (boundary.face(0, -1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow &&
        valid.lo[0] == domain.lo[0])
      visitor(local, 0, -1, domain.lo[0],
              Box2D{{domain.lo[0] - depth, tangential_lo}, {domain.lo[0] - 1, tangential_hi}});
    if (boundary.face(0, 1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow &&
        valid.hi[0] == domain.hi[0])
      visitor(local, 0, 1, domain.hi[0],
              Box2D{{domain.hi[0] + 1, tangential_lo}, {domain.hi[0] + depth, tangential_hi}});

    tangential_lo = valid.lo[0] - depth;
    tangential_hi = valid.hi[0] + depth;
    if (boundary.face(0, -1).law != HyperbolicBoundaryLaw::Periodic)
      tangential_lo = std::max(tangential_lo, domain.lo[0]);
    if (boundary.face(0, 1).law != HyperbolicBoundaryLaw::Periodic)
      tangential_hi = std::min(tangential_hi, domain.hi[0]);
    if (boundary.face(1, -1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow &&
        valid.lo[1] == domain.lo[1])
      visitor(local, 1, -1, domain.lo[1],
              Box2D{{tangential_lo, domain.lo[1] - depth}, {tangential_hi, domain.lo[1] - 1}});
    if (boundary.face(1, 1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow &&
        valid.hi[1] == domain.hi[1])
      visitor(local, 1, 1, domain.hi[1],
              Box2D{{tangential_lo, domain.hi[1] + 1}, {tangential_hi, domain.hi[1] + depth}});
  }
}

template <class Model>
PreparedBoundaryPlan::CharacteristicNoInflowFill make_characteristic_no_inflow_fill(
    const Model& model, const PreparedHyperbolicBoundary<2>& boundary) {
  if (!boundary.has_characteristic_no_inflow())
    return {};
  if constexpr (!HasCharacteristicNoInflow<Model>) {
    throw std::runtime_error(
        "characteristic no-inflow requires the exact block-model flux-Jacobian provider; "
        "no component-wise or Euler-specific fallback exists");
  } else {
    std::array<typename Model::State, 4> references{};
    for (int face = 0; face < 4; ++face) {
      const auto& prepared = boundary.face(face / 2, face % 2 == 0 ? -1 : 1);
      if (prepared.law != HyperbolicBoundaryLaw::CharacteristicNoInflow)
        continue;
      if (prepared.fixed_state.size() != static_cast<std::size_t>(Model::n_vars))
        throw std::runtime_error(
            "characteristic no-inflow reference does not cover the exact model state");
      for (int component = 0; component < Model::n_vars; ++component)
        references[static_cast<std::size_t>(face)][component] =
            prepared.fixed_state[static_cast<std::size_t>(component)];
    }
    return [model, boundary, references](MultiFab& state, const Box2D& domain,
                                         CommunicatorView communicator) {
      const int depth = state.n_grow();
      const bool characteristic_x =
          boundary.face(0, -1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow ||
          boundary.face(0, 1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow;
      const bool characteristic_y =
          boundary.face(1, -1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow ||
          boundary.face(1, 1).law == HyperbolicBoundaryLaw::CharacteristicNoInflow;
      if ((characteristic_x && depth > domain.nx()) || (characteristic_y && depth > domain.ny()))
        throw std::invalid_argument(
            "characteristic no-inflow does not support multi-reflection ghost depth");
      long invalid_local = 0;
      for_each_characteristic_no_inflow_region(
          boundary, state, domain,
          [&](int local, int axis, int side, int coordinate, const Box2D& region) {
            const int face = 2 * axis + (side > 0 ? 1 : 0);
            invalid_local += static_cast<long>(for_each_cell_reduce_sum(
                region, CharacteristicNoInflowPreflightKernel<Model>{
                            model, state.fab(local).const_array(),
                            references[static_cast<std::size_t>(face)], axis, side, coordinate}));
          });
      const long invalid = all_reduce_sum(invalid_local, communicator);
      if (invalid != 0)
        throw std::runtime_error(
            "characteristic no-inflow lost a real prepared spectrum (failed cells=" +
            std::to_string(invalid) + ")");
      for_each_characteristic_no_inflow_region(
          boundary, state, domain,
          [&](int local, int axis, int side, int coordinate, const Box2D& region) {
            const int face = 2 * axis + (side > 0 ? 1 : 0);
            for_each_cell(region,
                          CharacteristicNoInflowCommitKernel<Model>{
                              model, state.fab(local).array(), state.fab(local).const_array(),
                              references[static_cast<std::size_t>(face)], axis, side, coordinate});
          });
    };
  }
}

}  // namespace detail

/// PER-CELL (one cell) cons <-> prim conversions of the MODEL, type-erased over arrays of
/// Model::n_vars doubles. First = primitive -> conservative (M.to_conservative, init from the
/// primitives), second = conservative -> primitive through one PreparedVariableRecovery method.
/// The second closure returns a RecoveryReport and writes its output only after recovery succeeds.
/// Captures the model by value (frozen when the block is added). For a model WITHOUT a conversion
/// (pure scalar, no hyperbolic brick) both formulas are the IDENTITY -- exact for scalar transport
/// (prim == cons) -- while the recovery route still rejects non-finite publication.
/// This flat ABI requires Model::Prim to share the Model::n_vars width of State; make_cell_convert
/// enforces that additional constraint at compile time because HyperbolicPhysicalModel itself only
/// types the forward/inverse maps. Shared by add_block (native) and add_compiled_model (compiled):
/// the SAME conversion serves both paths.
template <class Model>
std::pair<std::function<void(const double*, double*)>,
          std::function<RecoveryReport(const double*, double*)>>
make_cell_convert(const Model& m) {
  constexpr int NV = Model::n_vars;
  const auto recovery_plan = prepare_model_variable_recovery(m);
  if constexpr (HasPrimitiveVars<Model>) {
    static_assert(
        requires { std::integral_constant<int, Model::Prim::size()>{}; },
        "make_cell_convert requires a compile-time primitive-state width");
    if constexpr (requires { std::integral_constant<int, Model::Prim::size()>{}; })
      static_assert(
          Model::Prim::size() == NV,
          "make_cell_convert requires primitive and conservative states to have equal arity");
    auto p2c = [m](const double* in, double* out) {
      typename Model::Prim p{};
      for (int c = 0; c < NV; ++c)
        p[c] = static_cast<Real>(in[c]);
      const typename Model::State u = m.to_conservative(p);
      for (int c = 0; c < NV; ++c)
        out[c] = static_cast<double>(u[c]);
    };
    auto c2p = [recovery_plan](const double* in, double* out) {
      constexpr int N = Model::n_vars;
      Real conserved[N] = {};
      Real initial_guess[N] = {};
      for (int c = 0; c < N; ++c)
        conserved[c] = initial_guess[c] = static_cast<Real>(in[c]);
      const RecoveryOutcome<N> outcome =
          recover_prepared_variable(recovery_plan, conserved, initial_guess);
      if (!outcome.publication_permitted())
        return recovery_report(outcome);
      for (int c = 0; c < N; ++c)
        out[c] = static_cast<double>(outcome.value[c]);
      return recovery_report(outcome);
    };
    auto validated_p2c = detail::make_recovery_validated_forward_conversion<NV>(p2c, c2p);
    return {std::function<void(const double*, double*)>(std::move(validated_p2c)),
            std::function<RecoveryReport(const double*, double*)>(c2p)};
  } else {
    auto p2c = [](const double* in, double* out) {
      for (int c = 0; c < NV; ++c)
        out[c] = in[c];
    };
    auto c2p = [recovery_plan](const double* in, double* out) {
      constexpr int N = Model::n_vars;
      Real conserved[N] = {};
      Real initial_guess[N] = {};
      for (int c = 0; c < N; ++c)
        conserved[c] = initial_guess[c] = static_cast<Real>(in[c]);
      const RecoveryOutcome<N> outcome =
          recover_prepared_variable(recovery_plan, conserved, initial_guess);
      if (!outcome.publication_permitted())
        return recovery_report(outcome);
      for (int c = 0; c < N; ++c)
        out[c] = static_cast<double>(outcome.value[c]);
      return recovery_report(outcome);
    };
    auto validated_p2c = detail::make_recovery_validated_forward_conversion<NV>(p2c, c2p);
    return {std::function<void(const double*, double*)>(std::move(validated_p2c)),
            std::function<RecoveryReport(const double*, double*)>(c2p)};
  }
}

}  // namespace pops
