#pragma once

#include <pops/coupling/amr/amr_coupler_mp.hpp>  // AmrCouplerMP, AmrLevelMP
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // pops::saxpy (level_source = full - flux-only residual, ADC-508)
#include <pops/mesh/layout/refinement.hpp>  // coarsen_index
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>  // SourceFreeModel (flux-only Program primitive)
#include <pops/parallel/comm.hpp>              // n_ranks
#include <pops/runtime/amr/amr_runtime.hpp>    // AmrRuntimeBlock (type-erased multi-block registry)
#include <pops/runtime/amr/composite_reduction.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/block/block_builder.hpp>  // detail::make_poisson_rhs (rhs += elliptic_rhs(U))
#include <pops/runtime/builders/scheme_dispatch.hpp>  // dispatch_limiter: ONE limiter-route dispatch generator (ADC-640)
#include <pops/runtime/config/dispatch_tags.hpp>  // UNIQUE tag registry (validate_limiter/riemann)
#include <pops/runtime/config/route_ids.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// @brief add_compiled_model on the AmrSystem side: wires a COMPILED model (a CompositeModel, generated
///        by the DSL or hand-written, known at COMPILE time) as a block of an AMR hierarchy,
///        EXACTLY the production path of AmrSystem::add_block but WITHOUT going through the ModelSpec
///        dispatch (the model is already a concrete type). One or many compiled/native blocks use
///        the same AmrRuntime engine and materialize each block as an AmrRuntimeBlock.
///
/// Refined counterpart of add_compiled_model(System&, ...) (dsl_block.hpp). The runtime block build
/// is instantiated here on the concrete Model type and enters AmrSystem through one type-erased
/// AmrCompiledBlockBuilder.

namespace pops {

/// Spatial bundle (limiter, Riemann flux) materialized into an AmrRuntimeBlock and consumed by
/// ProgramGraph residual evaluations. Unique definition: the native path of amr_system.cpp goes
/// through this same header (no duplicate DiscLF on the .cpp side).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

template <class Limiter, class Flux, class Model>
void compute_amr_face_fluxes(const Model& model, const MultiFab& state, const MultiFab& aux,
                             MultiFab& flux_x, MultiFab& flux_y, Real dx, Real dy,
                             bool reconstruct_primitive, Real positivity_floor, Real weno_epsilon,
                             const std::shared_ptr<MultiFab>& wave_speed_cache) {
  if constexpr (std::is_same_v<Flux, HLLFlux>) {
    if (wave_speed_cache) {
      compute_face_fluxes_hll_cached<Limiter>(model, state, aux, flux_x, flux_y, *wave_speed_cache,
                                              dx, dy, reconstruct_primitive, positivity_floor,
                                              weno_epsilon);
      return;
    }
  }
  compute_face_fluxes<Limiter, Flux>(model, state, aux, flux_x, flux_y, dx, dy,
                                     reconstruct_primitive, positivity_floor, weno_epsilon);
}

// Projection ponctuelle post-pas appliquee PAR NIVEAU (ADC-177) : miroir de PointwiseProject
// (block_builder.hpp) mais sur la pile de niveaux AMR ; aux = lev.aux (cable par AmrRuntime).
// Defini en tete du namespace pour build_amr_block, situe plus bas (la recherche qualifiee detail::
// exige la declaration AVANT le point d'usage). No-op (else) si le modele ne declare pas m.project.
template <class Model>
void apply_pointwise_project_amr_state(const Model& m, MultiFab& U, const MultiFab& a) {
  if constexpr (HasPointwiseProjection<Model>) {
    for (int li = 0; li < U.local_size(); ++li)
      for_each_cell(U.box(li),
                    ProjectCellKernel<Model>{m, U.fab(li).array(), U.fab(li).const_array(),
                                             a.fab(li).const_array()});
  } else {
    (void)m;
    (void)U;
    (void)a;
  }
}

/// SHARED layout of a multi-block AMR hierarchy, frozen at construction. All
/// blocks allocate their levels on EXACTLY this layout (same BoxArray + DistributionMapping +
/// dx/dy per level) -> same_layout_or_throw passes by construction. The default facade preserves its
/// historical coarse + one central fine seed; the explicit bootstrap can carry any supported count.
/// We expose the BoxArrays /
/// dmaps / dx/dy per level, the coarse grid (Geometry + ba) for the Poisson, and the ownership
/// policy. build_amr_block allocates the block on top of it.
struct SharedAmrLayout {
  Geometry geom;                        // geometry of the coarse level (Poisson)
  BoxArray ba_coarse;                   // BoxArray of the coarse grid
  DistributionMapping dm_coarse;        // DistributionMapping of the coarse grid
  std::vector<BoxArray> ba;             // [level] shared BoxArray (coarse + fines)
  std::vector<DistributionMapping> dm;  // [level] shared DistributionMapping
  std::vector<Real> dx, dy;             // [level] mesh spacing
  std::vector<int> refinement_ratios;   // transition k -> k+1
  std::shared_ptr<const PreparedLoadBalanceAuthority>
      load_balance;               // exact authority for all future ownership decisions
  bool replicated_coarse = true;  // ownership of level 0
  BCRec poisson_bc;               // BC of the coarse Poisson
  ActiveRegionProvider2D wall;    // conducting-wall predicate (empty = none)
  int nx = 128;                   // coarse cells along x
  int ny = 128;                   // coarse cells along y
  Periodicity base_per{};         // periodicity of the base domain
  /// Per-block prepared boundary authorities owned by AmrSystem::Impl. The map and plans outlive
  /// every deferred block builder/closure.
  const std::map<std::string, std::shared_ptr<PreparedBoundaryPlan>>* boundary_plans = nullptr;

  int nlev() const { return static_cast<int>(ba.size()); }

  AmrHierarchyLayout runtime_hierarchy() const {
    return AmrHierarchyLayout{ba, dm, dx, dy, refinement_ratios, load_balance};
  }
};

/// Builds a ratio-2 shared hierarchy with an explicit level count.  Every fine seed is the central
/// half of its parent patch, refined into the child's index space.  This is the native bootstrap for
/// already N-level-generic transport/reflux/runtime kernels; public hierarchy lowering owns the
/// eventual authored BoxArrays and may replace these deterministic seeds before execution.
inline SharedAmrLayout make_shared_amr_layout_levels(const AmrBuildParams& bp, int level_count) {
  if (level_count < 1)
    throw std::runtime_error("make_shared_amr_layout_levels: level_count must be >= 1");
  if (!bp.mesh.load_balance)
    throw std::invalid_argument(
        "make_shared_amr_layout_levels requires a prepared load-balance authority");
  SharedAmrLayout S;
  S.load_balance = bp.mesh.load_balance;
  const int base_nx = bp.mesh.cells_x(), base_ny = bp.mesh.cells_y();
  const double Lx = bp.mesh.length_x(), Ly = bp.mesh.length_y();
  S.geom = Geometry{Box2D::from_extents(base_nx, base_ny), bp.mesh.xlo, bp.mesh.xlo + Lx,
                    bp.mesh.ylo, bp.mesh.ylo + Ly};
  S.nx = base_nx;
  S.ny = base_ny;
  S.base_per = bp.mesh.periodicity;
  S.replicated_coarse = !bp.mesh.distribute_coarse;
  S.poisson_bc = bp.poisson.bc;
  S.wall = bp.poisson.wall;
  const double dxc = Lx / base_nx, dyc = Ly / base_ny;
  const auto [bac, dmc] = detail::coupler_make_coarse_layout(
      base_nx, base_ny, bp.mesh.distribute_coarse, bp.mesh.coarse_max_grid, *bp.mesh.load_balance);
  S.ba_coarse = bac;
  S.dm_coarse = dmc;
  S.ba = {bac};
  S.dm = {dmc};
  S.dx = {dxc};
  S.dy = {dyc};
  Box2D parent_seed = S.geom.domain;
  double spacing_x = dxc, spacing_y = dyc;
  for (int level = 1; level < level_count; ++level) {
    const int nx = parent_seed.nx(), ny = parent_seed.ny();
    if (nx < 4 || ny < 4)
      throw std::runtime_error("make_shared_amr_layout_levels: cannot create level " +
                               std::to_string(level) +
                               " because the parent seed is smaller than 4 cells per axis");
    const Box2D selected{
        {parent_seed.lo[0] + nx / 4, parent_seed.lo[1] + ny / 4},
        {parent_seed.lo[0] + (3 * nx) / 4 - 1, parent_seed.lo[1] + (3 * ny) / 4 - 1}};
    const Box2D fine_seed = selected.refine(kAmrRefRatio);
    BoxArray fine_ba(std::vector<Box2D>{fine_seed});
    DistributionMapping fine_dm = bp.mesh.load_balance->distribute(fine_ba, n_ranks());
    spacing_x /= static_cast<double>(kAmrRefRatio);
    spacing_y /= static_cast<double>(kAmrRefRatio);
    S.ba.push_back(fine_ba);
    S.dm.push_back(fine_dm);
    S.dx.push_back(spacing_x);
    S.dy.push_back(spacing_y);
    S.refinement_ratios.push_back(kAmrRefRatio);
    parent_seed = fine_seed;
  }
  return S;
}

/// Historical facade: one level for the Program parity route, otherwise the unchanged two-level
/// seed.  Keeping this wrapper preserves every existing caller while the final hierarchy lowering
/// adopts make_shared_amr_layout_levels with its resolved transition count.
inline SharedAmrLayout make_shared_amr_layout(const AmrBuildParams& bp, bool single_level = false) {
  return make_shared_amr_layout_levels(bp, single_level ? 1 : 2);
}

/// Builds ONE type-erased AMR block (AmrRuntimeBlock) on the SHARED layout @p S, for a composite
/// Model + concrete (Limiter, Flux). It allocates the level
/// stack of the block on the SAME BoxArray/dmap as all the others (guarantees same_layout_or_throw),
/// sets the complete initial state when provided (otherwise density component 0) + coarse->fine
/// injection, and CAPTURES the concrete spatial scheme in Program primitives and Poisson providers.
/// The kernel stays COMPILED; only the block list is type-erased (AMR analog of make_block /
/// PoissonRhs on the flat System side). @p substeps and @p stride remain authored cadence metadata;
/// normalized Programs, never this builder, decide temporal ordering.
template <class Model, class Limiter, class Flux>
AmrRuntimeBlock build_amr_block(const Model& model, const SharedAmrLayout& S,
                                const std::string& name, const std::vector<double>& density,
                                bool has_density, double gamma, int substeps, bool recon_prim,
                                int stride = 1, const std::vector<double>* state = nullptr,
                                double pos_floor = 0.0,
                                double weno_epsilon = static_cast<double>(kWenoEpsilon),
                                bool wave_speed_cache = false) {
  const int nc = Model::n_vars;
  const int ng = Limiter::n_ghost;
  const int nlev = S.nlev();
  std::shared_ptr<const PreparedBoundaryPlan> boundary_plan;
  if (S.boundary_plans != nullptr) {
    auto found = S.boundary_plans->find(name);
    if (found != S.boundary_plans->end())
      boundary_plan = found->second;
  }
  BCRec transport_bc;
  if (!S.base_per.x)
    transport_bc.xlo = transport_bc.xhi = BCType::Foextrap;
  if (!S.base_per.y)
    transport_bc.ylo = transport_bc.yhi = BCType::Foextrap;
  auto transport_boundary_fill = std::make_shared<const AmrBoundaryFillAuthority>(
      make_amr_boundary_fill_authority(transport_bc));
  auto boundary_field_registry = std::make_shared<GridContext::BoundaryFieldRegistryFactory>();
  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->reserve(nlev);
  for (int k = 0; k < nlev; ++k) {
    MultiFab U(S.ba[k], S.dm[k], nc, ng);
    U.set_val(Real(0));
    levels->push_back(AmrLevelMP{std::move(U), nullptr, S.dx[k], S.dy[k]});
  }
  // Coarse seed + conservative-linear prolongation to the fines: COMPLETE CONSERVATIVE STATE
  // (set_conservative_state, preferred) otherwise density (component 0, rest at rest) otherwise zero.
  if (state && !state->empty())
    detail::coupler_write_coarse_state((*levels)[0].U, *state, S.nx, S.ny, nc);
  else if (has_density)
    detail::coupler_write_coarse((*levels)[0].U, density, S.nx, S.ny, nc, gamma);
  for (int k = 1; k < nlev; ++k)
    detail::coupler_inject_coarse_to_fine_mb(
        (*levels)[k - 1].U, (*levels)[k].U, amr_level_index_domain(S.geom.domain, k - 1),
        amr_level_index_domain(S.geom.domain, k), (k == 1) && S.replicated_coarse, S.base_per);

  AmrRuntimeBlock b;
  b.name = name;
  b.ncomp = nc;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.aux_ncomp = aux_comps<Model>();  // aux width READ by the model (B_z/T_e -> > kAuxBaseComps)
  b.reconstruction_order = Limiter::formal_order;
  b.reconstruction_ghost_depth = Limiter::n_ghost;
  b.cons_vars =
      Model::conservative_vars();  // names + ROLES: role resolution -> component of coupled sources
  b.levels = levels;
  b.boundary_plan = boundary_plan;
  b.boundary_field_registry = boundary_field_registry;
  b.transport_boundary_fill = transport_boundary_fill;
  const bool rprim = recon_prim;
  const Real pf = static_cast<Real>(pos_floor);
  const Real weps = static_cast<Real>(weno_epsilon);
  std::shared_ptr<MultiFab> ws_cache =
      wave_speed_cache ? std::make_shared<MultiFab>() : std::shared_ptr<MultiFab>{};
  // Projection is a typed Program primitive over candidate state, never a live-state post-step hook.
  if constexpr (HasPointwiseProjection<Model>)
    b.project_level_state = [model](MultiFab& U, const MultiFab& aux) {
      detail::apply_pointwise_project_amr_state(model, U, aux);
    };
  // Contribution of the block to the SUMMED Poisson RHS: rhs += elliptic_rhs(U) on the coarse grid
  // through the generic Kokkos pointwise primitive. SAME functor as the flat System -> each
  // block accumulates (+=) into the SAME cells of the shared coarse grid (per-cell co-location).
  b.add_elliptic_rhs = make_poisson_rhs(model);
  // PER-LEVEL SEMI-DISCRETE RESIDUAL (epic ADC-508, compiled-Program AMR driver): R <- -div F + S over a
  // level's grid, reusing BlockRhsEval<Limiter, Flux, Model> -- the SAME device-clean evaluator System
  // wires for block_rhs_into. The closure builds a per-call GridContext from the passed-in level
  // geometry + shared aux (the AmrProgramContext hands it the current level's metric and aux_[k]) so the
  // ONE closure serves every level. The transport BC is derived from the base periodicity (periodic ->
  // periodic ghosts; non-periodic -> Foextrap), matching System::make_bc. The recon_prim flag matches
  // the block's transport. Device contract: BlockRhsEval is a named functor (no cross-TU extended
  // lambda), instantiated HERE on the concrete Model/Limiter/Flux, so the kernel stays compiled and runs
  // Serial / OpenMP / CUDA identically. These closures are read only by an installed Program.
  {
    const BCRec tbc = transport_bc;
    b.level_rhs = [model, rprim, pf, weps, ws_cache, tbc, boundary_plan](
                      MultiFab& U, const MultiFab& aux, const Geometry& geom, MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      detail::BlockRhsEval<Limiter, Flux, Model>{model, &gc, rprim, pf, ws_cache, weps}(U, R);
    };
    b.level_rhs_at_point =
        [model, rprim, pf, weps, ws_cache, tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& R) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::BlockRhsEval<Limiter, Flux, Model>{model, &gc, rprim, pf, ws_cache, weps}(point,
                                                                                            U, R);
        };
    b.level_neg_div_flux = [model, rprim, pf, weps, ws_cache, tbc, boundary_plan](
                               MultiFab& U, const MultiFab& aux, const Geometry& geom,
                               MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      detail::BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>>{
          SourceFreeModel<Model>{model}, &gc, rprim, pf, ws_cache, weps}(U, R);
    };
    b.level_neg_div_flux_at_point =
        [model, rprim, pf, weps, ws_cache, tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& R) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>>{
              SourceFreeModel<Model>{model}, &gc, rprim, pf, ws_cache, weps}(point, U, R);
        };
    b.level_rhs_core_at_point = [model, rprim, pf, weps, ws_cache, tbc, boundary_plan,
                                 boundary_field_registry](
                                    const runtime::multiblock::BoundaryEvaluationPoint& point,
                                    MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                    MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      gc.boundary_field_registry = *boundary_field_registry;
      detail::RhsCoreInto<Limiter, Flux, Model>{model, gc, rprim, pf, ws_cache, weps}(point, U, R);
    };
    b.level_neg_div_flux_core_at_point =
        [model, rprim, pf, weps, ws_cache, tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& R) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::RhsCoreInto<Limiter, Flux, SourceFreeModel<Model>>{
              SourceFreeModel<Model>{model}, gc, rprim, pf, ws_cache, weps}(point, U, R);
        };
    b.level_boundary_residual_at_point =
        [tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& C) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          add_grid_boundary_residual(U, C, gc, point);
        };
    b.level_boundary_jvp_at_point = [tbc, boundary_plan, boundary_field_registry](
                                        const runtime::multiblock::BoundaryEvaluationPoint& point,
                                        MultiFab& U, const MultiFab& V, const MultiFab& aux,
                                        const Geometry& geom, MultiFab& J) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      gc.boundary_field_registry = *boundary_field_registry;
      apply_grid_boundary_jvp(U, V, J, gc, point);
    };
    b.level_rhs_core_at_point_prepared =
        [model, rprim, pf, weps, ws_cache, tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& R,
            const PreparedGridBoundarySession& boundary) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::RhsCoreInto<Limiter, Flux, Model>{model, gc, rprim, pf, ws_cache, weps}(
              point, U, R, boundary);
        };
    b.level_neg_div_flux_core_at_point_prepared =
        [model, rprim, pf, weps, ws_cache, tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& R,
            const PreparedGridBoundarySession& boundary) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::RhsCoreInto<Limiter, Flux, SourceFreeModel<Model>>{
              SourceFreeModel<Model>{model}, gc, rprim, pf, ws_cache, weps}(point, U, R, boundary);
        };
    b.level_boundary_residual_at_point_prepared =
        [tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& C,
            const PreparedGridBoundarySession& boundary) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::BoundaryResidualInto{gc}(point, U, C, boundary);
        };
    b.level_boundary_jvp_at_point_prepared =
        [tbc, boundary_plan, boundary_field_registry](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& V, const MultiFab& aux, const Geometry& geom, MultiFab& J,
            const PreparedGridBoundarySession& boundary) {
          GridContext gc;
          gc.dom = geom.domain;
          gc.bc = tbc;
          gc.geom = geom;
          gc.aux = const_cast<MultiFab*>(&aux);
          gc.boundary_plan = boundary_plan;
          gc.boundary_field_registry = *boundary_field_registry;
          detail::BoundaryJvpInto{gc}(point, U, V, J, boundary);
        };
    if (boundary_plan && boundary_plan->has_omitted_faces()) {
      b.level_rhs_without_prepared_interfaces = b.level_rhs_at_point;
      b.level_neg_div_flux_without_prepared_interfaces = b.level_neg_div_flux_at_point;
    }
    // SOURCE-ONLY: R <- S(U, aux) directly through the same pointwise source kernel as System.
    // A source evaluation has no stencil and therefore must not manufacture two transport RHS
    // evaluations merely to subtract their fluxes: doing so would require an unrelated boundary
    // evaluation point and could change the source when a prepared boundary is installed.
    b.level_source = [model, tbc, boundary_plan](MultiFab& U, const MultiFab& aux,
                                                 const Geometry& geom, MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      detail::SourceInto<Model>{model, gc}(U, R);
    };
    // CONSERVATIVE-REFLUX CAPTURE (ADC-639): the flux-materialising twin of level_rhs / level_neg_div_flux.
    // Instead of the fused assemble_rhs (which computes -div F and DISCARDS the face fluxes), it writes the
    // face fluxes with compute_face_fluxes<Limiter, Flux> THEN derives R with mf_eval_rhs from those SAME
    // fluxes. compute_face_fluxes uses the identical reconstruction + numerical flux as assemble_rhs, so R
    // == the fused level_rhs residual bit-for-bit (face_flux.hpp:236-238) while Fx/Fy stay visible to the
    // reflux register. The physical ghost fill (fill_ghosts, the SAME BlockRhsEval does before assembling)
    // is done here first so the flux at the domain boundary matches the fused path; the fine-level C/F ghost
    // refresh is done by the caller (AmrRuntime::level_rhs_capture_into, like level_rhs_into). Fx/Fy are
    // sized by the caller (xface_box/yface_box, ncomp = Model::n_vars, 0 ghost). recon_prim + the level
    // metric match level_rhs. Read ONLY on the reflux path (nlev>1). Same <Limiter, Flux, Model> capture.
    b.level_flux_capture = [model, rprim, pf, weps, ws_cache, tbc, boundary_plan](
                               MultiFab& U, const MultiFab& aux, const Geometry& geom, MultiFab& Fx,
                               MultiFab& Fy, MultiFab& R) {
      if (boundary_plan)
        throw std::logic_error(
            "resolved AMR reflux boundary plan requires its persistent prepared session");
      pops::fill_ghosts(U, geom.domain, tbc);
      detail::compute_amr_face_fluxes<Limiter, Flux>(model, U, aux, Fx, Fy, geom.dx(), geom.dy(),
                                                     rprim, pf, weps, ws_cache);
      pops::mf_eval_rhs(model, U, aux, Fx, Fy, geom.dx(), geom.dy(), R);
    };
    b.level_flux_capture_neg_div = [model, rprim, pf, weps, ws_cache, tbc, boundary_plan](
                                       MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                       MultiFab& Fx, MultiFab& Fy, MultiFab& R) {
      const SourceFreeModel<Model> sm{model};
      if (boundary_plan)
        throw std::logic_error(
            "resolved AMR reflux boundary plan requires its persistent prepared session");
      pops::fill_ghosts(U, geom.domain, tbc);
      detail::compute_amr_face_fluxes<Limiter, Flux>(sm, U, aux, Fx, Fy, geom.dx(), geom.dy(),
                                                     rprim, pf, weps, ws_cache);
      pops::mf_eval_rhs(sm, U, aux, Fx, Fy, geom.dx(), geom.dy(), R);
    };
    b.level_flux_capture_prepared = [model, rprim, pf, weps, ws_cache](
                                        const runtime::multiblock::BoundaryEvaluationPoint& point,
                                        MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                        MultiFab& Fx, MultiFab& Fy, MultiFab& R,
                                        const PreparedGridBoundarySession& boundary) {
      boundary.fill_same_level_and_physical(U, point);
      detail::compute_amr_face_fluxes<Limiter, Flux>(model, U, aux, Fx, Fy, geom.dx(), geom.dy(),
                                                     rprim, pf, weps, ws_cache);
      detail::zero_prepared_interface_fluxes(Fx, Fy, boundary.context());
      pops::mf_eval_rhs(model, U, aux, Fx, Fy, geom.dx(), geom.dy(), R);
    };
    b.level_flux_capture_neg_div_prepared =
        [model, rprim, pf, weps, ws_cache](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
            const MultiFab& aux, const Geometry& geom, MultiFab& Fx, MultiFab& Fy, MultiFab& R,
            const PreparedGridBoundarySession& boundary) {
          const SourceFreeModel<Model> sm{model};
          boundary.fill_same_level_and_physical(U, point);
          detail::compute_amr_face_fluxes<Limiter, Flux>(sm, U, aux, Fx, Fy, geom.dx(), geom.dy(),
                                                         rprim, pf, weps, ws_cache);
          detail::zero_prepared_interface_fluxes(Fx, Fy, boundary.context());
          pops::mf_eval_rhs(sm, U, aux, Fx, Fy, geom.dx(), geom.dy(), R);
        };
  }
  // CFL SPEED of the block: SAME policy as System (make_max_speed) -- stability lambda*
  // (HasStabilitySpeed trait) if the model declares it, otherwise max_wave_speed (historical fallback,
  // bit-identical). The Riemann solvers always read max_wave_speed.
  if constexpr (HasStabilitySpeed<Model>) {
    b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
      return max_stability_speed_mf(model, U, aux);
    };
  } else {
    b.max_speed = [model](const MultiFab& U, const MultiFab& aux) {
      return max_wave_speed_mf(model, U, aux);
    };
  }
  // OPTIONAL STEP BOUNDS (AMR StabilityPolicy): same reductions as System
  // (max_source_frequency_mf / min_stability_dt_mf), evaluated by Program-owned CFL selection on
  // the COARSE grid. Closures left EMPTY when the model does not declare the trait.
  if constexpr (HasSourceFrequency<Model>) {
    b.source_frequency = [model](const MultiFab& U, const MultiFab& aux) {
      return max_source_frequency_mf(model, U, aux);
    };
  }
  if constexpr (HasStabilityDt<Model>) {
    b.stability_dt = [model](const MultiFab& U, const MultiFab& aux) {
      return min_stability_dt_mf(model, U, aux);
    };
  }
  const Geometry g = S.geom;
  const bool repl = S.replicated_coarse;
  b.mass = [levels, g, repl] {
    const MultiFab& U = (*levels)[0].U;
    const Real dV = g.dx() * g.dy();
    Real M = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      M += for_each_cell_reduce_sum(U.box(li),
                                    [u, dV] POPS_HD(int i, int j) { return u(i, j, 0) * dV; });
    }
    return repl ? M : all_reduce_sum(M);
  };
  const int base_nx = S.nx, base_ny = S.ny;
  b.density = [levels, base_nx, base_ny, repl] {
    return detail::coupler_read_coarse((*levels)[0].U, base_nx, base_ny, repl);
  };
  b.potential = [base_nx, base_ny, repl](const MultiFab& aux0) {
    return detail::coupler_read_coarse_phi(aux0, base_nx, base_ny, repl);
  };
  return b;
}

// ADC-359 per-flux branches of dispatch_amr_block, factored so the compressible AMR seam compiles ONE
// flux per TU (build_amr_block_for_flux -> these). Each body is the corresponding `if (riem == "<flux>")`
// branch of dispatch_amr_block VERBATIM (same leaves, same hllc/roe `if constexpr` capability guards, same
// messages); validate_riemann/limiter run in the caller (dispatch_amr_block, or the compressible thin
// dispatcher python/amr_block_compressible.cpp). dispatch_amr_block (below, unchanged) still serves the
// exb/isothermal seam, where the if constexpr guards prune hllc/roe.
template <class Model>
AmrRuntimeBlock dispatch_amr_block_rusanov(const Model& m, const std::string& lim,
                                           const SharedAmrLayout& S, const std::string& name,
                                           const std::vector<double>& density, bool has_density,
                                           double gamma, int substeps, bool recon_prim, int stride,
                                           const std::vector<double>* state, double pos_floor,
                                           double weno_epsilon, bool wave_speed_cache) {
  return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                          "add_block(AmrSystem, multi-block)", [&](auto tag) {
                            using L = typename decltype(tag)::type;
                            return build_amr_block<Model, L, RusanovFlux>(
                                m, S, name, density, has_density, gamma, substeps, recon_prim,
                                stride, state, pos_floor, weno_epsilon, wave_speed_cache);
                          });
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_hll(const Model& m, const std::string& lim,
                                       const SharedAmrLayout& S, const std::string& name,
                                       const std::vector<double>& density, bool has_density,
                                       double gamma, int substeps, bool recon_prim, int stride,
                                       const std::vector<double>* state, double pos_floor,
                                       double weno_epsilon, bool wave_speed_cache) {
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, HLLFlux>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  stride, state, pos_floor, weno_epsilon, wave_speed_cache);
                            });
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'hll' requires signed wave "
        "speeds (model.wave_speeds); this transport -> 'rusanov'");
  }
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_hllc(const Model& m, const std::string& lim,
                                        const SharedAmrLayout& S, const std::string& name,
                                        const std::vector<double>& density, bool has_density,
                                        double gamma, int substeps, bool recon_prim, int stride,
                                        const std::vector<double>* state, double pos_floor,
                                        double weno_epsilon, bool wave_speed_cache) {
  if constexpr (HasHLLCStructure<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, HLLCFlux>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  stride, state, pos_floor, weno_epsilon, wave_speed_cache);
                            });
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'hllc' requires the model's HLLC capability "
        "(pressure + wave_speeds + contact_speed + hllc_star_state, cf. "
        "HasHLLCStructure); this transport -> "
        "'hll'/'rusanov'");
  }
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_roe(const Model& m, const std::string& lim,
                                       const SharedAmrLayout& S, const std::string& name,
                                       const std::vector<double>& density, bool has_density,
                                       double gamma, int substeps, bool recon_prim, int stride,
                                       const std::vector<double>* state, double pos_floor,
                                       double weno_epsilon, bool wave_speed_cache) {
  if constexpr (HasRoeDissipation<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, RoeFlux>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  stride, state, pos_floor, weno_epsilon, wave_speed_cache);
                            });
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'roe' requires the model's Roe capability "
        "(roe_dissipation, cf. HasRoeDissipation); "
        "this transport -> 'hll'/'rusanov'");
  }
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> build_amr_block. HLLC/Roe require
/// the model's exact Riemann capability HasHLLCStructure / HasRoeDissipation. Time integration and
/// implicit solves are not part of this spatial seam.
template <class Model>
AmrRuntimeBlock dispatch_amr_block(const Model& m, const std::string& lim, const std::string& riem,
                                   const SharedAmrLayout& S, const std::string& name,
                                   const std::vector<double>& density, bool has_density,
                                   double gamma, int substeps, bool recon_prim, int stride = 1,
                                   const std::vector<double>* state = nullptr,
                                   double pos_floor = 0.0,
                                   double weno_epsilon = static_cast<double>(kWenoEpsilon),
                                   bool wave_speed_cache = false) {
  // CENTRALIZED VALIDATION (dispatch_tags.hpp registry) BEFORE the dispatch: same tags accepted /
  // rejected as before, identical messages. The template if/else dispatch that follows is UNCHANGED; the
  // capability guards (hllc/roe: 2D Euler or capability) stay `if constexpr` PER MODEL.
  validate_riemann(riem, /*polar=*/false, "add_block(AmrSystem, multi-block)");
  validate_limiter(lim, "add_block(AmrSystem, multi-block)");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::runtime_error("add_block(AmrSystem, multi-block): finite weno_epsilon > 0 required");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon) && lim != "weno5")
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): weno_epsilon applies to limiter='weno5' only");
  if (wave_speed_cache && riem != "hll")
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): wave_speed_cache requires flux='hll'");
  // ADC-359: delegate to the flux-pinned dispatch_amr_block_<flux> helpers above (factored so the
  // compressible seam compiles one flux per TU). Behavior is unchanged: same leaves, same hllc/roe
  // capability guards, same throws. exb/isothermal route here as before (their guards prune hllc/roe).
  // ADC-641: parse the validated tag ONCE into the typed RiemannRouteId; the switch decodes it and the
  switch (parse_riemann_route(riem, "add_block(AmrSystem, multi-block)")) {
    case RiemannRouteId::kRusanov:
      return dispatch_amr_block_rusanov(m, lim, S, name, density, has_density, gamma, substeps,
                                        recon_prim, stride, state, pos_floor, weno_epsilon,
                                        wave_speed_cache);
    case RiemannRouteId::kHll:
      return dispatch_amr_block_hll(m, lim, S, name, density, has_density, gamma, substeps,
                                    recon_prim, stride, state, pos_floor, weno_epsilon,
                                    wave_speed_cache);
    case RiemannRouteId::kHllc:
      return dispatch_amr_block_hllc(m, lim, S, name, density, has_density, gamma, substeps,
                                     recon_prim, stride, state, pos_floor, weno_epsilon,
                                     wave_speed_cache);
    case RiemannRouteId::kRoe:
      return dispatch_amr_block_roe(m, lim, S, name, density, has_density, gamma, substeps,
                                    recon_prim, stride, state, pos_floor, weno_epsilon,
                                    wave_speed_cache);
  }
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "flux", riem);
}

}  // namespace detail

/// Wires @p model (concrete CompositeModel) as an AMR block of @p sys, with the requested scheme. The
/// build is DEFERRED (like add_block): the captured closures are invoked at the first
/// step/mass/density via ensure_built(), after set_refinement / set_poisson / set_density.
///
/// Every block count materializes the same type-erased AmrRuntimeBlock on the shared layout.
/// @p time: legacy authoring token normalized into the installed ProgramGraph. The runtime block
/// builder does not select or execute a time method.
/// @p stride: HOLD-THEN-CATCH-UP cadence of the block (1 = each macro-step).
/// @p implicit_vars / @p implicit_roles: compatibility authoring inputs. Every non-empty selector
///   currently fails closed because the AMR target has no typed implicit-source Program primitive.
/// @p pos_floor: Zhang-Shu positivity floor (ADC-322; 0 = inactive, bit-identical). Stored on the block
///   and forwarded to the runtime builder, so the .so floors the Density-role face states like a
///   native add_block.
/// @throws std::runtime_error if the system is already built or if time/recon are out of domain.
template <class Model>
void add_compiled_model(
    AmrSystem& sys, const std::string& name, Model model, const std::string& limiter = "minmod",
    const std::string& riemann = "rusanov", const std::string& recon = "conservative",
    const std::string& time = "explicit", double gamma = static_cast<double>(kPhysicalDefaultGamma),
    int substeps = 1, int stride = 1, const std::vector<std::string>& implicit_vars = {},
    const std::vector<std::string>& implicit_roles = {}, double pos_floor = 0.0,
    double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false) {
  if (substeps < 1)
    throw std::runtime_error("add_compiled_model(AmrSystem): substeps >= 1");
  // The flat loader ABI carries the canonical authoring token. It is retained for Program
  // normalization and introspection; the spatial block builder never executes this method.
  if (time != route_token(TimeRouteId::kExplicitSsprk2) &&
      time != route_token(TimeRouteId::kForwardEuler) &&
      time != route_token(TimeRouteId::kSsprk3) && time != route_token(TimeRouteId::kImex))
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): time '" + time +
        "' unknown (available here: " + std::string(route_token(TimeRouteId::kExplicitSsprk2)) +
        "|" + route_token(TimeRouteId::kForwardEuler) + "|" + route_token(TimeRouteId::kSsprk3) +
        "|" + route_token(TimeRouteId::kImex) + ")");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("add_compiled_model(AmrSystem): recon unknown '" + recon +
                             "' (valid: " + kReconRouteTokensCsv + ")");
  const bool recon_prim = (recon == "primitive");
  const bool imex = (time == "imex");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::runtime_error("add_compiled_model(AmrSystem): finite weno_epsilon > 0 required");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon) && limiter != "weno5")
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): weno_epsilon applies to limiter='weno5' only");
  if (wave_speed_cache && riemann != "hll")
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): wave_speed_cache requires riemann='hll'");
  if (wave_speed_cache && imex)
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): wave_speed_cache is supported by explicit AMR transport "
        "only");
  // The runtime builder captures the concrete Model/scheme and materializes an AmrRuntimeBlock on
  // the shared layout for both one and many blocks. set_compiled_block validates the canonical time
  // token before this deferred spatial builder is invoked; the builder itself receives only spatial
  // metadata. The lambda materializes concrete spatial Program primitives and contains no device
  // kernel.
  auto runtime_builder = [model, limiter, riemann](
                             const detail::SharedAmrLayout& S, const std::string& bname,
                             const std::vector<double>& density, bool has_density,
                             const std::vector<double>& state, bool has_state, double bgamma,
                             int bsub, bool brecon_prim, int bstride,
                             const std::vector<std::string>& /*ivars*/,
                             const std::vector<std::string>& /*iroles*/, double bpos_floor,
                             double bweno_epsilon, bool bwave_speed_cache) {
    // pos_floor (ADC-322): the .so flat ABI now carries the Zhang-Shu floor; forward it to the SAME
    // dispatch_amr_block -> build_amr_block leaf as a native multi-block. Runtime initial state is
    // carried by this deferred builder (it is bound after the .so installs the concrete model).
    return detail::dispatch_amr_block(
        model, limiter, riemann, S, bname, density, has_density, bgamma, bsub, brecon_prim, bstride,
        has_state ? &state : nullptr, bpos_floor, bweno_epsilon, bwave_speed_cache);
  };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(runtime_builder), name,
                         recon_prim, time, stride, implicit_vars, implicit_roles, pos_floor,
                         weno_epsilon, wave_speed_cache);
}

}  // namespace pops
