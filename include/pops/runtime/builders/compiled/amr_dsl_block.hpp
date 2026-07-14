#pragma once

#include <pops/coupling/amr/amr_coupler_mp.hpp>                      // AmrCouplerMP, AmrLevelMP
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/mf_arith.hpp>   // pops::saxpy (level_source = full - flux-only residual, ADC-508)
#include <pops/mesh/layout/refinement.hpp>  // coarsen_index
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>  // SourceFreeModel (explicit IMEX half-step, transport only)
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // backward_euler_source + ImplicitMask (stiff IMEX source)
#include <pops/parallel/comm.hpp>                   // n_ranks
#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntimeBlock (type-erased multi-block registry)
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/block/block_builder.hpp>  // detail::make_poisson_rhs (rhs += elliptic_rhs(U))
#include <pops/runtime/builders/scheme_dispatch.hpp>  // dispatch_limiter: ONE limiter-route dispatch generator (ADC-640)
#include <pops/runtime/config/dispatch_tags.hpp>  // UNIQUE tag registry (validate_limiter/riemann)
#include <pops/runtime/config/route_ids.hpp>

#include <algorithm>  // std::find, std::sort (resolving the partial IMEX mask of a compiled block)
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief add_compiled_model on the AmrSystem side: wires a COMPILED model (a CompositeModel, generated
///        by the DSL or hand-written, known at COMPILE time) as a block of an AMR hierarchy,
///        EXACTLY the production path of AmrSystem::add_block but WITHOUT going through the ModelSpec
///        dispatch (the model is already a concrete type). A SINGLE compiled block -> historical
///        mono-block AmrCouplerMP path (bit-identical); SEVERAL compiled blocks or a MIX of compiled +
///        native (capstone v, multi-block production DSL) -> AmrRuntime runtime engine on the shared
///        hierarchy, the compiled block being materialized there as a type-erased AmrRuntimeBlock.
///
/// Refined counterpart of add_compiled_model(System&, ...) (dsl_block.hpp). The AMR coupler build
/// machinery (AmrCouplerMP<Model> + conservative reflux + regrid) is instantiated HERE, from the CALLING
/// translation unit, on the concrete Model type -- like block_builder.hpp for the flat System.
/// The type-erased closures enter AmrSystem through AmrSystem::set_compiled_block (a non-template method)
/// which freezes TWO builders: the mono-block one (detail::build_amr_compiled / dispatch_amr_compiled,
/// SHARED with the native ModelSpec path of add_block once the type is resolved by detail::dispatch_model)
/// AND the multi-block one (detail::dispatch_amr_block, also SHARED with add_block in native multi-block).

namespace pops {

/// Bundle (limiter, Riemann flux) expected by AmrCouplerMP::step<Disc>. Unique definition: the
/// native path of amr_system.cpp goes through this same header (no more DiscLF duplicated on the .cpp side).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

// Projection ponctuelle post-pas appliquee PAR NIVEAU (ADC-177) : miroir de PointwiseProject
// (block_builder.hpp) mais sur la pile de niveaux AMR ; aux = lev.aux (cable par AmrRuntime).
// Defini en tete du namespace : utilise par build_amr_compiled (mono-bloc) ET build_amr_block
// (multi-bloc natif), tous deux situes plus bas (la recherche qualifiee detail:: exige la
// declaration AVANT le point d'usage). No-op (else) si le modele ne declare pas m.project.
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

template <class Model>
void apply_pointwise_project_amr(const Model& m, std::vector<AmrLevelMP>& levels) {
  if constexpr (HasPointwiseProjection<Model>) {
    for (auto& lev : levels)
      apply_pointwise_project_amr_state(m, lev.U, *lev.aux);
  } else {
    (void)m;
    (void)levels;
  }
}

/// Builds the AMR coupler for a composite Model + concrete (Limiter, Flux) and fills the type-erased
/// hooks. Two levels: coarse + one central seed fine patch, reshaped by the regrid. This is the header
/// counterpart of AmrSystem::Impl::build, instantiated from the calling TU on the Model type. The
/// coarse helpers (layout, write/read/inject) are SHARED with the native path via
/// amr_coupler_mp.hpp (detail::coupler_*), so replicated and distributed follow exactly the same logic.
template <class Model, class Limiter, class Flux>
AmrCompiledHooks build_amr_compiled(const Model& model, const AmrBuildParams& bp) {
  using Coupler = AmrCouplerMP<Model>;
  const int nc = Model::n_vars;
  const Geometry g{Box2D::from_extents(bp.mesh.n, bp.mesh.n), 0.0, bp.mesh.L, 0.0, bp.mesh.L};
  const double dxc = bp.mesh.L / bp.mesh.n, dxf = dxc / 2;
  // Level 0 (coarse): layout decided by the ownership policy (replicated mono-box by default,
  // distributed multi-box if bp.mesh.distribute_coarse). When replicated, dmap = my_rank() everywhere (the box
  // lives on each rank; a round-robin would place it on rank 0 only -> out-of-bounds fab elsewhere,
  // segfault under np>1). The fine seed (allocated below ONLY when refinement is configured) starts on a
  // one-box round-robin dmap (box 0 -> rank 0); the initial regrid REBUILDS it then REDISTRIBUTES
  // round-robin (DistributionMapping(nfine, n_ranks())) -> multi-GPU distribution of the fine patches.
  // When distributed, the coarse is distributed TOO (AMR strong-scaling).
  const auto [bac, dm] =
      coupler_make_coarse_layout(bp.mesh.n, bp.mesh.distribute_coarse, bp.mesh.coarse_max_grid);
  const int ng = Limiter::n_ghost;  // limiter stencil (1 NoSlope, 2 MUSCL): scheme parity
  MultiFab Uc(bac, dm, nc, ng);
  Uc.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(Uc), nullptr, dxc, dxc});
  // Level 1 (central seed fine patch, reshaped by the regrid) is allocated when refinement is configured.
  // With the 1e30 sentinel the build-time regrid below (cpl->regrid(crit)) tags
  //     nothing and amr_regrid_finest is a deliberate no-op on zero tags, so the seed would NEVER be reshaped
  //     or removed -- it would persist as a SINGLE un-chopped fine box on the coarse dmap (box 0 -> rank 0),
  //     dead weight that starves MPI strong-scaling (rank 0 carries its coarse boxes PLUS the whole fine
  //     patch). Gating on refine_threshold keeps the no-refinement hierarchy MONO-LEVEL, so the coarse
  //     distributes cleanly. When refinement IS configured the seed is
  //     allocated and the first build regrid chops + distributes it round-robin exactly as before (UNCHANGED).
  if (bp.regrid.threshold < kAmrRefinementDisabledThreshold) {
    const int I0 = bp.mesh.n / 4, I1 = 3 * bp.mesh.n / 4 - 1, J0 = bp.mesh.n / 4,
              J1 = 3 * bp.mesh.n / 4 - 1;
    Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
    BoxArray baf(std::vector<Box2D>{fb});
    // The single-box fine seed carries its OWN coherent one-entry dmap (round-robin of one box ->
    // rank 0), NOT the coarse dm: with distribute_coarse the coarse dm has one entry per box,
    // so reusing it here pairs a 1-box BoxArray with a longer DistributionMapping (rejected by the
    // MultiFab layout check). Round-robin places the box on rank 0 -- identical to the previous
    // dm[0] (replicated dm[0]==my_rank(), so in serial np=1 both are rank 0; distributed dm[0]==0),
    // so ownership, MPI comms and the trajectory are unchanged. Mirrors make_shared_amr_layout.
    MultiFab Uf(baf, DistributionMapping(baf.size(), n_ranks()), nc, ng);
    Uf.set_val(Real(0));
    levels.push_back({std::move(Uf), nullptr, dxf, dxf});
  }

  auto cpl = std::make_shared<Coupler>(model, g, bac, bp.poisson.bc, std::move(levels),
                                       bp.poisson.wall, !bp.mesh.distribute_coarse);
  // Coarse seed: COMPLETE conservative state (preferred, set_conservative_state) otherwise density
  // only (historical). coupler_inject_coarse_to_fine_mb prolongs ALL components (loop k<nc), so the
  // momentum of the seed propagates freely to the fine levels -- no change of prolongation.
  // has_state==false -> bit-identical density path (NO-DEFAULT-CHANGE).
  if (bp.initial.has_state)
    coupler_write_coarse_state(cpl->coarse(), bp.initial.state, bp.mesh.n, nc);
  else if (bp.initial.has_density)
    coupler_write_coarse(cpl->coarse(), bp.initial.density, bp.mesh.n, nc, bp.physics.gamma);
  auto& Lv = cpl->levels();
  for (std::size_t k = 1; k < Lv.size(); ++k)
    coupler_inject_coarse_to_fine_mb(cpl->coarse(), Lv[k].U, !bp.mesh.distribute_coarse);

  const double thr = bp.regrid.threshold;
  auto crit = [thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; };
  if (cpl->levels().size() > 1)
    cpl->regrid(crit);  // no regrid on a mono-level hierarchy (amr-schur)
  // ADC-645: opt-in COMPOSITE FAC field solve (set_poisson(composite=true)). The coupler's
  // compute_aux gate (2 levels, ONE mono-box fine patch, replicated coarse) silently falls back to
  // Option A outside its scope; surface it HERE, at build (before the first update), as a loud
  // refusal instead -- the caller explicitly opted out of the Option A solve. Checked after the
  // initial regrid so the mono-box condition sees the materialized fine patch. composite=false
  // (default) skips all of this: Option A, bit-identical.
  if (bp.poisson.composite) {
    const bool replicated = !bp.mesh.distribute_coarse;
    const bool two_levels = cpl->levels().size() == 2;
    const bool mono_box_fine = two_levels && cpl->levels()[1].U.box_array().size() == 1;
    if (!replicated || !two_levels || !mono_box_fine)
      throw std::runtime_error(
          "AmrSystem::set_poisson : composite=true requires the coupler's composite scope (2 "
          "levels, ONE mono-box fine patch, replicated coarse) ; got levels=" +
          std::to_string(cpl->levels().size()) + (replicated ? "" : ", distributed coarse") +
          (two_levels && !mono_box_fine ? ", multi-box fine patch" : "") +
          ". Use composite=false (the Option A coarse solve + gradient injection).");
    cpl->set_composite_poisson(true);
    // Composite-FAC knobs: the <= 0 -> kFAC*-default idiom shared with the Schur stage below.
    CompositeFacOptions fo;
    if (bp.poisson.fac_max_iters > 0)
      fo.max_iters = bp.poisson.fac_max_iters;
    if (bp.poisson.fac_fine_sweeps > 0)
      fo.fine_sweeps = bp.poisson.fac_fine_sweeps;
    if (bp.poisson.fac_tol > 0.0)
      fo.tol = static_cast<Real>(bp.poisson.fac_tol);
    if (bp.poisson.fac_coarse_rel_tol > 0.0)
      fo.coarse_rel_tol = static_cast<Real>(bp.poisson.fac_coarse_rel_tol);
    if (bp.poisson.fac_coarse_cycles > 0)
      fo.coarse_cycles = bp.poisson.fac_coarse_cycles;
    fo.verbose = bp.poisson.fac_verbose;
    cpl->set_fac_options(fo);
  }
  // model-NAMED aux (ADC-291): seed the static named fields onto the coupler's shared aux BEFORE the
  // first update/step (like density/B_z seeding). The coupler re-applies them in compute_aux each
  // update, so they persist across regrid and reach every level via the aux injection. Empty -> no-op.
  for (const auto& kv : bp.named_aux.fields)
    cpl->set_named_aux(kv.first, std::vector<Real>(kv.second.begin(), kv.second.end()));
  // ADC-369: per-field aux halo policies (compute_aux applies them after the shared fill).
  for (const auto& kv : bp.named_aux.halo_policies)
    cpl->set_named_aux_bc(kv.first, kv.second);
  cpl->update();

  AmrCompiledHooks h;
  h.coupler_holder = cpl;  // lifetime: the closures capture cpl (shared_ptr)
  const int sub = bp.physics.substeps;
  const bool rprim = bp.physics.recon_prim;
  const bool imex =
      bp.physics.imex;  // implicit stiff source (backward_euler) rather than forward Euler
  const int regrid_every = bp.mesh.regrid_every;
  // NEWTON OPTIONS of the mono-block IMEX source (wave 3): threaded to cpl->step -> advance_amr ->
  // backward_euler_source. DEFAULT {} (newton_options not set) = historical constants (2 iters) ->
  // bit-identical path (2a). Captured BY VALUE (POD) in the h.step closure.
  const NewtonOptions nopts = bp.physics.newton_options;
  // TIME METHOD mono-block: integer of the flat ABI (bp.physics.time_method) -> AmrTimeMethod, threaded
  // to cpl->step -> advance_amr. 0 (default / older .so loader) = historical kEuler, bit-identical.
  const AmrTimeMethod tmethod =
      bp.physics.time_method == 1 ? AmrTimeMethod::kSsprk3 : AmrTimeMethod::kEuler;
  // Zhang-Shu positivity floor (ADC-259): threaded to cpl->step / advance_transport -> advance_amr ->
  // compute_face_fluxes + C/F ghost clamp. bp.physics.pos_floor == 0 (default) -> inactive, bit-identical.
  const Real pf = static_cast<Real>(bp.physics.pos_floor);
  auto step_state = std::make_shared<int>(0);  // step counter shared by the closure
  h.base.step = [cpl, crit, sub, rprim, imex, regrid_every, step_state, nopts, tmethod, model,
                 pf](double dt) {
      if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0)
        cpl->regrid(crit);
      const double h2 = dt / sub;
      // NEWTON OPTIONS threaded to the coupler (mono-block): nopts={} by default => iters=2 historical,
      // bit-identical; non-default nopts (set_density + pops.IMEX(newton_*)) drives the local Newton.
      // tmethod (kEuler default) selects SSPRK3 if requested (time='ssprk3'); kEuler bit-identical.
      for (int s = 0; s < sub; ++s)
        cpl->template step<AmrDiscLF<Limiter, Flux>>(h2, rprim, imex, nopts, tmethod, pf);
      // PROJECTION PONCTUELLE post-pas (ADC-177) PAR NIVEAU, APRES transport + source de tous les
      // substeps. No-op si le modele ne declare pas m.project (HasPointwiseProjection false).
      detail::apply_pointwise_project_amr(model, cpl->levels());
      ++*step_state;
    };
  // RESTORATION of the CADENCE PHASE (IO v1, parity with System::set_clock): AmrSystem::set_clock sets
  // the macro-step counter of the mono-block (the regrid cadence reads *step_state) on restart. Shares the
  // SAME step_state as the step closure above -> the regrid phase resumes exactly. Without the call,
  // *step_state stays at 0 (default, bit-identical).
  h.checkpoint.set_macro_step = [step_state](int s) { *step_state = s; };
  // CFL SPEED: lambda* (HasStabilitySpeed trait) if declared, otherwise max_wave_speed of the coupler
  // (historical fallback, bit-identical) -- SAME policy as System/make_max_speed, evaluated on the
  // COARSE grid (the AMR mono-block CFL lives at the coarse step).
  if constexpr (HasStabilitySpeed<Model>) {
    h.base.max_speed = [cpl, model] {
      return static_cast<double>(max_stability_speed_mf(model, cpl->coarse(), cpl->aux0()));
    };
  } else {
    h.base.max_speed = [cpl] { return static_cast<double>(cpl->max_wave_speed()); };
  }
  // OPTIONAL STEP BOUNDS (AMR mono-block StabilityPolicy): same reductions as System,
  // hooks left EMPTY without the trait (AmrSystem::step_cfl then keeps the historical formula).
  if constexpr (HasSourceFrequency<Model>) {
    h.stability.source_frequency = [cpl, model] {
      return static_cast<double>(max_source_frequency_mf(model, cpl->coarse(), cpl->aux0()));
    };
  }
  if constexpr (HasStabilityDt<Model>) {
    h.stability.stability_dt = [cpl, model] {
      return static_cast<double>(min_stability_dt_mf(model, cpl->coarse(), cpl->aux0()));
    };
  }
  h.base.mass = [cpl] { return static_cast<double>(cpl->mass()); };
  h.base.n_patches = [cpl] {
    auto& L = cpl->levels();
    int count = 0;
    for (std::size_t k = 1; k < L.size(); ++k)
      count += L[k].U.box_array().size();
    return count;
  };
  // Index-space signatures of the fine patches (mono-block counterpart of AmrRuntime::patch_boxes).
  // Captures the SAME cpl as the other hooks (no new lifetime concern), reads the already materialized
  // BoxArray -> query between steps, zero cost on the hot path (h.step untouched).
  h.stability.patch_boxes = [cpl] {
    auto& L = cpl->levels();
    std::vector<pops::PatchBox> out;
    for (std::size_t k = 1; k < L.size(); ++k) {
      const auto& bxs = L[k].U.box_array().boxes();
      for (const pops::Box2D& b : bxs)
        out.push_back(pops::PatchBox{static_cast<int>(k), b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  };
  // Coarse-level (base) box counts (ADC-319, MPI ownership diagnostic): per-rank OWNED fabs of level 0
  // (local_size()) and the GLOBAL base box count (box_array().size()). Same cpl capture as the other
  // hooks (no new lifetime concern); a query between steps, zero cost on the hot path. distribute_coarse
  // -> local < total per rank (distributed coarse transport); replicated/single-box -> local == total.
  h.mpi_gather.coarse_local_boxes = [cpl] { return cpl->coarse().local_size(); };
  h.mpi_gather.coarse_total_boxes = [cpl] { return cpl->coarse().box_array().size(); };
  // AMR CHECKPOINT / RESTART single-rank (ADC-65): COMPLETE conservative state per level + phi
  // (warm-start) + imposing the saved fine hierarchy. Capture the SAME cpl (shared_ptr) as the
  // other hooks (no new lifetime concern). Single-rank: the coupler accessors loop over local_size()
  // (no gather) -- the facade rejects np>1 / multi-block upstream. These hooks are QUERIES/SETTERS
  // between steps: zero cost on the hot path (h.step untouched).
  h.checkpoint.n_levels = [cpl] { return cpl->nlev(); };
  h.checkpoint.n_vars = [] { return Model::n_vars; };
  h.checkpoint.level_state = [cpl](int k) { return cpl->level_state(k); };
  h.checkpoint.set_level_state = [cpl](int k, const std::vector<double>& s) { cpl->set_level_state(k, s); };
  h.checkpoint.level_potential = [cpl](int k) { return cpl->level_potential(k); };
  h.checkpoint.set_level_potential = [cpl](int k, const std::vector<double>& p) {
    cpl->set_level_potential(k, p);
  };
  // GLOBAL (np>1 gather) counterparts (ADC-509): the facade routes to these under MPI np>1 so a
  // bit-identical checkpoint gathers the distributed per-level fabs onto rank 0 (COLLECTIVE, all ranks
  // call). Mono-rank they return the same array as the non-global hooks above (reduce = identity).
  h.mpi_gather.level_state_global = [cpl](int k) { return cpl->level_state_global(k); };
  h.mpi_gather.level_potential_global = [cpl](int k) { return cpl->level_potential_global(k); };
  h.checkpoint.set_hierarchy = [cpl](const std::vector<pops::PatchBox>& boxes) {
    // Mono-block: all patches live at level 1 -> we filter level == 1 and convert to Box2D
    // (INCLUSIVE corners, fine-level index space), then impose this BoxArray on the coupler.
    std::vector<pops::Box2D> fb;
    for (const pops::PatchBox& b : boxes)
      if (b.level == 1)
        fb.push_back(pops::Box2D{{b.ilo, b.jlo}, {b.ihi, b.jhi}});
    cpl->set_hierarchy(fb);
  };
  const int nn = bp.mesh.n;
  const bool repl = !bp.mesh.distribute_coarse;
  h.base.density = [cpl, nn, repl] { return coupler_read_coarse(cpl->coarse(), nn, repl); };
  // Coarse phi: we refresh (update() = sync_down + compute_aux, hence coarse Poisson solve)
  // then read aux0 component 0. Counterpart of System::potential() which calls ensure_elliptic: the
  // value is current even if no step has run yet. update() is already called at each step,
  // so the overhead exists only on a call outside the loop (diagnostic).
  h.base.potential = [cpl, nn, repl] {
    cpl->update();
    return coupler_read_coarse_phi(cpl->aux0(), nn, repl);
  };
  return h;
}

/// SHARED layout of a multi-block AMR hierarchy, frozen at construction. All
/// blocks allocate their levels on EXACTLY this layout (same BoxArray + DistributionMapping +
/// dx/dy per level) -> same_layout_or_throw passes by construction. The default facade preserves its
/// historical coarse + one central fine seed; the explicit bootstrap can carry any supported count.
/// We expose the BoxArrays /
/// dmaps / dx/dy per level, the coarse grid (Geometry + ba) for the Poisson, and the ownership
/// policy. build_amr_block allocates the block on top of it.
struct SharedAmrLayout {
  Geometry geom;                         // geometry of the coarse level (Poisson)
  BoxArray ba_coarse;                    // BoxArray of the coarse grid
  DistributionMapping dm_coarse;         // DistributionMapping of the coarse grid
  std::vector<BoxArray> ba;              // [level] shared BoxArray (coarse + fines)
  std::vector<DistributionMapping> dm;   // [level] shared DistributionMapping
  std::vector<Real> dx, dy;              // [level] mesh spacing
  std::vector<int> refinement_ratios;    // transition k -> k+1
  bool replicated_coarse = true;         // ownership of level 0
  BCRec poisson_bc;                      // BC of the coarse Poisson
  std::function<bool(Real, Real)> wall;  // conducting-wall predicate (empty = none)
  int n = 128;                           // coarse cells per direction
  Periodicity base_per{true, true};      // periodicity of the base domain
  /// Per-block prepared boundary authorities owned by AmrSystem::Impl. The map and plans outlive
  /// every deferred block builder/closure.
  const std::map<std::string, std::shared_ptr<PreparedBoundaryPlan>>* boundary_plans = nullptr;

  int nlev() const { return static_cast<int>(ba.size()); }

  AmrHierarchyLayout runtime_hierarchy() const {
    return AmrHierarchyLayout{ba, dm, dx, dy, refinement_ratios};
  }
};

/// Builds a ratio-2 shared hierarchy with an explicit level count.  Every fine seed is the central
/// half of its parent patch, refined into the child's index space.  This is the native bootstrap for
/// already N-level-generic transport/reflux/runtime kernels; public hierarchy lowering owns the
/// eventual authored BoxArrays and may replace these deterministic seeds before execution.
inline SharedAmrLayout make_shared_amr_layout_levels(const AmrBuildParams& bp, int level_count) {
  if (level_count < 1)
    throw std::runtime_error(
        "make_shared_amr_layout_levels: level_count must be >= 1");
  SharedAmrLayout S;
  S.geom = Geometry{Box2D::from_extents(bp.mesh.n, bp.mesh.n), 0.0, bp.mesh.L, 0.0, bp.mesh.L};
  S.n = bp.mesh.n;
  S.replicated_coarse = !bp.mesh.distribute_coarse;
  S.poisson_bc = bp.poisson.bc;
  S.wall = bp.poisson.wall;
  const double dxc = bp.mesh.L / bp.mesh.n;
  const auto [bac, dmc] =
      detail::coupler_make_coarse_layout(bp.mesh.n, bp.mesh.distribute_coarse,
                                         bp.mesh.coarse_max_grid);
  S.ba_coarse = bac;
  S.dm_coarse = dmc;
  S.ba = {bac};
  S.dm = {dmc};
  S.dx = {dxc};
  S.dy = {dxc};
  Box2D parent_seed = S.geom.domain;
  double spacing = dxc;
  for (int level = 1; level < level_count; ++level) {
    const int nx = parent_seed.nx(), ny = parent_seed.ny();
    if (nx < 4 || ny < 4)
      throw std::runtime_error(
          "make_shared_amr_layout_levels: cannot create level " + std::to_string(level) +
          " because the parent seed is smaller than 4 cells per axis");
    const Box2D selected{{parent_seed.lo[0] + nx / 4, parent_seed.lo[1] + ny / 4},
                         {parent_seed.lo[0] + (3 * nx) / 4 - 1,
                          parent_seed.lo[1] + (3 * ny) / 4 - 1}};
    const Box2D fine_seed = selected.refine(kAmrRefRatio);
    BoxArray fine_ba(std::vector<Box2D>{fine_seed});
    DistributionMapping fine_dm(fine_ba.size(), n_ranks());
    spacing /= static_cast<double>(kAmrRefRatio);
    S.ba.push_back(fine_ba);
    S.dm.push_back(fine_dm);
    S.dx.push_back(spacing);
    S.dy.push_back(spacing);
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
/// Model + concrete (Limiter, Flux). Multi-block counterpart of build_amr_compiled: allocates the level
/// stack of the block on the SAME BoxArray/dmap as all the others (guarantees same_layout_or_throw),
/// sets the initial density (component 0) + coarse->fine injection, and CAPTURES the concrete scheme
/// in the closures (advance via advance_amr<Limiter, Flux>, add_elliptic_rhs via PoissonRhs).
/// The kernel stays COMPILED; only the block list is type-erased (AMR analog of make_block /
/// PoissonRhs on the flat System side). @p density (empty = coarse at zero), @p substeps sub-steps of the
/// block, @p stride hold-then-catch-up cadence of the block (1 = each macro-step). substeps and stride are
/// carried by AmrRuntime::step (the advance closure does just ONE advance_amr): they thus do NOT touch
/// the scheme capture, only the substeps/stride fields of the AmrRuntimeBlock.
///
/// TIME TREATMENT (capstone vii): @p imex selects the SOURCE treatment. We populate
/// TWO distinct closures set on the AmrRuntimeBlock and AmrRuntime::step chooses (b.imex):
///   - advance: AMR transport + EXPLICIT source (forward Euler) -- historical path unchanged;
///   - imex_advance: SOURCE-FREE AMR transport + stiff IMPLICIT source backward_euler_source per
///     level (mask @p implicit_components for partial IMEX) + cascade. The SEMANTICS of the splitting
///     mirror the IMEX branch of AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper), and
///     AT substeps=1 is IDENTICAL to it. This closure does ONE Lie step; AmrRuntime::step calls it
///     substeps times (on the effective step / substeps), so for substeps>1 the runtime SUB-CYCLES the
///     IMEX splitting where compile-time applies it once on the effective step. ASSUMED divergence
///     and sound (cf. IMEX SEMANTICS UNDER substeps in amr_runtime.hpp).
/// @p implicit_components: indices of the components treated IMPLICITLY (partial IMEX, carried by the
/// BLOCK, takes priority over the model default); EMPTY (default) -> inactive mask -> full backward-Euler
/// (all components implicit), bit-identical behavior to IMEX without a mask. Ignored if imex==false.
template <class Model, class Limiter, class Flux>
AmrRuntimeBlock build_amr_block(
    const Model& model, const SharedAmrLayout& S, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps,
    bool recon_prim, bool imex, int stride = 1, const std::vector<int>& implicit_components = {},
    const NewtonOptions& nopts = {}, const std::vector<double>* state = nullptr,
    bool newton_diagnostics = false, AmrTimeMethod time_method = AmrTimeMethod::kEuler,
    double pos_floor = 0.0) {
  const int nc = Model::n_vars;
  const int ng = Limiter::n_ghost;  // limiter stencil (scheme parity, like build_amr_compiled)
  const int nlev = S.nlev();
  std::shared_ptr<const PreparedBoundaryPlan> boundary_plan;
  if (S.boundary_plans != nullptr) {
    auto found = S.boundary_plans->find(name);
    if (found != S.boundary_plans->end())
      boundary_plan = found->second;
  }
  auto boundary_field_registry =
      std::make_shared<GridContext::BoundaryFieldRegistryFactory>();
  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->reserve(nlev);
  for (int k = 0; k < nlev; ++k) {
    MultiFab U(S.ba[k], S.dm[k], nc, ng);
    U.set_val(Real(0));
    levels->push_back(AmrLevelMP{std::move(U), nullptr, S.dx[k], S.dy[k]});
  }
  // Coarse seed + piecewise-constant injection to the fines, exactly like
  // build_amr_compiled: COMPLETE CONSERVATIVE STATE (set_conservative_state, wave 3: now
  // wired in multi-block, preferred) otherwise density (component 0, rest at rest) otherwise zero.
  if (state && !state->empty())
    detail::coupler_write_coarse_state((*levels)[0].U, *state, S.n, nc);
  else if (has_density)
    detail::coupler_write_coarse((*levels)[0].U, density, S.n, nc, gamma);
  for (int k = 1; k < nlev; ++k)
    detail::coupler_inject_coarse_to_fine_mb((*levels)[k - 1].U, (*levels)[k].U,
                                              (k == 1) && S.replicated_coarse);

  AmrRuntimeBlock b;
  b.name = name;
  b.ncomp = nc;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.imex = imex;  // time treatment of the block: selects advance vs imex_advance in step()
  b.aux_ncomp = aux_comps<Model>();  // aux width READ by the model (B_z/T_e -> > kAuxBaseComps)
  b.cons_vars =
      Model::conservative_vars();  // names + ROLES: role resolution -> component of coupled sources
  b.levels = levels;
  b.boundary_plan = boundary_plan;
  b.boundary_field_registry = boundary_field_registry;

  const bool rprim = recon_prim;
  // advance: ONE AMR transport sub-step of the block (conservative Berger-Oliger + reflux + average_down)
  // of size dt, with ITS scheme (Limiter, Flux) on ITS level stack, source in
  // FORWARD EULER (imex=false always here: the IMEX path lives in imex_advance, selected by
  // step()). The sub-step loop (substeps) and the stride cadence are CARRIED by AmrRuntime::step,
  // not by this closure: thus the multirate semantics are in ONE place in the engine (mirror
  // of AmrSystemCoupler::step) and stay disableable / testable there. Implicit FUNCTOR:
  // advance_amr<Limiter, Flux> is a named template function (no cross-TU extended lambda);
  // we capture it in a std::function from THIS TU (device-clean recipe #64/#97).
  // tmethod (kEuler default) selects SSPRK3 (time='ssprk3') for the explicit transport of the block;
  // kEuler -> historical forward Euler, bit-identical. The explicit source stays carried by advance_amr.
  b.advance = [model, rprim, time_method, pos_floor](std::vector<AmrLevelMP>& L,
                                                     const Box2D& dom, Real dt,
                                                     Periodicity per, bool repl) {
    advance_amr<Limiter, Flux>(model, L, dom, dt, per, repl, rprim, /*imex=*/false, NewtonOptions{},
                               time_method, static_cast<Real>(pos_floor));
  };
  // imex_advance (capstone vii): ONE Lie step [source-free transport; implicit source] whose
  // SEMANTICS mirror the IMEX branch of AmrSystemCoupler::step (SourceFreeModel + AmrImplicitSourceStepper),
  // populated ONLY if imex. (1) EXPLICIT transport on the SOURCE-FREE model (SourceFreeModel<Model>:
  // flux/CFL of the model, null source) by the SAME AMR engine (conservative reflux); (2) stiff source
  // IMPLICIT backward_euler_source AT EACH LEVEL (local Newton), with the mask @p implicit_components
  // carried by the BLOCK (partial IMEX); (3) cascade fine -> coarse (mf_average_down_mb) for the coherence
  // of the covered coarse cells. AmrRuntime::step calls this closure substeps times: at
  // substeps=1 this is exactly the compile-time IMEX branch, for substeps>1 the runtime SUB-CYCLES the
  // splitting (assumed decision, cf. IMEX SEMANTICS UNDER substeps in amr_runtime.hpp).
  // We CAPTURE the mask in an ImplicitMask<Model::n_vars> (device-clean POD) once here (the
  // width n_vars is known only at build, the mask is inactive if implicit_components is empty ->
  // full backward-Euler, bit-identical to IMEX without a mask). SourceFreeModel<Model> is a concrete
  // type instantiated IN this TU: its advance_amr<Limiter, Flux> stays compiled (no cross-TU extended
  // lambda), captured in the std::function of identical signature to advance. The reconstruction
  // of the source-free half-step stays CONSERVATIVE (recon_prim=false): SAME choice as AmrSystemCoupler::step
  // (which calls advance_amr on SourceFreeModel with the default), and SourceFreeModel does not expose
  // the primitive variables anyway (cf. its header). The EXPLICIT block, for its part, keeps recon_prim=rprim.
  if (imex) {
    ImplicitMask<Model::n_vars> mask;
    for (int c : implicit_components)
      if (c >= 0 && c < Model::n_vars) {
        mask.active = true;
        mask.flag[c] = true;
      }
    // NEWTON DIAGNOSTICS (wave 3): we allocate the AGGREGATE report of the block in a shared_ptr
    // (STABLE address even after moving the AmrRuntimeBlock into the engine registry) and capture its
    // raw pointer in the imex_advance closure. Explicit diagnostics and fail_policy warn/throw need
    // this report: warn/throw events must be structured, not stderr text. No diagnostics and
    // fail_policy=none -> nreport=nullptr -> backward_euler_source FAST path, bit-identical. The RESET
    // of the report is the responsibility of AmrRuntime::step (head of the block advance), like
    // System::AdvanceImex.
    std::shared_ptr<NewtonReport> nrep;
    if (newton_diagnostics || nopts.fail_policy != NewtonOptions::kFailNone) {
      nrep = std::make_shared<NewtonReport>();
      b.newton_diagnostics = true;
      b.newton_report = nrep;
    }
    NewtonReport* nreport = nrep.get();  // null without diagnostics; stable address otherwise
    b.imex_advance = [model, mask, nopts, nreport, pos_floor](
                         std::vector<AmrLevelMP>& L, const Box2D& dom, Real dt,
                         Periodicity per, bool repl) {
      // (1) explicit source-free transport (-div F only), reflux carries the hyperbolic conservation.
      // The Zhang-Shu floor (ADC-259) applies to the source-free TRANSPORT (the half-step that
      // reconstructs faces); the stiff implicit source backward_euler_source below stays unfloored
      // (cell-local, parity with the uniform System IMEX). SourceFreeModel<Model> forwards
      // conservative_vars(), so positivity_comp resolves the SAME Density-role component.
      advance_amr<Limiter, Flux>(SourceFreeModel<Model>{model}, L, dom, dt, per, repl,
                                 /*recon_prim=*/false, /*imex=*/false, NewtonOptions{},
                                 AmrTimeMethod::kEuler, static_cast<Real>(pos_floor));
      // (2) stiff implicit source backward-Euler PER LEVEL (local Newton, block mask). The report
      // nreport (null without diagnostics) AGGREGATES over the levels: backward_euler_source does its own
      // max/sum + MPI all_reduce into *nreport (no reset here -> it also accumulates over the sub-steps,
      // step() having reset at the head of the advance). nreport==nullptr -> fast bit-identical path.
      const int nlev_l = static_cast<int>(L.size());
      for (int k = 0; k < nlev_l; ++k)
        backward_euler_source<Model>(model, *L[k].aux, L[k].U, dt, nopts, mask, nreport);
      // (3) COVERAGE INVARIANT (cf. AmrImplicitSourceStepper): the implicit source was solved
      // level by level, so a COVERED coarse cell would carry a phantom coarse source
      // instead of the 2x2 average of its children. Cascade fine -> coarse for the coherence (the mass,
      // sum of the coarse grid alone, then does not count the patch source twice). Mono-level: empty loop
      // -> bit-identical. The source remaining CELL-LOCAL (not a face flux), it does NOT enter
      // the reflux registers: conservation at the coarse-fine interfaces stays intact.
      for (int k = nlev_l - 1; k >= 1; --k)
        mf_average_down_mb(L[k].U, L[k - 1].U);
    };
  }
  // PROJECTION PONCTUELLE post-pas (ADC-177) : cablee SEULEMENT si le modele declare m.project
  // (HasPointwiseProjection). AmrRuntime::step l'applique PAR NIVEAU a la FIN de l'avance du bloc
  // (substeps + reflux/cascade faits). Vide sinon -> trajectoire bit-identique. Capture le `model`
  // concret comme advance / imex_advance (foncteur device-clean, pas de lambda etendue cross-TU).
  if constexpr (HasPointwiseProjection<Model>)
    b.project_per_level = [model](std::vector<AmrLevelMP>& L) {
      detail::apply_pointwise_project_amr(model, L);
    };
  if constexpr (HasPointwiseProjection<Model>)
    b.project_level_state = [model](MultiFab& U, const MultiFab& aux) {
      detail::apply_pointwise_project_amr_state(model, U, aux);
    };
  // Contribution of the block to the SUMMED Poisson RHS: rhs += elliptic_rhs(U) on the coarse grid (pure
  // host loop). SAME functor as the flat System (make_poisson_rhs -> detail::PoissonRhs) -> each
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
  // Serial / OpenMP / CUDA identically. These closures are read ONLY by an installed compiled Program;
  // the native AMR step never calls them.
  {
    BCRec tbc;  // transport BC of the level, derived from the base periodicity (parity System::make_bc)
    if (!S.base_per.x)
      tbc.xlo = tbc.xhi = BCType::Foextrap;
    if (!S.base_per.y)
      tbc.ylo = tbc.yhi = BCType::Foextrap;
    b.level_rhs = [model, rprim, tbc, boundary_plan](MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                      MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      detail::BlockRhsEval<Limiter, Flux, Model>{model, &gc, rprim, Real(0), nullptr}(U, R);
    };
    b.level_rhs_at_point = [model, rprim, tbc, boundary_plan, boundary_field_registry](
        const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
        const MultiFab& aux, const Geometry& geom, MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      gc.boundary_field_registry = *boundary_field_registry;
      detail::BlockRhsEval<Limiter, Flux, Model>{model, &gc, rprim, Real(0), nullptr}(
          point, U, R);
    };
    b.level_neg_div_flux = [model, rprim, tbc, boundary_plan](MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                               MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      detail::BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>>{SourceFreeModel<Model>{model}, &gc,
                                                                  rprim, Real(0), nullptr}(U, R);
    };
    b.level_neg_div_flux_at_point = [model, rprim, tbc, boundary_plan,
                                     boundary_field_registry](
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
          SourceFreeModel<Model>{model}, &gc, rprim, Real(0), nullptr}(point, U, R);
    };
    b.level_rhs_core_at_point = [model, rprim, tbc, boundary_plan,
                                 boundary_field_registry](
        const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
        const MultiFab& aux, const Geometry& geom, MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      gc.boundary_field_registry = *boundary_field_registry;
      detail::RhsCoreInto<Limiter, Flux, Model>{
          model, gc, rprim, Real(0), nullptr}(point, U, R);
    };
    b.level_neg_div_flux_core_at_point = [model, rprim, tbc, boundary_plan,
                                          boundary_field_registry](
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
          SourceFreeModel<Model>{model}, gc, rprim, Real(0), nullptr}(point, U, R);
    };
    b.level_boundary_residual_at_point = [tbc, boundary_plan,
                                          boundary_field_registry](
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
    b.level_boundary_jvp_at_point = [tbc, boundary_plan,
                                     boundary_field_registry](
        const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
        const MultiFab& V, const MultiFab& aux, const Geometry& geom, MultiFab& J) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      gc.boundary_field_registry = *boundary_field_registry;
      apply_grid_boundary_jvp(U, V, J, gc, point);
    };
    if (boundary_plan && boundary_plan->has_omitted_faces()) {
      b.level_rhs_without_prepared_interfaces = b.level_rhs_at_point;
      b.level_neg_div_flux_without_prepared_interfaces = b.level_neg_div_flux_at_point;
    }
    // SOURCE-ONLY: R <- S(U, aux) only. Computed as the full residual minus the flux-only residual
    // (R_full - R_flux), reusing the two BlockRhsEval instances above -- no new source kernel (parity
    // with the System split, which evaluates m.source per cell; here R = (-div F + S) - (-div F) = S
    // is bit-identical and device-clean: it is two named-functor residuals + a saxpy).
    b.level_source = [model, rprim, tbc, boundary_plan](MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                         MultiFab& R) {
      GridContext gc;
      gc.dom = geom.domain;
      gc.bc = tbc;
      gc.geom = geom;
      gc.aux = const_cast<MultiFab*>(&aux);
      gc.boundary_plan = boundary_plan;
      detail::BlockRhsEval<Limiter, Flux, Model>{model, &gc, rprim, Real(0), nullptr}(U, R);
      MultiFab Rf(R.box_array(), R.dmap(), R.ncomp(), 0);
      detail::BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>>{SourceFreeModel<Model>{model}, &gc,
                                                                  rprim, Real(0), nullptr}(U, Rf);
      pops::saxpy(R, Real(-1), Rf);  // R <- (-div F + S) - (-div F) = S
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
    b.level_flux_capture = [model, rprim, tbc, boundary_plan](MultiFab& U, const MultiFab& aux, const Geometry& geom,
                                               MultiFab& Fx, MultiFab& Fy, MultiFab& R) {
      if (boundary_plan)
        boundary_plan->fill_same_level_and_physical(U, geom.domain);
      else
        pops::fill_ghosts(U, geom.domain, tbc);
      pops::compute_face_fluxes<Limiter, Flux>(model, U, aux, Fx, Fy, geom.dx(), geom.dy(), rprim);
      pops::mf_eval_rhs(model, U, aux, Fx, Fy, geom.dx(), geom.dy(), R);
    };
    b.level_flux_capture_neg_div = [model, rprim, tbc, boundary_plan](MultiFab& U, const MultiFab& aux,
                                                       const Geometry& geom, MultiFab& Fx, MultiFab& Fy,
                                                       MultiFab& R) {
      const SourceFreeModel<Model> sm{model};
      if (boundary_plan)
        boundary_plan->fill_same_level_and_physical(U, geom.domain);
      else
        pops::fill_ghosts(U, geom.domain, tbc);
      pops::compute_face_fluxes<Limiter, Flux>(sm, U, aux, Fx, Fy, geom.dx(), geom.dy(), rprim);
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
  // (max_source_frequency_mf / min_stability_dt_mf), evaluated by AmrRuntime::step_cfl on the
  // COARSE grid. Closures left EMPTY when the model does not declare the trait (bit-identical).
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
  const int nn = S.n;
  b.density = [levels, nn, repl] { return detail::coupler_read_coarse((*levels)[0].U, nn, repl); };
  b.potential = [nn, repl](const MultiFab& aux0) {
    return detail::coupler_read_coarse_phi(aux0, nn, repl);
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
AmrRuntimeBlock dispatch_amr_block_rusanov(
    const Model& m, const std::string& lim, const SharedAmrLayout& S, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps,
    bool recon_prim, bool imex, int stride, const std::vector<int>& implicit_components,
    const NewtonOptions& nopts, const std::vector<double>* state, bool newton_diagnostics,
    AmrTimeMethod time_method, double pos_floor) {
  return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                          "add_block(AmrSystem, multi-block)", [&](auto tag) {
                            using L = typename decltype(tag)::type;
                            return build_amr_block<Model, L, RusanovFlux>(
                                m, S, name, density, has_density, gamma, substeps, recon_prim, imex,
                                stride, implicit_components, nopts, state, newton_diagnostics,
                                time_method, pos_floor);
                          });
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_hll(const Model& m, const std::string& lim,
                                       const SharedAmrLayout& S, const std::string& name,
                                       const std::vector<double>& density, bool has_density,
                                       double gamma, int substeps, bool recon_prim, bool imex,
                                       int stride, const std::vector<int>& implicit_components,
                                       const NewtonOptions& nopts, const std::vector<double>* state,
                                       bool newton_diagnostics, AmrTimeMethod time_method,
                                       double pos_floor) {
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, HLLFlux>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
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
                                        double gamma, int substeps, bool recon_prim, bool imex,
                                        int stride, const std::vector<int>& implicit_components,
                                        const NewtonOptions& nopts,
                                        const std::vector<double>* state, bool newton_diagnostics,
                                        AmrTimeMethod time_method, double pos_floor) {
  // ADC-590 split, same rationale as dispatch_amr_compiled_hllc: the generic HLLCFlux is
  // capability-only (static_assert without HasHLLCStructure); the canonical Euler layout routes the
  // explicit EulerHLLCFlux2D (bit-identical on the true Euler brick).
  if constexpr (HasHLLCStructure<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, HLLCFlux>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
                            });
  } else if constexpr (Model::n_vars == 4 &&
                       requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, EulerHLLCFlux2D>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
                            });
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'hllc' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's HLLC capability (pressure + wave_speeds + contact_speed + "
        "hllc_star_state, cf. HasHLLCStructure); this transport -> "
        "'hll'/'rusanov'");
  }
}

template <class Model>
AmrRuntimeBlock dispatch_amr_block_roe(const Model& m, const std::string& lim,
                                       const SharedAmrLayout& S, const std::string& name,
                                       const std::vector<double>& density, bool has_density,
                                       double gamma, int substeps, bool recon_prim, bool imex,
                                       int stride, const std::vector<int>& implicit_components,
                                       const NewtonOptions& nopts, const std::vector<double>* state,
                                       bool newton_diagnostics, AmrTimeMethod time_method,
                                       double pos_floor) {
  // ADC-590 split, same rationale as dispatch_amr_compiled_roe: generic RoeFlux is capability-only;
  // the canonical Euler layout routes the explicit EulerRoeFlux2D.
  if constexpr (HasRoeDissipation<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, RoeFlux>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
                            });
  } else if constexpr (Model::n_vars == 4 &&
                       requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_block(AmrSystem, multi-block)"),
                            "add_block(AmrSystem, multi-block)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_block<Model, L, EulerRoeFlux2D>(
                                  m, S, name, density, has_density, gamma, substeps, recon_prim,
                                  imex, stride, implicit_components, nopts, state,
                                  newton_diagnostics, time_method, pos_floor);
                            });
  } else {
    throw std::runtime_error(
        "add_block(AmrSystem, multi-block): flux 'roe' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's Roe capability (roe_dissipation, cf. HasRoeDissipation); "
        "this transport -> 'hll'/'rusanov'");
  }
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> build_amr_block. SAME guards as
/// dispatch_amr_compiled (hllc/roe require the model's Riemann capability HasHLLCStructure /
/// HasRoeDissipation, OR the canonical Euler 2D layout: 4 variables + pressure).
/// Multi-block counterpart of dispatch_amr_compiled. @p implicit_components: partial IMEX mask carried
/// by the block (indices of the implicit components; empty = full backward-Euler), threaded to build_amr_block.
template <class Model>
AmrRuntimeBlock dispatch_amr_block(
    const Model& m, const std::string& lim, const std::string& riem, const SharedAmrLayout& S,
    const std::string& name, const std::vector<double>& density, bool has_density, double gamma,
    int substeps, bool recon_prim, bool imex, int stride = 1,
    const std::vector<int>& implicit_components = {}, const NewtonOptions& nopts = {},
    const std::vector<double>* state = nullptr, bool newton_diagnostics = false,
    AmrTimeMethod time_method = AmrTimeMethod::kEuler, double pos_floor = 0.0) {
  // CENTRALIZED VALIDATION (dispatch_tags.hpp registry) BEFORE the dispatch: same tags accepted /
  // rejected as before, identical messages. The template if/else dispatch that follows is UNCHANGED; the
  // capability guards (hllc/roe: 2D Euler or capability) stay `if constexpr` PER MODEL.
  validate_riemann(riem, /*polar=*/false, "add_block(AmrSystem, multi-block)");
  validate_limiter(lim, "add_block(AmrSystem, multi-block)");
  // ADC-359: delegate to the flux-pinned dispatch_amr_block_<flux> helpers above (factored so the
  // compressible seam compiles one flux per TU). Behavior is unchanged: same leaves, same hllc/roe
  // capability guards, same throws. exb/isothermal route here as before (their guards prune hllc/roe).
  // ADC-641: parse the validated tag ONCE into the typed RiemannRouteId; the switch decodes it and the
  // euler_* fall-through keeps the fusion self-documenting.
  switch (parse_riemann_route(riem, "add_block(AmrSystem, multi-block)")) {
    case RiemannRouteId::kRusanov:
      return dispatch_amr_block_rusanov(m, lim, S, name, density, has_density, gamma, substeps,
                                        recon_prim, imex, stride, implicit_components, nopts, state,
                                        newton_diagnostics, time_method, pos_floor);
    case RiemannRouteId::kHll:
      return dispatch_amr_block_hll(m, lim, S, name, density, has_density, gamma, substeps,
                                    recon_prim, imex, stride, implicit_components, nopts, state,
                                    newton_diagnostics, time_method, pos_floor);
    // hllc / euler_hllc share the leaf: on the true Euler brick the generic HLLCFlux (via
    // HasHLLCStructure) and the explicit EulerHLLCFlux2D are bit-identical (ADC-590). The native
    // compressible transport that reaches AMR carries the capability, so both route here; euler_hllc
    // on a non-Euler transport is refused by the dispatch_amr_block_hllc capability guard (same
    // message). Same for roe / euler_roe.
    case RiemannRouteId::kHllc:
    case RiemannRouteId::kEulerHllc:
      return dispatch_amr_block_hllc(m, lim, S, name, density, has_density, gamma, substeps,
                                     recon_prim, imex, stride, implicit_components, nopts, state,
                                     newton_diagnostics, time_method, pos_floor);
    case RiemannRouteId::kRoe:
    case RiemannRouteId::kEulerRoe:
      return dispatch_amr_block_roe(m, lim, S, name, density, has_density, gamma, substeps,
                                    recon_prim, imex, stride, implicit_components, nopts, state,
                                    newton_diagnostics, time_method, pos_floor);
  }
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "flux", riem);
}

// ADC-359 per-flux branches of dispatch_amr_compiled, factored so the compressible compiled AMR seam
// compiles ONE flux per TU (build_amr_compiled_for_flux -> these). Each body is the corresponding
// `if (riem == "<flux>")` branch of dispatch_amr_compiled VERBATIM (same leaves, guards, messages);
// validate_* run in the caller. dispatch_amr_compiled (below, unchanged) still serves exb/isothermal.
template <class Model>
AmrCompiledHooks dispatch_amr_compiled_rusanov(const Model& m, const std::string& lim,
                                               const AmrBuildParams& bp) {
  return dispatch_limiter(parse_limiter_route(lim, "add_compiled_model(AmrSystem)"),
                          "add_compiled_model(AmrSystem)", [&](auto tag) {
                            using L = typename decltype(tag)::type;
                            return build_amr_compiled<Model, L, RusanovFlux>(m, bp);
                          });
}

template <class Model>
AmrCompiledHooks dispatch_amr_compiled_hll(const Model& m, const std::string& lim,
                                           const AmrBuildParams& bp) {
  if constexpr (requires(const Model mm, typename Model::State s, Aux a, Real r) {
                  mm.wave_speeds(s, a, 0, r, r);
                }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_compiled_model(AmrSystem)"),
                            "add_compiled_model(AmrSystem)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_compiled<Model, L, HLLFlux>(m, bp);
                            });
  } else {
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): flux 'hll' requires signed wave "
        "speeds (model.wave_speeds: declare a primitive 'p'); "
        "this transport -> 'rusanov'");
  }
}

template <class Model>
AmrCompiledHooks dispatch_amr_compiled_hllc(const Model& m, const std::string& lim,
                                            const AmrBuildParams& bp) {
  // ADC-590 split (AMR mirror of make_block_hllc / make_block_euler_hllc): the generic HLLCFlux is
  // capability-ONLY -- it static_asserts without HasHLLCStructure -- so instantiating it behind the
  // old "capability OR canonical layout" gate broke the WHOLE generated TU for a capability-free
  // 4-var Euler DSL model (no target='amr_system' .so could compile, whatever flux was requested;
  // ADC-634 fallout). A capable model keeps the generic leaves (bit-identical: the native Euler brick
  // carries the capability); a capability-free canonical Euler layout routes the explicit
  // EulerHLLCFlux2D (the ADC-590 canonical route, bit-identical on the true Euler brick); anything
  // else keeps the runtime refusal.
  if constexpr (HasHLLCStructure<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "add_compiled_model(AmrSystem)"),
                            "add_compiled_model(AmrSystem)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_compiled<Model, L, HLLCFlux>(m, bp);
                            });
  } else if constexpr (Model::n_vars == 4 &&
                       requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_compiled_model(AmrSystem)"),
                            "add_compiled_model(AmrSystem)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_compiled<Model, L, EulerHLLCFlux2D>(m, bp);
                            });
  } else {
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): flux 'hllc' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's HLLC capability (pressure + wave_speeds + contact_speed + "
        "hllc_star_state, cf. HasHLLCStructure); this transport -> "
        "'hll'/'rusanov'");
  }
}

template <class Model>
AmrCompiledHooks dispatch_amr_compiled_roe(const Model& m, const std::string& lim,
                                           const AmrBuildParams& bp) {
  // ADC-590 split, same rationale as dispatch_amr_compiled_hllc above: generic RoeFlux is
  // capability-only; the canonical Euler layout routes the explicit EulerRoeFlux2D.
  if constexpr (HasRoeDissipation<Model>) {
    return dispatch_limiter(parse_limiter_route(lim, "add_compiled_model(AmrSystem)"),
                            "add_compiled_model(AmrSystem)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_compiled<Model, L, RoeFlux>(m, bp);
                            });
  } else if constexpr (Model::n_vars == 4 &&
                       requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
    return dispatch_limiter(parse_limiter_route(lim, "add_compiled_model(AmrSystem)"),
                            "add_compiled_model(AmrSystem)", [&](auto tag) {
                              using L = typename decltype(tag)::type;
                              return build_amr_compiled<Model, L, EulerRoeFlux2D>(m, bp);
                            });
  } else {
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): flux 'roe' requires a "
        "compressible Euler 2D transport (4 variables + pressure) OR the "
        "model's Roe capability (roe_dissipation, cf. HasRoeDissipation); "
        "this transport -> 'hll'/'rusanov'");
  }
}

/// Dispatch of the spatial scheme (limiter x Riemann flux) -> build_amr_compiled. Same guards as
/// AmrSystem::add_block (hllc/roe require the model's Riemann capability HasHLLCStructure /
/// HasRoeDissipation, OR the canonical Euler 2D layout: 4 variables + pressure).
template <class Model>
AmrCompiledHooks dispatch_amr_compiled(const Model& m, const std::string& lim,
                                       const std::string& riem, const AmrBuildParams& bp) {
  // CENTRALIZED VALIDATION (dispatch_tags.hpp registry) BEFORE the dispatch: same tags accepted /
  // rejected as before. Template if/else dispatch UNCHANGED; per-model hllc/roe capability guards.
  validate_riemann(riem, /*polar=*/false, "add_compiled_model(AmrSystem)");
  validate_limiter(lim, "add_compiled_model(AmrSystem)");
  // ADC-359: delegate to the flux-pinned dispatch_amr_compiled_<flux> helpers above. Behavior unchanged
  // (same leaves, guards, throws); exb/isothermal route here as before (their guards prune hllc/roe).
  // ADC-641: parse the validated tag ONCE into the typed RiemannRouteId; the euler_* fall-through keeps
  // the fusion self-documenting. The default is the defense-in-depth registry/dispatch guard.
  switch (parse_riemann_route(riem, "add_compiled_model(AmrSystem)")) {
    case RiemannRouteId::kRusanov:
      return dispatch_amr_compiled_rusanov(m, lim, bp);
    case RiemannRouteId::kHll:
      return dispatch_amr_compiled_hll(m, lim, bp);
    // hllc / euler_hllc (and roe / euler_roe) share the leaf: bit-identical on the true Euler brick
    // (ADC-590); euler_* on a non-Euler transport is refused by the same capability guard.
    case RiemannRouteId::kHllc:
    case RiemannRouteId::kEulerHllc:
      return dispatch_amr_compiled_hllc(m, lim, bp);
    case RiemannRouteId::kRoe:
    case RiemannRouteId::kEulerRoe:
      return dispatch_amr_compiled_roe(m, lim, bp);
  }
  throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "flux", riem);
}

}  // namespace detail

/// Resolves the partial IMEX MASK (implicit_vars / implicit_roles) of a COMPILED block into indices of
/// conserved components, against the conservative descriptor @p cons of the CONCRETE Model (known here).
/// SAME strict logic as resolve_implicit_components of amr_system.cpp (missing name/role -> error;
/// unique sorted indices) -- replicated here because this header does not depend on the facade .cpp. EMPTY
/// input -> empty -> inactive mask (full backward-Euler). Used by the multi-block runtime builder.
inline std::vector<int> resolve_implicit_components_compiled(
    const std::string& block, const VariableSet& cons, const std::vector<std::string>& names,
    const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end())
      out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) {
        idx = i;
        break;
      }
    if (idx < 0)
      throw std::runtime_error("add_compiled_model(AmrSystem): implicit_vars: variable '" + nm +
                               "' missing from block '" + block + "'");
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const VariableRole role = role_from_name(rn);
    const int idx = cons.index_of(role);
    if (role == VariableRole::Custom || idx < 0)
      throw std::runtime_error("add_compiled_model(AmrSystem): implicit_roles: role '" + rn +
                               "' missing from block '" + block + "'");
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Wires @p model (concrete CompositeModel) as an AMR block of @p sys, with the requested scheme. The
/// build is DEFERRED (like add_block): the captured closures are invoked at the first
/// step/mass/density via ensure_built(), after set_refinement / set_poisson / set_density.
///
/// MONO-BLOCK (a single add_compiled_model): historical AmrCouplerMP<Model> path (mono_builder),
/// bit-identical. MULTI-BLOCK (>= 2 blocks, compiled and/or native mixed; capstone v): the block is
/// materialized as a type-erased AmrRuntimeBlock on the layout SHARED by the multi_builder, exactly
/// like native add_block. We freeze BOTH builders here (the facade chooses the routing at ensure_built).
/// @p time: "explicit" (forward Euler source) or "imex" (stiff implicit source via
/// backward_euler_source, explicit transport carried by the reflux). Any other treatment is refused.
/// @p stride: HOLD-THEN-CATCH-UP cadence of the block in multi-block (1 = each macro-step).
/// @p implicit_vars / @p implicit_roles: partial IMEX mask of the block (multi-block; requires time=imex).
/// @p pos_floor: Zhang-Shu positivity floor (ADC-322; 0 = inactive, bit-identical). Stored on the block
///   (mono path reads AmrBuildParams::pos_floor) AND forwarded to the multi-block builder, so the .so
///   floors the Density-role face states like a native add_block.
/// @throws std::runtime_error if the system is already built or if time/recon are out of domain.
template <class Model>
void add_compiled_model(AmrSystem& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit",
                        double gamma = static_cast<double>(kPhysicalDefaultGamma),
                        int substeps = 1,
                        int stride = 1, const std::vector<std::string>& implicit_vars = {},
                        const std::vector<std::string>& implicit_roles = {},
                        double pos_floor = 0.0) {
  if (substeps < 1)
    throw std::runtime_error("add_compiled_model(AmrSystem): substeps >= 1");
  // PROJECTION PONCTUELLE post-pas (ADC-177) : DESORMAIS CABLEE sur AmrSystem. Appliquee PAR NIVEAU
  // a la fin de l'avance du pas (apres le reflux), aussi bien sur le coupleur mono-bloc
  // (build_amr_compiled -> cpl->levels()) que sur le multi-bloc natif (build_amr_block ->
  // AmrRuntime::step -> project_per_level). Cell-local + idempotente : conservation preservee (les
  // flux-registres sont deja regles). No-op si le modele ne declare pas m.project.
  // SSPRK3 IS NOT carried by the COMPILED path: neither the mono_builder nor the multi_builder
  // freezes AmrBuildParams::time_method / passes AmrTimeMethod to dispatch_amr_block (the flat ABI of the
  // .so loader does not marshal the method). EXPLICIT rejection rather than a silent kEuler fallback; an
  // SSPRK3 block must be NATIVE (AmrSystem::add_block / dispatch_amr_block, which threads it).
  if (time == "ssprk3")
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): time='ssprk3' not carried by the "
        "compiled path (.so); use a native block pops.Model(...).");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error(
        "add_compiled_model(AmrSystem): time '" + time + "' unknown (available here: " +
        std::string(route_token(TimeRouteId::kExplicitSsprk2)) + "|" +
        route_token(TimeRouteId::kImex) + ")");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("add_compiled_model(AmrSystem): recon unknown '" + recon +
                             "' (valid: " + kReconRouteTokensCsv + ")");
  const bool recon_prim = (recon == "primitive");
  const bool imex = (time == "imex");
  // (1) MONO-BLOCK builder: captures the concrete Model + the scheme, materializes the AmrCouplerMP at the
  // lazy build (refine/poisson/density parameters frozen at that point).
  auto mono_builder = [model, limiter, riemann, recon_prim, imex](const AmrBuildParams& bp) {
    AmrBuildParams p = bp;
    p.physics.recon_prim = recon_prim;
    p.physics.imex = imex;
    return detail::dispatch_amr_compiled(model, limiter, riemann, p);
  };
  // (2) MULTI-BLOCK builder: captures the SAME concrete Model/scheme, materializes the AmrRuntimeBlock of the
  // block on the SHARED layout (common to all blocks, created once at ensure_built). Resolves ITSELF
  // the partial IMEX mask against cons_vars of the concrete Model (known here), then calls dispatch_amr_block
  // -- EXACTLY the native path of add_block, only the point of type resolution differs (here at
  // the add, there from a ModelSpec at build). FUNCTOR without a cross-TU extended lambda in the kernel:
  // dispatch_amr_block captures advance_amr<Limiter, Flux> (named template function), device-clean
  // recipe #64/#97; the outer lambda only orchestrates (no device kernel in its body).
  auto multi_builder = [model, limiter, riemann](
                           const detail::SharedAmrLayout& S, const std::string& bname,
                           const std::vector<double>& density, bool has_density, double bgamma,
                           int bsub, bool brecon_prim, bool bimex, int bstride,
                           const std::vector<std::string>& ivars,
                           const std::vector<std::string>& iroles, double bpos_floor) {
    const std::vector<int> impl_components =
        bimex
            ? resolve_implicit_components_compiled(bname, Model::conservative_vars(), ivars, iroles)
            : std::vector<int>{};
    // pos_floor (ADC-322): the .so flat ABI now carries the Zhang-Shu floor; forward it to the SAME
    // dispatch_amr_block -> build_amr_block leaf as a native multi-block. The compiled path transports
    // NEITHER Newton options/state/diagnostics NOR SSPRK3 (rejected at the facade / add_compiled_model),
    // so those intermediate arguments stay at their historical defaults (kEuler, no Newton, no state).
    return detail::dispatch_amr_block(
        model, limiter, riemann, S, bname, density, has_density, bgamma, bsub, brecon_prim, bimex,
        bstride, impl_components, NewtonOptions{},
        /*state=*/nullptr, /*newton_diagnostics=*/false, AmrTimeMethod::kEuler, bpos_floor);
  };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(mono_builder),
                         std::move(multi_builder), name, recon_prim, imex, stride, implicit_vars,
                         implicit_roles, pos_floor);
}

}  // namespace pops
