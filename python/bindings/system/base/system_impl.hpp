// ADC-632: BINDING-PRIVATE definition of System::Impl and the System-facade helpers, hoisted out
// of system.cpp so the responsibility-split sibling TUs (system_install / system_fields /
// system_io / system_profiling / system_program.cpp) share ONE Impl definition. Impl is binding-
// internal (NOT public API) -> it stays under python/bindings, not include/pops. The formerly
// anonymous-namespace helpers become header-inline (ODR-safe single definition across the TUs).
// VERBATIM move: bodies unchanged, no logic touched -> production trajectories bit-identical.
// native_loader.hpp is deliberately NOT included here; the install TU includes it AFTER this
// header (its templates instantiate on Impl, kept "lower down" per-TU as before).
#pragma once

#include <pops/runtime/system.hpp>

#include <pops/core/state/variables.hpp>  // VariableSet + VariableRole: role descriptor carried by each block
#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>  // pops::abi_key + detail::abi_key_string (ABI boundary of the native loader)
#include <pops/runtime/builders/block/block_builder.hpp>  // GridContext + make_block/make_max_speed (compiled closures)
#include <pops/runtime/builders/block/block_seam.hpp>  // ADC-335: per-transport build seam (build_block_exb/.../polar)
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model + compiled bricks
#include <pops/runtime/dynamic/model_registry.hpp>  // validate_transport: single-source transport rejection (ADC-331)
#include <pops/coupling/source/coupled_source_program.hpp>  // CoupledSourceKernel: generic coupled source (DSL P5, bytecode)
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>  // PolarPoissonSolver (direct polar Poisson, REUSED)
#include <pops/runtime/system/system_field_solver.hpp>  // SystemFieldSolver: elliptic solve + field derivation (Batch B)
#include <pops/runtime/system/system_stepper.hpp>  // SystemStepper: time advance (step/advance/step_cfl/step_adaptive) (Batch B)
#include <pops/runtime/system/system_block_store.hpp>  // SystemBlockStore: block management (BlockState + registry + index/copy/write) (Batch B.3)
#include <pops/runtime/system/system_runtime_params.hpp>  // SystemRuntimeParamsRegistry: per-AOT-block runtime params (ADC-578)
#include <pops/runtime/system/system_diagnostics_registry.hpp>  // SystemDiagnosticsRegistry: block/stage options + Newton reports (ADC-578)
#include <pops/runtime/system/system_coupling_registry.hpp>  // SystemCouplingRegistry: couplings + dt bounds + frequency bounds (ADC-578)
#include <pops/runtime/system/system_domain.hpp>  // SystemDomain: geometry/layout + shared aux + embedded-boundary (ADC-578)
#include <pops/runtime/system/system_lifecycle.hpp>  // SystemLifecycle: typed freeze state machine (ADC-578)
#include <pops/runtime/builders/block/block_builder_polar.hpp>  // POLAR block closures (assemble_rhs_polar, REUSED)
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // backward_euler_source
#include <pops/numerics/time/integrators/time_steppers.hpp>  // ForwardEuler, SSPRK2Step (core RK math)
#include <pops/numerics/spatial_operator.hpp>  // assemble_rhs, SourceFreeModel, max_wave_speed_mf, load_state

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // sum
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>      // fill_ghosts, fill_boundary
#include <pops/runtime/dynamic/dynamic_model.hpp>  // IModel: model loaded at runtime (dynamic block)
#include <pops/runtime/context/wall_predicate.hpp>  // detail::wall_predicate (wall shared by System/AmrSystem)
#include <pops/runtime/program/module_metadata.hpp>  // read_module_metadata / required_aux: install-time requirement validation (ADC-446)
#include <pops/runtime/program/program_context.hpp>  // ProgramContext: wraps the System for the .so dt_bound call (ADC-417)
#include <pops/runtime/program/profiler.hpp>  // Profiler / ProfileScope: per-node / per-brick timing (ADC-459)
#include <pops/runtime/program/program_runtime_state.hpp>  // ProgramRuntimeState: the extracted compiled-Program subsystem (ADC-594)

#include <algorithm>
#include <cmath>
#include <pops/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW layer (ADC-99); <dlfcn.h> on POSIX
#include <functional>
#include <limits>  // std::numeric_limits (per-block CFL: dt = min over blocks)
#include <map>     // std::map (per-block runtime params registry, P7-b)
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace pops {

// The structured DIAGNOSTIC trace of the solve_fields path is owned by SystemFieldSolver
// (namespace field_solver); it stays env-gated (POPS_TRACE_SOLVE_FIELDS) and inert by default.
// resolve_implicit_components moved to model_factory.hpp (pops::detail) so the per-transport seam TUs
// (python/system_<transport>.cpp, ADC-335) share one definition; it is otherwise unchanged.

// System-facade helpers (formerly anonymous-namespace, now header-inline for the split TUs).
// Collective multi-box/multi-rank gather of @p mf into a GLOBAL buffer of size ncomp*gny*gnx,
// component-major ((c*gny + j)*gnx + i; for ncomp == 1 this collapses to j*gnx + i). Zero-init,
// then each rank writes ONLY its LOCAL boxes at their GLOBAL indices (disjoint boxes -> each cell
// owned by exactly one rank; a rank without a box writes nothing), and all_reduce_sum_inplace
// makes the full field appear on every rank. Mono-rank: the box covers the domain and the reduce
// is the identity. The CALLER owns the device_fence and the single-box fast path (SystemBlockStore).
// Factored out of the five copy-pasted gather sites (ADC-264); the loops are verbatim, so each site
// stays bit-identical (copy_comp0 / copy_state and density_global / state_global / potential_global).
inline std::vector<double> gather_global(const MultiFab& mf, int ncomp, int gnx, int gny) {
  std::vector<double> out(static_cast<std::size_t>(ncomp) * gnx * gny, 0.0);
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 u = mf.fab(li).const_array();
    const Box2D v = mf.box(li);
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[(static_cast<std::size_t>(c) * gny + j) * gnx + i] = static_cast<double>(u(i, j, c));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

inline bool newton_options_non_default(const NewtonOptions& newton, bool diagnostics = false) {
  return newton.max_iters != kNewtonDefaultMaxIters || newton.rel_tol != kNewtonDefaultRelTol ||
         newton.abs_tol != kNewtonDefaultAbsTol || newton.fd_eps != kNewtonDefaultFdEps ||
         diagnostics || newton.damping != kNewtonDefaultDamping ||
         newton.fail_policy != kNewtonDefaultFailPolicy;
}

inline EffectiveNewtonOptions effective_newton_options(const NewtonOptions& newton, bool diagnostics) {
  EffectiveNewtonOptions out;
  out.max_iters = newton.max_iters;
  out.rel_tol = static_cast<double>(newton.rel_tol);
  out.abs_tol = static_cast<double>(newton.abs_tol);
  out.fd_eps = static_cast<double>(newton.fd_eps);
  out.damping = static_cast<double>(newton.damping);
  out.fail_policy = newton_fail_policy_name(newton.fail_policy);
  out.diagnostics = diagnostics;
  out.non_default = newton_options_non_default(newton, diagnostics);
  return out;
}

inline EffectiveBlockOptions make_system_block_options(
    const std::string& name, const ModelSpec& model, const std::string& route,
    const std::string& limiter, const std::string& riemann, const std::string& recon,
    const std::string& time, const std::string& method, bool imex, int substeps, bool evolve,
    int stride, const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles, const NewtonOptions& newton,
    bool newton_diagnostics, double positivity_floor, bool wave_speed_cache) {
  EffectiveBlockOptions out;
  out.name = name;
  out.route = route;
  out.compiled = false;
  out.transport = model.transport;
  out.source = model.source;
  out.elliptic = model.elliptic;
  out.limiter = limiter;
  out.riemann = riemann;
  out.recon = recon;
  out.time = time;
  out.time_method = method;
  out.imex = imex;
  out.substeps = substeps;
  out.stride = stride;
  out.evolve = evolve;
  out.n_ghost = block_n_ghost(limiter);
  out.implicit_vars = implicit_vars;
  out.implicit_roles = implicit_roles;
  out.newton = effective_newton_options(newton, newton_diagnostics);
  out.positivity_floor = positivity_floor;
  out.wave_speed_cache = wave_speed_cache;
  out.gamma = model.gamma;
  out.B0 = model.B0;
  out.cs2 = model.cs2;
  out.vacuum_floor = model.vacuum_floor;
  out.qom = model.qom;
  out.q = model.q;
  out.alpha = model.alpha;
  out.n0 = model.n0;
  out.sign = model.sign;
  out.four_pi_G = model.four_pi_G;
  out.rho0 = model.rho0;
  return out;
}

inline int coupling_role_index_reported(const VariableSet& vs, VariableRole role, int fallback,
                                 const char* origin, const std::string& block) {
  if (vs.roles.empty())
    record_fallback(FallbackCounter::kRolelessComponentIndex);
  return coupling_role_index(vs, role, fallback, origin, block);
}

struct System::Impl {
  // BLOCK MANAGEMENT extracted into SystemBlockStore (Batch B.3, last P0 extraction from the god-class):
  // the block struct (formerly Species, renamed BlockState), the ordered registry (blocks_.blocks), the
  // by-name access (index / find) and the state marshaling (copy_comp0 / copy_state / write_state) now
  // live there. See include/pops/runtime/system_block_store.hpp.
  //
  // COMPATIBILITY ALIASES. The already-extracted header templates (SystemFieldSolver, SystemStepper,
  // native_loader) iterate `owner_->sp` / `P->sp` and name `Impl::Species`; we keep these two
  // access points identical (zero churn outside this file):
  //  - `Species` = the block type carried by the store (init via positional aggregate unchanged);
  //  - `sp` = a REFERENCE to the store registry (same object, same iteration, same indexing).
  using Species = SystemBlockStore::BlockState;

  // GEOMETRY / LAYOUT + SHARED aux + embedded-boundary domain, EXTRACTED into SystemDomain (ADC-578,
  // include/pops/runtime/system/system_domain.hpp). domain_ is constructed FIRST (before fields_ /
  // stepper_) so its exact historical init-list (cfg, geom, polar_, pgeom_, ba, dm, bc_, dom, per_,
  // periodic_, aux) runs before any back-pointer reads it. Every member is re-exposed under its exact
  // historical name via a REFERENCE ALIAS (the proven `sp = blocks_.blocks` idiom): SystemStepper /
  // SystemFieldSolver / native_loader read them via owner_-> / P-> unchanged, and the block closures
  // capture a stable `&aux == &domain_.aux` -> those headers and the MockImpl stay byte-unchanged.
  pops::runtime::system::SystemDomain domain_;
  SystemConfig& cfg = domain_.cfg;
  Geometry& geom = domain_.geom;
  bool& polar_ = domain_.polar_;
  PolarGeometry& pgeom_ = domain_.pgeom_;
  BoxArray& ba = domain_.ba;
  DistributionMapping& dm = domain_.dm;
  BCRec& bc_ = domain_.bc_;
  Box2D& dom = domain_.dom;
  Periodicity& per_ = domain_.per_;
  bool& periodic_ = domain_.periodic_;
  MultiFab& aux = domain_.aux;
  int& aux_ncomp_ = domain_.aux_ncomp_;
  detail::DiscDomain& eb_domain_ = domain_.eb_domain_;
  bool& eb_set_ = domain_.eb_set_;
  MultiFab& domain_mask_ = domain_.domain_mask_;
  // ADC-615: the cut-cell / EB thresholds (kappa_min, face_open_eps, cut_theta_min) resolved by
  // set_disc_domain. Defaults are the kEb* constants, so an unconfigured EB run is bit-identical AND
  // the cut_theta_min is passed to BOTH the EB transport (assemble_rhs_eb) and the elliptic wall.
  EbThresholds eb_thresholds_;
  bool& ws_cache_block_ = domain_.ws_cache_block_;
  GeometryMode& geometry_mode_ = domain_.geometry_mode_;
  // aux APPLICATION fields (bz_field_, te_src_) and apply_bz/apply_te buffers EXTRACTED into
  // fields_ (SystemFieldSolver, Batch B); the SHARED aux and its width stay here (common channel).
  // Block registry OWNED by the store (Batch B.3). `sp` is a REFERENCE to blocks_.blocks: same
  // object (no copy), so owner_->sp / P->sp in the header templates stay bit-identical.
  SystemBlockStore blocks_;
  std::vector<Species>& sp = blocks_.blocks;
  // P7-b: RUNTIME parameter values per AOT block, EXTRACTED into SystemRuntimeParamsRegistry
  // (ADC-578, include/pops/runtime/system/system_runtime_params.hpp). The vector is SHARED
  // (shared_ptr) with the compiled block closures: writing into it (set_block_params) changes the
  // block behavior at the next step WITHOUT recompiling. `block_params_` is a REFERENCE ALIAS to the
  // registry's map (SAME object): native_loader.hpp registers into `P->block_params_[name]`, so the
  // exact name is kept and the loader header stays byte-unchanged.
  pops::runtime::system::SystemRuntimeParamsRegistry params_;
  std::map<std::string, std::shared_ptr<std::vector<double>>>& block_params_ = params_.block_params;
  // Effective numerical/physical block/stage options + OPT-IN IMEX Newton reports, EXTRACTED into
  // SystemDiagnosticsRegistry (ADC-578, include/pops/runtime/system/system_diagnostics_registry.hpp).
  // Stepper-invisible: accessed only here via diagnostics_.* (no alias needed, no MockImpl impact).
  pops::runtime::system::SystemDiagnosticsRegistry diagnostics_;
  double t = 0;
  int macro_step_ = 0;  // macro-step counter (0-indexed): feeds the per-block stride filter
  // COMPILED TIME-PROGRAM RUNTIME STATE (ADC-594): the whole compiled-Program subsystem -- installed
  // step + dt bound, cadence, IR-hash guard, name-based block map, runtime params, recorded
  // diagnostics, the profiler, the scheduler cache and the multistep history rings -- extracted out of
  // this god-object into ONE inspectable struct (include/pops/runtime/program/program_runtime_state.hpp),
  // SHARED verbatim with AmrSystem::Impl (the documented common contract). The stepper reads only
  // program_.step_ / substeps_ / stride_ / dt_bound_; the diagnostics /
  // params / cache / history / profiler are System-owned and NOT stepper-visible. Program invariants live
  // here, block/field/layout invariants stay on Impl.
  pops::runtime::program::ProgramRuntimeState program_;
  // RUNTIME FREEZE LIFECYCLE (ADC-592 / ADC-578): the typed state machine that replaced the single
  // `bool bound_`. Assembling while the composition is mutable; Bound once mark_bound() runs (the
  // Python bind flow calls it LAST, after every install call). When frozen (Bound / Checkpointed /
  // Finalized) the structural setters reject; the runtime-data setters stay allowed. It is the
  // authoritative NATIVE lifecycle source the Python freeze gates query. NOT referenced by
  // SystemStepper -> no MockImpl impact (MockImpl never had bound_); Assembling for a direct engine
  // script that never binds (the low-level C++ tests) -> historical behavior unchanged. See
  // include/pops/runtime/system/system_lifecycle.hpp.
  pops::runtime::system::SystemLifecycle lifecycle_;
  // INTER-SPECIES COUPLING SUBSYSTEM, EXTRACTED into SystemCouplingRegistry (ADC-578,
  // include/pops/runtime/system/system_coupling_registry.hpp): the splitting-source operators, the
  // GLOBAL host dt bounds, the constant / per-cell coupled-source frequency bounds, and the typed
  // coupling-operator inspect views. The stepper reads couplings / dt_bounds_ / coupled_freqs_ /
  // coupled_freq_exprs_, so those keep their exact names via REFERENCE ALIASES into the registry
  // (SAME objects) -> system_stepper.hpp and the MockImpl stay byte-unchanged. coupled_operators_ is
  // METADATA ONLY (never stepper-read) -> accessed registry-direct. The GlobalDtBound / CoupledFreq /
  // CoupledFreqExpr struct types now live in the registry header; the `using` aliases below keep the
  // `Impl::GlobalDtBound{...}` construction sites in this TU unchanged.
  pops::runtime::system::SystemCouplingRegistry coupling_;
  using GlobalDtBound = pops::runtime::system::GlobalDtBound;
  using CoupledFreq = pops::runtime::system::CoupledFreq;
  using CoupledFreqExpr = pops::runtime::system::CoupledFreqExpr;
  std::vector<std::function<void(Real)>>& couplings = coupling_.operators;
  std::vector<GlobalDtBound>& dt_bounds_ = coupling_.dt_bounds;
  std::vector<CoupledFreq>& coupled_freqs_ = coupling_.coupled_freqs;
  std::vector<CoupledFreqExpr>& coupled_freq_exprs_ = coupling_.coupled_freq_exprs;

  // stride_due (hold-then-catch-up cadence filter) EXTRACTED into stepper_ (SystemStepper, Batch B):
  // it serves exclusively the time advance. macro_step_ (above) stays a SHARED member of Impl
  // (read by time() indirectly via t, incremented by stepper_ via owner_->macro_step_).

  // Elliptic solve + field derivation EXTRACTED into fields_ (SystemFieldSolver, Batch B,
  // cf. docs/SYSTEM_CPP_EXTRACTION_PLAN.md section 2): the Poisson configuration (p_rhs/p_solver/
  // p_bc/p_wall/p_wall_radius/p_eps_), the coefficient fields (eps(x), eps_x/eps_y, kappa), the
  // solvers (Cartesian ell_, polar pell_) and the aux application buffers (B_z, T_e) live there
  // now. fields_ reads the SHARED aux/sp/cfg/geom/pgeom_/ba/dm/bc_/dom/per_ of Impl via its
  // back-pointer. Declared after the shared members it captures (initialized in the constructor).

  // Geometry/layout helpers moved to SystemDomain (ADC-578); thin static forwarders keep the
  // Impl::polar_nr(cfg) / Impl::polar_ntheta(cfg) call sites in this TU unchanged.
  static int polar_nr(const SystemConfig& c) {
    return pops::runtime::system::SystemDomain::polar_nr(c);
  }
  static int polar_ntheta(const SystemConfig& c) {
    return pops::runtime::system::SystemDomain::polar_ntheta(c);
  }
  static Box2D index_domain(const SystemConfig& c) {
    return pops::runtime::system::SystemDomain::index_domain(c);
  }
  // Number of cells of a cell-defined field (n*n Cartesian / nr*ntheta polar), for the
  // size check of named aux fields (set_aux_field).
  std::size_t aux_field_cell_count() const;

  // domain_ (SystemDomain) is constructed FIRST from the config: it owns the exact historical
  // geometry/layout init-list (cfg, geom, polar_, pgeom_, ba, dm, bc_, dom, per_, periodic_, aux) so
  // fields_ / stepper_ back-pointers read a fully-built layout. The reference aliases above then bind
  // to domain_.*, and fields_(this) / stepper_(this) capture Impl (bit-identical addresses).
  explicit Impl(const SystemConfig& c)
      : domain_(c),
        fields_(this),
        stepper_(this) {}

  // Elliptic solve + field derivation (Batch B). OWNS the solvers (ell_/pell_), the Poisson
  // config, the coefficient fields and the aux application buffers (B_z, T_e). owner_ = this: the
  // helper reads the SHARED aux/sp/cfg/geom/pgeom_/ba/dm/bc_/dom/per_/periodic_/polar_ of Impl. None of
  // these accesses dereferences Impl at CONSTRUCTION (pure back-pointer) -> init at end of list without
  // ordering dependency. See include/pops/runtime/system_field_solver.hpp.
  field_solver::SystemFieldSolver<Impl> fields_;

  // Time advance (Batch B). ORCHESTRATES step / advance / step_cfl / step_adaptive, the cadence filter
  // (stride_due) and the couplings (apply_couplings). owner_ = this: the stepper reads the SHARED sp /
  // fields_ / aux / couplings / t / macro_step_ / geom / pgeom_ / polar_
  // of Impl via its back-pointer. Pure back-pointer at construction (no dereferencing) ->
  // init at end of list without ordering dependency. See include/pops/runtime/system_stepper.hpp.
  stepper::SystemStepper<Impl> stepper_;

  // Guarantees an aux width >= ncomp (SHARED channel). Reallocating the aux KEEPS its address (member:
  // the block closures capture &aux via grid_ctx) and re-applies B_z. No-op if already wide enough.
  void ensure_aux_width(int ncomp) {
    if (ncomp <= aux_ncomp_)
      return;
    aux_ncomp_ = ncomp;
    aux = MultiFab(ba, dm, aux_ncomp_, 1);
    fields_.apply_bz();
    fields_.apply_te();
    fields_
        .apply_named_aux();  // re-applies the NAMED aux fields (ADC-70): the redistributed MultiFab starts at zero
  }

  // apply_bz (population of the B_z component of the aux channel) EXTRACTED into fields_ (SystemFieldSolver).

  // Guarantees that the state U of block @p name carries at least @p ng ghosts (spatial scheme stencil).
  // WENO5 reads 3 ghosts, > the 2 allocated by default in install_block; without this width,
  // fill_ghosts + assemble_rhs would read out of bounds (cf. AmrSystem which allocates with Limiter::n_ghost,
  // PR #22). Reallocates the MultiFab and COPIES the valid cells (set_density may have preceded);
  // no-op if U already has enough ghosts -> allocation and data bit-identical to before for MUSCL.
  void set_block_ghosts(const std::string& name, int ng) {
    Species& s = find(name);
    if (s.U.n_grow() >= ng)
      return;
    MultiFab nu(s.U.box_array(), s.U.dmap(), s.ncomp, ng);
    nu.set_val(Real(0));
    for (int li = 0; li < s.U.local_size(); ++li) {
      const ConstArray4 old = s.U.fab(li).const_array();
      Array4 dst = nu.fab(li).array();
      const Box2D v = s.U.box(li);  // valid cells (excluding ghost): copied as-is
      for (int c = 0; c < s.ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            dst(i, j, c) = old(i, j, c);
    }
    s.U = std::move(nu);
  }

  // kTeComp (canonical T_e component) and apply_te (population of T_e = p/rho of the source block)
  // EXTRACTED into fields_ (SystemFieldSolver): T_e is part of the aux field application.

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodic by default
    if (c.geometry == "polar") {
      // POLAR: r (dir 0, xlo/xhi) carries a PHYSICAL BC (wall / free outflow, Foextrap); theta
      // (dir 1, ylo/yhi) is PERIODIC (the ring covers [0, 2pi)). This is the convention of
      // test_polar_transport_mms and of assemble_rhs_polar (periodic theta, physical r).
      b.xlo = b.xhi = BCType::Foextrap;
      b.ylo = b.yhi = BCType::Periodic;
      return b;
    }
    if (!c.periodic)
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  // By-name access DELEGATED to the store (Batch B.3): same linear search, same indexing by
  // insertion order, same error message ("System: unknown block '...'").
  Species& find(const std::string& name) { return blocks_.find(name); }
  const Species& find(const std::string& name) const { return blocks_.find(name); }
  int index(const std::string& name) const { return blocks_.index(name); }

  // apply_couplings (inter-species coupling sources by splitting, AFTER transport) is extracted into
  // stepper_ (SystemStepper, Batch B) and invoked by step / step_cfl / step_adaptive. It reads the
  // SHARED state via owner_->. The
  // couplings list (above) stays a member of Impl (populated by add_ionization / add_collision / ...).

  // --- elliptic solver (system Poisson) -----------------------------
  // poisson_bc / wall_active / ensure_elliptic / apply_epsilon_field / apply_epsilon_anisotropic_field
  // / apply_reaction_field / ell_rhs / ell_phi / ell_solve / ensure_elliptic_polar / solve_fields_polar
  // / solve_fields EXTRACTED into fields_ (SystemFieldSolver, Batch B). See the header.

  // --- compiled spatial schemes -------------------------------------------
  // Method-of-lines evaluator of a block (L/F/Model frozen): ghosts then R = -div F + S.
  // Construction of the block closures (advance + residual + Poisson) moved to the header
  // (pops/runtime/block_builder.hpp: make_block / make_max_speed / make_poisson_rhs) so that the
  // production template path is instantiable outside this unit (AOT compilation of a
  // generated model). Here we only provide the grid context to pass to them.
  // GridContext: mesh + BC + aux + EMBEDDED-BOUNDARY geometry (project T5-PR3). domain_mask_ /
  // eb_domain_ are STABLE-address MEMBERS -> the block closures (build_block) read them by pointer at
  // each step, so the add_block / set_disc_domain order is irrelevant (the mask is materialized / the
  // descriptor set before the 1st step; as long as !eb_set_ the stepper does not select the
  // embedded-boundary advance).
  GridContext grid_ctx() {
    // ADC-615: carry the resolved cut-cell / EB thresholds into the context so the EB transport
    // advance (BlockRhsEvalEb -> assemble_rhs_eb) uses them; defaults = kEb* (bit-identical).
    return GridContext{dom,
                       bc_,
                       geom,
                       &aux,
                       &domain_mask_,
                       &eb_domain_,
                       eb_thresholds_.kappa_min,
                       eb_thresholds_.face_open_eps,
                       eb_thresholds_.cut_theta_min};
  }

  // POLAR grid context (ring pgeom_ + r/theta BC + aux) for the polar block closures
  // (block_builder_polar.hpp). Counterpart of grid_ctx(); never called in Cartesian.
  PolarGridContext grid_ctx_polar() { return PolarGridContext{dom, bc_, pgeom_, &aux}; }

  // ensure_elliptic_polar / solve_fields_polar / solve_fields (body) EXTRACTED into fields_
  // (SystemFieldSolver, Batch B). Pure delegation: the Cartesian/polar dispatch, the device_fence and
  // the order of fill_ghosts/fill_boundary now live in the header (bit-identical).
  //
  // PROFILER kernel count (ADC-459, Spec 3 section 29): the elliptic field solve is the per-step
  // kernel-dispatch chokepoint the NATIVE step actually hits (SystemStepper::step calls P->solve_fields
  // once per Lie step / three times per Strang step), so counting here moves "kernels" on the native
  // host path -- no SystemStepper edit (Profiler stays an Impl member the stepper never reads). The
  // count() is a single predictable branch when profiling is off (zero hot-path cost).
  SolveReport solve_fields() {
    program_.profiler_.count("kernels");
    return fields_.solve_fields();
  }
  // Per-stage field solve (ADC-409): re-solve + re-fill the shared aux from a stage state of the
  // target block (the rest of the blocks keep their live s.U). Same delegation idiom as solve_fields.
  SolveReport solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
    return fields_.solve_fields_from_state(block_idx, U_stage);
  }
  // Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): re-solve + re-fill the shared aux
  // from the SIMULTANEOUS stage states of all blocks (assemble_poisson_rhs_from_blocks). Same
  // delegation idiom as solve_fields / solve_fields_from_state.
  SolveReport solve_fields_from_blocks(const std::vector<const MultiFab*>& U_stages) {
    return fields_.solve_fields_from_blocks(U_stages);
  }
  // NAMED multi-elliptic field (ADC-428): a SECOND elliptic solve for the user-named @p field, from a
  // stage state of @p block_idx, written to the field's OWN aux components. The default Poisson path
  // (solve_fields / solve_fields_from_state) is untouched. Same delegation idiom.
  SolveReport solve_named_field_from_state(const std::string& field, int block_idx,
                                           const MultiFab& U_stage) {
    return fields_.solve_named_field_from_state(field, block_idx, U_stage);
  }
  void register_elliptic_field(const std::string& block, const std::string& field,
                               int phi_comp, int gx_comp, int gy_comp) {
    fields_.register_named_field(block, field, phi_comp, gx_comp, gy_comp);
  }

  // State marshaling DELEGATED to the store (Batch B.3): copy_comp0 / copy_state / write_state carry the
  // device_fence, the layout (component-major) and the size error identically. Kept as
  // helpers of Impl because native_loader and the facade methods call them via P->copy_state /
  // P->write_state / P->copy_comp0 (unchanged access point, zero churn outside this file).
  //
  // MULTI-BOX (theta split of the polar transport, ADC-67). The store marshals via fab(0) -- valid
  // for the unique local box of the Cartesian and of the polar mono-box (local_size() <= 1, including MPI mono-
  // box where a rank without a box returns {}). With theta_boxes > 1, a rank carries SEVERAL local boxes
  // (local_size() > 1): we then rebuild the GLOBAL field (size dom.nx() x dom.ny()) placing
  // each box at its GLOBAL indices, exactly like density_global / state_global (collective gather,
  // all_reduce_sum; mono-rank identity). We DO NOT TOUCH the store (VERBATIM bit-identical extraction):
  // the local_size() <= 1 branch delegates as-is -> Cartesian and polar mono-box UNCHANGED.
  std::vector<double> copy_comp0(const MultiFab& mf) const {
    if (mf.local_size() <= 1)
      return blocks_.copy_comp0(mf);
    device_fence();
    return gather_global(mf, 1, dom.nx(), dom.ny());
  }
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    if (mf.local_size() <= 1)
      return blocks_.copy_state(mf, ncomp);
    device_fence();
    return gather_global(mf, ncomp, dom.nx(), dom.ny());
  }
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    if (mf.local_size() <= 1) {
      blocks_.write_state(mf, ncomp, in);
      return;
    }
    // Multi-box SCATTER: @p in is the GLOBAL field (component-major (c*gny + j)*gnx + i, same layout
    // as copy_state). Each rank writes ONLY the cells of its local boxes (reading at the
    // global indices) -- no communication. Mono-rank: writes all bands.
    const int gnx = dom.nx(), gny = dom.ny();
    const std::size_t need = static_cast<std::size_t>(ncomp) * gnx * gny;
    if (in.size() != need)
      throw std::runtime_error("System::set_state : size != ncomp*nr*ntheta (multi-box theta)");
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 u = mf.fab(li).array();
      const Box2D v = mf.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            u(i, j, c) = in[(static_cast<std::size_t>(c) * gny + j) * gnx + i];
    }
  }

  // push_dynamic<NV> (DYNAMIC IModel<NV> block loaded from a .so) was EXTRACTED VERBATIM into
  // pops::native_loader::push_dynamic (include/pops/runtime/native_loader.hpp, template over Impl);
  // add_dynamic_block below instantiates it with System::Impl. See SYSTEM_CPP_EXTRACTION_PLAN.md.

  /// Execute one public macro-step against an accepted snapshot. Any exception restores every
  /// runtime-owned value a Program/native step may have published before failing; the typed
  /// StepAttemptRejected signal therefore leaves no partial state and no consumed clock tick.
  template <class Body>
  decltype(auto) execute_step_transaction(Body&& body) {
    struct Snapshot {
      std::vector<MultiFab> states;
      MultiFab aux;
      typename pops::field_solver::SystemFieldSolver<Impl>::StepSnapshot fields;
      double time;
      int macro_step;
      Real last_program_dt;
      std::map<std::string, Real> program_diagnostics;
      pops::runtime::program::CacheManager cache;
      pops::runtime::program::HistoryManager history;

      explicit Snapshot(Impl& impl)
          : aux(impl.aux),
            fields(impl.fields_.step_snapshot()),
            time(impl.t),
            macro_step(impl.macro_step_),
            last_program_dt(impl.program_.last_dt_),
            program_diagnostics(impl.program_.diagnostics_),
            cache(impl.program_.cache_),
            history(impl.program_.hist_) {
        states.reserve(impl.sp.size());
        for (const auto& block : impl.sp)
          states.emplace_back(block.U);
      }

      void restore(Impl& impl) const {
        for (std::size_t i = 0; i < states.size(); ++i)
          impl.sp[i].U = states[i];
        impl.aux = aux;
        impl.fields_.restore_step_snapshot(fields);
        impl.t = time;
        impl.macro_step_ = macro_step;
        impl.program_.last_dt_ = last_program_dt;
        impl.program_.diagnostics_ = program_diagnostics;
        impl.program_.cache_ = cache;
        impl.program_.hist_ = history;
      }
    };

    Snapshot accepted(*this);
    try {
      return std::forward<Body>(body)();
    } catch (...) {
      accepted.restore(*this);
      throw;
    }
  }
};

// Config / geometry / lifecycle guards (formerly anonymous-namespace, now header-inline).
// Geometry guard (polar-grid project). The geometry CHOICE is carried by the config
// (pops.CartesianMesh / pops.PolarMesh). "cartesian": historical path, bit-identical. "polar": global
// ring (r, theta) wired into System.step (Phase 2b): polar transport (assemble_rhs_polar) +
// polar Poisson (PolarPoissonSolver) + aux in local basis (e_r, e_theta). We validate HERE the radial
// bounds of the ring (r_max > r_min >= 0); the Python (PolarMesh) already validates them, but a caller
// that builds the SystemConfig by hand must also be protected. Any other token is an error.
inline void check_geometry(const SystemConfig& c) {
  if (c.geometry == "cartesian")
    return;
  if (c.geometry == "polar") {
    if (!(c.r_max > c.r_min && c.r_min >= 0.0))
      throw std::runtime_error(
          "System : geometry='polar' requires a ring r_max > r_min >= 0 (r_min > 0 avoids the "
          "r=0 coordinate singularity) ; cf. pops.PolarMesh");
    // nr >= 3 ENFORCED: the radial derivative of the aux (derive_aux_polar) uses a 2nd-order
    // OFF-CENTERED stencil at both walls (reads phi(i+1),phi(i+2) at r_min and phi(i-1),phi(i-2) at r_max). phi is
    // allocated WITHOUT ghost by the direct solver (its valid box IS its allocation): nr < 3 would read
    // phi out of bounds (UB). We reject it HERE (same fallback computation as Impl::polar_nr: nr or n).
    const int nr = c.nr > 0 ? c.nr : c.n;
    if (nr < 3)
      throw std::runtime_error(
          "System : geometry='polar' requires nr >= 3 (2nd-order off-centered radial stencil at "
          "the walls ; "
          "phi without ghost) ; cf. pops.PolarMesh");
    // THETA SPLIT of the transport (theta_boxes, ADC-67). 1 (default) = mono-box, bit-identical. > 1:
    // theta bands -- we require 1 <= theta_boxes <= ntheta (at least one azimuthal cell per band) AND
    // theta_boxes DIVIDES ntheta (EQUAL bands: the per-box split must not depend on the remainder,
    // and the periodic ring stitches back cleanly). PolarMesh already validates on the Python side; a caller that
    // builds the SystemConfig by hand is protected here.
    const int nth = c.ntheta > 0 ? c.ntheta : c.n;
    if (c.theta_boxes < 1)
      throw std::runtime_error(
          "System : geometry='polar' requires theta_boxes >= 1 (cf. pops.PolarMesh)");
    if (c.theta_boxes > nth)
      throw std::runtime_error(
          "System : geometry='polar' requires theta_boxes <= ntheta (at least one azimuthal cell "
          "per "
          "band) ; cf. pops.PolarMesh");
    if (nth % c.theta_boxes != 0)
      throw std::runtime_error(
          "System : geometry='polar' requires that theta_boxes DIVIDES ntheta (equal azimuthal "
          "bands) ; "
          "cf. pops.PolarMesh");
    return;
  }
  throw std::runtime_error("System : geometry '" + c.geometry +
                           "' unknown (cartesian | polar) ; cf. pops.CartesianMesh / pops.PolarMesh");
}

// UPSTREAM configuration guard (ADC-299): validate the SystemConfig invariants BEFORE constructing
// Impl. Impl's constructor already derives the geometry, the box array, the distribution mapping and
// allocates the shared aux MultiFab -- all sized from c.n. An invalid n / L does not crash there, it
// silently builds a DEGENERATE grid (empty box, dx = L/0 = +inf or negative dx) that only surfaces
// far downstream; we reject it here so the error names the real cause. n / L were wholly unchecked on
// the Cartesian path (check_geometry returns immediately for "cartesian"); the geometry token and the
// polar ring / nr / theta_boxes invariants stay in check_geometry, called last.
inline void validate_system_config(const SystemConfig& c) {
  if (c.n < 1)
    throw std::runtime_error("System : n >= 1 required (cells per direction) ; got n = " +
                             std::to_string(c.n));
  if (!(c.L > 0.0))
    throw std::runtime_error("System : L > 0 required (square domain [0,L]^2) ; got L = " +
                             std::to_string(c.L));
  check_geometry(c);  // geometry token + polar ring (r_max>r_min>=0, nr>=3, theta_boxes) invariants
}

// RUNTIME FREEZE LIFECYCLE guard (ADC-592): a STRUCTURAL setter must not mutate the composition once
// pops.bind has completed (@p lifecycle is frozen: Bound / Checkpointed / Finalized). @p what names
// the refused method. The message speaks the
// BIND vocabulary and points at the pops.Case / pops.compile / pops.bind path -- it NEVER recommends a
// legacy setter as the remedy, so it cannot read as a validation bypass. mark_bound() is called LAST by
// the Python bind flow, so the install sequence itself never trips this; a direct engine script that
// never binds keeps bound == false and is unaffected. Called at the TOP of each structural setter.
inline void require_assembling(const pops::runtime::system::SystemLifecycle& lifecycle, const char* what) {
  if (lifecycle.frozen())
    throw std::runtime_error(
        std::string("System::") + what +
        ": the composition is frozen once pops.bind completes (runtime lifecycle 'bound'); declare "
        "it on the pops.Case (blocks / field problems / AMR layout / source stage / refinement / "
        "solver routes / aux layout / installed Program) and lower it with pops.compile(...) + "
        "pops.bind(...). Only runtime data / params / checkpoint / diagnostics may change on a "
        "bound simulation.");
}

}  // namespace pops
