#include <pops/runtime/system.hpp>

#include <pops/core/state/variables.hpp>  // VariableSet + VariableRole: role descriptor carried by each block
#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>  // pops::abi_key + detail::abi_key_string (ABI boundary of the native loader)
#include <pops/runtime/builders/block/block_builder.hpp>  // GridContext + make_block/make_max_speed (compiled closures)
#include <pops/runtime/builders/block/block_seam.hpp>  // ADC-335: per-transport build seam (build_block_exb/.../polar)
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model + compiled bricks
#include <pops/runtime/dynamic/model_registry.hpp>  // unknown_transport_msg: single-source transport rejection (ADC-331)
#include <pops/coupling/schur/source/condensed_schur_source_stepper.hpp>  // Schur-condensed source stage (pops.Split / CondensedSchur, #126)
#include <pops/coupling/schur/source/polar_condensed_schur_source_stepper.hpp>  // POLAR counterpart of the condensed source stage (Path A step 2c, #212)
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
#include <pops/runtime/builders/compiled/native_loader.hpp>  // .so loading (JIT/AOT/native) + ABI guard: VERBATIM, included after the Impl def below (templates instantiated lower down)
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
#include <variant>
#include <vector>

namespace pops {

// The structured DIAGNOSTIC trace of the solve_fields path is owned by SystemFieldSolver
// (namespace field_solver); it stays env-gated (POPS_TRACE_SOLVE_FIELDS) and inert by default.
// resolve_implicit_components moved to model_factory.hpp (pops::detail) so the per-transport seam TUs
// (python/system_<transport>.cpp, ADC-335) share one definition; it is otherwise unchanged.

// MODULE ABI key (frozen at compile time of this TU). Defined here so the _pops module
// exports it (POPS_EXPORT): add_native_block compares it to the key baked into the loader .so.
POPS_EXPORT std::string abi_key() {
  return detail::abi_key_string();
}

// Convenience static method (Python binding + add_native_block): delegates to the module's free key.
std::string System::abi_key() {
  return pops::abi_key();
}

namespace {
// Collective multi-box/multi-rank gather of @p mf into a GLOBAL buffer of size ncomp*gny*gnx,
// component-major ((c*gny + j)*gnx + i; for ncomp == 1 this collapses to j*gnx + i). Zero-init,
// then each rank writes ONLY its LOCAL boxes at their GLOBAL indices (disjoint boxes -> each cell
// owned by exactly one rank; a rank without a box writes nothing), and all_reduce_sum_inplace
// makes the full field appear on every rank. Mono-rank: the box covers the domain and the reduce
// is the identity. The CALLER owns the device_fence and the single-box fast path (SystemBlockStore).
// Factored out of the five copy-pasted gather sites (ADC-264); the loops are verbatim, so each site
// stays bit-identical (copy_comp0 / copy_state and density_global / state_global / potential_global).
std::vector<double> gather_global(const MultiFab& mf, int ncomp, int gnx, int gny) {
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

bool newton_options_non_default(const NewtonOptions& newton, bool diagnostics = false) {
  return newton.max_iters != kNewtonDefaultMaxIters || newton.rel_tol != kNewtonDefaultRelTol ||
         newton.abs_tol != kNewtonDefaultAbsTol || newton.fd_eps != kNewtonDefaultFdEps ||
         diagnostics || newton.damping != kNewtonDefaultDamping ||
         newton.fail_policy != kNewtonDefaultFailPolicy;
}

EffectiveNewtonOptions effective_newton_options(const NewtonOptions& newton, bool diagnostics) {
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

EffectiveBlockOptions make_system_block_options(
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

int coupling_role_index_reported(const VariableSet& vs, VariableRole role, int fallback,
                                 const char* origin, const std::string& block) {
  if (vs.roles.empty())
    record_fallback(FallbackCounter::kRolelessComponentIndex);
  return coupling_role_index(vs, role, fallback, origin, block);
}
}  // namespace

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
  std::string time_scheme_ = "lie";
  std::string gauss_policy_ = "restart";
  double t = 0;
  int macro_step_ = 0;  // macro-step counter (0-indexed): feeds the per-block stride filter
  // COMPILED TIME-PROGRAM RUNTIME STATE (ADC-594): the whole compiled-Program subsystem -- installed
  // step + dt bound, cadence, IR-hash guard, name-based block map, runtime params, recorded
  // diagnostics, the profiler, the scheduler cache and the multistep history rings -- extracted out of
  // this god-object into ONE inspectable struct (include/pops/runtime/program/program_runtime_state.hpp),
  // SHARED verbatim with AmrSystem::Impl (the documented common contract). The stepper reads only
  // program_.step_ / substeps_ / stride_ / dt_bound_ (so the
  // tests/cpp/unit/numerics/test_strang_splitting.cpp MockImpl embeds the SAME struct); the diagnostics /
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
  // (stride_due), the condensed source stage (run_source_stage) and the couplings (apply_couplings). owner_
  // = this: the stepper reads the SHARED sp / fields_ / aux / couplings / t / macro_step_ / geom / pgeom_ / polar_
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

  // apply_couplings (inter-species coupling sources by splitting, AFTER transport) and
  // run_source_stage (Schur-condensed source stage, OPT-IN) EXTRACTED into stepper_ (SystemStepper,
  // Batch B): these are time-advance steps, invoked by step / step_cfl / step_adaptive.
  // They read the SHARED state via owner_-> (couplings, fields_.ell_phi(), aux, kAuxBaseComps). The
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
  GridContext grid_ctx() { return GridContext{dom, bc_, geom, &aux, &domain_mask_, &eb_domain_}; }

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
  void solve_fields() {
    program_.profiler_.count("kernels");
    fields_.solve_fields();
  }
  // Per-stage field solve (ADC-409): re-solve + re-fill the shared aux from a stage state of the
  // target block (the rest of the blocks keep their live s.U). Same delegation idiom as solve_fields.
  void solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
    fields_.solve_fields_from_state(block_idx, U_stage);
  }
  // Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): re-solve + re-fill the shared aux
  // from the SIMULTANEOUS stage states of all blocks (assemble_poisson_rhs_from_blocks). Same
  // delegation idiom as solve_fields / solve_fields_from_state.
  void solve_fields_from_blocks(const std::vector<const MultiFab*>& U_stages) {
    fields_.solve_fields_from_blocks(U_stages);
  }
  // NAMED multi-elliptic field (ADC-428): a SECOND elliptic solve for the user-named @p field, from a
  // stage state of @p block_idx, written to the field's OWN aux components. The default Poisson path
  // (solve_fields / solve_fields_from_state) is untouched. Same delegation idiom.
  void solve_named_field_from_state(const std::string& field, int block_idx,
                                    const MultiFab& U_stage) {
    fields_.solve_named_field_from_state(field, block_idx, U_stage);
  }
  void register_elliptic_field(const std::string& field, int phi_comp, int gx_comp, int gy_comp) {
    fields_.register_named_field(field, phi_comp, gx_comp, gy_comp);
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
};

namespace {
// Geometry guard (polar-grid project). The geometry CHOICE is carried by the config
// (pops.CartesianMesh / pops.PolarMesh). "cartesian": historical path, bit-identical. "polar": global
// ring (r, theta) wired into System.step (Phase 2b): polar transport (assemble_rhs_polar) +
// polar Poisson (PolarPoissonSolver) + aux in local basis (e_r, e_theta). We validate HERE the radial
// bounds of the ring (r_max > r_min >= 0); the Python (PolarMesh) already validates them, but a caller
// that builds the SystemConfig by hand must also be protected. Any other token is an error.
void check_geometry(const SystemConfig& c) {
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
void validate_system_config(const SystemConfig& c) {
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
void require_assembling(const pops::runtime::system::SystemLifecycle& lifecycle, const char* what) {
  if (lifecycle.frozen())
    throw std::runtime_error(
        std::string("System::") + what +
        ": the composition is frozen once pops.bind completes (runtime lifecycle 'bound'); declare "
        "it on the pops.Case (blocks / field problems / AMR layout / source stage / refinement / "
        "solver routes / aux layout / installed Program) and lower it with pops.compile(...) + "
        "pops.bind(...). Only runtime data / params / checkpoint / diagnostics may change on a "
        "bound simulation.");
}
}  // namespace

System::System(const SystemConfig& c) {
  validate_system_config(c);  // BEFORE any allocation/derivation (Impl builds geom/ba/dm/aux)
  p_ = std::make_unique<Impl>(c);
}
System::~System() = default;
System::System(System&&) noexcept = default;
System& System::operator=(System&&) noexcept = default;

void System::add_block(const std::string& name, const ModelSpec& model, const std::string& limiter,
                       const std::string& riemann, const std::string& recon,
                       const std::string& time, int substeps, bool evolve, int stride,
                       const std::vector<std::string>& implicit_vars,
                       const std::vector<std::string>& implicit_roles, const NewtonOptions& newton,
                       bool newton_diagnostics, double positivity_floor, bool wave_speed_cache) {
  Impl* P = p_.get();
  require_assembling(P->lifecycle_, "add_block");  // frozen once pops.bind completes (ADC-592)
  // Completeness contract of the model (ADC-290): transport / elliptic must be chosen explicitly.
  // Validated HERE, before the transport string routing below (which would otherwise report a
  // cryptic "unknown transport ''" for an unset tag) -- a default-constructed ModelSpec no longer
  // means a silent Euler + Poisson-charge composition.
  detail::validate_model_spec(model);
  if (substeps < 1)
    throw std::runtime_error("System::add_block : substeps >= 1");
  if (stride < 1)
    throw std::runtime_error("System::add_block : stride >= 1");
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error("System::add_block : positivity_floor >= 0 and finite (0 = inactive)");
  // Validation of the NEWTON OPTIONS POD (ADC-214): range check shared with AmrSystem::add_block
  // (validate_newton_options, in implicit_stepper.hpp). Whether non-default options are ALLOWED
  // (the time='imex' gate below) stays here -- it differs from the AMR path.
  validate_newton_options(newton, "System::add_block");
  // @p time carries the TREATMENT and, in explicit, the RK SCHEME: "explicit"/"ssprk2" = SSPRK2
  // (historical default), "ssprk3" = SSPRK3 (order 3), "euler" = ForwardEuler (order 1, fidelity to
  // first-order references -- validation), "imex" = explicit transport + local backward-Euler implicit
  // stiff source (order 1), "imexrk_ars222" = IMEX-RK family scheme ARS(2,2,2)
  // (order 2, distinct PARALLEL advance, Cartesian only). The RK math stays a CORE FUNCTOR
  // (build_block). "imex" and "imexrk_ars222" share the @c imex flag; @c method distinguishes them.
  if (time != "explicit" && time != "ssprk2" && time != "ssprk3" && time != "euler" &&
      time != "imex" && time != "imexrk_ars222")
    throw std::runtime_error(
        "System::add_block : time 'explicit'|'ssprk2'|'ssprk3'|'euler'|'imex'|'imexrk_ars222' "
        "(received '" +
        time + "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (received '" +
                             recon + "')");
  const bool imexrk = (time == "imexrk_ars222");
  const bool imex = (time == "imex" || imexrk);  // both go through the implicit source step
  const bool recon_prim = (recon == "primitive");
  // Wave speed cache (opt-in): only engages for the HLL flux and the explicit advance. Requesting it
  // elsewhere would be SILENTLY without effect -> explicit error (no silent ignore). The polar path has
  // its own factory (make_block_polar) without this cache.
  if (wave_speed_cache) {
    if (riemann != "hll")
      throw std::runtime_error(
          "System::add_block : wave_speed_cache requires riemann='hll' (the wave "
          "speed cache only applies to the HLL flux ; received riemann='" +
          riemann + "')");
    if (imex)
      throw std::runtime_error("System::add_block : wave_speed_cache not supported with time='" +
                               time +
                               "' (wired on the explicit advance ; use time "
                               "'explicit'/'ssprk2'/'ssprk3'/'euler')");
    if (P->polar_)
      throw std::runtime_error(
          "System::add_block : wave_speed_cache not supported on the polar "
          "geometry (ring)");
    // EMBEDDED-BOUNDARY transport mode already active: the stepper routes to advance_masked /
    // advance_eb, which do not carry the cache -> requesting it would be WITHOUT EFFECT. Explicit
    // rejection (no silent ignore). The reverse order (set_disc_domain AFTER a cached block) is
    // rejected by set_disc_domain / set_geometry_mode.
    if (P->eb_set_ && P->geometry_mode_ != GeometryMode::None)
      throw std::runtime_error(
          "System::add_block : wave_speed_cache incompatible with an active "
          "embedded-boundary transport mode (staircase/cutcell) ; the cache is only "
          "wired on the full Cartesian advance (remove wave_speed_cache or mode='none')");
    P->ws_cache_block_ = true;  // a block requested the cache -> locks the switch to disc mode
  }
  const std::string method = imexrk ? std::string("imexrk_ars222")
                                    : ((time == "ssprk3")  ? std::string("ssprk3")
                                       : (time == "euler") ? std::string("euler")
                                                           : std::string("ssprk2"));
  // The implicit mask (implicit_vars / implicit_roles) applies only to the IMEX source step. Requesting
  // it in explicit is an ERROR (no silent ignore): the explicit has no implicit step.
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "System::add_block : implicit_vars / implicit_roles require time='imex' "
        "(the implicit mask applies only to the IMEX source step ; received time='" +
        time + "')");
  // IMEX-RK ARS(2,2,2): FULLY implicit source (the stage consistency relation assumes a homogeneous
  // solve). A partial mask would be SILENTLY ignored there -> we reject it explicitly. The
  // partial mask stays available on time='imex' (local backward-Euler).
  if (imexrk && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "System::add_block : implicit_vars / implicit_roles (partial IMEX mask) unsupported by "
        "time='imexrk_ars222' (its source is FULLY implicit). Use time='imex' for a "
        "partial mask, or remove implicit_vars / implicit_roles.");
  // Same rules for the Newton options/diagnostics: they only drive the IMEX source step.
  // Non-default values in explicit would be SILENTLY ignored -> explicit error.
  const bool newton_non_default = newton_options_non_default(newton, newton_diagnostics);
  if (!imex && newton_non_default)
    throw std::runtime_error(
        "System::add_block : the Newton options (newton_max_iters/rel_tol/"
        "abs_tol/fd_eps/diagnostics) require time='imex' (received time='" +
        time + "')");

  int ncomp = 1;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  std::function<Real(const MultiFab&)> src_freq, stab_dt;  // optional step bounds (model traits)
  CellConvert prim_to_cons, cons_to_prim;  // pointwise model conversions (set/get_primitive_state)
  VariableSet cons_vs, prim_vs;
  detail::BuiltBlock bb;
  if (P->polar_) {
    // POLAR PATH (ring): closures built by block_builder_polar.hpp (assemble_rhs_polar + scalar polar
    // transport ExBVelocityPolar OR fluid IsothermalFluxPolar + scalar polar Poisson), via the polar
    // seam (python/system_polar.cpp, ADC-335). IMEX is not supported on the ring at this stage: the
    // electrostatic coupling goes through an explicit LOCAL source (non-stiff regime, Path A step 1);
    // we reject it explicitly rather than silently running the transport alone.
    if (imex)
      throw std::runtime_error(
          "System::add_block (polar) : time='" + time +
          "' (IMEX / IMEX-RK ARS(2,2,2)) unsupported "
          "(ring : coupling by explicit local source, no stiff source to handle implicitly "
          "at this stage). Use 'explicit'/'ssprk2'/'ssprk3'.");
    const PolarGridContext pctx = P->grid_ctx_polar();
    bb = detail::build_block_polar(model, limiter, riemann, pctx, recon_prim, method,
                                   static_cast<Real>(positivity_floor), &P->aux);
    // ADC-291: widen the shared aux to the polar block's read width (canonical extras AND model-named
    // extra[k]), mirroring the Cartesian branch below. ensure_aux_width keeps the aux ADDRESS captured
    // by the closures and re-applies B_z / named aux on realloc; without it a polar n_aux>3 model read
    // out of bounds. No-op for a base (n_aux=3) model -> bit-identical.
    P->ensure_aux_width(bb.aux_width);
  } else {
    const GridContext ctx = P->grid_ctx();
    // Newton options of the IMEX implicit source (defaults = historical constants, bit-identical).
    // The report lives in diagnostics_.newton_reports in a shared_ptr -> STABLE address captured by
    // the closures even when the map reallocates at a later add_block. It is allocated for explicit
    // diagnostics and for fail_policy warn/throw, because those policies must surface as structured
    // report events rather than stderr text.
    const NewtonOptions& nopts = newton;
    NewtonReport* nreport = nullptr;
    if (newton_diagnostics || nopts.fail_policy != NewtonOptions::kFailNone) {
      auto rep = std::make_shared<NewtonReport>();
      P->diagnostics_.newton_reports[name] = rep;
      nreport = rep.get();
    }
    // Transport-axis seam (ADC-335): each per-transport TU (python/system_<transport>.cpp) runs the
    // SAME source/elliptic dispatch + make_block + makers as before (detail::build_block_for), but
    // instantiates ONLY its own transport's leaves -- so the combinatorial product splits across files
    // for `-j`. This string if/else mirrors detail::dispatch_transport (same unknown-transport message).
    // aux_width is widened host-side AFTER the build (was P->ensure_aux_width inside the visitor;
    // ensure_aux_width keeps the aux ADDRESS captured by the closures, so order vs make_block is
    // immaterial -- byte-identical).
    const detail::BlockBuildArgs args{name,
                                      limiter,
                                      riemann,
                                      ctx,
                                      imex,
                                      recon_prim,
                                      method,
                                      implicit_vars,
                                      implicit_roles,
                                      nopts,
                                      nreport,
                                      static_cast<Real>(positivity_floor),
                                      wave_speed_cache};
    if (model.transport == "exb") {
      bb = detail::build_block_exb(model, args);
    } else if (model.transport == "compressible") {
      // Compressible/Euler is flux-subdivided (ADC-335): all four fluxes are valid (4-var + pressure),
      // so we run the SAME validation as make_block (validate_riemann then validate_limiter, identical
      // messages) and dispatch the riemann string to the matching per-flux sub-TU. An unknown flux hits
      // the same registry throw as make_block's tail (validate_riemann already rejected it).
      validate_riemann(riemann, /*polar=*/false, "System");
      validate_limiter(limiter, "System");
      if (riemann == "rusanov") {
        bb = detail::build_block_compressible_rusanov(model, args);
      } else if (riemann == "hll") {
        bb = detail::build_block_compressible_hll(model, args);
      } else if (riemann == "hllc" || riemann == "euler_hllc") {
        // On the true Euler brick the EXPLICIT euler_hllc route and the generic hllc route are the
        // SAME arithmetic (the native Euler now provides HasHLLCStructure with the canonical-Euler
        // formulas, so HLLCFlux == the former EulerHLLCFlux2D fallback bit-for-bit): both share this
        // seam leaf (ADC-590). euler_hllc's 4-var+pressure gate is satisfied by CompressibleFlux.
        bb = detail::build_block_compressible_hllc(model, args);
      } else if (riemann == "roe" || riemann == "euler_roe") {
        bb = detail::build_block_compressible_roe(model, args);
      } else {
        throw_registry_dispatch_mismatch("System", "flux", riemann);
      }
    } else if (model.transport == "isothermal") {
      // Isothermal is flux-subdivided (ADC-342): only rusanov + hll are reachable (3-var, no pressure
      // for hllc/roe). The per-flux seams call make_block_<flux> directly, so -- like compressible --
      // we run make_block's validation here (validate_riemann then validate_limiter, identical messages)
      // before dispatching; hllc/roe and any unknown flux hit the registry throw (explicit, no UB).
      validate_riemann(riemann, /*polar=*/false, "System");
      validate_limiter(limiter, "System");
      if (riemann == "rusanov") {
        bb = detail::build_block_isothermal_rusanov(model, args);
      } else if (riemann == "hll") {
        bb = detail::build_block_isothermal_hll(model, args);
      } else {
        throw_registry_dispatch_mismatch("System", "flux", riemann);
      }
    } else {
      throw std::runtime_error(unknown_transport_msg(model.transport));
    }
    P->ensure_aux_width(bb.aux_width);
  }
  ncomp = bb.ncomp;
  cons_vs = std::move(bb.cons_vs);
  prim_vs = std::move(bb.prim_vs);
  clo = std::move(bb.clo);
  max_speed = std::move(bb.max_speed);
  add_poisson_rhs = std::move(bb.add_poisson_rhs);
  src_freq = std::move(bb.src_freq);
  stab_dt = std::move(bb.stab_dt);
  prim_to_cons = std::move(bb.prim_to_cons);
  cons_to_prim = std::move(bb.cons_to_prim);
  // Common installation (same path as add_compiled_model for a DSL-generated model):
  // the closures run on the REAL System MultiFabs (MPI halos via fill_boundary, device
  // via Kokkos), without copy.
  install_block(name, ncomp, cons_vs, prim_vs, model.gamma, std::move(clo), std::move(max_speed),
                std::move(add_poisson_rhs), substeps, evolve, stride);
  EffectiveBlockOptions block_options =
      make_system_block_options(name, model, "native_model", limiter, riemann, recon, time, method,
                                imex, substeps, evolve, stride, implicit_vars, implicit_roles,
                                newton, newton_diagnostics, positivity_floor, wave_speed_cache);
  block_options.ncomp = ncomp;
  block_options.conservative_vars = cons_vs.names;
  block_options.primitive_vars = prim_vs.names;
  P->diagnostics_.block_options[name] = std::move(block_options);
  set_block_conversion(name, std::move(prim_to_cons), std::move(cons_to_prim));
  set_block_dt_bounds(name, std::move(src_freq), std::move(stab_dt));
  // SCHEME GHOSTS: WENO5 reads a 5-point stencil (3 ghosts) > the 2 allocated by default in
  // install_block. We reallocate the block state with block_n_ghost(limiter) if needed (cf. AmrSystem which
  // allocates with Limiter::n_ghost, PR #22) so that fill_ghosts + assemble_rhs do not read out of
  // bounds. minmod/vanleer (2 ghosts): no-op, allocation and result bit-identical to before.
  P->set_block_ghosts(name, block_n_ghost(limiter));
}

// Real grid context (mesh + BC + aux): used by the add_compiled_model template to build
// the closures of an AOT-compiled model on the real System fields (native parity, without marshaling).
POPS_EXPORT GridContext System::grid_context() {
  return p_->grid_ctx();
}

// Installs a block from already-built closures (by dispatch_model on the add_block side, or by
// block_builder on the add_compiled_model side). Centralizes the creation of the species (U, names, scheme).
POPS_EXPORT void System::install_block(const std::string& name, int ncomp,
                                      const VariableSet& cons_vars, const VariableSet& prim_vars,
                                      double gamma, BlockClosures closures,
                                      std::function<Real(const MultiFab&)> max_speed,
                                      std::function<void(const MultiFab&, MultiFab&)> poisson_rhs,
                                      int substeps, bool evolve, int stride) {
  if (stride < 1)
    throw std::runtime_error("System::install_block : stride >= 1");
  Impl* P = p_.get();
  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), ncomp, substeps, evolve,
                                stride, gamma, std::move(closures.advance),
                                std::move(closures.rhs_into), std::move(max_speed),
                                std::move(poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
  P->sp.back().cons_vars = cons_vars;
  P->sp.back().prim_vars = prim_vars;
  // EMBEDDED-BOUNDARY transport advances (project T5-PR3): empty unless build_block built them
  // (Cartesian block with domain_mask_/eb_domain_ provided). Empty -> the stepper falls back on advance
  // (bit-identical).
  P->sp.back().advance_masked = std::move(closures.advance_masked);
  P->sp.back().advance_eb = std::move(closures.advance_eb);
  P->sp.back().hotspot = std::move(closures.hotspot);  // dt_hotspot diagnostic (ADC-182)
  // Projection ponctuelle post-pas (ADC-177) : vide sauf si le modele declare le trait
  // HasPointwiseProjection (make_block). Vide -> le stepper ne l'interroge pas (bit-identique).
  P->sp.back().project = std::move(closures.project);
  // FLUX-ONLY residual -div F(U) (ADC-425): set for native blocks (build_block builds it via
  // SourceFreeModel<Model>); empty for paths that do not (the host .so prototype loader) ->
  // block_neg_div_flux_into fails loud rather than silently leaking the default source.
  P->sp.back().rhs_flux_only = std::move(closures.rhs_flux_only);
  // SOURCE-ONLY residual S(U, aux) (ADC-430): set for native blocks (build_block builds it via
  // SourceInto<Model>); empty for paths that do not (the host .so prototype loader) ->
  // block_source_into fails loud rather than silently leaking the flux.
  P->sp.back().source_only = std::move(closures.source_only);
  EffectiveBlockOptions& opt = P->diagnostics_.block_options[name];
  opt.name = name;
  if (opt.route.empty())
    opt.route = "closure_install";
  opt.ncomp = ncomp;
  opt.n_ghost = P->sp.back().U.n_grow();
  opt.substeps = substeps;
  opt.stride = stride;
  opt.evolve = evolve;
  opt.gamma = gamma;
  opt.conservative_vars = cons_vars.names;
  opt.primitive_vars = prim_vars.names;
}

// Width-aware reallocation of a block state (delegates to Impl::set_block_ghosts). Exposed
// (POPS_EXPORT) so that the add_compiled_model header template (native path, .so loader) can
// widen the compiled block to block_n_ghost(limiter) -- 3 for weno5 -- as add_block does.
POPS_EXPORT void System::set_block_ghosts(const std::string& name, int n_ghost) {
  p_->set_block_ghosts(name, n_ghost);
  if (EffectiveBlockOptions* opt = p_->diagnostics_.block_options_ptr(name))
    opt->n_ghost = p_->find(name).U.n_grow();
}

// OPTIONAL step bounds of a block (model traits): set after install_block, read by
// step_cfl / step_adaptive. Empty functions = the block imposes no bound (historical).
void System::set_block_dt_bounds(const std::string& name,
                                 std::function<Real(const MultiFab&)> source_frequency,
                                 std::function<Real(const MultiFab&)> stability_dt) {
  Impl::Species& s = p_->find(name);  // raises if unknown block
  s.source_frequency = std::move(source_frequency);
  s.stability_dt = std::move(stability_dt);
}

// GLOBAL step bound (host, one evaluation per step): multi-block coupling, Schur/Poisson,
// scheduler, user policy. cf. SystemStepper::step_cfl for the aggregation.
void System::add_dt_bound(const std::string& label, std::function<double()> fn) {
  require_assembling(p_->lifecycle_, "add_dt_bound");  // frozen once pops.bind completes (ADC-592)
  if (!fn)
    throw std::runtime_error("System::add_dt_bound : empty bound function");
  p_->dt_bounds_.push_back(Impl::GlobalDtBound{label, std::move(fn)});
}

// ACTIVE bound of the last step_cfl (step-policy diagnostic). "" before the first step.
std::string System::last_dt_bound() const {
  return p_->stepper_.last_dt_reason();
}

// dt_hotspot diagnostic (ADC-182): the GLOBAL cell (i, j) that dominates the transport CFL
// bound of block @p name, and its speed w = max(wx, wy). ON DEMAND (two reduction
// passes, cf. max_wave_speed_hotspot_mf) -- step/step_cfl do not touch it. Block without
// closure (historical non-rewireable paths, e.g. dynamic) -> EXPLICIT error.
std::array<double, 3> System::dt_hotspot(const std::string& name) {
  Impl::Species& s = p_->find(name);
  if (!s.hotspot)
    throw std::runtime_error("System::dt_hotspot : block '" + name +
                             "' without hotspot diagnostic (non-rewireable add path)");
  Real w = 0;
  int i = -1, j = -1;
  s.hotspot(s.U, w, i, j);
  return {static_cast<double>(w), static_cast<double>(i), static_cast<double>(j)};
}

// Newton report (OPT-IN IMEX diagnostics) of the block: flat copy of the NewtonReport aggregated by the
// LAST advance of the block (reset at the start of the advance by AdvanceImex*). Clear error if the block did
// not enable newton_diagnostics (no silently empty report).
System::SourceNewtonReport System::newton_report(const std::string& name) const {
  p_->index(name);  // raises if unknown block
  const NewtonReport* rp = p_->diagnostics_.newton_report_ptr(name);
  if (rp == nullptr)
    throw std::runtime_error(
        "System::newton_report : Newton diagnostics not enabled for block '" + name +
        "' ; add the block with newton_diagnostics=true "
        "(pops.IMEX(newton_diagnostics=True) / pops.SourceImplicit(newton_diagnostics=True)) "
        "or newton_fail_policy='warn'/'throw'");
  const NewtonReport& r = *rp;
  return SourceNewtonReport{r.enabled,
                            r.converged,
                            static_cast<double>(r.max_residual),
                            static_cast<double>(r.max_iters_used),
                            r.n_failed,
                            r.failed_i,
                            r.failed_j,
                            r.failed_comp,
                            r.diagnostics.events};
}

// Body EXTRACTED VERBATIM into pops::native_loader::add_dynamic_block (native_loader.hpp); instantiated
// here with System::Impl (defined above, private to this TU). Bit-identical: pure delegation.
void System::add_dynamic_block(const std::string& name, const std::string& so_path, int substeps,
                               const std::vector<std::string>& names, const std::string& recon) {
  require_assembling(p_->lifecycle_, "add_dynamic_block");  // frozen once pops.bind completes (ADC-592)
  native_loader::add_dynamic_block(this, p_.get(), name, so_path, substeps, names, recon);
  EffectiveBlockOptions& opt = p_->diagnostics_.block_options[name];
  opt.route = "dynamic_loader";
  opt.compiled = false;
  opt.transport = "dynamic_model";
  opt.source = "dynamic_model";
  opt.elliptic = "dynamic_model";
  opt.limiter = recon;
  opt.riemann = "rusanov_global";
  opt.recon = recon;
  opt.time = "explicit";
  opt.time_method = "host_euler";
}

// Body EXTRACTED VERBATIM into pops::native_loader::add_compiled_block (native_loader.hpp); instantiated
// here with System::Impl. Bit-identical: pure delegation.
void System::add_compiled_block(const std::string& name, const std::string& so_path,
                                const std::string& limiter, const std::string& riemann,
                                const std::string& recon, const std::string& time, int substeps,
                                const std::vector<std::string>& names, double positivity_floor) {
  require_assembling(p_->lifecycle_, "add_compiled_block");  // frozen once pops.bind completes (ADC-592)
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "System::add_compiled_block : positivity_floor >= 0 and finite (0 = inactive)");
  native_loader::add_compiled_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                    substeps, names, positivity_floor);
  EffectiveBlockOptions& opt = p_->diagnostics_.block_options[name];
  opt.route = "aot_loader";
  opt.compiled = true;
  opt.transport = "compiled_artifact";
  opt.source = "compiled_artifact";
  opt.elliptic = "compiled_artifact";
  opt.limiter = limiter;
  opt.riemann = riemann;
  opt.recon = recon;
  opt.time = time;
  opt.time_method = time;
  opt.imex = (time == "imex");
  opt.substeps = substeps;
  opt.positivity_floor = positivity_floor;
}

// P7-b: overwrites the SHARED vector of runtime parameter values of block @p name. add_compiled_block
// registered this vector in p_->block_params_ AND captured it in the block closures: writing
// into it suffices to change the behavior at the next step, WITHOUT recompiling the .so. Explicit error if
// the block has no runtime params (vector absent) or if values does not have the right size.
void System::set_block_params(const std::string& name, const std::vector<double>& values) {
  // index() raises "System: unknown block '...'" if the block does not exist (same diagnostic as everywhere).
  (void)p_->blocks_.index(name);
  auto it = p_->block_params_.find(name);
  if (it == p_->block_params_.end())
    throw std::runtime_error(
        "System::set_block_params : block '" + name +
        "' has no runtime parameter (declare dsl.Param(..., kind='runtime') and wire via "
        "backend='aot' / add_compiled_block ; const params are frozen at compile time)");
  std::vector<double>& pv = *it->second;
  if (values.size() != pv.size())
    throw std::runtime_error("System::set_block_params : block '" + name + "' expects " +
                             std::to_string(pv.size()) + " runtime parameters, received " +
                             std::to_string(values.size()));
  pv = values;  // the vector is SHARED with the closures (shared_ptr): effect at the next step
}

// Body EXTRACTED VERBATIM into pops::native_loader::add_native_block (native_loader.hpp); instantiated
// here with System::Impl. Bit-identical: pure delegation (this marshals to the unchanged native loader).
void System::add_native_block(const std::string& name, const std::string& so_path,
                              const std::string& limiter, const std::string& riemann,
                              const std::string& recon, const std::string& time, double gamma,
                              int substeps, bool evolve, int stride, double positivity_floor) {
  require_assembling(p_->lifecycle_, "add_native_block");  // frozen once pops.bind completes (ADC-592)
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "System::add_native_block : positivity_floor >= 0 and finite (0 = inactive)");
  native_loader::add_native_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                  gamma, substeps, evolve, stride, positivity_floor);
  EffectiveBlockOptions& opt = p_->diagnostics_.block_options[name];
  opt.route = "native_loader";
  opt.compiled = true;
  opt.transport = "compiled_artifact";
  opt.source = "compiled_artifact";
  opt.elliptic = "compiled_artifact";
  opt.limiter = limiter;
  opt.riemann = riemann;
  opt.recon = recon;
  opt.time = time;
  if (time == "imex")
    opt.time_method = "imex";
  else if (time == "ssprk3")
    opt.time_method = "ssprk3";
  else if (time == "euler")
    opt.time_method = "euler";
  else
    opt.time_method = "ssprk2";
  opt.imex = (time == "imex");
  opt.substeps = substeps;
  opt.stride = stride;
  opt.evolve = evolve;
  opt.gamma = gamma;
  opt.positivity_floor = positivity_floor;
}

void System::set_poisson(const std::string& rhs, const std::string& solver, const std::string& bc,
                         const std::string& wall, double wall_radius, double epsilon,
                         double abs_tol) {
  require_assembling(p_->lifecycle_, "set_poisson");  // frozen once pops.bind completes (ADC-592)
  if (epsilon == 0.0)
    throw std::runtime_error("System::set_poisson : epsilon != 0 required");
  if (abs_tol < 0.0)
    throw std::runtime_error("System::set_poisson : abs_tol >= 0 required");
  p_->fields_.p_rhs = rhs;
  p_->fields_.p_solver = solver;
  p_->fields_.p_bc = bc;
  p_->fields_.p_wall = wall;
  p_->fields_.p_wall_radius = wall_radius;
  p_->fields_.p_eps_ = static_cast<Real>(epsilon);
  p_->fields_.p_abs_tol_ =
      static_cast<Real>(abs_tol);  // absolute floor of the V-cycle (0 = relative only)
  p_->fields_.ell_.reset();
}

namespace {
// Translates the Python disc transport mode ("none"|"staircase"|"cutcell") into a GeometryMode. EXPLICIT
// error on an unknown mode (never a silent fallback). Single source of the name table.
GeometryMode parse_geometry_mode(const std::string& mode, const char* err_context) {
  if (mode == "none")
    return GeometryMode::None;
  if (mode == "staircase")
    return GeometryMode::Staircase;
  if (mode == "cutcell")
    return GeometryMode::CutCell;
  throw std::runtime_error(std::string(err_context) + " : unknown geometry mode '" + mode +
                           "' (none|staircase|cutcell)");
}
}  // namespace

void System::set_disc_domain(double cx, double cy, double R, const std::string& mode) {
  Impl* P = p_.get();
  require_assembling(P->lifecycle_, "set_disc_domain");  // frozen once pops.bind completes (ADC-592)
  // CARTESIAN only: polar already bounds the ring by its radial walls (r_min / r_max,
  // zero radial flux) -> a Cartesian disc mask makes no sense on the (r, theta) grid.
  if (P->polar_)
    throw std::runtime_error(
        "System::set_disc_domain : polar geometry (the ring is already bounded by its radial "
        "walls r_min/r_max ; the Cartesian disc mask does not apply)");
  if (!(R > 0.0))
    throw std::runtime_error("System::set_disc_domain : radius R > 0 required");
  // Validate the mode BEFORE any mutation (an unknown mode must not leave the disc half-set).
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_disc_domain");
  // wave_speed_cache (ADC-199) is only wired on the full Cartesian advance: a disc mode
  // (staircase/cutcell) borrows advance_masked / advance_eb which ignore the cache -> explicit rejection.
  if (gmode != GeometryMode::None && P->ws_cache_block_)
    throw std::runtime_error(
        "System::set_disc_domain : mode '" + mode +
        "' incompatible with wave_speed_cache (a block enabled the HLL wave speed "
        "cache, only wired on the full Cartesian advance ; remove wave_speed_cache "
        "or use mode='none')");
  P->eb_domain_ = detail::DiscDomain{cx, cy, R};
  P->eb_set_ = true;
  // Materializes the 0/1 cell-centered mask (1 ghost, so the mask-aware transport reads the
  // i-1/i+1/j-1/j+1 neighbors up to the edge). Same layout as the blocks (ba/dm). Cell active when
  // its CENTER is inside the disc (level set < 0, SAME convention as the conducting wall).
  P->domain_mask_ = MultiFab(P->ba, P->dm, 1, 1);
  const detail::DiscDomain disc = P->eb_domain_;
  const Geometry geom = P->geom;
  for (int li = 0; li < P->domain_mask_.local_size(); ++li) {
    Array4 m = P->domain_mask_.fab(li).array();
    // box WITH ghosts: we also classify the ghosts (the mask-aware transport reads the edge neighbors).
    const Box2D g = P->domain_mask_.fab(li).grown_box();
    for_each_cell(g, [=] POPS_HD(int i, int j) {
      m(i, j, 0) = disc.cell_active(geom.x_cell(i), geom.y_cell(j)) ? Real(1) : Real(0);
    });
  }
  // TRANSPORT ROUTING (project T5-PR3). mode == "none": the mask is materialized (queryable
  // via disc_mask()) but the transport stays FULL Cartesian -> bit-identical. mode != "none": the
  // stepper routes the advance to assemble_rhs_masked (staircase) / assemble_rhs_eb (cutcell).
  P->geometry_mode_ = gmode;
}

void System::set_geometry_mode(const std::string& mode) {
  Impl* P = p_.get();
  require_assembling(P->lifecycle_, "set_geometry_mode");  // frozen once pops.bind completes (ADC-592)
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_geometry_mode");
  // An embedded-boundary mode (staircase/cutcell) only makes sense with a fixed domain: otherwise the
  // stepper would fall back on the full transport (the mask / level set does not exist), a silent
  // footgun -> we reject.
  if (gmode != GeometryMode::None && !P->eb_set_)
    throw std::runtime_error(
        "System::set_geometry_mode : embedded-boundary mode '" + mode +
        "' requested without a fixed level-set domain ; call set_disc_domain(cx, cy, R) first");
  // wave_speed_cache (ADC-199) is not carried by the disc advances -> explicit rejection (cf.
  // set_disc_domain) rather than a cache silently ignored in staircase/cutcell mode.
  if (gmode != GeometryMode::None && P->ws_cache_block_)
    throw std::runtime_error(
        "System::set_geometry_mode : mode '" + mode +
        "' incompatible with wave_speed_cache (a block enabled the HLL wave speed "
        "cache, only wired on the full Cartesian advance ; remove wave_speed_cache "
        "or use mode='none')");
  P->geometry_mode_ = gmode;
}

std::vector<double> System::disc_mask() const {
  Impl* P = p_.get();
  device_fence();
  const Box2D v = P->dom;
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  if (!P->eb_set_) {
    // CONTRACT: without a fixed domain, the transport subdomain is the whole domain -> all active.
    out.assign(static_cast<std::size_t>(v.nx()) * v.ny(), 1.0);
    return out;
  }
  const ConstArray4 m = P->domain_mask_.fab(0).const_array();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      out.push_back(static_cast<double>(m(i, j, 0)));
  return out;
}

void System::set_epsilon_field(const std::vector<double>& eps) {
  require_assembling(p_->lifecycle_, "set_epsilon_field");  // frozen once pops.bind completes (ADC-592)
  const int n = p_->cfg.n;
  if (static_cast<int>(eps.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_field : size != n*n");
  for (double e : eps)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_field : permittivity eps(x) > 0 required");
  p_->fields_.p_eps_field_ = eps;
  p_->fields_.has_eps_field_ = true;
  p_->fields_.ell_
      .reset();  // the operator will be rebuilt with the eps field at the next solve_fields
}

void System::set_epsilon_anisotropic_field(const std::vector<double>& eps_x,
                                           const std::vector<double>& eps_y) {
  require_assembling(p_->lifecycle_,
                     "set_epsilon_anisotropic_field");  // frozen once pops.bind completes (ADC-592)
  const int n = p_->cfg.n;
  if (static_cast<int>(eps_x.size()) != n * n || static_cast<int>(eps_y.size()) != n * n)
    throw std::runtime_error(
        "System::set_epsilon_anisotropic_field : size != n*n (eps_x and eps_y)");
  for (double e : eps_x)
    if (!(e > 0.0))
      throw std::runtime_error(
          "System::set_epsilon_anisotropic_field : permittivity eps_x(x) > 0 required");
  for (double e : eps_y)
    if (!(e > 0.0))
      throw std::runtime_error(
          "System::set_epsilon_anisotropic_field : permittivity eps_y(x) > 0 required");
  p_->fields_.p_eps_x_field_ = eps_x;
  p_->fields_.p_eps_y_field_ = eps_y;
  p_->fields_.has_eps_xy_field_ = true;
  p_->fields_.ell_
      .reset();  // operator rebuilt as div(diag(eps_x, eps_y) grad phi) at the next solve_fields
}

void System::set_reaction_field(const std::vector<double>& kappa) {
  require_assembling(p_->lifecycle_, "set_reaction_field");  // frozen once pops.bind completes (ADC-592)
  const int n = p_->cfg.n;
  if (static_cast<int>(kappa.size()) != n * n)
    throw std::runtime_error("System::set_reaction_field : size != n*n");
  for (double k : kappa)
    if (!(k >= 0.0))
      throw std::runtime_error(
          "System::set_reaction_field : reaction term kappa(x) >= 0 required "
          "(well-posed elliptic operator and convergent multigrid)");
  p_->fields_.p_kappa_field_ = kappa;
  p_->fields_.has_kappa_field_ = true;
  p_->fields_.ell_.reset();  // operator rebuilt with - kappa phi at the next solve_fields
}

POPS_EXPORT void System::ensure_aux_width(int ncomp) {
  p_->ensure_aux_width(ncomp);
}

void System::set_magnetic_field(const std::vector<double>& bz) {
  // Expected size of the B_z(x) field row-major (slow axis = 2nd box index, fast axis = 1st):
  //   Cartesian = n * n (square, BIT-IDENTICAL); POLAR = nr * ntheta (ring, i = r fast, cf.
  //   apply_bz / polar set_density). The layout is the SAME as set_density (flat[j * nr + i]).
  if (p_->polar_) {
    const int nr = Impl::polar_nr(p_->cfg), nth = Impl::polar_ntheta(p_->cfg);
    if (static_cast<int>(bz.size()) != nr * nth)
      throw std::runtime_error("System::set_magnetic_field : size != nr*ntheta (polar)");
  } else {
    const int n = p_->cfg.n;
    if (static_cast<int>(bz.size()) != n * n)
      throw std::runtime_error("System::set_magnetic_field : size != n*n");
  }
  p_->fields_.bz_field_.assign(bz.begin(), bz.end());
  p_->fields_
      .apply_bz();  // apply right away if a block already reads B_z; otherwise keep for ensure_aux_width
}

void System::set_electron_temperature_from(const std::string& name) {
  require_assembling(p_->lifecycle_,
                     "set_electron_temperature_from");  // frozen once pops.bind completes (ADC-592)
  const int idx = p_->index(name);  // raises if unknown block
  if (p_->sp[static_cast<std::size_t>(idx)].ncomp != 4)
    throw std::runtime_error(
        "System::set_electron_temperature_from : block '" + name +
        "' must be compressible (4 vars : rho, rho u, rho v, E) for T = p/rho");
  p_->fields_.te_src_ = idx;
  // T_e (canonical comp 4) DERIVED: recomputed at each solve_fields. Inert as long as no block
  // reads T_e (n_aux=5 -> ensure_aux_width(5)), like set_magnetic_field for B_z.
  p_->fields_.apply_te();
}

// Expected size of a cell-defined field (Cartesian n*n / polar nr*ntheta). Member of Impl:
// a free caller could not name the private type System::Impl.
std::size_t System::Impl::aux_field_cell_count() const {
  if (polar_) {
    const int nr = polar_nr(cfg), nth = polar_ntheta(cfg);
    return static_cast<std::size_t>(nr) * nth;
  }
  return static_cast<std::size_t>(cfg.n) * cfg.n;
}

void System::set_aux_field_component(int comp, const std::vector<double>& field) {
  Impl* P = p_.get();
  // RESERVED components (phi/grad/B_z/T_e): a named aux field starts at kAuxNamedBase (= 5).
  // B_z and T_e keep their dedicated paths -> redirecting message (the Python facade already intercepts
  // the canonical names, this guard covers a direct C++ call).
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "System::set_aux_field : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase) +
        " (B_z -> set_magnetic_field, T_e -> "
        "set_electron_temperature_from)");
  const std::size_t expect = P->aux_field_cell_count();
  if (field.size() != expect)
    throw std::runtime_error("System::set_aux_field : size " + std::to_string(field.size()) +
                             " != " + std::to_string(expect) + " (grid cells)");
  // The aux channel must be wide enough: a block declaring this field (n_aux = kAuxNamedBase + k + 1) has
  // already called ensure_aux_width at its add time. Otherwise the field would be read by no model -> error.
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::set_aux_field : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp) +
        " (add the block that reads it before set_aux_field)");
  std::vector<Real> f(field.begin(), field.end());
  p_->fields_.apply_named_aux_one(comp, f);     // populate right away (channel wide enough)
  p_->fields_.named_aux_[comp] = std::move(f);  // keep for a later reallocation of the channel
}

void System::set_aux_field_halo_component(int comp, int bc_type, double value) {
  Impl* P = p_.get();
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "System::set_aux_field (halo) : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase));
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::set_aux_field (halo) : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp));
  // Only the PHYSICAL-face policies are meaningful per field (Foextrap / Dirichlet). A periodic face is
  // a domain property kept by aux_halo_override, so a per-field 'periodic' is not offered.
  if (bc_type != static_cast<int>(BCType::Foextrap) &&
      bc_type != static_cast<int>(BCType::Dirichlet))
    throw std::runtime_error("System::set_aux_field (halo) : unsupported halo type " +
                             std::to_string(bc_type) + " ; use foextrap or dirichlet");
  P->fields_.named_aux_bc_[comp] =
      AuxHaloPolicy{static_cast<BCType>(bc_type), static_cast<Real>(value)};
}

std::vector<double> System::aux_field_component(int comp) const {
  Impl* P = p_.get();
  if (comp < kAuxNamedBase)
    throw std::runtime_error("System::aux_field : component " + std::to_string(comp) +
                             " reserved (phi/grad_x/grad_y/B_z/T_e) ; read phi via potential(), a "
                             "named aux field starts "
                             "at index " +
                             std::to_string(kAuxNamedBase));
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::aux_field : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp));
  device_fence();
  // Rank without a box (MPI mono-box): EMPTY return (cf. potential / copy_comp0). The Python facade is
  // mono-rank; the multi-rank global field would be a dedicated collective accessor (follow-up).
  if (P->aux.local_size() == 0)
    return {};
  const ConstArray4 a = P->aux.fab(0).const_array();
  const Box2D v = P->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      out.push_back(static_cast<double>(a(i, j, comp)));
  return out;
}

// The named inter-species couplings (System::add_ionization / add_collision / add_thermal_exchange)
// are removed (ADC-595): they are Python presets (python/pops/physics/coupling_presets.py) that emit the
// same formulas as a generic CoupledSource and register through add_coupling_operator with a declared
// conservation contract. Impl::couplings / coupled_freqs_ / coupled_freq_exprs_ STORAGE stays untouched
// (SystemStepper::apply_couplings / step_cfl read them); only the entry methods go.

void System::add_coupled_source(const CoupledSourceProgram& prog_desc, double frequency,
                                const std::string& label) {
  require_assembling(p_->lifecycle_, "add_coupled_source");  // frozen once pops.bind completes (ADC-592)
  // Bytecode description grouped into a POD (ADC-214): local aliases to keep the body readable (the
  // names and the semantics are strictly those of the old flat parameters).
  const std::vector<std::string>& in_blocks = prog_desc.in_blocks;
  const std::vector<std::string>& in_roles = prog_desc.in_roles;
  const std::vector<double>& consts = prog_desc.consts;
  const std::vector<std::string>& out_blocks = prog_desc.out_blocks;
  const std::vector<std::string>& out_roles = prog_desc.out_roles;
  const std::vector<int>& prog_ops = prog_desc.prog_ops;
  const std::vector<int>& prog_args = prog_desc.prog_args;
  const std::vector<int>& prog_lens = prog_desc.prog_lens;
  const std::vector<int>& freq_prog_ops = prog_desc.freq_prog_ops;
  const std::vector<int>& freq_prog_args = prog_desc.freq_prog_args;
  Impl* P = p_.get();
  const int n_in = static_cast<int>(in_blocks.size());
  const int n_const = static_cast<int>(consts.size());
  const int n_terms = static_cast<int>(out_blocks.size());
  // --- shape validation (before any step, EXPLICIT errors) ------------------------------------
  if (n_terms == 0)
    throw std::runtime_error("System::add_coupled_source : no source term (out_blocks empty)");
  if (static_cast<int>(in_roles.size()) != n_in)
    throw std::runtime_error(
        "System::add_coupled_source : in_blocks / in_roles of different sizes");
  if (static_cast<int>(out_roles.size()) != n_terms ||
      static_cast<int>(prog_lens.size()) != n_terms)
    throw std::runtime_error(
        "System::add_coupled_source : out_blocks / out_roles / prog_lens of different "
        "sizes");
  if (prog_ops.size() != prog_args.size())
    throw std::runtime_error(
        "System::add_coupled_source : prog_ops / prog_args of different sizes");
  if (n_in + n_const > kCsMaxReg)
    throw std::runtime_error(
        "System::add_coupled_source : too many registers (inputs + constants > " +
        std::to_string(kCsMaxReg) + ")");
  if (n_terms > kCsMaxTerms)
    throw std::runtime_error("System::add_coupled_source : too many source terms (> " +
                             std::to_string(kCsMaxTerms) + ")");
  // Resolves role -> component via the CONSERVATIVE descriptor of the block. The role is addressed BY
  // NAME: a canonical role name OR a user-defined role label (index_of(string), ADC-292). An unknown
  // block raises via P->index().
  auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
    const int sidx = P->index(block);  // raises if unknown block
    const VariableSet& vs = P->sp[static_cast<std::size_t>(sidx)].cons_vars;
    // STRICT (no silent fallback): a DSL coupled source targets a (block, role) EXPLICITLY requested
    // by the user. If the block does NOT expose this role (neither a canonical role nor a declared
    // user-role label), it is an error: a fallback on component 0 would apply the source to the wrong
    // field SILENTLY. We raise, listing what the block actually exposes.
    const int comp = vs.index_of(role);
    if (comp < 0)
      throw std::runtime_error(
          "System::add_coupled_source : block '" + block + "' does not expose role '" + role +
          "' (roles: " + (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
          ", no silent fallback on component 0)");
    return {sidx, comp};
  };
  // Inputs: (species, component) read per cell. Captured by INDEX (the fabs may be
  // reallocated between registration and application: we rebuild the Array4 at EACH step).
  struct InRef {
    int sidx, comp;
  };
  std::vector<InRef> ins(static_cast<std::size_t>(n_in));
  for (int c = 0; c < n_in; ++c) {
    auto [s, comp] =
        resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
    ins[static_cast<std::size_t>(c)] = {s, comp};
  }
  struct OutRef {
    int sidx, comp;
    CsProgram prog;
  };
  std::vector<OutRef> outs(static_cast<std::size_t>(n_terms));
  int off = 0;
  for (int t = 0; t < n_terms; ++t) {
    auto [s, comp] =
        resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
    const int len = prog_lens[static_cast<std::size_t>(t)];
    if (len < 0 || len > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : program of term " + std::to_string(t) +
                               " too long (> " + std::to_string(kCsMaxProg) + ")");
    if (off + len > static_cast<int>(prog_ops.size()))
      throw std::runtime_error("System::add_coupled_source : prog_lens inconsistent with prog_ops");
    CsProgram pg;
    pg.len = len;
    for (int k = 0; k < len; ++k) {
      const int opc = prog_ops[static_cast<std::size_t>(off + k)];
      const int a = prog_args[static_cast<std::size_t>(off + k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : invalid opcode");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "System::add_coupled_source : register out of bounds in the program");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    validate_cs_program_stack(pg, "System::add_coupled_source term " + std::to_string(t));
    outs[static_cast<std::size_t>(t)] = {s, comp, pg};
    off += len;
  }
  // All touched species (inputs + outputs) share the System DistributionMapping (one box
  // round-robin distributed), so same local_size() and same local indexing -> we would iterate in parallel
  // over the local fabs. Conversion to CAPTURED values (no reference to the C++ lambda's 'this').
  std::vector<Real> kconsts(consts.begin(), consts.end());
  // Optional PER-CELL frequency (CoupledSource.frequency with an Expr, refinement of the
  // CONSTANT frequency): a bytecode program mu(U) on the SAME register table as the terms
  // (inputs then constants). Validates HERE its SHAPE (opcodes / bounded registers) BEFORE any push -- the
  // bound must be registered only after a complete validation (anti-phantom-bound rule). Empty
  // (default) -> no per-cell frequency (historical path).
  const bool has_freq_expr = !freq_prog_ops.empty() || !freq_prog_args.empty();
  CsProgram freq_pg;
  if (has_freq_expr) {
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error(
          "System::add_coupled_source : freq_prog_ops / freq_prog_args of different "
          "sizes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : frequency program too long (> " +
                               std::to_string(kCsMaxProg) + ")");
    freq_pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < freq_pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : invalid opcode in the frequency");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "System::add_coupled_source : register out of bounds in the frequency");
      freq_pg.op[k] = opc;
      freq_pg.arg[k] = a;
    }
    validate_cs_program_stack(freq_pg, "System::add_coupled_source frequency");
  }
  // CONSTANT declared frequency of the coupling (audit wave 3): registered for the step bound of
  // step_cfl / step_adaptive (dt <= cfl/mu on the MACRO-step). <= 0 = no bound (historical). Pushed
  // AFTER all the validation (source AND frequency have raised if invalid): a rejected coupling must
  // leave NO phantom bound -- otherwise a script that try/excepts the failure would keep a throttled step without
  // matching physics.
  if (frequency > 0.0)
    P->coupled_freqs_.push_back(Impl::CoupledFreq{label, frequency});
  // PER-CELL frequency: same rule (push after complete validation). The inputs REUSE the
  // resolve() resolution (ins); the constants are the same as the source (kconsts). The program
  // mu(U) is reduced (MAX) at each step in step_cfl / step_adaptive.
  if (has_freq_expr) {
    Impl::CoupledFreqExpr ce;
    ce.label = label;
    ce.prog = freq_pg;
    ce.n_in = n_in;
    ce.ins.resize(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c)
      ce.ins[static_cast<std::size_t>(c)] = {ins[static_cast<std::size_t>(c)].sidx,
                                             ins[static_cast<std::size_t>(c)].comp};
    ce.kconsts = kconsts;
    P->coupled_freq_exprs_.push_back(std::move(ce));
  }
  P->couplings.push_back([P, ins, outs, kconsts, n_in, n_const, n_terms](Real dt) {
    // MPI-safe: iteration over the LOCAL fabs of the first input block (or output if no
    // input). local_size()==0 on a rank without a box -> empty loop, no-op (no hard-coded fab(0)).
    const int sref = n_in > 0 ? ins[0].sidx : outs[0].sidx;
    MultiFab& Uref = P->sp[static_cast<std::size_t>(sref)].U;
    for (int li = 0; li < Uref.local_size(); ++li) {
      CoupledSourceKernel kern;
      kern.dt = dt;
      kern.n_in = n_in;
      kern.n_const = n_const;
      kern.n_terms = n_terms;
      for (int c = 0; c < n_in; ++c) {
        kern.in[c] = P->sp[static_cast<std::size_t>(ins[static_cast<std::size_t>(c)].sidx)]
                         .U.fab(li)
                         .array();
        kern.in_comp[c] = ins[static_cast<std::size_t>(c)].comp;
      }
      for (int c = 0; c < n_const; ++c)
        kern.consts[c] = kconsts[static_cast<std::size_t>(c)];
      for (int t = 0; t < n_terms; ++t) {
        kern.out[t] = P->sp[static_cast<std::size_t>(outs[static_cast<std::size_t>(t)].sidx)]
                          .U.fab(li)
                          .array();
        kern.out_comp[t] = outs[static_cast<std::size_t>(t)].comp;
        kern.prog[t] = outs[static_cast<std::size_t>(t)].prog;
      }
      for_each_cell(Uref.box(li), kern);  // NAMED functor (device-clean), additive forward-Euler
    }
  });
  // Inspect metadata (ADC-595): a raw add_coupled_source declares NO conservation contract, so it
  // registers an "unchecked" view (empty ConservationContract) carrying the label and the frequency
  // bound. add_coupling_operator overwrites this behavior by pushing the DECLARED contract instead.
  CouplingOperatorView view;
  view.label = label;
  view.frequency.constant_mu = frequency;
  view.frequency.per_cell = has_freq_expr;
  P->coupling_.coupled_operators.push_back(std::move(view));
}

void System::add_coupling_operator(const CouplingOperator& op) {
  // Validate the DECLARED conservation contract against the actual output terms BEFORE anything is
  // stored (host, fail-loud): a coupling that declares a role conserved whose terms do not cancel
  // raises here and leaves no partial state (anti-phantom-registration, like add_coupled_source's
  // frequency-bound rule). An unchecked (empty) contract is a no-op check.
  validate_coupling_contract(op, "System::add_coupling_operator");
  // Lower through the SAME flat path (bit-identical numerics); it pushes an "unchecked" inspect view
  // at its tail. We then replace that view's contract with the DECLARED one so coupled_operators()
  // reports the typed contract rather than "unchecked".
  add_coupled_source(op.program, op.frequency.constant_mu, op.label);
  p_->coupling_.coupled_operators.back().conservation = op.conservation;
}

const std::vector<CouplingOperatorView>& System::coupled_operators() const {
  return p_->coupling_.coupled_operators;
}

void System::set_source_stage(const std::string& name, const std::string& kind, double theta,
                              double alpha, const SourceStageOptions& opts) {
  require_assembling(p_->lifecycle_, "set_source_stage");  // frozen once pops.bind completes (ADC-592)
  // Settings grouped into a POD (ADC-214): local aliases to keep the body readable (the names and the
  // semantics are strictly those of the old flat parameters).
  const double krylov_tol = opts.krylov_tol;
  const int krylov_max_iters = opts.krylov_max_iters;
  const std::string& density = opts.density;
  const std::string& momentum_x = opts.momentum_x;
  const std::string& momentum_y = opts.momentum_y;
  const std::string& energy = opts.energy;
  const int bz_aux_component = opts.bz_aux_component;
  Impl* P = p_.get();
  Impl::Species& s = P->find(name);  // raises if unknown block
  // ONLY kind wired for now: ElectrostaticLorentzCondensation (cf. CondensedSchurSourceStepper).
  // Other kinds may be added without touching the facade (explicit rejection, no silent ignore).
  if (kind != "electrostatic_lorentz")
    throw std::runtime_error("System::set_source_stage : kind '" + kind +
                             "' unknown (only 'electrostatic_lorentz' is supported)");
  if (!(theta > 0.0 && theta <= 1.0))
    throw std::runtime_error("System::set_source_stage : theta must be in (0, 1] (received " +
                             std::to_string(theta) + ")");
  // Tolerance / budget of the stage Krylov solve (audit 2026-06: the constants 1e-10 / 400 (cart)
  // / 600 (polar) are no longer frozen). krylov_tol <= 0 / krylov_max_iters <= 0 = "historical
  // stepper default" (we do not touch the setting of the constructed stepper).
  if (krylov_tol > 0.0 && !(krylov_tol < 1.0))
    throw std::runtime_error("System::set_source_stage : krylov_tol must be in (0, 1)");
  // GEOMETRY: the condensed source stage is wired in CARTESIAN (CondensedSchurSourceStepper, #126) AND in
  // POLAR (PolarCondensedSchurSourceStepper, #212, Path A step 2c). The dispatch below builds the
  // stepper adapted to the System geometry. Any other geometry is REJECTED explicitly (no
  // silent ignore).
  const bool polar = (P->cfg.geometry == "polar");
  if (P->cfg.geometry != "cartesian" && !polar)
    throw std::runtime_error(
        "System::set_source_stage : condensed source stage supports the "
        "cartesian and polar geometries (received '" +
        P->cfg.geometry + "')");
  // The POLAR condensed source stage is now MULTI-RANK MPI (PolarTensorKrylovSolver / polar
  // Schur distributed by AZIMUTHAL split; check_radial_columns layout guard in the
  // solver). On the FACADE side, the System builds for now ONE box covering the ring (P->ba mono-box),
  // so under MPI the box lives on rank 0 and the other ranks have local_size()==0: the solve stays CORRECT
  // (collective dot/project_mean called on all ranks, zero contributions from the empty ranks) and
  // BIT-IDENTICAL to the mono-rank, but without real parallelism at this level. The effective theta split
  // (true multi-rank scaling) takes place at the C++ API level (PolarCondensedSchurSourceStepper with a
  // BoxArray split in theta); the facade-side theta distribution is deferred (Extend). No mono-rank
  // guard here: the PolarTensorKrylovSolver raises a clear error if the layout ever cuts r.
  // ROLE CONTRACT: the block must expose Density / MomentumX / MomentumY (Energy optional). We read the
  // CONSERVATIVE descriptor of the block (populated by add_block / the .so with roles, including the compiled DSL which
  // declares the electrons with roles). A required role absent raises an EXPLICIT error HERE (before the step)
  // -- the stepper constructor would raise it too, but we diagnose on the named-block side.
  const VariableSet& vs = s.cons_vars;
  // DESCRIPTOR RESOLUTION (audit wave 2: roles/transported fields in the ABI). An
  // EMPTY descriptor = canonical role (historical, bit-identical). Otherwise: stable ROLE name
  // first (role_from_name), then block VARIABLE name. Failure = explicit error with remedy.
  auto resolve_field = [&](const std::string& spec, VariableRole canonical,
                           const char* label) -> int {
    if (spec.empty()) {
      const int idx = vs.index_of(canonical);
      if (idx < 0)
        throw std::runtime_error(
            "System::set_source_stage : block '" + name + "' does not expose the role " + label +
            " required by pops.CondensedSchur (the model must declare Density / MomentumX / "
            "MomentumY ; Energy optional), and no explicit descriptor is provided (pass "
            "density=/momentum=... with a role name or a block variable name).");
      return idx;
    }
    const VariableRole r = role_from_name(spec);
    if (r != VariableRole::Custom) {
      const int idx = vs.index_of(r);
      if (idx < 0)
        throw std::runtime_error("System::set_source_stage : block '" + name +
                                 "' does not expose role '" + spec + "' (" + label + ")");
      return idx;
    }
    for (std::size_t i = 0; i < vs.names.size(); ++i)
      if (vs.names[i] == spec)
        return static_cast<int>(i);
    throw std::runtime_error("System::set_source_stage : '" + spec +
                             "' is neither a stable role nor a variable of block '" + name + "' (" +
                             label + ")");
  };
  const int c_rho = resolve_field(density, VariableRole::Density, "Density");
  const int c_mx = resolve_field(momentum_x, VariableRole::MomentumX, "MomentumX");
  const int c_my = resolve_field(momentum_y, VariableRole::MomentumY, "MomentumY");
  const int c_E = (energy == "none")
                      ? -1
                      : (energy.empty() ? vs.index_of(VariableRole::Energy)
                                        : resolve_field(energy, VariableRole::Energy, "Energy"));
  // B_z MANDATORY: the Lorentz stage reads Omega = B_z. We require set_magnetic_field called
  // (bz_field_ provided) and we widen the aux channel to the B_z channel (kAuxBaseComps) so that apply_bz
  // populates it and solve_fields fills its ghosts. An absent B_z raises an EXPLICIT error.
  if (P->fields_.bz_field_.empty())
    throw std::runtime_error("System::set_source_stage : block '" + name +
                             "' has no B_z field (aux Omega) ; "
                             "pops.CondensedSchur requires set_magnetic_field(B_z) (the Lorentz "
                             "term reads Omega = B_z).");
  // Aux channel of the magnetic field: canonical (kAuxBaseComps) by default, redirectable by
  // bz_aux_component (transported descriptor). NOTE: apply_bz populates the CANONICAL channel; a
  // different component assumes the caller populates it itself (derived/custom aux field).
  const int c_bz = bz_aux_component >= 0 ? bz_aux_component : kAuxBaseComps;
  P->ensure_aux_width(c_bz + 1);  // guarantees the channel in the shared aux + re-applies B_z
  const double effective_krylov_tol =
      krylov_tol > 0.0 ? krylov_tol : static_cast<double>(kKrylovDefaultRelTol);
  const int effective_krylov_max_iters =
      krylov_max_iters > 0
          ? krylov_max_iters
          : (polar ? kSchurKrylovPolarMaxIters : kSchurKrylovCartesianMaxIters);
  // Builds the condensed source stage on the REAL System layout (ba/dm/geom) with the Poisson BC.
  // The stepper allocates its buffers ONCE; step() reuses them (cf. its lifecycle). alpha =
  // electrostatic coupling constant of the source subsystem.
  if (polar) {
    // POLAR (Path A step 2c): PolarCondensedSchurSourceStepper on the ring pgeom_, SAME Poisson BC
    // (radial Dirichlet/Neumann, theta always periodic on the solver side). RadialLine preconditioner
    // (default). run_source_stage invokes it exactly like the Cartesian (identical step() signature).
    // schur stays nullptr (Cartesian path untouched). EXPLICIT components resolved above
    // (empty descriptors -> canonical roles -> bit-identical): the POLAR stepper accepts the
    // overrides since wave 3 (ctor with explicit components, Cartesian parity).
    s.schur_polar = std::make_shared<PolarCondensedSchurSourceStepper>(
        vs, c_rho, c_mx, c_my, c_E, P->pgeom_, P->ba, P->fields_.poisson_bc(),
        static_cast<Real>(alpha));
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur_polar->set_krylov(static_cast<Real>(effective_krylov_tol),
                                effective_krylov_max_iters);
  } else {
    // CARTESIAN (#126): EXPLICIT components resolved above (empty descriptors -> canonical
    // roles -> same indices as the historical, bit-identical).
    s.schur = std::make_shared<CondensedSchurSourceStepper>(vs, c_rho, c_mx, c_my, c_E, P->geom,
                                                            P->ba, P->fields_.poisson_bc(),
                                                            static_cast<Real>(alpha));
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur->set_krylov(static_cast<Real>(effective_krylov_tol), effective_krylov_max_iters);
  }
  s.schur_bz_comp = c_bz;
  s.schur_theta = theta;
  EffectiveSourceStageOptions stage;
  stage.block = name;
  stage.kind = kind;
  stage.geometry = polar ? "polar" : "cartesian";
  stage.theta = theta;
  stage.alpha = alpha;
  stage.requested_krylov_tol = krylov_tol;
  stage.requested_krylov_max_iters = krylov_max_iters;
  stage.effective_krylov_tol = effective_krylov_tol;
  stage.effective_krylov_max_iters = effective_krylov_max_iters;
  stage.density = density;
  stage.momentum_x = momentum_x;
  stage.momentum_y = momentum_y;
  stage.energy = energy;
  stage.bz_aux_component = c_bz;
  P->diagnostics_.source_stage_options[name] = std::move(stage);
}

void System::set_time_scheme(const std::string& scheme) {
  require_assembling(p_->lifecycle_, "set_time_scheme");  // frozen once pops.bind completes (ADC-592)
  // Routes the splitting policy of the system stepper (default Lie = bit-identical). The Strang
  // scheme reuses the SAME bricks (s.advance for the transport half-advances, run_source_stage
  // for the full source stage); it RE-SOLVES solve_fields between the stages (cf. SystemStepper::step_strang
  // and docs/HOFFART_STEP_SEQUENCE.md). An unknown scheme raises an EXPLICIT error (no silent ignore).
  if (scheme == "lie") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Lie);
    p_->time_scheme_ = "lie";
  } else if (scheme == "strang") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Strang);
    p_->time_scheme_ = "strang";
  } else {
    throw std::runtime_error("System::set_time_scheme : scheme '" + scheme +
                             "' unknown (expected 'lie' or 'strang')");
  }
}

void System::set_gauss_policy(const std::string& policy) {
  require_assembling(p_->lifecycle_, "set_gauss_policy");  // frozen once pops.bind completes (ADC-592)
  // Gauss's law policy (project R0, Hoffart reproduction). "restart" (default): solve_fields
  // re-solves -Delta phi = f at each step (bit-identical to the historical). "evolve": after the first
  // solve (phi^0), solve_fields NO LONGER re-solves the Poisson; it derives the aux from the CURRENT phi that
  // the Schur source stage evolves in-place in ell_phi() -> -Delta phi evolution without restart of the
  // paper (the Gauss constraint is imposed only at t=0). Has effect ONLY with a condensed source stage
  // (without it phi would stay frozen after t=0). The gauss_solved_once_ lock is reset to zero here so
  // that a policy change BEFORE the first solve stays consistent (the 1st solve always solves).
  if (policy == "restart") {
    p_->fields_.gauss_evolve_ = false;
    p_->gauss_policy_ = "restart";
  } else if (policy == "evolve") {
    p_->fields_.gauss_evolve_ = true;
    p_->gauss_policy_ = "evolve";
  } else {
    throw std::runtime_error("System::set_gauss_policy : policy '" + policy +
                             "' unknown (expected 'restart' or 'evolve')");
  }
  p_->fields_.gauss_solved_once_ = false;
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const Real gm1 = Real(s.gamma) - Real(1);
  // Local helper: sets density + rest state on ONE cell (same formulas as the historical).
  auto set_cell = [&](Array4& u, int i, int j, Real r) {
    u(i, j, 0) = r;
    if (s.ncomp >= 3) {
      u(i, j, 1) = 0;
      u(i, j, 2) = 0;
    }  // momentum at rest
    if (s.ncomp == 4)
      u(i, j, 3) = r / gm1;  // E = p/(g-1), p = rho
  };
  // MULTI-BOX (theta_boxes > 1, polar): @p rho is the GLOBAL field (nr x ntheta, layout flat[j*gnx+i]
  // identical to the mono-box below). We write each local box at its GLOBAL indices. local_size() <= 1
  // (Cartesian / polar mono-box, including MPI mono-box): historical path UNCHANGED, bit-identical.
  if (s.U.local_size() > 1) {
    const int gnx = p_->dom.nx(), gny = p_->dom.ny();
    if (static_cast<int>(rho.size()) != gnx * gny)
      throw std::runtime_error("System::set_density : size != nr*ntheta (multi-box theta)");
    for (int li = 0; li < s.U.local_size(); ++li) {
      Array4 u = s.U.fab(li).array();
      const Box2D b = s.U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          set_cell(u, i, j, rho[static_cast<std::size_t>(j) * gnx + i]);
    }
    return;
  }
  // Row-major layout of the input array: (ni x nj) = extents of the state box. In Cartesian
  // ni = nj = cfg.n (indexing and size bit-identical to before). In polar ni = nr, nj = ntheta:
  // we index by the real extents of the box (and not n*n), so nr != ntheta is correctly handled.
  const Box2D v = s.U.box(0);
  const int ni = v.nx(), nj = v.ny();
  if (static_cast<int>(rho.size()) != ni * nj)
    throw std::runtime_error("System::set_density : size != nr*ntheta (or n*n in Cartesian)");
  Array4 u = s.U.fab(0).array();
  // LAYOUT CONVENTION (unchanged vs the historical): slow axis = 2nd box index (j), fast axis =
  // 1st (i), i.e. flat[(j-lo) * ni + (i-lo)]. In Cartesian ni = n, lo = 0 -> flat[j*n+i] (bit-identical
  // to before). In polar the array is thus (nr, ntheta) radial-line-by-line: j = theta (slow
  // axis), i = r (fast axis), SAME order as density()/copy_comp0 -> consistent.
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      set_cell(u, i, j, rho[static_cast<std::size_t>(j - v.lo[1]) * ni + (i - v.lo[0])]);
}

POPS_EXPORT void System::set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                             CellConvert cons_to_prim) {
  Impl::Species& s = p_->find(name);
  s.prim_to_cons = std::move(prim_to_cons);
  s.cons_to_prim = std::move(cons_to_prim);
}

void System::set_primitive_state(const std::string& name, const std::vector<double>& prim) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn =
      static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (prim.size() != static_cast<std::size_t>(nc) * nn)
    throw std::runtime_error(
        "System::set_primitive_state : size != ncomp*nr*ntheta (n*n Cartesian) (block '" + name +
        "' has " + std::to_string(nc) + " variables)");
  if (!s.prim_to_cons)
    throw std::runtime_error(
        "System::set_primitive_state : the model of block '" + name +
        "' does not expose a primitive -> conservative conversion (.so generated before "
        "this project ?) ; use set_state (direct conservative state)");
  // CELL-BY-CELL conversion via the block model: we read the nc primitives component-major
  // (prim[c*nn + k]) into a small contiguous buffer, convert, and write the conservatives at the
  // same place in an output buffer. Then write_state pushes everything to the MultiFab (set_state
  // path, identical marshaling). Reuses therefore the existing marshaling (copy/write_state).
  std::vector<double> cons(prim.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c)
      cell_in[c] = prim[static_cast<std::size_t>(c) * nn + k];
    s.prim_to_cons(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c)
      cons[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  p_->write_state(s.U, nc, cons);
}

std::vector<double> System::get_primitive_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn =
      static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (!s.cons_to_prim)
    throw std::runtime_error(
        "System::get_primitive_state : the model of block '" + name +
        "' does not expose a conservative -> primitive conversion (.so generated before "
        "this project ?) ; use get_state (direct conservative state)");
  const std::vector<double> cons = p_->copy_state(s.U, nc);  // get_state path (same marshaling)
  std::vector<double> prim(cons.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c)
      cell_in[c] = cons[static_cast<std::size_t>(c) * nn + k];
    s.cons_to_prim(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c)
      prim[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  return prim;
}

void System::solve_fields() {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "field_solve");
  p_->solve_fields();
  // ELLIPTIC-SOLVER NATIVE COUNTERS (Spec 5 sec.13.11.1, ADC-479 criteria 42/43). The opaque
  // "field_solve" scope hides where the elliptic solve (96-99.9% of step cost) spends its time: read
  // the active solver's per-solve stats back HERE -- after p_->solve_fields() returns, so AFTER its
  // internal device_fence() (system_field_solver.hpp CRITICAL invariant: the V-cycle must be done
  // before phi is read), preserving the device-fence ordering. Cheap int/double reads, all guarded
  // by enabled() -> ZERO cost when profiling is off (count/record are no-ops too, but the accessor
  // reads are skipped entirely).
  if (p_->program_.profiler_.enabled()) {
    // mg_cycles / krylov_iters ACCUMULATE (total elliptic iteration work over the run); elliptic_bottom
    // records the coarsest-grid self-time as a timing sample. mg_levels is a STRUCTURAL CONSTANT (the
    // hierarchy depth), so count_max (peak) reports the actual level count instead of summing it per
    // step (same idiom as scratch_peak_bytes). All four are honest 0 for a direct FFT solver.
    p_->program_.profiler_.count("mg_cycles", p_->fields_.last_mg_cycles());
    p_->program_.profiler_.count("krylov_iters", p_->fields_.last_krylov_iters());
    p_->program_.profiler_.count_max("mg_levels", p_->fields_.last_num_levels());
    p_->program_.profiler_.record("elliptic_bottom", p_->fields_.last_bottom_seconds());
  }
}

// --- profiling (ADC-459) -------------------------------------------------------------------------
// enable_profiling / profile_report drive the System-owned Profiler. Today the System wraps its
// coarse phases (step, field_solve); the per-Program-node / per-native-brick granularity is wired
// through the compiled-program ProgramContext as a follow-up.
void System::enable_profiling() {
  p_->program_.profiler_.enable();
}
void System::disable_profiling() {
  p_->program_.profiler_.disable();
}
bool System::is_profiling() const {
  return p_->program_.profiler_.enabled();
}
void System::reset_profiling() {
  p_->program_.profiler_.reset();
}
std::string System::profile_report() const {
  return p_->program_.profiler_.report();
}
std::vector<RuntimeDiagnosticEvent> System::solver_diagnostics() const {
  return p_->fields_.combined_diagnostics_report().events;
}
// The System-owned Profiler reference (ADC-459): the compiled-program ProgramContext::profile_node
// times each Program node into it, so per-node scopes accumulate in the SAME table as the coarse
// step / field_solve phases. POPS_EXPORT: resolved by a generated problem.so across the dlopen boundary.
POPS_EXPORT pops::runtime::program::Profiler& System::profiler() {
  return p_->program_.profiler_;
}

void System::solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
  p_->solve_fields_from_state(block_idx, U_stage);
}

// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): forwards to the field solver, which
// assembles the system Poisson RHS as Sum_s elliptic_rhs_s(U_s) reading EVERY block's stage state at
// once (U_stages indexed by block index; nullptr -> the block's live state), then re-fills the shared
// aux. POPS_EXPORT: resolved by a generated problem.so (ProgramContext) across the dlopen boundary.
POPS_EXPORT void System::solve_fields_from_blocks(const std::vector<const MultiFab*>& U_stages) {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "field_solve");
  p_->solve_fields_from_blocks(U_stages);
  // Same elliptic-solver counters as System::solve_fields (ADC-479 criteria 42/43), read back AFTER
  // the coupled solve returns -- i.e. after its internal device_fence() (system_field_solver.hpp). The
  // coupled multi-block solve uses the SAME ell_ solver, so the stats are populated identically.
  if (p_->program_.profiler_.enabled()) {
    p_->program_.profiler_.count("mg_cycles", p_->fields_.last_mg_cycles());
    p_->program_.profiler_.count("krylov_iters", p_->fields_.last_krylov_iters());
    p_->program_.profiler_.count_max("mg_levels", p_->fields_.last_num_levels());
    p_->program_.profiler_.record("elliptic_bottom", p_->fields_.last_bottom_seconds());
  }
}

// NAMED multi-elliptic field (ADC-428): a SECOND elliptic solve for @p field from block @p block_idx's
// stage state. Forwards to the field solver, which assembles the per-field RHS (sum of the blocks'
// named bricks), solves with a dedicated native solver, and writes the field's OWN aux components.
POPS_EXPORT void System::solve_fields_from_state(const std::string& field, int block_idx,
                                                const MultiFab& U_stage) {
  p_->solve_named_field_from_state(field, block_idx, U_stage);
}

// Register a named elliptic field (ADC-428): records WHERE the field's solved phi / centered grad land
// in the aux channel (@p phi_comp / @p gx_comp / @p gy_comp, the model's named aux slots). The native
// loader calls this for each m.elliptic_field after the block is installed. POPS_EXPORT: resolved by the
// generated problem.so / native loader across the dlopen boundary.
POPS_EXPORT void System::register_elliptic_field(const std::string& field, int phi_comp, int gx_comp,
                                                int gy_comp) {
  p_->register_elliptic_field(field, phi_comp, gx_comp, gy_comp);
}

// Attach a named elliptic-field RHS closure to block @p block_name (ADC-428): the per-field Poisson
// right-hand side brick += elliptic_field_rhs(U). The native loader builds it (make_poisson_rhs of the
// named brick) and attaches it here; solve_fields_from_state(field, ...) then sums it over the blocks.
// @throws if the block is unknown. POPS_EXPORT: resolved across the dlopen boundary.
POPS_EXPORT void System::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab&, MultiFab&)> rhs) {
  p_->blocks_.find(block_name).named_poisson_rhs[field] = std::move(rhs);
}

// Time advance EXTRACTED into stepper_ (SystemStepper, Batch B). Pure delegation: the Cartesian/polar
// dispatch of the physical step h, the per-block CFL formula (substeps/stride), the
// hold-then-catch-up semantics of the macro-step counter, the condensed source stage and the couplings live
// now in the header (bit-identical). The public API stays unchanged.
void System::step(double dt) {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "step");
  p_->program_.profiler_.count("steps");
  p_->stepper_.step(dt);
}
void System::advance(double dt, int nsteps) {
  p_->stepper_.advance(dt, nsteps);
}
double System::step_cfl(double cfl) {
  return p_->stepper_.step_cfl(cfl);
}
double System::step_adaptive(double cfl) {
  return p_->stepper_.step_adaptive(cfl);
}

// System clock (IO v1, audit wave 2): macro_step is REQUIRED by the restart (the
// hold-then-catch-up stride cadence reads macro_step % stride; t alone is not enough).
int System::macro_step() const {
  return p_->macro_step_;
}

// Potential phi restoration (IO v1, restart): writes the VALID cells of component 0 of the
// solver phi (multigrid warm start; physical state in gauss_policy="evolve"). Mono-box
// (same marshaling convention as potential / set_density).
void System::set_potential(const std::vector<double>& phi) {
  Impl* P = p_.get();
  device_fence();
  if (P->polar_) {
    P->fields_.ensure_elliptic_polar();
    MultiFab& ph = P->fields_.pell_->phi();
    // Rank without a box (MPI mono-box): NO-OP (the owning rank restores phi). Allows restart on
    // all ranks with the GLOBAL field. Mono-rank: local_size()==1, UNCHANGED.
    if (ph.local_size() == 0)
      return;
    const Box2D v = ph.box(0);
    if (static_cast<int>(phi.size()) != v.nx() * v.ny())
      throw std::runtime_error("System::set_potential : size != nr*ntheta");
    Array4 a = ph.fab(0).array();
    std::size_t k = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        a(i, j, 0) = phi[k++];
    return;
  }
  P->fields_.ensure_elliptic();
  MultiFab& ph = P->fields_.ell_phi();
  if (ph.local_size() == 0)
    return;  // rank without a box: no-op (cf. polar branch)
  const Box2D v = ph.box(0);
  if (static_cast<int>(phi.size()) != v.nx() * v.ny())
    throw std::runtime_error("System::set_potential : size != n*n");
  Array4 a = ph.fab(0).array();
  std::size_t k = 0;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      a(i, j, 0) = phi[k++];
}
void System::set_clock(double t, int macro_step) {
  if (macro_step < 0)
    throw std::runtime_error("System::set_clock : macro_step >= 0 (restart)");
  p_->t = t;
  p_->macro_step_ = macro_step;
}

std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);
  return p_->copy_state(R, s.ncomp);
}

// Compiled time-program seam (epic ADC-399 / ADC-401): a generated problem.so installs its macro-step
// body and reaches per-block storage through these accessors (Impl is private to this TU).
void System::install_program_step(std::function<void(double)> step) {
  p_->program_.step_ = std::move(step);
}
// Compiled-Program macro-step cadence (ADC-411): SYSTEM-level substeps + stride around the installed
// program closure (cf. SystemStepper::step). Kept separate from install_program so the .so ABI is
// untouched. Validates substeps >= 1 && stride >= 1 (fail-loud: a non-positive cadence is meaningless).
void System::set_program_cadence(int substeps, int stride) {
  require_assembling(p_->lifecycle_, "set_program_cadence");  // frozen once pops.bind completes (ADC-592)
  // Program subsystem owns the cadence validation + storage (ADC-594): the guard message names
  // "System::set_program_cadence" verbatim (unchanged wording), keeping the pinned error intact.
  p_->program_.set_cadence(substeps, stride, "System");
}
// Read the installed GLOBAL cadence (ADC-594): the tiny const getters the ProgramRuntimeReport reads
// through the bindings (there was no Python-visible getter before). Default 1/1 with no program.
int System::program_substeps() const {
  return p_->program_.substeps_;
}
int System::program_stride() const {
  return p_->program_.stride_;
}
int System::n_blocks() const {
  return static_cast<int>(p_->sp.size());
}
MultiFab& System::block_state(int b) {
  return p_->sp[static_cast<std::size_t>(b)].U;
}
void System::block_rhs_into(int b, MultiFab& U, MultiFab& R) {
  p_->sp[static_cast<std::size_t>(b)].rhs_into(U, R);
}
// FLUX-ONLY residual R <- -div F(U) (ADC-425): the block's SourceFreeModel<Model> rhs path (built in
// build_block), bit-identical to rhs_into minus the default source. Fails loud on a block that did not
// build it (the host .so prototype loader) instead of silently leaking the source.
void System::block_neg_div_flux_into(int b, MultiFab& U, MultiFab& R) {
  Impl::Species& s = p_->sp[static_cast<std::size_t>(b)];
  if (!s.rhs_flux_only)
    throw std::runtime_error(
        "System::block_neg_div_flux_into: block '" + s.name +
        "' has no flux-only residual closure (the host .so prototype loader does not build one); a "
        "flux-only RHS (P.rhs(flux=True, sources without 'default')) needs a native block "
        "(add_block / production-backend compiled block)");
  s.rhs_flux_only(U, R);
}
// SOURCE-ONLY residual R <- S(U, aux) (ADC-430): the block's SourceInto<Model> path (built in
// build_block), the exact mirror of block_neg_div_flux_into and bit-identical to the source half of
// rhs_into. Fails loud on a block that did not build it (the host .so prototype loader) instead of
// silently leaking the flux.
void System::block_source_into(int b, MultiFab& U, MultiFab& R) {
  Impl::Species& s = p_->sp[static_cast<std::size_t>(b)];
  if (!s.source_only)
    throw std::runtime_error(
        "System::block_source_into: block '" + s.name +
        "' has no source-only residual closure (the host .so prototype loader does not build one); "
        "a "
        "source-only RHS (P.rhs(flux=False, sources with 'default')) needs a native block "
        "(add_block / production-backend compiled block)");
  s.source_only(U, R);
}
// Max |wave speed| of block b on U: the SAME BlockState::max_speed closure step_cfl reads (set at
// add_block time -- HasStabilitySpeed / max_wave_speed of the model). REUSES it, does not recompute.
Real System::block_max_speed(int b, const MultiFab& U) const {
  return p_->sp[static_cast<std::size_t>(b)].max_speed(U);
}
// MIN physical cell size of the grid: Cartesian min(dx, dy) / polar min(dr, r_min*dtheta), the exact
// formula SystemStepper::cfl_grid_h uses for the native CFL (kept consistent so a Program dt bound and
// the native CFL share the same hmin).
Real System::cfl_min_dx() const {
  return p_->polar_ ? std::min(p_->pgeom_.dr(), p_->pgeom_.r_min * p_->pgeom_.dtheta())
                    : std::min(p_->geom.dx(), p_->geom.dy());
}
// Collective scalar reduction over a NAMED block's state -- the native seam the Python diagnostics
// driver (ADC-542) drives to fire a declared typed measure (Norm / Integral / MinMax) each cadence
// tick. Resolves the block by name (Impl::find, insertion order) and folds its U with the pops::
// free functions. Per-component kinds read component @p comp; the full-state "_all" kinds fold over
// EVERY component. Unknown kind -> throw (fail loud, no silent 0). COLLECTIVE like dot.
double System::reduce_component(const std::string& block, const std::string& kind, int comp) const {
  const Impl::Species& s = p_->find(block);
  const MultiFab& u = s.U;
  const int nc = s.ncomp;
  if (kind == "sum")
    return static_cast<double>(pops::reduce_sum(u, comp));
  if (kind == "min")
    return static_cast<double>(pops::reduce_min(u, comp));
  if (kind == "max")
    return static_cast<double>(pops::reduce_max(u, comp));
  if (kind == "abs_sum")
    return static_cast<double>(pops::reduce_abs_sum(u, comp));
  if (kind == "sum_sq")  // L2 squared: dot(u, u, comp); the driver takes sqrt
    return static_cast<double>(pops::dot(u, u, comp));
  if (kind == "abs_max")  // LInf: collective max |u(.,.,comp)|
    return all_reduce_max(static_cast<double>(pops::norm_inf(u, comp)));
  // Full-state (unscoped) folds over ALL components -- host O(ncomp) composition of the native
  // per-component collectives (no field leaves the ranks; only ncomp scalars).
  if (kind == "sum_all") {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c)
      acc += static_cast<double>(pops::reduce_sum(u, c));
    return acc;
  }
  if (kind == "abs_sum_all") {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c)
      acc += static_cast<double>(pops::reduce_abs_sum(u, c));
    return acc;
  }
  if (kind == "sum_sq_all")
    return static_cast<double>(pops::dot_all(u, u));
  if (kind == "abs_max_all") {
    double m = 0.0;
    for (int c = 0; c < nc; ++c)
      m = std::max(m, all_reduce_max(static_cast<double>(pops::norm_inf(u, c))));
    return m;
  }
  throw std::runtime_error(
      "System::reduce_component: unknown reduction kind '" + kind + "' for block '" + block +
      "' (expected one of: sum, min, max, abs_sum, sum_sq, abs_max, "
      "sum_all, abs_sum_all, sum_sq_all, abs_max_all)");
}
MultiFab System::alloc_scalar_field(int n_comp, int n_ghost) {
  // Co-distributed with the block storage (Impl::ba / Impl::dm -- the same (ba, dm) every block U is
  // built with, P->ba/P->dm above), so a matrix-free apply pairs this field with the state/aux by
  // local fab index. Zero-initialized like a fresh block state (install_block sets U to 0).
  MultiFab f(p_->ba, p_->dm, n_comp, n_ghost);
  f.set_val(Real(0));
  return f;
}

// Multistep history seam (ADC-406a): a generated problem.so declares / reads / writes a named history
// field across macro-steps (Adams-Bashforth), reaching the SYSTEM-OWNED ring buffers through these
// accessors. The rings live in Impl::program_.hist_ (the extracted Program subsystem, ADC-594) so a
// later checkpoint slice (ADC-406b) can serialize them without touching the .so ABI.
MultiFab& System::register_history(const std::string& name, int lag) {
  if (lag < 1)
    throw std::runtime_error("System::register_history: lag must be >= 1 (got " +
                             std::to_string(lag) + ") for history '" + name + "'");
  if (p_->sp.empty())
    throw std::runtime_error(
        "System::register_history: no block exists yet; a history is co-distributed with block 0's "
        "state (add the block before installing the program)");
  const int want_depth = lag + 1;
  auto it = p_->program_.hist_.histories.find(name);
  if (it != p_->program_.hist_.histories.end()) {
    // Idempotent re-registration: the ring depth is the MAX lag any caller requests. A read at the
    // declared max lag and the store (which only needs the current slot, register_history(name, 1))
    // can register in EITHER order without conflict -- a smaller request is a no-op (returns the
    // existing current slot), a larger one grows the ring (appending zero-filled deeper slots; the
    // current slot [0] and the already-stored slots are preserved). A program reads each name at one
    // fixed lag, so the depth converges in the first step and never changes again.
    if (want_depth > p_->program_.hist_.depth[name]) {
      const int ncomp = it->second[0].ncomp();
      for (int k = p_->program_.hist_.depth[name]; k < want_depth; ++k) {
        MultiFab slot(p_->ba, p_->dm, ncomp, 1);
        slot.set_val(Real(0));
        it->second.push_back(std::move(slot));
      }
      p_->program_.hist_.depth[name] = want_depth;
    }
    return it->second[0];
  }
  // The ring holds the block's ncomp (so a slot can carry a full RHS / state), co-distributed with the
  // block storage (ba/dm) so a per-cell kernel and the arithmetic pair it with the state by local fab
  // index. One ghost layer like a block state; zero-initialized (the cold-start fill happens on the
  // first store, but a never-stored read still fails loud on the !initialized flag below).
  const int ncomp = p_->sp[0].ncomp;
  std::vector<MultiFab> ring;
  ring.reserve(static_cast<std::size_t>(want_depth));
  for (int k = 0; k < want_depth; ++k) {
    MultiFab slot(p_->ba, p_->dm, ncomp, 1);
    slot.set_val(Real(0));
    ring.push_back(std::move(slot));
  }
  auto& stored = p_->program_.hist_.histories.emplace(name, std::move(ring)).first->second;
  p_->program_.hist_.depth[name] = want_depth;
  p_->program_.hist_.initialized[name] = false;
  return stored[0];
}

MultiFab& System::read_history(const std::string& name, int lag) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::read_history: unknown history '" + name +
                             "' (register it first)");
  if (lag < 0 || lag >= p_->program_.hist_.depth[name])
    throw std::runtime_error("System::read_history: lag=" + std::to_string(lag) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(p_->program_.hist_.depth[name]) + ")");
  if (!p_->program_.hist_.initialized[name])
    throw std::runtime_error("history '" + name + "' with lag=" + std::to_string(lag) +
                             " was requested but not initialized");
  return it->second[static_cast<std::size_t>(lag)];
}

void System::store_history(const std::string& name, const MultiFab& value) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::store_history: unknown history '" + name +
                             "' (register it first)");
  std::vector<MultiFab>& ring = it->second;
  // Copy the valid cells of value into the current slot [0] (identical layout: ring slots and the
  // block state share (ba, dm); lincomb(dst, 1, src, 0, src) is a valid-cell deep copy).
  pops::lincomb(ring[0], Real(1), value, Real(0), value);
  // PER-SLOT dt (ADC-626): tag slot 0 with the dt that produced it (the last dt the stepper handed to
  // program_.step_). slot_dt is co-sized with the ring and rotated alongside it, so a selective-
  // persistence restart re-steps the recomputed slots with the exact dt sequence. Grown lazily here so
  // a program that never uses a checkpoint policy still pays only a small scalar vector.
  std::vector<Real>& dts = p_->program_.hist_.slot_dt[name];
  if (dts.size() != ring.size())
    dts.assign(ring.size(), Real(0));
  dts[0] = p_->program_.last_dt_;
  if (!p_->program_.hist_.initialized[name]) {
    // COLD START (first store): broadcast into every deeper slot so a multistep step 0 reads the same
    // value at every lag (degenerating to a one-step method). Deterministic + machine-precision exact.
    // The dt broadcasts the same way so every cold-start slot carries the step-0 dt.
    for (std::size_t k = 1; k < ring.size(); ++k) {
      pops::lincomb(ring[k], Real(1), value, Real(0), value);
      dts[k] = p_->program_.last_dt_;
    }
    p_->program_.hist_.initialized[name] = true;
  }
}

void System::rotate_histories() {
  // Shift each ring one step at the end of a macro-step (O(1) std::swap chain, buffer recycled into
  // slot [0]); the grid-free ring bookkeeping lives in the extracted Program subsystem (ADC-594).
  p_->program_.hist_.rotate();
}

// Multistep history checkpoint/restart seam (ADC-406b): the System owns the rings, so the checkpoint
// facade (sim.checkpoint / sim.restart) gathers and restores them DIRECTLY -- reusing the SAME global
// gather (gather_global) / scatter (write_state) machinery as the block state, so the round-trip is
// MPI-safe and bit-identical under np>1. No .so checkpoint_extra ABI is needed for the buffers.
std::vector<std::string> System::history_names() const {
  // enumeration lives in the extracted Program subsystem (ADC-594)
  return p_->program_.hist_.names();
}
int System::history_depth(const std::string& name) const {
  auto it = p_->program_.hist_.depth.find(name);
  if (it == p_->program_.hist_.depth.end())
    throw std::runtime_error("System::history_depth: unknown history '" + name + "'");
  return it->second;
}
int System::history_ncomp(const std::string& name) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_ncomp: unknown history '" + name + "'");
  return it->second[0].ncomp();
}
std::vector<double> System::history_global(const std::string& name, int slot) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_global: unknown history '" + name + "'");
  const std::vector<MultiFab>& ring = it->second;
  if (slot < 0 || slot >= static_cast<int>(ring.size()))
    throw std::runtime_error("System::history_global: slot=" + std::to_string(slot) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(ring.size()) + ")");
  device_fence();
  return gather_global(ring[static_cast<std::size_t>(slot)], ring[0].ncomp(), nx(), ny());
}
bool System::history_initialized(const std::string& name) const {
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::history_initialized: unknown history '" + name + "'");
  return it->second;
}
void System::restore_history(const std::string& name, int slot, const std::vector<double>& values) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end()) {
    // The program will re-register the ring on its first post-restart step, but we restore BEFORE that
    // step; register it now (depth = slot + 1, grown as deeper slots arrive) so the values land. Uses
    // the SAME co-distributed (ba, dm, block 0 ncomp) ring as register_history.
    register_history(name, slot >= 1 ? slot : 1);
    it = p_->program_.hist_.histories.find(name);
  }
  std::vector<MultiFab>& ring = it->second;
  if (slot < 0)
    throw std::runtime_error("System::restore_history: slot=" + std::to_string(slot) +
                             " must be >= 0 for history '" + name + "'");
  if (slot >= static_cast<int>(ring.size())) {
    // A deeper slot than currently registered: grow the ring (zero-filled tail) so it fits, matching
    // register_history's idempotent growth.
    const int ncomp = ring[0].ncomp();
    for (int k = static_cast<int>(ring.size()); k <= slot; ++k) {
      MultiFab s(p_->ba, p_->dm, ncomp, 1);
      s.set_val(Real(0));
      ring.push_back(std::move(s));
    }
    p_->program_.hist_.depth[name] = static_cast<int>(ring.size());
  }
  // Scatter the GLOBAL component-major buffer into the slot's fab: reuse the Impl multi-box
  // write_state (the SAME scatter set_state uses), the true inverse of the multi-box gather
  // (gather_global / state_global). It dispatches on the slot's local_size(): the mono-box / MPI
  // mono-box path (owner rank writes its box, others no-op) and, for theta_boxes > 1, the multi-box
  // scatter that places each local band at its global indices -- matching how history_global gathers.
  p_->write_state(ring[static_cast<std::size_t>(slot)], ring[0].ncomp(), values);
}
void System::set_history_initialized(const std::string& name, bool initialized) {
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::set_history_initialized: unknown history '" + name +
                             "' (restore its slots first)");
  it->second = initialized;
}

// Selective history persistence + deterministic ring replay (ADC-626). A history-persistence policy
// (pops.time.Dense / Interval / Revolve) stores only a SUBSET of a ring's slots in a checkpoint; the
// per-slot dt is serialized alongside so the restart can replay the recomputed slots with the exact dt
// sequence (variable-dt histories round-trip bit-for-bit). rebuild_history_slots reconstructs the
// missing slots by re-stepping the installed Program from the nearest older stored slot.
double System::history_slot_dt(const std::string& name, int slot) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_slot_dt: unknown history '" + name + "'");
  if (slot < 0 || slot >= static_cast<int>(it->second.size()))
    throw std::runtime_error("System::history_slot_dt: slot=" + std::to_string(slot) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(it->second.size()) + ")");
  auto dt_it = p_->program_.hist_.slot_dt.find(name);
  if (dt_it == p_->program_.hist_.slot_dt.end() ||
      slot >= static_cast<int>(dt_it->second.size()))
    return 0.0;  // a never-stepped ring: no dt recorded yet (the dense/zero-fill case)
  return static_cast<double>(dt_it->second[static_cast<std::size_t>(slot)]);
}

void System::restore_history_slot_dt(const std::string& name, int slot, double dt) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::restore_history_slot_dt: unknown history '" + name +
                             "' (restore its slots first)");
  if (slot < 0)
    throw std::runtime_error("System::restore_history_slot_dt: slot=" + std::to_string(slot) +
                             " must be >= 0 for history '" + name + "'");
  std::vector<Real>& dts = p_->program_.hist_.slot_dt[name];
  if (slot >= static_cast<int>(dts.size()))
    dts.resize(static_cast<std::size_t>(slot) + 1, Real(0));
  dts[static_cast<std::size_t>(slot)] = static_cast<Real>(dt);
}

int System::rebuild_history_slots(const std::string& name, const std::vector<int>& stored_slots) {
  // Contract (ADC-626): the STORED slots of ring `name` are already restored (restore_history), the
  // per-slot dt is restored (restore_history_slot_dt), and the SAME Program the checkpoint recorded is
  // installed (the program-hash guard upstream ensures this). The ring stores the block-0 state (the
  // keep_history lowering emits store_history(name, U.n)), so a stored slot IS that block's state at
  // that lag. We reconstruct the missing slots by seeding block 0 from the nearest OLDER stored slot
  // and re-stepping the installed Program forward, capturing the intermediate block states.
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::rebuild_history_slots: unknown history '" + name + "'");
  if (!p_->program_.step_)
    throw std::runtime_error(
        "System::rebuild_history_slots: no compiled Program is installed; the ring cannot be replayed "
        "(install_program before restart, or checkpoint the ring with Dense())");
  std::vector<MultiFab>& ring = it->second;
  const int depth = static_cast<int>(ring.size());
  std::vector<int> anchors = stored_slots;
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  if (anchors.empty() || anchors.back() != depth - 1)
    throw std::runtime_error(
        "System::rebuild_history_slots: the oldest slot " + std::to_string(depth - 1) +
        " of history '" + name + "' is not stored; the ring is unreconstructable (nothing older to "
        "replay it from). The persistence policy must store the oldest slot.");
  // A fully-stored ring (Dense): nothing to recompute.
  const std::size_t stored_count = anchors.size();
  if (static_cast<int>(stored_count) == depth)
    return 0;
  // SAVE bracket: deep-copy every block state, the scheduler cache, and the WHOLE history subsystem
  // (rings + slot_dt + initialized) so the replay's own store_history / rotate_histories side effects
  // are fully undone -- the live state U and cache_ are identity after replay, and only the missing
  // ring slots we place by index below survive.
  std::vector<MultiFab> saved_states;
  saved_states.reserve(p_->sp.size());
  for (auto& block : p_->sp)
    saved_states.push_back(block.U);  // deep copy
  const pops::runtime::program::CacheManager saved_cache = p_->program_.cache_;
  const pops::runtime::program::HistoryManager saved_hist = p_->program_.hist_;

  // The per-slot dt each store produced, captured from the SAVED snapshot into a stable local vector.
  // CRITICAL: the replay's own store_history / rotate_histories MUTATE p_->program_.hist_.slot_dt, so
  // reading the live map inside the loop would give a moving target -- dts[j] is the dt that produced
  // the state now in slot j on the ORIGINAL forward run, which is exactly what re-stepping needs.
  std::vector<Real> dts(static_cast<std::size_t>(depth), Real(0));
  auto saved_dt_it = saved_hist.slot_dt.find(name);
  if (saved_dt_it != saved_hist.slot_dt.end()) {
    const std::vector<Real>& sd = saved_dt_it->second;
    for (int k = 0; k < depth && k < static_cast<int>(sd.size()); ++k)
      dts[static_cast<std::size_t>(k)] = sd[static_cast<std::size_t>(k)];
  }

  // Reconstruct the block-0 state trajectory: for each gap between adjacent anchors (older anchor at a
  // LARGER index, newer at a SMALLER one; time increases as the index decreases), seed block 0 from the
  // older stored slot then step forward, recording the post-step block state into each intervening slot.
  // Placement is BY INDEX (no rotate) -> the ADC-538 rotation-invalidation edge is sidestepped.
  std::vector<MultiFab> reconstructed(static_cast<std::size_t>(depth));
  for (std::size_t a = 0; a + 1 < anchors.size(); ++a) {
    const int older = anchors[a + 1];  // larger index = further back in time
    const int newer = anchors[a];       // smaller index = closer to now
    // Seed block 0 with the older stored slot's state (the ring holds the block-0 state at that lag).
    pops::lincomb(p_->sp[0].U, Real(1), saved_hist.histories.at(name)[static_cast<std::size_t>(older)],
                  Real(0), p_->sp[0].U);
    // Step forward from `older` down to `newer`, capturing each intermediate slot. The dt for the store
    // that produced slot j is dts[j] (recorded on the forward run), so re-stepping with it reproduces a
    // variable-dt history exactly.
    for (int j = older - 1; j >= newer; --j) {
      p_->program_.last_dt_ = dts[static_cast<std::size_t>(j)];
      p_->program_.step_(static_cast<double>(dts[static_cast<std::size_t>(j)]));
      // Record the fresh block state for slot j. Slot `newer` is a stored anchor (its restored value is
      // reinstated below), so recording it here is harmless; the non-anchor slots are the real output.
      reconstructed[static_cast<std::size_t>(j)] = p_->sp[0].U;  // deep copy the fresh block state
    }
  }

  // RESTORE bracket: undo every replay side effect (block states, cache, whole history subsystem).
  for (std::size_t b = 0; b < p_->sp.size(); ++b)
    p_->sp[b].U = std::move(saved_states[b]);
  p_->program_.cache_ = saved_cache;
  p_->program_.hist_ = saved_hist;

  // Place ONLY the recomputed slots (the anchors keep their restored values). Re-fetch the ring after
  // restoring hist_ (the restore replaced the vector).
  std::vector<MultiFab>& out_ring = p_->program_.hist_.histories.at(name);
  std::vector<bool> is_stored(static_cast<std::size_t>(depth), false);
  for (int s : anchors)
    is_stored[static_cast<std::size_t>(s)] = true;
  int recomputed = 0;
  for (int j = 0; j < depth; ++j) {
    if (is_stored[static_cast<std::size_t>(j)])
      continue;
    pops::lincomb(out_ring[static_cast<std::size_t>(j)], Real(1),
                  reconstructed[static_cast<std::size_t>(j)], Real(0),
                  out_ring[static_cast<std::size_t>(j)]);
    ++recomputed;
  }
  return recomputed;
}

// Load a generated problem.so and install its compiled time Program. Mirrors add_native_block
// (native_loader.hpp): self-promote this module to the global scope so the .so resolves the System
// seam accessors (POPS_EXPORT) against it, dlopen, fail-loud on ABI-key mismatch, then call
// pops_install_program(this) which wraps the System in a ProgramContext and installs the macro-step
// closure. The .so stays loaded for the process lifetime (the closure runs every step).
POPS_EXPORT void System::install_program(const std::string& so_path) {
  require_assembling(p_->lifecycle_, "install_program");  // frozen once pops.bind completes (ADC-592)
#if defined(_WIN32)
  // Windows: the generated .dll links against _pops.lib at compile time; no global promotion needed.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h) {
    throw std::runtime_error("System::install_program: LoadLibrary('" + so_path +
                             "'): " + pops::dynlib::last_error());
  }
#else
  {
    // Promote the already-loaded module (found via an exported symbol) to the global scope so the
    // .so's undefined System seam symbols (POPS_EXPORT) resolve against it. macOS: harmless (the .so
    // is built with -undefined dynamic_lookup).
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&pops::abi_key), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error(
        "System::install_program: dlopen('" + so_path + "'): " + std::string(e ? e : "?") +
        " (the pops::System seam accessors must be exported AND the module loaded "
        "globally; cf. POPS_EXPORT)");
  }
#endif
  auto key_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_abi_key"));
  if (!key_fn) {
    pops::dynlib::close(h);
    throw std::runtime_error("System::install_program: pops_program_abi_key missing from '" +
                             so_path +
                             "' (regenerate the problem module with the current pops headers)");
  }
  const std::string loader_key = key_fn();
  const std::string module_key = pops::abi_key();
  if (loader_key != module_key) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "System::install_program: compiled program ABI mismatch: expected '" + module_key +
        "', got '" + loader_key +
        "'. Recompile the problem module with the SAME compiler, C++ standard and "
        "pops headers as the _pops module.");
  }
  // Route registry guard (ADC-599): refuse a problem.so whose embedded route manifest
  // (pops_program_route_manifest) disagrees with the current registry, right after the ABI-key
  // check. Optional symbol: a pre-ADC-599 .so carries nothing -> verify_route_manifest("") no-op.
  {
    auto manifest_fn = reinterpret_cast<const char* (*)()>(
        pops::dynlib::sym(h, "pops_program_route_manifest"));
    try {
      pops::verify_route_manifest(
          manifest_fn ? std::string(manifest_fn()) : std::string(), "install_program");
    } catch (...) {
      pops::dynlib::close(h);
      throw;
    }
  }
  auto install = reinterpret_cast<void (*)(void*)>(pops::dynlib::sym(h, "pops_install_program"));
  if (!install) {
    pops::dynlib::close(h);
    throw std::runtime_error("System::install_program: pops_install_program missing from '" +
                             so_path + "'");
  }
#if !defined(_WIN32)
  // Spec-2 criterion 24 (ADC-446): install-time requirement validation. The problem.so carries, per
  // operator, the aux fields its body reads (pops_module_operator_requirements -> read_module_metadata).
  // Reject BEFORE installing the program if the simulation did not provide a required field, with a
  // spec-style message, instead of a cryptic failure mid-step. A pre-Spec-2 .so (present == false) or
  // an operator with no aux requirement carries nothing to check -> skip (backward compatible). Only
  // the user-supplied application fields (B_z, T_e) are hard requirements; derived/lazy fields cannot
  // block (see SystemFieldSolver::provides_aux). POSIX only: read_module_metadata uses dlsym directly.
  {
    const auto meta = pops::runtime::program::read_module_metadata(h);
    const std::vector<std::string> sys_block_names = block_names();
    const std::string configured_solver = poisson_solver();
    auto has_block = [&sys_block_names](const std::string& want) {
      for (const auto& nm : sys_block_names) {
        if (nm == want) {
          return true;
        }
      }
      return false;
    };
    for (const auto& op : meta.operators) {
      // (a) AUX FIELD requirements (ADC-446): the user-supplied application fields B_z / T_e. Only
      // these are hard requirements (provides_aux); the derived fields phi/grad cannot block.
      for (const auto& aux : pops::runtime::program::required_aux(op.requirements)) {
        if (!p_->fields_.provides_aux(aux)) {
          pops::dynlib::close(h);
          throw std::runtime_error(
              "System::install_program: operator '" + op.name + "' requires aux field '" + aux +
              "', but simulation did not provide it (B_z -> set_magnetic_field, T_e -> "
              "set_electron_temperature_from, before install_program)");
        }
      }
      // (b) BLOCK-INSTANCE requirements (ADC-466, Spec criterion 24): an operator that reads another
      // species (e.g. collisions) names the block instance it needs; reject if it was not added. The
      // verbatim spec message names the operator and the missing instance.
      for (const auto& blk : pops::runtime::program::required_blocks(op.requirements)) {
        if (!has_block(blk)) {
          pops::dynlib::close(h);
          throw std::runtime_error("operator '" + op.name + "' requires block instance '" + blk +
                                   "'");
        }
      }
      // (c) SOLVER requirement (ADC-466): a field operator that requires a named field solver is
      // rejected at install when the configured Poisson solver (set_poisson) does not match. The
      // verbatim spec message names the field operator and the required solver.
      const std::string need_solver = pops::runtime::program::required_solver(op.requirements);
      if (!need_solver.empty() && need_solver != configured_solver) {
        pops::dynlib::close(h);
        throw std::runtime_error("field operator '" + op.name + "' requires solver '" + need_solver +
                                 "'");
      }
    }
  }
#endif
  // NAME-based block binding (Spec 3 criterion 23, ADC-457). A compiled Program numbers its blocks in
  // P.state declaration order (the .so's pops_program_block_name table); the System numbers its blocks
  // in add order (block_names). They need NOT agree -- bind by NAME, not add-order. Read the .so's
  // block names, map each Program block index to the System block of that name, and store the
  // program-index -> system-index map (read by ProgramContext to resolve every ctx.state / rhs_into /
  // commit). A Program block whose name has no instantiated System block fails loud with the spec
  // message. A pre-Spec-3 .so (the count symbol absent) carries no table -> clear the map (identity),
  // i.e. the historical positional convention. Built BEFORE install() so the step closure (which
  // captures a ProgramContext) sees the map on its first run.
  {
    using count_t = int (*)();
    using name_t = const char* (*)(int);
    auto block_count = reinterpret_cast<count_t>(pops::dynlib::sym(h, "pops_program_block_count"));
    auto block_name = reinterpret_cast<name_t>(pops::dynlib::sym(h, "pops_program_block_name"));
    if (block_count && block_name) {
      const std::vector<std::string> sys_names = block_names();
      const int n = block_count();
      std::vector<int> prog_to_sys(static_cast<std::size_t>(n), -1);
      for (int p = 0; p < n; ++p) {
        const std::string want = block_name(p);
        int found = -1;
        for (std::size_t s = 0; s < sys_names.size(); ++s)
          if (sys_names[s] == want) {
            found = static_cast<int>(s);
            break;
          }
        if (found < 0) {
          pops::dynlib::close(h);
          throw std::runtime_error("Program requires block instance '" + want +
                                   "', but simulation did not instantiate it");
        }
        prog_to_sys[static_cast<std::size_t>(p)] = found;
      }
      set_program_block_map(prog_to_sys);
    } else {
      set_program_block_map({});  // pre-Spec-3 .so: no name table -> identity (positional convention)
    }
  }
  // RUNTIME PARAMETERS (ADC-510, Spec 5 C5). A Program whose physics reads dsl.Param(..., kind="runtime")
  // exports a pops_program_param_* table: per flat parameter, its PROGRAM block index, its stable index
  // WITHIN that block (sorted-name order, matching the lowered params.get(index)) and its declaration
  // default. Group the defaults per block (in index order) and seed each block's RuntimeParams to those
  // defaults, so an install WITHOUT a runtime set behaves as with a const param. A later Python params=
  // route overwrites the supplied values via set_program_params. A Program with no runtime param (the
  // count symbol absent or 0) seeds nothing -> the param store stays empty (program_params returns
  // count 0, the lowered kernels read no param). Built BEFORE install() so the step closure (which
  // captures a ProgramContext) reads the seeded value on its first run.
  {
    using count_t = int (*)();
    using ival_t = int (*)(int);
    using dval_t = double (*)(int);
    auto pcount = reinterpret_cast<count_t>(pops::dynlib::sym(h, "pops_program_param_count"));
    auto pblock = reinterpret_cast<ival_t>(pops::dynlib::sym(h, "pops_program_param_block"));
    auto pindex = reinterpret_cast<ival_t>(pops::dynlib::sym(h, "pops_program_param_index"));
    auto pdef = reinterpret_cast<dval_t>(pops::dynlib::sym(h, "pops_program_param_default"));
    if (pcount && pblock && pindex && pdef) {
      const int np = pcount();
      std::map<int, std::vector<double>> defaults_by_block;  // program block -> defaults in index order
      for (int i = 0; i < np; ++i) {
        const int blk = pblock(i);
        const int idx = pindex(i);
        std::vector<double>& d = defaults_by_block[blk];
        if (static_cast<int>(d.size()) <= idx)
          d.resize(static_cast<std::size_t>(idx) + 1, 0.0);
        d[static_cast<std::size_t>(idx)] = pdef(i);
      }
      for (const auto& kv : defaults_by_block)
        seed_program_params(kv.first, kv.second);
    }
  }
  install(static_cast<void*>(this));
  // Record the program's IR hash (ADC-406b): the optional pops_program_hash export (a stable IR key,
  // cf. _PROGRAM_CPP_TEMPLATE) is serialized in the checkpoint so a restart against a DIFFERENT
  // compiled Program is rejected fail-loud. Missing symbol (older module) -> empty hash, no guard.
  auto hash_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_hash"));
  p_->program_.installed_hash_ = hash_fn ? std::string(hash_fn()) : std::string();
  // OPTIONAL dt bound (epic ADC-399 / ADC-417, spec s18). A Program may export a SECOND ABI pair --
  // pops_program_has_dt_bound() and pops_program_dt_bound(ProgramContext*, Real cfl) -- alongside
  // pops_install_program. When present AND has_dt_bound() is true, store a closure that builds a
  // ProgramContext over THIS System and runs the .so's lowered dt_bound expression for a given cfl;
  // step_cfl tightens dt to min(native CFL, program dt bound). A Program WITHOUT a dt bound (older
  // module / has_dt_bound() == false) clears the closure -> the native CFL is used UNCHANGED.
  using has_dt_t = bool (*)();
  using dt_bound_t = pops::Real (*)(pops::runtime::program::ProgramContext*, pops::Real);
  auto has_dt = reinterpret_cast<has_dt_t>(pops::dynlib::sym(h, "pops_program_has_dt_bound"));
  auto dt_bound = reinterpret_cast<dt_bound_t>(pops::dynlib::sym(h, "pops_program_dt_bound"));
  if (has_dt && dt_bound && has_dt()) {
    System* self = this;
    p_->program_.dt_bound_ = [self, dt_bound](Real cfl) -> Real {
      pops::runtime::program::ProgramContext ctx(self);
      return dt_bound(&ctx, cfl);
    };
  } else {
    p_->program_.dt_bound_ = nullptr;  // no program dt bound -> native CFL unchanged
  }
  // .so left loaded for the duration of the process (the installed closure points to code in it).
}
std::string System::installed_program_hash() const {
  return p_->program_.installed_hash_;
}
// RUNTIME FREEZE LIFECYCLE (ADC-592 / ADC-578). mark_bound() is the ONE transition into the frozen
// state; the Python bind flow calls it LAST (after every install call), so the install sequence itself
// never trips require_assembling. A second call throws (a composition binds exactly once).
// lifecycle_state() reports "assembling" (not bound), "bound" (bound, no macro-step advanced),
// "running" (bound AND macro_step_ > 0) -- the running edge is derived from the macro-step counter, so
// it needs no extra state (and SystemStepper never reads lifecycle_ -> no MockImpl impact). The new
// checkpointed / finalized phases (SystemLifecycle) are reachable only through explicit transitions
// with no current caller, so the observable strings above are preserved bit-for-bit.
void System::mark_bound() {
  p_->lifecycle_.to_bound();  // Assembling -> Bound; throws the same message on a second bind
}
std::string System::lifecycle_state() const {
  // "running" stays DERIVED from the macro-step counter (the stepper never touches lifecycle_), so
  // the observable three strings are unchanged; the new checkpointed / finalized states surface only
  // when explicitly transitioned (no current caller).
  return p_->lifecycle_.state(p_->macro_step_);
}
// SCHEDULER VALUE CACHE (ADC-458): the System-owned CacheManager every ProgramContext forwards to. The
// .so resolves this across the dlopen boundary (POPS_EXPORT), so the step closure's cache_store_aux /
// cache_should_update reach the SAME manager the checkpoint serializes.
POPS_EXPORT pops::runtime::program::CacheManager& System::program_cache() { return p_->program_.cache_; }
// Scheduler-cache checkpoint/restart seam (ADC-458, Spec 3 section 30): the System owns the cache, so
// the facade (sim.checkpoint / sim.restart) gathers and restores it DIRECTLY -- reusing the SAME global
// gather (gather_global, via copy_state) / scatter (write_state) machinery as the block state and the
// history rings, so the round-trip is MPI-safe and bit-identical under np>1. Mirrors the history seam.
std::vector<int> System::program_cache_nodes() const { return p_->program_.cache_.node_ids(); }
std::string System::program_cache_name(int node_id) const {
  return p_->program_.cache_.name_of(node_id);
}
int System::program_cache_last_update_step(int node_id) const {
  return p_->program_.cache_.last_update_step(node_id);
}
double System::program_cache_accumulated_dt(int node_id) const {
  return static_cast<double>(p_->program_.cache_.accumulated_dt_of(node_id));
}
int System::program_cache_ncomp(int node_id) const { return p_->program_.cache_.ncomp_of(node_id); }
int System::program_cache_ngrow(int node_id) const { return p_->program_.cache_.ngrow_of(node_id); }
std::vector<double> System::program_cache_global(int node_id) const {
  // Reuse the Impl multi-box gather (copy_state -> gather_global): the cache value is co-distributed
  // with block 0's storage (ba/dm), so this is the SAME component-major gather state_global / history_
  // global use (device_fence + all_reduce). All ranks call it; @throws if @p node_id is absent.
  const MultiFab& v = p_->program_.cache_.value_of(node_id);
  return p_->copy_state(v, v.ncomp());
}
void System::restore_program_cache(int node_id, int ncomp, int ngrow, int last_update_step,
                                   double accumulated_dt, const std::string& name,
                                   const std::vector<double>& values) {
  if (p_->sp.empty())
    throw std::runtime_error(
        "System::restore_program_cache: no block exists yet; the cache value is co-distributed with "
        "block 0's storage (replay the composition before restart)");
  // Allocate a value co-distributed with block 0 (ba/dm, @p ncomp comps, @p ngrow ghosts -- the SAME
  // ghost width the slot was cached with: 1 for the aux, the block-state width for a held scratch) and
  // scatter the GLOBAL buffer into it via the SAME write_state set_state uses (owner rank writes,
  // others no-op) -- the true inverse of program_cache_global. Then re-key the slot with its
  // bookkeeping. MPI-safe (all ranks call), bit-identical under np>1.
  MultiFab value(p_->ba, p_->dm, ncomp, ngrow);
  value.set_val(Real(0));
  p_->write_state(value, ncomp, values);
  p_->program_.cache_.restore_slot(node_id, std::move(value), last_update_step,
                                  static_cast<Real>(accumulated_dt), name);
}
// Configured field (Poisson) solver token, owned by SystemFieldSolver (p_solver, default
// "geometric_mg"). Read by install_program (Spec criterion 24, solver requirement) and exposed for
// introspection. Returns the last set_poisson solver, never empty (the default stands).
std::string System::poisson_solver() const {
  return p_->fields_.p_solver;
}
// NAME-based block binding seam (Spec 3 criterion 23, ADC-457). install_program builds the map after
// matching the .so's block names; ProgramContext reads it to translate a Program block index to the
// name-matched System block index. POPS_EXPORT: resolved by the generated .so across the dlopen boundary.
void System::set_program_block_map(const std::vector<int>& prog_to_sys) {
  p_->program_.block_map_ = prog_to_sys;
}
const std::vector<int>& System::program_block_map() const {
  return p_->program_.block_map_;
}
// Block positivity projection (ADC-177) reached by a compiled Program (ProgramContext::apply_projection,
// spec op 21). REUSES the block's own projection closure; a block without one is a no-op.
void System::block_project(int b, MultiFab& u) {
  std::function<void(MultiFab&)>& proj = p_->sp[static_cast<std::size_t>(b)].project;
  if (proj)
    proj(u);
}
// Compiled-Program scalar diagnostics (ADC-414, spec op 23): the installed program writes named scalars
// via P.record_scalar (ProgramContext::record_scalar); Python reads them after the step. Delegated to
// the extracted Program subsystem (ADC-594); the read keeps the "System::program_diagnostic" wording.
void System::record_program_diagnostic(const std::string& name, Real value) {
  p_->program_.record_diagnostic(name, value);
}
Real System::program_diagnostic(const std::string& name) const {
  return p_->program_.diagnostic(name, "System");
}
std::map<std::string, Real> System::program_diagnostics() const {
  return p_->program_.diagnostics();
}
// COMPILED-PROGRAM RUNTIME PARAMETERS (ADC-510, Spec 5 C5). Seed/overwrite/read the per-PROGRAM-block
// RuntimeParams the installed step closure reads through ProgramContext::program_params. Delegated to
// the extracted Program subsystem (ADC-594): the store lives in program_ so a value change reaches the
// captured ctx -- the no-recompile contract mirrored from the AOT-native set_block_params. The fail-loud
// messages keep the "System::set_program_params" wording (unchanged). install_program seeds the
// defaults; Python's _install_params overwrites the supplied values (validated against the .so metadata).
void System::seed_program_params(int prog_block, const std::vector<double>& defaults) {
  p_->program_.seed_params(prog_block, defaults);  // idempotent: re-seeding resets to the baseline
}
void System::set_program_params(int prog_block, const std::vector<double>& values) {
  p_->program_.set_params(prog_block, values, "System");
}
RuntimeParams System::program_params(int prog_block) const {
  return p_->program_.params(prog_block);
}
std::vector<double> System::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
int System::n_vars(const std::string& name) const {
  return p_->find(name).ncomp;
}
std::vector<std::string> System::variable_names(const std::string& name,
                                                const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  if (kind == "conservative")
    return s.cons_vars.names;
  if (kind == "primitive")
    return s.prim_vars.names;
  throw std::runtime_error(
      "System::variable_names : kind 'conservative' | 'primitive' (received '" + kind + "')");
}
std::vector<std::string> System::variable_roles(const std::string& name,
                                                const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  const VariableSet* vs = nullptr;
  if (kind == "conservative")
    vs = &s.cons_vars;
  else if (kind == "primitive")
    vs = &s.prim_vars;
  else
    throw std::runtime_error(
        "System::variable_roles : kind 'conservative' | 'primitive' (received '" + kind + "')");
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(vs->size));
  for (int i = 0; i < vs->size; ++i)
    out.push_back(role_name(vs->at(i).role));  // 'custom' if absent
  return out;
}
double System::block_gamma(const std::string& name) const {
  return p_->find(name).gamma;
}

int System::nx() const {
  return p_->cfg.n;
}
// SLOW axis of the field (rows of the (ny, nx) array). We read it from the INDEX domain (dom = nx() x ny()),
// SINGLE SOURCE of the extents for both geometries: Cartesian dom = n x n -> ny() == nx() == n (square,
// UNCHANGED); polar dom = nr x ntheta -> nx() == nr (fast, i), ny() == ntheta (slow, j). It is this
// dimension that sizes the numpy array on the bindings side: a polar field has nx()*ny() = nr*ntheta
// values, and with nr != ntheta the square reshape (nx, nx) overflows the buffer (teardown bug).
int System::ny() const {
  return p_->dom.ny();
}
double System::time() const {
  return p_->t;
}
int System::n_species() const {
  return p_->blocks_.size();
}
std::vector<std::string> System::block_names() const {
  // SINGLE block registry (store), populated by all add paths: a block loaded via
  // add_dynamic_block / add_compiled_block (.so) appears there just like an add_block.
  return p_->blocks_.names();
}

EffectiveOptionsReport System::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "system";
  report.time_scheme = p_->time_scheme_;
  report.gauss_policy = p_->gauss_policy_;
  report.poisson.rhs = p_->fields_.p_rhs;
  report.poisson.solver = p_->fields_.p_solver;
  report.poisson.bc = p_->fields_.p_bc;
  report.poisson.wall = p_->fields_.p_wall;
  report.poisson.wall_radius = p_->fields_.p_wall_radius;
  report.poisson.epsilon = static_cast<double>(p_->fields_.p_eps_);
  report.poisson.abs_tol = static_cast<double>(p_->fields_.p_abs_tol_);
  report.poisson.has_epsilon_field = p_->fields_.has_eps_field_;
  report.poisson.has_anisotropic_epsilon = p_->fields_.has_eps_xy_field_;
  report.poisson.has_reaction_field = p_->fields_.has_kappa_field_;

  for (const Impl::Species& s : p_->sp) {
    EffectiveBlockOptions row;
    if (const EffectiveBlockOptions* opt = p_->diagnostics_.block_options_ptr(s.name))
      row = *opt;
    row.name = s.name;
    row.ncomp = s.ncomp;
    row.n_ghost = s.U.n_grow();
    row.substeps = s.substeps;
    row.stride = s.stride;
    row.evolve = s.evolve;
    row.gamma = s.gamma;
    row.conservative_vars = s.cons_vars.names;
    row.primitive_vars = s.prim_vars.names;
    report.blocks.push_back(std::move(row));
  }

  for (const auto& kv : p_->diagnostics_.source_stage_options)
    report.source_stages.push_back(kv.second);
  return report;
}
double System::mass(const std::string& name) const {
  const Impl::Species& s = p_->find(name);
  if (!p_->polar_)
    return sum(s.U, 0);  // Cartesian: bare sum of the cells (bit-identical)
  // POLAR: FV mass = Sum_ij n_ij r_i dr dtheta (annular cell volume r dr dtheta). This is the
  // quantity CONSERVED by assemble_rhs_polar (cf. test_polar_transport_mms). Host loop over the valid
  // cells (mono-rank: a single local fab), reduced over the ranks by symmetry (n_ranks==1).
  device_fence();
  const PolarGeometry& g = p_->pgeom_;
  const Real dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m += static_cast<double>(u(i, j, 0)) * static_cast<double>(g.r_cell(i) * dr * dth);
  }
  return all_reduce_sum(m);
}
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  device_fence();
  // POLAR: phi comes from the polar Poisson (pell_), not from the Cartesian solver (ell_). We build it
  // lazily if needed (a call before any step) and we read phi() of PolarPoissonSolver.
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    // Rank without a box (MPI mono-box): EMPTY return (no fab(0)). Cf. copy_comp0; the multi-rank
    // global field goes through System::potential_global.
    if (p_->aux.local_size() == 0)
      return {};
    const ConstArray4 ph = p_->fields_.pell_->phi().fab(0).const_array();
    const Box2D v = p_->aux.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out.push_back(ph(i, j));
    return out;
  }
  p_->fields_.ensure_elliptic();
  if (p_->aux.local_size() == 0)
    return {};  // rank without a box: empty (cf. potential_global)
  const ConstArray4 ph = p_->fields_.ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      out.push_back(ph(i, j));
  return out;
}

// --- GLOBAL accessors (collective MPI-safe), IO v1 multi-rank --------------------------------
// All three delegate to gather_global (anon namespace, top of file): a GLOBAL buffer filled by the
// LOCAL fabs at GLOBAL indices then all_reduce_sum_inplace, component-major. Mono-rank: the box
// covers the domain and the reduce is the identity -> array bit-identical to the non-global
// accessors (density / get_state / potential). The device_fence is owned here (before the gather).
std::vector<double> System::density_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  return gather_global(s.U, 1, nx(), ny());
}
std::vector<double> System::state_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  return gather_global(s.U, s.ncomp, nx(), ny());
}
std::vector<double> System::potential_global() {
  device_fence();
  // Resolve phi, solving the Poisson (polar or Cartesian) if needed: COLLECTIVE, like the gather.
  const MultiFab* phi = nullptr;
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    phi = &p_->fields_.pell_->phi();
  } else {
    p_->fields_.ensure_elliptic();
    phi = &p_->fields_.ell_phi();
  }
  return gather_global(*phi, 1, nx(), ny());
}

// --- LOCAL per-fab accessors (NON collective): parallel HDF5 write by hyperslabs (PR-IO-3) --
// Local counterpart of the _global accessors: they aggregate nothing (no MPI comm), they expose per rank
// the LOCAL boxes (in GLOBAL indices, as carried by the fab box) and the state of each fab.
// The facade sim.write(format='hdf5', parallel=True) creates the global datasets then each rank writes
// ITS boxes in hyperslabs. A rank without a box -> local_size()==0 -> empty list (never a hard-coded fab(0)).
std::vector<std::array<int, 4>> System::local_boxes(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  std::vector<std::array<int, 4>> out;
  out.reserve(s.U.local_size());
  for (int li = 0; li < s.U.local_size(); ++li) {
    const Box2D v = s.U.box(li);
    out.push_back({v.lo[0], v.lo[1], v.hi[0], v.hi[1]});  // (ilo, jlo, ihi, jhi) GLOBAL
  }
  return out;
}
std::vector<double> System::local_state(const std::string& name, int li) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  if (li < 0 || li >= s.U.local_size())
    throw std::out_of_range("System::local_state : local fab index out of bounds (0.." +
                            std::to_string(s.U.local_size() - 1) + ")");
  const int nc = s.ncomp;
  const ConstArray4 u = s.U.fab(li).const_array();
  const Box2D v = s.U.box(li);
  const int bnx = v.nx(), bny = v.ny();  // dimensions of the LOCAL box (valid cells)
  std::vector<double> out(static_cast<std::size_t>(nc) * bnx * bny, 0.0);
  // Layout = state_global mapped to the local box: (c*bny + jl)*bnx + il, component-major, so
  // reshapeable into (nc, bny, bnx) for a hyperslab dset[:, jlo:jhi+1, ilo:ihi+1].
  for (int c = 0; c < nc; ++c)
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[(static_cast<std::size_t>(c) * bny + (j - v.lo[1])) * bnx + (i - v.lo[0])] =
            static_cast<double>(u(i, j, c));
  return out;
}

}  // namespace pops
