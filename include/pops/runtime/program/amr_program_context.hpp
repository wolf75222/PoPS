#pragma once

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>     // Real, POPS_HD
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts
#include <pops/mesh/execution/for_each.hpp>    // for_each_cell, device_fence
#include <pops/mesh/geometry/geometry.hpp>     // Geometry
#include <pops/mesh/storage/mf_arith.hpp>      // saxpy / lincomb
#include <pops/mesh/storage/multifab.hpp>      // MultiFab
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>             // GeometricMG (Krylov precond)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>    // apply_laplacian
#include <pops/runtime/amr/amr_runtime.hpp>     // AmrRuntime (the engine the driver wraps)
#include <pops/runtime/amr_system.hpp>          // AmrSystem (the facade: params / block map / engine)
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams

/// @file
/// @brief AmrProgramContext -- the AMR counterpart of ProgramContext (epic ADC-508, Spec 6).
///
/// A compiled time Program lowers its macro-step body referencing ONLY the variable `ctx` (never the
/// type ProgramContext). The SAME generated body therefore compiles against any object exposing
/// ProgramContext's method surface. AmrProgramContext is that duck-typed structural mirror, driving the
/// lowered body PER LEVEL over the AMR hierarchy (an AmrRuntime). The `{amr_install}` codegen slot wraps
/// the identical body in a per-level loop and constructs an AmrProgramContext instead of a ProgramContext.
///
/// SCOPE (v1): SYNCHRONOUS (non-subcycled) multilevel step -- every level advances with the SAME dt, then
/// the levels couple via average_down (fine -> coarse) and the next macro-step's head-of-step
/// solve_fields re-solves the coarse Poisson. This is NOT Berger-Oliger subcycling (the native non-Program
/// AMR path keeps subcycling). The single coarse system Poisson per macro-step (OncePerStep) is injected
/// coarse -> fine; solve_fields() runs exactly once per macro-step (a level-0 guard). DEFERRED (fail-loud
/// or documented): Berger-Oliger subcycling under a Program, conservative reflux at coarse-fine interfaces
/// (v1 ships average_down-only), multi-block AMR Program coupling, named multi-elliptic fields, per-stage
/// field re-solve at FINE levels. Multistep history rings (keep_history / T.prev) ARE supported (ADC-631:
/// per-level ring slots remapped through regrid + v3 checkpoint with native replay). GPU (CUDA) run is the ROMEO step (device-clean by
/// construction: every per-cell op is for_each_cell / a POPS_HD named functor reused from the engine).
namespace pops {
namespace runtime {
namespace program {

class AmrProgramContext {
 public:
  /// Wrap an AmrSystem passed as a flat void* (what pops_install_program_amr(void* sys) receives). The
  /// ctor pulls the AmrRuntime engine out of the facade (engine() returns the built runtime; the AMR
  /// blocks must be materialized -- install_program forces the build before install()).
  explicit AmrProgramContext(void* sys)
      : facade_(static_cast<AmrSystem*>(sys)), eng_(facade_->engine()) {
    if (eng_ == nullptr)
      throw std::runtime_error(
          "AmrProgramContext: the AMR runtime engine is not built; install_program must force the "
          "multi-block AmrRuntime build before installing a compiled time Program over the hierarchy");
  }
  /// Direct ctor (C++ tests / the driver): an engine + the facade carrying the param / block-map stores.
  AmrProgramContext(AmrRuntime* eng, AmrSystem* facade) : facade_(facade), eng_(eng) {}

  // --- driver state (mutable: every seam method is const, like ProgramContext::mg_precond_) ----------
  void set_level(int k) const { level_ = k; }
  int level() const { return level_; }
  int nlev() const { return eng_->nlev(); }
  /// Reset the per-macro-step flags (called by the install wrapper at the top of each macro-step).
  void reset_step() const { solved_this_step_ = false; }

  /// Register the macro-step body (forwards to AmrSystem::install_program_step). @p step is the per-level
  /// loop wrapper the codegen emits; it runs ONE macro-step over dt.
  void install(std::function<void(double)> step) const {
    facade_->install_program_step(std::move(step));
  }

  /// Translate a PROGRAM block index to its AMR block index (Spec 3 criterion 23). Empty map = identity.
  int sys_block(int b) const {
    const std::vector<int>& m = facade_->program_block_map();
    return (b >= 0 && b < static_cast<int>(m.size())) ? m[static_cast<std::size_t>(b)] : b;
  }
  int n_blocks() const { return static_cast<int>(eng_->n_blocks()); }

  // --- inter-level coupling (the synchronous driver's (B) phase) --------------------------------------
  /// fine -> coarse average_down over ALL blocks (covered coarse cell <- 2x2 fine average), then a fresh
  /// coarse Poisson + injection so every level's aux is consistent for the next macro-step.
  ///
  /// CONSERVATION LIMITATION (v1, conservation-affecting -- NOT a cosmetic deferral). This ships
  /// average_down ONLY, with NO conservative reflux at coarse-fine interfaces. average_down corrects the
  /// covered coarse CELL VALUES (fine -> coarse restriction) but does NOT correct the coarse FLUX at the
  /// C/F boundary by the sum of the fine face fluxes (a reflux). So on a GENUINELY MULTILEVEL Program run
  /// (a real fine patch under the coarse), the total conserved quantity (mass / momentum / energy) is NOT
  /// conserved across the C/F interface -- it drifts by the un-refluxed face-flux mismatch. This is exact
  /// only for the single-LEVEL (coarse-only) Program layout, where there is no C/F interface (the
  /// bit-identical parity gate). A multilevel Program that must conserve at the C/F interface needs the
  /// native AMR route (reflux + average_down) -- pops.compile(Case(layout=AMR(...))) -- or System.
  void couple_levels() const {
    for (int k = nlev() - 1; k >= 1; --k)
      for (int b = 0; b < n_blocks(); ++b)
        eng_->average_down_level(static_cast<std::size_t>(b), k);
  }
  /// Head-of-step regrid at the engine's cadence (the SAME union-tags regrid the native step runs).
  void regrid_if_due(int macro_step) const { eng_->regrid_if_due(macro_step); }

  // --- state / RHS seam over the CURRENT level -------------------------------------------------------
  MultiFab& state(int b) const {
    return eng_->level_state(static_cast<std::size_t>(sys_block(b)), level_);
  }
  void rhs_into(int b, MultiFab& u, MultiFab& r) const {
    count_kernel();
    eng_->level_rhs_into(static_cast<std::size_t>(sys_block(b)), level_, u, r);
  }
  void neg_div_flux_default_into(int b, MultiFab& u, MultiFab& r) const {
    count_kernel();
    eng_->level_neg_div_flux_into(static_cast<std::size_t>(sys_block(b)), level_, u, r);
  }
  void source_default_into(int b, MultiFab& u, MultiFab& r) const {
    count_kernel();
    eng_->level_source_into(static_cast<std::size_t>(sys_block(b)), level_, u, r);
  }
  void apply_projection(int b, MultiFab& u) const {
    // v1: per-level positivity projection is wired through the native block path (project_per_level),
    // not the per-stage Program seam. A Program that requests it on AMR is a documented deferral.
    (void)b;
    (void)u;
    throw std::runtime_error(
        "AmrProgramContext::apply_projection: a per-stage positivity projection under a compiled "
        "Program on AMR is deferred (v1); the native AMR block applies it per level at end-of-step. "
        "Drop P.project from the AMR Program or use System.");
  }

  // --- dt bound primitives (evaluated at the COARSE level, where the AMR CFL lives) -----------------
  Real hmin() const { return eng_->level_hmin(level_); }
  Real max_wave_speed(int b, const MultiFab& u) const {
    return eng_->level_max_speed(static_cast<std::size_t>(sys_block(b)), level_, u);
  }

  // --- field solve (the SHARED coarse Poisson) ------------------------------------------------------
  /// The default head-of-step elliptic solve: the coarse system Poisson + coarse->fine aux injection.
  /// v1 runs it EXACTLY ONCE per macro-step (a level-0 / not-yet-solved guard): calling it again at fine
  /// levels within the same macro-step is a no-op cache-hit (parity: the body stays atomic, the solve
  /// fires once -- the OncePerStep cadence the native AMR step uses).
  void solve_fields() const {
    if (level_ == 0 || !solved_this_step_) {
      eng_->solve_fields();
      solved_this_step_ = true;
    }
  }
  /// Per-stage re-solve from a stage state. v1: honored ONLY at the coarse level (level_ == 0), where a
  /// re-solve from the stage state pushes through the summed coarse Poisson. At fine levels it re-uses
  /// the already-injected aux_[level_] from the head-of-step coarse solve (exact for OncePerStep
  /// Programs such as the SSPRK2 parity Program, where the coupling is frozen across the RK stages).
  void solve_fields_from_state(int b, MultiFab& u_stage) const {
    if (level_ == 0) {
      MultiFab& live = state(b);
      MultiFab saved = live;          // stash the live state
      live = u_stage;                 // solve from the stage state
      eng_->solve_fields();
      live = std::move(saved);        // restore the live state (the commit owns it)
      solved_this_step_ = true;
    }
    // fine level: reuse the injected aux (documented v1 fallback).
  }
  /// Named multi-elliptic field re-solve: deferred on AMR (ADC-513 companion). Fail loud.
  void solve_fields_from_state(const std::string& field, int /*b*/, MultiFab& /*u_stage*/) const {
    throw std::runtime_error(
        "AmrProgramContext::solve_fields_from_state(field='" + field +
        "'): named multi-elliptic fields under a compiled Program on AMR are deferred (ADC-513). Use "
        "System, or a single (default) field.");
  }
  /// Coupled multi-block field solve: deferred on AMR (the per-block summed-RHS at fine levels needs the
  /// coupled re-solve path). Fail loud rather than a silent half-implementation.
  void solve_fields_from_blocks(const std::vector<const MultiFab*>& /*u_stages*/) const {
    throw std::runtime_error(
        "AmrProgramContext::solve_fields_from_blocks: a coupled multi-block field solve under a "
        "compiled Program on AMR is deferred (v1, Spec 3 criterion 24). Use System, or a single-block "
        "AMR Program.");
  }

  /// The SHARED aux of the current level (phi / grad / B_z), the channel solve_fields fills.
  MultiFab& aux() const { return const_cast<MultiFab&>(eng_->aux(level_)); }
  /// The current level's metric (dx/dy >> level, domain << level).
  Geometry geom() const { return eng_->level_geom(level_); }

  // --- scratch (per-level) --------------------------------------------------------------------------
  MultiFab alloc_scalar_field(int n_comp = 1, int n_ghost = 1) const {
    return eng_->level_scalar_field(level_, n_comp, n_ghost);
  }
  MultiFab rhs_scratch_like(const MultiFab& u) const {
    MultiFab scratch(u.box_array(), u.dmap(), u.ncomp(), u.n_grow());
    count_scratch(scratch);
    return scratch;
  }
  MultiFab scratch_state_like(const MultiFab& u) const { return rhs_scratch_like(u); }

  // --- linear algebra (LEVEL-AGNOSTIC: operate on the MultiFab handed in) ---------------------------
  void axpy(MultiFab& u, Real a, const MultiFab& r) const {
    count_kernel();
    pops::saxpy(u, a, r);
  }
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    count_kernel();
    pops::lincomb(z, a, x, b, y);
  }

  // --- matrix-free elliptic primitives over the CURRENT level (parity with ProgramContext) ----------
  void laplacian(MultiFab& out, MultiFab& in) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(in, g.domain, eng_->transport_bc());
    apply_laplacian(in, g, out);
  }
  void gradient(MultiFab& out, MultiFab& phi) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(phi, g.domain, eng_->transport_bc());
    const Real cx = Real(1) / (Real(2) * g.dx());
    const Real cy = Real(1) / (Real(2) * g.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(fx, g.domain, eng_->transport_bc());
    if (&fy != &fx)
      fill_ghosts(fy, g.domain, eng_->transport_bc());
    apply_divergence(fx, fy, g, out, /*cx=*/0, /*cy=*/1);
  }
  void geometric_mg_precond_apply(MultiFab& out, const MultiFab& in) const {
    count_kernel();
    if (!mg_precond_) {
      const Geometry g = eng_->level_geom(level_);
      const MultiFab tmpl = eng_->level_scalar_field(level_, 1, 1);
      mg_precond_.emplace(g, tmpl.box_array(), eng_->poisson_bc());
    }
    GeometricMG& mg = *mg_precond_;
    lincomb(mg.rhs(), Real(1), in, Real(0), in);
    mg.phi().set_val(Real(0));
    mg.vcycle();
    lincomb(out, Real(1), mg.phi(), Real(0), out);
  }

  // --- reductions (COLLECTIVE all_reduce, called on every rank; per-level field) --------------------
  Real sum_component(const MultiFab& u, int comp) const { return pops::reduce_sum(u, comp); }
  Real max_component(const MultiFab& u, int comp) const { return pops::reduce_max(u, comp); }
  Real min_component(const MultiFab& u, int comp) const { return pops::reduce_min(u, comp); }
  Real abs_sum_component(const MultiFab& u, int comp) const { return pops::reduce_abs_sum(u, comp); }
  Real sum(const MultiFab& u) const { return pops::reduce_sum(u, 0); }
  Real max(const MultiFab& u) const { return pops::reduce_max(u, 0); }
  Real min(const MultiFab& u) const { return pops::reduce_min(u, 0); }
  Real abs_sum(const MultiFab& u) const { return pops::reduce_abs_sum(u, 0); }

  void fill_boundary(MultiFab& x) const {
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(x, g.domain, eng_->transport_bc());
  }

  // --- history (ADC-631): per-level ring slots on the AmrRuntime engine, driven by the SAME lowered
  // body as Uniform (the level index is the driver's set_level cursor -- no scheme dispatch, no new IR).
  // register/read/store address the CURRENT level; rotate fires ONCE per macro-step, guarded to the LAST
  // level -- the body's terminal rotate_histories() runs once per level in the AMR per-level loop, so
  // the level_==nlev-1 guard is the AMR analogue of the Uniform once-per-step rotate (design plan sec.2).
  void register_history(const std::string& name, int lag) const {
    pops::detail::AmrHistoryOps::register_history(*eng_, name, lag);  // idempotent; allocates every level
  }
  MultiFab& history(const std::string& name, int lag = 1) const {
    return pops::detail::AmrHistoryOps::read_history(*eng_, name, lag, level_);
  }
  void store_history(const std::string& name, const MultiFab& value) const {
    pops::detail::AmrHistoryOps::store_history(*eng_, name, level_, value, facade_->program_last_dt());
  }
  void rotate_histories() const {
    if (level_ == nlev() - 1)
      pops::detail::AmrHistoryOps::rotate_histories(*eng_);
  }

  // --- diagnostics / runtime params (forward to the facade store) -----------------------------------
  void record_scalar(const std::string& name, Real value) const {
    facade_->record_program_diagnostic(name, value);
  }
  /// Program block @p b's CURRENT RuntimeParams (keyed by PROGRAM index, NOT sys_block -- parity with
  /// ProgramContext: the store is keyed by program index, the same index set_program_params writes).
  RuntimeParams program_params(int b) const { return facade_->program_params(b); }

  // --- profiling counters (forward to the facade profiler; no-op when disabled) ---------------------
  void count_kernel(std::int64_t by = 1) const { facade_->profiler_handle().count("kernels", by); }
  void count_scratch(const MultiFab& mf) const {
    Profiler& prof = facade_->profiler_handle();
    if (!prof.enabled())
      return;
    prof.count("scratch_allocs");
    std::int64_t bytes = 0;
    for (int li = 0; li < mf.local_size(); ++li)
      bytes += mf.fab(li).size() * static_cast<std::int64_t>(sizeof(Real));
    prof.count_max("scratch_peak_bytes", bytes);
  }
  Profiler& profiler() const { return facade_->profiler_handle(); }
  ProfileScope profile_node(const std::string& name) const {
    return ProfileScope(facade_->profiler_handle(), name);
  }
  void profile_record(const std::string& name, std::chrono::steady_clock::time_point t0) const {
    const auto t1 = std::chrono::steady_clock::now();
    facade_->profiler_handle().record(name,
                                      std::chrono::duration<double>(t1 - t0).count());
  }

  int macro_step() const { return facade_->macro_step(); }

  // --- condensed-Schur / named-flux primitives: DEFERRED on AMR (v1), fail loud ---------------------
  // The codegen can lower a condensed-Schur (ADC-421/422) or named-flux (ADC-419) Program against these
  // seams. ProgramContext implements them on the single-level System (coefficiented matrix-free elliptic
  // + the fused Schur RHS / reconstruct / energy + the named-flux divergence); AMR v1 does NOT wire them
  // (they would need the per-level coefficient assembly + the coarse-fine elliptic coupling). A SILENT
  // lower would produce an AMR .so that either does not compile (a missing member) or runs the WRONG
  // arithmetic. The signatures match ProgramContext EXACTLY (const-ness, return type, args) so the SAME
  // lowered body still type-checks against AmrProgramContext (the duck-typing contract); each one throws.
  void assemble_schur_coeffs(MultiFab& /*eps_x*/, MultiFab& /*eps_y*/, MultiFab& /*a_xy*/,
                             MultiFab& /*a_yx*/, const MultiFab& /*state*/, Real /*c*/,
                             Real /*th_dt*/, int /*c_rho*/, int /*c_bz*/) const {
    deferred_op("assemble_schur_coeffs",
                "a condensed-Schur Program on AMR is deferred; use System, or a native AMR block "
                "(pops.compile with a per-block time policy).");
  }
  void apply_laplacian_coeff(MultiFab& /*out*/, MultiFab& /*in*/, const MultiFab& /*eps_x*/,
                             const MultiFab& /*eps_y*/, const MultiFab& /*a_xy*/,
                             const MultiFab& /*a_yx*/) const {
    deferred_op(
        "apply_laplacian_coeff",
        "the coefficiented matrix-free elliptic operator of a condensed-Schur Program on AMR "
        "is deferred; use System, or a native AMR block.");
  }
  void schur_explicit_flux(MultiFab& /*out*/, const MultiFab& /*state*/, Real /*th_dt*/,
                           int /*c_mx*/, int /*c_my*/, int /*c_bz*/) const {
    deferred_op("schur_explicit_flux",
                "a condensed-Schur Program on AMR is deferred; use System, or a native AMR block.");
  }
  void assemble_schur_rhs(MultiFab& /*rhs*/, MultiFab& /*phi_n*/, const MultiFab& /*state*/,
                          Real /*th_dt*/, Real /*g*/, int /*c_mx*/, int /*c_my*/,
                          int /*c_bz*/) const {
    deferred_op("assemble_schur_rhs",
                "a condensed-Schur Program on AMR is deferred; use System, or a native AMR block.");
  }
  void schur_reconstruct(MultiFab& /*state*/, MultiFab& /*phi*/, Real /*th_dt*/, int /*c_rho*/,
                         int /*c_mx*/, int /*c_my*/, int /*c_bz*/) const {
    deferred_op("schur_reconstruct",
                "a condensed-Schur Program on AMR is deferred; use System, or a native AMR block.");
  }
  void schur_energy(MultiFab& /*state*/, const MultiFab& /*state_old*/, int /*c_rho*/, int /*c_mx*/,
                    int /*c_my*/, int /*c_E*/) const {
    deferred_op("schur_energy",
                "a condensed-Schur Program on AMR is deferred; use System, or a native AMR block.");
  }
  void neg_div_flux_into(MultiFab& /*r*/, MultiFab& /*fx*/, MultiFab& /*fy*/) const {
    deferred_op(
        "neg_div_flux_into",
        "a named-flux (-div F) Program on AMR is deferred; use System, or a native AMR block "
        "whose flux IR runs through the level RHS.");
  }

  // --- scheduler value cache: DEFERRED on AMR (v1), fail loud ----------------------------------------
  // The codegen lowers a held / scheduled field-solve node (ADC-458) against these cache seams. The
  // CacheManager that backs them is owned by System (per-installed-Program, keyed by node id); AMR v1
  // has no AmrSystem cache store, so a held schedule on AMR is not wired. Same EXACT signatures as
  // ProgramContext; each throws rather than silently caching nothing (which would read a stale value).
  bool cache_should_update(int /*node_id*/, int /*every_n*/) const {
    deferred_op("cache_should_update",
                "a held / scheduled field solve under a compiled Program on AMR is deferred; use "
                "System (the scheduler cache lives on System), or drop the schedule.");
  }
  void cache_store_aux(int /*node_id*/) const {
    deferred_op("cache_store_aux",
                "the scheduler aux cache under a compiled Program on AMR is deferred; use System.");
  }
  void cache_restore_aux(int /*node_id*/) const {
    deferred_op("cache_restore_aux",
                "the scheduler aux cache under a compiled Program on AMR is deferred; use System.");
  }
  void cache_store_scratch(int /*node_id*/, const MultiFab& /*scratch*/) const {
    deferred_op(
        "cache_store_scratch",
        "the scheduler scratch cache under a compiled Program on AMR is deferred; use System.");
  }
  void cache_restore_scratch(int /*node_id*/, MultiFab& /*scratch*/) const {
    deferred_op(
        "cache_restore_scratch",
        "the scheduler scratch cache under a compiled Program on AMR is deferred; use System.");
  }
  void cache_accumulate_dt(int /*node_id*/, Real /*dt*/) const {
    deferred_op(
        "cache_accumulate_dt",
        "the scheduler accumulate_dt policy under a compiled Program on AMR is deferred; use "
        "System.");
  }
  Real cache_effective_dt(int /*node_id*/, Real /*dt_now*/) const {
    deferred_op(
        "cache_effective_dt",
        "the scheduler accumulate_dt policy under a compiled Program on AMR is deferred; use "
        "System.");
  }
  /// Fail loud: an `error`-policy scheduled node reached off cadence. ProgramContext throws here; the
  /// AMR path never installs a schedule (the cache seams above fail loud first), so this only fires if
  /// the body reaches the off-cadence branch directly. Keep the exact [[noreturn]] signature.
  [[noreturn]] void scheduler_error(const std::string& what) const {
    throw std::runtime_error("pops Program scheduler (AMR): " + what +
                             " (scheduled Programs on AMR are deferred in v1; use System)");
  }

 private:
  /// Fail loud for an op the codegen can emit but the v1 AMR Program path does not yet wire (Schur /
  /// named-flux / scheduled Programs). [[noreturn]] so a non-void stub needs no dummy return -- the
  /// caller's signature stays byte-faithful to ProgramContext (the duck-typing requirement) without
  /// fabricating a value. @p op names the seam; @p detail names the alternative (System or the native
  /// AMR route) so the message is actionable, mirroring the inline fail-loud stubs above.
  [[noreturn]] static void deferred_op(const char* op, const char* detail) {
    throw std::runtime_error(std::string("AmrProgramContext: ") + op +
                             " is not wired on the AMR Program path (v1); " + detail);
  }

  AmrSystem* facade_;
  AmrRuntime* eng_;
  mutable int level_ = 0;
  mutable bool solved_this_step_ = false;
  mutable std::optional<GeometricMG> mg_precond_;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
