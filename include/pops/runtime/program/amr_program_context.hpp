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
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>  // ApplyFn / cg / bicgstab / gmres / richardson (flat solve_linear_matfree)
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>             // GeometricMG (Krylov precond)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>    // apply_laplacian
#include <pops/runtime/amr/amr_runtime.hpp>     // AmrRuntime (the engine the driver wraps)
#include <pops/runtime/amr/amr_schur.hpp>       // AmrSchurElliptic (the composite tensor elliptic, ADC-633)
#include <pops/runtime/context/grid_context.hpp>  // GridContext (per-level Schur assembly seam, ADC-633)
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

  /// The grid context of the CURRENT level (ADC-633): the AMR counterpart of System::grid_context(),
  /// per level. It bundles the transport BC + the level geometry + the live level aux pointer, exactly
  /// what System::grid_context() returns for the uniform mesh. Used by the context-generic condensed-
  /// Schur free functions (condensed_schur_operator.hpp) so their per-cell assembly reads the CURRENT
  /// level's geom / aux / BC as direct body calls (they read the level_ cursor live). transport_bc()
  /// (not poisson_bc()) matches the uniform Program's gc.bc, so the flat-hierarchy phi-ghost fill is
  /// byte-identical to the uniform Program (the flat bit-parity gate). BY VALUE, like ProgramContext.
  GridContext grid_context() const {
    const Geometry g = eng_->level_geom(level_);
    GridContext gc;
    gc.dom = g.domain;
    gc.bc = eng_->transport_bc();
    gc.geom = g;
    gc.aux = &const_cast<MultiFab&>(eng_->aux(level_));
    return gc;
  }

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
  // @p ncomp mirrors ProgramContext::register_history so the SAME lowered body (a single problem.so)
  // compiles against BOTH contexts. The narrow-ring AMR phi^n carry (ADC-427 commit 3) threads @p ncomp
  // into AmrHistoryOps; until then the AMR ring keeps block 0's width (the multistep ring, unchanged),
  // so @p ncomp is accepted-and-ignored here -- a signature-only addition, no AMR semantics change.
  void register_history(const std::string& name, int lag, int ncomp = -1) const {
    (void)ncomp;
    pops::detail::AmrHistoryOps::register_history(*eng_, name, lag);  // idempotent; allocates every level
  }
  MultiFab& history(const std::string& name, int lag = 1) const {
    return pops::detail::AmrHistoryOps::read_history(*eng_, name, lag, level_);
  }
  // ZERO COLD-START read (ADC-427), mirroring ProgramContext::history_zero_start so the SAME lowered
  // body compiles on both contexts: a read-first cross-step carry reads the zero-filled slots on its
  // very first read instead of failing loud. @p ncomp is accepted-and-ignored like register_history
  // above (the AMR narrow ring lands with the ADC-427 AMR commits; block-0 width until then).
  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    (void)ncomp;
    pops::detail::AmrHistoryOps::register_history(*eng_, name, lag);
    if (!pops::detail::AmrHistoryOps::initialized(*eng_, name))
      pops::detail::AmrHistoryOps::set_initialized(*eng_, name, true);
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

  // --- condensed-Schur primitives on the hierarchy (ADC-633): WIRED per level -----------------------
  // The codegen lowers a condensed-Schur (ADC-421/422) Program against the context-generic free kernels
  // pops::coupling::schur::program::<op>(ctx, ...) (condensed_schur_operator.hpp), templated on Ctx. With
  // this context's grid_context() (per level) + assembly_target / assembly_source (write/read redirect),
  // those kernels run the SAME assembly PER LEVEL as direct body calls (they read the level_ cursor
  // live). On a FLAT hierarchy the emitted matrix-free BiCGStab runs on level 0, bit-identical to the
  // uniform Program; on a REFINED hierarchy the tensor elliptic is solved compositely (AmrSchurElliptic
  // + CompositeFacPoisson). The former vestigial deferred_op stubs (and their delegating free-function
  // overloads) are GONE -- the templated kernels bind directly to this context.

  /// Schur assembly WRITE redirection (ADC-633). On a REFINED hierarchy each assembled coefficient /
  /// RHS / flux field must live on the CURRENT level, not the level-0-bound emitted scratch: the kernel
  /// writes THROUGH here into AmrSchurElliptic's per-level buffer. On a FLAT hierarchy (no fine patch)
  /// the emitted level-0 field IS the whole system, so this is the identity (byte-for-byte the uniform
  /// path -- the flat bit-parity gate). @p role is a SchurTargetRole (eps_x / eps_y / a_xy / a_yx / rhs
  /// / flux).
  MultiFab& assembly_target(MultiFab& field, int role) const {
    AmrSchurElliptic& s = schur();
    if (!s.has_fine_patches())
      return field;  // flat / no fine patch: the emitted level-0 field is correct as-is.
    return s.target(role, level_);
  }
  /// Schur reconstruction READ redirection (ADC-633): the fine-level reconstruction reads the level's
  /// published composite potential (the emitted level-0 solution cannot hold a fine level's phi). Flat /
  /// no fine patch: identity (returns the emitted solution). @p role is kSchurPhi.
  MultiFab& assembly_source(MultiFab& field, int /*role*/) const {
    AmrSchurElliptic& s = schur();
    if (!s.has_fine_patches())
      return field;
    return s.phi(level_);
  }
  /// Solve the matrix-free condensed-Schur linear system A(phi) = rhs on the hierarchy (ADC-633). FLAT
  /// (no fine patch): the SAME matrix-free Krylov call as the uniform Program (identical numerics, the
  /// flat bit-parity path -- the load-bearing acceptance). REFINED (>= one fine patch): drive
  /// AmrSchurElliptic::solve_composite (the composite FAC over the tower), which reads the per-level
  /// coefficients / RHS the emitted assembly already wrote through assembly_target and publishes each
  /// level's potential for assembly_source to read; the emitted @p apply / @p precond are UNUSED on this
  /// branch (the FAC has its own operator). @p method is a SchurSolveMethod id (program_context.hpp).
  ///
  /// REFINED-HIERARCHY ORDERING (documented limitation). The emitted per-level loop interleaves
  /// assemble / solve / reconstruct per level, while a composite solve wants every level's coefficients
  /// assembled BEFORE it solves. The composite path is therefore correct for the coarse-only / flat
  /// layout the Program driver ships (the tested acceptance); a genuinely refined Schur Program (a real
  /// fine patch under a Program) needs the native AMR source-stage route (add_equation(Strang(source=
  /// CondensedSchur)), which assembles the whole tower then solves once) for a conservative, order-exact
  /// result. This branch is the composite-solve scaffold, not a bit-exact multilevel driver.
  void solve_linear_matfree(MultiFab& sol, const MultiFab& rhs, const ApplyFn& apply,
                          const ApplyFn& precond, int method, Real tol, int max_iter,
                          int restart) const {
    (void)restart;
    AmrSchurElliptic& s = schur();
    if (!s.has_fine_patches()) {
      switch (method) {
        case kLinearSolveCg:
          (void)pops::cg_solve(apply, sol, rhs, tol, max_iter);
          break;
        case kLinearSolveGmres:
          (void)pops::gmres_solve(apply, precond, sol, rhs, tol, max_iter, restart);
          break;
        case kLinearSolveRichardson:
          (void)pops::richardson_solve(apply, sol, rhs, static_cast<Real>(1), tol, max_iter);
          break;
        default:  // kLinearSolveBicgstab
          (void)pops::bicgstab_solve(apply, precond, sol, rhs, tol, max_iter);
          break;
      }
      return;
    }
    // Refined: the per-level coefficients / RHS are already assembled into AmrSchurElliptic (through
    // assembly_target on the prior per-level assembly calls); drive the composite FAC over the whole tower.
    s.solve_composite();
  }

  // --- named-flux primitive: DEFERRED on AMR (v1), fail loud -----------------------------------------
  // The named-flux divergence is a ProgramContext method the codegen can lower for a named-flux (ADC-419)
  // Program. The AMR named-flux -div path is NOT wired (out of ADC-633 scope); it fails loud so the SAME
  // lowered body compiles on target='amr_system' and throws only when the op is REACHED at run.
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
  /// Fail loud for an op the codegen can emit but the v1 AMR Program path does not yet wire (named-flux /
  /// scheduled Programs). [[noreturn]] so a non-void stub needs no dummy return -- the caller's signature
  /// stays byte-faithful to ProgramContext (the duck-typing requirement) without fabricating a value. @p
  /// op names the seam; @p detail names the alternative (System or the native AMR route) so the message
  /// is actionable, mirroring the inline fail-loud stubs above.
  [[noreturn]] static void deferred_op(const char* op, const char* detail) {
    throw std::runtime_error(std::string("AmrProgramContext: ") + op +
                             " is not wired on the AMR Program path (v1); " + detail);
  }

  /// The block-0 composite tensor-elliptic driver a condensed-Schur Program routes to on a REFINED
  /// hierarchy (ADC-633). Lazily created (a flat Program never touches it beyond has_fine_patches()).
  /// Held via shared_ptr so a copy of the context (the install closure captures ctx BY VALUE) SHARES
  /// the same per-Program elliptic driver -- AmrSchurElliptic owns a unique_ptr (move-only), so a bare
  /// value member would delete the context copy constructor the [=] install lambda needs.
  AmrSchurElliptic& schur() const {
    if (!schur_)
      schur_ = std::make_shared<AmrSchurElliptic>(eng_, sys_block(0));
    return *schur_;
  }

  AmrSystem* facade_;
  AmrRuntime* eng_;
  mutable int level_ = 0;
  mutable bool solved_this_step_ = false;
  mutable std::optional<GeometricMG> mg_precond_;
  mutable std::shared_ptr<AmrSchurElliptic> schur_;
};

}  // namespace program
}  // namespace runtime

// ADC-633: the former AmrProgramContext overloads of the condensed-Schur FREE kernels are GONE. The
// kernels (condensed_schur_operator.hpp) are now TEMPLATES on the context type Ctx, so they instantiate
// directly for AmrProgramContext -- reaching this context's grid_context() / assembly_target / assembly_source
// per level. No delegating overload is needed (and a non-templated one would ambiguate the template).
}  // namespace pops
