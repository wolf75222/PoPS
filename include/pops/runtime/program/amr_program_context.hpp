#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>     // Real, POPS_HD
#include <pops/coupling/schur/core/schur_condensation.hpp>  // Schur kernels + NegateKernel
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts
#include <pops/mesh/execution/for_each.hpp>    // for_each_cell, device_fence
#include <pops/mesh/storage/fab2d.hpp>         // Array4 / ConstArray4
#include <pops/mesh/geometry/geometry.hpp>     // Geometry
#include <pops/mesh/storage/mf_arith.hpp>      // saxpy / lincomb
#include <pops/mesh/storage/multifab.hpp>      // MultiFab
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>             // GeometricMG (Krylov precond)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>    // apply_laplacian
#include <pops/numerics/linalg/lorentz_eliminator.hpp>  // LorentzEliminator
#include <pops/runtime/amr/amr_runtime.hpp>     // AmrRuntime (the engine the driver wraps)
#include <pops/runtime/amr_system.hpp>          // AmrSystem (the facade: params / block map / engine)
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams
#include <pops/runtime/program/cache_manager.hpp>  // CacheManager (held-node value cache)
#include <pops/runtime/program/program_context.hpp>  // detail::Schur*KernelC shared with System ProgramContext

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
/// field re-solve at FINE levels, AMR history rings. GPU (CUDA) run is the ROMEO step (device-clean by
/// construction: every per-cell op is for_each_cell / a POPS_HD named functor reused from the engine).
namespace pops {
namespace runtime {
namespace program {

class AmrProgramContext {
 public:
  /// Wrap an AmrSystem passed as a flat void* (what pops_install_program_amr(void* sys) receives). The
  /// ctor pulls the AmrRuntime engine out of the facade (engine() returns the built runtime; the AMR
  /// blocks must be materialized -- install_problem forces the build before install()).
  explicit AmrProgramContext(void* sys)
      : facade_(static_cast<AmrSystem*>(sys)), eng_(facade_->engine()) {
    if (eng_ == nullptr)
      throw std::runtime_error(
          "AmrProgramContext: the AMR runtime engine is not built; install_problem must force the "
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
    count_kernel();
    eng_->project_level_state(static_cast<std::size_t>(sys_block(b)), level_, u);
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
  /// Named multi-elliptic field re-solve: routed through the AMR runtime's named-field solver.
  void solve_fields_from_state(const std::string& field, int b, MultiFab& u_stage) const {
    if (level_ == 0) {
      eng_->solve_named_fields_from_state(field, static_cast<std::size_t>(sys_block(b)), u_stage);
      solved_this_step_ = true;
    }
  }
  /// Coupled multi-block field solve over simultaneous stage states.
  void solve_fields_from_blocks(const std::vector<const MultiFab*>& u_stages) const {
    if (level_ != 0)
      return;
    const std::vector<int>& m = facade_->program_block_map();
    if (m.empty()) {
      eng_->solve_fields_from_blocks(u_stages);
    } else {
      std::vector<const MultiFab*> remapped(static_cast<std::size_t>(eng_->n_blocks()), nullptr);
      for (std::size_t p = 0; p < m.size() && p < u_stages.size(); ++p)
        remapped[static_cast<std::size_t>(m[p])] = u_stages[p];
      eng_->solve_fields_from_blocks(remapped);
    }
    solved_this_step_ = true;
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
  Real sum(const MultiFab& u) const { return pops::reduce_sum(u, 0); }
  Real max(const MultiFab& u) const { return pops::reduce_max(u, 0); }
  Real min(const MultiFab& u) const { return pops::reduce_min(u, 0); }

  void fill_boundary(MultiFab& x) const {
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(x, g.domain, eng_->transport_bc());
  }

  // --- history (closure-owned ring, same semantics as System HistoryManager) ------------------------
  void register_history(const std::string& name, int lag) const {
    if (lag < 1)
      throw std::runtime_error("AmrProgramContext::register_history: lag must be >= 1 for history '" +
                               name + "'");
    HistorySlot& h = histories_[name];
    h.depth = std::max(h.depth, lag + 1);
  }
  MultiFab& history(const std::string& name, int lag = 1) const {
    auto it = histories_.find(name);
    if (it == histories_.end())
      throw std::runtime_error("AmrProgramContext::history: unknown history '" + name +
                               "' (register it first)");
    HistorySlot& h = it->second;
    if (!h.initialized)
      throw std::runtime_error("AmrProgramContext::history: history '" + name +
                               "' was requested before its first store");
    if (lag < 0 || lag >= h.depth)
      throw std::runtime_error("AmrProgramContext::history: lag out of range for history '" + name +
                               "'");
    return h.ring[static_cast<std::size_t>(lag)];
  }
  void store_history(const std::string& name, const MultiFab& value) const {
    HistorySlot& h = histories_[name];
    if (h.depth < 2)
      h.depth = 2;
    ensure_history_allocated(h, value);
    lincomb(h.ring[0], Real(1), value, Real(0), value);
    if (!h.initialized) {
      for (std::size_t k = 1; k < h.ring.size(); ++k)
        lincomb(h.ring[k], Real(1), value, Real(0), value);
      h.initialized = true;
    }
  }
  void rotate_histories() const {
    for (auto& [name, h] : histories_) {
      (void)name;
      if (!h.initialized || h.ring.size() < 2)
        continue;
      for (std::size_t k = h.ring.size() - 1; k >= 1; --k) {
        std::swap(h.ring[k], h.ring[k - 1]);
        if (k == 1)
          break;
      }
    }
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

  // --- condensed-Schur / named-flux primitives over the CURRENT AMR level ---------------------------
  void assemble_schur_coeffs(MultiFab& eps_x, MultiFab& eps_y, MultiFab& a_xy, MultiFab& a_yx,
                             const MultiFab& state, Real c, Real th_dt, int c_rho,
                             int c_bz) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    const MultiFab& a = aux();
    for (int li = 0; li < eps_x.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = a.fab(li).const_array();
      for_each_cell(eps_x.box(li),
                    detail::SchurOperatorCoeffKernelC{s, b, eps_x.fab(li).array(),
                                                      eps_y.fab(li).array(), a_xy.fab(li).array(),
                                                      a_yx.fab(li).array(), c, th_dt, c_rho, c_bz});
    }
    const BCRec ebc = coeff_bc(eng_->transport_bc());
    fill_ghosts(eps_x, g.domain, ebc);
    fill_ghosts(eps_y, g.domain, ebc);
    fill_ghosts(a_xy, g.domain, ebc);
    fill_ghosts(a_yx, g.domain, ebc);
  }
  void apply_laplacian_coeff(MultiFab& out, MultiFab& in, const MultiFab& eps_x,
                             const MultiFab& eps_y, const MultiFab& a_xy,
                             const MultiFab& a_yx) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(in, g.domain, eng_->transport_bc());
    apply_laplacian(in, g, out, /*coef=*/nullptr, /*eps=*/&eps_x, /*kappa=*/nullptr,
                    /*eps_y=*/&eps_y, /*a_xy=*/&a_xy, /*a_yx=*/&a_yx);
  }
  void schur_explicit_flux(MultiFab& out, const MultiFab& state, Real th_dt, int c_mx, int c_my,
                           int c_bz) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    const MultiFab& a = aux();
    for (int li = 0; li < out.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = a.fab(li).const_array();
      for_each_cell(out.box(li),
                    detail::SchurExplicitFluxKernelC{s, b, out.fab(li).array(), th_dt, c_mx,
                                                     c_my, c_bz});
    }
    fill_ghosts(out, g.domain, coeff_bc(eng_->transport_bc()));
  }
  void assemble_schur_rhs(MultiFab& rhs, MultiFab& phi_n, const MultiFab& state, Real th_dt,
                          Real gcoef, int c_mx, int c_my, int c_bz) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    const MultiFab& a = aux();
    const BoxArray& ba = rhs.box_array();
    const DistributionMapping& dm = rhs.dmap();
    fill_ghosts(phi_n, g.domain, eng_->transport_bc());
    MultiFab lap(ba, dm, 1, 0);
    apply_laplacian(phi_n, g, lap);
    MultiFab neg_lap(ba, dm, 1, 0);
    for (int li = 0; li < neg_lap.local_size(); ++li)
      for_each_cell(neg_lap.box(li),
                    pops::detail::NegateKernel{lap.fab(li).const_array(), neg_lap.fab(li).array()});
    MultiFab fx(ba, dm, 2, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = a.fab(li).const_array();
      for_each_cell(fx.box(li), detail::SchurExplicitFluxKernelC{s, b, fx.fab(li).array(), th_dt,
                                                                 c_mx, c_my, c_bz});
    }
    fill_ghosts(fx, g.domain, coeff_bc(eng_->transport_bc()));
    const Real half_idx = Real(1) / (Real(2) * g.dx());
    const Real half_idy = Real(1) / (Real(2) * g.dy());
    for (int li = 0; li < rhs.local_size(); ++li)
      for_each_cell(rhs.box(li), detail::SchurRhsAssembleKernelC{
                                     neg_lap.fab(li).const_array(), fx.fab(li).const_array(),
                                     rhs.fab(li).array(), gcoef, half_idx, half_idy});
  }
  void schur_reconstruct(MultiFab& state, MultiFab& phi, Real th_dt, int c_rho, int c_mx,
                         int c_my, int c_bz) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    const MultiFab& a = aux();
    fill_ghosts(phi, g.domain, eng_->transport_bc());
    const Real half_idx = Real(1) / (Real(2) * g.dx());
    const Real half_idy = Real(1) / (Real(2) * g.dy());
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 ph = phi.fab(li).const_array();
      const ConstArray4 b = a.fab(li).const_array();
      Array4 st = state.fab(li).array();
      for_each_cell(state.box(li),
                    detail::SchurReconstructKernelC{ph, b, st, th_dt, half_idx, half_idy, c_rho,
                                                    c_mx, c_my, c_bz});
    }
  }
  void schur_energy(MultiFab& state, const MultiFab& state_old, int c_rho, int c_mx, int c_my,
                    int c_E) const {
    count_kernel();
    for (int li = 0; li < state.local_size(); ++li) {
      Array4 st = state.fab(li).array();
      const ConstArray4 so = state_old.fab(li).const_array();
      for_each_cell(state.box(li), detail::SchurEnergyKernelC{st, so, c_rho, c_mx, c_my, c_E});
    }
  }
  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(fx, g.domain, eng_->transport_bc());
    fill_ghosts(fy, g.domain, eng_->transport_bc());
    MultiFab divc(r.box_array(), r.dmap(), 1, 0);
    for (int c = 0; c < r.ncomp(); ++c) {
      apply_divergence(fx, fy, g, divc, /*cx=*/c, /*cy=*/c);
      for (int li = 0; li < r.local_size(); ++li) {
        const ConstArray4 d = divc.fab(li).const_array();
        Array4 rv = r.fab(li).array();
        const int comp = c;
        for_each_cell(r.box(li), [=] POPS_HD(int i, int j) { rv(i, j, comp) = -d(i, j, 0); });
      }
    }
  }

  // --- scheduler value cache ------------------------------------------------------------------------
  bool cache_should_update(int node_id, int every_n) const {
    const bool due = cache_.is_due(node_id, macro_step(), every_n);
    if (due) {
      facade_->profiler_handle().count("cache_misses");
      facade_->profiler_handle().count("nodes_due");
    } else {
      facade_->profiler_handle().count("cache_hits");
      facade_->profiler_handle().count("nodes_skipped");
    }
    return due;
  }
  void cache_store_aux(int node_id) const {
    cache_.store(node_id, aux(), macro_step());
  }
  void cache_restore_aux(int node_id) const {
    aux() = cache_.retrieve(node_id);
  }
  void cache_store_scratch(int node_id, const MultiFab& scratch) const {
    cache_.store(node_id, scratch, macro_step());
  }
  void cache_restore_scratch(int node_id, MultiFab& scratch) const {
    scratch = cache_.retrieve(node_id);
  }
  void cache_accumulate_dt(int node_id, Real dt) const {
    cache_.accumulate_dt(node_id, dt);
  }
  Real cache_effective_dt(int node_id, Real dt_now) const {
    return cache_.effective_dt(node_id, dt_now);
  }
  /// Fail loud: an `error`-policy scheduled node reached off cadence. ProgramContext throws here; the
  /// AMR path never installs a schedule (the cache seams above fail loud first), so this only fires if
  /// the body reaches the off-cadence branch directly. Keep the exact [[noreturn]] signature.
  [[noreturn]] void scheduler_error(const std::string& what) const {
    throw std::runtime_error("pops Program scheduler (AMR): " + what);
  }

 private:
  struct HistorySlot {
    int depth = 0;
    bool initialized = false;
    std::vector<MultiFab> ring;
  };
  static void ensure_history_allocated(HistorySlot& h, const MultiFab& tmpl) {
    if (!h.ring.empty())
      return;
    h.ring.reserve(static_cast<std::size_t>(h.depth));
    for (int k = 0; k < h.depth; ++k) {
      MultiFab slot(tmpl.box_array(), tmpl.dmap(), tmpl.ncomp(), tmpl.n_grow());
      slot.set_val(Real(0));
      h.ring.push_back(std::move(slot));
    }
  }
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo);
    b.xhi = fo(bc.xhi);
    b.ylo = fo(bc.ylo);
    b.yhi = fo(bc.yhi);
    return b;
  }

  AmrSystem* facade_;
  AmrRuntime* eng_;
  mutable int level_ = 0;
  mutable bool solved_this_step_ = false;
  mutable std::optional<GeometricMG> mg_precond_;
  mutable CacheManager cache_;
  mutable std::map<std::string, HistorySlot> histories_;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
