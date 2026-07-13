#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
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
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_ledger.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>     // AmrRuntime (the engine the driver wraps)
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>  // AmrTensorElliptic composite provider
#include <pops/runtime/context/grid_context.hpp>  // GridContext (per-level Schur assembly seam, ADC-633)
#include <pops/runtime/amr_system.hpp>          // AmrSystem (the facade: params / block map / engine)
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams
#include <pops/runtime/program/wire_ids.hpp>       // stable compiled-Program numeric protocol

/// @file
/// @brief AmrProgramContext -- the AMR counterpart of ProgramContext (epic ADC-508, Spec 6).
///
/// A compiled time Program lowers its macro-step body referencing ONLY the variable `ctx` (never the
/// type ProgramContext). The SAME generated body therefore compiles against any object exposing
/// ProgramContext's method surface. AmrProgramContext is that duck-typed structural mirror, driving the
/// lowered body on explicit parent/child level clocks over the AMR hierarchy. The `{amr_install}` slot
/// installs one recursive Berger-Oliger driver: child steps partition the parent window, each rate reads
/// a mandatory old/new dense-output interpolation at its exact Program abscissa, and level sync is
/// conservative reflux followed by average-down. The single coarse system Poisson per macro-step
/// (OncePerStep) is injected coarse -> fine; unsupported per-stage fine re-solves fail loudly. Multistep
/// history rings (keep_history / T.prev) are owner/space/clock-qualified; their per-level slots are
/// remapped through regrid and v3 checkpoint native replay. GPU execution stays device-clean by
/// construction: every per-cell op is for_each_cell / a POPS_HD named functor reused from the engine.
namespace pops {
namespace runtime {
namespace program {

class AmrProgramContext {
 public:
  enum class SyncPhase { Reflux, AverageDown };
  struct SyncEvent {
    int parent_level = 0;
    int child_level = 0;
    int block = 0;
    SyncPhase phase = SyncPhase::Reflux;
  };
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
  bool has_refined_hierarchy() const { return tensor_elliptic().has_fine_patches(); }
  /// Reset the per-macro-step flags (called by the install wrapper at the top of each macro-step). Also
  /// clears the per-step effective-flux ledger + the live-state-ring record (ADC-639); the PERSISTENT
  /// per-ring flux strips (ring_flux_) survive across steps, as the multistep ring itself does.
  void reset_step() const {
    solved_this_step_ = false;
    flux_ledger_.clear();
    rate_provenance_.clear();
    synchronized_flux_.clear();
    sync_report_.clear();
    live_state_rings_.clear();
  }
  /// True iff a genuine coarse-fine interface exists (nlev > 1): the reflux capture path activates. On a
  /// coarse-only / flat Program (nlev == 1) this is false, the capture code is never reached, and the
  /// trajectory is byte-identical to System / Uniform (the load-bearing bit-parity gate).
  bool capturing() const { return eng_->nlev() > 1; }
  /// Number of populated ledger entries (test seam): the parity gate unit-asserts this is 0 on a
  /// coarse-only / flat Program (the capture path is unreachable at nlev == 1).
  std::size_t ledger_size() const { return flux_ledger_.size(); }
  const std::vector<SyncEvent>& sync_report() const { return sync_report_; }

  /// Execute one recursively subcycled hierarchy attempt.  The body is the exact lowered Program body;
  /// it is evaluated once on the parent and once per declared child clock interval.  Context-owned
  /// clocks, history-flux publications and the conservative ledger form one nested transaction so a
  /// StepAttemptRejected cannot leave residue for the retry.
  template <class Body>
  void advance_hierarchy(double dt, Body&& body) const {
    advance_attempt_(dt, "AmrProgramContext::advance_hierarchy",
                     [&](const amr::ClockWindow& root) { advance_level_(0, root, body); });
  }

  /// Execute one hierarchy-wide Program body inside the same accepted-step transaction as the
  /// recursively subcycled driver. This scheduler is reserved for operators whose authored scope is
  /// the complete hierarchy: every level gathers, one global solve crosses the hierarchy barrier, and
  /// every level publishes before conservative synchronization. Replaying that body at child substeps
  /// would silently change a hierarchy-scoped operator into a level operator, so it is forbidden here.
  template <class Body>
  void advance_synchronized_hierarchy(double dt, Body&& body) const {
    advance_attempt_(dt, "AmrProgramContext::advance_synchronized_hierarchy",
                     [&](const amr::ClockWindow& root) {
                       current_window_ = root;
                       current_level_dt_ = dt;
                       active_parent_.reset();
                       stage_time_ = amr::Rational(0, 1);
                       body(dt);
                       for (int k = 0; k < nlev(); ++k) {
                         const amr::ClockStamp accepted{k, root.end.macro_step, root.end.phase,
                                                        root.end.physical_time};
                         flush_level_flux_(k, dt, accepted);
                       }
                     });
  }

  /// Exact local abscissa of the next rate evaluation, emitted from the Program's TimePoint.
  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    stage_time_ = amr::Rational(numerator, denominator);
    if (stage_time_ < amr::Rational(0, 1) || amr::Rational(1, 1) < stage_time_)
      throw std::runtime_error("Program stage time is outside [0,1]");
  }
  ClockScheduleState::SubcycleScope subcycle_scope(
      const std::string& parent, const std::string& child, int count) const {
    return clock_schedule_.subcycle(parent, child, count);
  }
  void synchronize_sample_and_hold(const std::string& source, const std::string& target,
                                   int step, Real offset) const {
    clock_schedule_.synchronize_sample_and_hold(source, target, step, static_cast<double>(offset));
  }

  /// Register the macro-step body (forwards to AmrSystem::install_program_step). @p step is the per-level
  /// loop wrapper the codegen emits; it runs ONE macro-step over dt.
  void install(std::function<void(double)> step) const {
    ensure_level_clocks_();
    if (facade_->program_accepted_state().empty())
      publish_program_accepted_state_();
    facade_->install_program_step(std::move(step));
  }

  /// Translate a PROGRAM block index to its explicit, name-matched AMR block index (Spec 3 criterion
  /// 23). Empty, incomplete and invalid maps fail before any hierarchy storage is accessed; positional
  /// block identity is never inferred.
  int sys_block(int b) const {
    const std::vector<int>& m = facade_->program_block_map();
    if (m.empty())
      throw block_map_error_(
          "AmrProgramContext::sys_block: no explicit program-to-AMR block map is installed; "
          "positional block identity is not supported");
    if (b < 0 || b >= static_cast<int>(m.size()))
      throw block_map_error_(
          "AmrProgramContext::sys_block: program block index " + std::to_string(b) +
          " is outside the explicit block map [0, " + std::to_string(m.size()) + ")");
    const int mapped = m[static_cast<std::size_t>(b)];
    const int count = static_cast<int>(eng_->n_blocks());
    if (mapped < 0 || mapped >= count)
      throw block_map_error_(
          "AmrProgramContext::sys_block: program block index " + std::to_string(b) +
          " maps to invalid AMR block index " + std::to_string(mapped) +
          " for an AmrRuntime with " + std::to_string(count) + " blocks");
    return mapped;
  }
  int n_blocks() const { return static_cast<int>(eng_->n_blocks()); }
  Real physical_time() const { return static_cast<Real>(facade_->time()); }
  void set_field_logical_timepoint(const std::string& field,
                                   const FieldLogicalTimePoint& point) const {
    facade_->set_field_logical_timepoint(field, point);
  }
  void set_field_boundary_parameters(const std::string& field,
                                     const std::vector<double>& parameters) const {
    facade_->set_field_boundary_parameters(field, parameters);
  }
  void set_field_boundary_kernel(const std::string& field,
                                 const CompiledFieldBoundaryKernel& kernel) const {
    facade_->set_field_boundary_kernel(field, kernel);
  }

  // --- inter-level coupling -------------------------------------------------------------------------
  /// fine -> coarse reflux then average_down over ALL blocks, followed by a fresh
  /// coarse Poisson + injection so every level's aux is consistent for the next macro-step.
  ///
  /// CONSERVATIVE C/F COUPLING (ADC-639). Per interface, finest first: conservative reflux routes the
  /// effective-flux mismatch into bordering coarse cells, THEN average_down overwrites covered cells.
  /// The effective flux at each level was captured through the Program's own
  /// linear combination (the flux ledger, section 2b), so the coarse cell's flux at the C/F interface is
  /// corrected by exactly (fine effective flux - coarse effective flux) -- the total conserved quantity
  /// (mass / momentum / energy) is now conserved across the interface to ROUND-OFF, matching the native
  /// reflux. After the reflux, ADC-631 history rings stay consistent: slot 0 of any ring that stored the
  /// COMMITTED live state (no live-state write followed the store) is re-synced from the now-corrected
  /// state, then the deferred macro-step rotate fires (it was withheld from the body so the reflux runs
  /// against the right live state). A ring storing the RHS (AB2) or a pre-commit state (BDF2) is untouched.
  /// On a
  /// coarse-only / flat Program (nlev == 1) the loop is empty, the ledger is empty, and the deferred
  /// rotate collapses to the original store->rotate order -> the trajectory is bit-identical (the parity
  /// gate). Sync is reported by its fixed phase order: reflux first, average-down second.
  void couple_levels() const {
    for (int k = nlev() - 1; k >= 1; --k)
      for (int b = 0; b < n_blocks(); ++b) {
        const std::size_t sb = static_cast<std::size_t>(sys_block(b));
        if (capturing()) {
          sync_report_.push_back({k - 1, k, b, SyncPhase::Reflux});
          const EdgeFlux& coarse_role = synchronized_flux_at_(b, k - 1);
          const EdgeFlux& fine_role = synchronized_flux_at_(b, k);
          pops::detail::route_reflux_program(*eng_, sb, k, coarse_role, fine_role);
        }
        sync_report_.push_back({k - 1, k, b, SyncPhase::AverageDown});
        eng_->average_down_level(sb, k);
      }
    if (capturing() && rotate_pending_) {
      resync_history_slot0_();  // re-copy the corrected live state into any live-state ring's slot 0
      rotate_ring_flux_();      // rotate the persistent flux strips in lockstep with the ring
      rotate_ring_clocks_();
      pops::detail::AmrHistoryOps::rotate_histories(*eng_);
      rotate_pending_ = false;
    }
  }
  /// Head-of-step regrid at the engine's cadence (the SAME union-tags regrid the native step runs).
  void regrid_if_due(int macro_step) const { eng_->regrid_if_due(macro_step); }

  // --- state / RHS seam over the CURRENT level -------------------------------------------------------
  MultiFab& state(int b) const {
    return eng_->level_state(static_cast<std::size_t>(sys_block(b)), level_);
  }
  void rhs_into(int b, MultiFab& u, MultiFab& r, int rate_id = -1) const {
    count_kernel();
    if (capturing()) {
      capture_into_(b, u, r, ResidualCapture::FullRate, rate_id);
      return;
    }
    eng_->level_rhs_into(static_cast<std::size_t>(sys_block(b)), level_, u, r);
  }
  void neg_div_flux_default_into(int b, MultiFab& u, MultiFab& r, int rate_id = -1) const {
    count_kernel();
    if (capturing()) {
      capture_into_(b, u, r, ResidualCapture::FluxOnly, rate_id);
      return;
    }
    eng_->level_neg_div_flux_into(static_cast<std::size_t>(sys_block(b)), level_, u, r);
  }
  void source_default_into(int b, MultiFab& u, MultiFab& r) const {
    count_kernel();
    eng_->level_source_into(static_cast<std::size_t>(sys_block(b)), level_, u, r);
  }
  void apply_projection(int b, MultiFab& u) const {
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
  SolveReport solve_fields() const {
    if (level_ == 0 || !solved_this_step_) {
      eng_->solve_fields();
      solved_this_step_ = true;
    }
    SolveReport report;
    report.mark_solved();
    return report;
  }
  /// Per-stage re-solve from a stage state is currently a coarse-only capability.  A fine-level request
  /// is rejected explicitly; it never consumes a stale injected auxiliary field.
  SolveReport solve_fields_from_state(int b, MultiFab& u_stage) const {
    if (level_ == 0) {
      MultiFab& live = state(b);
      MultiFab saved = live;          // stash the live state
      live = u_stage;                 // solve from the stage state
      eng_->solve_fields();
      live = std::move(saved);        // restore the live state (the commit owns it)
      solved_this_step_ = true;
      SolveReport report;
      report.mark_solved();
      return report;
    }
    throw std::runtime_error(
        "AmrProgramContext::solve_fields_from_state: per-stage fine-level field re-solve is not "
        "implemented; use OncePerStep field cadence or provide a composite stage solver");
  }
  /// Named multi-elliptic field re-solve: deferred on AMR (ADC-513 companion). Fail loud.
  SolveReport solve_fields_from_state(const std::string& field, int b,
                                      MultiFab& u_stage) const {
    if (level_ != 0) {
      SolveReport report;
      report.mark_solved();
      return report;  // the coarse solve publishes/injects every level once per stage
    }
    (void)eng_->provider_potential(field);  // authenticate the exact resolved field route
    MultiFab& live = state(b);
    MultiFab published = live;
    SolveReport report;
    try {
      live = u_stage;
      report = eng_->solve_named_fields(&field);
      live = std::move(published);
    } catch (...) {
      live = std::move(published);
      throw;
    }
    if (report.solved())
      solved_this_step_ = true;
    return report;
  }
  /// Coupled multi-block field solve: deferred on AMR (the per-block summed-RHS at fine levels needs the
  /// coupled re-solve path). Fail loud rather than a silent half-implementation.
  SolveReport solve_fields_from_blocks(
      const std::vector<const MultiFab*>& /*u_stages*/) const {
    throw std::runtime_error(
        "AmrProgramContext::solve_fields_from_blocks: a coupled multi-block field solve under a "
        "compiled Program on AMR is deferred (v1, Spec 3 criterion 24). Use System, or a single-block "
        "AMR Program.");
  }

  SolveReport solve_fields_from_blocks(const std::string& field,
                                       const std::vector<const MultiFab*>& u_stages) const {
    if (level_ != 0) {
      SolveReport report;
      report.mark_solved();
      return report;
    }
    if (u_stages.size() != static_cast<std::size_t>(n_blocks()))
      throw std::runtime_error(
          "AmrProgramContext::solve_fields_from_blocks(field): stage vector size mismatch");
    (void)eng_->provider_potential(field);
    std::vector<MultiFab*> live;
    std::vector<MultiFab> published;
    live.reserve(u_stages.size());
    published.reserve(u_stages.size());
    for (std::size_t p = 0; p < u_stages.size(); ++p) {
      if (u_stages[p] == nullptr)
        continue;
      MultiFab& state_value = state(static_cast<int>(p));
      live.push_back(&state_value);
      published.push_back(state_value);
      state_value = *u_stages[p];
    }
    auto restore = [&]() {
      for (std::size_t i = 0; i < live.size(); ++i)
        *live[i] = std::move(published[i]);
    };
    SolveReport report;
    try {
      report = eng_->solve_named_fields(&field);
      restore();
    } catch (...) {
      restore();
      throw;
    }
    if (report.solved())
      solved_this_step_ = true;
    return report;
  }

  /// The SHARED aux of the current level (phi / grad / B_z), the channel solve_fields fills.
  MultiFab& aux() const { return const_cast<MultiFab&>(eng_->aux(level_)); }
  /// The current level's metric (dx/dy >> level, domain << level).
  Geometry geom() const { return eng_->level_geom(level_); }
  /// AMR v1 is Cartesian.  The metric queries still exist on this context so one generated Program
  /// body instantiates against both runtime contexts without geometry-specific source-stage classes.
  bool is_polar_geometry() const { return false; }
  Real radial_origin() const { return Real(0); }
  Real radial_spacing() const { return geom().dx(); }

  /// The grid context of the CURRENT level (ADC-633): the AMR counterpart of System::grid_context(),
  /// per level. It bundles the transport BC + the level geometry + the live level aux pointer, exactly
  /// what System::grid_context() returns for the uniform mesh. Used by the emitted condensed-implicit
  /// assembly kernels so their per-cell assembly reads the CURRENT level's geom / aux / BC as direct
  /// body calls (they read the level_ cursor live). transport_bc() (not poisson_bc()) matches the
  /// uniform Program's gc.bc, so the flat-hierarchy phi-ghost fill is byte-identical to the uniform
  /// Program (the flat bit-parity gate). BY VALUE, like ProgramContext.
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
    if (capturing()) {
      ledger_axpy_(u, a, r);  // shadow the state combine on the effective-flux strip: ledger[u] += a*ledger[r]
      note_live_write_(&u);   // a write to the live state invalidates any earlier live-state ring snapshot
    }
  }
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    count_kernel();
    pops::lincomb(z, a, x, b, y);
    if (capturing()) {
      ledger_lincomb_(z, a, x, b, y);  // ledger[z] = a*ledger[x] + b*ledger[y]
      note_live_write_(&z);
    }
  }
  void commit_many(
      std::initializer_list<std::pair<MultiFab*, const MultiFab*>> commits) const {
    std::vector<MultiFab*> targets;
    targets.reserve(commits.size());
    for (const auto& [target, source] : commits) {
      if (target == nullptr || source == nullptr)
        throw std::invalid_argument("AmrProgramContext::commit_many received a null state");
      if (std::find(targets.begin(), targets.end(), target) != targets.end())
        throw std::invalid_argument("AmrProgramContext::commit_many received a duplicate target");
      if (target->ncomp() != source->ncomp() ||
          target->box_array().boxes() != source->box_array().boxes())
        throw std::invalid_argument("AmrProgramContext::commit_many state layout mismatch");
      targets.push_back(target);
    }
    for (const auto& [target, source] : commits)
      if (target != source)
        lincomb(*target, Real(0), *target, Real(1), *source);
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
  // register/read/store address the CURRENT level; rotate fires ONCE per accepted hierarchy step. The
  // recursively invoked body requests rotation after each local advance, and the context defers the
  // actual hierarchy-wide rotation until synchronization has corrected the committed states.
  // @p ncomp mirrors ProgramContext::register_history so the SAME lowered body (a single problem.so)
  // compiles against BOTH contexts. The narrow-ring AMR phi^n carry (ADC-427) threads @p ncomp into
  // AmrHistoryOps: ncomp < 0 uses the owner-qualified program block's width; an
  // explicit ncomp >= 1 (the 1-component condensed-Schur phi^n carry) narrows the per-level ring, which
  // rides the same alloc / remap / replay machinery (each slot is sized by ncomp internally).
  void register_history(const std::string& name, int lag, int ncomp, int owner,
                        const std::string& state_identity,
                        const std::string& space_identity) const {
    register_history(name, lag, ncomp, owner, state_identity, space_identity,
                     "legacy:macro", "exact");
  }
  void register_history(const std::string& name, int lag, int ncomp, int owner,
                        const std::string& state_identity,
                        const std::string& space_identity,
                        const std::string& clock_identity,
                        const std::string& interpolation_identity) const {
    if (state_identity.empty() || space_identity.empty())
      throw std::runtime_error("AMR history requires qualified state and space identities");
    if (clock_identity.empty() || interpolation_identity.empty())
      throw std::runtime_error(
          "AMR history requires qualified logical-clock and interpolation identities");
    const auto prior_owner = history_owners_.find(name);
    const auto prior_state = history_state_ids_.find(name);
    const auto prior_space = history_space_ids_.find(name);
    const auto prior_clock = history_clock_ids_.find(name);
    const auto prior_interpolation = history_interpolation_ids_.find(name);
    if ((prior_owner != history_owners_.end() && prior_owner->second != owner) ||
        (prior_state != history_state_ids_.end() && prior_state->second != state_identity) ||
        (prior_space != history_space_ids_.end() && prior_space->second != space_identity) ||
        (prior_clock != history_clock_ids_.end() && prior_clock->second != clock_identity) ||
        (prior_interpolation != history_interpolation_ids_.end() &&
         prior_interpolation->second != interpolation_identity))
      throw std::runtime_error("AMR history '" + name +
                               "' cannot be re-registered with a different identity");
    pops::detail::AmrHistoryOps::register_history(
        *eng_, static_cast<std::size_t>(sys_block(owner)), name, lag, ncomp);
    history_owners_[name] = owner;
    history_state_ids_[name] = state_identity;
    history_space_ids_[name] = space_identity;
    history_clock_ids_[name] = clock_identity;
    history_interpolation_ids_[name] = interpolation_identity;
  }
  MultiFab& history(const std::string& name, int lag, int owner) const {
    require_history_owner_(name, owner);
    validate_history_clock_(name, lag);
    MultiFab& mf = pops::detail::AmrHistoryOps::read_history(*eng_, name, lag, level_);
    if (capturing())
      restore_ring_flux_(name, lag, mf);  // re-publish the lagged buffer's flux strip into the live ledger
    return mf;
  }
  // ZERO COLD-START read (ADC-427), mirroring ProgramContext::history_zero_start so the SAME lowered
  // body compiles on both contexts: a read-first cross-step carry reads the zero-filled slots on its
  // very first read instead of failing loud. @p ncomp binds the ring width at the first register (the
  // codegen prelude locks it before any read), exactly like register_history above: ncomp < 0 keeps
  // the owner-qualified block width; explicit ncomp >= 1 narrows the ring.
  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp, int owner) const {
    require_history_owner_(name, owner);
    if (history_state_ids_.find(name) == history_state_ids_.end() ||
        history_space_ids_.find(name) == history_space_ids_.end())
      throw std::runtime_error("AMR history '" + name + "' has no registered state/space identity");
    pops::detail::AmrHistoryOps::register_history(
        *eng_, static_cast<std::size_t>(sys_block(owner)), name, lag, ncomp);
    if (!pops::detail::AmrHistoryOps::initialized(*eng_, name))
      pops::detail::AmrHistoryOps::set_initialized(*eng_, name, true);
    MultiFab& mf = pops::detail::AmrHistoryOps::read_history(*eng_, name, lag, level_);
    if (capturing())
      restore_ring_flux_(name, lag, mf);
    return mf;
  }
  void store_history(const std::string& name, const MultiFab& value, int owner) const {
    require_history_owner_(name, owner);
    pops::detail::AmrHistoryOps::store_history(*eng_, name, level_, value,
                                               static_cast<Real>(current_level_dt_));
    record_history_clock_(name);
    if (capturing())
      save_ring_flux_(name, value, owner);
  }
  void rotate_histories() const {
    // ADC-631/639: DEFER the rotate. The body's terminal rotate fires inside the recursive advance, before
    // couple_levels reflux touches the coarse live state; deferring it to couple_levels keeps a multistep
    // Program's lag read consistent with the refluxed live state. On nlev==1 (or no reflux) the deferral
    // collapses to the original store->rotate order -> bit-identical. Guarded to the last level like v1.
    if (level_ != nlev() - 1)
      return;
    if (capturing()) {
      rotate_pending_ = true;  // couple_levels executes it after the reflux + slot-0 resync
      return;
    }
    pops::detail::AmrHistoryOps::rotate_histories(*eng_);
  }
  void rotate_histories(const std::string& clock_identity) const {
    if (clock_identity.empty())
      throw std::runtime_error("AMR history rotation requires a logical-clock identity");
    bool found = false;
    for (const auto& [name, identity] : history_clock_ids_)
      if (identity == clock_identity)
        found = true;
    if (!found)
      return;
    for (const auto& [name, identity] : history_clock_ids_)
      if (identity != clock_identity)
        throw std::runtime_error(
            "AMR selective history rotation cannot mix logical clocks in one hierarchy step");
    rotate_histories();
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

  // --- condensed-implicit elliptic primitives on the hierarchy (ADC-633 / ADC-637): WIRED per level ---
  // The codegen lowers a condensed-implicit (ADC-637) Program to inline block-inverse assembly kernels
  // referencing ONLY the variable `ctx`, so the SAME emitted body compiles against this context. With its
  // grid_context() (per level) + assembly_target / assembly_source (write/read redirect), those kernels
  // run the SAME assembly PER LEVEL as direct body calls (they read the level_ cursor live). On a FLAT
  // hierarchy the emitted matrix-free BiCGStab runs on level 0, bit-identical to the uniform Program; on a
  // REFINED hierarchy the tensor elliptic is solved compositely (AmrTensorElliptic + CompositeFacPoisson).
  // No coupling/schur call remains on any path -- the generated .so carries all scheme kernels.

  /// Assembly WRITE redirection (ADC-633). On a REFINED hierarchy each assembled coefficient / RHS / flux
  /// field must live on the CURRENT level, not the level-0-bound emitted scratch: the kernel writes
  /// THROUGH here into AmrTensorElliptic's per-level buffer. On a FLAT hierarchy (no fine patch) the
  /// emitted level-0 field IS the whole system, so this is the identity (byte-for-byte the uniform path --
  /// the flat bit-parity gate). @p role is an AssemblyFieldRole (eps_x / eps_y / a_xy / a_yx / rhs / flux).
  MultiFab& assembly_target(MultiFab& field, int role) const {
    validate_assembly_write_role(role, "AmrProgramContext::assembly_target");
    AmrTensorElliptic& s = tensor_elliptic();
    if (!s.has_fine_patches())
      return field;  // flat / no fine patch: the emitted level-0 field is correct as-is.
    return s.target(role, level_);
  }
  /// Reconstruction READ redirection (ADC-633): the fine-level reconstruction reads the level's published
  /// composite potential (the emitted level-0 solution cannot hold a fine level's phi). Flat / no fine
  /// patch: identity (returns the emitted solution). @p role is kPhi.
  MultiFab& assembly_source(MultiFab& field, int role) const {
    validate_assembly_read_role(role, "AmrProgramContext::assembly_source");
    AmrTensorElliptic& s = tensor_elliptic();
    if (!s.has_fine_patches())
      return field;
    return s.phi(level_);
  }
  /// Resolve a hierarchy-scoped solve value for the current publish/reconstruct pass.  Flat AMR is
  /// the identity; a refined hierarchy returns the level solution published by the one composite solve.
  MultiFab& linear_solution(MultiFab& field) const {
    AmrTensorElliptic& s = tensor_elliptic();
    return s.has_fine_patches() ? s.phi(level_) : field;
  }
  /// Gather the authored initial guess for the current hierarchy level.  The no-argument overload is
  /// the explicit zero-guess contract; the field overload carries a per-level scalar history such as
  /// condensed-Schur phi^n.  Emitted only inside the refined gather loop.
  void stage_linear_initial_guess() const { tensor_elliptic().stage_initial_guess(level_, nullptr); }
  void stage_linear_initial_guess(const MultiFab& guess) const {
    tensor_elliptic().stage_initial_guess(level_, &guess);
  }
  /// Solve the matrix-free condensed-implicit linear system A(phi) = rhs on the hierarchy (ADC-633).
  /// FLAT (no fine patch): the SAME matrix-free Krylov call as the uniform Program (identical numerics,
  /// the flat bit-parity path -- the load-bearing acceptance). REFINED (>= one fine patch): drive
  /// AmrTensorElliptic::solve_composite (the composite FAC over the tower), which reads the per-level
  /// coefficients / RHS the emitted assembly already wrote through assembly_target and publishes each
  /// level's potential for assembly_source to read; the emitted @p apply / @p precond are UNUSED on this
  /// branch (the FAC has its own operator). @p method is a LinearSolveMethod id (program_context.hpp).
  ///
  /// On a refined hierarchy the generated driver gathers coefficients, RHS and the authored initial
  /// guess for every level before calling this function once; only a successful SolveReport publishes
  /// the per-level potentials, after which reconstruction runs over every level and coupling/reflux
  /// closes the macro-step.
  SolveReport solve_linear_matfree(MultiFab& sol, const MultiFab& rhs, const ApplyFn& apply,
                                   const ApplyFn& precond, int method, Real tol, int max_iter,
                                   int restart, Real omega) const {
    validate_linear_solve_method(method, "AmrProgramContext::solve_linear_matfree");
    AmrTensorElliptic& s = tensor_elliptic();
    if (!s.has_fine_patches()) {
      switch (method) {
        case kLinearSolveCg:
          return pops::cg_solve(apply, sol, rhs, tol, max_iter);
        case kLinearSolveGmres:
          return pops::gmres_solve(apply, precond, sol, rhs, tol, max_iter, restart);
        case kLinearSolveRichardson:
          return pops::richardson_solve(apply, sol, rhs, omega, tol, max_iter);
        case kLinearSolveBicgstab:
          return pops::bicgstab_solve(apply, precond, sol, rhs, tol, max_iter);
        default:
          SolveReport report;
          report.mark_failed(SolveStatus::kInvalidInput, SolveAction::kRejectAttempt);
          return report;  // validated above
      }
    }
    // Refined: the hierarchy driver calls this seam once, after every level has assembled.  FAC publishes
    // every level before the reconstruct pass starts, so no level can observe a partial solution.
    return s.solve_composite(tol, max_iter);
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
  enum class ResidualCapture { FullRate, FluxOnly };
  static std::runtime_error block_map_error_(std::string message) {
    return std::runtime_error(std::move(message));
  }

  /// Fail loud for an op the codegen can emit but the v1 AMR Program path does not yet wire (named-flux /
  /// scheduled Programs). [[noreturn]] so a non-void stub needs no dummy return -- the caller's signature
  /// stays byte-faithful to ProgramContext (the duck-typing requirement) without fabricating a value. @p
  /// op names the seam; @p detail names the alternative (System or the native AMR route) so the message
  /// is actionable, mirroring the inline fail-loud stubs above.
  [[noreturn]] static void deferred_op(const char* op, const char* detail) {
    throw std::runtime_error(std::string("AmrProgramContext: ") + op +
                             " is not wired on the AMR Program path (v1); " + detail);
  }

  /// The block-0 composite tensor-elliptic driver a condensed-implicit Program routes to on a REFINED
  /// hierarchy (ADC-633). Lazily created (a flat Program never touches it beyond has_fine_patches()).
  /// Held via shared_ptr so a copy of the context (the install closure captures ctx BY VALUE) SHARES
  /// the same per-Program elliptic driver -- AmrTensorElliptic owns a unique_ptr (move-only), so a bare
  /// value member would delete the context copy constructor the [=] install lambda needs.
  AmrTensorElliptic& tensor_elliptic() const {
    if (!tensor_elliptic_)
      tensor_elliptic_ = std::make_shared<AmrTensorElliptic>(eng_, sys_block(0));
    return *tensor_elliptic_;
  }

  // --- ADC-639 conservative-reflux helpers ---------------------------------------------------------
  /// The flux-materialising residual + interface-strip sampling of the CURRENT level (the reflux capture
  /// branch of rhs_into / neg_div_flux_default_into). Sizes the transient level face fluxes Fx/Fy from the
  /// level box_array (xface_box/yface_box), computes R == the fused residual bit-for-bit via the engine
  /// capture seam, then samples the coarse-role strip (if a child level exists) and the fine-role strip (if
  /// level_ >= 1) and OVERWRITE-SETS the ledger for R. The source S is cell-local (never in Fx/Fy), so it
  /// is correctly excluded from the strip -- the native reflux invariant.
  void capture_into_(int b, MultiFab& u, MultiFab& r, ResidualCapture mode, int rate_id) const {
    const std::size_t sb = static_cast<std::size_t>(sys_block(b));
    const MultiFab& ref = eng_->level_state(sb, level_);  // the level grid (state layout)
    const int nc = ref.ncomp();
    // One face box per LOCAL level box, co-distributed with the level state (parity with the native path).
    std::vector<Box2D> fxb, fyb;
    const BoxArray& ba = ref.box_array();
    for (int g = 0; g < ba.size(); ++g) {
      fxb.push_back(pops::xface_box(ba[g]));
      fyb.push_back(pops::yface_box(ba[g]));
    }
    MultiFab Fx(BoxArray(std::move(fxb)), ref.dmap(), nc, 0);
    MultiFab Fy(BoxArray(std::move(fyb)), ref.dmap(), nc, 0);
    if (active_parent_ && active_parent_->child_level == level_) {
      const amr::Rational target_phase =
          active_parent_->child_window.begin.phase +
          stage_time_ * (active_parent_->child_window.end.phase -
                         active_parent_->child_window.begin.phase);
      const amr::Rational alpha =
          (target_phase - active_parent_->parent_window.begin.phase) /
          (active_parent_->parent_window.end.phase - active_parent_->parent_window.begin.phase);
      const double target_physical =
          active_parent_->parent_window.begin.physical_time +
          alpha.value() * (active_parent_->parent_window.end.physical_time -
                           active_parent_->parent_window.begin.physical_time);
      runtime::amr::TemporalTransferContext target{
          {0, active_parent_->parent_window.begin.physical_time},
          {1, active_parent_->parent_window.end.physical_time}, {0, target_physical}};
      const MultiFab& old_parent = active_parent_->old_states.at(static_cast<std::size_t>(b));
      const MultiFab& new_parent = active_parent_->new_states.at(static_cast<std::size_t>(b));
      if (mode == ResidualCapture::FluxOnly)
        eng_->level_neg_div_flux_capture_into_temporal(sb, level_, u, r, Fx, Fy, old_parent,
                                                       new_parent, target);
      else
        eng_->level_rhs_capture_into_temporal(sb, level_, u, r, Fx, Fy, old_parent, new_parent,
                                              target);
    } else if (mode == ResidualCapture::FluxOnly) {
      eng_->level_neg_div_flux_capture_into(sb, level_, u, r, Fx, Fy);
    } else {
      eng_->level_rhs_capture_into(sb, level_, u, r, Fx, Fy);
    }
    EdgeFlux ef;
    // COARSE role: the level-k coarse flux at the faces bordering each level-(k+1) patch (a child exists).
    if (level_ + 1 < nlev())
      pops::detail::sample_coarse_role_strip(Fx, Fy,
                                             eng_->level_state(sb, level_ + 1).box_array(), nc, ef);
    // FINE role: the coarse-face-averaged level-k flux at level-k's own patch edges (level_ borders k-1).
    if (level_ >= 1)
      pops::detail::sample_fine_role_strip(Fx, Fy, ba, nc, ef);
    flux_ledger_[key_(&r)] = std::move(ef);  // OVERWRITE-set (a residual op initialises the strip)
    rate_provenance_[key_(&r)] = {rate_id};
  }

  void ensure_level_clocks_() const {
    if (static_cast<int>(level_clocks_.size()) == nlev())
      return;
    const double accepted_time =
        level_clocks_.empty() ? facade_->time() : level_clocks_[0].physical_time;
    level_clocks_.clear();
    level_clocks_.reserve(static_cast<std::size_t>(nlev()));
    for (int k = 0; k < nlev(); ++k)
      level_clocks_.push_back({k, macro_step(), amr::Rational(0, 1), accepted_time});
  }

  template <class Advance>
  void advance_attempt_(double dt, const char* operation, Advance&& advance) const {
    if (!(dt > 0.0))
      throw std::invalid_argument(std::string(operation) + " requires dt > 0");
    // The Program writes directly into the live hierarchy while evaluating stages.  The flux ledger
    // is transactional too, but it cannot restore state, topology, elliptic warm starts, histories,
    // counters or field materialisations.  Snapshot the complete accepted engine state before even
    // importing Program-owned rings so a typed rejected attempt is observationally atomic.
    const auto saved_engine = eng_->step_snapshot();
    import_program_accepted_state_();
    const auto saved_flux = flux_ledger_;
    const auto saved_sync = synchronized_flux_;
    const auto saved_rate_provenance = rate_provenance_;
    const auto saved_sync_report = sync_report_;
    const auto saved_ring = ring_flux_;
    const auto saved_ring_init = ring_flux_init_;
    const auto saved_ring_clocks = ring_clocks_;
    const auto saved_ring_identities = ring_identities_;
    const auto saved_history_owners = history_owners_;
    const auto saved_history_states = history_state_ids_;
    const auto saved_history_spaces = history_space_ids_;
    const auto saved_clocks = level_clocks_;
    const auto saved_live = live_state_rings_;
    const bool saved_rotate = rotate_pending_;
    const bool saved_solved = solved_this_step_;
    const int saved_level = level_;
    const amr::Rational saved_stage_time = stage_time_;
    const auto saved_active_parent = active_parent_;
    const auto saved_window = current_window_;
    const double saved_level_dt = current_level_dt_;
    conservative_ledger_.begin();
    try {
      reset_step();
      regrid_if_due(macro_step());
      ensure_level_clocks_();
      const double t0 = level_clocks_[0].physical_time;
      const amr::ClockWindow root{{0, macro_step(), amr::Rational(0, 1), t0},
                                  {0, macro_step(), amr::Rational(1, 1), t0 + dt}};
      advance(root);
      couple_levels();
      for (int k = 0; k < nlev(); ++k)
        level_clocks_[static_cast<std::size_t>(k)] =
            amr::ClockStamp{k, macro_step() + 1, amr::Rational(0, 1), t0 + dt};
      conservative_ledger_.commit();
      conservative_ledger_.clear();
      level_ = saved_level;
      stage_time_ = saved_stage_time;
      active_parent_ = saved_active_parent;
      current_window_ = saved_window;
      current_level_dt_ = saved_level_dt;
      publish_program_accepted_state_();
    } catch (...) {
      conservative_ledger_.rollback();
      eng_->restore_step_snapshot(saved_engine);
      flux_ledger_ = saved_flux;
      synchronized_flux_ = saved_sync;
      rate_provenance_ = saved_rate_provenance;
      sync_report_ = saved_sync_report;
      ring_flux_ = saved_ring;
      ring_flux_init_ = saved_ring_init;
      ring_clocks_ = saved_ring_clocks;
      ring_identities_ = saved_ring_identities;
      history_owners_ = saved_history_owners;
      history_state_ids_ = saved_history_states;
      history_space_ids_ = saved_history_spaces;
      level_clocks_ = saved_clocks;
      live_state_rings_ = saved_live;
      rotate_pending_ = saved_rotate;
      solved_this_step_ = saved_solved;
      level_ = saved_level;
      stage_time_ = saved_stage_time;
      active_parent_ = saved_active_parent;
      current_window_ = saved_window;
      current_level_dt_ = saved_level_dt;
      throw;
    }
  }

  void validate_program_accepted_state_(const AmrProgramAcceptedState& state) const {
    if (state.level_clocks.size() != static_cast<std::size_t>(nlev()))
      throw std::runtime_error(
          "AMR Program accepted state does not match the restored hierarchy level count");
    for (int level = 0; level < nlev(); ++level) {
      const amr::ClockStamp& clock = state.level_clocks[static_cast<std::size_t>(level)];
      if (clock.level != level || clock.phase != amr::Rational(0, 1))
        throw std::runtime_error(
            "AMR Program accepted state contains a non-accepted or misqualified level clock");
    }
    if (state.history_owners.size() != state.history_states.size() ||
        state.history_owners.size() != state.history_spaces.size() ||
        state.history_owners.size() != state.ring_clocks.size() ||
        state.history_owners.size() != state.ring_identities.size() ||
        state.history_owners.size() != state.ring_flux.size() ||
        state.history_owners.size() != state.ring_flux_initialized.size())
      throw std::runtime_error(
          "AMR Program accepted state has inconsistent qualified history registries");
    for (const auto& [name, owner] : state.history_owners) {
      const auto clocks = state.ring_clocks.find(name);
      const auto identities = state.ring_identities.find(name);
      const auto fluxes = state.ring_flux.find(name);
      const auto initialized = state.ring_flux_initialized.find(name);
      if (state.history_states.count(name) == 0 || state.history_spaces.count(name) == 0 ||
          clocks == state.ring_clocks.end() || identities == state.ring_identities.end() ||
          fluxes == state.ring_flux.end() || initialized == state.ring_flux_initialized.end())
        throw std::runtime_error("AMR Program accepted state lacks history qualification for '" +
                                 name + "'");
      (void)sys_block(owner);  // prove the qualified owner is still installed by name.
      const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
      if (static_cast<int>(clocks->second.size()) != depth ||
          static_cast<int>(identities->second.size()) != depth ||
          static_cast<int>(fluxes->second.size()) != depth ||
          initialized->second.size() != static_cast<std::size_t>(nlev()))
        throw std::runtime_error("AMR Program accepted state has wrong ring depth for '" + name +
                                 "'");
      for (int slot = 0; slot < depth; ++slot) {
        const auto& slot_clocks = clocks->second[static_cast<std::size_t>(slot)];
        const auto& slot_identities = identities->second[static_cast<std::size_t>(slot)];
        const auto& slot_fluxes = fluxes->second[static_cast<std::size_t>(slot)];
        if (slot_clocks.size() != static_cast<std::size_t>(nlev()) ||
            slot_identities.size() != static_cast<std::size_t>(nlev()) ||
            slot_fluxes.size() != static_cast<std::size_t>(nlev()))
          throw std::runtime_error("AMR Program accepted state has wrong level axis for history '" +
                                   name + "'");
        for (int level = 0; level < nlev(); ++level) {
          const auto& identity = slot_identities[static_cast<std::size_t>(level)];
          if (!identity)
            continue;
          const amr::ClockStamp& clock = slot_clocks[static_cast<std::size_t>(level)];
          if (identity->owner != "program.block." + std::to_string(owner) ||
              identity->state != state.history_states.at(name) ||
              identity->space != state.history_spaces.at(name) || identity->level != level ||
              !(identity->clock == clock))
            throw std::runtime_error("AMR Program accepted state has mismatched identity for history '" +
                                     name + "'");
        }
      }
    }
  }

  AmrProgramAcceptedState accepted_state_() const {
    AmrProgramAcceptedState state{
        level_clocks_, history_owners_, history_state_ids_, history_space_ids_, ring_clocks_,
        ring_identities_, ring_flux_, ring_flux_init_};
    // A flat hierarchy captures no C/F flux strips, and a declared ring may still be cold. Persist
    // those cases as explicit zero/empty axes rather than omitting semantic registry entries: strict
    // restore can then distinguish "no flux exists" from a truncated checkpoint.
    for (const auto& [name, owner] : history_owners_) {
      (void)owner;
      const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
      auto& clocks = state.ring_clocks[name];
      auto& identities = state.ring_identities[name];
      auto& fluxes = state.ring_flux[name];
      clocks.resize(static_cast<std::size_t>(depth));
      identities.resize(static_cast<std::size_t>(depth));
      fluxes.resize(static_cast<std::size_t>(depth));
      for (int slot = 0; slot < depth; ++slot) {
        clocks[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
        identities[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
        fluxes[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
      }
      state.ring_flux_initialized[name].resize(static_cast<std::size_t>(nlev()), 0);
    }
    return state;
  }

  void import_program_accepted_state_() const {
    const std::uint64_t revision = facade_->program_accepted_state_revision();
    if (revision == accepted_state_revision_)
      return;
    const std::vector<std::uint8_t> bytes = facade_->program_accepted_state();
    if (bytes.empty())
      throw std::runtime_error(
          "compiled AMR Program restart lacks its accepted clock/history/flux state");
    AmrProgramAcceptedState state = deserialize_amr_program_accepted_state(bytes);
    validate_program_accepted_state_(state);
    level_clocks_ = std::move(state.level_clocks);
    history_owners_ = std::move(state.history_owners);
    history_state_ids_ = std::move(state.history_states);
    history_space_ids_ = std::move(state.history_spaces);
    ring_clocks_ = std::move(state.ring_clocks);
    ring_identities_ = std::move(state.ring_identities);
    ring_flux_ = std::move(state.ring_flux);
    ring_flux_init_ = std::move(state.ring_flux_initialized);
    accepted_state_revision_ = revision;
  }

  void publish_program_accepted_state_() const {
    if (conservative_ledger_.in_transaction() || !conservative_ledger_.empty())
      throw std::runtime_error("cannot checkpoint a non-accepted AMR conservative ledger");
    facade_->restore_program_accepted_state(
        serialize_amr_program_accepted_state(accepted_state_()));
    accepted_state_revision_ = facade_->program_accepted_state_revision();
  }

  struct ActiveParentWindow {
    int child_level = 0;
    amr::ClockWindow parent_window;
    amr::ClockWindow child_window;
    std::vector<MultiFab> old_states;
    std::vector<MultiFab> new_states;
  };
  struct LiveStateRing {
    int level = 0;
    int owner = 0;
    std::string name;
  };

  template <class Body>
  void advance_level_(int level, const amr::ClockWindow& window, Body& body) const {
    const auto saved_window = current_window_;
    const double saved_dt = current_level_dt_;
    current_window_ = window;
    set_level(level);
    std::vector<MultiFab> old_states;
    old_states.reserve(static_cast<std::size_t>(n_blocks()));
    for (int b = 0; b < n_blocks(); ++b)
      old_states.push_back(eng_->level_state(static_cast<std::size_t>(sys_block(b)), level));

    stage_time_ = amr::Rational(0, 1);
    const double local_dt = window.end.physical_time - window.begin.physical_time;
    current_level_dt_ = local_dt;
    body(local_dt);
    flush_level_flux_(level, local_dt, window.end);

    if (level + 1 >= nlev()) {
      current_window_ = saved_window;
      current_level_dt_ = saved_dt;
      return;
    }
    std::vector<MultiFab> new_states;
    new_states.reserve(static_cast<std::size_t>(n_blocks()));
    for (int b = 0; b < n_blocks(); ++b)
      new_states.push_back(eng_->level_state(static_cast<std::size_t>(sys_block(b)), level));

    const int ratio = eng_->parent_child_temporal_ratio(level + 1);
    const amr::ParentChildClockRelation relation(
        level, level + 1, amr::Rational(ratio, 1), amr::RemainderPolicy::IntegralOnly);
    const std::optional<ActiveParentWindow> saved_parent = active_parent_;
    for (const amr::ChildSubstep& substep : relation.partition(window)) {
      active_parent_ = ActiveParentWindow{level + 1, window, substep.window, old_states, new_states};
      advance_level_(level + 1, substep.window, body);
    }
    active_parent_ = saved_parent;
    current_window_ = saved_window;
    current_level_dt_ = saved_dt;
  }

  static EdgeFlux orientation_payload_(const EdgeFlux& source, amr::FluxOrientation orientation) {
    EdgeFlux out = source;
    auto filter = [orientation](EdgeStrip& strip) {
      if (orientation != amr::FluxOrientation::XMinus) {
        strip.cL.clear();
        strip.fL.clear();
      }
      if (orientation != amr::FluxOrientation::XPlus) {
        strip.cR.clear();
        strip.fR.clear();
      }
      if (orientation != amr::FluxOrientation::YMinus) {
        strip.cB.clear();
        strip.fB.clear();
      }
      if (orientation != amr::FluxOrientation::YPlus) {
        strip.cT.clear();
        strip.fT.clear();
      }
    };
    for (EdgeStrip& strip : out.coarse)
      filter(strip);
    for (EdgeStrip& strip : out.fine)
      filter(strip);
    return out;
  }

  void flush_level_flux_(int level, double dt, const amr::ClockStamp& clock) const {
    for (int b = 0; b < n_blocks(); ++b) {
      const MultiFab* state = &eng_->level_state(static_cast<std::size_t>(sys_block(b)), level);
      const auto found = flux_ledger_.find({level, static_cast<const void*>(state)});
      if (found == flux_ledger_.end())
        continue;
      pops::detail::edge_flux_axpy(synchronized_flux_[{b, level}], Real(1), found->second);
      std::string rate_identity = "program.rate";
      const auto provenance = rate_provenance_.find({level, static_cast<const void*>(state)});
      if (provenance != rate_provenance_.end())
        for (int node : provenance->second)
          rate_identity += ".node." + std::to_string(node);
      const Geometry geometry = eng_->level_geom(level);
      const std::pair<amr::FluxOrientation, double> directions[] = {
          {amr::FluxOrientation::XMinus, static_cast<double>(geometry.dy())},
          {amr::FluxOrientation::XPlus, static_cast<double>(geometry.dy())},
          {amr::FluxOrientation::YMinus, static_cast<double>(geometry.dx())},
          {amr::FluxOrientation::YPlus, static_cast<double>(geometry.dx())}};
      int direction_id = 0;
      for (const auto& [orientation, face_measure] : directions) {
        EdgeFlux normalized = orientation_payload_(found->second, orientation);
        pops::detail::edge_flux_axpy(normalized,
                                     Real(1) / static_cast<Real>(dt * face_measure) - Real(1),
                                     normalized);
        conservative_ledger_.accumulate(
            {"program.block." + std::to_string(b), "conservative_state", rate_identity,
             "default_flux.orientation." + std::to_string(direction_id++), level, clock},
            {amr::Rational(1, 1), orientation, face_measure, dt}, std::move(normalized));
      }
    }
    for (auto it = flux_ledger_.begin(); it != flux_ledger_.end();) {
      if (it->first.first == level)
        it = flux_ledger_.erase(it);
      else
        ++it;
    }
    for (auto it = rate_provenance_.begin(); it != rate_provenance_.end();) {
      if (it->first.first == level)
        it = rate_provenance_.erase(it);
      else
        ++it;
    }
  }

  const EdgeFlux& synchronized_flux_at_(int block, int level) const {
    static const EdgeFlux kEmpty;
    const auto found = synchronized_flux_.find({block, level});
    return found == synchronized_flux_.end() ? kEmpty : found->second;
  }

  std::pair<int, const void*> key_(const MultiFab* mf) const {
    return {level_, static_cast<const void*>(mf)};
  }

  /// ledger[u] += a * ledger[r] (component-wise on the interface strips), skipping if &r is absent (zero
  /// flux). The native lockstep discipline: the register accumulates sum_i w_i F_i with the SAME weights
  /// the state combine applies, reproduced through the Program's axpy without any scheme dispatch.
  void ledger_axpy_(MultiFab& u, Real a, const MultiFab& r) const {
    const auto it = flux_ledger_.find(key_(&r));
    if (it == flux_ledger_.end())
      return;
    pops::detail::edge_flux_axpy(flux_ledger_[key_(&u)], a, it->second);
    const auto provenance = rate_provenance_.find(key_(&r));
    if (provenance != rate_provenance_.end())
      rate_provenance_[key_(&u)].insert(provenance->second.begin(), provenance->second.end());
  }
  /// ledger[z] = a*ledger[x] + b*ledger[y] (overwrite-set; missing operand = zero flux).
  void ledger_lincomb_(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    EdgeFlux out;
    const auto itx = flux_ledger_.find(key_(&x));
    if (itx != flux_ledger_.end())
      pops::detail::edge_flux_axpy(out, a, itx->second);
    const auto ity = flux_ledger_.find(key_(&y));
    if (ity != flux_ledger_.end())
      pops::detail::edge_flux_axpy(out, b, ity->second);
    flux_ledger_[key_(&z)] = std::move(out);
    std::set<int> provenance;
    const auto px = rate_provenance_.find(key_(&x));
    if (px != rate_provenance_.end())
      provenance.insert(px->second.begin(), px->second.end());
    const auto py = rate_provenance_.find(key_(&y));
    if (py != rate_provenance_.end())
      provenance.insert(py->second.begin(), py->second.end());
    rate_provenance_[key_(&z)] = std::move(provenance);
  }

  /// Persist the stored buffer's flux strip into ring_flux_[name] slot 0 at the current level, so a later
  /// lag read (history()) can re-publish it. Also record whether the stored value ALIASES the live level
  /// state (a state ring), for the couple_levels slot-0 resync after reflux.
  void save_ring_flux_(const std::string& name, const MultiFab& value, int owner) const {
    std::vector<std::vector<EdgeFlux>>& ring = ring_flux_[name];
    const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
    if (static_cast<int>(ring.size()) != depth)
      ring.assign(static_cast<std::size_t>(depth < 1 ? 1 : depth), {});
    for (auto& slot : ring)
      if (static_cast<int>(slot.size()) < nlev())
        slot.assign(static_cast<std::size_t>(nlev()), {});
    const auto it = flux_ledger_.find(key_(&value));
    EdgeFlux ef = (it != flux_ledger_.end()) ? it->second : EdgeFlux{};
    // PER-RING PER-LEVEL COLD START (mirror of AmrHistoryOps::store_history): the FIRST store of a (name,
    // level) broadcasts into EVERY deeper slot, so a multistep step 0 reads the same flux at each lag; from
    // then on only slot 0 is written (the ring rotate carries the older slots). Tracked with our own flag
    // set, independent of the engine's hist_init_ (which is already set by the engine store above).
    std::vector<char>& init = ring_flux_init_[name];
    if (static_cast<int>(init.size()) < nlev())
      init.assign(static_cast<std::size_t>(nlev()), 0);
    ring[0][static_cast<std::size_t>(level_)] = ef;
    if (!init[static_cast<std::size_t>(level_)]) {
      for (std::size_t s = 1; s < ring.size(); ++s)
        ring[s][static_cast<std::size_t>(level_)] = ef;
      init[static_cast<std::size_t>(level_)] = 1;
    }
    // Record a live-state-aliasing ring for the post-reflux slot-0 resync (AB2 stores the RHS -> absent).
    if (&value == &eng_->level_state(static_cast<std::size_t>(sys_block(owner)), level_))
      live_state_rings_.push_back({level_, owner, name});
  }
  /// Re-publish ring_flux_[name][lag][level] into the live ledger for the ring buffer @p mf, so the commit
  /// combine carries the lagged flux's weight (acceptance e). Absent = zero flux (no ring strip recorded).
  void restore_ring_flux_(const std::string& name, int lag, MultiFab& mf) const {
    const auto it = ring_flux_.find(name);
    if (it == ring_flux_.end())
      return;
    const std::vector<std::vector<EdgeFlux>>& ring = it->second;
    if (lag < 0 || lag >= static_cast<int>(ring.size()) ||
        level_ >= static_cast<int>(ring[static_cast<std::size_t>(lag)].size()))
      return;
    flux_ledger_[key_(&mf)] = ring[static_cast<std::size_t>(lag)][static_cast<std::size_t>(level_)];
  }
  /// Rotate ring_flux_ one slot in lockstep with the ADC-631 history ring (called by couple_levels when
  /// the deferred rotate fires). O(1) vector-of-strip swaps, the exact chain AmrHistoryOps::rotate uses.
  void rotate_ring_flux_() const {
    for (auto& [name, ring] : ring_flux_)
      for (std::size_t s = ring.size(); s-- > 1;)
        std::swap(ring[s], ring[s - 1]);
  }
  void rotate_ring_clocks_() const {
    for (auto& [name, ring] : ring_clocks_) {
      (void)name;
      for (std::size_t s = ring.size(); s-- > 1;)
        std::swap(ring[s], ring[s - 1]);
    }
    for (auto& [name, ring] : ring_identities_) {
      (void)name;
      for (std::size_t s = ring.size(); s-- > 1;)
        std::swap(ring[s], ring[s - 1]);
    }
  }
  void require_history_owner_(const std::string& name, int owner) const {
    const auto found = history_owners_.find(name);
    if (found == history_owners_.end() || found->second != owner)
      throw std::runtime_error("history '" + name + "' is not qualified by owner program.block." +
                               std::to_string(owner));
  }
  void validate_history_clock_(const std::string& name, int lag) const {
    const auto clocks = ring_clocks_.find(name);
    const auto identities = ring_identities_.find(name);
    if (clocks == ring_clocks_.end() || identities == ring_identities_.end() || lag < 0 ||
        lag >= static_cast<int>(clocks->second.size()) ||
        lag >= static_cast<int>(identities->second.size()) ||
        level_ >= static_cast<int>(clocks->second[static_cast<std::size_t>(lag)].size()) ||
        level_ >= static_cast<int>(identities->second[static_cast<std::size_t>(lag)].size()) ||
        !identities->second[static_cast<std::size_t>(lag)][static_cast<std::size_t>(level_)])
      throw std::runtime_error("history '" + name + "' has no qualified clock for this lag/level");
    const amr::ClockStamp& clock =
        clocks->second[static_cast<std::size_t>(lag)][static_cast<std::size_t>(level_)];
    const amr::HistoryIdentity& identity =
        *identities->second[static_cast<std::size_t>(lag)][static_cast<std::size_t>(level_)];
    const int owner = history_owners_.at(name);
    if (identity.owner != "program.block." + std::to_string(owner) ||
        identity.state != history_state_ids_.at(name) || identity.space != history_space_ids_.at(name) ||
        identity.level != level_ || !(identity.clock == clock))
      throw std::runtime_error("history '" + name + "' identity does not match its qualified slot");
  }
  void record_history_clock_(const std::string& name) const {
    if (!current_window_)
      throw std::runtime_error("history store requires an active qualified level clock");
    const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
    auto& ring = ring_clocks_[name];
    if (static_cast<int>(ring.size()) != depth)
      ring.assign(static_cast<std::size_t>(depth),
                  std::vector<amr::ClockStamp>(static_cast<std::size_t>(nlev())));
    for (auto& slot : ring)
      if (static_cast<int>(slot.size()) != nlev())
        slot.resize(static_cast<std::size_t>(nlev()));
    const int owner = history_owners_.at(name);
    const amr::HistoryIdentity identity{
        "program.block." + std::to_string(owner), history_state_ids_.at(name),
        history_space_ids_.at(name), level_, current_window_->end};
    auto& identities = ring_identities_[name];
    if (static_cast<int>(identities.size()) != depth)
      identities.assign(
          static_cast<std::size_t>(depth),
          std::vector<std::optional<amr::HistoryIdentity>>(static_cast<std::size_t>(nlev())));
    for (auto& slot : identities)
      if (static_cast<int>(slot.size()) != nlev())
        slot.resize(static_cast<std::size_t>(nlev()));
    // AmrHistoryOps cold-start-fills every data slot on the first store of a name/level.  Mirror that
    // publication for clocks and identities so every declared lag denotes the same accepted step-zero
    // value; later stores update only slot zero and rotation carries the older qualified slots.
    if (!identities[0][static_cast<std::size_t>(level_)]) {
      for (std::size_t slot = 0; slot < ring.size(); ++slot) {
        ring[slot][static_cast<std::size_t>(level_)] = current_window_->end;
        identities[slot][static_cast<std::size_t>(level_)] = identity;
      }
    } else {
      ring[0][static_cast<std::size_t>(level_)] = current_window_->end;
      identities[0][static_cast<std::size_t>(level_)] = identity;
    }
  }
  /// The ledger entry for (level, &mf), or a static empty EdgeFlux (zero flux) if absent -- used by
  /// couple_levels to fetch the coarse-role / fine-role strips of a level's committed buffer.
  const EdgeFlux& ledger_at_(int level, const MultiFab* mf) const {
    static const EdgeFlux kEmpty;
    const auto it = flux_ledger_.find({level, static_cast<const void*>(mf)});
    return it == flux_ledger_.end() ? kEmpty : it->second;
  }
  /// A live-state write (a stage axpy or the terminal commit lincomb into level state) INVALIDATES any
  /// earlier ring snapshot of that level's live state: after such a write the stored slot-0 no longer
  /// equals the live state, so it must NOT be resynced from the corrected state at couple_levels. BDF2
  /// stores U^n BEFORE its commit, so the commit clears its record -> no resync (correct). A scheme that
  /// stores the COMMITTED state (store after the commit) keeps its record -> the reflux correction reaches
  /// slot 0. Cheap: an erase of the (level, *) records after every live-state write.
  void note_live_write_(const MultiFab* dst) const {
    if (live_state_rings_.empty())
      return;
    live_state_rings_.erase(
        std::remove_if(live_state_rings_.begin(), live_state_rings_.end(),
                       [&](const LiveStateRing& ring) {
                         return dst == &eng_->level_state(
                                           static_cast<std::size_t>(sys_block(ring.owner)),
                                           ring.level);
                       }),
        live_state_rings_.end());
  }
  /// After the reflux corrects the coarse live state, re-copy it into slot 0 of any ring that stored the
  /// COMMITTED live state this step (still recorded in live_state_rings_ -- i.e. no live-state write
  /// followed the store), so the stored state and the live state stay consistent for a multistep Program's
  /// next-step lag read (ADC-631 consistency, section 2c). AB2 stores the RHS (never the live state) and
  /// BDF2's commit clears its pre-commit U^n record, so neither is touched -- the resync fires only for a
  /// scheme that genuinely stores the post-commit state.
  void resync_history_slot0_() const {
    for (const LiveStateRing& ring : live_state_rings_) {
      if (ring.level >= nlev())
        continue;
      const std::size_t sb = static_cast<std::size_t>(sys_block(ring.owner));
      const Real stored_dt = static_cast<Real>(
          pops::detail::AmrHistoryOps::slot_dt(*eng_, ring.name, 0));
      pops::detail::AmrHistoryOps::store_history(
          *eng_, ring.name, ring.level, eng_->level_state(sb, ring.level), stored_dt);
    }
  }

  AmrSystem* facade_;
  AmrRuntime* eng_;
  mutable int level_ = 0;
  mutable bool solved_this_step_ = false;
  mutable std::optional<GeometricMG> mg_precond_;
  mutable std::shared_ptr<AmrTensorElliptic> tensor_elliptic_;

  // --- ADC-639 conservative-reflux state -----------------------------------------------------------
  // The effective-flux LEDGER: for each tracked MultiFab (keyed by (level, address)) the interface-strip
  // effective flux (EdgeFlux: coarse-role per child patch + fine-role per this-level patch). The Program's
  // linear combination is SHADOWED on these strips (axpy / lincomb mirror), so a commit's strip holds
  // dt * Feff = dt * sum_i w_i F_i -- the native effective flux reproduced from the Program text. Cleared
  // per macro-step by reset_step(); populated ONLY when capturing() (nlev > 1). On a coarse-only / flat
  // Program it stays EMPTY (the capture branch is never reached) -- the bit-identical parity gate.
  mutable std::map<std::pair<int, const void*>, EdgeFlux> flux_ledger_;
  mutable std::map<std::pair<int, const void*>, std::set<int>> rate_provenance_;
  mutable std::map<std::pair<int, int>, EdgeFlux> synchronized_flux_;
  mutable std::vector<SyncEvent> sync_report_;
  mutable amr::TransactionalFluxLedger<EdgeFlux> conservative_ledger_;
  mutable std::vector<amr::ClockStamp> level_clocks_;
  mutable std::uint64_t accepted_state_revision_ = 0;
  mutable amr::Rational stage_time_{0, 1};
  mutable std::optional<ActiveParentWindow> active_parent_;
  mutable std::optional<amr::ClockWindow> current_window_;
  mutable double current_level_dt_ = 0.0;
  // PERSISTENT per-history-ring flux strips (NOT cleared per step): name -> [slot][level] -> EdgeFlux. A
  // multistep scheme (AB2 / BDF2) reads a lagged RHS / state from the ring; store_history saves that
  // buffer's ledger strip here (slot 0), rotate_histories rotates it in lockstep with the ring, and
  // history() restores it into the live ledger so the commit combine carries the lagged flux's weight --
  // the reflux stays conservative for a multistep Program across steps (acceptance e).
  mutable std::map<std::string, std::vector<std::vector<EdgeFlux>>> ring_flux_;
  // Per-ring per-level cold-start flags for ring_flux_ (mirror of the engine hist_init_), so the first
  // store of a (name, level) broadcasts its strip into every slot. PERSISTENT (not cleared per step).
  mutable std::map<std::string, std::vector<char>> ring_flux_init_;
  mutable std::map<std::string, int> history_owners_;
  mutable std::map<std::string, std::string> history_state_ids_;
  mutable std::map<std::string, std::string> history_space_ids_;
  mutable std::map<std::string, std::string> history_clock_ids_;
  mutable std::map<std::string, std::string> history_interpolation_ids_;
  mutable std::map<std::string, std::vector<std::vector<amr::ClockStamp>>> ring_clocks_;
  mutable std::map<
      std::string,
      std::vector<std::vector<std::optional<amr::HistoryIdentity>>>> ring_identities_;
  // Deferred-rotate flag (ADC-631 consistency, section 2c): the body's terminal rotate_histories() fires
  // inside the recursive level advance, before couple_levels reflux modifies the coarse live state. We defer the
  // rotate to couple_levels (after the reflux + slot-0 resync), so a multistep Program never reads a
  // pre-reflux ring slot against a post-reflux live state. On nlev==1 the deferral collapses to the
  // original store->rotate order -> bit-identical to Uniform / v1.
  mutable bool rotate_pending_ = false;
  mutable ClockScheduleState clock_schedule_;
  // Per-step record of (level, ring name) whose slot-0 was written FROM the live level state this step:
  // after the reflux corrects that live state, couple_levels re-copies it into slot 0 so the stored state
  // and the live state stay consistent. A ring storing a non-live buffer (AB2 stores the RHS) is absent.
  mutable std::vector<LiveStateRing> live_state_rings_;
};

}  // namespace program
}  // namespace runtime

// ADC-637: the condensed-implicit assembly kernels are emitted INLINE (block_inverse + pops::apply_laplacian)
// referencing only the variable `ctx`, so the SAME generated body instantiates directly against
// AmrProgramContext -- reaching this context's grid_context() / assembly_target / assembly_source per level.
// No coupling/schur free function or delegating overload remains on any path.
}  // namespace pops
