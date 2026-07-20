#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/core/foundation/types.hpp>           // Real, POPS_HD
#include <pops/mesh/boundary/physical_bc.hpp>       // fill_ghosts
#include <pops/mesh/execution/for_each.hpp>         // for_each_cell, device_fence
#include <pops/mesh/geometry/geometry.hpp>          // Geometry
#include <pops/mesh/layout/field_distribution.hpp>  // FieldDistribution
#include <pops/mesh/storage/mf_arith.hpp>           // saxpy / lincomb
#include <pops/mesh/storage/multifab.hpp>           // MultiFab
#include <pops/parallel/execution_lane.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_ledger.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntime (the engine the driver wraps)
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/runtime/context/grid_context.hpp>  // GridContext (per-level Schur assembly seam, ADC-633)
#include <pops/runtime/amr_system.hpp>  // AmrSystem (the facade: params / block map / engine)
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/step_transaction.hpp>
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
    amr::ClockStamp clock;
  };
  struct HistoryFluxTopology {
    std::uint64_t epoch = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::vector<Box2D>> boxes;
    std::vector<std::vector<int>> owners;

    bool bound() const { return epoch != std::numeric_limits<std::uint64_t>::max(); }
  };
  /// Logical-clock child interval nested inside the current AMR level window.  The hierarchy driver
  /// remains the authority for level substeps; this move-only companion further partitions that exact
  /// window for Program.subcycle and restores it on normal, nested, and exceptional exits.
  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const AmrProgramContext& owner, int iteration, int count)
        : owner_(&owner),
          saved_window_(owner.current_window_),
          saved_dt_(owner.current_level_dt_),
          saved_stage_(owner.stage_time_) {
      if (count <= 0 || iteration < 0 || iteration >= count)
        throw std::invalid_argument(
            "AMR Program logical evaluation requires a valid child iteration");
      if (!saved_window_ || !std::isfinite(saved_dt_) || saved_dt_ <= 0.0)
        throw std::logic_error(
            "AMR Program logical evaluation requires a prepared parent level window");
      const double child_dt = saved_dt_ / static_cast<double>(count);
      const double child_physical_time =
          saved_window_->begin.physical_time + static_cast<double>(iteration) * child_dt;
      if (!std::isfinite(child_dt) || child_dt <= 0.0 || !std::isfinite(child_physical_time))
        throw std::overflow_error("AMR Program logical evaluation child window is not finite");
      const amr::Rational parent_span = saved_window_->end.phase - saved_window_->begin.phase;
      const amr::Rational child_begin_phase =
          saved_window_->begin.phase + parent_span * amr::Rational(iteration, count);
      const amr::Rational child_end_phase =
          saved_window_->begin.phase + parent_span * amr::Rational(iteration + 1, count);
      amr::ClockStamp child_begin = saved_window_->begin;
      child_begin.phase = child_begin_phase;
      child_begin.physical_time = child_physical_time;
      amr::ClockStamp child_end = child_begin;
      child_end.phase = child_end_phase;
      child_end.physical_time = child_physical_time + child_dt;

      owner.invalidate_active_operator_snapshot_();
      child_dt_ = child_dt;
      owner.current_window_ = amr::ClockWindow{child_begin, child_end};
      owner.current_level_dt_ = child_dt;
      owner.stage_time_ = amr::Rational(0, 1);
    }
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          saved_window_(other.saved_window_),
          saved_dt_(other.saved_dt_),
          saved_stage_(other.saved_stage_),
          child_dt_(other.child_dt_) {}
    LogicalEvaluationScope& operator=(LogicalEvaluationScope&&) = delete;
    ~LogicalEvaluationScope() noexcept { restore_(); }

    Real dt() const {
      if (owner_ == nullptr)
        throw std::logic_error("AMR Program logical evaluation scope is no longer active");
      return static_cast<Real>(child_dt_);
    }

   private:
    void restore_() noexcept {
      if (owner_ == nullptr)
        return;
      owner_->current_window_ = saved_window_;
      owner_->current_level_dt_ = saved_dt_;
      owner_->stage_time_ = saved_stage_;
      owner_->invalidate_active_operator_snapshot_();
      owner_ = nullptr;
    }

    const AmrProgramContext* owner_ = nullptr;
    std::optional<amr::ClockWindow> saved_window_;
    double saved_dt_ = 0.0;
    amr::Rational saved_stage_{0, 1};
    double child_dt_ = 0.0;
  };
  /// Wrap an AmrSystem passed as a flat void* (what pops_install_program_amr(void* sys) receives). The
  /// ctor pulls the AmrRuntime engine out of the facade (engine() returns the built runtime; the AMR
  /// blocks must be materialized -- install_program forces the build before install()).
  explicit AmrProgramContext(void* sys)
      : facade_(static_cast<AmrSystem*>(sys)), eng_(facade_->engine()) {
    if (eng_ == nullptr)
      throw std::runtime_error(
          "AmrProgramContext: the AMR runtime engine is not built; install_program must force the "
          "multi-block AmrRuntime build before installing a compiled time Program over the "
          "hierarchy");
    require_supported_program_refinement_ratios_(*eng_);
    stage_restore_scratch_.reserve(eng_->n_blocks());
    hierarchy_tensor_solver_registry_ = facade_->hierarchy_tensor_solver_provider_registry();
  }
  /// Direct ctor (C++ tests / the driver): an engine + the facade carrying the param / block-map stores.
  AmrProgramContext(AmrRuntime* eng, AmrSystem* facade) : facade_(facade), eng_(eng) {
    // Keep the established contract-test seam: argument/rate validation must be exercisable before
    // any topology lookup. Every executable driver supplies a real engine and is ratio-validated here;
    // the production void* constructor above remains fail-closed when the engine was not built.
    if (eng_ != nullptr) {
      require_supported_program_refinement_ratios_(*eng_);
      stage_restore_scratch_.reserve(eng_->n_blocks());
    }
    if (facade_ != nullptr)
      hierarchy_tensor_solver_registry_ = facade_->hierarchy_tensor_solver_provider_registry();
  }

  // --- driver state (mutable: every seam method mirrors the const ProgramContext surface) ------------
  void set_level(int k) const { level_ = k; }
  int level() const { return level_; }
  /// Epoch used by generated install-time resource bundles. A regrid/rebalance changes this value;
  /// generated Programs then rematerialize every per-level persistent field/problem/workspace once,
  /// before entering the next hierarchy advance. Compatible steps retain the same bundles.
  std::uint64_t program_resource_topology_epoch() const { return eng_->topology_epoch(); }
  /// Runtime-only companion to the checkpointed epoch. This changes whenever hierarchy storage is
  /// reconstructed, including checkpoint restore and rejected-attempt rollback, so an equal restored
  /// epoch/nlev pair can never authenticate stale layout-bound Program resources.
  std::uint64_t program_resource_topology_generation() const {
    return eng_->topology_materialization_generation();
  }
  /// Exact ownership of the level whose install-time Program resources are being materialized.
  /// Generated prepared solvers must not infer replication from rank-local DistributionMapping
  /// metadata: a replicated level intentionally names the current rank as its local owner.
  const PreparedVectorDistribution& program_resource_vector_distribution() const {
    return eng_->level_is_replicated(level_) ? PreparedVectorDistribution::Replicated
                                             : PreparedVectorDistribution::Distributed;
  }
  FieldDistribution program_resource_field_storage_distribution() const {
    return eng_->level_is_replicated(level_) ? FieldDistribution::Replicated
                                             : FieldDistribution::Distributed;
  }
  int program_resource_field_level() const { return level_; }
  /// Materialize absolute hierarchy metadata for the current level's persistent prepared resource.
  /// The generated solver remains topology-agnostic; AMR alone owns level numbering, per-level
  /// ownership and metric resolution.
  void configure_program_resource_field_nullspace(FieldNullspacePlan& plan) const {
    if (level_ < 0 || level_ >= nlev())
      throw std::out_of_range("AMR Program field-nullspace resource level is out of range");
    const Geometry geometry = eng_->level_geom(level_);
    const Real measure = geometry.dx() * geometry.dy();
    for (FieldNullspaceBasis& basis : plan.bases) {
      basis.cell_measure.assign(static_cast<std::size_t>(nlev()), Real(0));
      basis.cell_measure[static_cast<std::size_t>(level_)] = measure;
    }
  }
  int nlev() const { return eng_->nlev(); }
  bool uses_prepared_krylov_fallback() const {
    return configured_hierarchy_tensor_solver_().execution_path() ==
           HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
  }
  /// Reset the per-macro-step flags (called by the install wrapper at the top of each macro-step). Also
  /// clears the per-step effective-flux ledger + the live-state-ring record (ADC-639); the PERSISTENT
  /// per-ring flux strips (ring_flux_) survive across steps, as the multistep ring itself does.
  void reset_step() const {
    default_solve_report_.reset();
    for (auto& [_, report] : named_solve_reports_)
      report = SolveReport{};
    flux_ledger_.clear();
    flux_contributions_.clear();
    rate_provenance_.clear();
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
                     [&](const amr::ClockWindow& root) { advance_level_(0, root, dt, body); });
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
  void configure_primary_clock(const std::string& clock) const {
    clock_schedule_.configure_primary_clock(clock);
    primary_clock_ = clock;
  }
  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    clock_schedule_.declare_relation(parent, child, count);
  }
  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level, level_, macro_step())
        .has_value();
  }
  bool schedule_is_due(int node_id, int every_n, ScheduleDomainKind kind, const std::string& clock,
                       const std::string& stage_identity, int level) const {
    if (node_id < 0 || every_n <= 0)
      throw std::runtime_error("AMR Program schedule requires a valid node and positive period");
    const auto coordinate =
        clock_schedule_.coordinate(kind, clock, stage_identity, level, level_, macro_step());
    return coordinate && coordinate->value % every_n == 0;
  }
  bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                         const std::string& stage_identity, int level) const {
    const auto coordinate =
        clock_schedule_.coordinate(kind, clock, stage_identity, level, level_, macro_step());
    return coordinate && coordinate->value == 0;
  }

  /// Exact parity with ProgramContext: record one typed scheduler decision. Cache-backed AMR
  /// policies still fail loud at their cache action seams below; this counter never pretends that
  /// an AMR cache exists, it only reports the decision that was actually reached.
  bool schedule_decision(int node_id, bool due, bool cache_backed) const {
    if (node_id < 0)
      throw std::runtime_error("AMR Program schedule decision requires a valid node");
    return facade_->profiler_handle().schedule_decision(due, cache_backed);
  }
  ClockScheduleState::SubcycleScope subcycle_scope(const std::string& parent,
                                                   const std::string& child, int count) const {
    return clock_schedule_.subcycle(parent, child, count);
  }
  [[nodiscard]] LogicalEvaluationScope logical_evaluation_scope(int iteration, int count) const {
    return LogicalEvaluationScope(*this, iteration, count);
  }
  void synchronize_sample_and_hold(const std::string& source, const std::string& target, int step,
                                   Real offset) const {
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
      throw block_map_error_("AmrProgramContext::sys_block: program block index " +
                             std::to_string(b) + " is outside the explicit block map [0, " +
                             std::to_string(m.size()) + ")");
    const int mapped = m[static_cast<std::size_t>(b)];
    const int count = static_cast<int>(eng_->n_blocks());
    if (mapped < 0 || mapped >= count)
      throw block_map_error_("AmrProgramContext::sys_block: program block index " +
                             std::to_string(b) + " maps to invalid AMR block index " +
                             std::to_string(mapped) + " for an AmrRuntime with " +
                             std::to_string(count) + " blocks");
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
        if (!current_sync_clock_)
          throw std::runtime_error("AMR synchronization has no exact accepted clock");
        amr::ClockStamp sync_clock = *current_sync_clock_;
        sync_clock.level = k - 1;
        if (capturing()) {
          sync_report_.push_back({k - 1, k, b, SyncPhase::Reflux, sync_clock});
          // The accepted exact ledger is the sole numerical reflux authority.  Reconstruct the two
          // dt-integrated strips on demand from its qualified entries; there is no independently
          // accumulated synchronized-flux shadow that can drift across rejection/restart.
          const EdgeFlux coarse_role = reflux_flux_from_ledger_(b, k - 1);
          const EdgeFlux fine_role = reflux_flux_from_ledger_(b, k);
          pops::detail::route_reflux_program(*eng_, sb, k, coarse_role, fine_role);
        }
        sync_report_.push_back({k - 1, k, b, SyncPhase::AverageDown, sync_clock});
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
  /// A changed topology also rebinds lagged conservative histories and their compact interface-flux
  /// authority before the Program reads prev(k); a layout-identical regrid remains bit-identical.
  void regrid_if_due(int macro_step) const {
    const HistoryFluxTopology before = history_flux_topology_snapshot_();
    if (history_flux_topology_.bound() &&
        !same_history_flux_topology_(history_flux_topology_, before))
      throw std::runtime_error(
          "AMR lagged-flux topology authority differs from the accepted hierarchy");
    history_flux_topology_ = before;
    eng_->regrid_if_due(macro_step);
    const HistoryFluxTopology after = history_flux_topology_snapshot_();
    if (after.epoch != before.epoch && !same_history_flux_layout_(before, after))
      rebind_history_flux_topology_(before, after);
    history_flux_topology_ = after;
  }
  std::uint64_t history_flux_topology_epoch() const {
    return history_flux_topology_.bound() ? history_flux_topology_.epoch : eng_->topology_epoch();
  }
  bool history_flux_topology_bound() const { return history_flux_topology_.bound(); }
  int history_flux_topology_rebind_count() const { return history_flux_topology_rebind_count_; }

  // --- state / RHS seam over the CURRENT level -------------------------------------------------------
  MultiFab& state(int b) const {
    return eng_->level_state(static_cast<std::size_t>(sys_block(b)), level_);
  }
  void rhs_into(int b, MultiFab& u, MultiFab& r, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel();
    if (capturing()) {
      capture_into_(b, u, r, ResidualCapture::FullRate, rate_id);
      return;
    }
    eng_->level_rhs_into_at(static_cast<std::size_t>(sys_block(b)), level_,
                            boundary_point_(rate_id), u, r);
  }
  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage_id) const {
    return boundary_point_(stage_id);
  }
  bool has_boundary_linearization(int b) const {
    return eng_->has_boundary_linearization(static_cast<std::size_t>(sys_block(b)));
  }
  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                        MultiFab& u, MultiFab& r, bool flux_only) const {
    count_kernel();
    eng_->level_rhs_core_into_at(static_cast<std::size_t>(sys_block(b)), point.level, point, u, r,
                                 flux_only);
  }
  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                        MultiFab& u, MultiFab& r, bool flux_only,
                        const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    eng_->level_rhs_core_into_at(static_cast<std::size_t>(sys_block(b)), point.level, point, u, r,
                                 flux_only, boundary);
  }
  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                                 MultiFab& u, MultiFab& c) const {
    count_kernel();
    eng_->level_boundary_residual_into_at(static_cast<std::size_t>(sys_block(b)), point.level,
                                          point, u, c);
  }
  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                                 MultiFab& u, MultiFab& c,
                                 const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    eng_->level_boundary_residual_into_at(static_cast<std::size_t>(sys_block(b)), point.level,
                                          point, u, c, boundary);
  }
  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                            MultiFab& u, const MultiFab& v, MultiFab& j) const {
    count_kernel();
    eng_->level_boundary_jvp_into_at(static_cast<std::size_t>(sys_block(b)), point.level, point, u,
                                     v, j);
  }
  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                            MultiFab& u, const MultiFab& v, MultiFab& j,
                            const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    eng_->level_boundary_jvp_into_at(static_cast<std::size_t>(sys_block(b)), point.level, point, u,
                                     v, j, boundary);
  }
  struct RhsGroupRequest {
    RhsGroupRequest(int block_value, MultiFab* state_value, MultiFab* rhs_value, int rate_id_value,
                    int flux_only_value)
        : block(block_value),
          state(state_value),
          rhs(rhs_value),
          rate_id(rate_id_value),
          flux_only(flux_only_value) {}

    int block;
    MultiFab* state;
    MultiFab* rhs;
    int rate_id;
    int flux_only;
  };
  /// @p group_id identifies the authored atomic evaluation independently of each request's exact
  /// rate-node identity.  Neither identity may be anonymous or inferred from iteration order.
  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    require_group_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("AMR Program RHS group cannot be empty");
    std::vector<int> rate_ids;
    rate_ids.reserve(requests.size());
    for (const auto& request : requests) {
      require_rate_identity_(request.rate_id);
      if (request.rate_id == group_id ||
          std::find(rate_ids.begin(), rate_ids.end(), request.rate_id) != rate_ids.end())
        throw std::invalid_argument(
            "AMR Program RHS group and member rate identities must be distinct");
      if (request.state == nullptr || request.rhs == nullptr ||
          (request.flux_only != 0 && request.flux_only != 1))
        throw std::invalid_argument("AMR Program RHS group contains an invalid request");
      rate_ids.push_back(request.rate_id);
    }
    if (capturing()) {
      if (eng_->has_level_interfaces(level_))
        deferred_op("refined_shared_block_interfaces",
                    "shared block interfaces across a refined hierarchy require a prepared "
                    "interface-flux reflux ledger; coarse-only execution is supported");
      for (const auto& request : requests) {
        count_kernel();
        capture_into_(request.block, *request.state, *request.rhs,
                      request.flux_only ? ResidualCapture::FluxOnly : ResidualCapture::FullRate,
                      request.rate_id);
      }
      return;
    }
    std::vector<int> blocks;
    std::vector<MultiFab*> states;
    std::vector<MultiFab*> rhs;
    std::vector<int> flux_only;
    blocks.reserve(requests.size());
    states.reserve(requests.size());
    rhs.reserve(requests.size());
    flux_only.reserve(requests.size());
    for (const auto& request : requests) {
      count_kernel();
      blocks.push_back(sys_block(request.block));
      states.push_back(request.state);
      rhs.push_back(request.rhs);
      flux_only.push_back(request.flux_only);
    }
    eng_->level_rhs_group(level_, boundary_point_(group_id), blocks, states, rhs, flux_only);
  }
  void neg_div_flux_default_into(int b, MultiFab& u, MultiFab& r, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel();
    if (capturing()) {
      capture_into_(b, u, r, ResidualCapture::FluxOnly, rate_id);
      return;
    }
    eng_->level_neg_div_flux_into_at(static_cast<std::size_t>(sys_block(b)), level_,
                                     boundary_point_(rate_id), u, r);
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
  /// The AMR runtime runs it EXACTLY ONCE per macro-step (a level-0 / not-yet-solved guard):
  /// calling it again at fine levels within the same macro-step is a no-op cache-hit (parity: the
  /// body stays atomic, the solve fires once -- the OncePerStep cadence the native AMR step uses).
  SolveReport solve_fields() const {
    if (level_ == 0 || !default_solve_report_) {
      default_solve_report_.reset();
      const SolveReport report = eng_->solve_default_field();
      if (report.solved())
        default_solve_report_ = report;
      return report;
    }
    return *default_solve_report_;
  }
  /// Per-stage re-solve from a stage state is currently a coarse-only capability.  A fine-level request
  /// is rejected explicitly; it never consumes a stale injected auxiliary field.
  SolveReport solve_fields_from_state(int b, MultiFab& u_stage) const {
    if (level_ == 0) {
      MultiFab& live = state(b);
      MultiFab& saved = stage_state_scratch_for_(b, level_, live);
      PureFieldAlgebra::copy_allocated(saved, live);
      default_solve_report_.reset();
      SolveReport report;
      try {
        PureFieldAlgebra::copy_allocated(live, u_stage);
        report = eng_->solve_default_field();
        PureFieldAlgebra::copy_allocated(live, saved);
      } catch (...) {
        PureFieldAlgebra::copy_allocated(live, saved);
        throw;
      }
      if (report.solved())
        default_solve_report_ = report;
      return report;
    }
    deferred_op(
        "solve_fields_from_state_default",
        "the default per-stage fine-level field re-solve requires a composite stage solver; use "
        "OncePerStep field cadence or an exact named field provider");
  }
  SolveReport solve_fields_from_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                         const std::string& provider_slot, int b,
                                         MultiFab& u_stage) const {
    if (provider_slot.empty())
      throw std::invalid_argument(
          "AmrProgramContext::solve_fields_from_state_at requires an exact provider slot");
    if (point.level < 0 || point.level >= eng_->nlev())
      throw std::out_of_range(
          "AmrProgramContext::solve_fields_from_state_at level is out of range");
    if (point.level != 0)
      deferred_op("solve_fields_from_state_at_fine_level",
                  "a fine-level stage perturbation requires a composite field solver");
    named_solve_reports_.erase(provider_slot);
    MultiFab& live = eng_->level_state(static_cast<std::size_t>(sys_block(b)), point.level);
    MultiFab& saved = stage_state_scratch_for_(b, point.level, live);
    PureFieldAlgebra::copy_allocated(saved, live);
    SolveReport report;
    try {
      PureFieldAlgebra::copy_allocated(live, u_stage);
      report = eng_->solve_named_fields(&provider_slot);
      PureFieldAlgebra::copy_allocated(live, saved);
    } catch (...) {
      PureFieldAlgebra::copy_allocated(live, saved);
      named_solve_reports_.insert_or_assign(provider_slot, SolveReport{});
      throw;
    }
    named_solve_reports_.insert_or_assign(provider_slot, report);
    return report;
  }
  template <class Body>
  void evaluate_with_field_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                    const std::string& provider_slot, int b,
                                    MultiFab& evaluation_state, MultiFab& restore_state,
                                    Body&& body) const {
    const auto restore = [&]() {
      const SolveReport restored =
          solve_fields_from_state_at(point, provider_slot, b, restore_state);
      if (!restored.solved_value_available())
        throw_field_solve_failure_(restored, "restoring the frozen field state");
    };
    const SolveReport prepared =
        solve_fields_from_state_at(point, provider_slot, b, evaluation_state);
    if (!prepared.solved_value_available()) {
      restore();
      throw_field_solve_failure_(prepared, "evaluating the perturbed field state");
    }
    try {
      std::forward<Body>(body)();
    } catch (...) {
      const std::exception_ptr failure = std::current_exception();
      restore();
      std::rethrow_exception(failure);
    }
    restore();
  }
  /// Named multi-elliptic field re-solve. The coarse solve publishes and injects every level once;
  /// fine levels consume only that exact provider-qualified report.
  SolveReport solve_fields_from_state(const std::string& field, int b, MultiFab& u_stage) const {
    if (level_ != 0) {
      const auto cached = named_solve_reports_.find(field);
      if (cached == named_solve_reports_.end() || !cached->second.solved())
        throw std::runtime_error(
            "AmrProgramContext::solve_fields_from_state(field): fine-level reuse requires an "
            "accepted coarse SolveReport");
      return cached->second;  // the coarse solve publishes/injects every level once per stage
    }
    MultiFab& live = state(b);
    MultiFab& published = stage_state_scratch_for_(b, level_, live);
    PureFieldAlgebra::copy_allocated(published, live);
    SolveReport report;
    try {
      PureFieldAlgebra::copy_allocated(live, u_stage);
      report = eng_->solve_named_fields(&field);
      PureFieldAlgebra::copy_allocated(live, published);
    } catch (...) {
      PureFieldAlgebra::copy_allocated(live, published);
      named_solve_reports_.insert_or_assign(field, SolveReport{});
      throw;
    }
    named_solve_reports_.insert_or_assign(field, report);
    return report;
  }
  /// Retained default-provider overload: the final Program IR always carries an exact field identity,
  /// while an unqualified coupled solve has no provider authority and therefore fails loud.
  SolveReport solve_fields_from_blocks(const std::vector<const MultiFab*>& /*u_stages*/) const {
    deferred_op(
        "solve_fields_from_blocks_default",
        "an unqualified coupled multi-block field solve has no AMR provider authority; use the "
        "exact field-qualified Program operation");
  }

  SolveReport solve_fields_from_blocks(const std::string& field,
                                       const std::vector<const MultiFab*>& u_stages) const {
    if (level_ != 0) {
      const auto cached = named_solve_reports_.find(field);
      if (cached == named_solve_reports_.end() || !cached->second.solved())
        throw std::runtime_error(
            "AmrProgramContext::solve_fields_from_blocks(field): fine-level reuse requires an "
            "accepted coarse SolveReport");
      return cached->second;
    }
    if (u_stages.size() != static_cast<std::size_t>(n_blocks()))
      throw std::runtime_error(
          "AmrProgramContext::solve_fields_from_blocks(field): stage vector size mismatch");
    stage_restore_scratch_.clear();
    for (std::size_t p = 0; p < u_stages.size(); ++p) {
      if (u_stages[p] == nullptr)
        continue;
      MultiFab& state_value = state(static_cast<int>(p));
      MultiFab& published = stage_state_scratch_for_(static_cast<int>(p), level_, state_value);
      PureFieldAlgebra::copy_allocated(published, state_value);
      stage_restore_scratch_.push_back({&state_value, &published});
      PureFieldAlgebra::copy_allocated(state_value, *u_stages[p]);
    }
    auto restore = [&]() {
      for (const auto& [live, published] : stage_restore_scratch_)
        PureFieldAlgebra::copy_allocated(*live, *published);
    };
    SolveReport report;
    try {
      report = eng_->solve_named_fields(&field);
      restore();
    } catch (...) {
      restore();
      named_solve_reports_.insert_or_assign(field, SolveReport{});
      throw;
    }
    named_solve_reports_.insert_or_assign(field, report);
    return report;
  }

  /// The SHARED aux of the current level (phi / grad / B_z), the channel solve_fields fills.
  MultiFab& aux() const { return const_cast<MultiFab&>(eng_->aux(level_)); }
  /// The current level's metric (dx/dy >> level, domain << level).
  Geometry geom() const { return eng_->level_geom(level_); }
  /// The installed AMR route is Cartesian.  The metric queries still exist on this context so one
  /// generated Program body instantiates against both runtime contexts without geometry-specific
  /// source-stage classes.
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

  /// Block/level-qualified active-cell mask for generated pointwise operators.  Current Cartesian AMR
  /// levels return nullptr.  Geometry-aware AMR providers can populate GridContext::domain_mask and
  /// automatically obtain the same protocol without changing generated Program code.
  const MultiFab* pointwise_active_mask(int block, const MultiFab& field) const {
    const GridContext context =
        eng_->level_grid_context(static_cast<std::size_t>(sys_block(block)), level_);
    if (context.domain_mask == nullptr)
      return nullptr;
    pops::detail::validate_relative_cell_measure(
        field, RelativeCellMeasure{context.domain_mask, nullptr},
        "AmrProgramContext pointwise active-cell mask");
    return context.domain_mask;
  }

  /// AMR counterpart of ProgramContext::pointwise_status_max.  The exact mask view used by the
  /// pointwise kernel is authenticated again before the collective reduction.
  Real pointwise_status_max(int block, const MultiFab& status,
                            const MultiFab* active_cells) const {
    const MultiFab* expected = pointwise_active_mask(block, status);
    if (expected != active_cells)
      throw std::invalid_argument(
          "AmrProgramContext pointwise status reduction received a different active-cell mask");
    const Real reduced =
        pops::reduce_max(status, 0, RelativeCellMeasure{active_cells, nullptr});
    return reduced == -std::numeric_limits<Real>::infinity() ? Real(0) : reduced;
  }

  std::shared_ptr<PreparedGridBoundarySession> prepare_mesh_boundary_session(
      const MultiFab&, const ExecutionLane& lane) const {
    return std::make_shared<PreparedGridBoundarySession>(grid_context(), lane);
  }

  std::shared_ptr<PreparedGridBoundarySession> prepare_block_boundary_session(
      int block, MultiFab& prototype, const runtime::multiblock::BoundaryEvaluationPoint& point,
      const ExecutionLane& lane) const {
    return std::make_shared<PreparedGridBoundarySession>(
        eng_->level_grid_context(static_cast<std::size_t>(sys_block(block)), level_), lane,
        prototype, point);
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
      ledger_axpy_(
          u, a,
          r);  // shadow the state combine on the effective-flux strip: ledger[u] += a*ledger[r]
      note_live_write_(
          &u);  // a write to the live state invalidates any earlier live-state ring snapshot
    }
  }
  void axpy(MultiFab& u, Real a, const MultiFab& r, Real dt,
            std::initializer_list<ExactCoefficientTerm> exact) const {
    count_kernel();
    pops::saxpy(u, a, r);
    if (capturing()) {
      ledger_axpy_exact_(u, a, r, dt, exact);
      note_live_write_(&u);
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
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y, Real dt,
               std::initializer_list<ExactCoefficientTerm> exact_a,
               std::initializer_list<ExactCoefficientTerm> exact_b) const {
    count_kernel();
    pops::lincomb(z, a, x, b, y);
    if (capturing()) {
      ledger_lincomb_exact_(z, a, x, b, y, dt, exact_a, exact_b);
      note_live_write_(&z);
    }
  }
  void commit_many(std::initializer_list<std::pair<MultiFab*, const MultiFab*>> commits) const {
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
  void laplacian(MultiFab& out, MultiFab& in, const ExecutionLane& lane) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(in, g.domain, eng_->transport_bc(), lane);
    apply_laplacian(in, g, out);
  }
  void laplacian(MultiFab& out, MultiFab& in, const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(in);
    apply_laplacian(in, boundary.context().geom, out);
  }
  void laplacian(MultiFab& out, MultiFab& in, const PreparedGridBoundarySession& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(in, point);
    apply_laplacian(in, boundary.context().geom, out);
  }
  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(in, g.domain, eng_->transport_bc());
    apply_laplacian(in, g, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
  }
  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const ExecutionLane& lane) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(in, g.domain, eng_->transport_bc(), lane);
    apply_laplacian(in, g, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
  }
  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(in);
    apply_laplacian(in, boundary.context().geom, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
  }
  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const PreparedGridBoundarySession& boundary,
                        const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(in, point);
    apply_laplacian(in, boundary.context().geom, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
  }
  void gradient(MultiFab& out, MultiFab& phi) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(phi, g.domain, eng_->transport_bc());
    const Real cx = Real(1) / (Real(2) * g.dx());
    const Real cy = Real(1) / (Real(2) * g.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }
  void gradient(MultiFab& out, MultiFab& phi, const ExecutionLane& lane) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(phi, g.domain, eng_->transport_bc(), lane);
    const Real cx = Real(1) / (Real(2) * g.dx());
    const Real cy = Real(1) / (Real(2) * g.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }
  void gradient(MultiFab& out, MultiFab& phi, const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(phi);
    const Geometry& g = boundary.context().geom;
    const Real cx = Real(1) / (Real(2) * g.dx());
    const Real cy = Real(1) / (Real(2) * g.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }
  void gradient(MultiFab& out, MultiFab& phi, const PreparedGridBoundarySession& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(phi, point);
    const Geometry& g = boundary.context().geom;
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
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy, const ExecutionLane& lane) const {
    count_kernel();
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(fx, g.domain, eng_->transport_bc(), lane);
    if (&fy != &fx)
      fill_ghosts(fy, g.domain, eng_->transport_bc(), lane);
    apply_divergence(fx, fy, g, out, /*cx=*/0, /*cy=*/1);
  }
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy,
                  const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(fx);
    if (&fy != &fx)
      boundary.fill(fy);
    apply_divergence(fx, fy, boundary.context().geom, out, /*cx=*/0, /*cy=*/1);
  }
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy,
                  const PreparedGridBoundarySession& boundary,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(fx, point);
    if (&fy != &fx)
      boundary.fill(fy, point);
    apply_divergence(fx, fy, boundary.context().geom, out, /*cx=*/0, /*cy=*/1);
  }
  // --- reductions (COLLECTIVE all_reduce, called on every rank; per-level field) --------------------
  Real sum_component(const MultiFab& u, int comp) const { return pops::reduce_sum(u, comp); }
  Real max_component(const MultiFab& u, int comp) const { return pops::reduce_max(u, comp); }
  Real min_component(const MultiFab& u, int comp) const { return pops::reduce_min(u, comp); }
  Real abs_sum_component(const MultiFab& u, int comp) const {
    return pops::reduce_abs_sum(u, comp);
  }
  Real sum(const MultiFab& u) const { return pops::reduce_sum(u, 0); }
  Real max(const MultiFab& u) const { return pops::reduce_max(u, 0); }
  Real min(const MultiFab& u) const { return pops::reduce_min(u, 0); }
  Real abs_sum(const MultiFab& u) const { return pops::reduce_abs_sum(u, 0); }

  void fill_boundary(MultiFab& x) const {
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(x, g.domain, eng_->transport_bc());
  }

  void fill_boundary(MultiFab& x, const ExecutionLane& lane) const {
    const Geometry g = eng_->level_geom(level_);
    fill_ghosts(x, g.domain, eng_->transport_bc(), lane);
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
                        const std::string& state_identity, const std::string& space_identity,
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
    pops::detail::AmrHistoryOps::register_history(*eng_, static_cast<std::size_t>(sys_block(owner)),
                                                  name, lag, ncomp);
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
      restore_ring_flux_(name, lag,
                         mf);  // re-publish the lagged buffer's flux strip into the live ledger
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
    pops::detail::AmrHistoryOps::register_history(*eng_, static_cast<std::size_t>(sys_block(owner)),
                                                  name, lag, ncomp);
    if (!pops::detail::AmrHistoryOps::initialized(*eng_, name))
      pops::detail::AmrHistoryOps::set_initialized(*eng_, name, true);
    MultiFab& mf = pops::detail::AmrHistoryOps::read_history(*eng_, name, lag, level_);
    if (capturing())
      restore_ring_flux_(name, lag, mf);
    return mf;
  }
  void store_history(const std::string& name, const MultiFab& value, int owner) const {
    require_history_owner_(name, owner);
    // The supported AMR histories belong to the primary macro clock.  The generated body is
    // evaluated at every child substep, but that implementation detail must not overwrite the same
    // logical ring slot at progressively later fine-level phases: doing so creates a "hierarchy
    // snapshot" whose coarse and fine slices denote different physical times and cannot be replayed
    // from one anchor.  Publish each level exactly once, at the first child window of the macro tick.
    // Explicit child-clock histories are rejected by the AMR lowering until they have their own
    // independently rotating provider.
    if (current_window_ && current_window_->begin.phase != amr::Rational(0, 1))
      return;
    // A primary-clock AMR ring rotates once per accepted MACRO step, even though its fine-level
    // value is stored once per child substep.  Its one scalar slot_dt therefore belongs to that
    // macro-step, not to whichever level happened to store last.  Using current_level_dt_ let the
    // finest level overwrite dt with dt/ref_ratio; selective restart then replayed only that
    // fraction of a macro-step.  AMR child-clock histories are rejected by the lowering until a
    // distinct per-clock ring provider exists, so the facade's installed-Program dt is the exact
    // clock authority for every supported ring here.
    pops::detail::AmrHistoryOps::store_history(*eng_, name, level_, value,
                                               static_cast<Real>(facade_->program_last_dt()));
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
    facade_->profiler_handle().record(name, std::chrono::duration<double>(t1 - t0).count());
  }

  int macro_step() const { return facade_->macro_step(); }

  // --- condensed-implicit elliptic primitives on the hierarchy (ADC-633 / ADC-637): WIRED per level ---
  // The codegen lowers a condensed-implicit (ADC-637) Program to inline block-inverse assembly kernels
  // referencing ONLY the variable `ctx`, so the SAME emitted body compiles against this context. With its
  // grid_context() (per level) + assembly_target / assembly_source (write/read redirect), those kernels
  // run the SAME assembly PER LEVEL as direct body calls (they read the level_ cursor live). On a FLAT
  // hierarchy a provider may delegate to the separately prepared level-local Krylov contract or own
  // storage, solve and publication directly. No coupling/schur call remains on either path -- the
  // generated .so carries all scheme kernels.

  /// Assembly WRITE redirection (ADC-633). On a REFINED hierarchy each assembled coefficient / RHS / flux
  /// field must live on the CURRENT level, not the level-0-bound emitted scratch: the kernel writes
  /// THROUGH here into AmrTensorElliptic's per-level buffer. On a FLAT hierarchy (no fine patch) the
  /// emitted level-0 field IS the whole system, so this is the identity (byte-for-byte the uniform path --
  /// the flat bit-parity gate). The prepared slot identity is opaque to this context and interpreted
  /// only by the selected hierarchy provider.
  MultiFab& assembly_target(MultiFab& field, std::string_view field_slot_identity) const {
    validate_prepared_field_slot(field_slot_identity, "AmrProgramContext::assembly_target");
    if (!hierarchy_tensor_solver_)
      return field;
    PreparedHierarchyTensorSolver& solver = *hierarchy_tensor_solver_;
    if (solver.execution_path() == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback)
      return field;
    if (std::find(hierarchy_tensor_assembly_field_slots_.begin(),
                  hierarchy_tensor_assembly_field_slots_.end(),
                  field_slot_identity) == hierarchy_tensor_assembly_field_slots_.end())
      throw std::invalid_argument("hierarchy assembly used an undeclared provider field slot");
    return solver.assembly_target(field_slot_identity, level_);
  }
  /// Reconstruction READ redirection (ADC-633): the fine-level reconstruction reads the level's published
  /// composite potential (the emitted level-0 solution cannot hold a fine level's phi). Flat / no fine
  /// patch: identity (returns the emitted solution). The provider-neutral slot identity remains
  /// authenticated even though the selected provider owns its published solution storage.
  MultiFab& assembly_source(MultiFab& field, std::string_view field_slot_identity) const {
    validate_prepared_field_slot(field_slot_identity, "AmrProgramContext::assembly_source");
    if (!hierarchy_tensor_solver_)
      return field;
    PreparedHierarchyTensorSolver& solver = *hierarchy_tensor_solver_;
    if (solver.execution_path() == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback)
      return field;
    if (field_slot_identity != hierarchy_tensor_solution_field_slot_)
      throw std::invalid_argument("hierarchy read used an undeclared provider solution slot");
    return solver.solution(level_);
  }
  /// Resolve a hierarchy-scoped solve value for the current publish/reconstruct pass.  Flat AMR is
  /// the identity; a refined hierarchy returns the level solution published by the one composite solve.
  MultiFab& linear_solution(MultiFab& field) const {
    if (!hierarchy_tensor_solver_)
      return field;
    PreparedHierarchyTensorSolver& solver = *hierarchy_tensor_solver_;
    return solver.execution_path() == HierarchyTensorSolverExecutionPath::DirectProvider
               ? solver.solution(level_)
               : field;
  }
  /// Resolve a provider-owned published solution when code generation authenticated that the flat
  /// execution strategy is direct. This overload avoids manufacturing unused Krylov storage merely
  /// to satisfy a fallback-shaped API.
  MultiFab& hierarchy_solution() const {
    PreparedHierarchyTensorSolver& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
      throw std::logic_error(
          "provider-owned hierarchy solution requested on a prepared Krylov fallback path");
    return solver.solution(level_);
  }
  /// Gather the authored initial guess for the current hierarchy level.  The no-argument overload is
  /// the explicit zero-guess contract; the field overload carries a per-level scalar history such as
  /// condensed-Schur phi^n.  Emitted only inside the refined gather loop.
  void stage_linear_initial_guess() const {
    PreparedHierarchyTensorSolver& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
      throw std::logic_error("hierarchy initial guess staging requires direct provider execution");
    solver.stage_initial_guess(level_, nullptr);
  }
  void stage_linear_initial_guess(const MultiFab& guess) const {
    PreparedHierarchyTensorSolver& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
      throw std::logic_error("hierarchy initial guess staging requires direct provider execution");
    solver.stage_initial_guess(level_, &guess);
  }
  /// Register a hierarchy provider carried by a verified compiled native component.  The facade
  /// authenticates and installs the declaration collectively; this context never branches on the
  /// provider identity and retains the same registry used by the subsequent generic configure call.
  void register_hierarchy_tensor_solver_provider(
      std::shared_ptr<const HierarchyTensorSolverProvider> provider) const {
    if (facade_ == nullptr || hierarchy_tensor_solver_registry_ == nullptr)
      throw std::logic_error(
          "hierarchy tensor-solver component registration requires an AMR facade");
    facade_->register_program_hierarchy_tensor_solver_provider(std::move(provider));
  }
  /// Install one exact hierarchy-solver binding. The block/component identity, operator envelope,
  /// field slots and opaque provider options are authenticated by code generation.
  void configure_hierarchy_tensor_solver(int program_block, int ncomp,
                                         const std::string& provider_identity,
                                         const std::string& plan_identity,
                                         const std::string& operator_contract_identity,
                                         const std::vector<std::string>& assembly_field_slots,
                                         const std::string& solution_field_slot,
                                         const PreparedProviderOptions& options) const {
    if (hierarchy_tensor_solver_registry_ == nullptr)
      throw std::logic_error("hierarchy tensor-solver registry is unavailable");
    if (operator_contract_identity.empty() || assembly_field_slots.empty() ||
        solution_field_slot.empty() ||
        std::any_of(assembly_field_slots.begin(), assembly_field_slots.end(),
                    [](const std::string& slot) { return slot.empty(); }) ||
        std::set<std::string>(assembly_field_slots.begin(), assembly_field_slots.end()).size() !=
            assembly_field_slots.size())
      throw std::invalid_argument("hierarchy tensor solver requires exact unique field slots");
    ExactContractBuilder selection;
    selection.text("pops.hierarchy.tensor-solver-selection")
        .scalar(std::uint32_t{1})
        .text(provider_identity)
        .text(plan_identity)
        .text(operator_contract_identity)
        .sequence(assembly_field_slots,
                  [](ExactContractBuilder& item, const std::string& slot) { item.text(slot); })
        .text(solution_field_slot)
        .bytes(options.exact_contract());
    const std::string selection_contract = std::move(selection).release();
    const std::uint64_t topology_epoch = eng_->topology_epoch();
    if (hierarchy_tensor_solver_) {
      if (hierarchy_tensor_selection_contract_ != selection_contract ||
          hierarchy_tensor_program_block_ != program_block || hierarchy_tensor_ncomp_ != ncomp)
        throw std::logic_error("hierarchy tensor solver is already configured differently");
      if (hierarchy_tensor_topology_epoch_ == topology_epoch)
        return;
    }
    HierarchyTensorSolverBuildRequest request;
    request.runtime = eng_;
    request.block = sys_block(program_block);
    request.components = ncomp;
    request.levels = eng_->nlev();
    request.level_populated.reserve(static_cast<std::size_t>(request.levels));
    request.level_distributions.reserve(static_cast<std::size_t>(request.levels));
    for (int level = 0; level < request.levels; ++level) {
      request.level_populated.push_back(
          eng_->level_state(static_cast<std::size_t>(request.block), level).box_array().size() !=
          0);
      request.level_distributions.push_back(eng_->level_is_replicated(level)
                                                ? FieldDistribution::Replicated
                                                : FieldDistribution::Distributed);
    }
    request.plan_identity = plan_identity;
    request.operator_contract_identity = operator_contract_identity;
    request.assembly_field_slots = assembly_field_slots;
    request.solution_field_slot = solution_field_slot;
    request.options = options;
    hierarchy_tensor_solver_ = prepare_hierarchy_tensor_solver_collectively(
        *hierarchy_tensor_solver_registry_, provider_identity, std::move(request));
    hierarchy_tensor_selection_contract_ = selection_contract;
    hierarchy_tensor_program_block_ = program_block;
    hierarchy_tensor_ncomp_ = ncomp;
    hierarchy_tensor_topology_epoch_ = topology_epoch;
    hierarchy_tensor_assembly_field_slots_ = assembly_field_slots;
    hierarchy_tensor_solution_field_slot_ = solution_field_slot;
  }
  OperatorEvaluationSnapshot operator_evaluation_snapshot(OperatorFingerprint authority,
                                                          const MultiFab& prototype,
                                                          OperatorFingerprint resources) const {
    if (!current_window_ || !std::isfinite(current_level_dt_) || current_level_dt_ <= 0.0)
      throw std::logic_error("AMR operator snapshot requested outside a prepared level window");
    OperatorFingerprint topology =
        ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
    ::pops::detail::fingerprint_geometry(topology, eng_->level_geom(level_));
    ::pops::detail::fingerprint_boundary(topology, eng_->transport_bc());
    ::pops::detail::fingerprint_mix(topology, "amr-level-local");
    ::pops::detail::fingerprint_mix(topology, static_cast<std::uint64_t>(level_));
    ::pops::detail::fingerprint_mix(topology, static_cast<std::uint64_t>(nlev()));
    if (operator_snapshot_revision_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR operator snapshot revision exhausted");
    const std::uint64_t revision = ++operator_snapshot_revision_;
    invalidate_active_operator_snapshot_();
    OperatorEvaluationSnapshot snapshot =
        operator_evaluation_snapshot_(authority, topology, resources, revision);
    active_operator_snapshot_revision_ = revision;
    return snapshot;
  }

  /// Recompute only the allocation-free dynamic identity. The requested evaluation revision is
  /// reproduced only while it remains the context's active mint; logical-scope entry/exit clears
  /// that authority even when the exact AMR parent window is later restored. The independent
  /// topology revision continues to change across engine-epoch or active-level transitions.
  OperatorEvaluationSnapshot probe_operator_evaluation(OperatorFingerprint authority,
                                                       OperatorFingerprint topology,
                                                       OperatorFingerprint resources,
                                                       std::uint64_t revision) const {
    const std::uint64_t probe_revision =
        revision == active_operator_snapshot_revision_ ? revision : UINT64_C(0);
    return operator_evaluation_snapshot_(authority, topology, resources, probe_revision);
  }

 private:
  void invalidate_active_operator_snapshot_() const noexcept {
    active_operator_snapshot_revision_ = 0;
  }

  std::uint64_t operator_topology_revision_() const {
    const std::uint64_t epoch = eng_->topology_epoch();
    if (observed_operator_topology_epoch_ != epoch || observed_operator_level_ != level_) {
      if (operator_topology_revision_counter_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("AMR operator topology revision exhausted");
      observed_operator_topology_epoch_ = epoch;
      observed_operator_level_ = level_;
      ++operator_topology_revision_counter_;
    }
    return operator_topology_revision_counter_;
  }

  OperatorEvaluationSnapshot operator_evaluation_snapshot_(OperatorFingerprint authority,
                                                           OperatorFingerprint topology,
                                                           OperatorFingerprint resources,
                                                           std::uint64_t revision) const {
    if (!current_window_ || !std::isfinite(current_level_dt_) || current_level_dt_ <= 0.0)
      throw std::logic_error("AMR operator snapshot requested outside a prepared level window");
    const amr::ClockStamp clock = evaluation_clock_();
    return {authority,
            revision,
            clock.macro_step,
            clock.phase.numerator,
            clock.phase.denominator,
            std::bit_cast<std::uint64_t>(current_level_dt_),
            std::bit_cast<std::uint64_t>(clock.physical_time),
            operator_topology_revision_(),
            topology,
            resources};
  }

 public:
  /// AMR counterpart of ProgramContext's compiled-artifact capability. The authority is checked
  /// against the level-local evaluation snapshot before the unverified hot path can be enabled.
  ::pops::detail::AuthenticatedProgramApplyToken authenticated_program_apply_token(
      OperatorFingerprint authority) const {
    if (facade_ == nullptr || !facade_->program_owns_operator_authority(authority))
      throw std::invalid_argument(
          "compiled AMR Program requested an operator authority not owned by its installed "
          "artifact");
    return ::pops::detail::AuthenticatedProgramApplyToken(authority);
  }

  /// Level-local prepared Krylov route. Direct composite hierarchy solves remain a separate typed
  /// backend and never discard an authored prepared operator.
  SolveReport solve_prepared_linear(const PreparedAffineLinearProblem& problem,
                                    KrylovWorkspace& workspace, MultiFab& sol, const MultiFab& rhs,
                                    const KrylovControls& controls) const {
    return pops::solve_prepared_affine(problem, workspace, sol, rhs, controls);
  }

  /// Solve the configured operator through the provider-owned direct hierarchy path. A provider may
  /// select this path for one or many levels; topology shape and solver family remain provider-private.
  SolveReport solve_hierarchy_tensor(int program_block, int ncomp, Real rel_tol, Real abs_tol,
                                     int max_iter) const {
    require_tensor_binding(program_block, ncomp);
    PreparedHierarchyTensorSolver& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidInput, SolveAction::kRejectAttempt);
      return report;
    }
    const HierarchyTensorSolveControls controls{rel_tol, abs_tol, max_iter};
    return solve_prepared_hierarchy_tensor_collectively(solver, controls);
  }

  // --- named-flux primitive: DEFERRED on AMR, fail loud ----------------------------------------------
  // The named-flux divergence is a ProgramContext method the codegen can lower for a named-flux (ADC-419)
  // Program. The AMR named-flux -div path is NOT wired (out of ADC-633 scope); it fails loud so the SAME
  // lowered body compiles on target='amr_system' and throws only when the op is REACHED at run.
  void neg_div_flux_into(MultiFab& /*r*/, MultiFab& /*fx*/, MultiFab& /*fy*/) const {
    deferred_op(
        "neg_div_flux_into",
        "a named-flux (-div F) Program on AMR is deferred; use System, or a native AMR block "
        "whose flux IR runs through the level RHS.");
  }

  // --- scheduler value cache: DEFERRED on AMR, fail loud ---------------------------------------------
  // The codegen lowers a held / scheduled field-solve node (ADC-458) against these cache seams. The
  // CacheManager that backs them is owned by System (per-installed-Program, keyed by node id); AMR
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
    deferred_op("scheduler_error",
                ("the scheduled error policy reached its off-cadence branch (" + what +
                 "); scheduled Programs on AMR are deferred; use System"));
  }

 private:
  enum class ResidualCapture { FullRate, FluxOnly };
  using FluxContribution = AmrProgramFluxContribution;
  static void require_supported_program_refinement_ratios_(const AmrRuntime& eng) {
    for (int child = 1; child < eng.nlev(); ++child) {
      const int parent_refinement = eng.level_refinement(child - 1);
      const int child_refinement = eng.level_refinement(child);
      const int ratio = child_refinement / parent_refinement;
      if (child_refinement != parent_refinement * ratio || ratio != kAmrRefRatio)
        throw std::runtime_error(
            "AmrProgramContext: the native Program reflux/average-down provider supports only "
            "refinement ratio " +
            std::to_string(kAmrRefRatio) + "; transition " + std::to_string(child - 1) + "->" +
            std::to_string(child) + " resolved ratio " + std::to_string(ratio) +
            ". Select a provider whose declared capabilities cover that transition.");
    }
  }
  static std::runtime_error block_map_error_(std::string message) {
    return std::runtime_error(std::move(message));
  }

  [[noreturn]] static void throw_field_solve_failure_(const SolveReport& report,
                                                      const char* detail) {
    if (report.action == SolveAction::kRejectAttempt)
      throw StepAttemptRejected(report.status, "prepared field evaluation", detail);
    throw std::runtime_error(std::string("prepared field evaluation failed: ") +
                             report.status_name() + " (" + detail + ")");
  }

  MultiFab& stage_state_scratch_for_(int program_block, int level,
                                     const MultiFab& prototype) const {
    const std::pair<int, int> key{program_block, level};
    auto insertion = stage_state_scratch_.try_emplace(key, prototype.box_array(), prototype.dmap(),
                                                      prototype.ncomp(), prototype.n_grow());
    MultiFab& scratch = insertion.first->second;
    const bool compatible = scratch.box_array().boxes() == prototype.box_array().boxes() &&
                            scratch.dmap().ranks() == prototype.dmap().ranks() &&
                            scratch.ncomp() == prototype.ncomp() &&
                            scratch.n_grow() == prototype.n_grow();
    if (!compatible)
      scratch =
          MultiFab(prototype.box_array(), prototype.dmap(), prototype.ncomp(), prototype.n_grow());
    return scratch;
  }

  /// Fail loud for an op the codegen can emit but the installed AMR Program path does not wire (named-flux /
  /// scheduled Programs). [[noreturn]] so a non-void stub needs no dummy return -- the caller's signature
  /// stays byte-faithful to ProgramContext (the duck-typing requirement) without fabricating a value. @p
  /// op names the seam; @p detail names the alternative (System or the native AMR route) so the message
  /// is actionable, mirroring the inline fail-loud stubs above.
  [[noreturn]] static void deferred_op(const char* op, std::string detail) {
    throw std::runtime_error(std::string("AmrProgramContext: ") + op +
                             " is not wired on the AMR Program path; " + std::move(detail));
  }

  PreparedHierarchyTensorSolver& configured_hierarchy_tensor_solver_() const {
    if (!hierarchy_tensor_solver_)
      throw std::logic_error("hierarchy tensor solver must be configured before hierarchy access");
    return *hierarchy_tensor_solver_;
  }

  void require_tensor_binding(int program_block, int ncomp) const {
    if (!hierarchy_tensor_solver_ || hierarchy_tensor_program_block_ != program_block ||
        hierarchy_tensor_ncomp_ != ncomp)
      throw std::logic_error(
          "hierarchy tensor block/component binding does not match the configured solver");
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
    const auto boundary_point = boundary_point_(rate_id);
    if (active_parent_ && active_parent_->child_level == level_) {
      const amr::Rational target_phase = active_parent_->child_window.begin.phase +
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
          {1, active_parent_->parent_window.end.physical_time},
          {0, target_physical},
          alpha.numerator,
          alpha.denominator};
      const MultiFab& old_parent = active_parent_->old_states.at(static_cast<std::size_t>(b));
      const MultiFab& new_parent = active_parent_->new_states.at(static_cast<std::size_t>(b));
      if (mode == ResidualCapture::FluxOnly)
        eng_->level_neg_div_flux_capture_into_temporal(sb, level_, boundary_point, u, r, Fx, Fy,
                                                       old_parent, new_parent, target);
      else
        eng_->level_rhs_capture_into_temporal(sb, level_, boundary_point, u, r, Fx, Fy, old_parent,
                                              new_parent, target);
    } else if (mode == ResidualCapture::FluxOnly) {
      eng_->level_neg_div_flux_capture_into(sb, level_, boundary_point, u, r, Fx, Fy);
    } else {
      eng_->level_rhs_capture_into(sb, level_, boundary_point, u, r, Fx, Fy);
    }
    EdgeFlux ef;
    // COARSE role: the level-k coarse flux at the faces bordering each level-(k+1) patch (a child exists).
    if (level_ + 1 < nlev()) {
      const MultiFab& child = eng_->level_state(sb, level_ + 1);
      pops::detail::CoarseRoleScratch& scratch = coarse_role_scratch_[{sb, level_}];
      pops::detail::sample_coarse_role_strip(ref, Fx, Fy, child, eng_->level_is_replicated(level_),
                                             eng_->topology_epoch(), nc, scratch, ef);
    }
    // FINE role: the coarse-face-averaged level-k flux at level-k's own patch edges (level_ borders k-1).
    if (level_ >= 1)
      pops::detail::sample_fine_role_strip(ref, Fx, Fy, nc, ef);
    flux_ledger_[key_(&r)] = ef;  // OVERWRITE-set (a residual op initialises the strip)
    flux_contributions_[key_(&r)] = {
        FluxContribution{rate_id, amr::Rational(1, 1), 0, 0.0, evaluation_clock_(), std::move(ef)}};
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
    const auto saved_program_accepted_state = facade_->program_accepted_state();
    const auto saved_flux = flux_ledger_;
    const auto saved_flux_contributions = flux_contributions_;
    const auto saved_rate_provenance = rate_provenance_;
    const auto saved_sync_report = sync_report_;
    const auto saved_accepted_flux_report = accepted_flux_report_;
    const auto saved_accepted_sync_report = accepted_sync_report_;
    const auto saved_ring = ring_flux_;
    const auto saved_ring_contributions = ring_flux_contributions_;
    const auto saved_ring_init = ring_flux_init_;
    const auto saved_ring_clocks = ring_clocks_;
    const auto saved_ring_identities = ring_identities_;
    const auto saved_history_owners = history_owners_;
    const auto saved_history_states = history_state_ids_;
    const auto saved_history_spaces = history_space_ids_;
    const auto saved_history_clocks = history_clock_ids_;
    const auto saved_history_interpolations = history_interpolation_ids_;
    const auto saved_history_flux_topology = history_flux_topology_;
    const int saved_history_flux_topology_rebind_count = history_flux_topology_rebind_count_;
    const auto saved_clocks = level_clocks_;
    const auto saved_clock_schedule = clock_schedule_;
    const auto saved_live = live_state_rings_;
    const bool saved_rotate = rotate_pending_;
    const auto saved_default_solve_report = default_solve_report_;
    const auto saved_named_solve_reports = named_solve_reports_;
    const int saved_level = level_;
    const amr::Rational saved_stage_time = stage_time_;
    const auto saved_active_parent = active_parent_;
    const auto saved_window = current_window_;
    const auto saved_sync_clock = current_sync_clock_;
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
      current_sync_clock_ = root.end;
      couple_levels();
      for (int k = 0; k < nlev(); ++k)
        level_clocks_[static_cast<std::size_t>(k)] =
            amr::ClockStamp{k, macro_step() + 1, amr::Rational(0, 1), t0 + dt};
      clock_schedule_.restore_accepted_ticks(clock_schedule_.accepted_ticks(macro_step() + 1),
                                             macro_step() + 1);
      conservative_ledger_.commit();
      accepted_flux_report_.clear();
      accepted_flux_report_.reserve(conservative_ledger_.entries().size());
      for (const auto& entry : conservative_ledger_.entries())
        accepted_flux_report_.push_back({entry.key, entry.measure});
      accepted_sync_report_.clear();
      accepted_sync_report_.reserve(sync_report_.size());
      for (const SyncEvent& event : sync_report_)
        accepted_sync_report_.push_back({event.parent_level, event.child_level, event.block,
                                         event.phase == SyncPhase::Reflux ? 0 : 1, event.clock});
      conservative_ledger_.clear();
      level_ = saved_level;
      stage_time_ = saved_stage_time;
      active_parent_ = saved_active_parent;
      current_window_ = saved_window;
      current_sync_clock_ = saved_sync_clock;
      current_level_dt_ = saved_level_dt;
      publish_program_accepted_state_();
    } catch (...) {
      if (conservative_ledger_.in_transaction())
        conservative_ledger_.rollback();
      else if (!conservative_ledger_.empty())
        conservative_ledger_.clear();
      eng_->restore_step_snapshot(saved_engine);
      facade_->restore_program_accepted_state(saved_program_accepted_state);
      accepted_state_revision_ = facade_->program_accepted_state_revision();
      flux_ledger_ = saved_flux;
      flux_contributions_ = saved_flux_contributions;
      rate_provenance_ = saved_rate_provenance;
      sync_report_ = saved_sync_report;
      accepted_flux_report_ = saved_accepted_flux_report;
      accepted_sync_report_ = saved_accepted_sync_report;
      ring_flux_ = saved_ring;
      ring_flux_contributions_ = saved_ring_contributions;
      ring_flux_init_ = saved_ring_init;
      ring_clocks_ = saved_ring_clocks;
      ring_identities_ = saved_ring_identities;
      history_owners_ = saved_history_owners;
      history_state_ids_ = saved_history_states;
      history_space_ids_ = saved_history_spaces;
      history_clock_ids_ = saved_history_clocks;
      history_interpolation_ids_ = saved_history_interpolations;
      history_flux_topology_ = saved_history_flux_topology;
      history_flux_topology_rebind_count_ = saved_history_flux_topology_rebind_count;
      level_clocks_ = saved_clocks;
      clock_schedule_ = saved_clock_schedule;
      live_state_rings_ = saved_live;
      rotate_pending_ = saved_rotate;
      default_solve_report_ = saved_default_solve_report;
      named_solve_reports_ = saved_named_solve_reports;
      level_ = saved_level;
      stage_time_ = saved_stage_time;
      active_parent_ = saved_active_parent;
      current_window_ = saved_window;
      current_sync_clock_ = saved_sync_clock;
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
      if (clock.level != level || clock.macro_step < 0 || clock.phase != amr::Rational(0, 1) ||
          !std::isfinite(clock.physical_time))
        throw std::runtime_error(
            "AMR Program accepted state contains a non-accepted or misqualified level clock");
    }
    const std::int64_t accepted_step =
        state.level_clocks.empty() ? macro_step() : state.level_clocks.front().macro_step;
    if (state.logical_clock_ticks != clock_schedule_.accepted_ticks(accepted_step))
      throw std::runtime_error(
          "AMR Program accepted state logical-clock ticks differ from the installed schedule");
    if (state.history_owners.size() != state.history_states.size() ||
        state.history_owners.size() != state.history_spaces.size() ||
        state.history_owners.size() != state.history_clocks.size() ||
        state.history_owners.size() != state.history_interpolations.size() ||
        state.history_owners.size() != state.ring_clocks.size() ||
        state.history_owners.size() != state.ring_identities.size() ||
        state.history_owners.size() != state.ring_flux.size() ||
        state.history_owners.size() != state.ring_flux_contributions.size() ||
        state.history_owners.size() != state.ring_flux_initialized.size())
      throw std::runtime_error(
          "AMR Program accepted state has inconsistent qualified history registries");
    for (const auto& [name, owner] : state.history_owners) {
      const auto clocks = state.ring_clocks.find(name);
      const auto identities = state.ring_identities.find(name);
      const auto fluxes = state.ring_flux.find(name);
      const auto contributions = state.ring_flux_contributions.find(name);
      const auto initialized = state.ring_flux_initialized.find(name);
      if (state.history_states.count(name) == 0 || state.history_spaces.count(name) == 0 ||
          state.history_clocks.count(name) == 0 || state.history_interpolations.count(name) == 0 ||
          clocks == state.ring_clocks.end() || identities == state.ring_identities.end() ||
          fluxes == state.ring_flux.end() || contributions == state.ring_flux_contributions.end() ||
          initialized == state.ring_flux_initialized.end())
        throw std::runtime_error("AMR Program accepted state lacks history qualification for '" +
                                 name + "'");
      (void)sys_block(owner);  // prove the qualified owner is still installed by name.
      const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
      if (static_cast<int>(clocks->second.size()) != depth ||
          static_cast<int>(identities->second.size()) != depth ||
          static_cast<int>(fluxes->second.size()) != depth ||
          static_cast<int>(contributions->second.size()) != depth ||
          initialized->second.size() != static_cast<std::size_t>(nlev()))
        throw std::runtime_error("AMR Program accepted state has wrong ring depth for '" + name +
                                 "'");
      for (int slot = 0; slot < depth; ++slot) {
        const auto& slot_clocks = clocks->second[static_cast<std::size_t>(slot)];
        const auto& slot_identities = identities->second[static_cast<std::size_t>(slot)];
        const auto& slot_fluxes = fluxes->second[static_cast<std::size_t>(slot)];
        const auto& slot_contributions = contributions->second[static_cast<std::size_t>(slot)];
        if (slot_clocks.size() != static_cast<std::size_t>(nlev()) ||
            slot_identities.size() != static_cast<std::size_t>(nlev()) ||
            slot_fluxes.size() != static_cast<std::size_t>(nlev()) ||
            slot_contributions.size() != static_cast<std::size_t>(nlev()))
          throw std::runtime_error("AMR Program accepted state has wrong level axis for history '" +
                                   name + "'");
        for (int level = 0; level < nlev(); ++level) {
          const auto& identity = slot_identities[static_cast<std::size_t>(level)];
          const auto& exact_contributions = slot_contributions[static_cast<std::size_t>(level)];
          if (!identity) {
            if (!exact_contributions.empty())
              throw std::runtime_error(
                  "AMR Program accepted state has history flux without a qualified identity");
            continue;
          }
          const amr::ClockStamp& clock = slot_clocks[static_cast<std::size_t>(level)];
          if (identity->owner != "program.block." + std::to_string(owner) ||
              identity->state != state.history_states.at(name) ||
              identity->space != state.history_spaces.at(name) || identity->level != level ||
              !(identity->clock == clock))
            throw std::runtime_error(
                "AMR Program accepted state has mismatched identity for history '" + name + "'");
          for (const FluxContribution& contribution : exact_contributions)
            if (contribution.rate_id < 0 || contribution.dt_power < 0 ||
                contribution.dt_power > 1 ||
                (contribution.dt_power == 0 && contribution.duration != 0.0) ||
                (contribution.dt_power == 1 &&
                 (!(contribution.duration > 0.0) || !std::isfinite(contribution.duration))) ||
                contribution.evaluation_clock.level != level ||
                !std::isfinite(contribution.evaluation_clock.physical_time))
              throw std::runtime_error(
                  "AMR Program accepted state has an invalid exact history flux contribution");
        }
      }
    }
    for (const AmrProgramFluxAuditEntry& entry : state.accepted_flux_ledger)
      if (entry.key.owner.empty() || entry.key.state.empty() || entry.key.rate.empty() ||
          entry.key.flux.empty() || entry.key.level < 0 ||
          entry.key.clock.level != entry.key.level || !(entry.measure.face_measure > 0.0) ||
          !std::isfinite(entry.measure.face_measure) || !(entry.measure.substep_duration > 0.0) ||
          !std::isfinite(entry.measure.substep_duration) ||
          !std::isfinite(entry.key.clock.physical_time))
        throw std::runtime_error("AMR Program accepted state has an invalid flux-ledger report");
    for (const AmrProgramSyncEvent& event : state.accepted_sync)
      if (event.parent_level < 0 || event.child_level != event.parent_level + 1 ||
          event.block < 0 || (event.phase != 0 && event.phase != 1) ||
          event.clock.level != event.parent_level || !std::isfinite(event.clock.physical_time))
        throw std::runtime_error(
            "AMR Program accepted state has an invalid synchronization report");
  }

  AmrProgramAcceptedState accepted_state_() const {
    AmrProgramAcceptedState state;
    state.level_clocks = level_clocks_;
    const std::int64_t accepted_step =
        level_clocks_.empty() ? macro_step() : level_clocks_.front().macro_step;
    state.logical_clock_ticks = clock_schedule_.accepted_ticks(accepted_step);
    state.history_owners = history_owners_;
    state.history_states = history_state_ids_;
    state.history_spaces = history_space_ids_;
    state.history_clocks = history_clock_ids_;
    state.history_interpolations = history_interpolation_ids_;
    state.ring_clocks = ring_clocks_;
    state.ring_identities = ring_identities_;
    state.ring_flux = ring_flux_;
    state.ring_flux_contributions = ring_flux_contributions_;
    state.ring_flux_initialized = ring_flux_init_;
    state.accepted_flux_ledger = accepted_flux_report_;
    state.accepted_sync = accepted_sync_report_;
    // A flat hierarchy captures no C/F flux strips, and a declared ring may still be cold. Persist
    // those cases as explicit zero/empty axes rather than omitting semantic registry entries: strict
    // restore can then distinguish "no flux exists" from a truncated checkpoint.
    for (const auto& [name, owner] : history_owners_) {
      (void)owner;
      const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
      auto& clocks = state.ring_clocks[name];
      auto& identities = state.ring_identities[name];
      auto& fluxes = state.ring_flux[name];
      auto& contributions = state.ring_flux_contributions[name];
      clocks.resize(static_cast<std::size_t>(depth));
      identities.resize(static_cast<std::size_t>(depth));
      fluxes.resize(static_cast<std::size_t>(depth));
      contributions.resize(static_cast<std::size_t>(depth));
      for (int slot = 0; slot < depth; ++slot) {
        clocks[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
        identities[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
        fluxes[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
        contributions[static_cast<std::size_t>(slot)].resize(static_cast<std::size_t>(nlev()));
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
    const std::int64_t accepted_step =
        level_clocks_.empty() ? macro_step() : level_clocks_.front().macro_step;
    clock_schedule_.restore_accepted_ticks(state.logical_clock_ticks, accepted_step);
    history_owners_ = std::move(state.history_owners);
    history_state_ids_ = std::move(state.history_states);
    history_space_ids_ = std::move(state.history_spaces);
    history_clock_ids_ = std::move(state.history_clocks);
    history_interpolation_ids_ = std::move(state.history_interpolations);
    ring_clocks_ = std::move(state.ring_clocks);
    ring_identities_ = std::move(state.ring_identities);
    ring_flux_ = std::move(state.ring_flux);
    ring_flux_contributions_ = std::move(state.ring_flux_contributions);
    ring_flux_init_ = std::move(state.ring_flux_initialized);
    history_flux_topology_ = history_flux_topology_snapshot_();
    accepted_flux_report_ = std::move(state.accepted_flux_ledger);
    accepted_sync_report_ = std::move(state.accepted_sync);
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
  void advance_level_(int level, const amr::ClockWindow& window, double local_dt,
                      Body& body) const {
    const auto saved_window = current_window_;
    const double saved_dt = current_level_dt_;
    current_window_ = window;
    set_level(level);
    std::vector<MultiFab> old_states;
    old_states.reserve(static_cast<std::size_t>(n_blocks()));
    for (int b = 0; b < n_blocks(); ++b)
      old_states.push_back(eng_->level_state(static_cast<std::size_t>(sys_block(b)), level));

    stage_time_ = amr::Rational(0, 1);
    // The authored step duration is the numerical authority.  Reconstructing it from accumulated
    // physical timestamps loses one ulp as soon as `(t + dt) - t != dt`; a flat hierarchy would then
    // execute a different AB/RK combine from System even though its exact clock phase is identical.
    // Child durations are derived below from this authoritative value and exact Rational spans.
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

    const amr::ParentChildClockRelation& relation = eng_->parent_child_temporal_relation(level + 1);
    const std::optional<ActiveParentWindow> saved_parent = active_parent_;
    const amr::Rational parent_span = window.end.phase - window.begin.phase;
    for (const amr::ChildSubstep& substep : relation.partition(window)) {
      active_parent_ =
          ActiveParentWindow{level + 1, window, substep.window, old_states, new_states};
      const amr::Rational child_span = substep.window.end.phase - substep.window.begin.phase;
      const double child_dt = local_dt * (child_span / parent_span).value();
      advance_level_(level + 1, substep.window, child_dt, body);
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
    (void)dt;
    (void)clock;
    for (int b = 0; b < n_blocks(); ++b) {
      const MultiFab* state = &eng_->level_state(static_cast<std::size_t>(sys_block(b)), level);
      const auto found = flux_ledger_.find({level, static_cast<const void*>(state)});
      if (found == flux_ledger_.end())
        continue;
      const auto contributions = flux_contributions_.find({level, static_cast<const void*>(state)});
      if (contributions == flux_contributions_.end() || contributions->second.empty())
        throw std::runtime_error(
            "AMR conservative state has numerical flux but no exact Program contribution ledger");
      const Geometry geometry = eng_->level_geom(level);
      const std::pair<amr::FluxOrientation, double> directions[] = {
          {amr::FluxOrientation::XMinus, static_cast<double>(geometry.dy())},
          {amr::FluxOrientation::XPlus, static_cast<double>(geometry.dy())},
          {amr::FluxOrientation::YMinus, static_cast<double>(geometry.dx())},
          {amr::FluxOrientation::YPlus, static_cast<double>(geometry.dx())}};
      for (const FluxContribution& contribution : contributions->second) {
        if (contribution.rate_id < 0 || contribution.dt_power != 1 ||
            !(contribution.duration > 0.0))
          throw std::runtime_error(
              "AMR conservative contribution is not exactly weight*dt*flux; artifact lowering "
              "must prove every tableau/program weight");
        int direction_id = 0;
        for (const auto& [orientation, face_measure] : directions)
          conservative_ledger_.accumulate(
              {"program.block." + std::to_string(b), "conservative_state",
               "program.rate.node." + std::to_string(contribution.rate_id),
               "default_flux.orientation." + std::to_string(direction_id++), level,
               contribution.evaluation_clock},
              {contribution.weight, orientation, face_measure, contribution.duration},
              orientation_payload_(contribution.payload, orientation));
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
    for (auto it = flux_contributions_.begin(); it != flux_contributions_.end();) {
      if (it->first.first == level)
        it = flux_contributions_.erase(it);
      else
        ++it;
    }
  }

  EdgeFlux reflux_flux_from_ledger_(int block, int level) const {
    const std::string owner = "program.block." + std::to_string(block);
    EdgeFlux result;
    for (const auto& entry : conservative_ledger_.entries()) {
      if (entry.key.owner != owner || entry.key.state != "conservative_state" ||
          entry.key.level != level || entry.key.flux.rfind("default_flux.orientation.", 0) != 0)
        continue;
      pops::detail::edge_flux_axpy(
          result, static_cast<Real>(amr::numerical_reflux_scale(entry.measure)), entry.payload);
    }
    return result;
  }

  std::pair<int, const void*> key_(const MultiFab* mf) const {
    return {level_, static_cast<const void*>(mf)};
  }

  amr::ClockStamp evaluation_clock_() const {
    if (!current_window_)
      throw std::runtime_error("AMR flux evaluation has no active exact clock window");
    const amr::Rational span = current_window_->end.phase - current_window_->begin.phase;
    const amr::Rational phase = current_window_->begin.phase + stage_time_ * span;
    const double physical = current_window_->begin.physical_time +
                            stage_time_.value() * (current_window_->end.physical_time -
                                                   current_window_->begin.physical_time);
    return {level_, current_window_->begin.macro_step, phase, physical};
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_point_(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !current_window_ || !std::isfinite(current_level_dt_) ||
        current_level_dt_ <= 0.0)
      throw std::runtime_error("AMR boundary evaluation has no prepared clock window/dt");
    const amr::ClockStamp stamp = evaluation_clock_();
    const auto phase = current_window_->begin.phase;
    if (phase.numerator < std::numeric_limits<int>::min() ||
        phase.numerator > std::numeric_limits<int>::max())
      throw std::overflow_error("AMR boundary substep identity exceeds int range");
    return {primary_clock_,
            stamp.macro_step,
            level_,
            static_cast<int>(phase.numerator),
            stage,
            stage_time_,
            current_level_dt_,
            stamp.physical_time};
  }

  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument(
          "AMR Program rate evaluation requires a non-negative authored node identity");
  }

  static void require_group_identity_(int group_id) {
    if (group_id < 0)
      throw std::invalid_argument(
          "AMR Program RHS group requires a non-negative authored group identity");
  }

  static std::vector<FluxContribution> scale_contributions_(
      const std::vector<FluxContribution>& source, Real dt,
      std::initializer_list<ExactCoefficientTerm> exact) {
    std::vector<FluxContribution> result;
    result.reserve(source.size() * exact.size());
    for (const FluxContribution& contribution : source)
      for (const ExactCoefficientTerm& term : exact) {
        if (term.dt_power < 0 || term.denominator <= 0)
          throw std::runtime_error("AMR conservative coefficient metadata is invalid");
        const amr::Rational factor(term.numerator, term.denominator);
        if (factor.numerator == 0)
          continue;
        FluxContribution scaled = contribution;
        scaled.weight = scaled.weight * factor;
        if (term.dt_power > 0) {
          if (!(dt > Real(0)))
            throw std::runtime_error("AMR conservative coefficient requires a positive dt");
          if (scaled.dt_power > 0 && scaled.duration != static_cast<double>(dt))
            throw std::runtime_error(
                "AMR conservative coefficient mixes distinct duration authorities");
          if (scaled.dt_power == 0)
            scaled.duration = static_cast<double>(dt);
        }
        scaled.dt_power += term.dt_power;
        result.push_back(std::move(scaled));
      }
    return result;
  }

  static std::vector<FluxContribution> scale_constant_contributions_(
      const std::vector<FluxContribution>& source, Real coefficient) {
    if (coefficient == Real(0))
      return {};
    if (coefficient != Real(1) && coefficient != Real(-1))
      throw std::runtime_error(
          "AMR conservative combine lacks exact coefficient metadata; artifact lowering must "
          "supply the authored polynomial");
    const ExactCoefficientTerm term{0, coefficient == Real(1) ? 1 : -1, 1};
    return scale_contributions_(source, Real(1), {term});
  }

  void append_scaled_contributions_(std::vector<FluxContribution>& destination,
                                    const MultiFab& source, Real dt,
                                    std::initializer_list<ExactCoefficientTerm> exact) const {
    const auto found = flux_contributions_.find(key_(&source));
    if (found == flux_contributions_.end())
      return;
    std::vector<FluxContribution> scaled = scale_contributions_(found->second, dt, exact);
    destination.insert(destination.end(), std::make_move_iterator(scaled.begin()),
                       std::make_move_iterator(scaled.end()));
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
    const auto contributions = flux_contributions_.find(key_(&r));
    if (contributions != flux_contributions_.end()) {
      std::vector<FluxContribution> scaled =
          scale_constant_contributions_(contributions->second, a);
      auto& destination = flux_contributions_[key_(&u)];
      destination.insert(destination.end(), std::make_move_iterator(scaled.begin()),
                         std::make_move_iterator(scaled.end()));
    }
  }

  void ledger_axpy_exact_(MultiFab& u, Real a, const MultiFab& r, Real dt,
                          std::initializer_list<ExactCoefficientTerm> exact) const {
    const auto it = flux_ledger_.find(key_(&r));
    if (it == flux_ledger_.end())
      return;
    pops::detail::edge_flux_axpy(flux_ledger_[key_(&u)], a, it->second);
    append_scaled_contributions_(flux_contributions_[key_(&u)], r, dt, exact);
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
    std::vector<FluxContribution> contributions;
    const auto cx = flux_contributions_.find(key_(&x));
    if (cx != flux_contributions_.end()) {
      auto scaled = scale_constant_contributions_(cx->second, a);
      contributions.insert(contributions.end(), std::make_move_iterator(scaled.begin()),
                           std::make_move_iterator(scaled.end()));
    }
    const auto cy = flux_contributions_.find(key_(&y));
    if (cy != flux_contributions_.end()) {
      auto scaled = scale_constant_contributions_(cy->second, b);
      contributions.insert(contributions.end(), std::make_move_iterator(scaled.begin()),
                           std::make_move_iterator(scaled.end()));
    }
    flux_contributions_[key_(&z)] = std::move(contributions);
  }

  void ledger_lincomb_exact_(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y,
                             Real dt, std::initializer_list<ExactCoefficientTerm> exact_a,
                             std::initializer_list<ExactCoefficientTerm> exact_b) const {
    EdgeFlux out;
    const auto itx = flux_ledger_.find(key_(&x));
    if (itx != flux_ledger_.end())
      pops::detail::edge_flux_axpy(out, a, itx->second);
    const auto ity = flux_ledger_.find(key_(&y));
    if (ity != flux_ledger_.end())
      pops::detail::edge_flux_axpy(out, b, ity->second);
    flux_ledger_[key_(&z)] = std::move(out);
    std::vector<FluxContribution> contributions;
    append_scaled_contributions_(contributions, x, dt, exact_a);
    append_scaled_contributions_(contributions, y, dt, exact_b);
    flux_contributions_[key_(&z)] = std::move(contributions);
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
    auto& contribution_ring = ring_flux_contributions_[name];
    const int depth = pops::detail::AmrHistoryOps::depth(*eng_, name);
    if (static_cast<int>(ring.size()) != depth)
      ring.assign(static_cast<std::size_t>(depth < 1 ? 1 : depth), {});
    if (static_cast<int>(contribution_ring.size()) != depth)
      contribution_ring.assign(static_cast<std::size_t>(depth < 1 ? 1 : depth), {});
    for (auto& slot : ring)
      if (static_cast<int>(slot.size()) < nlev())
        slot.assign(static_cast<std::size_t>(nlev()), {});
    for (auto& slot : contribution_ring)
      if (static_cast<int>(slot.size()) < nlev())
        slot.assign(static_cast<std::size_t>(nlev()), {});
    const auto it = flux_ledger_.find(key_(&value));
    EdgeFlux ef = (it != flux_ledger_.end()) ? it->second : EdgeFlux{};
    const auto contribution = flux_contributions_.find(key_(&value));
    const std::vector<FluxContribution> exact = contribution == flux_contributions_.end()
                                                    ? std::vector<FluxContribution>{}
                                                    : contribution->second;
    // PER-RING PER-LEVEL COLD START (mirror of AmrHistoryOps::store_history): the FIRST store of a (name,
    // level) broadcasts into EVERY deeper slot, so a multistep step 0 reads the same flux at each lag; from
    // then on only slot 0 is written (the ring rotate carries the older slots). Tracked with our own flag
    // set, independent of the engine's hist_init_ (which is already set by the engine store above).
    std::vector<char>& init = ring_flux_init_[name];
    if (static_cast<int>(init.size()) < nlev())
      init.assign(static_cast<std::size_t>(nlev()), 0);
    ring[0][static_cast<std::size_t>(level_)] = ef;
    contribution_ring[0][static_cast<std::size_t>(level_)] = exact;
    if (!init[static_cast<std::size_t>(level_)]) {
      for (std::size_t s = 1; s < ring.size(); ++s) {
        ring[s][static_cast<std::size_t>(level_)] = ef;
        contribution_ring[s][static_cast<std::size_t>(level_)] = exact;
      }
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
    const auto contributions = ring_flux_contributions_.find(name);
    if (contributions != ring_flux_contributions_.end() &&
        lag < static_cast<int>(contributions->second.size()) &&
        level_ < static_cast<int>(contributions->second[static_cast<std::size_t>(lag)].size()))
      flux_contributions_[key_(&mf)] =
          contributions->second[static_cast<std::size_t>(lag)][static_cast<std::size_t>(level_)];
  }
  /// Rotate ring_flux_ one slot in lockstep with the ADC-631 history ring (called by couple_levels when
  /// the deferred rotate fires). O(1) vector-of-strip swaps, the exact chain AmrHistoryOps::rotate uses.
  void rotate_ring_flux_() const {
    for (auto& [name, ring] : ring_flux_)
      for (std::size_t s = ring.size(); s-- > 1;)
        std::swap(ring[s], ring[s - 1]);
    for (auto& [name, ring] : ring_flux_contributions_) {
      (void)name;
      for (std::size_t s = ring.size(); s-- > 1;)
        std::swap(ring[s], ring[s - 1]);
    }
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

  HistoryFluxTopology history_flux_topology_snapshot_() const {
    HistoryFluxTopology result;
    result.epoch = eng_->topology_epoch();
    result.boxes.reserve(static_cast<std::size_t>(nlev()));
    result.owners.reserve(static_cast<std::size_t>(nlev()));
    for (int level = 0; level < nlev(); ++level) {
      const MultiFab& state = eng_->level_state(0, level);
      result.boxes.push_back(state.box_array().boxes());
      result.owners.push_back(state.dmap().ranks());
    }
    return result;
  }

  static bool same_history_flux_layout_(const HistoryFluxTopology& lhs,
                                        const HistoryFluxTopology& rhs) {
    return lhs.boxes == rhs.boxes && lhs.owners == rhs.owners;
  }

  static bool same_history_flux_topology_(const HistoryFluxTopology& lhs,
                                          const HistoryFluxTopology& rhs) {
    return lhs.epoch == rhs.epoch && same_history_flux_layout_(lhs, rhs);
  }

  static void clear_parent_interface_(EdgeFlux& flux) { flux.coarse.clear(); }
  static void clear_child_interface_(EdgeFlux& flux) { flux.fine.clear(); }

  bool has_lagged_conservative_flux_authority_(const std::string& name) const {
    const auto fluxes = ring_flux_.find(name);
    if (fluxes != ring_flux_.end())
      for (const auto& slot : fluxes->second)
        for (const EdgeFlux& flux : slot)
          if (!flux.empty())
            return true;
    const auto contributions = ring_flux_contributions_.find(name);
    if (contributions != ring_flux_contributions_.end())
      for (const auto& slot : contributions->second)
        for (const auto& level : slot)
          if (!level.empty())
            return true;
    return false;
  }

  /// A compact lagged EdgeFlux stores only the interface that existed when the rate was evaluated.
  /// Once that interface moves, missing new faces cannot be invented. For a ring with a conservative
  /// flux authority, add a parent-constant correction to each child group so the lagged fine average
  /// exactly matches its parent, then invalidate the obsolete payload. The residual and zero-mismatch
  /// ledger agree on the new topology while every retained old-fine fluctuation, temporal slot, exact
  /// contribution weight and clock survives. Rings without conservative flux contributions (for
  /// example a phi/state carry) keep the engine's normal overlap-preserving remap and are untouched.
  void rebind_history_flux_topology_(const HistoryFluxTopology& before,
                                     const HistoryFluxTopology& after) const {
    if (before.boxes.size() != after.boxes.size() || before.owners.size() != after.owners.size())
      throw std::runtime_error(
          "AMR lagged-flux topology rebind cannot change the resolved hierarchy depth");
    for (int child = 1; child < nlev(); ++child) {
      const std::size_t k = static_cast<std::size_t>(child);
      const int parent = child - 1;
      const std::size_t p = static_cast<std::size_t>(parent);
      const bool parent_changed =
          before.boxes[p] != after.boxes[p] || before.owners[p] != after.owners[p];
      const bool child_changed =
          before.boxes[k] != after.boxes[k] || before.owners[k] != after.owners[k];
      if (!parent_changed && !child_changed)
        continue;
      for (const auto& [name, owner] : history_owners_) {
        (void)owner;
        if (!has_lagged_conservative_flux_authority_(name))
          continue;
        pops::detail::AmrHistoryOps::match_conservative_ring_average_to_parent(*eng_, name, child,
                                                                               parent);
        auto flux_ring = ring_flux_.find(name);
        if (flux_ring != ring_flux_.end())
          for (auto& slot : flux_ring->second) {
            if (parent < static_cast<int>(slot.size()))
              clear_parent_interface_(slot[static_cast<std::size_t>(parent)]);
            if (child < static_cast<int>(slot.size()))
              clear_child_interface_(slot[k]);
          }
        auto contributions = ring_flux_contributions_.find(name);
        if (contributions != ring_flux_contributions_.end())
          for (auto& slot : contributions->second) {
            if (parent < static_cast<int>(slot.size()))
              for (FluxContribution& contribution : slot[static_cast<std::size_t>(parent)])
                clear_parent_interface_(contribution.payload);
            if (child < static_cast<int>(slot.size()))
              for (FluxContribution& contribution : slot[k])
                clear_child_interface_(contribution.payload);
          }
      }
      ++history_flux_topology_rebind_count_;
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
        identity.state != history_state_ids_.at(name) ||
        identity.space != history_space_ids_.at(name) || identity.level != level_ ||
        !(identity.clock == clock))
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
    const amr::HistoryIdentity identity{"program.block." + std::to_string(owner),
                                        history_state_ids_.at(name), history_space_ids_.at(name),
                                        level_, current_window_->end};
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
                         return dst ==
                                &eng_->level_state(static_cast<std::size_t>(sys_block(ring.owner)),
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
      const Real stored_dt =
          static_cast<Real>(pops::detail::AmrHistoryOps::slot_dt(*eng_, ring.name, 0));
      pops::detail::AmrHistoryOps::store_history(*eng_, ring.name, ring.level,
                                                 eng_->level_state(sb, ring.level), stored_dt);
    }
  }

  AmrSystem* facade_;
  AmrRuntime* eng_;
  mutable int level_ = 0;
  mutable std::optional<SolveReport> default_solve_report_;
  mutable std::map<std::string, SolveReport> named_solve_reports_;
  mutable std::map<std::pair<int, int>, MultiFab> stage_state_scratch_;
  mutable std::vector<std::pair<MultiFab*, MultiFab*>> stage_restore_scratch_;
  std::shared_ptr<const HierarchyTensorSolverProviderRegistry> hierarchy_tensor_solver_registry_;
  mutable std::unique_ptr<PreparedHierarchyTensorSolver> hierarchy_tensor_solver_;
  mutable std::string hierarchy_tensor_selection_contract_;
  mutable int hierarchy_tensor_program_block_{-1};
  mutable int hierarchy_tensor_ncomp_{0};
  mutable std::uint64_t hierarchy_tensor_topology_epoch_{std::numeric_limits<std::uint64_t>::max()};
  mutable std::vector<std::string> hierarchy_tensor_assembly_field_slots_;
  mutable std::string hierarchy_tensor_solution_field_slot_;

  // --- ADC-639 conservative-reflux state -----------------------------------------------------------
  // Persistent per-(runtime block,parent level) coarse-role redistribution targets. Their own
  // parallel_copy schedules survive across Program stages and macro-steps; CoarseRoleScratch rejects
  // and rebuilds them on topology-epoch, exact layout/ownership, or component-width changes.
  mutable std::map<std::pair<std::size_t, int>, pops::detail::CoarseRoleScratch>
      coarse_role_scratch_;
  // The effective-flux LEDGER: for each tracked MultiFab (keyed by (level, address)) the interface-strip
  // effective flux (EdgeFlux: coarse-role per child patch + fine-role per this-level patch). The Program's
  // linear combination is SHADOWED on these strips (axpy / lincomb mirror), so a commit's strip holds
  // dt * Feff = dt * sum_i w_i F_i -- the native effective flux reproduced from the Program text. Cleared
  // per macro-step by reset_step(); populated ONLY when capturing() (nlev > 1). On a coarse-only / flat
  // Program it stays EMPTY (the capture branch is never reached) -- the bit-identical parity gate.
  mutable std::map<std::pair<int, const void*>, EdgeFlux> flux_ledger_;
  mutable std::map<std::pair<int, const void*>, std::vector<FluxContribution>> flux_contributions_;
  mutable std::map<std::pair<int, const void*>, std::set<int>> rate_provenance_;
  mutable std::vector<SyncEvent> sync_report_;
  mutable amr::TransactionalFluxLedger<EdgeFlux> conservative_ledger_;
  mutable std::vector<AmrProgramFluxAuditEntry> accepted_flux_report_;
  mutable std::vector<AmrProgramSyncEvent> accepted_sync_report_;
  mutable std::vector<amr::ClockStamp> level_clocks_;
  mutable std::uint64_t accepted_state_revision_ = 0;
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::uint64_t active_operator_snapshot_revision_ = 0;  // zero is never minted
  mutable std::uint64_t operator_topology_revision_counter_ = 0;
  mutable std::uint64_t observed_operator_topology_epoch_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable int observed_operator_level_ = -1;
  mutable amr::Rational stage_time_{0, 1};
  mutable std::optional<ActiveParentWindow> active_parent_;
  mutable std::optional<amr::ClockWindow> current_window_;
  mutable std::optional<amr::ClockStamp> current_sync_clock_;
  mutable double current_level_dt_ = 0.0;
  // PERSISTENT per-history-ring flux strips (NOT cleared per step): name -> [slot][level] -> EdgeFlux. A
  // multistep scheme (AB2 / BDF2) reads a lagged RHS / state from the ring; store_history saves that
  // buffer's ledger strip here (slot 0), rotate_histories rotates it in lockstep with the ring, and
  // history() restores it into the live ledger so the commit combine carries the lagged flux's weight --
  // the reflux stays conservative for a multistep Program across steps (acceptance e).
  mutable std::map<std::string, std::vector<std::vector<EdgeFlux>>> ring_flux_;
  mutable std::map<std::string, std::vector<std::vector<std::vector<FluxContribution>>>>
      ring_flux_contributions_;
  // Per-ring per-level cold-start flags for ring_flux_ (mirror of the engine hist_init_), so the first
  // store of a (name, level) broadcasts its strip into every slot. PERSISTENT (not cleared per step).
  mutable std::map<std::string, std::vector<char>> ring_flux_init_;
  // Derived binding between the compact lagged interface strips and the hierarchy that sampled them.
  // It is reconstructed from the restored engine topology (the accepted-state payload already carries
  // the strips and the engine checkpoint carries the exact boxes/owners), then advanced transactionally.
  mutable HistoryFluxTopology history_flux_topology_;
  mutable int history_flux_topology_rebind_count_ = 0;
  mutable std::map<std::string, int> history_owners_;
  mutable std::map<std::string, std::string> history_state_ids_;
  mutable std::map<std::string, std::string> history_space_ids_;
  mutable std::map<std::string, std::string> history_clock_ids_;
  mutable std::map<std::string, std::string> history_interpolation_ids_;
  mutable std::map<std::string, std::vector<std::vector<amr::ClockStamp>>> ring_clocks_;
  mutable std::map<std::string, std::vector<std::vector<std::optional<amr::HistoryIdentity>>>>
      ring_identities_;
  // Deferred-rotate flag (ADC-631 consistency, section 2c): the body's terminal rotate_histories() fires
  // inside the recursive level advance, before couple_levels reflux modifies the coarse live state. We defer the
  // rotate to couple_levels (after the reflux + slot-0 resync), so a multistep Program never reads a
  // pre-reflux ring slot against a post-reflux live state. On nlev==1 the deferral collapses to the
  // original store->rotate order -> bit-identical to the Uniform route.
  mutable bool rotate_pending_ = false;
  mutable ClockScheduleState clock_schedule_;
  mutable std::string primary_clock_;
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
