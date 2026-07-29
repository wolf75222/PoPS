#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/program/step_transaction.hpp>

namespace pops::runtime::program {

/// Backend-independent Program operations shared by every execution topology.
///
/// The provider owns topology and storage only.  It exposes the narrow
/// ``program_execution_*`` hooks below; generated Program operations themselves live here exactly
/// once.  This is deliberately CRTP instead of a virtual facade: a generated artifact still calls
/// the concrete context directly, and the compiler resolves each provider hook without a second
/// runtime dispatch table.
template <class Provider>
class ProgramExecutionServices {
 public:
  struct FieldStageOverride {
    int program_block = -1;
    const MultiFab* state = nullptr;
  };

  struct CouplingStateOverride {
    int program_block = -1;
    MultiFab* state = nullptr;
  };

  enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };

  /// One topology-independent subdivision of the active logical interval.
  struct LogicalEvaluationInterval {
    int iteration = 0;
    int count = 1;
    double parent_dt = 0.0;
    double child_dt = 0.0;
    amr::Rational child_begin{0, 1};
    amr::Rational child_end{1, 1};
    amr::Rational child_span{1, 1};
  };

  /// One exact logical child-clock interval.
  ///
  /// Program semantics own validation, subdivision, move-only lifetime and rollback exactly once.
  /// ``Rollback`` is an opaque provider token: the shared authority neither names nor inspects
  /// topology state.
  template <class Rollback>
  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const Provider& owner, double child_dt, Rollback rollback)
        : owner_(&owner), child_dt_(child_dt), rollback_(std::move(rollback)) {
      static_assert(std::is_nothrow_move_constructible_v<Rollback>,
                    "Program logical-evaluation rollback tokens must be nothrow movable");
    }
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          child_dt_(other.child_dt_),
          rollback_(std::move(other.rollback_)) {}
    LogicalEvaluationScope& operator=(LogicalEvaluationScope&&) = delete;
    ~LogicalEvaluationScope() noexcept { restore_(); }

    Real dt() const {
      if (owner_ == nullptr)
        throw std::logic_error("Program logical evaluation scope is no longer active");
      return static_cast<Real>(child_dt_);
    }

   private:
    void restore_() noexcept {
      if (owner_ == nullptr)
        return;
      owner_->program_execution_restore_logical_evaluation_(rollback_);
      owner_ = nullptr;
    }

    const Provider* owner_ = nullptr;
    double child_dt_ = 0.0;
    Rollback rollback_;
  };

  [[nodiscard]] auto logical_evaluation_scope(int iteration, int count) const {
    if (count <= 0 || iteration < 0 || iteration >= count)
      throw std::invalid_argument("Program logical evaluation requires a valid child iteration");
    const double parent_dt = provider_().program_execution_logical_parent_dt_();
    if (!std::isfinite(parent_dt) || parent_dt <= 0.0)
      throw std::logic_error("Program logical evaluation requires a prepared parent dt");
    const double child_dt = parent_dt / static_cast<double>(count);
    if (!std::isfinite(child_dt) || child_dt <= 0.0)
      throw std::overflow_error("Program logical evaluation child dt is not finite");
    const LogicalEvaluationInterval interval{
        iteration,
        count,
        parent_dt,
        child_dt,
        amr::Rational(iteration, count),
        amr::Rational(iteration + 1, count),
        amr::Rational(1, count),
    };

    // Capture first, arm rollback second, mutate last. If applying the provider-specific clock
    // projection throws after any partial mutation, the local scope restores the exact parent state.
    auto rollback = provider_().program_execution_capture_logical_evaluation_();
    using Rollback = decltype(rollback);
    LogicalEvaluationScope<Rollback> scope(provider_(), child_dt, std::move(rollback));
    provider_().program_execution_apply_logical_evaluation_(interval);
    return scope;
  }

  template <class Body>
  void evaluate_with_field_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                    const std::string& provider_slot, int block,
                                    MultiFab& evaluation_state, MultiFab& restore_state,
                                    Body&& body) const {
    const auto restore = [&]() {
      const SolveReport restored = provider_().program_execution_solve_fields_from_state_at_(
          point, provider_slot, block, restore_state);
      if (!restored.solved_value_available())
        throw_field_solve_failure_(restored, "restoring the frozen field state");
    };
    const SolveReport prepared = provider_().program_execution_solve_fields_from_state_at_(
        point, provider_slot, block, evaluation_state);
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

  MultiFab rhs_scratch_like(const MultiFab& prototype) const {
    MultiFab scratch(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                     prototype.n_grow());
    count_scratch(scratch);
    return scratch;
  }

  MultiFab scratch_state_like(const MultiFab& prototype) const {
    return rhs_scratch_like(prototype);
  }

  MultiFab& rhs_scratch(std::int64_t value_id, int subslot, const MultiFab& prototype) const {
    return provider_().program_execution_scratch_(ScratchKind::Rhs, value_id, subslot, prototype,
                                                  prototype.ncomp(), prototype.n_grow());
  }

  MultiFab& scratch_state(std::int64_t value_id, int subslot, const MultiFab& prototype) const {
    return provider_().program_execution_scratch_(ScratchKind::State, value_id, subslot, prototype,
                                                  prototype.ncomp(), prototype.n_grow());
  }

  MultiFab& scalar_scratch(std::int64_t value_id, int subslot, const MultiFab& prototype,
                           int n_comp = 1, int n_ghost = 1) const {
    if (n_comp < 1 || n_ghost < 0)
      throw std::invalid_argument("Program scalar scratch requires n_comp >= 1 and n_ghost >= 0");
    return provider_().program_execution_scratch_(ScratchKind::Scalar, value_id, subslot, prototype,
                                                  n_comp, n_ghost);
  }

  void commit_many(std::initializer_list<std::pair<MultiFab*, const MultiFab*>> commits) const {
    std::vector<MultiFab*> targets;
    targets.reserve(commits.size());
    for (const auto& [target, source] : commits) {
      if (target == nullptr || source == nullptr)
        throw std::invalid_argument("Program commit_many received a null state");
      if (std::find(targets.begin(), targets.end(), target) != targets.end())
        throw std::invalid_argument("Program commit_many received a duplicate target");
      if (target->box_array().boxes() != source->box_array().boxes() ||
          target->dmap().ranks() != source->dmap().ranks() || target->ncomp() != source->ncomp())
        throw std::invalid_argument("Program commit_many state layout mismatch");
      targets.push_back(target);
    }
    const bool has_aliased_source =
        std::any_of(commits.begin(), commits.end(), [&targets](const auto& commit) {
          return commit.first != commit.second &&
                 std::find(targets.begin(), targets.end(), commit.second) != targets.end();
        });
    provider_().program_execution_validate_commit_aliases_(has_aliased_source);

    if (!has_aliased_source) {
      for (const auto& [target, source] : commits)
        if (target != source)
          provider_().program_execution_commit_copy_(*target, *source);
      return;
    }

    std::vector<std::pair<MultiFab*, const MultiFab*>> prepared(commits);
    std::vector<MultiFab> aliased_sources;
    aliased_sources.reserve(prepared.size());
    for (auto& [target, source] : prepared) {
      if (target != source && std::find(targets.begin(), targets.end(), source) != targets.end()) {
        source->sync_host();
        aliased_sources.emplace_back(*source);
        aliased_sources.back().sync_device();
        source = &aliased_sources.back();
      }
    }
    for (const auto& [target, source] : prepared) {
      if (target != source) {
        provider_().program_execution_commit_copy_(*target, *source);
        // Aliased snapshots are function-local. Complete their consumers before destruction.
        device_fence();
      }
    }
  }

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (denominator <= 0 || numerator < 0 || numerator > denominator)
      throw std::runtime_error("Program stage time is outside [0,1]");
    stage_time_ = amr::Rational(numerator, denominator);
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
    return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
  }

  bool schedule_is_due(int node_id, int every_n, ScheduleDomainKind kind, const std::string& clock,
                       const std::string& stage_identity, int level) const {
    if (node_id < 0 || every_n <= 0)
      throw std::runtime_error("Program schedule requires a valid node and positive period");
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value % every_n == 0;
  }

  bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                         const std::string& stage_identity, int level) const {
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value == 0;
  }

  bool schedule_decision(int node_id, bool due, bool cache_backed) const {
    if (node_id < 0)
      throw std::runtime_error("Program schedule decision requires a valid node");
    return profiler().schedule_decision(due, cache_backed);
  }

  ClockScheduleState::SubcycleScope subcycle_scope(const std::string& parent,
                                                   const std::string& child, int count) const {
    return clock_schedule_.subcycle(parent, child, count);
  }

  void synchronize_sample_and_hold(const std::string& source, const std::string& target, int step,
                                   Real offset) const {
    clock_schedule_.synchronize_sample_and_hold(source, target, step, static_cast<double>(offset));
  }

  int sys_block(int program_block) const {
    const std::vector<int>& block_map = provider_().program_execution_block_map_();
    if (block_map.empty())
      throw std::runtime_error(
          "Program execution has no explicit program-to-runtime block map; positional block "
          "identity is not supported");
    if (program_block < 0 || program_block >= static_cast<int>(block_map.size()))
      throw std::runtime_error("Program block index " + std::to_string(program_block) +
                               " is outside the explicit runtime block map [0, " +
                               std::to_string(block_map.size()) + ")");
    const int mapped = block_map[static_cast<std::size_t>(program_block)];
    const int count = provider_().program_execution_block_count_();
    if (mapped < 0 || mapped >= count)
      throw std::runtime_error("Program block index " + std::to_string(program_block) +
                               " maps to invalid runtime block index " + std::to_string(mapped) +
                               " for a runtime with " + std::to_string(count) + " blocks");
    return mapped;
  }

  int n_blocks() const { return provider_().program_execution_block_count_(); }

  Real physical_time() const { return provider_().program_execution_physical_time_(); }

  void record_scalar(const std::string& name, Real value) const {
    provider_().program_execution_record_scalar_(name, value);
  }

  void note_step_projection(const std::string& name) const {
    provider_().program_execution_note_step_projection_(name);
  }

  RuntimeParams program_params(int block) const {
    return provider_().program_execution_params_(block);
  }

  void set_field_logical_timepoint(const std::string& field,
                                   const FieldLogicalTimePoint& point) const {
    provider_().program_execution_set_field_timepoint_(field, point);
  }

  void set_field_boundary_parameters(const std::string& field,
                                     const std::vector<double>& parameters) const {
    provider_().program_execution_set_field_parameters_(field, parameters);
  }

  void set_field_boundary_kernel(const std::string& field,
                                 const CompiledFieldBoundaryKernel& kernel) const {
    provider_().program_execution_set_field_kernel_(field, kernel);
  }

  Profiler& profiler() const { return provider_().program_execution_profiler_(); }

  ProfileScope profile_node(const std::string& name) const {
    return ProfileScope(profiler(), name);
  }

  void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
    const auto end = std::chrono::steady_clock::now();
    profiler().record(name, std::chrono::duration<double>(end - start).count());
  }

  void count_kernel(std::int64_t by = 1) const { profiler().count("kernels", by); }

  void count_scratch(const MultiFab& field) const {
    Profiler& service = profiler();
    if (!service.enabled())
      return;
    service.count("scratch_allocs");
    std::int64_t bytes = 0;
    for (int local = 0; local < field.local_size(); ++local)
      bytes += field.fab(local).size() * static_cast<std::int64_t>(sizeof(Real));
    service.count_max("scratch_peak_bytes", bytes);
  }

  int macro_step() const { return provider_().program_execution_macro_step_(); }

  [[noreturn]] void scheduler_error(const std::string& what) const {
    throw std::runtime_error("pops Program scheduler: " + what);
  }

 protected:
  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument(
          "Program rate evaluation requires a non-negative authored node identity");
  }

  static void require_group_identity_(int group_id) {
    if (group_id < 0)
      throw std::invalid_argument(
          "Program RHS group requires a non-negative authored group identity");
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

  mutable ClockScheduleState clock_schedule_;
  mutable std::string primary_clock_;
  mutable amr::Rational stage_time_{0, 1};

 private:
  const Provider& provider_() const { return static_cast<const Provider&>(*this); }

  std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                         const std::string& clock,
                                                         const std::string& stage_identity,
                                                         int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level,
                                      provider_().program_execution_active_level_(), macro_step());
  }
};

}  // namespace pops::runtime::program
