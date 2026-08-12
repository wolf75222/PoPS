/// @file
/// @brief Exact compile-time-ranked execution boundary for generated AMR Programs.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/parallel/region_transfer.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

template <int Dim>
struct ProgramSpatialSnapshot {
  std::string spatial_contract;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;

  bool operator==(const ProgramSpatialSnapshot&) const = default;
};

/// One Program specialization over one immutable native rank.
///
/// The context never decodes a dimension tag and never pads an absent axis.  Its active level is a
/// compile-time-ranked `MultiFab<Dim>` selected from the exact `AmrRuntime<Dim>` hierarchy.  The
/// retained generated block owns geometry, physical boundaries, same-level/coarse-fine ghost fill,
/// residual assembly and integrated face fluxes.  Unsupported provider families fail before a valid
/// cell is changed.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrProgramContext {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "AmrProgramContext only supports dimensions 1, 2, and 3");
  static_assert(std::is_same_v<MemorySpace, typename Kokkos::DefaultExecutionSpace::memory_space>,
                "AmrProgramContext memory space must match its compiled AmrSystem leaf");

  static constexpr int dimension = Dim;
  using facade_type = ::pops::AmrSystem<Dim>;
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using hierarchy_type = typename runtime_type::hierarchy_type;
  using field_type = typename runtime_type::field_type;
  using level_evaluation_type = typename facade_type::PreparedLevelEvaluation;
  using runtime_state_type = ProgramRuntimeState<Dim>;
  using scalar_boundary_session_type = PreparedScalarBoundarySession<Dim>;
  class PreparedBlockBoundarySession {
   public:
    PreparedBlockBoundarySession(const PreparedBlockBoundarySession&) = default;
    PreparedBlockBoundarySession& operator=(const PreparedBlockBoundarySession&) = default;

   private:
    friend class AmrProgramContext;
    PreparedBlockBoundarySession(const facade_type* facade, int runtime_block,
                                 runtime::multiblock::BoundaryEvaluationPoint point,
                                 const ExecutionLane& lane,
                                 std::shared_ptr<scalar_boundary_session_type> transport)
        : facade_(facade),
          runtime_block_(runtime_block),
          point_(std::move(point)),
          lane_(&lane),
          transport_(std::move(transport)) {
      if (facade_ == nullptr || runtime_block_ < 0 || !transport_)
        throw std::invalid_argument("prepared AMR block boundary session is incomplete");
    }
    const facade_type* facade_ = nullptr;
    int runtime_block_ = -1;
    runtime::multiblock::BoundaryEvaluationPoint point_{};
    const ExecutionLane* lane_ = nullptr;
    std::shared_ptr<scalar_boundary_session_type> transport_;
  };
  using block_boundary_session_type = PreparedBlockBoundarySession;
  using subcycle_plan_type = ::pops::numerics::time::amr::PreparedAmrSubcyclePlan<Dim, MemorySpace>;
  using reflux_payload_type = std::vector<Real>;
  using multiblock_subcycling_type =
      ::pops::numerics::time::amr::PreparedMultiBlockAmrSubcyclingEngine<Dim, reflux_payload_type,
                                                                         MemorySpace>;
  using multiblock_level_group_type = typename multiblock_subcycling_type::LevelAdvanceGroup;
  using multiblock_reflux_context_type = typename multiblock_subcycling_type::RefluxContext;
  using multiblock_flux_ledger_type = typename multiblock_subcycling_type::ledger_type;
  using interface_flux_ledger_type =
      ::pops::amr::TransactionalInterfaceFluxLedger<AmrProgramFacePayload>;
  using flux_expression_budget_type = typename facade_type::PreparedAmrProgramFluxExpressionBudget;
  using hierarchy_tensor_provider_type = HierarchyTensorSolverProvider<Dim, MemorySpace>;
  using hierarchy_tensor_registry_type = HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>;
  using hierarchy_tensor_solver_type = PreparedHierarchyTensorSolver<Dim, MemorySpace>;
  using hierarchy_tensor_request_type = HierarchyTensorSolverBuildRequest<Dim>;

  struct ProgramResourceTopology {
    int levels = 0;
    std::uint64_t epoch = 0;
    std::uint64_t generation = 0;
  };

  struct FieldStageOverride {
    int program_block = -1;
    const field_type* state = nullptr;
  };

  struct RhsGroupRequest {
    RhsGroupRequest(int block_value, field_type* state_value, field_type* rhs_value,
                    int rate_id_value, int flux_only_value)
        : block(block_value),
          state(state_value),
          rhs(rhs_value),
          rate_id(rate_id_value),
          flux_only(flux_only_value) {}

    int block = -1;
    field_type* state = nullptr;
    field_type* rhs = nullptr;
    int rate_id = -1;
    int flux_only = 0;
  };

  struct CouplingStateOverride {
    int program_block = -1;
    field_type* state = nullptr;
  };

  struct HierarchyTensorSelection {
    int program_block = -1;
    int components = 0;
    std::string provider_identity;
    std::string plan_identity;
    std::string operator_contract_identity;
    std::vector<std::string> assembly_field_slots;
    std::string solution_field_slot;
    PreparedProviderOptions options;
    std::string exact_contract;
  };

  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const AmrProgramContext& owner, int iteration, int count)
        : owner_(&owner),
          prior_dt_(owner.current_dt_),
          prior_interval_start_time_(owner.current_interval_start_time_),
          prior_interval_begin_phase_(owner.current_interval_begin_phase_),
          prior_interval_end_phase_(owner.current_interval_end_phase_),
          prior_substep_(owner.logical_substep_) {
      if (iteration < 0 || count < 1 || iteration >= count || !std::isfinite(prior_dt_) ||
          !(prior_dt_ > 0.0))
        throw std::invalid_argument("AMR logical evaluation scope is invalid");
      owner_->current_dt_ = prior_dt_ / static_cast<double>(count);
      owner_->current_interval_start_time_ =
          prior_interval_start_time_ + static_cast<double>(iteration) * owner_->current_dt_;
      const ::pops::amr::Rational span = prior_interval_end_phase_ - prior_interval_begin_phase_;
      owner_->current_interval_begin_phase_ =
          prior_interval_begin_phase_ +
          span * ::pops::amr::Rational(iteration, static_cast<std::int64_t>(count));
      owner_->current_interval_end_phase_ =
          prior_interval_begin_phase_ +
          span * ::pops::amr::Rational(iteration + 1, static_cast<std::int64_t>(count));
      owner_->logical_substep_ = iteration;
    }
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          prior_dt_(other.prior_dt_),
          prior_interval_start_time_(other.prior_interval_start_time_),
          prior_interval_begin_phase_(other.prior_interval_begin_phase_),
          prior_interval_end_phase_(other.prior_interval_end_phase_),
          prior_substep_(other.prior_substep_) {}
    ~LogicalEvaluationScope() {
      if (owner_ != nullptr) {
        owner_->current_dt_ = prior_dt_;
        owner_->current_interval_start_time_ = prior_interval_start_time_;
        owner_->current_interval_begin_phase_ = prior_interval_begin_phase_;
        owner_->current_interval_end_phase_ = prior_interval_end_phase_;
        owner_->logical_substep_ = prior_substep_;
      }
    }
    Real dt() const { return static_cast<Real>(owner_->current_dt_); }

   private:
    const AmrProgramContext* owner_ = nullptr;
    double prior_dt_ = 0.0;
    double prior_interval_start_time_ = 0.0;
    ::pops::amr::Rational prior_interval_begin_phase_{0, 1};
    ::pops::amr::Rational prior_interval_end_phase_{1, 1};
    int prior_substep_ = 0;
  };

  explicit AmrProgramContext(facade_type* facade)
      : facade_(require_facade_(facade)), runtime_(require_runtime_(*facade_)) {
    facade_->refresh_prepared_amr_levels();
    hierarchy_tensor_solver_registry_ = facade_->hierarchy_tensor_solver_provider_registry();
    synchronize_resource_generation_();
  }

  AmrProgramContext(runtime_type* runtime, facade_type* facade)
      : facade_(require_facade_(facade)), runtime_(require_runtime_(runtime)) {
    if (facade_->engine() != runtime_)
      throw std::invalid_argument("AMR Program facade and runtime do not share one hierarchy");
    facade_->refresh_prepared_amr_levels();
    hierarchy_tensor_solver_registry_ = facade_->hierarchy_tensor_solver_provider_registry();
    synchronize_resource_generation_();
  }

  /// Spatial-only constructor used by preparation tests.  Execution methods require a facade.
  explicit AmrProgramContext(runtime_type& runtime) : runtime_(&runtime) {
    synchronize_resource_generation_();
  }

  runtime_type& runtime() const noexcept { return *runtime_; }
  hierarchy_type& hierarchy() const noexcept { return runtime_->hierarchy(); }
  /// Borrow the runtime-prepared AMR hierarchy lane; generated execution never materializes a
  /// second communicator or falls back to the process world.
  [[nodiscard]] const ExecutionLane& prepared_execution_lane() const {
    require_facade_execution_();
    return facade_->prepared_amr_multiblock_hierarchy_().lane();
  }
  const ::pops::amr::hierarchy::LevelLayout<Dim>& layout(std::size_t selected) const {
    return runtime_->hierarchy().layout(selected);
  }
  field_type& state(std::size_t selected) const {
    if (!active_attempt_states_.empty()) {
      if (selected != static_cast<std::size_t>(active_level_) ||
          active_attempt_states_.front() == nullptr)
        throw std::out_of_range("AMR Program attempt state lies outside the prepared hierarchy");
      return *active_attempt_states_.front();
    }
    return runtime_->hierarchy().state(selected);
  }

  ::pops::runtime::amr::PreparedTaggerCandidates<Dim> execute_prepared_tagging(
      int parent_level) const {
    if (facade_ == nullptr)
      throw std::logic_error(
          "AMR Program tagging execution requires the exact-ranked facade authority");
    return facade_->execute_prepared_tagging(parent_level);
  }

  bool regrid_from_prepared_tagging(int parent_level) const {
    if (facade_ == nullptr)
      throw std::logic_error(
          "AMR Program regrid publication requires the exact-ranked facade authority");
    require_history_free_for_topology_change_("prepared tagging regrid");
    return facade_->regrid_from_prepared_tagging(parent_level);
  }

  void begin_restart_regrid_history_sequence() const {
    require_facade_execution_();
    facade_->begin_restart_regrid_history_sequence();
  }

  void end_restart_regrid_history_sequence() const noexcept {
    facade_->end_restart_regrid_history_sequence();
  }

  ProgramSpatialSnapshot<Dim> spatial_snapshot() const {
    return {std::string(runtime_->spatial_contract()), runtime_->topology_epoch(),
            runtime_->materialization_generation()};
  }

  void require_live(const ProgramSpatialSnapshot<Dim>& snapshot) const {
    if (snapshot.spatial_contract != runtime_->spatial_contract() ||
        snapshot.topology_epoch != runtime_->topology_epoch() ||
        snapshot.materialization_generation != runtime_->materialization_generation())
      throw std::invalid_argument("AMR Program spatial snapshot is stale");
  }

  subcycle_plan_type prepare_subcycling(
      std::span<const int> temporal_substeps,
      ::pops::numerics::time::amr::AmrSubcyclePreparationBudget budget) const {
    return subcycle_plan_type::prepare(*runtime_, temporal_substeps, budget);
  }

  ::pops::amr::regridding::PreparedRegrid<Dim> prepare_regrid(
      std::size_t parent_level, ::pops::amr::RefinementRatio<Dim> ratio,
      ::pops::amr::tagging::ClusterResult<Dim> clustered,
      ::pops::amr::regridding::RegridPreparationBudget preparation_budget) const {
    const ExecutionLane& lane = prepared_execution_lane();
    return runtime_->prepare_regrid(parent_level, ratio, std::move(clustered), preparation_budget,
                                    lane);
  }

  void publish_regrid(::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                      std::optional<field_type> child_state) const {
    require_facade_execution_();
    require_history_free_for_topology_change_("regrid");
    const int parent_level = prepared.source_level().level;
    if (parent_level < 0)
      throw std::invalid_argument("AMR Program regrid has no source level");
    runtime_->publish_regrid(static_cast<std::size_t>(parent_level), std::move(prepared),
                             std::move(child_state));
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t selected, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget,
      const RebalancePolicy& policy) const {
    const ExecutionLane& lane = prepared_execution_lane();
    return runtime_->prepare_rebalance(selected, estimates, preparation_budget, policy, lane);
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t selected, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget) const {
    const ExecutionLane& lane = prepared_execution_lane();
    return runtime_->prepare_rebalance(selected, estimates, preparation_budget, lane);
  }

  void apply_rebalance(std::size_t selected, PreparedRebalanceDecision<Dim> decision,
                       field_type remapped_state) const {
    require_facade_execution_();
    require_history_free_for_topology_change_("rebalance");
    runtime_->apply_rebalance(selected, std::move(decision), std::move(remapped_state));
  }

  template <class Payload, class Axpy>
  ::pops::amr::reflux::MetricFaceReflux<Payload> reconcile_reflux(
      const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
      const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key, std::string_view state_identity,
      const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
      const ::pops::amr::reflux::MetricRefluxBudget& budget, Axpy&& axpy) const {
    return runtime_->reconcile_reflux(ledger, key, state_identity, mapping, budget,
                                      std::forward<Axpy>(axpy));
  }

  void install(std::function<void(double)> step) const {
    require_facade_execution_();
    if (!step)
      throw std::invalid_argument("AMR Program install requires a non-empty step");
    facade_->install_program_step(std::move(step));
  }

  void install(std::function<void(double)> step, std::shared_ptr<AmrProgramContext> keep_alive,
               std::function<void()> hierarchy_refresh = {}) const {
    require_facade_execution_();
    if (!step || !keep_alive || keep_alive.get() != this)
      throw std::invalid_argument("AMR Program install requires its exact owning context");
    facade_->install_program_step([step = std::move(step), keep_alive](double dt) { step(dt); });
    if (hierarchy_refresh)
      facade_->install_program_hierarchy_refresh(
          [owner = keep_alive, refresh = std::move(hierarchy_refresh)]() mutable {
            refresh();
            owner->refresh_accepted_hierarchy_state_();
          });
    else
      facade_->install_program_hierarchy_refresh(
          [owner = keep_alive]() { owner->refresh_accepted_hierarchy_state_(); });
    facade_->install_program_restart_hooks(
        [owner = keep_alive]() { owner->preflight_restart_regrid_(); },
        [owner = keep_alive]() { owner->restart_regrid_(); },
        [owner = keep_alive]() { owner->resync_after_restart_(); },
        [owner = keep_alive]() { return owner->accepted_context_snapshot(); });
  }

  std::unique_ptr<AcceptedProgramContextSnapshot> accepted_context_snapshot() const {
    return capture_accepted_context_snapshot_();
  }

  void begin_step(double dt) const {
    require_facade_execution_();
    if (!std::isfinite(dt) || !(dt > 0.0))
      throw std::invalid_argument("AMR Program step requires a finite positive dt");
    current_dt_ = dt;
    current_interval_start_time_ = facade_->time();
    current_interval_begin_phase_ = {0, 1};
    current_interval_end_phase_ = {1, 1};
    stage_time_ = ::pops::amr::Rational(0, 1);
    logical_substep_ = 0;
  }

  void configure_primary_clock(const std::string& clock) const {
    clock_schedule_.configure_primary_clock(clock);
    primary_clock_ = clock;
  }

  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    clock_schedule_.declare_relation(parent, child, count);
  }

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (denominator <= 0 || numerator < 0 || numerator > denominator)
      throw std::invalid_argument("AMR Program stage time is outside [0, 1]");
    stage_time_ = ::pops::amr::Rational(numerator, denominator);
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage) const {
    require_rate_identity_(stage);
    require_facade_execution_();
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || !(current_dt_ > 0.0))
      throw std::logic_error("AMR Program evaluation point lacks an active clock interval");
    return {.clock = primary_clock_,
            .tick = static_cast<std::int64_t>(facade_->macro_step()),
            .level = active_level_,
            .substep = logical_substep_,
            .stage = stage,
            .stage_fraction = stage_time_,
            .dt = current_dt_,
            .physical_time = current_interval_start_time_ + stage_time_.value() * current_dt_};
  }

  template <class Body>
  void advance_hierarchy(double dt, Body&& body) const {
    advance_prepared_hierarchy_(dt, std::forward<Body>(body), "advance_hierarchy");
  }

  template <class Body>
  void advance_synchronized_hierarchy(double dt, Body&& body) const {
    advance_prepared_hierarchy_(dt, std::forward<Body>(body), "advance_synchronized_hierarchy");
  }

  [[noreturn]] void prepare_same_level_cell_temporal_execution(std::string, std::int64_t,
                                                               int = 0) const {
    unavailable_("cell-local AMR temporal provider");
  }
  [[noreturn]] void advance_same_level_cell_temporal(double) const {
    unavailable_("cell-local AMR temporal provider");
  }

  bool uses_prepared_krylov_fallback() const {
    return configured_hierarchy_tensor_solver_().execution_path() ==
           HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
  }
  int nlev() const { return static_cast<int>(runtime_->hierarchy().num_levels()); }
  int level() const noexcept { return active_level_; }

  ProgramResourceTopology program_resource_topology() const {
    refresh_resources_();
    return {nlev(), runtime_->topology_epoch(), runtime_->materialization_generation()};
  }

  template <class Function>
  void for_each_program_resource_level(Function&& function) const {
    refresh_resources_();
    const int prior = active_level_;
    try {
      for (int selected = 0; selected < nlev(); ++selected) {
        active_level_ = selected;
        function(selected);
      }
      active_level_ = prior;
    } catch (...) {
      active_level_ = prior;
      throw;
    }
  }

  template <class Function>
  decltype(auto) with_program_resource_level(int selected, Function&& function) const {
    if (selected < 0 || selected >= nlev())
      throw std::out_of_range("AMR Program resource level lies outside the live hierarchy");
    const int prior = active_level_;
    active_level_ = selected;
    try {
      if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
        std::forward<Function>(function)();
        active_level_ = prior;
      } else {
        decltype(auto) result = std::forward<Function>(function)();
        active_level_ = prior;
        return result;
      }
    } catch (...) {
      active_level_ = prior;
      throw;
    }
  }

  int n_blocks() const {
    require_facade_execution_();
    return facade_->n_blocks();
  }

  int sys_block(int program_block) const {
    require_facade_execution_();
    const auto& map = facade_->program_block_map();
    if (program_block < 0 || static_cast<std::size_t>(program_block) >= map.size())
      throw std::out_of_range("AMR Program block has no authenticated runtime mapping");
    const int selected = map[static_cast<std::size_t>(program_block)];
    if (selected < 0 || selected >= facade_->n_blocks())
      throw std::runtime_error("AMR Program block mapping targets no runtime block");
    return selected;
  }

  field_type& state(int program_block) const {
    refresh_resources_();
    const int runtime_block = sys_block(program_block);
    if (!active_attempt_states_.empty())
      return *active_attempt_states_.at(static_cast<std::size_t>(runtime_block));
    return facade_->prepared_amr_block_state(runtime_block, active_level_);
  }

  /// Bind one generated consumer's compact provider view for the active AMR hierarchy level.
  /// The program block is authenticated before storage lookup; the qid resolves through that
  /// level's sealed plan, so neither generated code nor this context can fall back to ``ctx.aux``.
  /// This is a rank-local Fab hot path: collective resource refresh belongs to the enclosing
  /// resource traversal, never to one invocation of the local provider callback.
  template <int Count>
  [[nodiscard]] ProviderStorageView<Dim, Count> provider_values_view(std::string_view consumer_qid,
                                                                     int program_block,
                                                                     std::size_t local_fab) const {
    static_assert(Count >= 0, "a provider consumer count cannot be negative");
    if constexpr (Count == 0) {
      (void)consumer_qid;
      (void)program_block;
      (void)local_fab;
      return {};
    } else {
      if (sys_block(program_block) != 0)
        deferred_op("multi_block_state", "exact-ranked multi-block AMR state provider");
      if (resource_epoch_ != runtime_->topology_epoch() ||
          resource_generation_ != runtime_->materialization_generation())
        throw std::logic_error(
            "AMR Program provider binding requires a collectively refreshed resource traversal");
      if (active_level_ < 0 || active_level_ >= nlev())
        throw std::out_of_range("AMR Program provider level lies outside the live hierarchy");
      const field_type& state_field =
          runtime_->hierarchy().state(static_cast<std::size_t>(active_level_));
      const auto* const groups = facade_->prepared_amr_provider_storage_groups(active_level_);
      const auto& plan =
          facade_->prepared_amr_auxiliary_consumer_plan(std::string(consumer_qid), active_level_);
      runtime::system::require_pointwise_provider_groups<Dim, Count>(
          state_field, groups, &plan, "AmrProgramContext provider values");
      return runtime::system::bind_provider_storage_view<Dim, Count>(&plan, groups, local_fab);
    }
  }

  field_type rhs_scratch_like(const field_type& prototype) const {
    return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
  }
  field_type scratch_state_like(const field_type& prototype) const {
    return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
  }

  field_type& rhs_scratch(std::int64_t value_id, int subslot, const field_type& prototype) const {
    return persistent_scratch_(ScratchKind::Rhs, value_id, subslot, prototype, prototype.ncomp(),
                               prototype.ghosts());
  }
  field_type& scratch_state(std::int64_t value_id, int subslot, const field_type& prototype) const {
    return persistent_scratch_(ScratchKind::State, value_id, subslot, prototype, prototype.ncomp(),
                               prototype.ghosts());
  }
  field_type& scalar_scratch(std::int64_t value_id, int subslot, const field_type& prototype,
                             int ncomp = 1, int ghost_depth = 1) const {
    return persistent_scratch_(ScratchKind::Scalar, value_id, subslot, prototype, ncomp,
                               uniform_ghosts_(ghost_depth));
  }

  field_type alloc_scalar_field(int ncomp = 1, int ghost_depth = 1) const {
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument("AMR Program scalar allocation has an invalid shape");
    const field_type& prototype =
        runtime_->hierarchy().state(static_cast<std::size_t>(active_level_));
    return make_scratch_(prototype, ncomp, uniform_ghosts_(ghost_depth));
  }

  void rhs_into(int program_block, field_type& stage_state, field_type& rhs, int rate_id) const {
    const int runtime_block = sys_block(program_block);
    require_rate_identity_(rate_id);
    require_same_field_contract_(stage_state, rhs, "AMR Program residual");
    const auto point = boundary_evaluation_point(rate_id);
    const auto& evaluation =
        active_attempt_states_.empty()
            ? facade_->evaluate_prepared_amr_block_level_at(runtime_block, point, stage_state)
            : facade_->evaluate_prepared_amr_block_level_at(
                  runtime_block, point, stage_state, active_level_ - 1,
                  staged_parent_for_block_(runtime_block));
    copy_valid_(evaluation.residual, rhs);
    if (!active_attempt_states_.empty())
      attach_active_flux_basis_(runtime_block, evaluation, rhs);
    count_kernel_();
  }

  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    require_rate_identity_(group_id);
    std::vector<field_type> candidates;
    candidates.reserve(requests.size());
    for (const RhsGroupRequest& request : requests) {
      if (request.state == nullptr || request.rhs == nullptr || request.rate_id < 0 ||
          request.rate_id == group_id || request.flux_only != 0)
        throw std::invalid_argument("AMR Program RHS group contains an unsupported request");
      candidates.push_back(rhs_scratch_like(*request.rhs));
    }
    std::size_t index = 0;
    for (const RhsGroupRequest& request : requests)
      rhs_into(request.block, *request.state, candidates[index++], request.rate_id);
    index = 0;
    for (const RhsGroupRequest& request : requests) {
      copy_valid_(candidates[index], *request.rhs);
      copy_active_flux_expression_(candidates[index], *request.rhs);
      clear_active_flux_expression_(candidates[index]);
      ++index;
    }
  }

  void neg_div_flux_default_into(int program_block, field_type& stage_state, field_type& rhs,
                                 int rate_id) const {
    const int runtime_block = sys_block(program_block);
    require_rate_identity_(rate_id);
    require_same_field_contract_(stage_state, rhs, "AMR Program default flux residual");
    const auto point = boundary_evaluation_point(rate_id);
    const auto& evaluation =
        active_attempt_states_.empty()
            ? facade_->evaluate_prepared_amr_block_level_flux_at(runtime_block, point, stage_state)
            : facade_->evaluate_prepared_amr_block_level_flux_at(
                  runtime_block, point, stage_state, active_level_ - 1,
                  staged_parent_for_block_(runtime_block));
    copy_valid_(evaluation.residual, rhs);
    if (!active_attempt_states_.empty())
      attach_active_flux_basis_(runtime_block, evaluation, rhs);
    count_kernel_();
  }
  void source_default_into(int program_block, field_type& stage_state, field_type& rhs) const {
    const int runtime_block = sys_block(program_block);
    require_same_field_contract_(stage_state, rhs, "AMR Program default source residual");
    const auto point = boundary_evaluation_point(0);
    if (active_attempt_states_.empty())
      facade_->prepared_amr_block_level_source_into_at(runtime_block, point, stage_state, rhs);
    else
      facade_->prepared_amr_block_level_source_into_at(runtime_block, point, stage_state, rhs,
                                                       active_level_ - 1,
                                                       staged_parent_for_block_(runtime_block));
    clear_active_flux_expression_(rhs);
    count_kernel_();
  }

  [[nodiscard]] SolveOutcome solve_source_default(int program_block, field_type& stage_state,
                                                  Real dt, const NewtonOptions& options) const {
    const int runtime_block = sys_block(program_block);
    const auto point = boundary_evaluation_point(0);
    SolveOutcome outcome = active_attempt_states_.empty()
                               ? facade_->solve_prepared_amr_block_level_source_at(
                                     runtime_block, point, stage_state, dt, options)
                               : facade_->solve_prepared_amr_block_level_source_at(
                                     runtime_block, point, stage_state, dt, options,
                                     active_level_ - 1, staged_parent_for_block_(runtime_block));
    count_kernel_();
    return outcome;
  }

  void require_cartesian_generated_operator(int program_block, const std::string& operation) const {
    (void)sys_block(program_block);
    if (operation.empty())
      throw std::invalid_argument("AMR generated operator requires an operation identity");
  }

  void prepare_generated_state(int program_block, field_type& stage_state, int rate_id) const {
    const int runtime_block = sys_block(program_block);
    require_rate_identity_(rate_id);
    const auto point = boundary_evaluation_point(rate_id);
    if (active_attempt_states_.empty())
      facade_->prepare_generated_amr_block_level_state(runtime_block, point, stage_state);
    else
      facade_->prepare_generated_amr_block_level_state(runtime_block, point, stage_state,
                                                       active_level_ - 1,
                                                       staged_parent_for_block_(runtime_block));
  }

  void neg_div_named_flux_into(field_type& rhs, const std::array<field_type*, Dim>& fluxes) const {
    if (!active_attempt_states_.empty())
      throw std::runtime_error(
          "active AMR named-flux divergence has no authenticated face-fragment provider");
    const Geometry<Dim> geom = geometry();
    for (int axis = 0; axis < Dim; ++axis) {
      const field_type* flux = fluxes[static_cast<std::size_t>(axis)];
      if (flux == nullptr || flux->layout() != rhs.layout() ||
          flux->distribution() != rhs.distribution() || flux->local_rank() != rhs.local_rank() ||
          flux->local_size() != rhs.local_size() || flux->ncomp() != rhs.ncomp() ||
          flux->ghosts()[axis] < 1)
        throw std::invalid_argument("AMR named flux differs from its exact residual layout");
    }
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      std::array<FieldView<const Real, Dim>, Dim> views{};
      for (int axis = 0; axis < Dim; ++axis)
        views[static_cast<std::size_t>(axis)] =
            std::as_const(*fluxes[static_cast<std::size_t>(axis)]).fab(local).view();
      const FieldView<Real, Dim> output = rhs.fab(local).view();
      const int components = rhs.ncomp();
      for_each_cell(rhs.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component) {
          Real divergence = Real(0);
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> lower = cell;
            Index<Dim> upper = cell;
            --lower[axis];
            ++upper[axis];
            divergence += (views[static_cast<std::size_t>(axis)](upper, component) -
                           views[static_cast<std::size_t>(axis)](lower, component)) /
                          (Real(2) * geom.spacing(axis));
          }
          output(cell, component) = -divergence;
        }
      });
    }
    count_kernel_();
  }

  [[noreturn]] void apply_projection(int, field_type&) const {
    deferred_op("projection", "generated AMR projection provider");
  }

  Real max_wave_speed(int program_block, const field_type& stage_state) const {
    return facade_->prepared_amr_block_level_maximum_speed(sys_block(program_block), active_level_,
                                                           stage_state);
  }

  Real hmin() const {
    const Geometry<Dim> geom = geometry();
    Real result = geom.spacing(0);
    for (int axis = 1; axis < Dim; ++axis)
      result = std::min(result, geom.spacing(axis));
    return result;
  }

  RuntimeParams program_params(int program_block) const {
    (void)sys_block(program_block);
    return facade_->program_params(program_block);
  }

  void axpy(field_type& destination, Real factor, const field_type& source) const {
    require_same_field_contract_(destination, source, "AMR Program axpy");
    auto expression_update = prepare_active_axpy_flux_expression_(
        destination, source, exact_runtime_axpy_coefficient_(factor, source));
    pops::saxpy(destination, factor, source);
    publish_active_flux_expression_update_(std::move(expression_update));
    count_kernel_();
  }
  void axpy(field_type& destination, Real factor, const field_type& source, Real reference_dt,
            std::initializer_list<ExactCoefficientTerm> terms) const {
    require_same_field_contract_(destination, source, "AMR Program exact axpy");
    const auto coefficient = exact_coefficient_(factor, reference_dt, terms);
    auto expression_update = prepare_active_axpy_flux_expression_(destination, source, coefficient);
    pops::saxpy(destination, factor, source);
    publish_active_flux_expression_update_(std::move(expression_update));
    count_kernel_();
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right) const {
    require_same_field_contract_(destination, left, "AMR Program linear combination");
    require_same_field_contract_(destination, right, "AMR Program linear combination");
    auto expression_update = prepare_active_lincomb_flux_expression_(
        destination, left, exact_runtime_coefficient_(left_factor), right,
        exact_runtime_coefficient_(right_factor));
    pops::lincomb(destination, left_factor, left, right_factor, right);
    publish_active_flux_expression_update_(std::move(expression_update));
    count_kernel_();
  }
  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right, Real reference_dt,
               std::initializer_list<ExactCoefficientTerm> left_terms,
               std::initializer_list<ExactCoefficientTerm> right_terms) const {
    require_same_field_contract_(destination, left, "AMR Program exact linear combination");
    require_same_field_contract_(destination, right, "AMR Program exact linear combination");
    const auto left_coefficient = exact_coefficient_(left_factor, reference_dt, left_terms);
    const auto right_coefficient = exact_coefficient_(right_factor, reference_dt, right_terms);
    auto expression_update = prepare_active_lincomb_flux_expression_(
        destination, left, left_coefficient, right, right_coefficient);
    pops::lincomb(destination, left_factor, left, right_factor, right);
    publish_active_flux_expression_update_(std::move(expression_update));
    count_kernel_();
  }

  void commit_many(std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
    const auto& commit_lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
    std::vector<field_type*> targets;
    std::vector<field_type> snapshots;
    std::vector<FluxExpression> snapshot_flux_expressions;
    targets.reserve(commits.size());
    snapshots.reserve(commits.size());
    snapshot_flux_expressions.reserve(commits.size());
    std::exception_ptr snapshot_error;
    try {
      for (const auto& [target, source] : commits) {
        if (target == nullptr || source == nullptr ||
            std::find(targets.begin(), targets.end(), target) != targets.end())
          throw std::invalid_argument("AMR Program commit has null or duplicate storage");
        require_same_field_contract_(*target, *source, "AMR Program commit");
        targets.push_back(target);
        snapshots.emplace_back(*source);
        snapshot_flux_expressions.push_back(active_flux_expression_(*source));
      }
    } catch (...) {
      snapshot_error = std::current_exception();
    }
    if (all_reduce_max(snapshot_error ? 1L : 0L, commit_lane) != 0) {
      if (commit_lane.size() == 1 && snapshot_error)
        std::rethrow_exception(snapshot_error);
      throw std::runtime_error("AMR Program commit snapshot failed collectively");
    }

    const long commit_count = static_cast<long>(targets.size());
    if (all_reduce_min(commit_count, commit_lane) != all_reduce_max(commit_count, commit_lane))
      throw std::runtime_error("AMR Program commit count differs between MPI ranks");

    std::vector<std::optional<int>> runtime_blocks;
    runtime_blocks.reserve(targets.size());
    std::exception_ptr classification_error;
    try {
      for (field_type* target : targets)
        runtime_blocks.push_back(authenticated_runtime_block_for_state_target_(*target));
    } catch (...) {
      classification_error = std::current_exception();
    }
    if (all_reduce_max(classification_error ? 1L : 0L, commit_lane) != 0) {
      if (commit_lane.size() == 1 && classification_error)
        std::rethrow_exception(classification_error);
      throw std::runtime_error("AMR Program commit target classification failed collectively");
    }

    if (!active_attempt_states_.empty()) {
      std::map<const field_type*, FluxExpression> expression_candidate;
      std::exception_ptr active_commit_error;
      try {
        expression_candidate = active_flux_expressions_;
        for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
          const bool detached_group_candidate =
              std::find(active_attempt_states_.begin(), active_attempt_states_.end(),
                        targets[candidate]) != active_attempt_states_.end();
          const bool prepared_scratch =
              std::any_of(scratches_.begin(), scratches_.end(),
                          [&](const auto& entry) { return &entry.second == targets[candidate]; });
          if (!detached_group_candidate && !prepared_scratch)
            throw std::invalid_argument(
                "active AMR Program commit target is neither a detached group candidate nor a "
                "prepared scratch");
          for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block)
            if (targets[candidate] ==
                &facade_->prepared_amr_block_state(runtime_block, active_level_))
              throw std::invalid_argument(
                  "active AMR Program commit cannot target an accepted block carrier");
          expression_candidate[targets[candidate]] = snapshot_flux_expressions[candidate];
        }
      } catch (...) {
        active_commit_error = std::current_exception();
      }
      if (all_reduce_max(active_commit_error ? 1L : 0L, commit_lane) != 0) {
        if (commit_lane.size() == 1 && active_commit_error)
          std::rethrow_exception(active_commit_error);
        throw std::runtime_error("active AMR Program commit failed collectively");
      }
      for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
        *targets[candidate] = std::move(snapshots[candidate]);
      active_flux_expressions_.swap(expression_candidate);
      return;
    }

    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
      const std::optional<int> runtime_block = runtime_blocks[candidate];
      const long state_target = runtime_block ? 1L : 0L;
      if (all_reduce_min(state_target, commit_lane) != all_reduce_max(state_target, commit_lane))
        throw std::runtime_error(
            "AMR Program commit target state classification differs between MPI ranks");
      if (runtime_block)
        facade_->validate_prepared_amr_state_publication_candidate(*runtime_block, active_level_,
                                                                   snapshots[candidate]);
    }

    std::vector<std::size_t> accepted_snapshot_by_runtime(static_cast<std::size_t>(n_blocks()),
                                                          snapshots.size());
    std::size_t accepted_targets = 0;
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
      if (!runtime_blocks[candidate])
        continue;
      const int runtime_block = *runtime_blocks[candidate];
      if (targets[candidate] != &facade_->prepared_amr_block_state(runtime_block, active_level_))
        continue;
      ++accepted_targets;
      accepted_snapshot_by_runtime[static_cast<std::size_t>(runtime_block)] = candidate;
    }
    if (accepted_targets != 0) {
      std::vector<field_type> publication_candidates;
      std::exception_ptr publication_error;
      try {
        publication_candidates.reserve(static_cast<std::size_t>(n_blocks()));
        for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block) {
          const std::size_t snapshot =
              accepted_snapshot_by_runtime[static_cast<std::size_t>(runtime_block)];
          if (snapshot == snapshots.size())
            publication_candidates.emplace_back(
                facade_->prepared_amr_block_state(runtime_block, active_level_));
          else
            publication_candidates.emplace_back(snapshots[snapshot]);
        }
      } catch (...) {
        publication_error = std::current_exception();
      }
      if (all_reduce_max(publication_error ? 1L : 0L, commit_lane) != 0) {
        if (commit_lane.size() == 1 && publication_error)
          std::rethrow_exception(publication_error);
        throw std::runtime_error("AMR Program accepted-state publication pack failed collectively");
      }
      std::vector<field_type*> program_candidates(static_cast<std::size_t>(n_blocks()), nullptr);
      for (int program_block = 0; program_block < n_blocks(); ++program_block) {
        const int runtime_block = sys_block(program_block);
        program_candidates[static_cast<std::size_t>(program_block)] =
            &publication_candidates[static_cast<std::size_t>(runtime_block)];
      }
      facade_->publish_prepared_amr_program_candidates(
          active_level_, std::span<field_type* const>(program_candidates));
      for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
        if (!runtime_blocks[candidate])
          *targets[candidate] = std::move(snapshots[candidate]);
      return;
    }

    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
      *targets[candidate] = std::move(snapshots[candidate]);
  }

  void apply_coupling_operators(std::string_view graph_identity, std::string_view rate_identity,
                                std::string_view application_identity, Real dt,
                                std::initializer_list<CouplingStateOverride> candidates) const {
    require_facade_execution_();
    std::vector<field_type*> program_states;
    std::optional<runtime::multiblock::BoundaryEvaluationPoint> prepared_point;
    std::optional<runtime::multiblock::InterfaceFluxFragmentPublication> prepared_publication;
    std::exception_ptr local_error;
    try {
      program_states.assign(static_cast<std::size_t>(n_blocks()), nullptr);
      if (graph_identity.empty() || graph_identity != facade_->installed_program_hash() ||
          rate_identity.empty() || application_identity.empty())
        throw std::invalid_argument(
            "AMR Program coupling requires exact graph, rate, and application identities");
      const auto& coupling_budget = facade_->prepared_amr_program_flux_expression_budget();
      std::size_t identity_characters = graph_identity.size();
      if (rate_identity.size() > std::numeric_limits<std::size_t>::max() - identity_characters)
        throw std::length_error("AMR Program coupling identity characters exceed size_t");
      identity_characters += rate_identity.size();
      if (application_identity.size() >
          std::numeric_limits<std::size_t>::max() - identity_characters)
        throw std::length_error("AMR Program coupling identity characters exceed size_t");
      identity_characters += application_identity.size();
      if (identity_characters > coupling_budget.interface_coupling_identity_character_bound)
        throw std::length_error(
            "AMR Program coupling identities exceed the frozen artifact character bound");
      for (const CouplingStateOverride& candidate : candidates) {
        const int runtime_block = sys_block(candidate.program_block);
        if (candidate.state == nullptr ||
            program_states[static_cast<std::size_t>(candidate.program_block)] != nullptr)
          throw std::invalid_argument(
              "AMR Program coupling candidates are incomplete, duplicate, or null");
        require_same_field_contract_(
            *candidate.state, facade_->prepared_amr_block_state(runtime_block, active_level_),
            "AMR Program coupling candidate");
        if (!active_attempt_states_.empty()) {
          const bool detached_group_candidate =
              std::find(active_attempt_states_.begin(), active_attempt_states_.end(),
                        candidate.state) != active_attempt_states_.end();
          const bool prepared_scratch =
              std::any_of(scratches_.begin(), scratches_.end(),
                          [&](const auto& entry) { return &entry.second == candidate.state; });
          if (!detached_group_candidate && !prepared_scratch)
            throw std::invalid_argument(
                "active AMR Program coupling requires detached group candidates or prepared "
                "scratches");
          for (int accepted_block = 0; accepted_block < n_blocks(); ++accepted_block)
            if (candidate.state ==
                &facade_->prepared_amr_block_state(accepted_block, active_level_))
              throw std::invalid_argument(
                  "active AMR Program coupling cannot mutate an accepted block carrier");
        }
        program_states[static_cast<std::size_t>(candidate.program_block)] = candidate.state;
      }
      if (std::find(program_states.begin(), program_states.end(), nullptr) != program_states.end())
        throw std::invalid_argument(
            "AMR Program coupling requires every authenticated Program block candidate");
      const ::pops::amr::Rational interval_phase =
          active_subcycling_window_.begin.phase +
          (active_subcycling_window_.end.phase - active_subcycling_window_.begin.phase) *
              stage_time_;
      const double interval_duration = active_subcycling_window_.end.physical_time -
                                       active_subcycling_window_.begin.physical_time;
      const double evaluation_time =
          active_subcycling_window_.begin.physical_time + stage_time_.value() * interval_duration;
      if (!std::isfinite(interval_duration) || !(interval_duration > 0.0) ||
          !std::isfinite(static_cast<double>(dt)) || !(dt > Real(0)))
        throw std::logic_error("AMR Program interface publication has an invalid exact interval");
      prepared_point.emplace(runtime::multiblock::BoundaryEvaluationPoint{
          primary_clock_, static_cast<std::int64_t>(facade_->macro_step()), active_level_,
          logical_substep_, 0, stage_time_, interval_duration, evaluation_time,
          std::string(graph_identity), std::string(rate_identity),
          std::string(application_identity)});
      prepared_publication.emplace(runtime::multiblock::InterfaceFluxFragmentPublication{
          interface_flux_ledger_.get(),
          runtime_->topology_epoch(),
          nlev(),
          {active_level_, static_cast<std::int64_t>(facade_->macro_step()), interval_phase,
           evaluation_time},
          "program-stage:" + std::to_string(stage_time_.numerator) + "/" +
              std::to_string(stage_time_.denominator),
          active_subcycling_window_,
          exact_binary_rational_(dt / interval_duration),
          true});
    } catch (...) {
      local_error = std::current_exception();
    }
    const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR Program coupling candidate pack failed collectively");
    }
    count_kernel_(static_cast<std::int64_t>(facade_->apply_prepared_amr_program_candidates(
        active_level_, dt, std::span<field_type* const>(program_states), *prepared_point,
        interface_flux_ledger_->in_transaction() ? &*prepared_publication : nullptr)));
  }

  Real sum_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_sum(pops::reduce_sum_local(field, component), prepared_execution_lane()));
  }
  Real max_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_max(pops::reduce_max_local(field, component), prepared_execution_lane()));
  }
  Real min_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_min(pops::reduce_min_local(field, component), prepared_execution_lane()));
  }
  Real norm2(int, const field_type& field) const {
    return std::sqrt(static_cast<Real>(
        all_reduce_sum(pops::dot_local(field, field, 0), prepared_execution_lane())));
  }
  Real norm_inf(int, const field_type& field) const {
    return static_cast<Real>(all_reduce_max(pops::norm_inf(field, 0), prepared_execution_lane()));
  }
  Real dot(int, const field_type& left, const field_type& right) const {
    return static_cast<Real>(
        all_reduce_sum(pops::dot_local(left, right, 0), prepared_execution_lane()));
  }

  Geometry<Dim> geometry() const {
    require_facade_execution_();
    return facade_->prepared_amr_level_geometry(active_level_);
  }

  field_type& assembly_target(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("AMR Program assembly target requires an identity");
    if (!hierarchy_tensor_selection_)
      return field;
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback)
      return field;
    if (std::find(hierarchy_tensor_selection_->assembly_field_slots.begin(),
                  hierarchy_tensor_selection_->assembly_field_slots.end(),
                  identity) == hierarchy_tensor_selection_->assembly_field_slots.end())
      throw std::invalid_argument("AMR hierarchy assembly used an undeclared provider field slot");
    return solver.assembly_target(identity, active_level_);
  }

  field_type& assembly_source(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("AMR Program assembly source requires an identity");
    if (!hierarchy_tensor_selection_)
      return field;
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback)
      return field;
    if (identity != hierarchy_tensor_selection_->solution_field_slot)
      throw std::invalid_argument("AMR hierarchy read used an undeclared provider solution slot");
    return solver.solution(active_level_);
  }

  std::shared_ptr<scalar_boundary_session_type> prepare_mesh_boundary_session(
      field_type& prototype, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "AMR scalar boundary preparation");
    return std::make_shared<scalar_boundary_session_type>(
        geometry(), facade_->prepared_amr_boundary_topology(), prototype, lane,
        next_boundary_generation_());
  }

  std::shared_ptr<block_boundary_session_type> prepare_block_boundary_session(
      int program_block, field_type& prototype,
      const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "AMR block boundary preparation");
    const int runtime_block = sys_block(program_block);
    require_boundary_point_(point, "AMR block scalar boundary");
    require_same_field_contract_(prototype, state(program_block), "AMR block boundary prototype");
    auto transport = scalar_boundary_session_type::prepare_block(
        geometry(), facade_->prepared_amr_boundary_topology(), prototype, lane,
        next_boundary_generation_());
    return std::shared_ptr<block_boundary_session_type>(
        new block_boundary_session_type(facade_, runtime_block, point, lane, std::move(transport)));
  }

  void fill_boundary(field_type& field) const { fill_boundary(field, prepared_execution_lane()); }

  void fill_boundary(field_type& field, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "AMR boundary fill");
    scalar_boundary_session_type session(geometry(), facade_->prepared_amr_boundary_topology(),
                                         field, lane, next_boundary_generation_());
    session.fill(field);
  }

  void laplacian(field_type& output, field_type& input) const {
    require_scalar_stencil_(output, input, 1, "AMR Program Laplacian");
    fill_boundary(input);
    const Geometry<Dim> geom = geometry();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<Real, Dim> result = output.fab(local).view();
      const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        Real image = Real(0);
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell;
          Index<Dim> upper = cell;
          --lower[axis];
          ++upper[axis];
          const Real spacing = geom.spacing(axis);
          image +=
              (value(upper, 0) - Real(2) * value(cell, 0) + value(lower, 0)) / (spacing * spacing);
        }
        result(cell, 0) = image;
      });
    }
    count_kernel_();
  }

  void laplacian(field_type& output, field_type& input,
                 const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "AMR Laplacian boundary");
    boundary.fill(input);
    laplacian_without_fill_(output, input, boundary.geometry());
  }
  void laplacian(field_type& output, field_type& input,
                 const scalar_boundary_session_type& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "AMR Program Laplacian");
    laplacian(output, input, boundary);
  }

  void gradient(field_type& output, field_type& input) const {
    require_scalar_stencil_(output, input, Dim, "AMR Program gradient");
    fill_boundary(input);
    const Geometry<Dim> geom = geometry();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<Real, Dim> result = output.fab(local).view();
      const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell;
          Index<Dim> upper = cell;
          --lower[axis];
          ++upper[axis];
          result(cell, axis) = (value(upper, 0) - value(lower, 0)) / (Real(2) * geom.spacing(axis));
        }
      });
    }
    count_kernel_();
  }

  void gradient(field_type& output, field_type& input,
                const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "AMR gradient boundary");
    boundary.fill(input);
    gradient_without_fill_(output, input, boundary.geometry());
  }
  void gradient(field_type& output, field_type& input, const scalar_boundary_session_type& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "AMR Program gradient");
    gradient(output, input, boundary);
  }

  void divergence(field_type& output, field_type& flux) const {
    auto boundary = prepare_mesh_boundary_session(flux, prepared_execution_lane());
    divergence(output, flux, *boundary);
  }
  void divergence(field_type& output, field_type& flux,
                  const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "AMR divergence boundary");
    if (output.ncomp() != 1 || flux.ncomp() != Dim)
      throw std::invalid_argument("AMR Program divergence requires one exact native vector field");
    require_same_layout_(output, flux, "AMR Program divergence");
    boundary.fill(flux);
    const Geometry<Dim> geom = boundary.geometry();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<Real, Dim> result = output.fab(local).view();
      const FieldView<const Real, Dim> vector = std::as_const(flux).fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        Real value = Real(0);
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell;
          Index<Dim> upper = cell;
          --lower[axis];
          ++upper[axis];
          value += (vector(upper, axis) - vector(lower, axis)) / (Real(2) * geom.spacing(axis));
        }
        result(cell, 0) = value;
      });
    }
    count_kernel_();
  }
  void divergence(field_type& output, field_type& flux,
                  const scalar_boundary_session_type& boundary,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "AMR Program divergence");
    divergence(output, flux, boundary);
  }

  void pack_vector(field_type& output, const std::array<const field_type*, Dim>& components) const {
    if (output.ncomp() != Dim)
      throw std::invalid_argument("AMR Program vector packing requires Dim output components");
    for (const field_type* component : components)
      if (component == nullptr || component->ncomp() != 1)
        throw std::invalid_argument("AMR Program vector packing requires scalar components");
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      std::array<FieldView<const Real, Dim>, Dim> values{};
      for (int axis = 0; axis < Dim; ++axis) {
        require_same_layout_(output, *components[static_cast<std::size_t>(axis)],
                             "AMR Program vector packing");
        values[static_cast<std::size_t>(axis)] =
            components[static_cast<std::size_t>(axis)]->fab(local).view();
      }
      const FieldView<Real, Dim> result = output.fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int axis = 0; axis < Dim; ++axis)
          result(cell, axis) = values[static_cast<std::size_t>(axis)](cell, 0);
      });
    }
    count_kernel_();
  }

  [[noreturn]] void tensor_laplacian(field_type&, field_type&, const field_type&) const {
    unavailable_("AMR tensor-elliptic provider");
  }
  template <class... Arguments>
  [[noreturn]] void tensor_laplacian(Arguments&&...) const {
    unavailable_("AMR tensor-elliptic provider");
  }

  /// Copy one exact valid-cell component span without exposing distributed storage to generated
  /// code.  Aliasing copies select their component direction before the kernel launches, so an
  /// overlapping in-place pack cannot overwrite a value that has not yet been read.
  void copy_component_span(field_type& destination, int destination_component,
                           const field_type& source, int source_component,
                           int component_count) const {
    if (component_count <= 0 || destination_component < 0 || source_component < 0 ||
        destination_component > destination.ncomp() - component_count ||
        source_component > source.ncomp() - component_count)
      throw std::invalid_argument("AMR Program component-span copy has an invalid range");
    require_same_layout_(destination, source, "AMR Program component-span copy");
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      if (destination.global_index(local) != source.global_index(local))
        throw std::logic_error(
            "AMR Program component-span copy found inconsistent local ownership");
    }
    if (&destination == &source && destination_component == source_component)
      return;
    const bool copy_backward = &destination == &source &&
                               destination_component > source_component &&
                               destination_component < source_component + component_count;
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      const FieldView<Real, Dim> output = destination.fab(local).view();
      const FieldView<const Real, Dim> input = std::as_const(source).fab(local).view();
      for_each_cell(destination.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        if (copy_backward) {
          for (int offset = component_count; offset-- > 0;)
            output(cell, destination_component + offset) = input(cell, source_component + offset);
        } else {
          for (int offset = 0; offset < component_count; ++offset)
            output(cell, destination_component + offset) = input(cell, source_component + offset);
        }
      });
    }
    count_kernel_();
  }

  /// Register one level-qualified exact-ranked history ring.  The generated AMR installer invokes
  /// this while constructing each level bundle, so one authored history maps to one immutable
  /// layout contract per active level instead of a 2-D or owner-erased global buffer.
  void register_history(const std::string& name, int lag, int ncomp, int program_owner,
                        const std::string& state_identity, const std::string& space_identity,
                        const std::string& clock_identity,
                        const std::string& interpolation_identity) const {
    if (name.empty() || lag < 1 || program_owner < 0 || state_identity.empty() ||
        space_identity.empty() || clock_identity.empty() || interpolation_identity.empty())
      throw std::invalid_argument(
          "AMR Program history requires complete owner/state/space/clock identities");
    const int runtime_owner = sys_block(program_owner);
    refresh_resources_();
    const field_type& prototype = state(program_owner);
    const int components = ncomp < 0 ? prototype.ncomp() : ncomp;
    if (components < 1)
      throw std::invalid_argument("AMR Program history component count must be positive");
    const std::string key = history_key_(name, active_level_);
    auto& manager = runtime_state().hist_;
    const int depth = lag + 1;
    const auto found = manager.histories.find(key);
    if (found != manager.histories.end()) {
      const field_type& retained = found->second.front();
      if (manager.depth.at(key) != depth || manager.owner.at(key) != runtime_owner ||
          retained.layout() != prototype.layout() ||
          retained.distribution() != prototype.distribution() ||
          retained.local_rank() != prototype.local_rank() || retained.ncomp() != components ||
          retained.ghosts() != prototype.ghosts() ||
          manager.state_identity.at(key) != state_identity ||
          manager.space_identity.at(key) != space_identity ||
          manager.clock_identity.at(key) != clock_identity ||
          manager.interpolation_identity.at(key) != interpolation_identity)
        throw std::runtime_error(
            "AMR Program history identity changed after exact-ranked registration");
      history_levels_.insert_or_assign(key, active_level_);
      return;
    }

    std::vector<field_type> ring;
    ring.reserve(static_cast<std::size_t>(depth));
    for (int slot = 0; slot < depth; ++slot)
      ring.push_back(make_scratch_(prototype, components, prototype.ghosts()));
    manager.histories.emplace(key, std::move(ring));
    manager.depth[key] = depth;
    manager.initialized[key] = false;
    manager.fill_count[key] = 0;
    manager.store_pending[key] = false;
    manager.owner[key] = runtime_owner;
    manager.state_identity[key] = state_identity;
    manager.space_identity[key] = space_identity;
    manager.clock_identity[key] = clock_identity;
    manager.interpolation_identity[key] = interpolation_identity;
    manager.slot_dt[key] = std::vector<Real>(static_cast<std::size_t>(depth), Real(0));
    history_levels_.emplace(key, active_level_);
    if (history_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
      history_epoch_ = runtime_->topology_epoch();
      history_generation_ = runtime_->materialization_generation();
    }
  }

  field_type& history(const std::string& name, int lag, int program_owner) const {
    require_history_owner_(program_owner);
    return history_slot_(name, lag, /*zero_start=*/false, /*components=*/-1);
  }
  field_type& history(const std::string& name, int lag = 1) const {
    return history_slot_(name, lag, /*zero_start=*/false, /*components=*/-1);
  }
  field_type& history_zero_start(const std::string& name, int lag, int ncomp,
                                 int program_owner) const {
    require_history_owner_(program_owner);
    return history_slot_(name, lag, /*zero_start=*/true, ncomp);
  }
  field_type& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    return history_slot_(name, lag, /*zero_start=*/true, ncomp);
  }

  void store_history(const std::string& name, const field_type& value, int program_owner) const {
    require_history_owner_(program_owner);
    store_history_(name, value);
  }
  void store_history(const std::string& name, const field_type& value) const {
    store_history_(name, value);
  }

  void rotate_histories() const { rotate_histories_(std::nullopt); }
  void rotate_histories(const std::string& clock_identity) const {
    if (clock_identity.empty())
      throw std::invalid_argument("AMR Program history rotation requires a clock identity");
    rotate_histories_(clock_identity);
  }

  void interpolate_history_linear(field_type& output, const std::string& name, int max_lag,
                                  int program_owner, const std::string& source_clock,
                                  const std::string& target_clock, int target_step,
                                  Real target_offset) const {
    require_history_owner_(program_owner);
    if (max_lag < 1 || !std::isfinite(static_cast<double>(target_offset)))
      throw std::invalid_argument("AMR linear history interpolation has an invalid target");
    const std::string key = history_key_(name, active_level_);
    auto& manager = runtime_state().hist_;
    const auto found = manager.histories.find(key);
    if (found == manager.histories.end() || manager.depth.at(key) <= max_lag ||
        !manager.initialized.at(key))
      throw std::runtime_error(
          "AMR linear history interpolation requires an initialized retained ring");
    require_same_field_contract_(output, found->second.front(), "AMR linear history interpolation");

    const double source_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(source_clock));
    const double target_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(target_clock));
    const double coordinate =
        (static_cast<double>(target_step) + static_cast<double>(target_offset)) * source_ticks /
        target_ticks;
    if (!std::isfinite(coordinate) || coordinate > 0.0 ||
        coordinate < -static_cast<double>(max_lag))
      throw std::runtime_error(
          "AMR linear history interpolation target lies outside retained timestamps");
    if (coordinate == 0.0) {
      copy_valid_(found->second.front(), output);
      count_kernel_();
      return;
    }

    const int older_lag = static_cast<int>(std::ceil(-coordinate));
    if (older_lag < 1 || older_lag > max_lag)
      throw std::runtime_error(
          "AMR linear history interpolation could not select bracketing slots");
    double newer_time = static_cast<double>(physical_time());
    double older_time = newer_time;
    double bracket_dt = 0.0;
    for (int selected_lag = 1; selected_lag <= older_lag; ++selected_lag) {
      const double interval =
          static_cast<double>(manager.slot_dt.at(key)[static_cast<std::size_t>(selected_lag)]);
      if (!std::isfinite(interval) || !(interval > 0.0))
        throw std::runtime_error(
            "AMR linear history interpolation requires positive exact slot timestamps");
      bracket_dt = interval;
      older_time = newer_time - interval;
      if (selected_lag != older_lag)
        newer_time = older_time;
    }
    const double logical_fraction = coordinate + static_cast<double>(older_lag);
    const double target_time = older_time + logical_fraction * bracket_dt;
    const double alpha = (target_time - older_time) / (newer_time - older_time);
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0)
      throw std::runtime_error(
          "AMR linear history interpolation target does not bracket retained timestamps");
    lincomb(output, Real(1) - static_cast<Real>(alpha),
            found->second[static_cast<std::size_t>(older_lag)], static_cast<Real>(alpha),
            found->second[static_cast<std::size_t>(older_lag - 1)]);
  }

  /// The exact-ranked AMR cache is deliberately unavailable until its per-level values and elapsed
  /// windows participate in the AMR checkpoint/regrid transaction.  A cache-backed schedule is
  /// rejected at its decision seam, before its node body can mutate auxiliary or live storage.
  [[noreturn]] bool cache_should_update(int, int) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }
  [[noreturn]] void cache_store_aux(int) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }
  [[noreturn]] void cache_restore_aux(int) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }
  [[noreturn]] void cache_store_scratch(int, const field_type&) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }
  [[noreturn]] void cache_restore_scratch(int, field_type&) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }
  [[noreturn]] void cache_accumulate_dt(int, Real) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }
  [[noreturn]] Real cache_effective_dt(int, Real) const {
    unavailable_("checkpointed AMR scheduler cache provider");
  }

  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
  }
  bool schedule_is_due(int node_id, int every_n, ScheduleDomainKind kind, const std::string& clock,
                       const std::string& stage_identity, int level) const {
    if (node_id < 0 || every_n <= 0)
      throw std::invalid_argument("AMR Program schedule has an invalid node or period");
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
      throw std::invalid_argument("AMR Program schedule decision has an invalid node");
    if (cache_backed)
      unavailable_("checkpointed AMR scheduler cache provider");
    return runtime_state().profiler_.schedule_decision(due, false);
  }
  [[noreturn]] void scheduler_error(const std::string& message) const {
    throw std::runtime_error(message.empty() ? "AMR Program scheduled node is unavailable"
                                             : message);
  }

  const field_type* pointwise_active_mask(int program_block, const field_type& field) const {
    if (sys_block(program_block) != 0)
      unavailable_("exact-ranked multi-block AMR pointwise mask provider");
    refresh_resources_();
    require_same_layout_(field, state(program_block), "AMR Program pointwise mask");
    if (nlev() != 1)
      unavailable_("composite active-cell AMR pointwise mask provider");
    return nullptr;
  }
  Real pointwise_status_max(int program_block, const field_type& status,
                            const field_type* active_cells, const ExecutionLane& lane) const {
    if (&lane != &prepared_execution_lane())
      throw std::invalid_argument("AMR Program pointwise status requires its prepared lane");
    const field_type* expected = pointwise_active_mask(program_block, status);
    if (active_cells != expected)
      throw std::invalid_argument(
          "AMR Program pointwise status received a foreign active-cell mask");
    if (status.ncomp() < 1)
      throw std::invalid_argument("AMR Program pointwise status requires one component");
    const Real result = static_cast<Real>(
        all_reduce_max(static_cast<double>(pops::reduce_max_local(status, 0)), lane));
    return result == -std::numeric_limits<Real>::infinity() ? Real(0) : result;
  }

  void register_hierarchy_tensor_solver_provider(
      std::shared_ptr<const hierarchy_tensor_provider_type> provider) const {
    require_facade_execution_();
    if (!hierarchy_tensor_solver_registry_)
      throw std::logic_error("AMR hierarchy tensor-solver registry is unavailable");
    facade_->register_program_hierarchy_tensor_solver_provider(std::move(provider));
  }

  void configure_hierarchy_tensor_solver(int program_block, int components,
                                         const std::string& provider_identity,
                                         const std::string& plan_identity,
                                         const std::string& operator_contract_identity,
                                         const std::vector<std::string>& assembly_field_slots,
                                         const std::string& solution_field_slot,
                                         const PreparedProviderOptions& options) const {
    require_facade_execution_();
    if (!hierarchy_tensor_solver_registry_)
      throw std::logic_error("AMR hierarchy tensor-solver registry is unavailable");

    const ExecutionLane& lane = prepared_execution_lane();
    std::optional<HierarchyTensorSelection> staged;
    std::exception_ptr local_error;
    long local_failure = 0;
    try {
      (void)sys_block(program_block);
      if (components < 1 || provider_identity.empty() || plan_identity.empty() ||
          operator_contract_identity.empty() || assembly_field_slots.empty() ||
          solution_field_slot.empty() ||
          std::any_of(
              assembly_field_slots.begin(), assembly_field_slots.end(),
              [](const std::string& slot) { return slot.empty(); }) ||
          std::set<std::string>(assembly_field_slots.begin(), assembly_field_slots.end()).size() !=
              assembly_field_slots.size())
        throw std::invalid_argument(
            "AMR hierarchy tensor solver requires a complete exact field-slot envelope");
      ExactContractBuilder contract;
      contract.text("pops.hierarchy.tensor-solver-selection")
          .scalar(std::uint32_t{2})
          .scalar(std::int32_t{Dim})
          .scalar(std::int32_t{program_block})
          .scalar(std::int32_t{components})
          .text(provider_identity)
          .text(plan_identity)
          .text(operator_contract_identity)
          .sequence(assembly_field_slots,
                    [](ExactContractBuilder& item, const std::string& slot) { item.text(slot); })
          .text(solution_field_slot)
          .bytes(options.exact_contract());
      staged.emplace(HierarchyTensorSelection{
          program_block, components, provider_identity, plan_identity, operator_contract_identity,
          assembly_field_slots, solution_field_slot, options, std::move(contract).release()});
      if (hierarchy_tensor_selection_ &&
          hierarchy_tensor_selection_->exact_contract != staged->exact_contract)
        throw std::logic_error("AMR hierarchy tensor solver is already configured differently");
    } catch (...) {
      local_failure = 1;
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_failure, lane) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR hierarchy tensor selection failed on another MPI rank");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-hierarchy-tensor-selection", staged->exact_contract}}, lane))
      throw std::invalid_argument("AMR hierarchy tensor selection differs between MPI ranks");

    if (hierarchy_tensor_selection_ &&
        hierarchy_tensor_selection_->exact_contract == staged->exact_contract) {
      (void)configured_hierarchy_tensor_solver_();
      return;
    }

    std::unique_ptr<hierarchy_tensor_solver_type> prepared =
        prepare_hierarchy_tensor_solver_(*staged);
    hierarchy_tensor_selection_ = std::move(staged);
    hierarchy_tensor_solver_ = std::move(prepared);
    hierarchy_tensor_topology_epoch_ = runtime_->topology_epoch();
    hierarchy_tensor_materialization_generation_ = runtime_->materialization_generation();
  }

  SolveOutcome solve_hierarchy_tensor(int program_block, int components, Real relative_tolerance,
                                      Real absolute_tolerance, int maximum_iterations) const {
    require_hierarchy_tensor_binding_(program_block, components);
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidInput, SolveAction::kRejectAttempt);
      return SolveOutcome::collective_lane(std::move(report), prepared_execution_lane());
    }
    return solve_prepared_hierarchy_tensor_collectively(
        solver,
        HierarchyTensorSolveControls{relative_tolerance, absolute_tolerance, maximum_iterations},
        prepared_execution_lane());
  }

  field_type& hierarchy_solution() const {
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
      throw std::logic_error(
          "provider-owned hierarchy solution requested on a prepared Krylov fallback path");
    return solver.solution(active_level_);
  }
  field_type& linear_solution(field_type& fallback) const {
    require_same_layout_(fallback, state(0), "AMR Program linear solution");
    if (!hierarchy_tensor_selection_)
      return fallback;
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    return solver.execution_path() == HierarchyTensorSolverExecutionPath::DirectProvider
               ? solver.solution(active_level_)
               : fallback;
  }
  void stage_linear_initial_guess() const {
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
      throw std::logic_error("hierarchy initial guess requires direct provider execution");
    solver.stage_initial_guess(active_level_, nullptr);
  }
  void stage_linear_initial_guess(const field_type& guess) const {
    hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
    if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
      throw std::logic_error("hierarchy initial guess requires direct provider execution");
    solver.stage_initial_guess(active_level_, &guess);
  }

  ClockScheduleState::SubcycleScope subcycle_scope(const std::string& parent,
                                                   const std::string& child, int count) const {
    return clock_schedule_.subcycle(parent, child, count);
  }
  LogicalEvaluationScope logical_evaluation_scope(int iteration, int count) const {
    return LogicalEvaluationScope(*this, iteration, count);
  }
  void synchronize_sample_and_hold(const std::string& source, const std::string& target, int step,
                                   Real offset) const {
    clock_schedule_.synchronize_sample_and_hold(source, target, step, static_cast<double>(offset));
  }

  int macro_step() const { return facade_->macro_step(); }
  Real physical_time() const { return static_cast<Real>(facade_->time()); }

  void record_scalar(const std::string& name, Real value) const {
    facade_->record_program_diagnostic(name, static_cast<double>(value));
  }
  void record_balance_term(const std::string& route, const std::string& term, Real value) const {
    facade_->record_program_balance_term(route, term, static_cast<double>(value));
  }
  bool balance_consumer_is_due(const std::string& contract, const std::string& route,
                               int every_n) const {
    return facade_->program_balance_consumer_is_due(contract, route, every_n);
  }
  void note_automatic_balance_capture_due(bool due) const {
    runtime_state().note_automatic_balance_capture_due(due, "AmrProgramContext");
  }
  void note_step_projection(const std::string& name) const {
    runtime_state().note_step_projection(name);
  }
  void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    facade_->profiler_handle().record(name, std::chrono::duration<double>(elapsed).count());
  }

  runtime_state_type& runtime_state() const { return facade_->program_runtime_state_(); }

  const PreparedVectorDistribution<Dim>& program_resource_vector_distribution() const {
    refresh_resources_();
    vector_distribution_ = runtime_->hierarchy()
                                   .layout(static_cast<std::size_t>(active_level_))
                                   .distribution()
                                   .replicated()
                               ? PreparedVectorDistribution<Dim>::replicated()
                               : PreparedVectorDistribution<Dim>::distributed();
    return vector_distribution_;
  }
  int program_resource_field_level() const noexcept { return active_level_; }
  void configure_program_resource_field_nullspace(FieldNullspacePlan<Dim>& plan) const {
    Real cell_measure = Real(1);
    const Geometry<Dim> geom = geometry();
    for (int axis = 0; axis < Dim; ++axis)
      cell_measure *= geom.spacing(axis);
    for (FieldNullspaceBasis<Dim>& basis : plan.bases)
      basis.cell_measure = {cell_measure};
  }

  SolveOutcome solve_prepared_linear(const PreparedAffineLinearProblem<Dim>& problem,
                                     KrylovWorkspace<Dim>& workspace, field_type& solution,
                                     const field_type& rhs,
                                     const KrylovControls<Dim>& controls) const {
    require_prepared_lane_(workspace.execution_lane(), "AMR prepared linear solve");
    return pops::solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls);
  }

  OperatorEvaluationSnapshot operator_evaluation_snapshot(OperatorFingerprint authority,
                                                          const field_type& prototype,
                                                          OperatorFingerprint resources) const {
    if (operator_snapshot_revision_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program operator snapshot revision exhausted uint64_t");
    const OperatorFingerprint topology = operator_topology_(prototype);
    OperatorEvaluationSnapshot snapshot =
        current_operator_snapshot_(authority, topology, resources, ++operator_snapshot_revision_);
    if (!snapshot.valid())
      throw std::runtime_error("AMR Program produced an invalid exact operator snapshot");
    active_operator_snapshot_ = snapshot;
    return snapshot;
  }

  OperatorEvaluationSnapshot probe_operator_evaluation(OperatorFingerprint authority,
                                                       OperatorFingerprint topology,
                                                       OperatorFingerprint resources,
                                                       std::uint64_t revision) const {
    const bool active =
        active_operator_snapshot_ && active_operator_snapshot_->revision == revision;
    OperatorEvaluationSnapshot probe =
        current_operator_snapshot_(authority, topology, resources, active ? revision : 0);
    if (!active || probe != *active_operator_snapshot_) {
      active_operator_snapshot_.reset();
      probe.revision = 0;
    }
    return probe;
  }

  void set_field_boundary_kernel(const std::string& provider_slot,
                                 const CompiledFieldBoundaryKernel<Dim>& kernel) const {
    facade_->set_field_boundary_kernel(provider_slot, kernel);
  }
  void set_field_logical_timepoint(const std::string& provider_slot,
                                   const FieldLogicalTimePoint& point) const {
    facade_->set_field_logical_timepoint(provider_slot, point);
  }

  bool has_boundary_linearization(int program_block) const noexcept {
    try {
      return facade_ != nullptr &&
             facade_->has_prepared_amr_block_boundary_linearization(sys_block(program_block));
    } catch (...) {
      return false;
    }
  }
  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 int program_block, field_type& state, field_type& residual,
                                 const block_boundary_session_type& boundary) const {
    require_block_boundary_session_(point, program_block, boundary, "AMR boundary residual");
    facade_->prepared_amr_block_level_boundary_residual_into_at(boundary.runtime_block_, point,
                                                                state, residual);
  }
  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            int program_block, field_type& state, const field_type& direction,
                            field_type& result, const block_boundary_session_type& boundary) const {
    require_block_boundary_session_(point, program_block, boundary, "AMR boundary JVP");
    facade_->prepared_amr_block_level_boundary_jvp_into_at(boundary.runtime_block_, point, state,
                                                           direction, result);
  }
  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                        int program_block, field_type& state, field_type& residual, bool flux_only,
                        const block_boundary_session_type& boundary) const {
    require_block_boundary_session_(point, program_block, boundary, "AMR core RHS");
    facade_->prepared_amr_block_level_rhs_core_into_at(boundary.runtime_block_, point, state,
                                                       residual, flux_only);
  }
  void rhs_jacvec_pair_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                               int first_block, field_type& first_state, field_type& first_result,
                               bool first_flux_only, int second_block, field_type& second_state,
                               field_type& second_result, bool second_flux_only) const {
    require_boundary_point_(point, "AMR coupled Jacobian residual");
    if (first_block == second_block || n_blocks() != 2 || !(point.dt > 0.0))
      throw std::invalid_argument(
          "AMR coupled Jacobian residual requires two distinct complete Program blocks");
    std::unique_lock scratch_lock(coupled_jacvec_mutex_, std::try_to_lock);
    if (!scratch_lock.owns_lock())
      throw std::logic_error("AMR coupled Jacobian scratch is already in use");
    CoupledJacvecLevelScratch& scratch = require_coupled_jacvec_scratch_(
        first_block, first_state, first_result, second_block, second_state, second_result);
    // The generated block operators and the prepared multi-block coupling/interface scheduler
    // remain the only numerical engines. Evaluate the detached block candidates first, then run
    // the exact scheduler on copies and recover its rate from the transactional candidate delta.
    const int first_runtime = sys_block(first_block);
    const int second_runtime = sys_block(second_block);
    field_type& first_candidate = *scratch.residual[static_cast<std::size_t>(first_runtime)];
    field_type& second_candidate = *scratch.residual[static_cast<std::size_t>(second_runtime)];
    field_type& first_coupled = *scratch.coupled[static_cast<std::size_t>(first_runtime)];
    field_type& second_coupled = *scratch.coupled[static_cast<std::size_t>(second_runtime)];
    first_candidate.set_val(Real(0));
    second_candidate.set_val(Real(0));
    copy_full_(first_state, first_coupled);
    copy_full_(second_state, second_coupled);
    const auto& first_evaluation =
        first_flux_only
            ? facade_->evaluate_prepared_amr_block_level_flux_at(first_runtime, point, first_state)
            : facade_->evaluate_prepared_amr_block_level_at(first_runtime, point, first_state);
    copy_valid_(first_evaluation.residual, first_candidate);
    const auto& second_evaluation =
        second_flux_only
            ? facade_->evaluate_prepared_amr_block_level_flux_at(second_runtime, point,
                                                                 second_state)
            : facade_->evaluate_prepared_amr_block_level_at(second_runtime, point, second_state);
    copy_valid_(second_evaluation.residual, second_candidate);

    std::array<field_type*, 2> candidates{};
    candidates[static_cast<std::size_t>(first_block)] = &first_coupled;
    candidates[static_cast<std::size_t>(second_block)] = &second_coupled;
    if (!interface_flux_ledger_ || !interface_flux_ledger_->in_transaction())
      throw std::logic_error(
          "AMR coupled Jacobian residual requires its active interface-ledger transaction");
    runtime::multiblock::InterfaceFluxFragmentPublication publication{
        interface_flux_ledger_.get(),
        runtime_->topology_epoch(),
        nlev(),
        {active_level_, static_cast<std::int64_t>(facade_->macro_step()), point.stage_fraction,
         point.physical_time},
        "program-jacvec-pair",
        active_subcycling_window_,
        exact_binary_rational_(static_cast<Real>(point.dt) /
                               (active_subcycling_window_.end.physical_time -
                                active_subcycling_window_.begin.physical_time)),
        true};
    interface_flux_ledger_->begin();
    try {
      (void)facade_->apply_prepared_amr_program_candidates(
          active_level_, static_cast<Real>(point.dt),
          std::span<field_type* const>(candidates.data(), candidates.size()), point, &publication);
      interface_flux_ledger_->rollback();
    } catch (...) {
      if (interface_flux_ledger_->transaction_depth() > 1)
        interface_flux_ledger_->rollback();
      throw;
    }
    saxpy(first_coupled, Real(-1), first_state);
    saxpy(second_coupled, Real(-1), second_state);
    saxpy(first_candidate, Real(1) / static_cast<Real>(point.dt), first_coupled);
    saxpy(second_candidate, Real(1) / static_cast<Real>(point.dt), second_coupled);
    copy_valid_(first_candidate, first_result);
    copy_valid_(second_candidate, second_result);
  }
  template <class Function>
  void evaluate_with_field_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                    const std::string& provider_slot, int program_block,
                                    field_type& perturbed, const field_type& accepted,
                                    Function&& evaluate) const {
    require_boundary_point_(point, "AMR perturbed field-state provider");
    if (provider_slot.empty() || sys_block(program_block) != 0)
      throw std::invalid_argument(
          "AMR perturbed field-state provider requires one exact mono-block field route");
    require_same_field_contract_(perturbed, accepted, "AMR perturbed field state");
    facade_->with_program_field_candidate_at(
        point, provider_slot, active_level_, perturbed,
        std::function<void()>(std::forward<Function>(evaluate)));
  }

  [[nodiscard]] SolveOutcome solve_fields() const {
    refresh_resources_();
    return facade_->solve_program_default_field(active_level_);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_state_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int program_block, field_type& stage) const {
    refresh_resources_();
    require_boundary_point_(point, "AMR Program single-state field solve");
    if (provider_slot.empty())
      throw std::invalid_argument("AMR Program field solve requires an exact provider slot");
    if (sys_block(program_block) != 0)
      unavailable_("exact-ranked multi-block AMR field-state provider");
    require_same_field_contract_(stage, state(program_block), "AMR Program field stage override");
    return facade_->solve_program_field_at(point, provider_slot, active_level_, &stage);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_blocks_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t value_id,
      std::string_view field, std::initializer_list<FieldStageOverride> overrides) const {
    refresh_resources_();
    require_boundary_point_(point, "AMR Program simultaneous field solve");
    if (value_id < 0 || field.empty() || overrides.size() == 0)
      throw std::invalid_argument(
          "AMR Program simultaneous field solve requires an IR identity, field, and stages");

    GeneratedFieldRoute candidate;
    candidate.field.assign(field.data(), field.size());
    std::vector<const field_type*> runtime_stages(static_cast<std::size_t>(n_blocks()), nullptr);
    std::vector<const field_type*> unique_stages;
    unique_stages.reserve(overrides.size());
    for (const FieldStageOverride& override_value : overrides) {
      if (override_value.state == nullptr)
        throw std::invalid_argument("AMR Program simultaneous field stage override cannot be null");
      if (std::find(candidate.program_blocks.begin(), candidate.program_blocks.end(),
                    override_value.program_block) != candidate.program_blocks.end())
        throw std::invalid_argument(
            "AMR Program simultaneous field solve contains a duplicate Program block");
      if (std::find(unique_stages.begin(), unique_stages.end(), override_value.state) !=
          unique_stages.end())
        throw std::invalid_argument(
            "AMR Program simultaneous field solve aliases two stage overrides");
      const int runtime_block = sys_block(override_value.program_block);
      if (runtime_stages[static_cast<std::size_t>(runtime_block)] != nullptr)
        throw std::invalid_argument(
            "AMR Program simultaneous field solve maps two stages to one runtime block");
      if (runtime_block != 0)
        unavailable_("exact-ranked multi-block AMR field-state provider");
      require_same_field_contract_(*override_value.state, state(override_value.program_block),
                                   "AMR Program simultaneous field stage override");
      candidate.program_blocks.push_back(override_value.program_block);
      runtime_stages[static_cast<std::size_t>(runtime_block)] = override_value.state;
      unique_stages.push_back(override_value.state);
    }
    const auto existing = generated_field_routes_.find(value_id);
    if (existing == generated_field_routes_.end()) {
      generated_field_routes_.emplace(value_id, candidate);
    } else if (existing->second != candidate) {
      throw std::logic_error(
          "AMR Program simultaneous field IR identity changed its qualified route");
    }
    return facade_->solve_program_field_from_blocks_at(
        point, generated_field_routes_.at(value_id).field, active_level_, runtime_stages);
  }

  [[nodiscard]] SolveOutcome solve_default_field_on_coarse_level() const {
    refresh_resources_();
    return facade_->solve_program_default_field(0);
  }

 private:
  enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
  using ScratchKey = std::tuple<ScratchKind, int, std::int64_t, int>;
  using ExactPolynomial = std::map<int, ::pops::amr::Rational>;

  class AcceptedContextSnapshot final : public AcceptedProgramContextSnapshot {
   public:
    explicit AcceptedContextSnapshot(AmrProgramContext& owner)
        : owner_(&owner),
          clock_schedule_(owner.clock_schedule_),
          resource_epoch_(owner.resource_epoch_),
          resource_generation_(owner.resource_generation_),
          history_epoch_(owner.history_epoch_),
          history_generation_(owner.history_generation_),
          history_levels_(owner.history_levels_),
          accepted_temporal_partition_(owner.accepted_temporal_partition_),
          accepted_flux_budget_contract_(owner.accepted_flux_budget_contract_),
          accepted_coupling_contract_(owner.accepted_coupling_contract_),
          accepted_face_flux_(owner.accepted_face_flux_),
          interface_flux_ledger_(
              std::make_unique<interface_flux_ledger_type>(*owner.interface_flux_ledger_)),
          accepted_synchronization_events_(owner.accepted_synchronization_events_),
          accepted_state_revision_(owner.accepted_state_revision_) {}

    AcceptedContextSnapshot(const AcceptedContextSnapshot& accepted)
        : owner_(accepted.owner_),
          clock_schedule_(accepted.clock_schedule_),
          resource_epoch_(accepted.resource_epoch_),
          resource_generation_(accepted.resource_generation_),
          history_epoch_(accepted.history_epoch_),
          history_generation_(accepted.history_generation_),
          history_levels_(accepted.history_levels_),
          accepted_temporal_partition_(accepted.accepted_temporal_partition_),
          accepted_flux_budget_contract_(accepted.accepted_flux_budget_contract_),
          accepted_coupling_contract_(accepted.accepted_coupling_contract_),
          accepted_face_flux_(accepted.accepted_face_flux_),
          interface_flux_ledger_(
              std::make_unique<interface_flux_ledger_type>(*accepted.interface_flux_ledger_)),
          accepted_synchronization_events_(accepted.accepted_synchronization_events_),
          accepted_state_revision_(accepted.accepted_state_revision_) {}

    std::unique_ptr<AcceptedProgramContextSnapshot> prepare_restore() const override {
      return std::make_unique<AcceptedContextSnapshot>(*this);
    }

    void publish_restore() noexcept override {
      static_assert(std::is_nothrow_swappable_v<ClockScheduleState>);
      static_assert(std::is_nothrow_swappable_v<decltype(history_levels_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(accepted_flux_budget_contract_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(accepted_coupling_contract_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(interface_flux_ledger_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(discarded_scratches_)>);
      static_assert(std::is_nothrow_swappable_v<decltype(discarded_subcycling_contract_)>);
      std::swap(owner_->clock_schedule_, clock_schedule_);
      owner_->resource_epoch_ = resource_epoch_;
      owner_->resource_generation_ = resource_generation_;
      owner_->history_epoch_ = history_epoch_;
      owner_->history_generation_ = history_generation_;
      owner_->history_levels_.swap(history_levels_);
      owner_->scratches_.swap(discarded_scratches_);
      owner_->hierarchy_tensor_solver_.reset();
      owner_->hierarchy_tensor_topology_epoch_ = std::numeric_limits<std::uint64_t>::max();
      owner_->hierarchy_tensor_materialization_generation_ =
          std::numeric_limits<std::uint64_t>::max();
      owner_->multiblock_subcycling_.reset();
      owner_->multiblock_subcycling_epoch_ = std::numeric_limits<std::uint64_t>::max();
      owner_->multiblock_subcycling_generation_ = std::numeric_limits<std::uint64_t>::max();
      owner_->multiblock_subcycling_program_budget_contract_.swap(discarded_subcycling_contract_);
      std::swap(owner_->accepted_temporal_partition_, accepted_temporal_partition_);
      owner_->accepted_flux_budget_contract_.swap(accepted_flux_budget_contract_);
      owner_->accepted_coupling_contract_.swap(accepted_coupling_contract_);
      std::swap(owner_->accepted_face_flux_, accepted_face_flux_);
      owner_->interface_flux_commit_guard_.reset();
      owner_->interface_flux_ledger_.swap(interface_flux_ledger_);
      owner_->accepted_synchronization_events_.swap(accepted_synchronization_events_);
      owner_->accepted_state_revision_ = accepted_state_revision_;
    }

   private:
    AmrProgramContext* owner_ = nullptr;
    ClockScheduleState clock_schedule_;
    std::uint64_t resource_epoch_ = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t resource_generation_ = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t history_epoch_ = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t history_generation_ = std::numeric_limits<std::uint64_t>::max();
    std::map<std::string, int> history_levels_;
    CellTemporalPartitionAcceptedState accepted_temporal_partition_;
    std::string accepted_flux_budget_contract_;
    std::string accepted_coupling_contract_;
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
        accepted_face_flux_;
    std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger_;
    std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events_;
    std::uint64_t accepted_state_revision_ = std::numeric_limits<std::uint64_t>::max();
    std::map<ScratchKey, field_type> discarded_scratches_;
    std::string discarded_subcycling_contract_;
  };

  std::unique_ptr<AcceptedProgramContextSnapshot> capture_accepted_context_snapshot_() const {
    if (!active_attempt_states_.empty() || !interface_flux_ledger_ ||
        interface_flux_ledger_->in_transaction())
      throw std::logic_error("AMR Program accepted context snapshot crossed an active attempt");
    return std::make_unique<AcceptedContextSnapshot>(*const_cast<AmrProgramContext*>(this));
  }

  struct FluxBasisFace {
    ::pops::amr::reflux::FaceLedgerRole role = ::pops::amr::reflux::FaceLedgerRole::Coarse;
    int axis = 0;
    Index<Dim> face{};
    Index<Dim> coarse_face{};
    double face_measure = 0.0;
    std::vector<Real> flux_density;
  };

  struct FluxBasis {
    std::uint64_t identity = 0;
    std::size_t runtime_block = 0;
    int level = 0;
    ::pops::amr::Rational stage_fraction{0, 1};
    ::pops::amr::ClockWindow window{};
    std::vector<FluxBasisFace> faces;
  };

  struct FluxExpressionTerm {
    std::shared_ptr<const FluxBasis> basis;
    ExactPolynomial coefficient;
  };

  using FluxExpression = std::map<std::uint64_t, FluxExpressionTerm>;
  using FluxExpressionRegistry = std::map<const field_type*, FluxExpression>;
  using FluxExpressionUpdate = std::optional<FluxExpressionRegistry>;

  struct GeneratedFieldRoute {
    std::string field;
    std::vector<int> program_blocks;

    friend bool operator==(const GeneratedFieldRoute&, const GeneratedFieldRoute&) = default;
  };

  static facade_type* require_facade_(facade_type* facade) {
    if (facade == nullptr)
      throw std::invalid_argument("AmrProgramContext requires a non-null exact-ranked facade");
    return facade;
  }
  static runtime_type* require_runtime_(facade_type& facade) {
    return require_runtime_(facade.engine());
  }
  static runtime_type* require_runtime_(runtime_type* runtime) {
    if (runtime == nullptr)
      throw std::invalid_argument("AmrProgramContext requires a materialized exact-ranked runtime");
    return runtime;
  }
  void require_facade_execution_() const {
    if (facade_ == nullptr)
      throw std::logic_error("AMR Program execution requires its exact-ranked facade");
  }
  void require_prepared_lane_(const ExecutionLane& lane, std::string_view operation) const {
    const ExecutionLane& prepared = prepared_execution_lane();
    if (all_reduce_max(&lane == &prepared ? 0L : 1L, prepared) != 0)
      throw std::invalid_argument(std::string(operation) +
                                  " requires the context's authenticated execution lane");
  }
  void require_block_boundary_session_(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                       int program_block,
                                       const block_boundary_session_type& boundary,
                                       std::string_view operation) const {
    require_boundary_point_(point, operation);
    const ExecutionLane& lane = prepared_execution_lane();
    if (boundary.facade_ != facade_ || boundary.runtime_block_ != sys_block(program_block) ||
        boundary.point_ != point || boundary.lane_ != &lane || !boundary.transport_)
      throw std::invalid_argument(std::string(operation) +
                                  " received a foreign or stale prepared boundary session");
  }
  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument("AMR Program rate identity must be non-negative");
  }

  static void erase_zero_terms_(ExactPolynomial& polynomial) {
    for (auto term = polynomial.begin(); term != polynomial.end();) {
      if (term->second.numerator == 0)
        term = polynomial.erase(term);
      else
        ++term;
    }
  }

  static ExactPolynomial multiply_exact_polynomials_(const ExactPolynomial& left,
                                                     const ExactPolynomial& right) {
    ExactPolynomial result;
    for (const auto& [left_power, left_factor] : left)
      for (const auto& [right_power, right_factor] : right) {
        if (left_power > std::numeric_limits<int>::max() - right_power)
          throw std::overflow_error("AMR Program flux coefficient dt power exceeds int");
        const int power = left_power + right_power;
        const auto found = result.find(power);
        const ::pops::amr::Rational product = left_factor * right_factor;
        if (found == result.end())
          result.emplace(power, product);
        else
          found->second = found->second + product;
      }
    erase_zero_terms_(result);
    return result;
  }

  static void add_exact_polynomial_(ExactPolynomial& destination, const ExactPolynomial& source) {
    for (const auto& [power, factor] : source) {
      const auto found = destination.find(power);
      if (found == destination.end())
        destination.emplace(power, factor);
      else
        found->second = found->second + factor;
    }
    erase_zero_terms_(destination);
  }

  static Real evaluate_exact_polynomial_(const ExactPolynomial& polynomial, Real dt) {
    Real result = Real(0);
    for (const auto& [power, factor] : polynomial) {
      Real dt_power = Real(1);
      for (int exponent = 0; exponent < power; ++exponent)
        dt_power *= dt;
      result += static_cast<Real>(factor.value()) * dt_power;
    }
    return result;
  }

  ExactPolynomial exact_coefficient_unchecked_(
      Real factor, Real reference_dt, std::initializer_list<ExactCoefficientTerm> terms) const {
    if (!std::isfinite(static_cast<double>(factor)) ||
        !std::isfinite(static_cast<double>(reference_dt)) || reference_dt != current_dt_)
      throw std::invalid_argument("AMR Program flux coefficient does not name the active exact dt");
    ExactPolynomial polynomial;
    for (const ExactCoefficientTerm& term : terms) {
      if (term.dt_power < 0 || term.denominator <= 0)
        throw std::invalid_argument("AMR Program flux coefficient metadata is invalid");
      const ::pops::amr::Rational coefficient{term.numerator, term.denominator};
      if (coefficient.numerator != term.numerator || coefficient.denominator != term.denominator)
        throw std::invalid_argument("AMR Program flux coefficient metadata is not canonical");
      const auto found = polynomial.find(term.dt_power);
      if (found == polynomial.end())
        polynomial.emplace(term.dt_power, coefficient);
      else
        found->second = found->second + coefficient;
    }
    erase_zero_terms_(polynomial);
    if (evaluate_exact_polynomial_(polynomial, reference_dt) != factor)
      throw std::invalid_argument(
          "AMR Program numerical coefficient differs from its exact metadata");
    return polynomial;
  }

  static ::pops::amr::Rational exact_binary_rational_(Real value) {
    if (!std::isfinite(static_cast<double>(value)))
      throw std::invalid_argument("AMR Program coefficient is not finite");
    if (value == Real(0))
      return {0, 1};
    int exponent = 0;
    const double fraction = std::frexp(std::abs(static_cast<double>(value)), &exponent);
    constexpr int digits = std::numeric_limits<double>::digits;
    const auto mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, digits));
    const int binary_power = exponent - digits;
    if (binary_power >= 0) {
      if (binary_power >= 63 ||
          mantissa > (static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) >>
                      binary_power))
        throw std::overflow_error("AMR Program coefficient exceeds exact int64 metadata");
      const std::int64_t numerator = static_cast<std::int64_t>(mantissa << binary_power);
      return {value < Real(0) ? -numerator : numerator, 1};
    }
    if (-binary_power >= 63)
      throw std::overflow_error("AMR Program coefficient denominator exceeds exact int64 metadata");
    const std::int64_t numerator = static_cast<std::int64_t>(mantissa);
    const std::int64_t denominator = std::int64_t{1} << (-binary_power);
    return {value < Real(0) ? -numerator : numerator, denominator};
  }

  ExactPolynomial exact_runtime_coefficient_unchecked_(Real factor) const {
    if (active_attempt_states_.empty())
      return {};
    return {{0, exact_binary_rational_(factor)}};
  }

  template <class Build>
  auto prepare_flux_metadata_collectively_(Build&& build, std::string_view failure) const
      -> std::invoke_result_t<Build> {
    using result_type = std::invoke_result_t<Build>;
    if (active_attempt_states_.empty())
      return std::forward<Build>(build)();
    std::optional<result_type> result;
    std::exception_ptr local_error;
    try {
      result.emplace(std::forward<Build>(build)());
    } catch (...) {
      local_error = std::current_exception();
    }
    const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(std::string(failure));
    }
    return std::move(*result);
  }

  ExactPolynomial exact_coefficient_(Real factor, Real reference_dt,
                                     std::initializer_list<ExactCoefficientTerm> terms) const {
    return prepare_flux_metadata_collectively_(
        [&] { return exact_coefficient_unchecked_(factor, reference_dt, terms); },
        "AMR Program exact flux coefficient failed collectively");
  }

  ExactPolynomial exact_runtime_coefficient_(Real factor) const {
    return prepare_flux_metadata_collectively_(
        [&] { return exact_runtime_coefficient_unchecked_(factor); },
        "AMR Program runtime flux coefficient failed collectively");
  }

  ExactPolynomial exact_runtime_axpy_coefficient_(Real factor, const field_type& source) const {
    return prepare_flux_metadata_collectively_(
        [&] {
          if (active_attempt_states_.empty() || active_flux_expression_(source).empty())
            return ExactPolynomial{};
          if (!std::isfinite(current_dt_) || !(current_dt_ > 0.0))
            throw std::logic_error("AMR Program flux axpy lacks its active dt");
          const Real quotient = factor / static_cast<Real>(current_dt_);
          if (quotient * static_cast<Real>(current_dt_) != factor)
            throw std::invalid_argument(
                "AMR Program flux axpy factor has no exact symbolic dt coefficient");
          return ExactPolynomial{{1, exact_binary_rational_(quotient)}};
        },
        "AMR Program runtime flux axpy coefficient failed collectively");
  }

  FluxExpression active_flux_expression_(const field_type& field) const {
    if (active_attempt_states_.empty())
      return {};
    const auto found = active_flux_expressions_.find(&field);
    return found == active_flux_expressions_.end() ? FluxExpression{} : found->second;
  }

  void require_flux_expression_budget_(const FluxExpression& expression) const {
    std::map<std::size_t, std::size_t> bases_by_block;
    for (const auto& [identity, term] : expression) {
      (void)identity;
      if (!term.basis || term.basis->runtime_block >= prepared_rhs_basis_bounds_.size() ||
          term.basis->runtime_block >= prepared_coefficient_term_bounds_.size())
        throw std::logic_error("AMR Program flux expression has a foreign basis identity");
      const std::size_t block = term.basis->runtime_block;
      if (++bases_by_block[block] > prepared_rhs_basis_bounds_[block] ||
          term.coefficient.size() > prepared_coefficient_term_bounds_[block])
        throw std::length_error(
            "AMR Program flux expression exceeds its authenticated artifact budget");
    }
  }

  static FluxExpression scaled_flux_expression_(const FluxExpression& expression,
                                                const ExactPolynomial& coefficient) {
    FluxExpression result;
    if (coefficient.empty())
      return result;
    for (const auto& [identity, term] : expression) {
      ExactPolynomial scaled = multiply_exact_polynomials_(term.coefficient, coefficient);
      if (!scaled.empty())
        result.emplace(identity, FluxExpressionTerm{term.basis, std::move(scaled)});
    }
    return result;
  }

  static void add_flux_expression_(FluxExpression& destination, const FluxExpression& source) {
    for (const auto& [identity, term] : source) {
      const auto found = destination.find(identity);
      if (found == destination.end()) {
        destination.emplace(identity, term);
        continue;
      }
      if (found->second.basis != term.basis)
        throw std::logic_error("AMR Program flux basis identity aliases another evaluation");
      add_exact_polynomial_(found->second.coefficient, term.coefficient);
      if (found->second.coefficient.empty())
        destination.erase(found);
    }
  }

  FluxExpressionUpdate prepare_active_axpy_flux_expression_(
      field_type& destination, const field_type& source, const ExactPolynomial& coefficient) const {
    if (active_attempt_states_.empty())
      return std::nullopt;
    return prepare_flux_metadata_collectively_(
        [&]() {
          FluxExpressionRegistry candidate = active_flux_expressions_;
          FluxExpression result = active_flux_expression_(destination);
          add_flux_expression_(
              result, scaled_flux_expression_(active_flux_expression_(source), coefficient));
          require_flux_expression_budget_(result);
          candidate[&destination] = std::move(result);
          return FluxExpressionUpdate(std::move(candidate));
        },
        "AMR Program flux-expression axpy failed collectively");
  }

  FluxExpressionUpdate prepare_active_lincomb_flux_expression_(
      field_type& destination, const field_type& left, const ExactPolynomial& left_coefficient,
      const field_type& right, const ExactPolynomial& right_coefficient) const {
    if (active_attempt_states_.empty())
      return std::nullopt;
    return prepare_flux_metadata_collectively_(
        [&]() {
          FluxExpressionRegistry candidate = active_flux_expressions_;
          FluxExpression result =
              scaled_flux_expression_(active_flux_expression_(left), left_coefficient);
          add_flux_expression_(
              result, scaled_flux_expression_(active_flux_expression_(right), right_coefficient));
          require_flux_expression_budget_(result);
          candidate[&destination] = std::move(result);
          return FluxExpressionUpdate(std::move(candidate));
        },
        "AMR Program flux-expression linear combination failed collectively");
  }

  void publish_active_flux_expression_update_(FluxExpressionUpdate update) const noexcept {
    if (update)
      active_flux_expressions_.swap(*update);
  }

  void copy_active_flux_expression_(const field_type& source, field_type& destination) const {
    if (active_attempt_states_.empty())
      return;
    FluxExpressionRegistry candidate = prepare_flux_metadata_collectively_(
        [&] {
          FluxExpressionRegistry result = active_flux_expressions_;
          result[&destination] = active_flux_expression_(source);
          return result;
        },
        "AMR Program flux-expression copy failed collectively");
    active_flux_expressions_.swap(candidate);
  }

  void clear_active_flux_expression_(const field_type& field) const noexcept {
    if (!active_attempt_states_.empty())
      active_flux_expressions_.erase(&field);
  }

  template <class Body>
  void advance_prepared_hierarchy_(double dt, Body&& body, std::string_view operation) const {
    if (!std::isfinite(dt) || !(dt > 0.0))
      throw std::invalid_argument("AMR Program step requires a finite positive dt");
    const int prior_level = active_level_;
    const double prior_dt = current_dt_;
    const double prior_interval_start = current_interval_start_time_;
    const ::pops::amr::Rational prior_interval_begin = current_interval_begin_phase_;
    const ::pops::amr::Rational prior_interval_end = current_interval_end_phase_;
    const int prior_substep = logical_substep_;
    const ::pops::amr::Rational prior_stage = stage_time_;
    refresh_resources_();
    require_facade_execution_();
    if (!active_attempt_states_.empty())
      throw std::logic_error("AMR Program hierarchy advance cannot nest another attempt");
    // The prior candidate is only discarded when a later accepted transaction has already
    // captured its own rollback image.  During the transaction that published it, this guard kept
    // the pre-commit ledger alive even after the live ledger crossed its noexcept swap boundary.
    interface_flux_commit_guard_.reset();
    try {
      begin_step(dt);
      prepare_multiblock_subcycling_engine_();
      const ::pops::amr::ClockWindow root{
          {0, static_cast<std::int64_t>(facade_->macro_step()), {0, 1}, facade_->time()},
          {0, static_cast<std::int64_t>(facade_->macro_step()), {1, 1}, facade_->time() + dt}};
      const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
      std::optional<typename interface_flux_ledger_type::PreparedBegin> prepared_begin;
      std::exception_ptr ledger_error;
      try {
        prepared_begin.emplace(interface_flux_ledger_->prepare_begin());
      } catch (...) {
        ledger_error = std::current_exception();
      }
      if (all_reduce_max(ledger_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && ledger_error)
          std::rethrow_exception(ledger_error);
        throw std::runtime_error(
            "AMR Program interface-ledger begin preparation failed collectively");
      }
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("amr-program-interface-ledger-begin"),
                prepared_begin->exact_contract()}},
              lane))
        throw std::runtime_error(
            "AMR Program interface-ledger begin contract differs between MPI ranks");
      interface_flux_ledger_->publish_prepared_begin(*prepared_begin);
      multiblock_subcycling_->advance(
          root,
          [&](multiblock_level_group_type group) { advance_multiblock_level_group_(group, body); },
          [&](multiblock_reflux_context_type& reflux) { reconcile_multiblock_reflux_(reflux); },
          [&](std::size_t runtime_block, std::size_t level, const field_type& candidate) {
            facade_->validate_prepared_amr_state_publication_candidate(
                static_cast<int>(runtime_block), static_cast<int>(level), candidate);
          });
      std::optional<typename interface_flux_ledger_type::PreparedCommit> prepared_commit;
      ledger_error = nullptr;
      try {
        prepared_commit.emplace(interface_flux_ledger_->prepare_commit());
      } catch (...) {
        ledger_error = std::current_exception();
      }
      if (all_reduce_max(ledger_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && ledger_error)
          std::rethrow_exception(ledger_error);
        throw std::runtime_error(
            "AMR Program interface-ledger commit preparation failed collectively");
      }
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("amr-program-interface-ledger-commit"),
                prepared_commit->exact_contract()}},
              lane))
        throw std::runtime_error(
            "AMR Program interface-ledger commit contract differs between MPI ranks");
      static_assert(std::is_nothrow_move_constructible_v<
                    typename interface_flux_ledger_type::PreparedCommit>);
      static_assert(noexcept(interface_flux_commit_guard_.swap(prepared_commit)));
      interface_flux_commit_guard_.swap(prepared_commit);
      interface_flux_ledger_->publish_prepared_commit(*interface_flux_commit_guard_);
    } catch (...) {
      if (interface_flux_ledger_ && interface_flux_ledger_->in_transaction())
        interface_flux_ledger_->rollback();
      active_level_ = prior_level;
      current_dt_ = prior_dt;
      current_interval_start_time_ = prior_interval_start;
      current_interval_begin_phase_ = prior_interval_begin;
      current_interval_end_phase_ = prior_interval_end;
      logical_substep_ = prior_substep;
      stage_time_ = prior_stage;
      clear_active_multiblock_group_();
      throw;
    }
    active_level_ = prior_level;
    current_dt_ = prior_dt;
    current_interval_start_time_ = prior_interval_start;
    current_interval_begin_phase_ = prior_interval_begin;
    current_interval_end_phase_ = prior_interval_end;
    logical_substep_ = prior_substep;
    stage_time_ = prior_stage;
    (void)operation;
  }

  static std::size_t checked_product_(std::size_t left, std::size_t right,
                                      std::string_view operation) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::length_error(std::string(operation) + " exceeds size_t");
    return left * right;
  }

  static ::pops::amr::InterfaceFluxLedgerBudget inactive_interface_flux_budget_() {
    return {0, 0, 1, "pops.amr-program.interface-flux-budget/inactive"};
  }

  void prepare_multiblock_subcycling_engine_() const {
    auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
    std::vector<std::size_t> rhs_basis_bounds;
    std::vector<std::size_t> coefficient_term_bounds;
    std::string program_budget_contract;
    std::exception_ptr local_error;
    try {
      const flux_expression_budget_type& expression_budget =
          facade_->prepared_amr_program_flux_expression_budget();
      const auto& prepared_map = facade_->prepared_amr_program_block_map();
      const auto& program_map = facade_->program_block_map();
      const std::size_t blocks = hierarchy.block_count();
      if (expression_budget.program_hash.empty() ||
          expression_budget.program_hash != facade_->installed_program_hash() ||
          expression_budget.generation != runtime_->materialization_generation() ||
          expression_budget.exact_contract.empty() ||
          expression_budget.program_block_map.canonical_indices != prepared_map.canonical_indices ||
          expression_budget.program_block_map.hierarchy_contract !=
              prepared_map.hierarchy_contract ||
          expression_budget.program_block_map.exact_contract != prepared_map.exact_contract ||
          expression_budget.blocks.size() != blocks || program_map.size() != blocks ||
          prepared_map.canonical_indices.size() != blocks)
        throw std::logic_error(
            "AMR Program flux-expression budget is not authentic for the active carrier");

      rhs_basis_bounds.assign(blocks, 0);
      coefficient_term_bounds.assign(blocks, 0);
      std::vector<bool> seen(blocks, false);
      for (std::size_t program_block = 0; program_block < blocks; ++program_block) {
        const int mapped = program_map[program_block];
        const std::size_t canonical = prepared_map.canonical_indices[program_block];
        if (mapped < 0 || static_cast<std::size_t>(mapped) != canonical || canonical >= blocks ||
            seen[canonical])
          throw std::logic_error(
              "AMR Program flux-expression budget differs from its exact block permutation");
        seen[canonical] = true;
        const auto& block_budget = expression_budget.blocks[program_block];
        const bool active =
            block_budget.rhs_basis_bound != 0 || block_budget.coefficient_term_bound != 0;
        if (active &&
            (block_budget.rhs_basis_bound == 0 || block_budget.coefficient_term_bound == 0))
          throw std::logic_error(
              "AMR Program flux-expression budget has a partial per-block bound");
        rhs_basis_bounds[canonical] = block_budget.rhs_basis_bound;
        coefficient_term_bounds[canonical] = block_budget.coefficient_term_bound;
      }
      program_budget_contract = expression_budget.exact_contract;
    } catch (...) {
      local_error = std::current_exception();
    }
    const ExecutionLane& lane = hierarchy.lane();
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (hierarchy.lane().size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR Program flux-expression budget validation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("amr-program-flux-expression-budget"),
              std::string_view(program_budget_contract)}},
            lane))
      throw std::invalid_argument(
          "AMR Program flux-expression budget differs between prepared-lane ranks");

    std::vector<::pops::amr::ParentChildClockRelation> relations;
    ::pops::amr::InterfaceFluxLedgerBudget interface_budget;
    std::optional<typename interface_flux_ledger_type::PreparedBudget> prepared_interface_budget;
    local_error = nullptr;
    try {
      relations = facade_->prepared_program_temporal_relations();
      if (relations.size() + 1 != hierarchy.level_count())
        throw std::logic_error(
            "AMR Program subcycling lacks one temporal relation per live transition");
      interface_budget = facade_->prepared_amr_interface_flux_ledger_budget();
      prepared_interface_budget.emplace(interface_flux_ledger_->prepare_budget(interface_budget));
      ExactContractBuilder complete;
      complete.text("pops.amr-program.complete-flux-budget")
          .scalar(std::uint32_t{1})
          .bytes(program_budget_contract)
          .bytes(interface_budget.exact_contract);
      program_budget_contract = std::move(complete).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR Program interface budget preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("amr-program-interface-ledger-budget"),
              std::string_view(interface_budget.exact_contract)}},
            lane))
      throw std::invalid_argument(
          "AMR Program interface ledger budgets differ between prepared-lane ranks");
    interface_flux_ledger_->publish_prepared_budget(*prepared_interface_budget);

    if (multiblock_subcycling_ != nullptr &&
        multiblock_subcycling_epoch_ == runtime_->topology_epoch() &&
        multiblock_subcycling_generation_ == runtime_->materialization_generation() &&
        multiblock_subcycling_program_budget_contract_ == program_budget_contract)
      return;

    std::unique_ptr<multiblock_subcycling_type> prepared;
    local_error = nullptr;
    try {
      std::size_t maximum_patches = 1;
      for (std::size_t level = 0; level < hierarchy.level_count(); ++level)
        maximum_patches =
            std::max(maximum_patches,
                     hierarchy.topology_runtime().hierarchy().layout(level).patches().size());
      const std::size_t overlap_pairs = maximum_patches < 2
                                            ? 1
                                            : checked_product_(maximum_patches, maximum_patches - 1,
                                                               "AMR Program patch-overlap budget") /
                                                  2;

      // The ledger implementation requires a positive structural capacity even for an
      // authenticated source-only Program.  Flux-producing Programs derive every retained entry
      // from the exact per-block RHS-basis bound below; no evaluation-count fallback is used.
      std::size_t maximum_entries = 1;
      for (std::size_t parent = 0; parent < relations.size(); ++parent) {
        const auto ratio =
            hierarchy.topology_runtime().hierarchy().layout(parent + 1).ratio_from_parent();
        const auto temporal = relations[parent].temporal_ratio();
        const std::size_t substeps =
            static_cast<std::size_t>(temporal.numerator / temporal.denominator +
                                     (temporal.numerator % temporal.denominator == 0 ? 0 : 1));
        for (std::size_t block = 0; block < hierarchy.block_count(); ++block) {
          std::size_t block_entries = 0;
          for (const ProgramInterfaceFace& interface : program_interface_faces_(parent)) {
            std::size_t fine_faces = 1;
            for (int axis = 0; axis < Dim; ++axis)
              if (axis != interface.axis)
                fine_faces = checked_product_(fine_faces, static_cast<std::size_t>(ratio[axis]),
                                              "AMR Program fine-face budget");
            const std::size_t fine_evaluations =
                checked_product_(substeps, fine_faces, "AMR Program temporal fine-flux budget");
            if (fine_evaluations == std::numeric_limits<std::size_t>::max())
              throw std::length_error("AMR Program face-fragment budget exceeds size_t");
            const std::size_t fragments_per_basis = 1 + fine_evaluations;
            const std::size_t face_entries =
                checked_product_(rhs_basis_bounds[block], fragments_per_basis,
                                 "AMR Program authenticated face-flux expression budget");
            if (block_entries > std::numeric_limits<std::size_t>::max() - face_entries)
              throw std::length_error("AMR Program face-flux ledger budget exceeds size_t");
            block_entries += face_entries;
          }
          maximum_entries = std::max(maximum_entries, block_entries);
        }
      }

      ::pops::numerics::time::amr::MultiBlockAmrSubcyclingBudget budget;
      budget.transitions = {relations.size(), {maximum_patches, overlap_pairs}};
      budget.flux_ledger = {maximum_entries, maximum_entries, 1};
      prepared = std::make_unique<multiblock_subcycling_type>(
          multiblock_subcycling_type::prepare(hierarchy, relations, budget));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (hierarchy.lane().size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR Program subcycling preparation failed collectively");
    }

    multiblock_subcycling_ = std::move(prepared);
    prepared_rhs_basis_bounds_ = std::move(rhs_basis_bounds);
    prepared_coefficient_term_bounds_ = std::move(coefficient_term_bounds);
    multiblock_subcycling_program_budget_contract_ = std::move(program_budget_contract);
    multiblock_subcycling_epoch_ = runtime_->topology_epoch();
    multiblock_subcycling_generation_ = runtime_->materialization_generation();
  }

  template <class Body>
  void advance_multiblock_level_group_(multiblock_level_group_type group, Body& body) const {
    const std::size_t blocks = facade_->prepared_amr_multiblock_hierarchy_().block_count();
    if (group.size() != blocks || group.empty())
      throw std::logic_error("AMR Program level group lost its complete block pack");

    if (prepared_rhs_basis_bounds_.size() != blocks ||
        prepared_coefficient_term_bounds_.size() != blocks)
      throw std::logic_error(
          "AMR Program level group lacks its authenticated flux-expression budget");
    active_attempt_states_.assign(blocks, nullptr);
    active_staged_parents_.assign(blocks, nullptr);
    active_incoming_flux_.assign(blocks, nullptr);
    active_outgoing_flux_.assign(blocks, nullptr);
    active_block_identities_.assign(blocks, {});
    active_flux_basis_counts_.assign(blocks, 0);
    active_flux_expressions_.clear();
    next_active_flux_basis_identity_ = 0;
    for (auto& current : group) {
      if (current.block >= blocks || current.level != group.front().level ||
          current.substep != group.front().substep || current.attempt != group.front().attempt ||
          current.window.begin != group.front().window.begin ||
          current.window.end != group.front().window.end ||
          active_attempt_states_[current.block] != nullptr)
        throw std::logic_error("AMR Program level group is not canonical and simultaneous");
      active_attempt_states_[current.block] = &current.candidate;
      active_staged_parents_[current.block] = current.staged_parent;
      active_incoming_flux_[current.block] = current.incoming_flux;
      active_outgoing_flux_[current.block] = current.outgoing_flux;
      active_block_identities_[current.block] = current.block_identity;
    }
    if (std::find(active_attempt_states_.begin(), active_attempt_states_.end(), nullptr) !=
        active_attempt_states_.end())
      throw std::logic_error("AMR Program level group omits a runtime block candidate");

    active_level_ = static_cast<int>(group.front().level);
    logical_substep_ = group.front().substep;
    active_subcycling_attempt_ = group.front().attempt;
    active_subcycling_window_ = group.front().window;
    current_dt_ = group.front().window.end.physical_time - group.front().window.begin.physical_time;
    current_interval_start_time_ = group.front().window.begin.physical_time;
    current_interval_begin_phase_ = group.front().window.begin.phase;
    current_interval_end_phase_ = group.front().window.end.phase;
    stage_time_ = {0, 1};
    facade_->clear_prepared_amr_level_evaluations();
    try {
      body(current_dt_);
      for (std::size_t block = 0; block < blocks; ++block)
        materialize_active_flux_expression_(block, *active_attempt_states_[block]);
    } catch (...) {
      clear_active_multiblock_group_();
      throw;
    }
    clear_active_multiblock_group_();
  }

  void clear_active_multiblock_group_() const noexcept {
    active_attempt_states_.clear();
    active_staged_parents_.clear();
    active_incoming_flux_.clear();
    active_outgoing_flux_.clear();
    active_block_identities_.clear();
    active_flux_basis_counts_.clear();
    active_flux_expressions_.clear();
    next_active_flux_basis_identity_ = 0;
    active_subcycling_attempt_ = 0;
  }

  struct ProgramInterfaceFace {
    int axis = 0;
    Index<Dim> coarse_face{};
    Index<Dim> coarse_cell{};
    ::pops::amr::reflux::CoarseCellFaceSide side = ::pops::amr::reflux::CoarseCellFaceSide::Lower;
  };

  static std::array<int, Dim> index_key_(const Index<Dim>& index) {
    std::array<int, Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[static_cast<std::size_t>(axis)] = index[axis];
    return result;
  }

  static std::vector<Index<Dim>> cells_in_box_(const Box<Dim>& box) {
    const std::size_t cells = static_cast<std::size_t>(box.numPts());
    std::vector<Index<Dim>> result;
    result.reserve(cells);
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      std::size_t remainder = ordinal;
      Index<Dim> cell{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t length = static_cast<std::size_t>(box.length(axis));
        cell[axis] = box.lo[axis] + static_cast<int>(remainder % length);
        remainder /= length;
      }
      result.push_back(cell);
    }
    return result;
  }

  std::vector<ProgramInterfaceFace> program_interface_faces_(std::size_t parent_level) const {
    const auto& parent = runtime_->hierarchy().layout(parent_level);
    const auto& child = runtime_->hierarchy().layout(parent_level + 1);
    Extent<Dim> ratio{};
    for (int axis = 0; axis < Dim; ++axis)
      ratio[axis] = child.ratio_from_parent()[axis];
    std::set<std::array<int, Dim>> covered;
    for (const Box<Dim>& fine_patch : child.patches().boxes())
      for (const Index<Dim>& cell : cells_in_box_(pops::coarsen(fine_patch, ratio)))
        covered.insert(index_key_(cell));

    std::vector<ProgramInterfaceFace> result;
    for (const auto& coordinate : covered) {
      Index<Dim> inside{};
      for (int axis = 0; axis < Dim; ++axis)
        inside[axis] = coordinate[static_cast<std::size_t>(axis)];
      for (int axis = 0; axis < Dim; ++axis) {
        for (int direction : {-1, 1}) {
          Index<Dim> outside = inside;
          outside[axis] += direction;
          const bool parent_cell =
              std::any_of(parent.patches().boxes().begin(), parent.patches().boxes().end(),
                          [&](const Box<Dim>& patch) { return patch.contains(outside); });
          if (!parent.domain().contains(outside) || !parent_cell ||
              covered.contains(index_key_(outside)))
            continue;
          Index<Dim> face = inside;
          if (direction > 0)
            ++face[axis];
          result.push_back({axis, face, outside,
                            direction > 0 ? ::pops::amr::reflux::CoarseCellFaceSide::Lower
                                          : ::pops::amr::reflux::CoarseCellFaceSide::Upper});
        }
      }
    }
    return result;
  }

  std::vector<Real> collective_face_payload_(const level_evaluation_type& evaluation,
                                             const field_type& field, int axis,
                                             const Index<Dim>& face) const {
    std::vector<Real> payload;
    std::exception_ptr local_error;
    try {
      std::size_t selected = field.layout().size();
      for (std::size_t global = 0; global < field.layout().size(); ++global)
        if (nd::face_box(field.layout()[global], axis).contains(face)) {
          selected = global;
          break;
        }
      if (selected == field.layout().size())
        throw std::out_of_range("AMR Program interface face has no level flux patch");
      const Index<Dim> owner = field.distribution().replicated()
                                   ? field.rank_space().coordinate(0)
                                   : field.distribution().owner(selected);
      payload.assign(static_cast<std::size_t>(field.ncomp()), Real(0));
      if (owner == field.local_rank()) {
        const std::size_t local = field.local_index_of(selected);
        if (local == field_type::not_local || local >= evaluation.integrated_face_fluxes.size())
          throw std::runtime_error("AMR Program interface flux storage lost its local patch");
        const FieldView<const Real, Dim> values =
            evaluation.integrated_face_fluxes[local].view().axes[axis];
        for (int component = 0; component < field.ncomp(); ++component) {
          Real value = Real(0);
          Kokkos::parallel_reduce(
              "pops_program_amr_interface_face", Kokkos::RangePolicy<>(0, 1),
              [=] POPS_HD(int, Real& sum) { sum += values(face, component); }, value);
          payload[static_cast<std::size_t>(component)] = value;
        }
        Kokkos::fence();
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    const ExecutionLane& lane = prepared_execution_lane();
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR Program interface flux failed on another MPI rank");
    }
    for (Real& value : payload)
      value = all_reduce_sum(value, lane);
    return payload;
  }

  static void payload_axpy_(std::vector<Real>& destination, double factor,
                            const std::vector<Real>& source) {
    if (destination.empty())
      destination.assign(source.size(), Real(0));
    if (destination.size() != source.size())
      throw std::invalid_argument("AMR Program reflux payload component counts differ");
    for (std::size_t component = 0; component < source.size(); ++component)
      destination[component] += static_cast<Real>(factor) * source[component];
  }

  static double face_measure_(const Geometry<Dim>& geometry, int normal_axis) {
    double result = 1.0;
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != normal_axis)
        result *= static_cast<double>(geometry.spacing(axis));
    return result;
  }

  static double cell_measure_(const Geometry<Dim>& geometry) {
    double result = 1.0;
    for (int axis = 0; axis < Dim; ++axis)
      result *= static_cast<double>(geometry.spacing(axis));
    return result;
  }

  void apply_reflux_payload_(field_type& coarse, const Index<Dim>& cell,
                             const std::vector<Real>& correction) const {
    for (std::size_t local = 0; local < coarse.local_size(); ++local) {
      if (!coarse.box(local).contains(cell))
        continue;
      const FieldView<Real, Dim> values = coarse.fab(local).view();
      for (int component = 0; component < static_cast<int>(correction.size()); ++component) {
        const Real increment = correction[static_cast<std::size_t>(component)];
        for_each_cell(Box<Dim>{cell, cell}, [=] POPS_HD(const Index<Dim>& index) {
          values(index, component) += increment;
        });
      }
      return;
    }
  }

  void attach_active_flux_basis_(int runtime_block, const level_evaluation_type& evaluation,
                                 field_type& rhs) const {
    if (active_attempt_states_.empty())
      return;
    if (runtime_block < 0 ||
        static_cast<std::size_t>(runtime_block) >= active_attempt_states_.size() ||
        active_attempt_states_[static_cast<std::size_t>(runtime_block)] == nullptr)
      throw std::logic_error("AMR Program flux evaluation has no active block candidate");

    using fragment_role_type = ::pops::amr::reflux::FaceLedgerRole;
    struct PendingFace {
      fragment_role_type role = fragment_role_type::Coarse;
      int axis = 0;
      Index<Dim> face{};
      Index<Dim> coarse_face{};
      double measure = 0.0;
    };

    const std::size_t block = static_cast<std::size_t>(runtime_block);
    std::vector<PendingFace> pending;
    std::exception_ptr preparation_error;
    try {
      require_same_field_contract_(evaluation.residual, rhs,
                                   "AMR Program flux-expression RHS basis");
      if (evaluation.point.level != active_level_ || evaluation.point.substep != logical_substep_ ||
          evaluation.point.stage_fraction != stage_time_ || evaluation.point.dt != current_dt_ ||
          evaluation.topology_epoch != runtime_->topology_epoch() ||
          evaluation.materialization_generation != runtime_->materialization_generation() ||
          evaluation.integrated_face_fluxes.size() != rhs.local_size() ||
          active_block_identities_[block].empty() ||
          !(current_interval_begin_phase_ < current_interval_end_phase_) ||
          !std::isfinite(current_interval_start_time_) || !std::isfinite(current_dt_) ||
          !(current_dt_ > 0.0))
        throw std::logic_error(
            "AMR Program flux-expression basis differs from its active evaluation interval");
      if (block >= prepared_rhs_basis_bounds_.size() ||
          block >= prepared_coefficient_term_bounds_.size() ||
          prepared_rhs_basis_bounds_[block] == 0 || prepared_coefficient_term_bounds_[block] == 0)
        throw std::logic_error(
            "flux-producing AMR Program block has no authenticated expression budget");
      if (active_flux_basis_counts_.at(block) >= prepared_rhs_basis_bounds_[block])
        throw std::length_error(
            "AMR Program flux evaluations exceed their authenticated RHS-basis bound");
      if (next_active_flux_basis_identity_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("AMR Program flux basis identity exhausted uint64_t");

      if (active_outgoing_flux_[block] != nullptr) {
        const Geometry<Dim> geometry = facade_->prepared_amr_level_geometry(active_level_);
        for (const ProgramInterfaceFace& interface :
             program_interface_faces_(static_cast<std::size_t>(active_level_)))
          pending.push_back({fragment_role_type::Coarse, interface.axis, interface.coarse_face,
                             interface.coarse_face, face_measure_(geometry, interface.axis)});
      }
      if (active_incoming_flux_[block] != nullptr) {
        const std::size_t parent = static_cast<std::size_t>(active_level_ - 1);
        const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
        const auto ratio =
            hierarchy.topology_runtime().hierarchy().layout(parent + 1).ratio_from_parent();
        const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
            hierarchy.topology_runtime().hierarchy().layout(parent).domain().lo,
            hierarchy.topology_runtime().hierarchy().layout(parent + 1).domain().lo};
        const Geometry<Dim> geometry = facade_->prepared_amr_level_geometry(active_level_);
        for (const ProgramInterfaceFace& interface : program_interface_faces_(parent)) {
          ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
          query.owner = std::string(active_block_identities_[block]);
          query.state = query.owner + "/state";
          query.levels = {static_cast<int>(parent), static_cast<int>(parent + 1)};
          query.axis = interface.axis;
          query.coarse_face = interface.coarse_face;
          query.attempt = active_subcycling_attempt_;
          query.macro_step = evaluation.point.tick;
          query.window_begin = current_interval_begin_phase_;
          query.window_end = current_interval_end_phase_;
          std::size_t fine_count = 1;
          for (int axis = 0; axis < Dim; ++axis)
            if (axis != interface.axis)
              fine_count = checked_product_(fine_count, static_cast<std::size_t>(ratio[axis]),
                                            "AMR Program fine-face enumeration");
          const ::pops::amr::reflux::MetricRefluxBudget budget{fine_count, fine_count, 1};
          for (const Index<Dim>& fine_face :
               ::pops::amr::reflux::fine_faces_for_coarse_face(query, ratio, mapping, budget))
            pending.push_back({fragment_role_type::Fine, interface.axis, fine_face,
                               interface.coarse_face, face_measure_(geometry, interface.axis)});
        }
      }
    } catch (...) {
      preparation_error = std::current_exception();
    }
    const ExecutionLane& lane = prepared_execution_lane();
    if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
      if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && preparation_error)
        std::rethrow_exception(preparation_error);
      throw std::runtime_error("AMR Program face-flux preparation failed collectively");
    }

    std::vector<FluxBasisFace> faces;
    preparation_error = nullptr;
    try {
      faces.reserve(pending.size());
    } catch (...) {
      preparation_error = std::current_exception();
    }
    if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
      if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && preparation_error)
        std::rethrow_exception(preparation_error);
      throw std::runtime_error("AMR Program face-flux basis reservation failed collectively");
    }

    for (const PendingFace& face : pending) {
      std::vector<Real> payload = collective_face_payload_(evaluation, rhs, face.axis, face.face);
      std::optional<FluxBasisFace> prepared_face;
      std::exception_ptr payload_error;
      try {
        if (!(face.measure > 0.0) || !std::isfinite(face.measure))
          throw std::invalid_argument("AMR Program flux basis has an invalid face measure");
        for (Real& component : payload) {
          if (!std::isfinite(static_cast<double>(component)))
            throw std::invalid_argument("AMR Program flux basis contains a non-finite payload");
          // Generated Cartesian operators retain face-integrated fluxes.  The metric ledger
          // consumes a density and multiplies its authenticated face measure exactly once.
          component /= static_cast<Real>(face.measure);
        }
        prepared_face.emplace(FluxBasisFace{face.role, face.axis, face.face, face.coarse_face,
                                            face.measure, std::move(payload)});
      } catch (...) {
        payload_error = std::current_exception();
      }
      if (all_reduce_max(payload_error ? 1L : 0L, lane) != 0) {
        if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && payload_error)
          std::rethrow_exception(payload_error);
        throw std::runtime_error("AMR Program face-flux basis failed collectively");
      }
      faces.push_back(std::move(*prepared_face));
    }

    const std::uint64_t identity = next_active_flux_basis_identity_ + 1;
    FluxExpressionRegistry expression_candidate;
    std::exception_ptr expression_error;
    try {
      const ::pops::amr::ClockWindow interval{
          {active_level_, evaluation.point.tick, current_interval_begin_phase_,
           current_interval_start_time_},
          {active_level_, evaluation.point.tick, current_interval_end_phase_,
           current_interval_start_time_ + current_dt_}};
      auto basis = std::make_shared<const FluxBasis>(FluxBasis{identity, block, active_level_,
                                                               evaluation.point.stage_fraction,
                                                               interval, std::move(faces)});
      FluxExpression expression;
      expression.emplace(identity, FluxExpressionTerm{std::move(basis), {{0, {1, 1}}}});
      require_flux_expression_budget_(expression);
      expression_candidate = active_flux_expressions_;
      expression_candidate[&rhs] = std::move(expression);
    } catch (...) {
      expression_error = std::current_exception();
    }
    if (all_reduce_max(expression_error ? 1L : 0L, lane) != 0) {
      if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && expression_error)
        std::rethrow_exception(expression_error);
      throw std::runtime_error("AMR Program flux-expression attachment failed collectively");
    }
    active_flux_expressions_.swap(expression_candidate);
    ++active_flux_basis_counts_[block];
    next_active_flux_basis_identity_ = identity;
  }

  void materialize_active_flux_expression_(std::size_t runtime_block,
                                           const field_type& candidate) const {
    using fragment_key_type = ::pops::amr::reflux::FaceFluxFragmentKey<Dim>;
    using fragment_role_type = ::pops::amr::reflux::FaceLedgerRole;
    if (runtime_block >= active_attempt_states_.size() ||
        runtime_block >= active_incoming_flux_.size() ||
        runtime_block >= active_outgoing_flux_.size() ||
        runtime_block >= active_block_identities_.size() ||
        active_attempt_states_[runtime_block] != &candidate ||
        active_block_identities_[runtime_block].empty())
      throw std::logic_error("AMR Program final flux expression has no canonical block candidate");

    const FluxExpression expression = active_flux_expression_(candidate);
    require_flux_expression_budget_(expression);
    for (const auto& [identity, term] : expression) {
      if (!term.basis || term.basis->identity != identity ||
          term.basis->runtime_block != runtime_block || term.basis->level != active_level_ ||
          term.coefficient.size() != 1 || term.coefficient.begin()->first != 1)
        throw std::invalid_argument(
            "AMR Program final flux coefficient is not a supported exact dt integral");
      const ::pops::amr::Rational weight = term.coefficient.begin()->second;
      if (weight.denominator <= 0 ||
          ::pops::amr::Rational{weight.numerator, weight.denominator} != weight)
        throw std::invalid_argument(
            "AMR Program final flux coefficient lost its canonical rational metadata");
      const FluxBasis& basis = *term.basis;
      if (basis.window.begin.level != active_level_ || basis.window.end.level != active_level_ ||
          basis.window.begin.macro_step != active_subcycling_window_.begin.macro_step ||
          basis.window.end.macro_step != active_subcycling_window_.end.macro_step ||
          basis.window.begin.phase < active_subcycling_window_.begin.phase ||
          active_subcycling_window_.end.phase < basis.window.end.phase ||
          !(basis.window.begin.phase < basis.window.end.phase) ||
          basis.stage_fraction.denominator <= 0 || basis.stage_fraction.numerator < 0 ||
          basis.stage_fraction.numerator > basis.stage_fraction.denominator)
        throw std::invalid_argument(
            "AMR Program flux basis lies outside its canonical level/substep window");
      const double duration = basis.window.end.physical_time - basis.window.begin.physical_time;
      if (!std::isfinite(duration) || !(duration > 0.0))
        throw std::invalid_argument("AMR Program flux basis has an invalid physical duration");
      const ::pops::amr::Rational stage_phase =
          basis.window.begin.phase +
          (basis.window.end.phase - basis.window.begin.phase) * basis.stage_fraction;
      const double stage_physical_time =
          basis.window.begin.physical_time + basis.stage_fraction.value() * duration;
      const std::string stage = "pops.program-flux-expression.v1/basis/" +
                                std::to_string(identity) + "/dt-power/1/weight/" +
                                std::to_string(weight.numerator) + "/" +
                                std::to_string(weight.denominator) + "/stage/" +
                                std::to_string(basis.stage_fraction.numerator) + "/" +
                                std::to_string(basis.stage_fraction.denominator);

      for (const FluxBasisFace& face : basis.faces) {
        multiblock_flux_ledger_type* ledger = face.role == fragment_role_type::Coarse
                                                  ? active_outgoing_flux_[runtime_block]
                                                  : active_incoming_flux_[runtime_block];
        if (ledger == nullptr)
          throw std::logic_error(
              "AMR Program flux basis targets no active hierarchy-transition ledger");
        fragment_key_type key;
        key.owner = std::string(active_block_identities_[runtime_block]);
        key.state = key.owner + "/state";
        key.levels = face.role == fragment_role_type::Coarse
                         ? ::pops::amr::reflux::LevelTransition{active_level_, active_level_ + 1}
                         : ::pops::amr::reflux::LevelTransition{active_level_ - 1, active_level_};
        key.axis = face.axis;
        key.face = face.face;
        key.coarse_face = face.coarse_face;
        key.clock = {active_level_, basis.window.begin.macro_step, stage_phase,
                     stage_physical_time};
        key.stage = stage;
        key.attempt = active_subcycling_attempt_;
        key.role = face.role;
        ledger->accumulate(
            std::move(key),
            {weight, basis.window.begin.phase, basis.window.end.phase, duration, face.face_measure},
            face.flux_density);
      }
    }
  }

  void reconcile_multiblock_reflux_(multiblock_reflux_context_type& context) const {
    if (context.flux.published_size() == 0)
      return;
    const Geometry<Dim> geometry =
        facade_->prepared_amr_level_geometry(static_cast<int>(context.parent_level));
    std::size_t maximum_fine_faces = 1;
    for (int axis = 0; axis < Dim; ++axis)
      maximum_fine_faces = checked_product_(maximum_fine_faces,
                                            static_cast<std::size_t>(context.spatial_ratio[axis]),
                                            "AMR Program reflux fine-face budget");
    const ::pops::amr::reflux::MetricRefluxBudget budget{
        maximum_fine_faces, std::max<std::size_t>(context.flux.published_size(), 1),
        std::max<std::size_t>(context.flux.published_size(), 1)};
    const std::string state_identity = std::string(context.block_identity) + "/state";
    for (const ProgramInterfaceFace& interface : program_interface_faces_(context.parent_level)) {
      ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
      query.owner = std::string(context.block_identity);
      query.state = state_identity;
      query.levels = {static_cast<int>(context.parent_level),
                      static_cast<int>(context.parent_level + 1)};
      query.axis = interface.axis;
      query.coarse_face = interface.coarse_face;
      query.attempt = context.attempt;
      query.macro_step = context.parent_window.begin.macro_step;
      query.window_begin = context.parent_window.begin.phase;
      query.window_end = context.parent_window.end.phase;
      bool found_coarse = false;
      for (const auto& entry : context.flux.published_entries(interface.axis))
        found_coarse =
            found_coarse ||
            (entry.key.owner == query.owner && entry.key.state == query.state &&
             entry.key.levels == query.levels && entry.key.axis == query.axis &&
             entry.key.coarse_face == query.coarse_face && entry.key.attempt == query.attempt &&
             entry.key.role == ::pops::amr::reflux::FaceLedgerRole::Coarse);
      if (!found_coarse)
        throw std::runtime_error(
            "AMR Program reflux ledger lacks its block-qualified coarse face: owner=" +
            query.owner + " axis=" + std::to_string(query.axis) +
            " published=" + std::to_string(context.flux.published_size()));
      const auto reflux = facade_->prepared_amr_multiblock_hierarchy_()
                              .topology_runtime()
                              .reconcile_reflux_for_owner(
                                  context.flux, query, context.block_identity, state_identity,
                                  context.face_mapping, budget, payload_axpy_);
      const std::vector<Real> correction = ::pops::amr::reflux::coarse_cell_reflux_correction(
          reflux, cell_measure_(geometry), interface.side, payload_axpy_);
      apply_reflux_payload_(context.parent, interface.coarse_cell, correction);
    }
  }

  static std::optional<std::pair<int, std::string>> decode_history_key_(std::string_view key) {
    constexpr std::string_view prefix = "pops.amr.level-history.v1/";
    if (!key.starts_with(prefix))
      return std::nullopt;
    key.remove_prefix(prefix.size());
    const std::size_t slash = key.find('/');
    const std::size_t colon = key.find(':', slash == std::string_view::npos ? 0 : slash);
    if (slash == std::string_view::npos || colon == std::string_view::npos)
      throw std::invalid_argument("AMR Program history storage key is malformed");
    std::size_t consumed = 0;
    const int level = std::stoi(std::string(key.substr(0, slash)), &consumed);
    if (level < 0 || consumed != slash)
      throw std::invalid_argument("AMR Program history storage key has an invalid level");
    const std::string length_text(key.substr(slash + 1, colon - slash - 1));
    consumed = 0;
    const std::size_t length = std::stoull(length_text, &consumed);
    const std::string name(key.substr(colon + 1));
    if (consumed != length_text.size() || name.empty() || name.size() != length)
      throw std::invalid_argument("AMR Program history storage key has an invalid name");
    return std::pair<int, std::string>{level, name};
  }

  AmrProgramAcceptedState<Dim> accepted_state_() const {
    require_facade_execution_();
    AmrProgramAcceptedState<Dim> state;
    state.spatial_contract = runtime_->spatial_contract();
    state.topology_epoch = runtime_->topology_epoch();
    state.materialization_generation = runtime_->materialization_generation();
    state.level_clocks.reserve(runtime_->hierarchy().num_levels());
    for (std::size_t level = 0; level < runtime_->hierarchy().num_levels(); ++level)
      state.level_clocks.push_back(
          {static_cast<int>(level), facade_->macro_step(), {0, 1}, facade_->time()});
    state.logical_clock_ticks =
        clock_schedule_.accepted_ticks(static_cast<std::int64_t>(facade_->macro_step()));
    state.temporal_partition = accepted_temporal_partition_;

    const auto& manager = runtime_state().hist_;
    struct AccumulatedHistory {
      AmrProgramHistoryDescriptor descriptor;
      std::set<int> levels;
    };
    std::map<std::string, AccumulatedHistory> histories;
    for (const auto& [key, ring] : manager.histories) {
      const auto decoded = decode_history_key_(key);
      if (!decoded || ring.empty())
        throw std::runtime_error("AMR Program accepted history registry is malformed");
      const auto& [level, name] = *decoded;
      const int runtime_owner = manager.owner.at(key);
      int program_owner = -1;
      const auto& block_map = facade_->program_block_map();
      for (std::size_t program = 0; program < block_map.size(); ++program)
        if (block_map[program] == runtime_owner) {
          program_owner = static_cast<int>(program);
          break;
        }
      if (program_owner < 0)
        throw std::runtime_error("AMR Program history lost its authenticated block owner");
      AmrProgramHistoryDescriptor descriptor{name,
                                             program_owner,
                                             manager.state_identity.at(key),
                                             manager.space_identity.at(key),
                                             manager.clock_identity.at(key),
                                             manager.interpolation_identity.at(key),
                                             manager.depth.at(key),
                                             ring.front().ncomp()};
      auto [entry, inserted] =
          histories.try_emplace(name, AccumulatedHistory{descriptor, std::set<int>{level}});
      if (!inserted) {
        const auto& retained = entry->second.descriptor;
        if (retained.program_owner != descriptor.program_owner ||
            retained.state_identity != descriptor.state_identity ||
            retained.space_identity != descriptor.space_identity ||
            retained.clock_identity != descriptor.clock_identity ||
            retained.interpolation_identity != descriptor.interpolation_identity ||
            retained.depth != descriptor.depth || retained.components != descriptor.components ||
            !entry->second.levels.insert(level).second)
          throw std::runtime_error("AMR Program history differs between active levels");
      }
      const auto& dts = manager.slot_dt.at(key);
      if (dts.size() != ring.size())
        throw std::runtime_error("AMR Program history dt provenance has the wrong depth");
      for (std::size_t slot = 0; slot < ring.size(); ++slot)
        state.history_slots.push_back({name, level, static_cast<int>(slot),
                                       static_cast<double>(dts[slot]), manager.initialized.at(key),
                                       manager.fill_count.at(key)});
    }
    for (auto& [name, accumulated] : histories) {
      (void)name;
      if (accumulated.levels.size() != runtime_->hierarchy().num_levels())
        throw std::runtime_error("AMR Program history omits an active hierarchy level");
      state.histories.push_back(std::move(accumulated.descriptor));
    }
    std::sort(state.history_slots.begin(), state.history_slots.end(),
              [](const auto& left, const auto& right) {
                return std::tie(left.name, left.level, left.slot) <
                       std::tie(right.name, right.level, right.slot);
              });

    if (multiblock_subcycling_) {
      state.flux_budget_contract = multiblock_subcycling_program_budget_contract_;
      state.coupling_contract =
          std::string(facade_->prepared_amr_multiblock_hierarchy_().collective_contract());
      const std::string_view interface_contract =
          facade_->prepared_amr_multiblock_hierarchy_().interface_flux_provider_contract();
      if (!interface_contract.empty()) {
        ExactContractBuilder accepted_coupling;
        accepted_coupling.text("pops.amr-program.accepted-coupling")
            .scalar(std::uint32_t{1})
            .bytes(state.coupling_contract)
            .bytes(interface_contract);
        state.coupling_contract = std::move(accepted_coupling).release();
      }
      if (interface_flux_ledger_) {
        ExactContractBuilder budgeted_coupling;
        budgeted_coupling.text("pops.amr-program.accepted-budgeted-coupling")
            .scalar(std::uint32_t{1})
            .bytes(state.coupling_contract)
            .bytes(interface_flux_ledger_->budget().exact_contract)
            .scalar(static_cast<std::uint64_t>(
                interface_flux_ledger_->budget().max_fragments_per_window))
            .scalar(static_cast<std::uint64_t>(
                interface_flux_ledger_->budget().max_payload_terms_per_window))
            .scalar(
                static_cast<std::uint64_t>(interface_flux_ledger_->budget().max_transaction_depth));
        state.coupling_contract = std::move(budgeted_coupling).release();
      }
      const auto seal_contract = [](std::string_view prefix, std::string_view contract) {
        const auto* begin = reinterpret_cast<const std::uint8_t*>(contract.data());
        std::vector<std::uint8_t> bytes(begin, begin + contract.size());
        return std::string(prefix) + identity::sha256_hex(bytes);
      };
      state.flux_budget_contract = seal_contract("pops.amr-program.complete-flux-budget.v1:sha256:",
                                                 state.flux_budget_contract);
      state.coupling_contract =
          seal_contract("pops.amr-program.accepted-coupling.v1:sha256:", state.coupling_contract);
      for (std::size_t block = 0;
           block < facade_->prepared_amr_multiblock_hierarchy_().block_count(); ++block) {
        for (std::size_t parent = 0; parent + 1 < runtime_->hierarchy().num_levels(); ++parent) {
          for (const auto& ledger : multiblock_subcycling_->ledgers(block, parent))
            for (int axis = 0; axis < Dim; ++axis) {
              const auto& entries = ledger.published_entries(axis);
              auto& flux = state.accepted_face_flux[static_cast<std::size_t>(axis)];
              flux.insert(flux.end(), entries.begin(), entries.end());
            }
          const ::pops::amr::ClockStamp clock{
              static_cast<int>(parent), facade_->macro_step(), {0, 1}, facade_->time()};
          state.synchronization_events.push_back({static_cast<int>(parent),
                                                  static_cast<int>(parent + 1),
                                                  static_cast<int>(block), "reflux", clock});
          state.synchronization_events.push_back({static_cast<int>(parent),
                                                  static_cast<int>(parent + 1),
                                                  static_cast<int>(block), "average_down", clock});
        }
      }
      for (int axis = 0; axis < Dim; ++axis) {
        auto by_key = [](const auto& left, const auto& right) { return left.key < right.key; };
        std::sort(state.accepted_face_flux[static_cast<std::size_t>(axis)].begin(),
                  state.accepted_face_flux[static_cast<std::size_t>(axis)].end(), by_key);
      }
    } else {
      state.flux_budget_contract = accepted_flux_budget_contract_;
      state.coupling_contract = accepted_coupling_contract_;
      state.accepted_face_flux = accepted_face_flux_;
      state.synchronization_events = accepted_synchronization_events_;
    }
    if (interface_flux_ledger_) {
      if (interface_flux_ledger_->in_transaction())
        throw std::logic_error(
            "AMR Program checkpoint cannot observe an active interface-flux transaction");
      state.accepted_interface_flux = interface_flux_ledger_->published_entries();
      std::sort(state.accepted_interface_flux.begin(), state.accepted_interface_flux.end(),
                [](const auto& left, const auto& right) { return left.key < right.key; });
    }
    return state;
  }

  void import_accepted_state_(bool force) const {
    require_facade_execution_();
    const std::uint64_t revision = facade_->program_accepted_state_revision();
    if (!force && revision == accepted_state_revision_)
      return;
    prepare_multiblock_subcycling_engine_();
    const std::vector<std::uint8_t> bytes = facade_->program_accepted_state();
    if (bytes.empty())
      throw std::runtime_error("AMR Program accepted-state import received an empty image");
    AmrProgramAcceptedState<Dim> state =
        deserialize_amr_program_accepted_state<Dim>(bytes, &interface_flux_ledger_->budget());
    require_live_amr_program_checkpoint(state, *runtime_);
    const auto expected = accepted_state_();
    if (state.histories != expected.histories || state.history_slots != expected.history_slots)
      throw std::runtime_error(
          "AMR Program accepted-state history provenance differs from live restored rings");
    if (!expected.flux_budget_contract.empty() &&
        state.flux_budget_contract != expected.flux_budget_contract)
      throw std::runtime_error("AMR Program accepted-state flux budget is no longer authentic");
    if (!expected.coupling_contract.empty() &&
        state.coupling_contract != expected.coupling_contract)
      throw std::runtime_error(
          "AMR Program accepted-state coupling contract is no longer authentic");
    if (state.level_clocks.empty())
      throw std::runtime_error("AMR Program accepted-state import lacks its level clocks");
    const std::int64_t accepted_macro_step = state.level_clocks.front().macro_step;
    for (const auto& clock : state.level_clocks)
      if (clock.macro_step != accepted_macro_step)
        throw std::runtime_error(
            "AMR Program accepted-state levels disagree on their accepted macro step");
    clock_schedule_.restore_accepted_ticks(state.logical_clock_ticks, accepted_macro_step);
    accepted_temporal_partition_ = std::move(state.temporal_partition);
    accepted_flux_budget_contract_ = std::move(state.flux_budget_contract);
    accepted_coupling_contract_ = std::move(state.coupling_contract);
    accepted_face_flux_ = std::move(state.accepted_face_flux);
    interface_flux_commit_guard_.reset();
    auto interface_budget = interface_flux_ledger_->budget();
    interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
        restore_amr_program_interface_flux_ledger(state, std::move(interface_budget)));
    accepted_synchronization_events_ = std::move(state.synchronization_events);
    accepted_state_revision_ = revision;
  }

  void refresh_accepted_hierarchy_state_() const {
    require_facade_execution_();
    if (!active_attempt_states_.empty())
      throw std::logic_error("AMR Program accepted-state refresh crossed an active attempt");
    synchronize_resource_generation_();
    const auto state = accepted_state_();
    facade_->restore_program_accepted_state(serialize_amr_program_accepted_state(state));
    accepted_state_revision_ = facade_->program_accepted_state_revision();
    accepted_temporal_partition_ = state.temporal_partition;
    accepted_flux_budget_contract_ = state.flux_budget_contract;
    accepted_coupling_contract_ = state.coupling_contract;
    accepted_face_flux_ = state.accepted_face_flux;
    accepted_synchronization_events_ = state.synchronization_events;
  }

  void preflight_restart_regrid_() const {
    if (!active_attempt_states_.empty())
      throw std::logic_error("AMR RegridOnRestart requires an accepted Program boundary");
    import_accepted_state_(true);
    require_regrid_rematerializable_temporal_partition(accepted_temporal_partition_);
  }

  void restart_regrid_() const {
    preflight_restart_regrid_();
    begin_restart_regrid_history_sequence();
    try {
      for (int parent = 0; parent + 1 < facade_->configured_n_levels(); ++parent) {
        (void)facade_->execute_prepared_tagging(parent);
        if (!facade_->regrid_from_prepared_tagging(parent))
          break;
      }
      end_restart_regrid_history_sequence();
    } catch (...) {
      end_restart_regrid_history_sequence();
      throw;
    }
    multiblock_subcycling_.reset();
    accepted_face_flux_ = {};
    interface_flux_commit_guard_.reset();
    interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
        runtime_->topology_epoch(), inactive_interface_flux_budget_());
    accepted_synchronization_events_.clear();
    if (accepted_temporal_partition_.kind == TemporalPartitionKind::CellLocal)
      accepted_temporal_partition_.topology_epoch = runtime_->topology_epoch();
    refresh_accepted_hierarchy_state_();
  }

  void resync_after_restart_() const {
    if (!active_attempt_states_.empty())
      throw std::logic_error("AMR restart resynchronization crossed an active Program attempt");
    multiblock_subcycling_.reset();
    synchronize_resource_generation_();
    import_accepted_state_(true);
  }

  void require_history_free_for_topology_change_(std::string_view operation) const {
    if (!history_levels_.empty())
      throw std::runtime_error(
          "AmrProgramContext cannot " + std::string(operation) +
          " while exact-ranked history rings lack a prepared rematerialization transaction");
  }
  [[noreturn]] static void unavailable_(std::string_view provider) {
    throw std::runtime_error("AmrProgramContext has no prepared " + std::string(provider));
  }
  [[noreturn]] static void deferred_op(std::string_view id, std::string_view detail) {
    throw std::runtime_error("AmrProgramContext deferred capability '" + std::string(id) +
                             "': " + std::string(detail));
  }

  PhysicalBoundaryConditions<Dim> hierarchy_tensor_boundary_(const Geometry<Dim>& geometry) const {
    const BoundaryTopology<Dim> topology = facade_->prepared_amr_boundary_topology();
    std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
    RealVector<Dim> spacing{};
    for (int axis = 0; axis < Dim; ++axis) {
      spacing[axis] = geometry.spacing(axis);
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        if (topology.is_physical(face))
          faces[static_cast<std::size_t>(face.ordinal())] = {PhysicalBoundaryKind::dirichlet,
                                                             Real(0), Real(1), Real(0)};
      }
    }
    return PhysicalBoundaryConditions<Dim>{topology, faces, spacing};
  }

  std::unique_ptr<hierarchy_tensor_solver_type> prepare_hierarchy_tensor_solver_(
      const HierarchyTensorSelection& selection) const {
    hierarchy_tensor_request_type request;
    std::exception_ptr local_error;
    long local_failure = 0;
    try {
      const int runtime_block = sys_block(selection.program_block);
      if (runtime_block < 0)
        throw std::invalid_argument("AMR hierarchy tensor solver has an invalid runtime block");
      request.block = static_cast<std::size_t>(runtime_block);
      request.components = selection.components;
      request.plan_identity = selection.plan_identity;
      request.operator_contract_identity = selection.operator_contract_identity;
      request.assembly_field_slots = selection.assembly_field_slots;
      request.solution_field_slot = selection.solution_field_slot;
      request.options = selection.options;
      request.levels.reserve(runtime_->hierarchy().num_levels());
      if (runtime_->hierarchy().num_levels() > 1)
        request.ratios.reserve(runtime_->hierarchy().num_levels() - 1);
      for (std::size_t level = 0; level < runtime_->hierarchy().num_levels(); ++level) {
        const field_type& level_state = runtime_->hierarchy().state(level);
        const Geometry<Dim> level_geometry =
            facade_->prepared_amr_level_geometry(static_cast<int>(level));
        request.levels.push_back({level_geometry, hierarchy_tensor_boundary_(level_geometry),
                                  level_state.layout(), level_state.distribution(),
                                  level_state.local_rank()});
        if (level != 0)
          request.ratios.push_back(runtime_->hierarchy().layout(level).ratio_from_parent());
      }
    } catch (...) {
      local_failure = 1;
      local_error = std::current_exception();
    }
    const ExecutionLane& lane = prepared_execution_lane();
    if (all_reduce_max(local_failure, lane) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "AMR hierarchy tensor request construction failed on another MPI rank");
    }
    return prepare_hierarchy_tensor_solver_collectively(
        *hierarchy_tensor_solver_registry_, selection.provider_identity, std::move(request), lane);
  }

  hierarchy_tensor_solver_type& configured_hierarchy_tensor_solver_() const {
    if (!hierarchy_tensor_selection_)
      throw std::logic_error(
          "AMR hierarchy tensor solver must be configured before hierarchy access");
    refresh_resources_();
    if (!hierarchy_tensor_solver_ ||
        hierarchy_tensor_topology_epoch_ != runtime_->topology_epoch() ||
        hierarchy_tensor_materialization_generation_ != runtime_->materialization_generation()) {
      std::unique_ptr<hierarchy_tensor_solver_type> prepared =
          prepare_hierarchy_tensor_solver_(*hierarchy_tensor_selection_);
      hierarchy_tensor_solver_ = std::move(prepared);
      hierarchy_tensor_topology_epoch_ = runtime_->topology_epoch();
      hierarchy_tensor_materialization_generation_ = runtime_->materialization_generation();
    }
    return *hierarchy_tensor_solver_;
  }

  void require_hierarchy_tensor_binding_(int program_block, int components) const {
    if (!hierarchy_tensor_selection_ ||
        hierarchy_tensor_selection_->program_block != program_block ||
        hierarchy_tensor_selection_->components != components)
      throw std::logic_error(
          "AMR hierarchy tensor block/component binding differs from its prepared solver");
  }

  void synchronize_resource_generation_() const {
    if (interface_flux_ledger_ &&
        interface_flux_ledger_->topology_epoch() != runtime_->topology_epoch())
      interface_flux_commit_guard_.reset();
    prepare_coupled_jacvec_scratch_();
    if (!interface_flux_ledger_) {
      interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
          runtime_->topology_epoch(), inactive_interface_flux_budget_());
    } else {
      interface_flux_ledger_->advance_topology_epoch(runtime_->topology_epoch());
    }
    resource_epoch_ = runtime_->topology_epoch();
    resource_generation_ = runtime_->materialization_generation();
    std::map<std::string, int> indexed_histories;
    if (facade_ != nullptr) {
      for (const auto& [key, ring] : runtime_state().hist_.histories) {
        (void)ring;
        const auto decoded = decode_history_key_(key);
        if (!decoded ||
            static_cast<std::size_t>(decoded->first) >= runtime_->hierarchy().num_levels())
          throw std::runtime_error(
              "AMR Program history registry is not qualified by the live hierarchy");
        indexed_histories.emplace(key, decoded->first);
      }
    }
    history_levels_.swap(indexed_histories);
    if (history_levels_.empty()) {
      history_epoch_ = std::numeric_limits<std::uint64_t>::max();
      history_generation_ = std::numeric_limits<std::uint64_t>::max();
    } else {
      history_epoch_ = resource_epoch_;
      history_generation_ = resource_generation_;
    }
  }
  void refresh_resources_() const {
    if (facade_ != nullptr)
      facade_->refresh_prepared_amr_levels();
    if (resource_epoch_ == runtime_->topology_epoch() &&
        resource_generation_ == runtime_->materialization_generation())
      return;
    if (!history_levels_.empty() && (history_epoch_ != runtime_->topology_epoch() ||
                                     history_generation_ != runtime_->materialization_generation()))
      throw std::runtime_error(
          "AMR Program topology changed while retained histories still name the prior layouts");
    scratches_.clear();
    synchronize_resource_generation_();
    if (active_level_ >= nlev())
      active_level_ = 0;
  }

  static Extent<Dim> uniform_ghosts_(int depth) {
    if (depth < 0)
      throw std::invalid_argument("AMR Program ghost depth must be non-negative");
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = depth;
    return result;
  }

  static field_type make_scratch_(const field_type& prototype, int ncomp, Extent<Dim> ghosts) {
    if (ncomp < 1)
      throw std::invalid_argument("AMR Program scratch requires positive components");
    field_type result(prototype.layout(), prototype.distribution(), prototype.local_rank(), ncomp,
                      ghosts);
    result.set_val(Real(0));
    return result;
  }

  struct CoupledJacvecLevelScratch {
    std::array<std::unique_ptr<field_type>, 2> residual;
    std::array<std::unique_ptr<field_type>, 2> coupled;
  };

  struct CoupledJacvecScratch {
    std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
    std::vector<CoupledJacvecLevelScratch> levels;
  };

  void prepare_coupled_jacvec_scratch_() const {
    if (facade_ == nullptr)
      return;
    const std::uint64_t topology_epoch = runtime_->topology_epoch();
    const std::uint64_t materialization_generation = runtime_->materialization_generation();
    if (coupled_jacvec_scratch_ && coupled_jacvec_scratch_->topology_epoch == topology_epoch &&
        coupled_jacvec_scratch_->materialization_generation == materialization_generation)
      return;

    const ExecutionLane& lane = prepared_execution_lane();
    std::unique_ptr<CoupledJacvecScratch> candidate;
    std::exception_ptr preparation_error;
    try {
      if (facade_->n_blocks() == 2) {
        candidate = std::make_unique<CoupledJacvecScratch>();
        candidate->topology_epoch = topology_epoch;
        candidate->materialization_generation = materialization_generation;
        candidate->levels.resize(runtime_->hierarchy().num_levels());
        for (std::size_t level = 0; level < candidate->levels.size(); ++level)
          for (int runtime_block = 0; runtime_block < 2; ++runtime_block) {
            const field_type& prototype =
                facade_->prepared_amr_block_state(runtime_block, static_cast<int>(level));
            auto residual = std::make_unique<field_type>(
                prototype.layout(), prototype.distribution(), prototype.local_rank(),
                prototype.ncomp(), prototype.ghosts());
            auto coupled = std::make_unique<field_type>(
                prototype.layout(), prototype.distribution(), prototype.local_rank(),
                prototype.ncomp(), prototype.ghosts());
            residual->set_val(Real(0));
            coupled->set_val(Real(0));
            candidate->levels[level].residual[static_cast<std::size_t>(runtime_block)] =
                std::move(residual);
            candidate->levels[level].coupled[static_cast<std::size_t>(runtime_block)] =
                std::move(coupled);
          }
      }
    } catch (...) {
      preparation_error = std::current_exception();
    }
    if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && preparation_error)
        std::rethrow_exception(preparation_error);
      throw std::runtime_error("AMR coupled Jacobian scratch preparation failed collectively");
    }
    coupled_jacvec_scratch_ = std::move(candidate);
  }

  CoupledJacvecLevelScratch& require_coupled_jacvec_scratch_(
      int first_block, const field_type& first_state, const field_type& first_result,
      int second_block, const field_type& second_state, const field_type& second_result) const {
    if (!coupled_jacvec_scratch_ ||
        coupled_jacvec_scratch_->topology_epoch != runtime_->topology_epoch() ||
        coupled_jacvec_scratch_->materialization_generation !=
            runtime_->materialization_generation() ||
        active_level_ < 0 ||
        static_cast<std::size_t>(active_level_) >= coupled_jacvec_scratch_->levels.size())
      throw std::logic_error("AMR coupled Jacobian scratch is stale or unprepared");
    const int first_runtime = sys_block(first_block);
    const int second_runtime = sys_block(second_block);
    auto& level = coupled_jacvec_scratch_->levels[static_cast<std::size_t>(active_level_)];
    const auto require_block = [&](int runtime_block, const field_type& state,
                                   const field_type& result) {
      const std::size_t index = static_cast<std::size_t>(runtime_block);
      if (runtime_block < 0 || runtime_block >= 2 || !level.residual[index] ||
          !level.coupled[index])
        throw std::logic_error("AMR coupled Jacobian block scratch is incomplete");
      const field_type& prepared_state = *level.coupled[index];
      const field_type& prepared_result = *level.residual[index];
      require_same_field_contract_(state, prepared_state, "AMR coupled Jacobian state scratch");
      require_same_field_contract_(result, prepared_result,
                                   "AMR coupled Jacobian residual scratch");
      if (state.ghosts() != prepared_state.ghosts() ||
          result.ghosts() != prepared_result.ghosts() ||
          state.shares_storage_with(prepared_state) || state.shares_storage_with(prepared_result) ||
          result.shares_storage_with(prepared_state) || result.shares_storage_with(prepared_result))
        throw std::invalid_argument(
            "AMR coupled Jacobian scratch changed or aliases an invocation field");
    };
    require_block(first_runtime, first_state, first_result);
    require_block(second_runtime, second_state, second_result);
    return level;
  }

  field_type& persistent_scratch_(ScratchKind kind, std::int64_t value_id, int subslot,
                                  const field_type& prototype, int ncomp,
                                  Extent<Dim> ghosts) const {
    if (value_id < 0 || subslot < 0)
      throw std::invalid_argument("AMR Program scratch identity must be non-negative");
    refresh_resources_();
    const ScratchKey key{kind, active_level_, value_id, subslot};
    auto found = scratches_.find(key);
    if (found == scratches_.end())
      found = scratches_.emplace(key, make_scratch_(prototype, ncomp, ghosts)).first;
    field_type& result = found->second;
    if (result.layout() != prototype.layout() ||
        result.distribution() != prototype.distribution() ||
        result.local_rank() != prototype.local_rank() || result.ncomp() != ncomp ||
        result.ghosts() != ghosts)
      throw std::runtime_error("AMR Program scratch identity changed its exact field contract");
    result.set_val(Real(0));
    clear_active_flux_expression_(result);
    return result;
  }

  static std::string history_key_(const std::string& name, int level) {
    if (name.empty() || level < 0)
      throw std::invalid_argument("AMR Program history key is invalid");
    return "pops.amr.level-history.v1/" + std::to_string(level) + "/" +
           std::to_string(name.size()) + ":" + name;
  }

  void require_history_owner_(int program_owner) const {
    if (program_owner < 0 || sys_block(program_owner) < 0)
      throw std::invalid_argument("AMR Program history has a foreign block owner");
  }

  field_type& history_slot_(const std::string& name, int lag, bool zero_start,
                            int components) const {
    const std::string key = history_key_(name, active_level_);
    auto& manager = runtime_state().hist_;
    const auto found = manager.histories.find(key);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(key))
      throw std::out_of_range("AMR Program history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (components >= 0 && result.ncomp() != components)
      throw std::invalid_argument("AMR Program history component contract differs");
    if (!manager.initialized.at(key)) {
      if (!zero_start)
        throw std::runtime_error("AMR Program history has not been initialized");
    }
    return result;
  }

  void store_history_(const std::string& name, const field_type& value) const {
    const std::string key = history_key_(name, active_level_);
    auto& manager = runtime_state().hist_;
    const auto found = manager.histories.find(key);
    if (found == manager.histories.end())
      throw std::out_of_range("AMR Program history is not registered on the active level");
    require_same_field_contract_(found->second.front(), value, "AMR Program history store");
    if (!std::isfinite(current_dt_) || !(current_dt_ > 0.0))
      throw std::logic_error("AMR Program history store has no positive exact interval");
    found->second.front() = value;
    auto& dts = manager.slot_dt.at(key);
    if (dts.size() != found->second.size())
      throw std::logic_error("AMR Program history dt ledger differs from its ring depth");
    if (!manager.initialized.at(key)) {
      for (std::size_t slot = 1; slot < found->second.size(); ++slot) {
        found->second[slot] = value;
        dts[slot] = static_cast<Real>(current_dt_);
      }
    }
    manager.initialized[key] = true;
    manager.store_pending[key] = true;
    dts.front() = static_cast<Real>(current_dt_);
  }

  void rotate_histories_(std::optional<std::string> clock_identity) const {
    auto& manager = runtime_state().hist_;
    std::vector<std::string> selected;
    for (const auto& [key, level] : history_levels_) {
      if (level != active_level_)
        continue;
      const auto ring = manager.histories.find(key);
      const auto clock = manager.clock_identity.find(key);
      const auto dts = manager.slot_dt.find(key);
      if (ring == manager.histories.end() || clock == manager.clock_identity.end() ||
          dts == manager.slot_dt.end() || ring->second.size() != dts->second.size() ||
          static_cast<int>(ring->second.size()) != manager.depth.at(key))
        throw std::runtime_error("AMR Program history registry is incomplete");
      if (!clock_identity || clock->second == *clock_identity)
        selected.push_back(key);
    }
    for (const std::string& key : selected) {
      auto& ring = manager.histories.at(key);
      for (std::size_t slot = ring.size(); slot-- > 1;)
        std::swap(ring[slot], ring[slot - 1]);
      auto& dts = manager.slot_dt.at(key);
      for (std::size_t slot = dts.size(); slot-- > 1;)
        std::swap(dts[slot], dts[slot - 1]);
      if (manager.store_pending.at(key)) {
        manager.fill_count[key] =
            std::min(static_cast<int>(ring.size()), manager.fill_count.at(key) + 1);
        manager.store_pending[key] = false;
      }
    }
  }

  std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                         const std::string& clock,
                                                         const std::string& stage_identity,
                                                         int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level, active_level_,
                                      static_cast<std::int64_t>(macro_step()));
  }

  OperatorFingerprint operator_topology_(const field_type& prototype) const {
    require_same_layout_(prototype, state(0), "AMR Program operator topology");
    OperatorFingerprint fingerprint =
        ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
    ::pops::detail::fingerprint_geometry(fingerprint, geometry());
    ::pops::detail::fingerprint_mix(fingerprint, runtime_->spatial_contract());
    ::pops::detail::fingerprint_mix(fingerprint, runtime_->topology_epoch());
    ::pops::detail::fingerprint_mix(fingerprint, runtime_->materialization_generation());
    ::pops::detail::fingerprint_mix(fingerprint, static_cast<std::uint64_t>(active_level_));
    return fingerprint;
  }

  OperatorEvaluationSnapshot current_operator_snapshot_(OperatorFingerprint authority,
                                                        OperatorFingerprint topology,
                                                        OperatorFingerprint resources,
                                                        std::uint64_t revision) const {
    refresh_resources_();
    const std::uint64_t maximum_generation =
        std::max(runtime_->topology_epoch(), runtime_->materialization_generation());
    if (maximum_generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program topology revision exhausted uint64_t");
    const double evaluation_time =
        static_cast<double>(physical_time()) + stage_time_.value() * current_dt_;
    return {authority,
            revision,
            static_cast<std::int64_t>(macro_step()),
            stage_time_.numerator,
            stage_time_.denominator,
            std::bit_cast<std::uint64_t>(current_dt_),
            std::bit_cast<std::uint64_t>(evaluation_time),
            maximum_generation + 1,
            topology,
            resources};
  }

  static void require_same_layout_(const field_type& left, const field_type& right,
                                   std::string_view operation) {
    if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
        left.local_rank() != right.local_rank() || left.local_size() != right.local_size())
      throw std::invalid_argument(std::string(operation) + " fields have different exact layouts");
  }
  static void require_same_field_contract_(const field_type& left, const field_type& right,
                                           std::string_view operation) {
    require_same_layout_(left, right, operation);
    if (left.ncomp() != right.ncomp())
      throw std::invalid_argument(std::string(operation) + " fields have different components");
  }
  static void require_scalar_stencil_(const field_type& output, const field_type& input,
                                      int output_components, std::string_view operation) {
    require_same_layout_(output, input, operation);
    if (input.ncomp() != 1 || output.ncomp() != output_components)
      throw std::invalid_argument(std::string(operation) + " has an invalid component contract");
    for (int axis = 0; axis < Dim; ++axis)
      if (input.ghosts()[axis] < 1)
        throw std::invalid_argument(std::string(operation) + " requires one ghost per axis");
  }
  void require_boundary_point_(const runtime::multiblock::BoundaryEvaluationPoint& point,
                               std::string_view operation) const {
    if (point.level != active_level_ || point.clock.empty() || point.stage < 0 ||
        point.stage_fraction.denominator <= 0)
      throw std::invalid_argument(std::string(operation) + " has a foreign evaluation point");
  }

  const field_type* staged_parent_for_block_(int runtime_block) const {
    if (runtime_block < 0)
      throw std::out_of_range("AMR Program staged-parent block is out of range");
    if (active_staged_parents_.empty())
      return nullptr;
    if (static_cast<std::size_t>(runtime_block) >= active_staged_parents_.size())
      throw std::logic_error("AMR Program staged-parent registry is incomplete");
    return active_staged_parents_[static_cast<std::size_t>(runtime_block)];
  }

  std::optional<int> authenticated_runtime_block_for_state_target_(const field_type& target) const {
    require_facade_execution_();
    const auto& map = facade_->program_block_map();
    if (map.size() != static_cast<std::size_t>(n_blocks()))
      throw std::logic_error("AMR Program state target has no complete authenticated block map");
    std::optional<int> match;
    for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block) {
      const field_type* candidate = nullptr;
      if (!active_attempt_states_.empty())
        candidate = active_attempt_states_.at(static_cast<std::size_t>(runtime_block));
      const field_type* accepted = &facade_->prepared_amr_block_state(runtime_block, active_level_);
      if (&target != candidate && &target != accepted)
        continue;
      if (match)
        throw std::logic_error("AMR Program state target aliases two runtime blocks");
      match = runtime_block;
    }
    return match;
  }

  static void copy_full_(const field_type& source, field_type& destination) {
    require_same_field_contract_(source, destination, "AMR Program full-field copy");
    if (source.ghosts() != destination.ghosts() || source.shares_storage_with(destination))
      throw std::invalid_argument(
          "AMR Program full-field copy requires detached exact ghost storage");
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      if (source.global_index(local) != destination.global_index(local) ||
          source.fab(local).box() != destination.fab(local).box() ||
          source.fab(local).grown_box() != destination.fab(local).grown_box() ||
          source.fab(local).size() != destination.fab(local).size())
        throw std::invalid_argument("AMR Program full-field copy patch storage changed");
    }
    for (std::size_t local = 0; local < destination.local_size(); ++local)
      Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
  }

  static void copy_valid_(const field_type& source, field_type& destination) {
    require_same_field_contract_(source, destination, "AMR Program valid-field copy");
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      const auto input = source.fab(local).view();
      const auto output = destination.fab(local).view();
      const int components = destination.ncomp();
      for_each_cell(destination.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component)
          output(cell, component) = input(cell, component);
      });
    }
  }

  void laplacian_without_fill_(field_type& output, field_type& input,
                               const Geometry<Dim>& geom) const {
    require_scalar_stencil_(output, input, 1, "AMR Program Laplacian");
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const auto result = output.fab(local).view();
      const auto value = std::as_const(input).fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        Real image = Real(0);
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell;
          Index<Dim> upper = cell;
          --lower[axis];
          ++upper[axis];
          const Real spacing = geom.spacing(axis);
          image +=
              (value(upper, 0) - Real(2) * value(cell, 0) + value(lower, 0)) / (spacing * spacing);
        }
        result(cell, 0) = image;
      });
    }
    count_kernel_();
  }

  void gradient_without_fill_(field_type& output, field_type& input,
                              const Geometry<Dim>& geom) const {
    require_scalar_stencil_(output, input, Dim, "AMR Program gradient");
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const auto result = output.fab(local).view();
      const auto value = std::as_const(input).fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell;
          Index<Dim> upper = cell;
          --lower[axis];
          ++upper[axis];
          result(cell, axis) = (value(upper, 0) - value(lower, 0)) / (Real(2) * geom.spacing(axis));
        }
      });
    }
    count_kernel_();
  }

  std::uint64_t next_boundary_generation_() const {
    if (boundary_generation_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program boundary generation exhausted uint64_t");
    return ++boundary_generation_;
  }
  void count_kernel_(std::int64_t count = 1) const {
    if (facade_ != nullptr)
      facade_->profiler_handle().count("kernel_launches", count);
  }

  facade_type* facade_ = nullptr;
  runtime_type* runtime_ = nullptr;
  mutable int active_level_ = 0;
  mutable double current_dt_ = 0.0;
  mutable double current_interval_start_time_ = 0.0;
  mutable ::pops::amr::Rational current_interval_begin_phase_{0, 1};
  mutable ::pops::amr::Rational current_interval_end_phase_{1, 1};
  mutable int logical_substep_ = 0;
  mutable ::pops::amr::Rational stage_time_{0, 1};
  mutable std::string primary_clock_;
  mutable ClockScheduleState clock_schedule_;
  mutable std::uint64_t boundary_generation_ = 0;
  mutable std::uint64_t resource_epoch_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t resource_generation_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t history_epoch_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t history_generation_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::optional<OperatorEvaluationSnapshot> active_operator_snapshot_;
  mutable std::map<std::string, int> history_levels_;
  mutable std::map<ScratchKey, field_type> scratches_;
  mutable std::mutex coupled_jacvec_mutex_;
  mutable std::unique_ptr<CoupledJacvecScratch> coupled_jacvec_scratch_;
  mutable std::map<std::int64_t, GeneratedFieldRoute> generated_field_routes_;
  std::shared_ptr<const hierarchy_tensor_registry_type> hierarchy_tensor_solver_registry_;
  mutable std::optional<HierarchyTensorSelection> hierarchy_tensor_selection_;
  mutable std::unique_ptr<hierarchy_tensor_solver_type> hierarchy_tensor_solver_;
  mutable std::uint64_t hierarchy_tensor_topology_epoch_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t hierarchy_tensor_materialization_generation_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable PreparedVectorDistribution<Dim> vector_distribution_ =
      PreparedVectorDistribution<Dim>::distributed();
  mutable std::vector<field_type*> active_attempt_states_;
  mutable std::vector<const field_type*> active_staged_parents_;
  mutable std::vector<multiblock_flux_ledger_type*> active_incoming_flux_;
  mutable std::vector<multiblock_flux_ledger_type*> active_outgoing_flux_;
  mutable std::vector<std::string_view> active_block_identities_;
  mutable FluxExpressionRegistry active_flux_expressions_;
  mutable std::vector<std::size_t> active_flux_basis_counts_;
  mutable std::uint64_t next_active_flux_basis_identity_ = 0;
  mutable std::vector<std::size_t> prepared_rhs_basis_bounds_;
  mutable std::vector<std::size_t> prepared_coefficient_term_bounds_;
  mutable ::pops::amr::ClockWindow active_subcycling_window_{};
  mutable std::uint64_t active_subcycling_attempt_ = 0;
  mutable std::unique_ptr<multiblock_subcycling_type> multiblock_subcycling_;
  mutable std::uint64_t multiblock_subcycling_epoch_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t multiblock_subcycling_generation_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable std::string multiblock_subcycling_program_budget_contract_;
  mutable CellTemporalPartitionAcceptedState accepted_temporal_partition_;
  mutable std::string accepted_flux_budget_contract_;
  mutable std::string accepted_coupling_contract_;
  mutable std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>
      accepted_face_flux_;
  mutable std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger_;
  mutable std::optional<typename interface_flux_ledger_type::PreparedCommit>
      interface_flux_commit_guard_;
  mutable std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events_;
  mutable std::uint64_t accepted_state_revision_ = std::numeric_limits<std::uint64_t>::max();
};

template <int Dim>
std::shared_ptr<AmrProgramContext<Dim>> make_program_execution_provider(
    ::pops::AmrSystem<Dim>* system) {
  return std::make_shared<AmrProgramContext<Dim>>(system);
}

template <int Dim>
AmrProgramContext<Dim> make_program_execution_view(::pops::AmrSystem<Dim>* system) {
  return AmrProgramContext<Dim>(system);
}

}  // namespace pops::runtime::program
