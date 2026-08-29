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
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/prepared_tensor_boundary_session.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/source_mask.hpp>
#include <pops/runtime/program/same_level_cell_temporal_provider.hpp>
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
class ProgramExecutionServices;

namespace detail {
template <int Dim>
struct AmrProgramHistoryRemapCollectiveTestAccess;
template <int Dim, class MemorySpace>
class AmrStorageTopologyAdapter;
}  // namespace detail

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
template <int Dim, class MemorySpace>
class detail::AmrStorageTopologyAdapter {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "AmrStorageTopologyAdapter only supports dimensions 1, 2, and 3");
  static_assert(std::is_same_v<MemorySpace, typename Kokkos::DefaultExecutionSpace::memory_space>,
                "AmrStorageTopologyAdapter memory space must match its compiled AmrSystem leaf");

  static constexpr int dimension = Dim;
  using facade_type = ::pops::AmrSystem<Dim>;
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using hierarchy_type = typename runtime_type::hierarchy_type;
  using field_type = typename runtime_type::field_type;
  using level_evaluation_type = typename facade_type::PreparedLevelEvaluation;
  using runtime_state_type = ProgramRuntimeState<Dim>;
  using scalar_boundary_session_type = PreparedScalarBoundarySession<Dim>;
  using tensor_boundary_session_type = PreparedTensorBoundarySession<Dim>;
  class PreparedBlockBoundarySession {
   public:
    PreparedBlockBoundarySession(const PreparedBlockBoundarySession&) = default;
    PreparedBlockBoundarySession& operator=(const PreparedBlockBoundarySession&) = default;

   private:
    friend class AmrStorageTopologyAdapter;
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

  /// Bounded image-owned topology used while a DSO declares its candidate.  It deliberately
  /// carries no AmrSystem pointer: state prototypes and block routes are captured before the DSO
  /// callback, while runtime/lane are owned by the detached candidate graph until publication.
  struct PreparedAmrTopologyView final {
    runtime_type* runtime = nullptr;
    ProgramRuntimeState<Dim>* program_state = nullptr;
    const ExecutionLane* lane = nullptr;
    std::vector<int> program_block_map;
    std::vector<std::vector<field_type>> block_prototypes;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;

    void validate() const {
      if (runtime == nullptr || program_state == nullptr || lane == nullptr || !lane->active() ||
          program_block_map.empty())
        throw std::invalid_argument("AMR preparation topology view is incomplete");
      if (topology_epoch != runtime->topology_epoch() ||
          materialization_generation != runtime->materialization_generation())
        throw std::invalid_argument("AMR preparation topology view is stale");
      const std::size_t levels = runtime->hierarchy().num_levels();
      if (levels == 0 || block_prototypes.empty())
        throw std::invalid_argument("AMR preparation topology view has no level prototypes");
      for (const int runtime_block : program_block_map)
        if (runtime_block < 0 ||
            static_cast<std::size_t>(runtime_block) >= block_prototypes.size() ||
            block_prototypes[static_cast<std::size_t>(runtime_block)].size() != levels)
          throw std::invalid_argument("AMR preparation topology view has an invalid block map");
    }
  };

  struct ProgramResourceTopology {
    int levels = 0;
    std::uint64_t epoch = 0;
    std::uint64_t generation = 0;
  };

  using FieldStageOverride = ProgramFieldStageOverride<Dim>;
  using RhsGroupRequest = ProgramRhsGroupRequest<Dim>;

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

  struct HierarchyTensorLevelBoundary {
    Geometry<Dim> geometry;
    PhysicalBoundaryConditions<Dim> conditions;
  };

  struct PreparedHierarchyTensorState {
    std::unique_ptr<hierarchy_tensor_solver_type> solver;
    std::vector<HierarchyTensorLevelBoundary> boundaries;
  };

  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const AmrStorageTopologyAdapter& owner, int iteration, int count)
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
    const AmrStorageTopologyAdapter* owner_ = nullptr;
    double prior_dt_ = 0.0;
    double prior_interval_start_time_ = 0.0;
    ::pops::amr::Rational prior_interval_begin_phase_{0, 1};
    ::pops::amr::Rational prior_interval_end_phase_{1, 1};
    int prior_substep_ = 0;
  };

  /// Bind-sealed pointer packs and per-level rollback images for AMR Program hot paths.  All
  /// vectors and MultiFabs are materialized from the immutable topology/facade view before the
  /// first candidate step; execution only rewrites pointer slots and copies into those images.
  struct PreparedHotPathWorkspace {
    bool bound = false;
    std::size_t block_capacity = 0;
    std::size_t level_capacity = 0;
    std::vector<const RhsGroupRequest*> rhs_ordered;
    std::vector<int> rhs_runtime_blocks;
    std::vector<int> rhs_rates;
    std::vector<int> rhs_flux_only;
    std::vector<field_type*> rhs_residuals;
    std::vector<runtime::multiblock::BoundaryEvaluationPoint> rhs_points;
    std::vector<std::pair<int, int>> rhs_evaluation_targets;
    std::vector<const field_type*> rhs_staged_parents;
    std::vector<const level_evaluation_type*> rhs_evaluations;
    std::vector<field_type*> rhs_candidates;
    std::vector<field_type*> rhs_backups;
    std::vector<field_type*> coupling_states;
    std::vector<field_type*> commit_targets;
    std::vector<const field_type*> commit_sources;
    std::vector<std::optional<int>> commit_runtime_blocks;
    std::vector<field_type*> commit_snapshots;
    std::vector<std::size_t> accepted_snapshot_by_runtime;
    std::vector<field_type> publication_candidates;
    std::vector<field_type*> publication_program_candidates;
    std::vector<field_type> candidate_storage;
    std::vector<field_type> backup_storage;
    std::vector<field_type> commit_storage;

    template <class Getter>
    void bind(std::size_t blocks, std::size_t levels, Getter&& getter) {
      if (blocks == 0 || levels == 0)
        throw std::invalid_argument("AMR Program hot-path workspace has no blocks or levels");
      if (bound) {
        if (block_capacity != blocks || level_capacity != levels)
          throw std::logic_error("AMR Program hot-path workspace shape changed after preparation");
        return;
      }
      block_capacity = blocks;
      level_capacity = levels;
      rhs_ordered.resize(blocks);
      rhs_runtime_blocks.resize(blocks);
      rhs_rates.resize(blocks);
      rhs_flux_only.resize(blocks);
      rhs_residuals.resize(blocks);
      rhs_points.resize(blocks);
      rhs_evaluation_targets.resize(blocks);
      rhs_staged_parents.resize(blocks);
      rhs_evaluations.resize(blocks);
      rhs_candidates.resize(blocks);
      rhs_backups.resize(blocks);
      coupling_states.resize(blocks);
      commit_targets.resize(blocks);
      commit_sources.resize(blocks);
      commit_runtime_blocks.resize(blocks);
      commit_snapshots.resize(blocks);
      accepted_snapshot_by_runtime.resize(blocks);
      publication_program_candidates.resize(blocks);
      const std::size_t total = blocks * levels;
      candidate_storage.reserve(total);
      backup_storage.reserve(total);
      commit_storage.reserve(total);
      publication_candidates.reserve(total);
      for (std::size_t level = 0; level < levels; ++level)
        for (std::size_t block = 0; block < blocks; ++block) {
          const field_type& prototype = getter(block, level);
          candidate_storage.emplace_back(prototype);
          backup_storage.emplace_back(prototype);
          commit_storage.emplace_back(prototype);
          publication_candidates.emplace_back(prototype);
        }
      bound = true;
    }

    [[nodiscard]] field_type& candidate(std::size_t level, std::size_t block) {
      return candidate_storage.at(level * block_capacity + block);
    }
    [[nodiscard]] field_type& backup(std::size_t level, std::size_t block) {
      return backup_storage.at(level * block_capacity + block);
    }
    [[nodiscard]] field_type& commit_snapshot(std::size_t level, std::size_t block) {
      return commit_storage.at(level * block_capacity + block);
    }

    void require_bound(std::size_t count, const char* operation) const {
      if (!bound || count > block_capacity)
        throw std::logic_error(std::string(operation) +
                               " requires a bind-sealed AMR hot-path workspace");
    }
  };

  void bind_preparation_hot_path_workspace_() const {
    if (preparation_view_ == nullptr)
      return;
    const std::size_t blocks = preparation_view_->block_prototypes.size();
    const std::size_t levels = preparation_view_->runtime->hierarchy().num_levels();
    hot_path_workspace_.bind(blocks, levels,
                             [this](std::size_t block, std::size_t level) -> const field_type& {
                               return preparation_view_->block_prototypes.at(block).at(level);
                             });
  }

  void bind_accepted_hot_path_workspace_() const {
    if (facade_ == nullptr || runtime_ == nullptr)
      return;
    const std::size_t blocks = static_cast<std::size_t>(n_blocks());
    const std::size_t levels = static_cast<std::size_t>(nlev());
    hot_path_workspace_.bind(blocks, levels,
                             [this](std::size_t block, std::size_t level) -> const field_type& {
                               return facade_->program_prepared_amr_block_state_(
                                   static_cast<int>(block), static_cast<int>(level));
                             });
  }

  AmrStorageTopologyAdapter() = default;
  explicit AmrStorageTopologyAdapter(const PreparedAmrTopologyView* preparation_view)
      : preparation_view_(preparation_view), preparation_mode_(true) {
    if (preparation_view_ == nullptr)
      throw std::invalid_argument("AMR preparation adapter has no topology view");
    preparation_view_->validate();
    runtime_ = preparation_view_->runtime;
    synchronize_resource_generation_();
    bind_preparation_hot_path_workspace_();
  }

  /// Bind the finite ProgramResourcePlan slot space before a DSO prelude can request scratch.
  /// Slots are deliberately dense indexes, never generated value ids.
  void bind_prepared_scratch_slots(std::size_t count) const {
    if (preparation_view_ == nullptr)
      throw std::logic_error("AMR scratch slots can only bind on a detached preparation image");
    prepared_scratch_.clear();
    prepared_scratch_.resize(count);
    bind_preparation_hot_path_workspace_();
  }

  /// Called by the host only after every rank has accepted the complete image.  The retained DSO
  /// provider stops borrowing the detached candidate view before the owner is made reachable.
  void bind_accepted_facade(facade_type* facade) {
    facade_ = require_facade_(facade);
    runtime_ = require_runtime_(*facade_);
    preparation_view_ = nullptr;
    preparation_mode_ = false;
    hierarchy_tensor_solver_registry_ = facade_->hierarchy_tensor_solver_provider_registry();
    // AMR histories are materialized against the detached ProgramRuntimeState before the
    // owner-last ProgramRuntimeState exchange.  At this point the facade still exposes the prior
    // accepted Program image, so reindexing from it would erase the exact level-qualified keys
    // just prepared for the candidate.  Preserve that frozen index until the host publishes the
    // matching Program state.  History-free artifacts still initialize their resource carriers
    // here from the accepted facade.
    if (history_levels_.empty())
      synchronize_resource_generation_();
    bind_accepted_hot_path_workspace_();
  }
  /// A generated closure captures the public execution service by value.  The capture must retain
  /// the immutable facade binding and clock declaration, but it must never share an in-flight AMR
  /// scratch, solver, flux ledger, or mutex with the source view.  Rebuilding those attempt-local
  /// carriers lazily on the captured view preserves the single ProgramExecutionServices authority
  /// while making ordinary Uniform captures possible (their AMR adapter is simply disengaged).
  AmrStorageTopologyAdapter(const AmrStorageTopologyAdapter& other)
      : facade_(other.facade_),
        runtime_(other.runtime_),
        active_level_(other.active_level_),
        current_dt_(other.current_dt_),
        current_interval_start_time_(other.current_interval_start_time_),
        current_interval_begin_phase_(other.current_interval_begin_phase_),
        current_interval_end_phase_(other.current_interval_end_phase_),
        logical_substep_(other.logical_substep_),
        stage_time_(other.stage_time_),
        primary_clock_(other.primary_clock_),
        clock_schedule_(other.clock_schedule_),
        boundary_generation_(other.boundary_generation_),
        resource_epoch_(other.resource_epoch_),
        resource_generation_(other.resource_generation_),
        history_epoch_(other.history_epoch_),
        history_generation_(other.history_generation_),
        operator_snapshot_revision_(other.operator_snapshot_revision_),
        history_levels_(other.history_levels_),
        hierarchy_tensor_solver_registry_(other.hierarchy_tensor_solver_registry_),
        vector_distribution_(other.vector_distribution_),
        accepted_temporal_partition_(other.accepted_temporal_partition_),
        cell_temporal_configuration_(other.cell_temporal_configuration_),
        cell_temporal_diagnostics_(other.cell_temporal_diagnostics_),
        cell_temporal_interval_begin_tick_(other.cell_temporal_interval_begin_tick_),
        cell_temporal_interval_target_tick_(other.cell_temporal_interval_target_tick_),
        accepted_flux_budget_contract_(other.accepted_flux_budget_contract_),
        accepted_coupling_contract_(other.accepted_coupling_contract_),
        accepted_face_flux_(other.accepted_face_flux_),
        accepted_synchronization_events_(other.accepted_synchronization_events_),
        accepted_state_revision_(other.accepted_state_revision_),
        hot_path_workspace_(other.hot_path_workspace_) {}
  AmrStorageTopologyAdapter& operator=(const AmrStorageTopologyAdapter&) = delete;

  [[nodiscard]] ProgramHostDescriptor program_host_descriptor() const {
    return const_cast<facade_type*>(facade_)->program_host_descriptor();
  }

  // During DSO preparation the facade redirects this declaration to its non-owning candidate
  // graph.  The DSO never receives a registry pointer or an extra native callback.
  void stage_auxiliary_consumer_plan(
      runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) const {
    if (facade_ == nullptr)
      throw std::logic_error("AMR auxiliary consumer plan requires one execution facade");
    facade_->install_auxiliary_consumer_plan(std::move(plan));
  }

  // Class-scope responsibility fragments preserve the public nested-type identities and member
  // layout of AmrStorageTopologyAdapter while making each semantic authority independently auditable.
#include <pops/runtime/program/detail/program_execution_services_amr_spatial.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_public.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_public.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_spatial_operations.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_public.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_solver.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_private.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_polynomial.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_cell_temporal_configuration.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_basis_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_cell_temporal_level_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_services.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_cell_temporal_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_subcycling_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_basis.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_services.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_services.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_spatial_operations_services.hpp>

 public:
  /// Build the AMR accepted-service image for a forward topology without exposing the concrete
  /// snapshot implementation to the system transaction carrier.  The returned image is detached
  /// and therefore cannot restore or observe a facade until HiddenPublish rebinds it below.
  [[nodiscard]] static std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>
  detach_accepted_context_for_forward(const AcceptedProgramExecutionServicesSnapshot& accepted,
                                      std::uint64_t topology_epoch,
                                      std::uint64_t materialization_generation,
                                      AmrStorageTopologyAdapter*& rebind_owner) {
    void* opaque_rebind_owner = nullptr;
    auto detached = accepted.detach_for_forward(topology_epoch, materialization_generation,
                                                opaque_rebind_owner);
    rebind_owner = static_cast<AmrStorageTopologyAdapter*>(opaque_rebind_owner);
    if (!detached || rebind_owner == nullptr)
      throw std::logic_error("AMR Program forward regrid has no AMR accepted-service image");
    return detached;
  }

  /// Complete the detached image only after the forward hierarchy has become the live authority.
  /// This is pointer rebinding only; all state and capacity were prepared in Candidate.
  static void rebind_detached_accepted_context_after_publish(
      AcceptedProgramExecutionServicesSnapshot& accepted,
      AmrStorageTopologyAdapter& owner) noexcept {
    accepted.rebind_after_forward_publish(static_cast<void*>(&owner));
  }

  /// Cold-bind the finite interface-flux snapshot storage exactly once.  Transaction refresh and
  /// finalization intentionally have no access to this operation.
  static void prime_accepted_context_at_bind(AcceptedProgramExecutionServicesSnapshot& accepted) {
    accepted.prime_at_bind();
  }

  static void prime_copied_accepted_context_at_bind(
      AcceptedProgramExecutionServicesSnapshot& accepted) {
    accepted.prime_copied_image_at_bind();
  }

  template <int TestDim>
  friend struct detail::AmrProgramHistoryRemapCollectiveTestAccess;
  template <int TestDim>
  friend class ::pops::runtime::program::ProgramExecutionServices;

  facade_type* facade_ = nullptr;
  // A raw restart hierarchy rebuild may replace the facade-owned runtime object.  Rebind only at
  // the accepted hierarchy-refresh boundary; attempt-local execution never observes a change.
  mutable runtime_type* runtime_ = nullptr;
  bool preparation_mode_ = false;
  const PreparedAmrTopologyView* preparation_view_ = nullptr;
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
  /// [plan slot][rhs/state/scalar][subslot][AMR level].  Generated Program execution is bound
  /// exclusively to this finite storage; no generated scratch path may consult ``scratches_``.
  mutable std::vector<std::array<std::vector<std::optional<std::vector<field_type>>>, 3>>
      prepared_scratch_;
  mutable std::mutex coupled_jacvec_mutex_;
  mutable std::unique_ptr<CoupledJacvecScratch> coupled_jacvec_scratch_;
  mutable std::vector<GeneratedFieldRoute> generated_field_routes_;
  std::shared_ptr<const hierarchy_tensor_registry_type> hierarchy_tensor_solver_registry_;
  mutable std::optional<HierarchyTensorSelection> hierarchy_tensor_selection_;
  mutable std::unique_ptr<hierarchy_tensor_solver_type> hierarchy_tensor_solver_;
  mutable std::vector<HierarchyTensorLevelBoundary> hierarchy_tensor_boundaries_;
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
  /// Bind-sealed request/rollback storage used by rhs_group, commit_many and coupling.  The
  /// selected level is addressed by compact indices; no operation below begin_step grows this
  /// carrier or reconstructs a value-id/path map.
  mutable PreparedHotPathWorkspace hot_path_workspace_;
  // Persistent, exact-ranked provenance for values retained by the numeric history rings.  The
  // key is history_key_(name, level), hence this carrier never erases a level or rank boundary.
  // Bases are immutable samples; a lag read clones and rebases them into the current attempt
  // rather than retaining a pointer to a prior attempt's live registry.
  mutable std::map<std::string, std::vector<FluxExpression>> history_flux_expressions_;
  mutable std::map<std::string, AmrProgramPendingHistoryRemap> pending_history_remaps_;
  mutable std::map<std::string, field_type> deferred_history_lag_scratches_;
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
  mutable std::optional<CellTemporalConfiguration> cell_temporal_configuration_;
  mutable std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      cell_temporal_diagnostics_;
  mutable std::int64_t cell_temporal_interval_begin_tick_ = 0;
  mutable std::int64_t cell_temporal_interval_target_tick_ = 0;
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

}  // namespace pops::runtime::program
