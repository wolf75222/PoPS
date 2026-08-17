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

namespace detail {
template <int Dim>
struct AmrProgramHistoryRemapCollectiveTestAccess;
}

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
  using tensor_boundary_session_type = PreparedTensorBoundarySession<Dim>;
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

  // Class-scope responsibility fragments preserve the public nested-type identities and member
  // layout of AmrProgramContext while making each semantic authority independently auditable.
#include <pops/runtime/program/amr_program_context_spatial.inc>
#include <pops/runtime/program/amr_program_context_field_runtime_public.inc>
#include <pops/runtime/program/amr_program_context_flux_expression_public.inc>
#include <pops/runtime/program/amr_program_context_spatial_operations.inc>
#include <pops/runtime/program/amr_program_context_history_checkpoint_public.inc>
#include <pops/runtime/program/amr_program_context_field_runtime_solver.inc>
#include <pops/runtime/program/amr_program_context_field_runtime_private.inc>
#include <pops/runtime/program/amr_program_context_flux_expression_polynomial.inc>
#include <pops/runtime/program/amr_program_context_cell_temporal_configuration.inc>
#include <pops/runtime/program/amr_program_context_flux_basis_definitions.inc>
#include <pops/runtime/program/amr_program_context_flux_expression_definitions.inc>
#include <pops/runtime/program/amr_program_context_history_checkpoint_definitions.inc>
#include <pops/runtime/program/amr_program_context_cell_temporal_level_runtime.inc>
#include <pops/runtime/program/amr_program_context_field_runtime_definitions.inc>
#include <pops/runtime/program/amr_program_context_flux_expression_services.inc>
#include <pops/runtime/program/amr_program_context_cell_temporal_runtime.inc>
#include <pops/runtime/program/amr_program_context_subcycling_runtime.inc>
#include <pops/runtime/program/amr_program_context_flux_basis.inc>
#include <pops/runtime/program/amr_program_context_flux_expression_runtime.inc>
#include <pops/runtime/program/amr_program_context_history_checkpoint_runtime.inc>
#include <pops/runtime/program/amr_program_context_field_runtime_services.inc>
#include <pops/runtime/program/amr_program_context_history_checkpoint_services.inc>
#include <pops/runtime/program/amr_program_context_spatial_operations_services.inc>

  template <int TestDim>
  friend struct detail::AmrProgramHistoryRemapCollectiveTestAccess;

  facade_type* facade_ = nullptr;
  // A raw restart hierarchy rebuild may replace the facade-owned runtime object.  Rebind only at
  // the accepted hierarchy-refresh boundary; attempt-local execution never observes a change.
  mutable runtime_type* runtime_ = nullptr;
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
