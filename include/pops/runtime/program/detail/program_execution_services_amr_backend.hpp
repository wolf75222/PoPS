/// @file
/// @brief Exact compile-time-ranked execution boundary for generated AMR Programs.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
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
#include <pops/runtime/program/detail/program_execution_services_amr_cell_temporal_configuration.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstring>
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

// clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_backend_state.hpp>
// clang-format on
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
  // Keep the common ProgramExecutionServices dispatch independent from the AMR facade's
  // out-of-line private seam.  The AMR installation path supplies this resolver only when it
  // activates an accepted AMR image; Uniform-only users therefore never acquire an AmrSystem
  // link dependency merely by instantiating the shared public service.
  using accepted_runtime_state_resolver_type = runtime_state_type& (*)(facade_type*);
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
  using prepared_multiblock_type =
      ::pops::runtime::amr::PreparedMultiBlockAmrHierarchy<Dim, MemorySpace>;
  using multiblock_level_group_type = typename multiblock_subcycling_type::LevelAdvanceGroup;
  using multiblock_reflux_context_type = typename multiblock_subcycling_type::RefluxContext;
  using multiblock_flux_ledger_type = typename multiblock_subcycling_type::ledger_type;
  using prepared_metric_reflux_workspace_type =
      ::pops::amr::reflux::PreparedMetricRefluxWorkspace<Dim, reflux_payload_type>;
  using interface_flux_ledger_type =
      ::pops::amr::TransactionalInterfaceFluxLedger<AmrProgramFacePayload>;
  using flux_expression_budget_type = typename facade_type::PreparedAmrProgramFluxExpressionBudget;
  using hierarchy_tensor_provider_type = HierarchyTensorSolverProvider<Dim, MemorySpace>;
  using hierarchy_tensor_registry_type = HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>;
  using hierarchy_tensor_solver_type = PreparedHierarchyTensorSolver<Dim, MemorySpace>;
  using hierarchy_tensor_request_type = HierarchyTensorSolverBuildRequest<Dim>;
  struct CellTemporalResidentLevel;

  /// One topology-qualified coarse/fine interface face.  These records are constructed while the
  /// hierarchy image is cold and then retained by the prepared reflux routes; the accepted step
  /// never re-enumerates covered cells or rebuilds a set of interface coordinates.
  struct ProgramInterfaceFace {
    int axis = 0;
    Index<Dim> coarse_face{};
    Index<Dim> coarse_cell{};
    ::pops::amr::reflux::CoarseCellFaceSide side = ::pops::amr::reflux::CoarseCellFaceSide::Lower;
  };

  /// Bounded image-owned topology used while a DSO declares its candidate.  It deliberately
  /// carries no AmrSystem pointer: state prototypes and block routes are captured before the DSO
  /// callback, while runtime/lane are owned by the detached candidate graph until publication.
  struct PreparedAmrTopologyView final {
    runtime_type* runtime = nullptr;
    /// A forward regrid owns only a typed topology view, not an AmrRuntime facade.  Such an image
    /// is explicitly detached and may be used for Candidate bundle construction only.
    bool forward_detached = false;
    ProgramRuntimeState<Dim>* program_state = nullptr;
    const ExecutionLane* lane = nullptr;
    std::shared_ptr<const hierarchy_tensor_registry_type> hierarchy_tensor_registry;
    std::vector<int> program_block_map;
    std::vector<std::vector<field_type>> block_prototypes;
    /// Exact runtime-block boundary-linearization capability captured with the detached state
    /// prototypes.  Generated bindings must not turn a missing facade into a silent `false`.
    std::vector<bool> runtime_block_boundary_linearizations;
    /// Exact per-level geometry copied with the detached field prototypes.  Candidate-time
    /// cell-local provider construction must not consult an accepted facade for this authority.
    std::vector<Geometry<Dim>> level_geometries;
    /// Exact adjacent spatial ratios copied with the detached hierarchy.  A forward tensor
    /// request consumes this value rather than consulting the accepted runtime.
    std::vector<::pops::amr::RefinementRatio<Dim>> spatial_refinement_ratios;
    /// Topology identity paired with the detached level layouts.  Forward preparation has no
    /// AmrRuntime pointer, so every materialization receipt consumes this frozen value.
    std::string spatial_contract;
    double accepted_time = 0.0;
    /// Complete configured hierarchy authority.  `temporal_relations` is only the active
    /// prefix represented by this detached runtime; a cell-local Program must nevertheless
    /// qualify every future transition before its artifact can be accepted.
    std::vector<::pops::amr::ParentChildClockRelation> configured_temporal_relations;
    std::vector<::pops::amr::ParentChildClockRelation> temporal_relations;
    /// Canonical lower/upper entries for axes 0..Dim-1.
    std::vector<bool> periodic_faces;
    std::size_t coupling_count = 0;
    bool has_interface_flux_provider = false;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;

    // The subcycling engine is an accepted-step carrier, but its complete shape is known while
    // the Program installation image is still detached.  These non-owning witnesses are set by
    // the host only after the DSO has supplied its flux declarations and before the image is
    // activated.  They deliberately name the candidate multi-block hierarchy rather than an
    // AmrSystem facade: cold preparation must never rebuild through live accepted authority.
    prepared_multiblock_type* candidate_multiblock = nullptr;
    const typename prepared_multiblock_type::ProgramBlockMap* candidate_program_block_map = nullptr;
    const flux_expression_budget_type* candidate_flux_expression_budget = nullptr;
    const ::pops::amr::InterfaceFluxLedgerBudget* candidate_interface_flux_ledger_budget = nullptr;
    /// The detached Program binding owns one scheduler invocation arena per configured level.
    /// This is a frozen byte receipt, not a facade-derived estimate, and is folded into the
    /// resource plan before that binding becomes accepted.
    std::uint64_t candidate_prepared_coupling_workspace_bytes = 0;
    const AmrProgramAcceptedStateStagingCapacity<Dim>* candidate_accepted_state_staging_capacity =
        nullptr;
    const std::string* candidate_installed_hash = nullptr;

    void validate() const {
      if (program_state == nullptr || lane == nullptr || !lane->active() ||
          hierarchy_tensor_registry == nullptr || program_block_map.empty())
        throw std::invalid_argument("AMR preparation topology view is incomplete");
      if (runtime == nullptr && !forward_detached)
        throw std::invalid_argument("AMR preparation topology view has no runtime authority");
      if (runtime != nullptr &&
          (topology_epoch != runtime->topology_epoch() ||
           materialization_generation != runtime->materialization_generation()))
        throw std::invalid_argument("AMR preparation topology view is stale");
      const std::size_t levels =
          runtime != nullptr ? runtime->hierarchy().num_levels() : level_geometries.size();
      if (levels == 0 || block_prototypes.empty() || level_geometries.size() != levels ||
          spatial_refinement_ratios.size() + 1U != levels || spatial_contract.empty() ||
          runtime_block_boundary_linearizations.size() != block_prototypes.size())
        throw std::invalid_argument("AMR preparation topology view has no level prototypes");
      if (!std::isfinite(accepted_time) || temporal_relations.size() + 1 != levels ||
          periodic_faces.size() != static_cast<std::size_t>(2 * Dim))
        throw std::invalid_argument(
            "AMR preparation topology view has incomplete temporal authority");
      for (std::size_t child = 1; child < levels; ++child) {
        const auto& relation = temporal_relations[child - 1];
        const auto ratio = relation.temporal_ratio();
        if (relation.parent_level() != static_cast<int>(child - 1) ||
            relation.child_level() != static_cast<int>(child) || ratio.numerator <= 0 ||
            ratio.denominator <= 0)
          throw std::invalid_argument(
              "AMR preparation topology view has a non-canonical temporal relation");
      }
      for (std::size_t child = 1; child <= configured_temporal_relations.size(); ++child) {
        const auto& relation = configured_temporal_relations[child - 1];
        const auto ratio = relation.temporal_ratio();
        if (relation.parent_level() != static_cast<int>(child - 1) ||
            relation.child_level() != static_cast<int>(child) || ratio.numerator <= 0 ||
            ratio.denominator <= 0)
          throw std::invalid_argument(
              "AMR preparation topology view has a non-canonical configured temporal relation");
      }
      for (const int runtime_block : program_block_map)
        if (runtime_block < 0 ||
            static_cast<std::size_t>(runtime_block) >= block_prototypes.size() ||
            block_prototypes[static_cast<std::size_t>(runtime_block)].size() != levels)
          throw std::invalid_argument("AMR preparation topology view has an invalid block map");
    }

    void validate_subcycling_authority() const {
      validate();
      if (candidate_multiblock == nullptr || candidate_program_block_map == nullptr ||
          candidate_flux_expression_budget == nullptr ||
          candidate_interface_flux_ledger_budget == nullptr ||
          candidate_accepted_state_staging_capacity == nullptr ||
          candidate_installed_hash == nullptr || candidate_installed_hash->empty())
        throw std::invalid_argument("AMR preparation subcycling authority is incomplete");
      if (std::addressof(candidate_multiblock->topology_runtime()) != runtime ||
          candidate_multiblock->level_count() != runtime->hierarchy().num_levels() ||
          candidate_multiblock->block_count() != program_block_map.size() ||
          candidate_program_block_map->canonical_indices.size() !=
              candidate_multiblock->block_count() ||
          candidate_program_block_map->hierarchy_contract !=
              candidate_multiblock->collective_contract() ||
          candidate_program_block_map->exact_contract.empty())
        throw std::invalid_argument(
            "AMR preparation subcycling authority differs from its candidate hierarchy");
      if (candidate_flux_expression_budget->program_hash != *candidate_installed_hash ||
          candidate_flux_expression_budget->generation != materialization_generation ||
          candidate_flux_expression_budget->exact_contract.empty() ||
          candidate_flux_expression_budget->program_block_map.canonical_indices !=
              candidate_program_block_map->canonical_indices ||
          candidate_flux_expression_budget->program_block_map.hierarchy_contract !=
              candidate_program_block_map->hierarchy_contract ||
          candidate_flux_expression_budget->program_block_map.exact_contract !=
              candidate_program_block_map->exact_contract ||
          candidate_flux_expression_budget->blocks.size() != candidate_multiblock->block_count() ||
          candidate_interface_flux_ledger_budget->exact_contract.empty() ||
          candidate_accepted_state_staging_capacity->checkpoint_byte_capacity == 0 ||
          candidate_accepted_state_staging_capacity->level_count <
              candidate_multiblock->level_count() ||
          candidate_accepted_state_staging_capacity->configured_level_cell_bounds.size() !=
              candidate_accepted_state_staging_capacity->level_count ||
          candidate_accepted_state_staging_capacity->configured_patch_bounds.size() !=
              candidate_accepted_state_staging_capacity->level_count ||
          candidate_accepted_state_staging_capacity->configured_parent_child_pair_bounds.size() +
                  1U !=
              candidate_accepted_state_staging_capacity->level_count ||
          candidate_accepted_state_staging_capacity->configured_route_bounds.size() + 1U !=
              candidate_accepted_state_staging_capacity->level_count ||
          candidate_accepted_state_staging_capacity->configured_event_bounds.size() + 1U !=
              candidate_accepted_state_staging_capacity->level_count ||
          candidate_accepted_state_staging_capacity->configured_forward_storage_counts
                  .level_cell_bounds !=
              candidate_accepted_state_staging_capacity->configured_level_cell_bounds ||
          candidate_accepted_state_staging_capacity->configured_forward_storage_counts
                  .multifab_value_counts.size() !=
              runtime::program::PreparedAmrForwardStorageCounts::multifab_family_count ||
          candidate_accepted_state_staging_capacity->configured_live_subcycling_bytes == 0 ||
          candidate_accepted_state_staging_capacity->configured_forward_snapshot_bytes == 0 ||
          candidate_accepted_state_staging_capacity->configured_rank_bound == 0 ||
          candidate_accepted_state_staging_capacity->configured_subcycling_storage_contract.empty())
        throw std::invalid_argument(
            "AMR preparation subcycling authority has an unauthenticated Program budget");
      if (!candidate_multiblock->lane().active() || candidate_multiblock->lane().identity().empty())
        throw std::invalid_argument("AMR preparation subcycling authority has no active lane");
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

  /// Host-copied, topology-qualified projection of the ABI-v5 two-table flux authority.  All
  /// strings are deliberately consumed while this carrier is built; the execution path selects
  /// only a dense occurrence or final-term index.  The dynamic face payload arena is owned by
  /// the flux service and is addressed by the same global ``basis_slot``.
  struct PreparedFluxTableCarrier final {
    struct Basis final {
      struct Face final {
        int axis = 0;
        Index<Dim> face{};
        Index<Dim> coarse_face{};
        double measure = 0.0;
      };
      struct FaceRoute final {
        multiblock_flux_ledger_type* ledger = nullptr;
        std::int32_t level = -1;
        ::pops::amr::reflux::FaceLedgerRole role = ::pops::amr::reflux::FaceLedgerRole::Coarse;
        std::vector<Face> faces;
      };
      std::uint32_t basis_slot = 0;
      std::uint32_t expression_slot = 0;
      std::uint32_t runtime_block = 0;
      std::int32_t level = -1;
      std::int32_t rhs_identity = -1;
      std::uint8_t provider = 0;
      std::uint32_t components = 0;
      ::pops::amr::Rational stage{0, 1};
      std::vector<FaceRoute> face_routes;
    };
    struct LedgerRoute final {
      multiblock_flux_ledger_type* ledger = nullptr;
      std::int32_t level = -1;
      ::pops::amr::reflux::FaceLedgerRole role = ::pops::amr::reflux::FaceLedgerRole::Coarse;
      /// Fine routes use one fixed slice per child substep of their parent ledger.  Coarse routes
      /// have exactly one slice.  ``slots`` is laid out [substep][face-route index] so warm
      /// execution can select a prebound range without a string or map lookup.
      std::uint32_t substep_count = 1;
      std::vector<std::uint32_t> slots;
    };
    struct Term final {
      std::uint32_t slot = 0;
      std::uint32_t basis_slot = 0;
      std::uint32_t expression_slot = 0;
      ::pops::amr::Rational coefficient{0, 1};
      std::string stage_identity;
      std::vector<LedgerRoute> ledger_routes;
    };

    bool bound = false;
    std::vector<Basis> bases;
    std::vector<Term> terms;
    std::vector<std::vector<std::uint32_t>> basis_slots_by_runtime_block;
    std::vector<std::vector<std::uint32_t>> term_slots_by_runtime_block;
    /// Reset at the beginning of a level group; every emitting RHS consumes one prescribed
    /// occurrence.  This is intentionally not a value-id lookup.
    std::vector<std::size_t> next_basis_by_runtime_block;

    void clear() noexcept {
      bound = false;
      bases.clear();
      terms.clear();
      basis_slots_by_runtime_block.clear();
      term_slots_by_runtime_block.clear();
      next_basis_by_runtime_block.clear();
    }
  };

  /// Bind-sealed pointer packs and per-level rollback images for AMR Program hot paths.  All
  /// vectors and MultiFabs are materialized from the immutable topology/facade view before the
  /// first candidate step; execution only rewrites pointer slots and copies into those images.
  struct PreparedHotPathWorkspace {
    using sum_execution_space = Kokkos::DefaultExecutionSpace;

    /// One cold-sealed metric reflux route per concrete candidate ledger and coarse interface
    /// face.  `query` owns the static owner/state identity and its dynamic attempt/macro fields;
    /// the sealed parent interval remains separately immutable so sibling subcycling routes can
    /// never collapse onto one mutable query.  The ledger pointer intentionally names the
    /// engine-owned resident ledger: same-topology rollback images retain this exact route
    /// authority rather than rediscovering topology or rebuilding face sets.
    struct PreparedMetricRefluxRoute {
      const multiblock_flux_ledger_type* ledger = nullptr;
      std::size_t block = 0;
      std::size_t parent_level = 0;
      ProgramInterfaceFace interface{};
      ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query{};
      ::pops::amr::Rational bound_window_begin{0, 1};
      ::pops::amr::Rational bound_window_end{1, 1};
      ::pops::amr::RefinementRatio<Dim> ratio{};
      ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{};
      ::pops::amr::reflux::MetricRefluxBudget budget{};
      prepared_metric_reflux_workspace_type workspace{};
    };

    bool bound = false;
    std::size_t block_capacity = 0;
    std::size_t level_capacity = 0;
    std::vector<const RhsGroupRequest*> rhs_ordered;
    std::vector<int> rhs_runtime_blocks;
    std::vector<int> rhs_rates;
    std::vector<int> rhs_flux_only;
    std::vector<field_type*> rhs_residuals;
    std::vector<runtime::multiblock::BoundaryEvaluationPoint> rhs_points;
    /// Direct public operations run serially through one resident point.  Grouped RHS uses the
    /// per-request pack above because it retains every point through batch publication.
    runtime::multiblock::BoundaryEvaluationPoint direct_point;
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
    /// Resident authored-coupling carrier.  BoundaryEvaluationPoint and interface publication
    /// own strings, so their capacities are reserved during bind and the step path only assigns
    /// into fixed storage.  The exact witness carries identities plus a zero-initialized binary
    /// scalar record, never a hot ExactContractBuilder.
    runtime::multiblock::BoundaryEvaluationPoint coupling_point;
    runtime::multiblock::InterfaceFluxFragmentPublication coupling_publication;
    std::array<char, 96> coupling_stage_buffer{};
    std::array<char, 96> coupling_stage_denominator_buffer{};
    struct CouplingNumericWitness {
      std::uint64_t topology_epoch = 0;
      std::int64_t tick = 0;
      std::int64_t stage_numerator = 0;
      std::int64_t stage_denominator = 1;
      std::int32_t level = 0;
      std::int32_t substep = 0;
      std::int32_t active_levels = 0;
      double interval_duration = 0.0;
      double evaluation_time = 0.0;
      double dt = 0.0;
    } coupling_numeric{};
    std::array<char, sizeof(CouplingNumericWitness)> coupling_numeric_bytes{};
    std::array<ExactOrderedBytePair, 6> coupling_invocation_pairs{};
    bool coupling_invocation_bound = false;
    std::size_t coupling_identity_capacity = 0;
    /// One HostSpace SUM workspace is shared serially by every local Fab reduction.  It is bound
    /// from the largest exact local prototype before the Program can execute, so a reduction may
    /// not construct Kokkos completion storage or expand its capacity during a transaction.
    PreparedCellSumReduction<sum_execution_space> sum_reduction;
    std::int64_t sum_maximum_points = 0;
    bool sum_reduction_bound = false;
    std::vector<PreparedMetricRefluxRoute> prepared_metric_reflux_routes;

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

    void bind_boundary_point_clock(std::string_view clock) {
      if (!bound)
        throw std::logic_error("AMR Program boundary points bind before their workspace");
      for (auto& point : rhs_points) {
        point.clock.reserve(clock.size());
        point.clock.assign(clock);
      }
      direct_point.clock.reserve(clock.size());
      direct_point.clock.assign(clock);
    }

    void bind_coupling_invocation(std::size_t identity_capacity) {
      if (coupling_invocation_bound) {
        if (coupling_identity_capacity != identity_capacity)
          throw std::logic_error("AMR coupling identity capacity changed after bind");
        return;
      }
      coupling_identity_capacity = identity_capacity;
      coupling_point.clock.reserve(identity_capacity);
      coupling_point.graph_identity.reserve(identity_capacity);
      coupling_point.rate_identity.reserve(identity_capacity);
      coupling_point.application_identity.reserve(identity_capacity);
      coupling_publication.stage_identity.reserve(coupling_stage_buffer.size());
      coupling_invocation_pairs[0].first = "amr-coupling-graph-v1";
      coupling_invocation_pairs[1].first = "amr-coupling-rate-v1";
      coupling_invocation_pairs[2].first = "amr-coupling-application-v1";
      coupling_invocation_pairs[3].first = "amr-coupling-provider-v1";
      coupling_invocation_pairs[4].first = "amr-coupling-stage-v1";
      coupling_invocation_pairs[5].first = "amr-coupling-numeric-v1";
      coupling_invocation_bound = true;
    }

    bool prepare_coupling_invocation(std::string_view graph, std::string_view rate,
                                     std::string_view application,
                                     std::string_view provider_contract, std::string_view clock,
                                     std::uint64_t topology_epoch, int active_levels, int level,
                                     int substep, std::int64_t tick,
                                     ::pops::amr::Rational stage_fraction,
                                     ::pops::amr::Rational interval_phase, double interval_duration,
                                     double evaluation_time, Real dt,
                                     const ::pops::amr::ClockWindow& interval,
                                     runtime::multiblock::InterfaceFluxFragmentLedger* ledger) {
      if (!coupling_invocation_bound || clock.size() > coupling_point.clock.capacity() ||
          graph.size() > coupling_point.graph_identity.capacity() ||
          rate.size() > coupling_point.rate_identity.capacity() ||
          application.size() > coupling_point.application_identity.capacity() ||
          graph.size() > coupling_identity_capacity || rate.size() > coupling_identity_capacity ||
          application.size() > coupling_identity_capacity)
        return false;
      auto [numerator_end, numerator_error] = std::to_chars(
          coupling_stage_buffer.data(), coupling_stage_buffer.data() + coupling_stage_buffer.size(),
          stage_fraction.numerator);
      auto [denominator_end, denominator_error] = std::to_chars(
          coupling_stage_denominator_buffer.data(),
          coupling_stage_denominator_buffer.data() + coupling_stage_denominator_buffer.size(),
          stage_fraction.denominator);
      if (numerator_error != std::errc{} || denominator_error != std::errc{})
        return false;
      constexpr std::string_view prefix = "program-stage:";
      const std::size_t numerator_size =
          static_cast<std::size_t>(numerator_end - coupling_stage_buffer.data());
      const std::size_t denominator_size =
          static_cast<std::size_t>(denominator_end - coupling_stage_denominator_buffer.data());
      const std::size_t stage_identity_size = prefix.size() + numerator_size + 1 + denominator_size;
      if (stage_identity_size > coupling_publication.stage_identity.capacity())
        return false;
      coupling_publication.stage_identity.assign(prefix);
      coupling_publication.stage_identity.append(coupling_stage_buffer.data(), numerator_end);
      coupling_publication.stage_identity.push_back('/');
      coupling_publication.stage_identity.append(coupling_stage_denominator_buffer.data(),
                                                 denominator_end);
      coupling_point.clock.assign(clock);
      coupling_point.tick = tick;
      coupling_point.level = level;
      coupling_point.substep = substep;
      coupling_point.stage = 0;
      coupling_point.stage_fraction = stage_fraction;
      coupling_point.dt = interval_duration;
      coupling_point.physical_time = evaluation_time;
      coupling_point.graph_identity.assign(graph);
      coupling_point.rate_identity.assign(rate);
      coupling_point.application_identity.assign(application);
      coupling_publication.ledger = ledger;
      coupling_publication.topology_epoch = topology_epoch;
      coupling_publication.active_level_count = active_levels;
      coupling_publication.clock = {level, tick, interval_phase, evaluation_time};
      coupling_publication.interval = interval;
      coupling_publication.stage_weight = exact_binary_rational_(dt / interval_duration);
      coupling_publication.stage_weight_resolved = true;
      coupling_numeric = {topology_epoch,
                          tick,
                          stage_fraction.numerator,
                          stage_fraction.denominator,
                          level,
                          substep,
                          active_levels,
                          interval_duration,
                          evaluation_time,
                          static_cast<double>(dt)};
      std::memcpy(coupling_numeric_bytes.data(), &coupling_numeric, sizeof(coupling_numeric));
      coupling_invocation_pairs[0].second = coupling_point.graph_identity;
      coupling_invocation_pairs[1].second = coupling_point.rate_identity;
      coupling_invocation_pairs[2].second = coupling_point.application_identity;
      coupling_invocation_pairs[3].second = provider_contract;
      coupling_invocation_pairs[4].second = coupling_publication.stage_identity;
      coupling_invocation_pairs[5].second =
          std::string_view(coupling_numeric_bytes.data(), coupling_numeric_bytes.size());
      return true;
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

    template <class Getter>
    void bind_sum_reduction(std::size_t blocks, std::size_t levels, Getter&& getter) {
      if (blocks == 0 || levels == 0)
        throw std::invalid_argument("AMR Program SUM workspace has no prepared prototypes");
      std::int64_t maximum_points = 0;
      for (std::size_t level = 0; level < levels; ++level)
        for (std::size_t block = 0; block < blocks; ++block) {
          const field_type& prototype = getter(block, level);
          for (std::size_t local = 0; local < prototype.local_size(); ++local)
            maximum_points = std::max(maximum_points, prototype.box(local).numPts());
        }
      if (sum_reduction_bound) {
        if (sum_maximum_points != maximum_points)
          throw std::logic_error("AMR Program SUM workspace capacity changed after preparation");
        return;
      }
      sum_maximum_points = maximum_points;
      sum_reduction_bound = true;
      // An empty rank never dispatches a local reduction.  The prepared primitive owns both the
      // device reduction and host fold storage, so every configured execution space binds the
      // same finite capacity before begin_step().
      if (maximum_points > 0)
        sum_reduction.prepare(::pops::detail::default_execution_space(), maximum_points);
    }

    void require_sum_reduction(const char* operation) const {
      if (!sum_reduction_bound)
        throw std::logic_error(std::string(operation) +
                               " requires a bind-sealed prepared AMR SUM workspace");
    }

    /// Cold-copy companion for resident transaction images.  Standard string copies may retain
    /// only their current size even when the accepted carrier reserved a larger authenticated
    /// identity envelope.  Restore that finite capacity before the image can participate in a
    /// rollback; candidate-time capture is deliberately forbidden from calling this allocator.
    void prime_copied_capacities_from_cold_source(const PreparedHotPathWorkspace& source) {
      if (bound != source.bound || block_capacity != source.block_capacity ||
          level_capacity != source.level_capacity ||
          rhs_points.capacity() < source.rhs_points.capacity() ||
          coupling_invocation_bound != source.coupling_invocation_bound ||
          coupling_identity_capacity != source.coupling_identity_capacity)
        throw std::logic_error("AMR Program hot workspace cold copy changed its shape");
      const auto prime_string = [](std::string& destination, const std::string& input) {
        if (destination.capacity() < input.capacity())
          destination.reserve(input.capacity());
      };
      const auto prime_point = [&](runtime::multiblock::BoundaryEvaluationPoint& destination,
                                   const runtime::multiblock::BoundaryEvaluationPoint& input) {
        prime_string(destination.clock, input.clock);
        prime_string(destination.graph_identity, input.graph_identity);
        prime_string(destination.rate_identity, input.rate_identity);
        prime_string(destination.application_identity, input.application_identity);
      };
      // Grouped RHS keeps the complete bound point pack resident. Shrinking this vector would
      // destroy the prepared string capacities and make a later wider group allocate or observe
      // an empty clock on the hot path.
      if (rhs_points.size() != block_capacity || source.rhs_points.size() != block_capacity)
        throw std::logic_error("AMR Program hot workspace point pack changed after bind");
      for (std::size_t index = 0; index < rhs_points.size(); ++index)
        prime_point(rhs_points[index], source.rhs_points[index]);
      prime_point(direct_point, source.direct_point);
      prime_point(coupling_point, source.coupling_point);
      prime_string(coupling_publication.stage_identity, source.coupling_publication.stage_identity);
      if (prepared_metric_reflux_routes.size() != source.prepared_metric_reflux_routes.size())
        throw std::logic_error("AMR Program cold copy changed its prepared metric reflux routes");
      if (prepared_metric_reflux_routes.capacity() <
          source.prepared_metric_reflux_routes.capacity())
        prepared_metric_reflux_routes.reserve(source.prepared_metric_reflux_routes.capacity());
      for (std::size_t index = 0; index < prepared_metric_reflux_routes.size(); ++index) {
        auto& destination = prepared_metric_reflux_routes[index];
        const auto& input = source.prepared_metric_reflux_routes[index];
        if (destination.ledger != input.ledger || destination.block != input.block ||
            destination.parent_level != input.parent_level ||
            destination.interface.axis != input.interface.axis ||
            destination.interface.coarse_face != input.interface.coarse_face ||
            destination.interface.coarse_cell != input.interface.coarse_cell ||
            destination.interface.side != input.interface.side ||
            destination.bound_window_begin != input.bound_window_begin ||
            destination.bound_window_end != input.bound_window_end ||
            destination.ratio != input.ratio || destination.mapping != input.mapping ||
            destination.budget.max_fine_faces != input.budget.max_fine_faces ||
            destination.budget.max_published_entries != input.budget.max_published_entries ||
            destination.budget.max_clock_stage_slices != input.budget.max_clock_stage_slices)
          throw std::logic_error("AMR Program cold copy changed metric reflux route identity");
        prime_string(destination.query.owner, input.query.owner);
        prime_string(destination.query.state, input.query.state);
        destination.workspace.prime_copied_capacities_from_cold_source(input.workspace);
      }
      coupling_invocation_pairs[0].second = coupling_point.graph_identity;
      coupling_invocation_pairs[1].second = coupling_point.rate_identity;
      coupling_invocation_pairs[2].second = coupling_point.application_identity;
      coupling_invocation_pairs[4].second = coupling_publication.stage_identity;
      coupling_invocation_pairs[5].second =
          std::string_view(coupling_numeric_bytes.data(), coupling_numeric_bytes.size());
    }

    void require_copied_capacity_for(const PreparedHotPathWorkspace& source) const {
      if (bound != source.bound || block_capacity != source.block_capacity ||
          level_capacity != source.level_capacity ||
          rhs_points.capacity() < source.rhs_points.capacity() ||
          rhs_points.size() != block_capacity || source.rhs_points.size() != block_capacity ||
          coupling_invocation_bound != source.coupling_invocation_bound ||
          coupling_identity_capacity != source.coupling_identity_capacity)
        throw std::logic_error("AMR Program hot workspace changed after cold bind");
      const auto require_string = [](const std::string& destination, const std::string& input) {
        if (destination.capacity() < input.capacity())
          throw std::logic_error("AMR Program hot workspace string capacity was not primed");
      };
      const auto require_point =
          [&](const runtime::multiblock::BoundaryEvaluationPoint& destination,
              const runtime::multiblock::BoundaryEvaluationPoint& input) {
            require_string(destination.clock, input.clock);
            require_string(destination.graph_identity, input.graph_identity);
            require_string(destination.rate_identity, input.rate_identity);
            require_string(destination.application_identity, input.application_identity);
          };
      for (std::size_t index = 0; index < rhs_points.size(); ++index)
        require_point(rhs_points[index], source.rhs_points[index]);
      require_point(direct_point, source.direct_point);
      require_point(coupling_point, source.coupling_point);
      require_string(coupling_publication.stage_identity,
                     source.coupling_publication.stage_identity);
      if (prepared_metric_reflux_routes.size() != source.prepared_metric_reflux_routes.size() ||
          prepared_metric_reflux_routes.capacity() <
              source.prepared_metric_reflux_routes.capacity())
        throw std::logic_error(
            "AMR Program hot workspace metric reflux routes were not primed (destination " +
            std::to_string(prepared_metric_reflux_routes.size()) + "/" +
            std::to_string(prepared_metric_reflux_routes.capacity()) + ", source " +
            std::to_string(source.prepared_metric_reflux_routes.size()) + "/" +
            std::to_string(source.prepared_metric_reflux_routes.capacity()) + ")");
      for (std::size_t index = 0; index < prepared_metric_reflux_routes.size(); ++index) {
        const auto& destination = prepared_metric_reflux_routes[index];
        const auto& input = source.prepared_metric_reflux_routes[index];
        if (destination.ledger != input.ledger || destination.block != input.block ||
            destination.parent_level != input.parent_level ||
            destination.interface.axis != input.interface.axis ||
            destination.interface.coarse_face != input.interface.coarse_face ||
            destination.interface.coarse_cell != input.interface.coarse_cell ||
            destination.interface.side != input.interface.side ||
            destination.bound_window_begin != input.bound_window_begin ||
            destination.bound_window_end != input.bound_window_end ||
            destination.ratio != input.ratio || destination.mapping != input.mapping ||
            destination.query.owner != input.query.owner ||
            destination.query.state != input.query.state ||
            destination.query.owner.capacity() < input.query.owner.capacity() ||
            destination.query.state.capacity() < input.query.state.capacity())
          throw std::logic_error(
              "AMR Program hot workspace metric reflux route changed after bind");
        destination.workspace.require_preallocated_copy_from(input.workspace);
      }
    }

    /// Exact heap payload retained by the AMR hot workspace.  The carrier owns pointer packs,
    /// scalar publications and complete rollback field images; only MultiFab can report the
    /// latter's true local/ghost payload.  The reduction arena is a separate manifest family.
    [[nodiscard]] std::uint64_t resident_storage_bytes() const {
      const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error("AMR Program hot-path resident storage overflows uint64");
        total += value;
      };
      const auto vector_bytes = [](const auto& values) -> std::uint64_t {
        using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
        if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
          throw std::overflow_error("AMR Program hot-path vector storage overflows uint64");
        return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
      };
      const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
        const auto object_begin = reinterpret_cast<std::uintptr_t>(&value);
        const auto object_end = object_begin + sizeof(value);
        const auto data = reinterpret_cast<std::uintptr_t>(value.data());
        return data >= object_begin && data < object_end
                   ? 0
                   : static_cast<std::uint64_t>(value.capacity()) + 1U;
      };
      std::uint64_t total = 0;
      checked_add(total, vector_bytes(rhs_ordered));
      checked_add(total, vector_bytes(rhs_runtime_blocks));
      checked_add(total, vector_bytes(rhs_rates));
      checked_add(total, vector_bytes(rhs_flux_only));
      checked_add(total, vector_bytes(rhs_residuals));
      checked_add(total, vector_bytes(rhs_points));
      for (const auto& point : rhs_points)
        checked_add(total, external_string_bytes(point.clock));
      checked_add(total, external_string_bytes(direct_point.clock));
      checked_add(total, vector_bytes(rhs_evaluation_targets));
      checked_add(total, vector_bytes(rhs_staged_parents));
      checked_add(total, vector_bytes(rhs_evaluations));
      checked_add(total, vector_bytes(rhs_candidates));
      checked_add(total, vector_bytes(rhs_backups));
      checked_add(total, vector_bytes(coupling_states));
      checked_add(total, vector_bytes(commit_targets));
      checked_add(total, vector_bytes(commit_sources));
      checked_add(total, vector_bytes(commit_runtime_blocks));
      checked_add(total, vector_bytes(commit_snapshots));
      checked_add(total, vector_bytes(accepted_snapshot_by_runtime));
      checked_add(total, vector_bytes(publication_candidates));
      checked_add(total, vector_bytes(publication_program_candidates));
      checked_add(total, vector_bytes(candidate_storage));
      checked_add(total, vector_bytes(backup_storage));
      checked_add(total, vector_bytes(commit_storage));
      for (const auto* fields :
           {&publication_candidates, &candidate_storage, &backup_storage, &commit_storage})
        for (const field_type& field : *fields)
          checked_add(total, field.resident_storage_bytes());
      checked_add(total, external_string_bytes(coupling_point.clock));
      checked_add(total, external_string_bytes(coupling_point.graph_identity));
      checked_add(total, external_string_bytes(coupling_point.rate_identity));
      checked_add(total, external_string_bytes(coupling_point.application_identity));
      checked_add(total, external_string_bytes(coupling_publication.stage_identity));
      checked_add(total, vector_bytes(prepared_metric_reflux_routes));
      for (const auto& route : prepared_metric_reflux_routes) {
        checked_add(total, external_string_bytes(route.query.owner));
        checked_add(total, external_string_bytes(route.query.state));
        checked_add(total, route.workspace.resident_storage_bytes());
      }
      return total;
    }
  };

#include <pops/runtime/program/detail/program_execution_services_amr_flux_basis_definitions.hpp>

  struct PreparedSubcyclingBundle final {
    std::unique_ptr<multiblock_subcycling_type> engine;
    /// Geometry is part of the detached topology authority.  Static history images retain face
    /// measures, so their cold templates must never reconstruct metrics through an accepted
    /// facade after a forward regrid has staged a successor hierarchy.
    std::vector<Geometry<Dim>> level_geometries;
    PreparedFluxTableCarrier flux_tables;
    std::vector<FluxBasis> flux_basis_payloads;
    std::vector<std::uint8_t> flux_basis_active;
    std::string flux_collective_contract;
    std::vector<std::size_t> rhs_basis_bounds;
    std::vector<std::size_t> coefficient_term_bounds;
    std::string program_budget_contract;
    std::unique_ptr<interface_flux_ledger_type> interface_ledger;
    PreparedHotPathWorkspace hot_path_workspace;
    std::vector<field_type*> active_attempt_states;
    std::vector<const field_type*> active_staged_parents;
    std::vector<multiblock_flux_ledger_type*> active_incoming_flux;
    std::vector<multiblock_flux_ledger_type*> active_outgoing_flux;
    std::vector<std::string_view> active_block_identities;
    std::vector<std::size_t> active_flux_basis_counts;
    std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
    std::size_t block_count = 0;

    /// Exact detached forward carrier footprint.  This deliberately walks the dynamic route and
    /// payload trees: sizeof-only accounting would miss face lists, ledger slot slices and flux
    /// density vectors retained simultaneously with the accepted A image during publication.
    [[nodiscard]] std::uint64_t resident_storage_bytes() const {
      const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error("AMR Program prepared subcycling bundle overflows uint64");
        total += value;
      };
      const auto vector_bytes = [](const auto& values) -> std::uint64_t {
        using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
        if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
          throw std::overflow_error(
              "AMR Program prepared subcycling bundle vector overflows uint64");
        return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
      };
      const auto string_bytes = [](const std::string& value) {
        return ::pops::amr::reflux::detail::external_string_storage_bytes(value);
      };
      // The bundle value is inline in AcceptedContextSnapshot's optional and is already charged
      // by sizeof(AcceptedContextSnapshot); this method reports only its external ownership.
      std::uint64_t total = 0;
      if (engine) {
        checked_add(total, sizeof(*engine));
        checked_add(total, engine->resident_storage_bytes());
      }
      if (interface_ledger) {
        checked_add(total, sizeof(*interface_ledger));
        checked_add(total, interface_ledger->resident_storage_bytes());
      }
      checked_add(total, vector_bytes(level_geometries));
      checked_add(total, vector_bytes(flux_tables.bases));
      for (const auto& basis : flux_tables.bases) {
        checked_add(total, vector_bytes(basis.face_routes));
        for (const auto& route : basis.face_routes)
          checked_add(total, vector_bytes(route.faces));
      }
      checked_add(total, vector_bytes(flux_tables.terms));
      for (const auto& term : flux_tables.terms) {
        checked_add(total, string_bytes(term.stage_identity));
        checked_add(total, vector_bytes(term.ledger_routes));
        for (const auto& route : term.ledger_routes)
          checked_add(total, vector_bytes(route.slots));
      }
      for (const auto* index :
           {&flux_tables.basis_slots_by_runtime_block, &flux_tables.term_slots_by_runtime_block}) {
        checked_add(total, vector_bytes(*index));
        for (const auto& slots : *index)
          checked_add(total, vector_bytes(slots));
      }
      checked_add(total, vector_bytes(flux_tables.next_basis_by_runtime_block));
      checked_add(total, vector_bytes(flux_basis_payloads));
      for (const auto& basis : flux_basis_payloads) {
        checked_add(total, string_bytes(basis.point.clock));
        checked_add(total, string_bytes(basis.point.graph_identity));
        checked_add(total, string_bytes(basis.point.rate_identity));
        checked_add(total, string_bytes(basis.point.application_identity));
        checked_add(total, vector_bytes(basis.faces));
        for (const auto& face : basis.faces)
          checked_add(total, vector_bytes(face.flux_density));
      }
      checked_add(total, vector_bytes(flux_basis_active));
      checked_add(total, string_bytes(flux_collective_contract));
      checked_add(total, vector_bytes(rhs_basis_bounds));
      checked_add(total, vector_bytes(coefficient_term_bounds));
      checked_add(total, string_bytes(program_budget_contract));
      checked_add(total, hot_path_workspace.resident_storage_bytes());
      checked_add(total, vector_bytes(active_attempt_states));
      checked_add(total, vector_bytes(active_staged_parents));
      checked_add(total, vector_bytes(active_incoming_flux));
      checked_add(total, vector_bytes(active_outgoing_flux));
      checked_add(total, vector_bytes(active_block_identities));
      checked_add(total, vector_bytes(active_flux_basis_counts));
      return total;
    }
  };

  // clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_backend_preparation.hpp>
  // clang-format on
  facade_type* facade_ = nullptr;
  accepted_runtime_state_resolver_type accepted_runtime_state_resolver_ = nullptr;
  // Resolve the facade-owned Program state exactly once at accepted activation.  The Impl-owned
  // object has stable storage across AmrSystem moves and artifact replacements, whereas the
  // public facade wrapper may move before a pending transaction is rolled back during teardown.
  // Hot execution and no-throw restore therefore use this direct accepted authority only.
  runtime_state_type* accepted_runtime_state_ = nullptr;
  // A raw restart hierarchy rebuild may replace the facade-owned runtime object.  Rebind only at
  // the accepted hierarchy-refresh boundary; attempt-local execution never observes a change.
  mutable runtime_type* runtime_ = nullptr;
  bool preparation_mode_ = false;
  const PreparedAmrTopologyView* preparation_view_ = nullptr;
  // Narrow friend-only seam for the forward-ceiling regression.  It is copied into the detached
  // Candidate image and never participates in an installed artifact or a publication path.
  std::optional<std::uint64_t> test_forward_storage_ceiling_override_;
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
  mutable PreparedScratchStorage prepared_scratch_;
  /// Exact owner/component/ghost declaration for each dense scratch subslot. It mirrors the
  /// sealed resource plan and is the only authority used to rebuild forward arenas.
  mutable PreparedScratchDescriptors prepared_scratch_descriptors_;
  mutable std::mutex coupled_jacvec_mutex_;
  mutable std::unique_ptr<CoupledJacvecScratch> coupled_jacvec_scratch_;
  mutable std::vector<GeneratedFieldRoute> generated_field_routes_;
  /// Cold-bound Program-to-runtime indices for projection/CFL callbacks.  Accepted execution
  /// may only read these dense slots; it must not re-enter the facade map or rebuild consensus.
  mutable std::vector<int> projection_speed_routes_;
  mutable bool projection_speed_routes_bound_ = false;
  /// Each Program image owns a mutable clone of the host registry. DSO-defined providers are
  /// retained by that image/solver and are never published into the global AmrSystem registry.
  std::shared_ptr<hierarchy_tensor_registry_type> hierarchy_tensor_solver_registry_;
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
  /// v5 two-table authority compacted at bind.  Keep it adjacent to the legacy expression
  /// registry while migration callers still use that registry for unsupported history paths;
  /// the active two-table route selects only dense slots from this carrier.
  mutable PreparedFluxTableCarrier static_flux_tables_;
  /// Cold-prepared exact route witness checked before a static group can enter any per-face
  /// collective.  Keeping the bytes resident avoids constructing a contract in the hot path.
  mutable std::string static_flux_collective_contract_;
  /// One resident payload image per global ABI basis slot.  Face vectors and their component
  /// payloads are reserved during cold binding; hot RHS attachment overwrites only values.
  mutable std::vector<FluxBasis> static_flux_basis_payloads_;
  mutable std::vector<std::uint8_t> static_flux_basis_active_;
  /// Bind-sealed request/rollback storage used by rhs_group, commit_many and coupling.  The
  /// selected level is addressed by compact indices; no operation below begin_step grows this
  /// carrier or reconstructs a value-id/path map.
  mutable PreparedHotPathWorkspace hot_path_workspace_;
  /// Bind-reserved POPSAND5 candidate image.  The facade retains the independently owned accepted
  /// bytes; this buffer remains untouched until serialization and rank agreement have succeeded.
  mutable std::vector<std::uint8_t> accepted_checkpoint_candidate_bytes_;
  /// The sole accepted-step state assembly carrier.  It is cold-primed after bind/regrid and
  /// never rebuilt from `accepted_state_()` while a numerical attempt is hot.
  mutable AcceptedStateStaging<Dim> accepted_state_staging_;
  mutable std::shared_ptr<const AmrProgramAcceptedStateStagingCapacity<Dim>>
      accepted_forward_storage_capacity_;
  mutable std::vector<typename AcceptedStateStaging<Dim>::InterfaceFluxSerializationView>
      accepted_interface_flux_staging_sources_;
  /// Reused only while assembling the candidate POPSAND5 state; its vector storage is exchanged
  /// into the transient state object and returned by an RAII guard after serialization/publication.
  mutable std::vector<::pops::amr::ClockStamp> accepted_checkpoint_level_clock_slots_;
  // Persistent, exact-ranked provenance for values retained by the numeric history rings.  The
  // key is history_key_(name, level), hence this carrier never erases a level or rank boundary.
  // Bases are immutable samples; a lag read clones and rebases them into the current attempt
  // rather than retaining a pointer to a prior attempt's live registry.
  mutable std::map<std::string, std::vector<FluxExpression>> history_flux_expressions_;
  mutable std::map<std::string, AmrProgramPendingHistoryRemap> pending_history_remaps_;
  mutable std::map<std::string, field_type> deferred_history_lag_scratches_;
  /// Cold-bound mutation carrier for exact-ranked history rings.  Store/rotate only copy through
  /// these resident images: names, field rollback slots, provenance terms and collective bytes
  /// are all allocated before the first candidate attempt.
  mutable std::vector<PreparedHistoryMutationSlot> prepared_history_mutation_slots_;
  /// Dense staging ordinals are deliberately adapter-owned: raw references into a map must not
  /// be copied with an accepted snapshot.  Cold rebind authenticates every ordinal before a hot
  /// checkpoint refresh may dereference it.
  mutable std::vector<std::size_t> accepted_history_binding_mutation_slots_;
  mutable std::vector<const AmrProgramPendingHistoryRemap*>
      accepted_pending_history_ordinal_sources_;
  mutable const void* accepted_history_ordinal_owner_ = nullptr;
  mutable std::uint64_t accepted_history_ordinal_epoch_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t accepted_history_ordinal_generation_ =
      std::numeric_limits<std::uint64_t>::max();
  /// Cold-built route to wire-slot mappings.  Ledger vectors are never stored in the staging
  /// value object, so copying an accepted image cannot retain a dangling route pointer.
  mutable std::array<std::vector<AcceptedFaceFluxOrdinal>, Dim> accepted_face_flux_ordinals_;
  mutable const void* accepted_face_flux_ordinal_owner_ = nullptr;
  mutable std::uint64_t accepted_face_flux_ordinal_epoch_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t accepted_face_flux_ordinal_generation_ =
      std::numeric_limits<std::uint64_t>::max();
  /// Interface fragments are held in a dense ledger rather than individual stable Entry objects.
  /// The cold carrier seals its finite source-to-wire ordinal permutation; hot serialization may
  /// only consume that permutation after validating the complete canonical fragment ordering.
  mutable std::vector<std::size_t> accepted_interface_flux_wire_ordinals_;
  mutable const void* accepted_interface_flux_ordinal_owner_ = nullptr;
  mutable std::uint64_t accepted_interface_flux_ordinal_epoch_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t accepted_interface_flux_ordinal_generation_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable std::string prepared_history_rotation_contract_;
  mutable std::uint64_t prepared_history_mutation_epoch_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t prepared_history_mutation_generation_ =
      std::numeric_limits<std::uint64_t>::max();
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
  mutable std::size_t multiblock_subcycling_block_count_ = 0;
  mutable std::string multiblock_subcycling_program_budget_contract_;
  /// Clear the per-group static-flux image without destroying its cold-reserved face/payload
  /// vectors.  Inactive bytes are never consumed, but resetting every scalar/cursor keeps retry
  /// and rollback observations exact while preserving the no-allocation resident envelope.
  void reset_static_flux_active_state_() const noexcept {
    if (!static_flux_tables_.bound)
      return;
    std::fill(static_flux_basis_active_.begin(), static_flux_basis_active_.end(), std::uint8_t{0});
    std::fill(static_flux_tables_.next_basis_by_runtime_block.begin(),
              static_flux_tables_.next_basis_by_runtime_block.end(), std::size_t{0});
    for (FluxBasis& payload : static_flux_basis_payloads_) {
      payload.identity = 0;
      payload.runtime_block = 0;
      payload.level = 0;
      payload.point.tick = 0;
      payload.point.level = 0;
      payload.point.substep = 0;
      payload.point.stage = 0;
      payload.point.stage_fraction = {0, 1};
      payload.point.dt = 0.0;
      payload.point.physical_time = 0.0;
      payload.point.graph_identity.clear();
      payload.point.rate_identity.clear();
      payload.point.application_identity.clear();
      payload.rhs_identity = -1;
      payload.provider = FluxBasisProvider::PreparedResidual;
      payload.window = {};
      payload.face_count = 0;
    }
  }
  static void copy_cell_temporal_diagnostics_noexcept_(
      const std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>& target,
      const std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>&
          source) noexcept {
    if (target.size() != source.size())
      std::terminate();
    for (std::size_t slot = 0; slot < source.size(); ++slot) {
      if (!target[slot] || !source[slot])
        std::terminate();
      target[slot]->copy_accepted_into_preallocated_noexcept(*source[slot]);
    }
  }

  mutable CellTemporalPartitionAcceptedState accepted_temporal_partition_;
  mutable std::optional<CellTemporalConfiguration> cell_temporal_configuration_;
  mutable std::vector<std::shared_ptr<CellTemporalResidentLevel>> cell_temporal_resident_levels_;
  mutable std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      cell_temporal_diagnostic_workspace_;
  mutable std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      cell_temporal_diagnostics_;
  mutable std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      cell_temporal_diagnostic_rollback_;
  mutable std::int64_t cell_temporal_interval_begin_tick_ = 0;
  mutable std::int64_t cell_temporal_interval_target_tick_ = 0;
  // Set only around the resident cell-local FE callback.  This is a value-only execution mode:
  // its fixed provider diagnostic arena is the sole flux metadata authority, so generic symbolic
  // flux-map materialization is forbidden on that hot path.
  mutable bool active_cell_temporal_execution_ = false;
  mutable std::string accepted_flux_budget_contract_;
  mutable std::string accepted_coupling_contract_;
  mutable std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>
      accepted_face_flux_;
  /// Cold-resident nested envelopes used to copy a validated staging image into the logical
  /// accepted publication without constructing strings or payload vectors after consensus.
  mutable std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>
      accepted_face_flux_commit_slots_;
  mutable std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger_;
  mutable std::optional<typename interface_flux_ledger_type::PreparedCommit>
      interface_flux_commit_guard_;
  mutable std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events_;
  mutable std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_event_commit_slots_;
  mutable std::uint64_t accepted_state_revision_ = std::numeric_limits<std::uint64_t>::max();
};

}  // namespace pops::runtime::program
