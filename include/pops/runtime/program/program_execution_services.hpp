#pragma once

#include <algorithm>
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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/program/cache_manager.hpp>
#include <pops/runtime/context/grid_context.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/program/wire_ids.hpp>

namespace pops::runtime::program {

/// Backend-independent Program operations shared by every execution topology.
///
/// The provider owns topology, storage and explicitly qualified non-Cartesian stencil capabilities
/// only. It exposes the narrow ``program_execution_*`` hooks below; generated Program operations
/// themselves live here exactly once. This is deliberately CRTP instead of a virtual facade: a
/// generated artifact still calls the concrete context directly, and the compiler resolves each
/// provider hook without a second runtime dispatch table.
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

 protected:
  struct RhsGroupBatch {
    int group_id = -1;
    // Borrowed only for the synchronous provider call made before rhs_group returns.
    std::initializer_list<RhsGroupRequest> requests;
    std::vector<int> runtime_blocks;
    std::vector<MultiFab*> states;
    std::vector<MultiFab*> rhs;
    std::vector<int> rate_ids;
    std::vector<int> flux_only;
  };

 public:
  enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
  enum class SchedulerCacheOperation : std::uint8_t {
    ShouldUpdate,
    StoreAux,
    RestoreAux,
    StoreScratch,
    RestoreScratch,
    AccumulateDt,
    EffectiveDt,
  };
  enum class HistoryReadMode : std::uint8_t { RequireInitialized, ZeroStart };
  enum class HistoryRotationAction : std::uint8_t { Skip, Rotate, Defer };

  struct HistoryRegistration {
    std::string name;
    int lag = 1;
    int ncomp = -1;
    int program_owner = -1;
    int runtime_owner = -1;
    std::string state_identity;
    std::string space_identity;
    std::string clock_identity;
    std::string interpolation_identity;
    bool qualified = false;
  };

  struct HistoryStorePlan {
    bool due = true;
    std::optional<Real> outgoing_dt;
  };

  struct ProgramResourceStorage {
    const PreparedVectorDistribution& vector_distribution;
    FieldDistribution field_distribution;
    int field_level;
  };

  /// Immutable identity of the hierarchy on which generated persistent resources are materialized.
  ///
  /// ``epoch`` is checkpointed topology identity, while ``generation`` also changes after any
  /// process-local storage rebuild.  ``levels`` is captured in the same provider read so generated
  /// code cannot compare an epoch from one hierarchy against a level count from another.
  struct ProgramResourceTopology {
    std::uint64_t epoch = 0;
    std::uint64_t generation = 0;
    int levels = 1;
  };

 protected:
  /// Scope one mutable prepared workspace to a single synchronous Program operation.
  ///
  /// Uniform and AMR providers own different workspace storage, but they share the same
  /// fail-before-mutation and release-on-exit policy.  Keeping that policy here prevents a provider
  /// from silently forgetting the exceptional-exit release path.
  class ExclusiveUseGuard {
   public:
    ExclusiveUseGuard(bool& in_use, std::string_view conflict_message) : in_use_(&in_use) {
      if (in_use)
        throw std::logic_error(std::string(conflict_message));
      in_use = true;
    }
    ExclusiveUseGuard(const ExclusiveUseGuard&) = delete;
    ExclusiveUseGuard& operator=(const ExclusiveUseGuard&) = delete;
    ExclusiveUseGuard(ExclusiveUseGuard&&) = delete;
    ExclusiveUseGuard& operator=(ExclusiveUseGuard&&) = delete;
    ~ExclusiveUseGuard() noexcept { *in_use_ = false; }

   private:
    bool* in_use_;
  };

  /// Exact distributed storage/layout compatibility used by every prepared Program workspace.
  static bool field_layout_matches_(const MultiFab& field, const MultiFab& prototype, int n_comp,
                                    int n_ghost) {
    return field.box_array().boxes() == prototype.box_array().boxes() &&
           field.dmap().ranks() == prototype.dmap().ranks() && field.ncomp() == n_comp &&
           field.n_grow() == n_ghost;
  }

 public:
  /// Install one compiled macro-step through the execution provider's lifecycle authority.
  void install(std::function<void(double)> step) const {
    provider_().program_execution_install_(std::move(step));
  }

  /// Field-solve operations are authored once for every Program execution topology.  Providers own
  /// only the storage/publication transaction and hierarchy semantics behind these hooks.
  SolveOutcome solve_fields() const {
    return provider_().program_execution_solve_fields_outcome_();
  }

  SolveOutcome solve_fields_from_state(int block, MultiFab& state) const {
    return provider_().program_execution_solve_fields_from_state_outcome_(block, state);
  }

  SolveOutcome solve_fields_from_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                          const std::string& provider_slot, int block,
                                          MultiFab& state) const {
    if (provider_slot.empty())
      throw std::invalid_argument("Program field solve requires an exact provider slot");
    return provider_().program_execution_field_solve_from_state_at_outcome_(point, provider_slot,
                                                                            block, state);
  }

  SolveOutcome solve_fields_from_state(const std::string& field, int block, MultiFab& state) const {
    return provider_().program_execution_solve_named_field_from_state_outcome_(field, block, state);
  }

  SolveOutcome solve_fields_from_blocks(const std::vector<const MultiFab*>& states) const {
    return provider_().program_execution_solve_fields_from_blocks_outcome_(states);
  }

  SolveOutcome solve_fields_from_blocks(const std::string& field,
                                        const std::vector<const MultiFab*>& states) const {
    return provider_().program_execution_solve_named_field_from_blocks_outcome_(field, states);
  }

  SolveOutcome solve_fields_from_blocks(std::int64_t value_id, std::string_view field,
                                        std::initializer_list<FieldStageOverride> overrides) const {
    return provider_().program_execution_solve_generated_field_from_blocks_outcome_(value_id, field,
                                                                                    overrides);
  }

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
    LogicalEvaluationScope(const ProgramExecutionServices& services, double child_dt,
                           Rollback rollback)
        : services_(&services), child_dt_(child_dt), rollback_(std::move(rollback)) {
      static_assert(std::is_nothrow_move_constructible_v<Rollback>,
                    "Program logical-evaluation rollback tokens must be nothrow movable");
    }
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&& other) noexcept
        : services_(std::exchange(other.services_, nullptr)),
          child_dt_(other.child_dt_),
          rollback_(std::move(other.rollback_)) {}
    LogicalEvaluationScope& operator=(LogicalEvaluationScope&&) = delete;
    ~LogicalEvaluationScope() noexcept { restore_(); }

    Real dt() const {
      if (services_ == nullptr)
        throw std::logic_error("Program logical evaluation scope is no longer active");
      return static_cast<Real>(child_dt_);
    }

   private:
    void restore_() noexcept {
      if (services_ == nullptr)
        return;
      static_assert(
          noexcept(std::declval<const Provider&>().program_execution_restore_logical_evaluation_(
              std::declval<const Rollback&>())),
          "Program logical-evaluation rollback must be noexcept");
      services_->provider_().program_execution_restore_logical_evaluation_(rollback_);
      services_->invalidate_active_operator_snapshot_();
      services_ = nullptr;
    }

    const ProgramExecutionServices* services_ = nullptr;
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
    LogicalEvaluationScope<Rollback> scope(*this, child_dt, std::move(rollback));
    invalidate_active_operator_snapshot_();
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

  /// Validate and prepare one authored simultaneous multi-block residual evaluation.
  ///
  /// Group/member identity, pointer and flux-mode semantics are topology-independent and therefore
  /// live here exactly once. Providers receive only a complete, runtime-block-qualified batch and
  /// own its spatial execution or AMR interface-ledger publication.
  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    require_group_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("Program RHS group cannot be empty");

    RhsGroupBatch batch;
    batch.group_id = group_id;
    batch.requests = requests;
    batch.runtime_blocks.reserve(requests.size());
    batch.states.reserve(requests.size());
    batch.rhs.reserve(requests.size());
    batch.rate_ids.reserve(requests.size());
    batch.flux_only.reserve(requests.size());
    for (const auto& request : requests) {
      require_rate_identity_(request.rate_id);
      if (request.rate_id == group_id || std::find(batch.rate_ids.begin(), batch.rate_ids.end(),
                                                   request.rate_id) != batch.rate_ids.end())
        throw std::invalid_argument(
            "Program RHS group and member rate identities must be distinct");
      if (request.state == nullptr || request.rhs == nullptr ||
          (request.flux_only != 0 && request.flux_only != 1))
        throw std::invalid_argument("Program RHS group contains an invalid request");
      batch.rate_ids.push_back(request.rate_id);
    }
    for (const auto& request : requests) {
      batch.runtime_blocks.push_back(sys_block(request.block));
      batch.states.push_back(request.state);
      batch.rhs.push_back(request.rhs);
      batch.flux_only.push_back(request.flux_only);
    }
    provider_().program_execution_rhs_group_(batch);
  }

  /// Evaluate one authored rate at its exact stable identity.
  ///
  /// Identity validation and profiling belong to Program semantics. The provider receives an
  /// already-qualified runtime block and owns only the Uniform or active-level spatial execution.
  void rhs_into(int block, MultiFab& state, MultiFab& rhs, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel();
    provider_().program_execution_rhs_into_(block, sys_block(block), state, rhs, rate_id);
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage_id) const {
    return provider_().program_execution_boundary_point_(stage_id);
  }

  bool has_boundary_linearization(int block) const {
    return provider_().program_execution_has_boundary_linearization_(sys_block(block));
  }

  /// Fail before a generated pointwise operator touches storage when the selected runtime block
  /// cannot provide Cartesian cell semantics. The shared Program surface owns block qualification;
  /// providers own only the terminal topology capability check.
  void require_cartesian_generated_operator(int block, const std::string& operation) const {
    provider_().program_execution_require_cartesian_generated_operator_(sys_block(block),
                                                                        operation);
  }

  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int block,
                        MultiFab& state, MultiFab& rhs, bool flux_only) const {
    count_kernel();
    provider_().program_execution_rhs_core_into_at_(point, sys_block(block), state, rhs, flux_only,
                                                    nullptr);
  }

  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int block,
                        MultiFab& state, MultiFab& rhs, bool flux_only,
                        const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    provider_().program_execution_rhs_core_into_at_(point, sys_block(block), state, rhs, flux_only,
                                                    &boundary);
  }

  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 int block, MultiFab& state, MultiFab& residual) const {
    count_kernel();
    provider_().program_execution_boundary_residual_into_at_(point, sys_block(block), state,
                                                             residual, nullptr);
  }

  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 int block, MultiFab& state, MultiFab& residual,
                                 const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    provider_().program_execution_boundary_residual_into_at_(point, sys_block(block), state,
                                                             residual, &boundary);
  }

  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int block,
                            MultiFab& state, const MultiFab& direction, MultiFab& result) const {
    count_kernel();
    provider_().program_execution_boundary_jvp_into_at_(point, sys_block(block), state, direction,
                                                        result, nullptr);
  }

  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int block,
                            MultiFab& state, const MultiFab& direction, MultiFab& result,
                            const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    provider_().program_execution_boundary_jvp_into_at_(point, sys_block(block), state, direction,
                                                        result, &boundary);
  }

  /// Evaluate the authored transport divergence without the default source.
  void neg_div_flux_default_into(int block, MultiFab& state, MultiFab& rhs, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel();
    provider_().program_execution_neg_div_flux_default_into_(block, sys_block(block), state, rhs,
                                                             rate_id);
  }

  /// Assemble the centered divergence of authored named fluxes.
  ///
  /// The public overload set, scratch contract and kernel accounting are Program semantics. The
  /// provider receives only complete fields plus an optional execution lane and either executes its
  /// topology-qualified stencil or fails closed before touching storage.
  void neg_div_flux_into(MultiFab& rhs, MultiFab& flux_x, MultiFab& flux_y,
                         MultiFab& divergence_scratch) const {
    count_kernel();
    provider_().program_execution_neg_div_named_flux_into_(rhs, flux_x, flux_y, divergence_scratch,
                                                           nullptr);
  }

  void neg_div_flux_into(MultiFab& rhs, MultiFab& flux_x, MultiFab& flux_y) const {
    MultiFab divergence_scratch(rhs.box_array(), rhs.dmap(), 1, 0);
    neg_div_flux_into(rhs, flux_x, flux_y, divergence_scratch);
  }

  void neg_div_flux_into(MultiFab& rhs, MultiFab& flux_x, MultiFab& flux_y,
                         MultiFab& divergence_scratch, const ExecutionLane& lane) const {
    count_kernel();
    provider_().program_execution_neg_div_named_flux_into_(rhs, flux_x, flux_y, divergence_scratch,
                                                           &lane);
  }

  void neg_div_flux_into(MultiFab& rhs, MultiFab& flux_x, MultiFab& flux_y,
                         const ExecutionLane& lane) const {
    MultiFab divergence_scratch(rhs.box_array(), rhs.dmap(), 1, 0);
    neg_div_flux_into(rhs, flux_x, flux_y, divergence_scratch, lane);
  }

  /// r <- S(u, aux) for one Program block, without any flux divergence.
  ///
  /// Profiling and Program-to-runtime block qualification are topology-independent. The provider
  /// receives only the exact runtime block and owns the Uniform or level-qualified source kernel.
  void source_default_into(int block, MultiFab& state, MultiFab& rhs) const {
    count_kernel();
    provider_().program_execution_source_default_into_(sys_block(block), state, rhs);
  }

  /// Project one candidate state through the exact authored block closure.
  ///
  /// Program-to-runtime block qualification is topology-independent. The provider owns only the
  /// Uniform or level-qualified native projection call.
  void apply_projection(int block, MultiFab& state) const {
    provider_().program_execution_apply_projection_(sys_block(block), state);
  }

  /// Minimum physical cell size used by the native CFL authority.
  ///
  /// The Program owns one dt-bound operation. The provider supplies only the topology-qualified
  /// terminal query, so Uniform and AMR cannot grow separate public CFL dispatch paths.
  Real hmin() const { return provider_().program_execution_hmin_(); }

  /// Maximum wave speed for one authored block on the supplied state.
  ///
  /// Program-to-runtime block qualification happens exactly once here. The provider receives the
  /// authenticated runtime block and performs only its Uniform or active-level native reduction.
  Real max_wave_speed(int block, const MultiFab& state) const {
    return provider_().program_execution_max_wave_speed_(sys_block(block), state);
  }

  /// Whether generated metric-aware operators execute on a polar mesh.
  bool is_polar_geometry() const { return provider_().program_execution_is_polar_geometry_(); }

  /// Physical radial origin used by generated metric-aware operators.
  Real radial_origin() const { return provider_().program_execution_radial_origin_(); }

  /// Physical radial cell spacing used by generated metric-aware operators.
  Real radial_spacing() const { return provider_().program_execution_radial_spacing_(); }

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

  /// Zero-copy access to one authored block's live state through the provider's storage authority.
  MultiFab& state(int block) const {
    return provider_().program_execution_state_(sys_block(block));
  }

  /// Shared auxiliary field and topology-qualified mesh context of the active execution level.
  MultiFab& aux() const {
    MultiFab* field = grid_context().aux;
    if (field == nullptr)
      throw std::logic_error("Program execution context has no auxiliary field storage");
    return *field;
  }
  GridContext grid_context() const { return provider_().program_execution_default_grid_context_(); }
  Geometry geom() const { return grid_context().geom; }

  /// Materialize a lane-private mesh boundary authority with no block components.
  std::shared_ptr<PreparedGridBoundarySession> prepare_mesh_boundary_session(
      const MultiFab&, const ExecutionLane& lane) const {
    return std::make_shared<PreparedGridBoundarySession>(
        provider_().program_execution_default_grid_context_(), lane);
  }

  /// Materialize the exact boundary authority of one authenticated Program block.
  std::shared_ptr<PreparedGridBoundarySession> prepare_block_boundary_session(
      int block, MultiFab& prototype, const runtime::multiblock::BoundaryEvaluationPoint& point,
      const ExecutionLane& lane) const {
    return std::make_shared<PreparedGridBoundarySession>(
        provider_().program_execution_block_grid_context_(block), lane, prototype, point);
  }

  /// Resolve storage for one generated prepared-operator assembly write.
  MultiFab& assembly_target(MultiFab& field, std::string_view field_slot_identity) const {
    validate_prepared_field_slot(field_slot_identity, "Program assembly_target");
    return provider_().program_execution_assembly_target_(field, field_slot_identity);
  }

  /// Resolve storage for one generated prepared-operator reconstruction read.
  MultiFab& assembly_source(MultiFab& field, std::string_view field_slot_identity) const {
    validate_prepared_field_slot(field_slot_identity, "Program assembly_source");
    return provider_().program_execution_assembly_source_(field, field_slot_identity);
  }

  /// Resolve the topology-qualified published value of one prepared linear solve.
  MultiFab& linear_solution(MultiFab& field) const {
    return provider_().program_execution_linear_solution_(field);
  }

  /// Mint the unverified prepared-apply capability only for an authority owned by this artifact.
  ::pops::detail::AuthenticatedProgramApplyToken authenticated_program_apply_token(
      OperatorFingerprint authority) const {
    if (!provider_().program_execution_owns_operator_authority_(authority))
      throw std::invalid_argument(
          "compiled Program requested an operator authority not owned by its installed artifact");
    return ::pops::detail::AuthenticatedProgramApplyToken(authority);
  }

  /// Execute one already prepared affine problem with its bound persistent workspace.
  SolveOutcome solve_prepared_linear(const PreparedAffineLinearProblem& problem,
                                     KrylovWorkspace& workspace, MultiFab& solution,
                                     const MultiFab& rhs, const KrylovControls& controls) const {
    return pops::solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls);
  }

  /// Mint one exact operator-evaluation identity.
  ///
  /// Monotonic revisions and active-mint invalidation are Program semantics. Providers contribute
  /// only the topology fingerprint and the exact Uniform or active-level temporal identity.
  OperatorEvaluationSnapshot operator_evaluation_snapshot(OperatorFingerprint authority,
                                                          const MultiFab& prototype,
                                                          OperatorFingerprint resources) const {
    const OperatorFingerprint topology =
        provider_().program_execution_operator_topology_(prototype);
    if (operator_snapshot_revision_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program operator snapshot revision exhausted");
    const std::uint64_t revision = ++operator_snapshot_revision_;
    invalidate_active_operator_snapshot_();
    OperatorEvaluationSnapshot snapshot =
        provider_().program_execution_operator_evaluation_snapshot_(authority, topology, resources,
                                                                    revision);
    active_operator_snapshot_revision_ = revision;
    return snapshot;
  }

  /// Probe the current native identity without issuing a new revision.
  ///
  /// A prior mint remains reproducible only while no logical-scope transition or newer mint has
  /// invalidated it. Provider topology and clock revisions remain part of the returned identity.
  OperatorEvaluationSnapshot probe_operator_evaluation(OperatorFingerprint authority,
                                                       OperatorFingerprint topology,
                                                       OperatorFingerprint resources,
                                                       std::uint64_t revision) const {
    const std::uint64_t probe_revision =
        revision == active_operator_snapshot_revision_ ? revision : UINT64_C(0);
    return provider_().program_execution_operator_evaluation_snapshot_(authority, topology,
                                                                       resources, probe_revision);
  }

  /// Apply the topology-qualified scalar Laplacian.
  ///
  /// Halo ownership, profiling and the Cartesian stencil are Program semantics and live here once.
  /// A provider participates only when it advertises a non-Cartesian geometry; that hook is explicit
  /// and fail-closed, so AMR cannot silently reinterpret a polar operator as Cartesian.
  void laplacian(MultiFab& out, MultiFab& in) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(in, context.geom.domain, context.bc);
    apply_spatial_laplacian_(out, in, context.geom, nullptr, nullptr, nullptr, nullptr);
  }

  void laplacian(MultiFab& out, MultiFab& in, const ExecutionLane& lane) const {
    require_lane_or_prepared_laplacian_();
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(in, context.geom.domain, context.bc, lane);
    apply_spatial_laplacian_(out, in, context.geom, nullptr, nullptr, nullptr, nullptr);
  }

  void laplacian(MultiFab& out, MultiFab& in, const PreparedGridBoundarySession& boundary) const {
    require_lane_or_prepared_laplacian_();
    count_kernel();
    boundary.fill(in);
    apply_spatial_laplacian_(out, in, boundary.context().geom, nullptr, nullptr, nullptr, nullptr);
  }

  void laplacian(MultiFab& out, MultiFab& in, const PreparedGridBoundarySession& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_lane_or_prepared_laplacian_();
    count_kernel();
    boundary.fill(in, point);
    apply_spatial_laplacian_(out, in, boundary.context().geom, nullptr, nullptr, nullptr, nullptr);
  }

  /// Metric-aware tensor ``div(A grad(in))``.
  ///
  /// The four authored coefficient fields and the exact prepared boundary session are passed
  /// unchanged. Cartesian Uniform and AMR providers execute the same stencil below; only the
  /// explicitly advertised polar provider receives the non-Cartesian hook.
  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_grid_ghosts(in, context);
    apply_spatial_laplacian_(out, in, context.geom, &a_xx, &a_yy, &a_xy, &a_yx);
  }

  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const ExecutionLane& lane) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_grid_ghosts(in, context, lane);
    apply_spatial_laplacian_(out, in, context.geom, &a_xx, &a_yy, &a_xy, &a_yx);
  }

  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(in);
    apply_spatial_laplacian_(out, in, boundary.context().geom, &a_xx, &a_yy, &a_xy, &a_yx);
  }

  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const PreparedGridBoundarySession& boundary,
                        const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(in, point);
    apply_spatial_laplacian_(out, in, boundary.context().geom, &a_xx, &a_yy, &a_xy, &a_yx);
  }

  /// Centered gradient with the topology-qualified transport boundary.
  void gradient(MultiFab& out, MultiFab& phi) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(phi, context.geom.domain, context.bc);
    apply_spatial_gradient_(out, phi, context.geom);
  }

  void gradient(MultiFab& out, MultiFab& phi, const ExecutionLane& lane) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(phi, context.geom.domain, context.bc, lane);
    apply_spatial_gradient_(out, phi, context.geom);
  }

  void gradient(MultiFab& out, MultiFab& phi, const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(phi);
    apply_spatial_gradient_(out, phi, boundary.context().geom);
  }

  void gradient(MultiFab& out, MultiFab& phi, const PreparedGridBoundarySession& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(phi, point);
    apply_spatial_gradient_(out, phi, boundary.context().geom);
  }

  /// Centered divergence. The y flux may alias the x flux, matching the gradient layout.
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(fx, context.geom.domain, context.bc);
    if (&fy != &fx)
      fill_ghosts(fy, context.geom.domain, context.bc);
    apply_divergence(fx, fy, context.geom, out, /*cx=*/0, /*cy=*/1);
  }

  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy, const ExecutionLane& lane) const {
    count_kernel();
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(fx, context.geom.domain, context.bc, lane);
    if (&fy != &fx)
      fill_ghosts(fy, context.geom.domain, context.bc, lane);
    apply_divergence(fx, fy, context.geom, out, /*cx=*/0, /*cy=*/1);
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

  /// Declare one persistent Program history ring.
  ///
  /// Registration, identity binding, owner mapping and idempotent growth are Program semantics and
  /// therefore live once in this service. Providers only materialize the ring on their storage
  /// topology.
  void register_history(const std::string& name, int lag, int ncomp = -1) const {
    HistoryRegistration registration =
        history_registration_(name, lag, ncomp, /*program_owner=*/-1);
    provider_().program_execution_register_history_storage_(registration);
    history_bindings_.insert_or_assign(name, history_binding_(registration));
  }

  void register_history(const std::string& name, int lag, int ncomp, int owner,
                        const std::string& state_identity, const std::string& space_identity,
                        const std::string& clock_identity,
                        const std::string& interpolation_identity) const {
    if (owner < 0 || state_identity.empty() || space_identity.empty() || clock_identity.empty() ||
        interpolation_identity.empty())
      throw std::runtime_error(
          "Program qualified history requires owner/state/space/clock/interpolation identities");
    const int runtime_owner = sys_block(owner);
    HistoryRegistration registration{name,
                                     lag,
                                     ncomp,
                                     owner,
                                     runtime_owner,
                                     state_identity,
                                     space_identity,
                                     clock_identity,
                                     interpolation_identity,
                                     true};
    validate_history_registration_(registration);
    provider_().program_execution_register_history_storage_(registration);
    history_bindings_.insert_or_assign(name, history_binding_(registration));
  }

  /// Read one retained history slot. A normal read stays fail-loud before the first store.
  MultiFab& history(const std::string& name, int lag = 1) const {
    const HistoryRegistration registration =
        ensure_history_registered_(name, lag == 0 ? 1 : lag, -1, /*program_owner=*/-1);
    return provider_().program_execution_read_history_storage_(registration, lag,
                                                               HistoryReadMode::RequireInitialized);
  }

  MultiFab& history(const std::string& name, int lag, int owner) const {
    const HistoryRegistration registration =
        ensure_history_registered_(name, lag == 0 ? 1 : lag, -1, owner);
    return provider_().program_execution_read_history_storage_(registration, lag,
                                                               HistoryReadMode::RequireInitialized);
  }

  /// Read a declared zero cold-start history. The first read authenticates the zero-filled slots.
  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    const HistoryRegistration registration =
        ensure_history_registered_(name, lag, ncomp, /*program_owner=*/-1);
    if (!provider_().program_execution_history_initialized_storage_(registration))
      provider_().program_execution_set_history_initialized_storage_(registration, true);
    return provider_().program_execution_read_history_storage_(registration, lag,
                                                               HistoryReadMode::ZeroStart);
  }

  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp, int owner) const {
    const HistoryRegistration registration = ensure_history_registered_(name, lag, ncomp, owner);
    if (!provider_().program_execution_history_initialized_storage_(registration))
      provider_().program_execution_set_history_initialized_storage_(registration, true);
    return provider_().program_execution_read_history_storage_(registration, lag,
                                                               HistoryReadMode::ZeroStart);
  }

  /// Publish one current history value. Cadence and outgoing-time selection remain topology hooks;
  /// registration, validation and the single store decision live here.
  void store_history(const std::string& name, const MultiFab& value) const {
    store_history_(name, value, /*program_owner=*/-1);
  }

  void store_history(const std::string& name, const MultiFab& value, int owner) const {
    store_history_(name, value, owner);
  }

  /// Rotate every history ring once at an accepted Program boundary.
  void rotate_histories() const { rotate_histories_(""); }

  /// Rotate histories belonging to one logical clock. A topology without independent per-clock
  /// rings may execute this only when every qualified ring belongs to that same clock.
  void rotate_histories(const std::string& clock_identity) const {
    if (clock_identity.empty())
      throw std::runtime_error("Program history rotation requires a logical-clock identity");
    bool found = false;
    bool mixed = false;
    for (const auto& [name, binding] : history_bindings_) {
      (void)name;
      if (!binding.qualified)
        continue;
      found = found || binding.clock_identity == clock_identity;
      mixed = mixed || binding.clock_identity != clock_identity;
    }
    if (!found)
      return;
    if (mixed && !provider_().program_execution_history_supports_selective_rotation_())
      throw std::runtime_error(
          "Program history storage cannot rotate mixed logical clocks independently");
    rotate_histories_(
        provider_().program_execution_history_supports_selective_rotation_() ? clock_identity : "");
  }

  /// Allocate one scalar field on the provider's active storage topology.
  MultiFab alloc_scalar_field(int n_comp = 1, int n_ghost = 1) const {
    return provider_().program_execution_alloc_scalar_field_(n_comp, n_ghost);
  }

  /// Topology-qualified metadata used by generated persistent Program resources.
  ProgramResourceTopology program_resource_topology() const {
    const ProgramResourceTopology topology = provider_().program_execution_resource_topology_();
    if (topology.levels <= 0)
      throw std::runtime_error("Program resource topology requires at least one level");
    return topology;
  }

  /// Active provider level. The raw cursor remains provider-owned; its public access and every
  /// generated selection go through this shared service.
  int level() const { return provider_().program_execution_resource_level_(); }

  void set_level(int selected) const {
    const int levels = program_resource_topology().levels;
    if (selected < 0 || selected >= levels)
      throw std::out_of_range("Program resource level is out of range");
    select_program_resource_level_(selected);
  }

  /// Evaluate one generated resource phase at @p selected and restore the incoming provider cursor
  /// on success or exception. Provider selection itself is a no-throw raw topology operation.
  template <class Body>
  void with_program_resource_level(int selected, Body&& body) const {
    const int levels = program_resource_topology().levels;
    if (selected < 0 || selected >= levels)
      throw std::out_of_range("Program resource level is out of range");
    const int incoming = level();
    const int restored = incoming >= 0 && incoming < levels ? incoming : 0;
    select_program_resource_level_(selected);
    try {
      body();
    } catch (...) {
      select_program_resource_level_(restored);
      throw;
    }
    select_program_resource_level_(restored);
  }

  /// Visit every level of one captured hierarchy identity and restore the incoming cursor even when
  /// materialization fails part-way through. This replaces generated save/set/try/catch code.
  template <class Body>
  void for_each_program_resource_level(Body&& body) const {
    const ProgramResourceTopology topology = program_resource_topology();
    const int incoming = level();
    const int restored = incoming >= 0 && incoming < topology.levels ? incoming : 0;
    try {
      for (int selected = 0; selected < topology.levels; ++selected) {
        select_program_resource_level_(selected);
        body(selected);
      }
    } catch (...) {
      select_program_resource_level_(restored);
      throw;
    }
    select_program_resource_level_(restored);
  }

  const PreparedVectorDistribution& program_resource_vector_distribution() const {
    return provider_().program_execution_resource_storage_().vector_distribution;
  }
  FieldDistribution program_resource_field_storage_distribution() const {
    return provider_().program_execution_resource_storage_().field_distribution;
  }
  int program_resource_field_level() const {
    return provider_().program_execution_resource_storage_().field_level;
  }

  /// Resolve physical cell measures once, then apply them to every authored basis.
  ///
  /// The provider owns only the topology projection: a uniform mesh returns one measure, while an
  /// AMR level returns a hierarchy-sized vector whose active level carries the positive measure.
  void configure_program_resource_field_nullspace(FieldNullspacePlan& plan) const {
    const std::vector<Real> cell_measures = provider_().program_execution_resource_cell_measures_();
    for (FieldNullspaceBasis& basis : plan.bases)
      basis.cell_measure = cell_measures;
  }

  /// Return the exact active-cell mask for one block-qualified generated pointwise operator.
  ///
  /// A provider supplies only its topology-qualified GridContext.  The shared service owns the
  /// geometry-mode interpretation, layout authentication and fail-closed missing-mask behavior.
  const MultiFab* pointwise_active_mask(int block, const MultiFab& field) const {
    return active_mask_from_context_(provider_().program_execution_block_grid_context_(block),
                                     field, "Program pointwise active-cell mask");
  }

  /// Collectively publish one pointwise status over the exact domain used by the device kernel.
  Real pointwise_status_max(int block, const MultiFab& status, const MultiFab* active_cells) const {
    const MultiFab* expected = pointwise_active_mask(block, status);
    if (expected != active_cells)
      throw std::invalid_argument(
          "Program pointwise status reduction received a different active-cell mask");
    const Real reduced = pops::reduce_max(status, 0, RelativeCellMeasure{active_cells, nullptr});
    return reduced == -std::numeric_limits<Real>::infinity() ? Real(0) : reduced;
  }

  /// u <- u + a r over the valid physical cells.
  void axpy(MultiFab& u, Real a, const MultiFab& r) const {
    count_kernel();
    if (const MultiFab* active_cells = active_domain_mask_())
      pops::saxpy_active(u, a, r, *active_cells);
    else
      pops::saxpy(u, a, r);
    provider_().program_execution_publish_axpy_(u, a, r);
  }

  /// Exact-coefficient twin used by conservative AMR ledger publication.
  void axpy(MultiFab& u, Real a, const MultiFab& r, Real dt,
            std::initializer_list<ExactCoefficientTerm> exact) const {
    count_kernel();
    if (const MultiFab* active_cells = active_domain_mask_())
      pops::saxpy_active(u, a, r, *active_cells);
    else
      pops::saxpy(u, a, r);
    provider_().program_execution_publish_exact_axpy_(u, a, r, dt, exact);
  }

  /// z <- a x + b y over the valid physical cells.
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    count_kernel();
    if (const MultiFab* active_cells = active_domain_mask_())
      pops::lincomb_active(z, a, x, b, y, *active_cells);
    else
      pops::lincomb(z, a, x, b, y);
    provider_().program_execution_publish_lincomb_(z, a, x, b, y);
  }

  /// Exact-coefficient twin used by conservative AMR ledger publication.
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y, Real dt,
               std::initializer_list<ExactCoefficientTerm> exact_a,
               std::initializer_list<ExactCoefficientTerm> exact_b) const {
    count_kernel();
    if (const MultiFab* active_cells = active_domain_mask_())
      pops::lincomb_active(z, a, x, b, y, *active_cells);
    else
      pops::lincomb(z, a, x, b, y);
    provider_().program_execution_publish_exact_lincomb_(z, a, x, b, y, dt, exact_a, exact_b);
  }

  /// Collective reductions over one explicitly owned Program block.
  Real sum_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_sum(u, comp, relative_cell_measure_(owner));
  }
  Real max_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_max(u, comp, relative_cell_measure_(owner));
  }
  Real min_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_min(u, comp, relative_cell_measure_(owner));
  }
  Real abs_sum_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_abs_sum(u, comp, relative_cell_measure_(owner));
  }
  Real norm2(int owner, const MultiFab& u) const {
    return std::sqrt(pops::dot(u, u, 0, relative_cell_measure_(owner)));
  }
  Real norm_inf(int owner, const MultiFab& u) const {
    return pops::reduce_norm_inf(u, 0, relative_cell_measure_(owner));
  }
  Real dot(int owner, const MultiFab& left, const MultiFab& right) const {
    return pops::dot(left, right, 0, relative_cell_measure_(owner));
  }

  Real sum(int owner, const MultiFab& u) const { return sum_component(owner, u, 0); }
  Real max(int owner, const MultiFab& u) const { return max_component(owner, u, 0); }
  Real min(int owner, const MultiFab& u) const { return min_component(owner, u, 0); }
  Real abs_sum(int owner, const MultiFab& u) const { return abs_sum_component(owner, u, 0); }

  /// Legacy hand-written Cartesian stages may omit the owner only when topology says that is safe.
  Real sum_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_sum(u, comp);
  }
  Real max_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_max(u, comp);
  }
  Real min_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_min(u, comp);
  }
  Real abs_sum_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_abs_sum(u, comp);
  }
  Real sum(const MultiFab& u) const { return sum_component(u, 0); }
  Real max(const MultiFab& u) const { return max_component(u, 0); }
  Real min(const MultiFab& u) const { return min_component(u, 0); }
  Real abs_sum(const MultiFab& u) const { return abs_sum_component(u, 0); }

  /// Fill the transport halos of one field through the provider's current topology.
  void fill_boundary(MultiFab& x) const {
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(x, context.geom.domain, context.bc);
  }

  void fill_boundary(MultiFab& x, const ExecutionLane& lane) const {
    const GridContext context = provider_().program_execution_default_grid_context_();
    fill_ghosts(x, context.geom.domain, context.bc, lane);
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
          lincomb(*target, Real(0), *target, Real(1), *source);
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
        lincomb(*target, Real(0), *target, Real(1), *source);
        // Aliased snapshots are function-local. Complete their consumers before destruction.
        device_fence();
      }
    }
  }

  /// Apply every coupled-source operator to one complete, simultaneous candidate-state pack.
  ///
  /// Candidate ordering, exact layouts, block-map bijection, accepted-state alias rejection and
  /// workspace reentrancy are Program semantics.  The provider receives only the authenticated
  /// runtime-ordered pointers and performs its topology-specific native coupling call.
  void apply_coupling_operators(Real dt,
                                std::initializer_list<CouplingStateOverride> candidates) const {
    if (!std::isfinite(static_cast<double>(dt)) || dt < Real(0))
      throw std::invalid_argument("Program coupling application requires a finite non-negative dt");
    if (coupling_workspace_.in_use)
      throw std::logic_error("Program coupling workspace is already in use");
    prepare_coupling_workspace_(candidates);
    struct WorkspaceUse {
      bool& flag;
      explicit WorkspaceUse(bool& value) : flag(value) { flag = true; }
      ~WorkspaceUse() { flag = false; }
    } use(coupling_workspace_.in_use);
    const std::size_t applied =
        provider_().program_execution_apply_coupling_(dt, coupling_workspace_.runtime_states);
    count_kernel(static_cast<std::int64_t>(applied));
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

  /// Scheduler cache semantics shared by every capable Program storage provider.
  ///
  /// The service owns cadence, profiling and value movement.  A provider supplies only the
  /// checkpoint-visible CacheManager authority; a topology without that capability fails loudly at
  /// the hook instead of fabricating a cache detached from its restart lifecycle.
  bool cache_should_update(int node_id, int every_n) const {
    CacheManager& cache =
        provider_().program_execution_cache_(SchedulerCacheOperation::ShouldUpdate);
    const bool due = cache.is_due(node_id, macro_step(), every_n);
    if (due) {
      profiler().count("cache_misses");
      profiler().count("nodes_due");
    } else {
      profiler().count("cache_hits");
      profiler().count("nodes_skipped");
    }
    return due;
  }

  void cache_store_aux(int node_id) const {
    provider_()
        .program_execution_cache_(SchedulerCacheOperation::StoreAux)
        .store(node_id, aux(), macro_step());
  }

  void cache_restore_aux(int node_id) const {
    provider_()
        .program_execution_cache_(SchedulerCacheOperation::RestoreAux)
        .restore_into(node_id, aux());
  }

  void cache_store_scratch(int node_id, const MultiFab& scratch) const {
    provider_()
        .program_execution_cache_(SchedulerCacheOperation::StoreScratch)
        .store(node_id, scratch, macro_step());
  }

  void cache_restore_scratch(int node_id, MultiFab& scratch) const {
    provider_()
        .program_execution_cache_(SchedulerCacheOperation::RestoreScratch)
        .restore_into(node_id, scratch);
  }

  void cache_accumulate_dt(int node_id, Real dt) const {
    provider_()
        .program_execution_cache_(SchedulerCacheOperation::AccumulateDt)
        .accumulate_dt(node_id, dt);
  }

  Real cache_effective_dt(int node_id, Real dt_now) const {
    return provider_()
        .program_execution_cache_(SchedulerCacheOperation::EffectiveDt)
        .effective_dt(node_id, dt_now);
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

  static SolveReport consume_field_outcome_(SolveOutcome outcome) {
    return outcome.consume(outcome.report().solved_value_available()
                               ? SolveConsumption::kAccept
                               : (outcome.report().action == SolveAction::kRejectAttempt
                                      ? SolveConsumption::kRejectAttempt
                                      : SolveConsumption::kFailRun));
  }

  RelativeCellMeasure relative_cell_measure_(int owner) const {
    const GridContext context = provider_().program_execution_block_grid_context_(owner);
    if (!embedded_domain_enabled_(context))
      return {};
    if (context.domain_mask == nullptr)
      throw std::runtime_error("Program physical reduction has no prepared active-cell mask");
    if (context.geometry_mode != nullptr && *context.geometry_mode == GeometryMode::CutCell) {
      if (context.eb_inverse_volume_fraction == nullptr)
        throw std::runtime_error(
            "Program cut-cell reduction has no prepared inverse volume fraction");
      return {context.domain_mask, context.eb_inverse_volume_fraction};
    }
    return {context.domain_mask, nullptr};
  }

  void require_unqualified_reduction_safe_() const {
    const GridContext context = provider_().program_execution_default_grid_context_();
    if (embedded_domain_enabled_(context))
      throw std::runtime_error(
          "Program embedded-boundary reduction requires an explicit Program block owner");
  }

  const MultiFab* active_domain_mask_() const {
    const GridContext context = provider_().program_execution_default_grid_context_();
    if (!embedded_domain_enabled_(context))
      return nullptr;
    if (context.domain_mask == nullptr)
      throw std::runtime_error(
          "Program embedded-boundary algebra has no prepared active-cell mask");
    return context.domain_mask;
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
  struct HistoryBinding {
    int program_owner = -1;
    std::string state_identity;
    std::string space_identity;
    std::string clock_identity;
    std::string interpolation_identity;
    bool qualified = false;
  };

  struct CouplingWorkspace {
    std::vector<int> program_to_runtime;
    std::vector<MultiFab*> runtime_states;
    bool in_use = false;
  };

  const Provider& provider_() const { return static_cast<const Provider&>(*this); }

  void invalidate_active_operator_snapshot_() const noexcept {
    active_operator_snapshot_revision_ = 0;
  }
  void require_lane_or_prepared_laplacian_() const {
    if (provider_().program_execution_is_polar_geometry_())
      throw std::logic_error(
          "lane-isolated or prepared Program laplacian requires an explicit polar tensor "
          "operator");
  }

  void apply_spatial_laplacian_(MultiFab& out, MultiFab& in, const Geometry& geometry,
                                const MultiFab* a_xx, const MultiFab* a_yy, const MultiFab* a_xy,
                                const MultiFab* a_yx) const {
    const int coefficient_count =
        static_cast<int>(a_xx != nullptr) + static_cast<int>(a_yy != nullptr) +
        static_cast<int>(a_xy != nullptr) + static_cast<int>(a_yx != nullptr);
    if (coefficient_count != 0 && coefficient_count != 4)
      throw std::logic_error(
          "Program tensor Laplacian requires all four authored coefficient fields");
    if (provider_().program_execution_is_polar_geometry_()) {
      provider_().program_execution_apply_polar_tensor_(out, in, a_xx, a_yy, a_xy, a_yx);
      return;
    }
    if (coefficient_count == 0) {
      apply_laplacian(in, geometry, out);
      return;
    }
    apply_laplacian(in, geometry, out, nullptr, a_xx, nullptr, a_yy, a_xy, a_yx);
  }

  static void apply_spatial_gradient_(MultiFab& out, const MultiFab& phi,
                                      const Geometry& geometry) {
    const Real cx = Real(1) / (Real(2) * geometry.dx());
    const Real cy = Real(1) / (Real(2) * geometry.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  static HistoryBinding history_binding_(const HistoryRegistration& registration) {
    return {registration.program_owner,          registration.state_identity,
            registration.space_identity,         registration.clock_identity,
            registration.interpolation_identity, registration.qualified};
  }

  void validate_history_registration_(const HistoryRegistration& registration) const {
    if (registration.name.empty())
      throw std::runtime_error("Program history name must be non-empty");
    if (registration.lag < 1)
      throw std::runtime_error("Program history lag must be >= 1");
    if (registration.ncomp == 0)
      throw std::runtime_error("Program history component count cannot be zero");
    const auto prior = history_bindings_.find(registration.name);
    if (prior == history_bindings_.end() || !prior->second.qualified)
      return;
    const HistoryBinding& binding = prior->second;
    if (binding.program_owner != registration.program_owner ||
        binding.state_identity != registration.state_identity ||
        binding.space_identity != registration.space_identity ||
        binding.clock_identity != registration.clock_identity ||
        binding.interpolation_identity != registration.interpolation_identity)
      throw std::runtime_error("Program history '" + registration.name +
                               "' cannot be re-registered with a different identity");
  }

  HistoryRegistration history_registration_(const std::string& name, int lag, int ncomp,
                                            int program_owner) const {
    if (name.empty())
      throw std::runtime_error("Program history name must be non-empty");
    if (lag < 1)
      throw std::runtime_error("Program history lag must be >= 1");
    if (ncomp == 0)
      throw std::runtime_error("Program history component count cannot be zero");
    const auto prior = history_bindings_.find(name);
    if (prior != history_bindings_.end()) {
      const HistoryBinding& binding = prior->second;
      if (program_owner >= 0 && binding.qualified && binding.program_owner != program_owner)
        throw std::runtime_error("history '" + name + "' is not qualified by owner program.block." +
                                 std::to_string(program_owner));
      const int owner = binding.program_owner >= 0 ? binding.program_owner : program_owner;
      return {name,
              lag,
              ncomp,
              owner,
              owner < 0 ? -1 : sys_block(owner),
              binding.state_identity,
              binding.space_identity,
              binding.clock_identity,
              binding.interpolation_identity,
              binding.qualified};
    }
    return {name, lag, ncomp, program_owner, program_owner < 0 ? -1 : sys_block(program_owner), "",
            "",   "",  "",    false};
  }

  HistoryRegistration ensure_history_registered_(const std::string& name, int lag, int ncomp,
                                                 int program_owner) const {
    HistoryRegistration registration = history_registration_(name, lag, ncomp, program_owner);
    provider_().program_execution_register_history_storage_(registration);
    history_bindings_.insert_or_assign(name, history_binding_(registration));
    return registration;
  }

  void store_history_(const std::string& name, const MultiFab& value, int program_owner) const {
    const HistoryRegistration registration =
        ensure_history_registered_(name, /*lag=*/1, /*ncomp=*/-1, program_owner);
    const HistoryStorePlan plan = provider_().program_execution_history_store_plan_(registration);
    if (!plan.due)
      return;
    if (plan.outgoing_dt &&
        (!std::isfinite(static_cast<double>(*plan.outgoing_dt)) || *plan.outgoing_dt < Real(0)))
      throw std::runtime_error("Program history store requires a finite non-negative outgoing dt");
    provider_().program_execution_store_history_storage_(registration, value, plan.outgoing_dt);
  }

  void rotate_histories_(const std::string& clock_identity) const {
    switch (provider_().program_execution_history_rotation_action_()) {
      case HistoryRotationAction::Skip:
        return;
      case HistoryRotationAction::Defer:
        provider_().program_execution_defer_history_rotation_();
        return;
      case HistoryRotationAction::Rotate:
        provider_().program_execution_rotate_history_storage_(clock_identity);
        return;
    }
    throw std::logic_error("unknown Program history rotation action");
  }

  void select_program_resource_level_(int selected) const noexcept {
    static_assert(
        noexcept(std::declval<const Provider&>().program_execution_select_resource_level_(0)),
        "Program resource-level selection must be noexcept so shared scopes can always restore");
    provider_().program_execution_select_resource_level_(selected);
  }

  void prepare_coupling_workspace_(std::initializer_list<CouplingStateOverride> candidates) const {
    const std::vector<int>& block_map = provider_().program_execution_block_map_();
    const std::size_t runtime_blocks =
        static_cast<std::size_t>(provider_().program_execution_block_count_());
    if (block_map.empty())
      throw block_map_error_("Program coupling has no explicit program-to-runtime block map");
    if (block_map.size() != runtime_blocks || candidates.size() != block_map.size())
      throw std::invalid_argument(
          "Program coupling requires a complete candidate pack for every runtime block");

    const bool structure_changed = coupling_workspace_.program_to_runtime != block_map ||
                                   coupling_workspace_.runtime_states.size() != runtime_blocks;
    if (structure_changed) {
      coupling_workspace_.runtime_states.assign(runtime_blocks, nullptr);
      for (std::size_t program_block = 0; program_block < block_map.size(); ++program_block) {
        const int runtime_block = sys_block(static_cast<int>(program_block));
        MultiFab*& mapped =
            coupling_workspace_.runtime_states[static_cast<std::size_t>(runtime_block)];
        if (mapped != nullptr)
          throw std::invalid_argument(
              "Program coupling block map does not cover each runtime block exactly once");
        // A live-state pointer is only a non-null sentinel while authenticating the map.
        mapped = &state(static_cast<int>(program_block));
      }
      coupling_workspace_.program_to_runtime.assign(block_map.begin(), block_map.end());
    }

    std::fill(coupling_workspace_.runtime_states.begin(), coupling_workspace_.runtime_states.end(),
              nullptr);
    std::size_t ordinal = 0;
    for (const CouplingStateOverride& candidate : candidates) {
      if (candidate.program_block != static_cast<int>(ordinal) || candidate.state == nullptr)
        throw std::invalid_argument(
            "Program coupling candidates must be non-null and ordered by Program block");
      const MultiFab& live = state(candidate.program_block);
      if (candidate.state->box_array().boxes() != live.box_array().boxes() ||
          candidate.state->dmap().ranks() != live.dmap().ranks() ||
          candidate.state->ncomp() != live.ncomp() || candidate.state->n_grow() != live.n_grow())
        throw std::invalid_argument(
            "Program coupling candidate does not match its exact runtime layout");
      for (std::size_t other = 0; other < block_map.size(); ++other)
        if (candidate.state == &state(static_cast<int>(other)))
          throw std::invalid_argument(
              "Program coupling candidates cannot alias accepted live states");
      const int runtime_block =
          coupling_workspace_.program_to_runtime[static_cast<std::size_t>(candidate.program_block)];
      coupling_workspace_.runtime_states[static_cast<std::size_t>(runtime_block)] = candidate.state;
      ++ordinal;
    }
  }

  static bool embedded_domain_enabled_(const GridContext& context) {
    if (context.embedded_boundary_set != nullptr || context.geometry_mode != nullptr)
      return context.embedded_boundary_set != nullptr && *context.embedded_boundary_set &&
             context.geometry_mode != nullptr && *context.geometry_mode != GeometryMode::None;
    return context.domain_mask != nullptr;
  }

  static const MultiFab* active_mask_from_context_(const GridContext& context,
                                                   const MultiFab& field, const char* where) {
    if (!embedded_domain_enabled_(context))
      return nullptr;
    if (context.domain_mask == nullptr)
      throw std::runtime_error(std::string(where) + " has no prepared active-cell mask");
    pops::detail::validate_relative_cell_measure(
        field, RelativeCellMeasure{context.domain_mask, nullptr}, where);
    return context.domain_mask;
  }

  std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                         const std::string& clock,
                                                         const std::string& stage_identity,
                                                         int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level,
                                      provider_().program_execution_active_level_(), macro_step());
  }

  mutable CouplingWorkspace coupling_workspace_;
  mutable std::map<std::string, HistoryBinding> history_bindings_;
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::uint64_t active_operator_snapshot_revision_ = 0;  // zero is never a minted revision
};

/// Compile-time association between a public runtime facade and its topology/storage provider.
///
/// Generated modules name only the facade carried by their stable ABI.  The runtime headers own
/// the concrete provider selection, so adding a Program operation never requires a second codegen
/// dispatch table.  Each supported provider specializes this trait beside its own definition.
template <class RuntimeFacade>
struct ProgramExecutionProviderFor;

template <class RuntimeFacade>
using ProgramExecutionProviderForT = typename ProgramExecutionProviderFor<RuntimeFacade>::type;

/// Construct the provider selected by the runtime facade and verify that it consumes the one shared
/// Program semantic authority.  The returned shared owner is intentionally concrete: topology
/// drivers may use provider-owned hierarchy services, while topology-independent generated
/// operations continue to resolve through ProgramExecutionServices exactly once.
template <class RuntimeFacade>
std::shared_ptr<ProgramExecutionProviderForT<RuntimeFacade>> make_program_execution_provider(
    RuntimeFacade* runtime) {
  using Provider = ProgramExecutionProviderForT<RuntimeFacade>;
  static_assert(std::is_base_of_v<ProgramExecutionServices<Provider>, Provider>,
                "a Program execution provider must consume ProgramExecutionServices");
  if (runtime == nullptr)
    throw std::invalid_argument("Program execution provider requires a non-null runtime facade");
  return std::make_shared<Provider>(runtime);
}

/// Construct a short-lived provider view without heap allocation.
///
/// Read-only scalar queries such as the CFL/dt bound need the same facade-selected provider but do
/// not install a closure or extend its lifetime. Returning the provider prvalue gives guaranteed
/// copy elision and keeps the per-step query allocation-free.
template <class RuntimeFacade>
ProgramExecutionProviderForT<RuntimeFacade> make_program_execution_view(RuntimeFacade* runtime) {
  using Provider = ProgramExecutionProviderForT<RuntimeFacade>;
  static_assert(std::is_base_of_v<ProgramExecutionServices<Provider>, Provider>,
                "a Program execution provider must consume ProgramExecutionServices");
  if (runtime == nullptr)
    throw std::invalid_argument("Program execution view requires a non-null runtime facade");
  return Provider(runtime);
}

}  // namespace pops::runtime::program
