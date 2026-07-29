#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/program/cache_manager.hpp>
#include <pops/runtime/context/grid_context.hpp>
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
  struct CouplingWorkspace {
    std::vector<int> program_to_runtime;
    std::vector<MultiFab*> runtime_states;
    bool in_use = false;
  };

  const Provider& provider_() const { return static_cast<const Provider&>(*this); }

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
