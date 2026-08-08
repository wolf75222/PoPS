/// @file
/// @brief Exact compile-time-ranked execution boundary for generated AMR Programs.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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
  using runtime_state_type = ProgramRuntimeState<Dim>;
  using scalar_boundary_session_type = PreparedScalarBoundarySession<Dim>;
  using subcycle_plan_type = ::pops::numerics::time::amr::PreparedAmrSubcyclePlan<Dim, MemorySpace>;

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

  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const AmrProgramContext& owner, int iteration, int count)
        : owner_(&owner), prior_dt_(owner.current_dt_), prior_substep_(owner.logical_substep_) {
      if (iteration < 0 || count < 1 || iteration >= count || !std::isfinite(prior_dt_) ||
          !(prior_dt_ > 0.0))
        throw std::invalid_argument("AMR logical evaluation scope is invalid");
      owner_->current_dt_ = prior_dt_ / static_cast<double>(count);
      owner_->logical_substep_ = iteration;
    }
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          prior_dt_(other.prior_dt_),
          prior_substep_(other.prior_substep_) {}
    ~LogicalEvaluationScope() {
      if (owner_ != nullptr) {
        owner_->current_dt_ = prior_dt_;
        owner_->logical_substep_ = prior_substep_;
      }
    }
    Real dt() const { return static_cast<Real>(owner_->current_dt_); }

   private:
    const AmrProgramContext* owner_ = nullptr;
    double prior_dt_ = 0.0;
    int prior_substep_ = 0;
  };

  explicit AmrProgramContext(facade_type* facade)
      : facade_(require_facade_(facade)), runtime_(require_runtime_(*facade_)) {
    facade_->refresh_prepared_amr_levels();
    scalar_boundary_lane_.emplace(ExecutionLane::duplicate_world_collectively(
        "pops.program.amr.scalar-boundary.nd" + std::to_string(Dim)));
    synchronize_resource_generation_();
  }

  AmrProgramContext(runtime_type* runtime, facade_type* facade)
      : facade_(require_facade_(facade)), runtime_(require_runtime_(runtime)) {
    if (facade_->engine() != runtime_)
      throw std::invalid_argument("AMR Program facade and runtime do not share one hierarchy");
    facade_->refresh_prepared_amr_levels();
    scalar_boundary_lane_.emplace(ExecutionLane::duplicate_world_collectively(
        "pops.program.amr.scalar-boundary.nd" + std::to_string(Dim)));
    synchronize_resource_generation_();
  }

  /// Spatial-only constructor used by preparation tests.  Execution methods require a facade.
  explicit AmrProgramContext(runtime_type& runtime) : runtime_(&runtime) {
    synchronize_resource_generation_();
  }

  runtime_type& runtime() const noexcept { return *runtime_; }
  hierarchy_type& hierarchy() const noexcept { return runtime_->hierarchy(); }
  const ::pops::amr::hierarchy::LevelLayout<Dim>& layout(std::size_t selected) const {
    return runtime_->hierarchy().layout(selected);
  }
  field_type& state(std::size_t selected) const { return runtime_->hierarchy().state(selected); }

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
      ::pops::amr::regridding::RegridPreparationBudget preparation_budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return runtime_->prepare_regrid(parent_level, ratio, std::move(clustered), preparation_budget,
                                    lane);
  }

  void publish_regrid(::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                      std::optional<field_type> child_state) const {
    require_history_free_for_topology_change_("regrid");
    const int parent_level = prepared.source_level().level;
    if (parent_level < 0)
      throw std::invalid_argument("AMR Program regrid has no source level");
    runtime_->publish_regrid(static_cast<std::size_t>(parent_level), std::move(prepared),
                             std::move(child_state));
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t selected, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget, const RebalancePolicy& policy,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return runtime_->prepare_rebalance(selected, estimates, preparation_budget, policy, lane);
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t selected, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return runtime_->prepare_rebalance(selected, estimates, preparation_budget, lane);
  }

  void apply_rebalance(std::size_t selected, PreparedRebalanceDecision<Dim> decision,
                       field_type remapped_state) const {
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
    facade_->install_program_step(
        [step = std::move(step), keep_alive = std::move(keep_alive)](double dt) { step(dt); });
    if (hierarchy_refresh)
      facade_->install_program_hierarchy_refresh(std::move(hierarchy_refresh));
  }

  void begin_step(double dt) const {
    if (!std::isfinite(dt) || !(dt > 0.0))
      throw std::invalid_argument("AMR Program step requires a finite positive dt");
    current_dt_ = dt;
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
            .physical_time = facade_->time() + stage_time_.value() * current_dt_};
  }

  template <class Body>
  void advance_hierarchy(double dt, Body&& body) const {
    begin_step(dt);
    refresh_resources_();
    require_single_level_conservative_route_("advance_hierarchy");
    with_program_resource_level(0, [&] { std::forward<Body>(body)(dt); });
  }

  template <class Body>
  void advance_synchronized_hierarchy(double dt, Body&& body) const {
    begin_step(dt);
    refresh_resources_();
    require_single_level_conservative_route_("advance_synchronized_hierarchy");
    with_program_resource_level(0, [&] { std::forward<Body>(body)(dt); });
  }

  [[noreturn]] void prepare_same_level_cell_temporal_execution(std::string, std::int64_t,
                                                               int = 0) const {
    unavailable_("cell-local AMR temporal provider");
  }
  [[noreturn]] void advance_same_level_cell_temporal(double) const {
    unavailable_("cell-local AMR temporal provider");
  }

  bool uses_prepared_krylov_fallback() const noexcept { return true; }
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
    if (sys_block(program_block) != 0)
      unavailable_("exact-ranked multi-block AMR state provider");
    refresh_resources_();
    return runtime_->hierarchy().state(static_cast<std::size_t>(active_level_));
  }

  field_type& aux() const {
    require_facade_execution_();
    refresh_resources_();
    return facade_->prepared_amr_level_auxiliary(active_level_);
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
    if (sys_block(program_block) != 0)
      unavailable_("exact-ranked multi-block AMR residual provider");
    require_rate_identity_(rate_id);
    require_same_field_contract_(stage_state, rhs, "AMR Program residual");
    const auto& evaluation =
        facade_->evaluate_prepared_amr_level_at(boundary_evaluation_point(rate_id), stage_state);
    copy_valid_(evaluation.residual, rhs);
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
    for (const RhsGroupRequest& request : requests)
      copy_valid_(candidates[index++], *request.rhs);
  }

  [[noreturn]] void neg_div_flux_default_into(int, field_type&, field_type&, int) const {
    unavailable_("split default-flux AMR provider");
  }
  [[noreturn]] void source_default_into(int, field_type&, field_type&) const {
    unavailable_("split default-source AMR provider");
  }

  void require_cartesian_generated_operator(int program_block, const std::string& operation) const {
    (void)sys_block(program_block);
    if (operation.empty())
      throw std::invalid_argument("AMR generated operator requires an operation identity");
  }

  void prepare_generated_state(int program_block, field_type& stage_state, int rate_id) const {
    if (sys_block(program_block) != 0)
      unavailable_("exact-ranked multi-block AMR state preparation");
    require_rate_identity_(rate_id);
    facade_->prepare_generated_amr_level_state(boundary_evaluation_point(rate_id), stage_state);
  }

  void neg_div_named_flux_into(field_type& rhs, const std::array<field_type*, Dim>& fluxes) const {
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
    unavailable_("generated AMR projection provider");
  }

  Real max_wave_speed(int program_block, const field_type& stage_state) const {
    if (sys_block(program_block) != 0)
      unavailable_("exact-ranked multi-block AMR wave-speed provider");
    return facade_->prepared_amr_level_maximum_speed(active_level_, stage_state);
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
    pops::saxpy(destination, factor, source);
    count_kernel_();
  }
  void axpy(field_type& destination, Real factor, const field_type& source, Real,
            std::initializer_list<ExactCoefficientTerm>) const {
    axpy(destination, factor, source);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right) const {
    require_same_field_contract_(destination, left, "AMR Program linear combination");
    require_same_field_contract_(destination, right, "AMR Program linear combination");
    pops::lincomb(destination, left_factor, left, right_factor, right);
    count_kernel_();
  }
  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right, Real, std::initializer_list<ExactCoefficientTerm>,
               std::initializer_list<ExactCoefficientTerm>) const {
    lincomb(destination, left_factor, left, right_factor, right);
  }

  void commit_many(std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
    std::vector<field_type*> targets;
    std::vector<std::optional<field_type>> candidates;
    targets.reserve(commits.size());
    candidates.reserve(commits.size());
    for (const auto& [target, source] : commits) {
      if (target == nullptr || source == nullptr ||
          std::find(targets.begin(), targets.end(), target) != targets.end())
        throw std::invalid_argument("AMR Program commit has null or duplicate storage");
      require_same_field_contract_(*target, *source, "AMR Program commit");
      targets.push_back(target);
      candidates.emplace_back(target == source ? std::nullopt : std::optional<field_type>(*source));
    }
    std::size_t index = 0;
    for (const auto& [target, source] : commits) {
      if (target != source)
        *target = std::move(*candidates[index]);
      ++index;
    }
  }

  [[noreturn]] void apply_coupling_operators(Real,
                                             std::initializer_list<CouplingStateOverride>) const {
    unavailable_("exact-ranked multi-block AMR coupling provider");
  }

  Real sum_component(const field_type& field, int component) const {
    return pops::reduce_sum(field, component);
  }
  Real max_component(const field_type& field, int component) const {
    return pops::reduce_max(field, component);
  }
  Real min_component(const field_type& field, int component) const {
    return pops::reduce_min(field, component);
  }
  Real norm2(int, const field_type& field) const { return std::sqrt(pops::dot(field, field, 0)); }
  Real norm_inf(int, const field_type& field) const { return pops::reduce_norm_inf(field, 0); }
  Real dot(int, const field_type& left, const field_type& right) const {
    return pops::dot(left, right, 0);
  }

  Geometry<Dim> geometry() const {
    require_facade_execution_();
    return facade_->prepared_amr_level_geometry(active_level_);
  }

  field_type& assembly_target(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("AMR Program assembly target requires an identity");
    return field;
  }

  field_type& assembly_source(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("AMR Program assembly source requires an identity");
    return field;
  }

  std::shared_ptr<scalar_boundary_session_type> prepare_mesh_boundary_session(
      field_type& prototype, const ExecutionLane& lane) const {
    return std::make_shared<scalar_boundary_session_type>(
        geometry(), facade_->prepared_amr_boundary_topology(), prototype, lane,
        next_boundary_generation_());
  }

  std::shared_ptr<scalar_boundary_session_type> prepare_block_boundary_session(
      int program_block, field_type& prototype,
      const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
    (void)sys_block(program_block);
    require_boundary_point_(point, "AMR block scalar boundary");
    return prepare_mesh_boundary_session(prototype, lane);
  }

  void fill_boundary(field_type& field) const {
    if (!scalar_boundary_lane_)
      throw std::logic_error("AMR Program boundary fill requires an execution facade");
    fill_boundary(field, *scalar_boundary_lane_);
  }

  void fill_boundary(field_type& field, const ExecutionLane& lane) const {
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
    boundary.fill(input);
    gradient_without_fill_(output, input, boundary.geometry());
  }
  void gradient(field_type& output, field_type& input, const scalar_boundary_session_type& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "AMR Program gradient");
    gradient(output, input, boundary);
  }

  void divergence(field_type& output, field_type& flux) const {
    if (output.ncomp() != 1 || flux.ncomp() != Dim)
      throw std::invalid_argument("AMR Program divergence requires one exact native vector field");
    require_same_layout_(output, flux, "AMR Program divergence");
    fill_boundary(flux);
    const Geometry<Dim> geom = geometry();
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
  void divergence(field_type& output, field_type& flux, const scalar_boundary_session_type&) const {
    divergence(output, flux);
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
    if (sys_block(program_owner) != 0)
      unavailable_("exact-ranked multi-block AMR history provider");
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
      if (manager.depth.at(key) != depth || manager.owner.at(key) != 0 ||
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
    manager.owner[key] = 0;
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
                            const field_type* active_cells) const {
    const field_type* expected = pointwise_active_mask(program_block, status);
    if (active_cells != expected)
      throw std::invalid_argument(
          "AMR Program pointwise status received a foreign active-cell mask");
    if (status.ncomp() < 1)
      throw std::invalid_argument("AMR Program pointwise status requires one component");
    const Real result = pops::reduce_max(status, 0);
    return result == -std::numeric_limits<Real>::infinity() ? Real(0) : result;
  }

  [[noreturn]] field_type& hierarchy_solution() const {
    unavailable_("direct hierarchy linear-solution provider");
  }
  field_type& linear_solution(field_type& fallback) const {
    require_same_layout_(fallback, state(0), "AMR Program linear solution");
    return fallback;
  }
  [[noreturn]] void stage_linear_initial_guess() const {
    unavailable_("direct hierarchy initial-guess provider");
  }
  [[noreturn]] void stage_linear_initial_guess(const field_type&) const {
    unavailable_("direct hierarchy initial-guess provider");
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

  bool has_boundary_linearization(int) const noexcept { return false; }
  template <class... Arguments>
  [[noreturn]] void boundary_residual_into_at(Arguments&&...) const {
    unavailable_("AMR iterate-dependent field boundary provider");
  }
  template <class... Arguments>
  [[noreturn]] void boundary_jvp_into_at(Arguments&&...) const {
    unavailable_("AMR iterate-dependent field boundary provider");
  }
  template <class... Arguments>
  [[noreturn]] void rhs_core_into_at(Arguments&&...) const {
    unavailable_("AMR matrix-free split residual provider");
  }
  template <class... Arguments>
  [[noreturn]] void rhs_jacvec_pair_into_at(Arguments&&...) const {
    unavailable_("AMR matrix-free coupled Jacobian provider");
  }
  template <class Function>
  void evaluate_with_field_state_at(const runtime::multiblock::BoundaryEvaluationPoint&,
                                    const std::string&, int, field_type&, const field_type&,
                                    Function&&) const {
    unavailable_("AMR perturbed field-state provider");
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
  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument("AMR Program rate identity must be non-negative");
  }
  void require_single_level_conservative_route_(std::string_view operation) const {
    if (runtime_->hierarchy().num_levels() != 1)
      throw std::runtime_error(std::string("AmrProgramContext::") + std::string(operation) +
                               " requires a prepared exact-ranked conservative multi-level "
                               "synchronization provider before any level state is advanced");
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

  void synchronize_resource_generation_() const {
    resource_epoch_ = runtime_->topology_epoch();
    resource_generation_ = runtime_->materialization_generation();
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
    return result;
  }

  static std::string history_key_(const std::string& name, int level) {
    if (name.empty() || level < 0)
      throw std::invalid_argument("AMR Program history key is invalid");
    return "pops.amr.level-history.v1/" + std::to_string(level) + "/" +
           std::to_string(name.size()) + ":" + name;
  }

  void require_history_owner_(int program_owner) const {
    if (program_owner < 0 || sys_block(program_owner) != 0)
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
      manager.initialized[key] = true;
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
    if (!manager.initialized.at(key)) {
      for (std::size_t slot = 1; slot < found->second.size(); ++slot)
        found->second[slot] = value;
    }
    manager.initialized[key] = true;
    manager.store_pending[key] = true;
    manager.slot_dt.at(key).front() = static_cast<Real>(current_dt_);
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
  mutable int logical_substep_ = 0;
  mutable ::pops::amr::Rational stage_time_{0, 1};
  mutable std::string primary_clock_;
  mutable ClockScheduleState clock_schedule_;
  mutable std::optional<ExecutionLane> scalar_boundary_lane_;
  mutable std::uint64_t boundary_generation_ = 0;
  mutable std::uint64_t resource_epoch_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t resource_generation_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t history_epoch_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t history_generation_ = std::numeric_limits<std::uint64_t>::max();
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::optional<OperatorEvaluationSnapshot> active_operator_snapshot_;
  mutable std::map<std::string, int> history_levels_;
  mutable std::map<ScratchKey, field_type> scratches_;
  mutable std::map<std::int64_t, GeneratedFieldRoute> generated_field_routes_;
  mutable PreparedVectorDistribution<Dim> vector_distribution_ =
      PreparedVectorDistribution<Dim>::distributed();
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
