/// @file
/// @brief Exact compile-time-ranked execution boundary for generated Uniform Programs.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/system.hpp>
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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {

/// A generated Program and its Uniform runtime have one immutable native rank.
///
/// The context never decodes a dimension tag, infers a missing axis, or substitutes a legacy
/// two-dimensional grid provider.  Every state, scratch, history and cache value is a
/// `MultiFab<Dim>` copied from an already-authenticated runtime layout.  Operations whose providers
/// have not yet crossed the exact-ranked boundary fail before touching storage.
template <int Dim>
class ProgramContext {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ProgramContext only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using runtime_type = System<Dim>;
  using field_type = MultiFab<Dim>;
  template <int Count>
  using provider_values_view_type = ProviderStorageView<Dim, Count>;
  using runtime_state_type = ProgramRuntimeState<Dim>;
  using scalar_boundary_session_type = PreparedScalarBoundarySession<Dim>;

  /// Immutable authentication token for one generated block-boundary invocation.  It retains no
  /// closure, state field, or mutable boundary image: System remains the sole owner of the exact
  /// prepared authority installed with the block.
  class PreparedBlockBoundarySession {
   public:
    PreparedBlockBoundarySession(const PreparedBlockBoundarySession&) = default;
    PreparedBlockBoundarySession& operator=(const PreparedBlockBoundarySession&) = default;

   private:
    friend class ProgramContext;

    PreparedBlockBoundarySession(const runtime_type* system, int runtime_block,
                                 runtime::multiblock::BoundaryEvaluationPoint point,
                                 const ExecutionLane& lane,
                                 std::shared_ptr<scalar_boundary_session_type> transport)
        : system_(system),
          runtime_block_(runtime_block),
          point_(std::move(point)),
          lane_(&lane),
          transport_(std::move(transport)) {
      if (!transport_)
        throw std::invalid_argument("prepared block boundary session requires halo transport");
    }

    [[nodiscard]] int runtime_block() const noexcept { return runtime_block_; }
    [[nodiscard]] const runtime::multiblock::BoundaryEvaluationPoint& point() const noexcept {
      return point_;
    }
    [[nodiscard]] const ExecutionLane& lane() const noexcept { return *lane_; }
    [[nodiscard]] const runtime_type* system() const noexcept { return system_; }
    [[nodiscard]] const scalar_boundary_session_type& transport() const noexcept {
      return *transport_;
    }

    const runtime_type* system_ = nullptr;
    int runtime_block_ = -1;
    runtime::multiblock::BoundaryEvaluationPoint point_{};
    const ExecutionLane* lane_ = nullptr;
    std::shared_ptr<scalar_boundary_session_type> transport_;
  };

  using block_boundary_session_type = PreparedBlockBoundarySession;

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

  /// Move-only exact child interval used by generated subcycle bodies.
  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const ProgramContext& owner, int iteration, int count)
        : owner_(&owner),
          prior_dt_(owner.current_dt_),
          prior_stage_(owner.stage_time_),
          prior_phase_begin_(owner.logical_phase_begin_),
          prior_phase_span_(owner.logical_phase_span_),
          prior_physical_time_offset_(owner.logical_physical_time_offset_) {
      if (count < 1 || iteration < 0 || iteration >= count || !std::isfinite(prior_dt_) ||
          !(prior_dt_ > 0.0))
        throw std::invalid_argument("Program logical evaluation scope is invalid");
      const double child_dt = prior_dt_ / static_cast<double>(count);
      const double child_offset =
          prior_physical_time_offset_ + static_cast<double>(iteration) * child_dt;
      if (!std::isfinite(child_dt) || !(child_dt > 0.0) || !std::isfinite(child_offset))
        throw std::overflow_error("Program logical evaluation child window is not finite");
      const amr::Rational child_fraction(iteration, count);
      const amr::Rational child_span(1, count);
      owner_->current_dt_ = child_dt;
      owner_->stage_time_ = amr::Rational(0, 1);
      owner_->logical_phase_begin_ = prior_phase_begin_ + prior_phase_span_ * child_fraction;
      owner_->logical_phase_span_ = prior_phase_span_ * child_span;
      owner_->logical_physical_time_offset_ = child_offset;
      owner_->active_operator_snapshot_.reset();
    }
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          prior_dt_(other.prior_dt_),
          prior_stage_(other.prior_stage_),
          prior_phase_begin_(other.prior_phase_begin_),
          prior_phase_span_(other.prior_phase_span_),
          prior_physical_time_offset_(other.prior_physical_time_offset_) {}
    LogicalEvaluationScope& operator=(LogicalEvaluationScope&&) = delete;
    ~LogicalEvaluationScope() noexcept { restore_(); }

    Real dt() const {
      if (owner_ == nullptr)
        throw std::logic_error("Program logical evaluation scope is no longer active");
      return static_cast<Real>(owner_->current_dt_);
    }

   private:
    void restore_() noexcept {
      if (owner_ == nullptr)
        return;
      owner_->current_dt_ = prior_dt_;
      owner_->stage_time_ = prior_stage_;
      owner_->logical_phase_begin_ = prior_phase_begin_;
      owner_->logical_phase_span_ = prior_phase_span_;
      owner_->logical_physical_time_offset_ = prior_physical_time_offset_;
      owner_->active_operator_snapshot_.reset();
      owner_ = nullptr;
    }

    const ProgramContext* owner_ = nullptr;
    double prior_dt_ = 0.0;
    amr::Rational prior_stage_{0, 1};
    amr::Rational prior_phase_begin_{0, 1};
    amr::Rational prior_phase_span_{1, 1};
    double prior_physical_time_offset_ = 0.0;
  };

  explicit ProgramContext(runtime_type* system) : system_(require_system_(system)) {}

  void install(std::function<void(double)> step) const {
    system_->install_program_step(std::move(step));
  }

  void begin_step(double dt) const {
    (void)prepared_execution_lane();
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramContext step requires a finite positive dt");
    current_dt_ = dt;
    stage_time_ = amr::Rational(0, 1);
    logical_phase_begin_ = amr::Rational(0, 1);
    logical_phase_span_ = amr::Rational(1, 1);
    logical_physical_time_offset_ = 0.0;
    active_operator_snapshot_.reset();
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
      throw std::invalid_argument("ProgramContext stage time is outside [0, 1]");
    stage_time_ = amr::Rational(numerator, denominator);
    active_operator_snapshot_.reset();
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("ProgramContext boundary evaluation has no prepared clock and dt");
    const amr::Rational evaluation_stage = logical_phase_begin_ + stage_time_ * logical_phase_span_;
    return {primary_clock_,
            static_cast<std::int64_t>(macro_step()),
            0,
            0,
            stage,
            evaluation_stage,
            current_dt_,
            physical_time() + logical_physical_time_offset_ + stage_time_.value() * current_dt_};
  }

  int n_blocks() const { return system_->n_blocks(); }

  /// Borrow the runtime-owned lane authenticated during Uniform boundary preparation. Generated
  /// implicit reports use this same lane for every reduction, diagnostic selection, and outcome.
  [[nodiscard]] const ExecutionLane& prepared_execution_lane() const {
    return system_->prepared_boundary_execution_lane();
  }

  int sys_block(int program_block) const {
    const std::vector<int>& map = system_->program_block_map();
    if (map.empty())
      throw std::runtime_error(
          "ProgramContext has no explicit program-to-runtime block map; positional identity is "
          "not supported");
    if (program_block < 0 || program_block >= static_cast<int>(map.size()))
      throw std::out_of_range("ProgramContext block is outside the authenticated map");
    const int runtime_block = map[static_cast<std::size_t>(program_block)];
    if (runtime_block < 0 || runtime_block >= system_->n_blocks())
      throw std::runtime_error("ProgramContext block map targets an absent runtime block");
    return runtime_block;
  }

  field_type& state(int program_block) const {
    return system_->block_state(sys_block(program_block));
  }
  /// Bind one native consumer's exact compact provider ABI for one local state patch.
  ///
  /// The consumer qid resolves late at the prepared System boundary; generated packages never
  /// embed global group/component addresses.  ``Count == 0`` returns an empty device-copyable view
  /// without reading a consumer plan or provider storage, even if another block owns providers.
  template <int Count>
  [[nodiscard]] provider_values_view_type<Count> provider_values_view(std::string_view consumer_qid,
                                                                      int program_block,
                                                                      std::size_t local_fab) const {
    static_assert(Count >= 0, "a provider consumer count cannot be negative");
    if constexpr (Count == 0) {
      (void)consumer_qid;
      (void)program_block;
      (void)local_fab;
      return {};
    } else {
      const field_type& state_field = state(program_block);
      const auto* const groups = system_->prepared_block_provider_storage_groups();
      const auto& plan = system_->prepared_auxiliary_consumer_plan(std::string(consumer_qid));
      runtime::system::require_pointwise_provider_groups<Dim, Count>(
          state_field, groups, &plan, "ProgramContext provider values");
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
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument(
          "ProgramContext scalar scratch requires positive components and non-negative ghosts");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    return persistent_scratch_(ScratchKind::Scalar, value_id, subslot, prototype, ncomp, ghosts);
  }

  field_type alloc_scalar_field(int ncomp = 1, int ghost_depth = 1) const {
    return scalar_field_like_(state(0), ncomp, ghost_depth);
  }

  void rhs_into(int program_block, field_type& state_value, field_type& rhs, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel_();
    const auto point = boundary_evaluation_point(rate_id);
    const int runtime_block = sys_block(program_block);
    if (system_->requires_block_boundary_session(runtime_block)) {
      const ExecutionLane& lane = system_->prepared_boundary_execution_lane();
      auto boundary = prepare_block_boundary_session(program_block, state_value, point, lane);
      system_->block_rhs_into_at_prepared(
          point, runtime_block, state_value, rhs, boundary->system(), boundary->runtime_block(),
          boundary->point(), boundary->lane(), boundary->transport());
      return;
    }
    system_->block_rhs_into_at(point, runtime_block, state_value, rhs);
  }

  void neg_div_flux_default_into(int program_block, field_type& state_value, field_type& rhs,
                                 int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel_();
    system_->block_neg_div_flux_into_at(boundary_evaluation_point(rate_id),
                                        sys_block(program_block), state_value, rhs);
  }

  void source_default_into(int program_block, field_type& state_value, field_type& rhs) const {
    count_kernel_();
    system_->block_source_into(sys_block(program_block), state_value, rhs);
  }

  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    require_rate_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("ProgramContext RHS group cannot be empty");
    std::vector<int> blocks;
    std::vector<field_type*> states;
    std::vector<field_type*> residuals;
    std::vector<int> flux_only;
    blocks.reserve(requests.size());
    states.reserve(requests.size());
    residuals.reserve(requests.size());
    flux_only.reserve(requests.size());
    std::vector<int> rates;
    rates.reserve(requests.size());
    for (const RhsGroupRequest& request : requests) {
      require_rate_identity_(request.rate_id);
      if (request.rate_id == group_id || request.state == nullptr || request.rhs == nullptr ||
          (request.flux_only != 0 && request.flux_only != 1) ||
          std::find(rates.begin(), rates.end(), request.rate_id) != rates.end())
        throw std::invalid_argument("ProgramContext RHS group contains an invalid request");
      rates.push_back(request.rate_id);
      blocks.push_back(sys_block(request.block));
      states.push_back(request.state);
      residuals.push_back(request.rhs);
      flux_only.push_back(request.flux_only);
    }
    count_kernel_(static_cast<std::int64_t>(requests.size()));
    system_->block_rhs_group(boundary_evaluation_point(group_id), blocks, states, residuals,
                             flux_only);
  }

  void require_cartesian_generated_operator(int program_block, const std::string& operation) const {
    system_->require_cartesian_generated_operator(sys_block(program_block), operation);
  }

  /// Fill the state and shared auxiliary halos through the exact block package before a generated
  /// pointwise stencil reads neighbouring cells.  The Program contributes no topology or boundary
  /// policy; both are retained by `SystemBlockStore<Dim>`.
  void prepare_generated_state(int program_block, field_type& state_value, int rate_id) const {
    require_rate_identity_(rate_id);
    system_->block_prepare_generated_state_at(boundary_evaluation_point(rate_id),
                                              sys_block(program_block), state_value);
  }

  /// Assemble the centered negative divergence of one already-materialized named flux field per
  /// native axis.  Axis count and storage rank are the same compile-time constant, so 1D/2D/3D use
  /// one algorithm and no runtime dimension selector.
  void neg_div_named_flux_into(field_type& rhs, const std::array<field_type*, Dim>& fluxes) const {
    const Geometry<Dim> geometry = system_->prepared_block_geometry();
    for (int axis = 0; axis < Dim; ++axis) {
      const field_type* flux = fluxes[static_cast<std::size_t>(axis)];
      if (flux == nullptr || flux->layout() != rhs.layout() ||
          flux->distribution() != rhs.distribution() || flux->local_rank() != rhs.local_rank() ||
          flux->ncomp() != rhs.ncomp() || flux->local_size() != rhs.local_size() ||
          flux->ghosts()[axis] < 1)
        throw std::invalid_argument(
            "ProgramContext named flux does not match the exact ranked residual layout");
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
                          (Real(2) * geometry.spacing(axis));
          }
          output(cell, component) = -divergence;
        }
      });
    }
    count_kernel_();
  }

  void apply_projection(int program_block, field_type& state_value) const {
    count_kernel_();
    system_->block_project(sys_block(program_block), state_value);
  }

  Real max_wave_speed(int program_block, const field_type& state_value) const {
    return system_->block_max_speed(sys_block(program_block), state_value);
  }

  Real hmin() const { return system_->cfl_min_dx(); }
  RuntimeParams program_params(int program_block) const {
    return system_->program_params(program_block);
  }

  void axpy(field_type& destination, Real factor, const field_type& source) const {
    count_kernel_();
    pops::saxpy(destination, factor, source);
  }

  void axpy(field_type& destination, Real factor, const field_type& source, Real,
            std::initializer_list<ExactCoefficientTerm>) const {
    axpy(destination, factor, source);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right) const {
    count_kernel_();
    pops::lincomb(destination, left_factor, left, right_factor, right);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right, Real, std::initializer_list<ExactCoefficientTerm>,
               std::initializer_list<ExactCoefficientTerm>) const {
    lincomb(destination, left_factor, left, right_factor, right);
  }

  void commit_many(std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
    std::vector<field_type*> targets;
    std::vector<std::optional<field_type>> snapshots;
    targets.reserve(commits.size());
    snapshots.reserve(commits.size());
    for (const auto& [target, source] : commits) {
      if (target == nullptr || source == nullptr)
        throw std::invalid_argument("ProgramContext commit contains null storage");
      if (std::find(targets.begin(), targets.end(), target) != targets.end())
        throw std::invalid_argument("ProgramContext commit contains a duplicate target");
      require_same_field_contract_(*target, *source, "ProgramContext commit");
      targets.push_back(target);
      snapshots.emplace_back(target == source ? std::nullopt : std::optional<field_type>(*source));
    }
    std::size_t candidate = 0;
    for (const auto& [target, source] : commits) {
      const field_type& value = snapshots[candidate] ? *snapshots[candidate] : *source;
      for (int block = 0; block < system_->n_blocks(); ++block)
        if (target == &system_->block_state(block)) {
          system_->validate_program_state_publication_candidate_(block, value,
                                                                 prepared_execution_lane());
          break;
        }
      ++candidate;
    }
    candidate = 0;
    for (const auto& [target, source] : commits) {
      if (target != source)
        *target = std::move(*snapshots[candidate]);
      ++candidate;
    }
  }

  void apply_coupling_operators(Real dt,
                                std::initializer_list<CouplingStateOverride> candidates) const {
    std::vector<field_type*> runtime_states(static_cast<std::size_t>(system_->n_blocks()), nullptr);
    for (const CouplingStateOverride& candidate : candidates) {
      const int block = sys_block(candidate.program_block);
      if (candidate.state == nullptr || runtime_states[static_cast<std::size_t>(block)] != nullptr)
        throw std::invalid_argument("ProgramContext coupling candidates are incomplete or aliased");
      runtime_states[static_cast<std::size_t>(block)] = candidate.state;
    }
    if (std::find(runtime_states.begin(), runtime_states.end(), nullptr) != runtime_states.end())
      throw std::invalid_argument("ProgramContext coupling requires every runtime block candidate");
    count_kernel_(static_cast<std::int64_t>(system_->apply_coupling_operators(dt, runtime_states)));
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

  Geometry<Dim> geometry() const { return system_->prepared_block_geometry(); }

  field_type& assembly_target(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("ProgramContext assembly target requires an identity");
    return field;
  }

  field_type& assembly_source(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("ProgramContext assembly source requires an identity");
    return field;
  }

  std::shared_ptr<scalar_boundary_session_type> prepare_mesh_boundary_session(
      field_type& prototype, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "Program scalar boundary preparation");
    return scalar_boundary_session_type::prepare(geometry(), scalar_boundary_topology_(), prototype,
                                                 lane, next_scalar_boundary_generation_());
  }

  std::shared_ptr<block_boundary_session_type> prepare_block_boundary_session(
      int program_block, field_type& prototype,
      const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "Program block boundary preparation");
    int runtime_block = -1;
    std::exception_ptr local_error;
    try {
      runtime_block = sys_block(program_block);
      require_boundary_point_(point, "Program prepared block boundary");
      require_same_field_contract_(prototype, system_->block_state(runtime_block),
                                   "Program block boundary prototype");
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program prepared block boundary session failed collectively");
    }
    ExactContractBuilder contract;
    contract.text("pops.program.prepared-block-boundary-session")
        .scalar(std::int32_t{Dim})
        .scalar(std::int32_t{runtime_block})
        .text(lane.identity())
        .text(point.clock)
        .scalar(point.tick)
        .scalar(point.level)
        .scalar(point.substep)
        .scalar(point.stage)
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("program-prepared-block-boundary-session"), contract.view()}}, lane))
      throw std::runtime_error("Program prepared block boundary session differs across MPI ranks");
    auto transport = scalar_boundary_session_type::prepare_block(
        geometry(), scalar_boundary_topology_(), prototype, lane,
        next_scalar_boundary_generation_());
    std::shared_ptr<block_boundary_session_type> session;
    local_error = nullptr;
    try {
      session = std::shared_ptr<block_boundary_session_type>(new block_boundary_session_type(
          system_, runtime_block, point, lane, std::move(transport)));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "Program prepared block boundary session allocation failed collectively");
    }
    return session;
  }

  bool has_boundary_linearization(int program_block) const {
    return system_->has_block_boundary_linearization(sys_block(program_block));
  }

  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                        int program_block, field_type& state, field_type& residual, bool flux_only,
                        const block_boundary_session_type& boundary) const {
    const int runtime_block =
        resolve_prepared_program_block_(program_block, boundary.lane(), "Program core RHS block");
    system_->block_rhs_core_into_at(point, runtime_block, state, residual, flux_only,
                                    boundary.system(), boundary.runtime_block(), boundary.point(),
                                    boundary.lane(), boundary.transport());
  }

  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 int program_block, field_type& state, field_type& residual,
                                 const block_boundary_session_type& boundary) const {
    const int runtime_block = resolve_prepared_program_block_(program_block, boundary.lane(),
                                                              "Program boundary residual block");
    system_->block_boundary_residual_into_at(
        point, runtime_block, state, residual, boundary.system(), boundary.runtime_block(),
        boundary.point(), boundary.lane(), boundary.transport());
  }

  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            int program_block, field_type& state, const field_type& direction,
                            field_type& result, const block_boundary_session_type& boundary) const {
    const int runtime_block = resolve_prepared_program_block_(program_block, boundary.lane(),
                                                              "Program boundary JVP block");
    system_->block_boundary_jvp_into_at(point, runtime_block, state, direction, result,
                                        boundary.system(), boundary.runtime_block(),
                                        boundary.point(), boundary.lane(), boundary.transport());
  }

  void fill_boundary(field_type& field) const {
    auto session = scalar_boundary_session_type::prepare(geometry(), scalar_boundary_topology_(),
                                                         field, prepared_execution_lane(),
                                                         next_scalar_boundary_generation_());
    session->fill(field);
  }

  void fill_boundary(field_type& field, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "Program boundary fill");
    auto session = scalar_boundary_session_type::prepare(
        geometry(), scalar_boundary_topology_(), field, lane, next_scalar_boundary_generation_());
    session->fill(field);
  }

  void laplacian(field_type& output, field_type& input) const {
    auto boundary = prepare_mesh_boundary_session(input, prepared_execution_lane());
    laplacian(output, input, *boundary);
  }

  void laplacian(field_type& output, field_type& input,
                 const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "Program Laplacian boundary");
    require_scalar_stencil_(output, input, 1, "Program Laplacian");
    boundary.fill(input);
    const Geometry<Dim> geom = boundary.geometry();
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
                 const scalar_boundary_session_type& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program Laplacian");
    laplacian(output, input, boundary);
  }

  void gradient(field_type& output, field_type& input) const {
    auto boundary = prepare_mesh_boundary_session(input, prepared_execution_lane());
    gradient(output, input, *boundary);
  }

  void gradient(field_type& output, field_type& input,
                const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "Program gradient boundary");
    require_scalar_stencil_(output, input, Dim, "Program gradient");
    boundary.fill(input);
    const Geometry<Dim> geom = boundary.geometry();
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

  void gradient(field_type& output, field_type& input, const scalar_boundary_session_type& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program gradient");
    gradient(output, input, boundary);
  }

  void divergence(field_type& output, field_type& flux) const {
    auto boundary = prepare_mesh_boundary_session(flux, prepared_execution_lane());
    divergence(output, flux, *boundary);
  }

  void divergence(field_type& output, field_type& flux,
                  const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "Program divergence boundary");
    if (output.ncomp() != 1 || flux.ncomp() != Dim || output.layout() != flux.layout() ||
        output.distribution() != flux.distribution() || output.local_rank() != flux.local_rank() ||
        output.local_size() != flux.local_size())
      throw std::invalid_argument(
          "Program divergence requires one exact-ranked vector flux and scalar output");
    for (int axis = 0; axis < Dim; ++axis) {
      if (flux.ghosts()[axis] < 1)
        throw std::invalid_argument(
            "Program divergence flux requires one ghost on every native axis");
    }
    boundary.fill(flux);
    const Geometry<Dim> geom = boundary.geometry();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<const Real, Dim> value = std::as_const(flux).fab(local).view();
      const FieldView<Real, Dim> result = output.fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        Real image = Real(0);
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell;
          Index<Dim> upper = cell;
          --lower[axis];
          ++upper[axis];
          image += (value(upper, axis) - value(lower, axis)) / (Real(2) * geom.spacing(axis));
        }
        result(cell, 0) = image;
      });
    }
    count_kernel_();
  }

  void divergence(field_type& output, field_type& flux,
                  const scalar_boundary_session_type& boundary,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program divergence");
    divergence(output, flux, boundary);
  }

  void pack_vector(field_type& output, const std::array<const field_type*, Dim>& components) const {
    if (output.ncomp() != Dim)
      throw std::invalid_argument("Program vector output must carry one component per native axis");
    for (int axis = 0; axis < Dim; ++axis) {
      const field_type* component = components[static_cast<std::size_t>(axis)];
      if (component == nullptr || component->ncomp() != 1 ||
          component->layout() != output.layout() ||
          component->distribution() != output.distribution() ||
          component->local_rank() != output.local_rank() ||
          component->local_size() != output.local_size())
        throw std::invalid_argument(
            "Program vector component does not match the exact scalar layout");
    }
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      std::array<FieldView<const Real, Dim>, Dim> values{};
      for (int axis = 0; axis < Dim; ++axis)
        values[static_cast<std::size_t>(axis)] =
            std::as_const(*components[static_cast<std::size_t>(axis)]).fab(local).view();
      const FieldView<Real, Dim> result = output.fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int axis = 0; axis < Dim; ++axis)
          result(cell, axis) = values[static_cast<std::size_t>(axis)](cell, 0);
      });
    }
    count_kernel_();
  }

  void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                        const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "Program tensor Laplacian boundary");
    require_scalar_stencil_(output, input, 1, "Program tensor Laplacian");
    require_tensor_stencil_(input, tensor, "Program tensor Laplacian");
    boundary.fill(input);
    const Geometry<Dim> geom = boundary.geometry();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<Real, Dim> result = output.fab(local).view();
      const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
      const FieldView<const Real, Dim> coefficient = std::as_const(tensor).fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        Real image = Real(0);
        for (int row = 0; row < Dim; ++row) {
          Index<Dim> lower_row = cell;
          Index<Dim> upper_row = cell;
          --lower_row[row];
          ++upper_row[row];
          Real lower_flux = Real(0);
          Real upper_flux = Real(0);
          for (int column = 0; column < Dim; ++column) {
            const int component = row * Dim + column;
            if (column == row) {
              const Real center_coefficient = coefficient(cell, component);
              const Real lower_coefficient = coefficient(lower_row, component);
              const Real upper_coefficient = coefficient(upper_row, component);
              const Real lower_sum = center_coefficient + lower_coefficient;
              const Real upper_sum = center_coefficient + upper_coefficient;
              const Real lower_face = lower_sum != Real(0) ? Real(2) * center_coefficient *
                                                                 lower_coefficient / lower_sum
                                                           : Real(0);
              const Real upper_face = upper_sum != Real(0) ? Real(2) * center_coefficient *
                                                                 upper_coefficient / upper_sum
                                                           : Real(0);
              lower_flux += lower_face * (value(cell, 0) - value(lower_row, 0)) / geom.spacing(row);
              upper_flux += upper_face * (value(upper_row, 0) - value(cell, 0)) / geom.spacing(row);
            } else {
              Index<Dim> lower_column = cell;
              Index<Dim> upper_column = cell;
              Index<Dim> lower_row_lower_column = lower_row;
              Index<Dim> lower_row_upper_column = lower_row;
              Index<Dim> upper_row_lower_column = upper_row;
              Index<Dim> upper_row_upper_column = upper_row;
              --lower_column[column];
              ++upper_column[column];
              --lower_row_lower_column[column];
              ++lower_row_upper_column[column];
              --upper_row_lower_column[column];
              ++upper_row_upper_column[column];
              const Real lower_face =
                  Real(0.5) * (coefficient(cell, component) + coefficient(lower_row, component));
              const Real upper_face =
                  Real(0.5) * (coefficient(cell, component) + coefficient(upper_row, component));
              const Real tangent_scale = Real(4) * geom.spacing(column);
              const Real lower_tangent =
                  (value(upper_column, 0) - value(lower_column, 0) +
                   value(lower_row_upper_column, 0) - value(lower_row_lower_column, 0)) /
                  tangent_scale;
              const Real upper_tangent =
                  (value(upper_column, 0) - value(lower_column, 0) +
                   value(upper_row_upper_column, 0) - value(upper_row_lower_column, 0)) /
                  tangent_scale;
              lower_flux += lower_face * lower_tangent;
              upper_flux += upper_face * upper_tangent;
            }
          }
          image += (upper_flux - lower_flux) / geom.spacing(row);
        }
        result(cell, 0) = image;
      });
    }
    count_kernel_();
  }

  void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                        const scalar_boundary_session_type& boundary,
                        const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program tensor Laplacian");
    tensor_laplacian(output, input, tensor, boundary);
  }

  void register_history(const std::string& name, int lag, int ncomp = -1, int program_owner = -1,
                        const std::string& state_identity = {},
                        const std::string& space_identity = {},
                        const std::string& clock_identity = {},
                        const std::string& interpolation_identity = {}) const {
    if (name.empty() || lag < 1)
      throw std::invalid_argument("ProgramContext history requires a name and positive lag");
    const int owner = program_owner < 0 ? 0 : sys_block(program_owner);
    const field_type& prototype = system_->block_state(owner);
    const int components = ncomp < 0 ? prototype.ncomp() : ncomp;
    if (components < 1)
      throw std::invalid_argument("ProgramContext history component count must be positive");
    auto& history = runtime_state().hist_;
    const int ring_depth = lag + 1;
    const auto existing = history.histories.find(name);
    if (existing != history.histories.end()) {
      if (history.depth.at(name) != ring_depth || history.owner.at(name) != owner ||
          existing->second.front().ncomp() != components ||
          history.state_identity.at(name) != state_identity ||
          history.space_identity.at(name) != space_identity ||
          history.clock_identity.at(name) != clock_identity ||
          history.interpolation_identity.at(name) != interpolation_identity)
        throw std::runtime_error("ProgramContext history identity changed after registration");
      return;
    }
    std::vector<field_type> ring;
    ring.reserve(static_cast<std::size_t>(ring_depth));
    for (int slot = 0; slot < ring_depth; ++slot)
      ring.push_back(make_scratch_(prototype, components, prototype.ghosts()));
    history.histories.emplace(name, std::move(ring));
    history.depth[name] = ring_depth;
    history.initialized[name] = false;
    history.fill_count[name] = 0;
    history.store_pending[name] = false;
    history.owner[name] = owner;
    history.state_identity[name] = state_identity;
    history.space_identity[name] = space_identity;
    history.clock_identity[name] = clock_identity;
    history.interpolation_identity[name] = interpolation_identity;
    history.slot_dt[name] = std::vector<Real>(static_cast<std::size_t>(ring_depth), Real(0));
  }

  field_type& history(const std::string& name, int lag, int ncomp = -1) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(name))
      throw std::out_of_range("ProgramContext history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramContext history component contract differs");
    if (!manager.initialized.at(name))
      throw std::runtime_error("ProgramContext history has not been initialized");
    return result;
  }

  field_type& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(name))
      throw std::out_of_range("ProgramContext history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramContext history component contract differs");
    return result;
  }

  void store_history(const std::string& name, const field_type& value) const {
    store_history_(name, value, std::nullopt);
  }
  void store_history(const std::string& name, const field_type& value, double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramContext history dt must be finite and positive");
    store_history_(name, value, static_cast<Real>(dt));
  }
  void rotate_histories() const { runtime_state().hist_.rotate(); }
  void rotate_histories(const std::string& clock) const { runtime_state().hist_.rotate(clock); }

  bool cache_should_update(int node_id, int every_n) const {
    const bool due = runtime_state().cache_.is_due(node_id, macro_step(), every_n);
    runtime_state().profiler_.count(due ? "cache_misses" : "cache_hits");
    return due;
  }
  void cache_store_scratch(int node_id, const field_type& scratch) const {
    runtime_state().cache_.store(node_id, scratch, macro_step());
  }
  void cache_restore_scratch(int node_id, field_type& scratch) const {
    runtime_state().cache_.restore_into(node_id, scratch);
  }
  void cache_accumulate_dt(int node_id, Real dt) const {
    runtime_state().cache_.accumulate_dt(node_id, dt);
  }
  Real cache_effective_dt(int node_id, Real dt) const {
    return runtime_state().cache_.effective_dt(node_id, dt);
  }

  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
  }
  bool schedule_is_due(int node_id, int every_n, ScheduleDomainKind kind, const std::string& clock,
                       const std::string& stage_identity, int level) const {
    if (node_id < 0 || every_n <= 0)
      throw std::invalid_argument("ProgramContext schedule has an invalid node or period");
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
      throw std::invalid_argument("ProgramContext schedule decision has an invalid node");
    return runtime_state().profiler_.schedule_decision(due, cache_backed);
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

  int macro_step() const { return system_->macro_step(); }
  Real physical_time() const { return static_cast<Real>(system_->time()); }

  void record_scalar(const std::string& name, Real value) const {
    runtime_state().record_diagnostic(name, value);
  }
  void record_balance_term(const std::string& route, const std::string& term, Real value) const {
    system_->record_program_balance_term(route, term, value);
  }
  bool balance_consumer_is_due(const std::string& contract, const std::string& route,
                               int every_n) const {
    return system_->program_balance_consumer_is_due(contract, route, every_n);
  }
  void note_automatic_balance_capture_due(bool due) const {
    runtime_state().note_automatic_balance_capture_due(due, "ProgramContext");
  }
  void note_step_projection(const std::string& name) const {
    runtime_state().note_step_projection(name);
  }

  void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    runtime_state().profiler_.record(name, std::chrono::duration<double>(elapsed).count());
  }

  runtime_state_type& runtime_state() const { return system_->program_runtime_state_(); }

  const PreparedVectorDistribution<Dim>& program_resource_vector_distribution() const {
    return PreparedVectorDistribution<Dim>::Distributed;
  }

  int program_resource_field_level() const noexcept { return 0; }

  void configure_program_resource_field_nullspace(FieldNullspacePlan<Dim>& plan) const {
    const Geometry<Dim> geom = geometry();
    Real cell_measure = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      cell_measure *= geom.spacing(axis);
    if (!std::isfinite(cell_measure) || !(cell_measure > Real(0)))
      throw std::runtime_error("Program nullspace cell measure is invalid");
    for (FieldNullspaceBasis<Dim>& basis : plan.bases)
      basis.cell_measure = {cell_measure};
  }

  SolveOutcome solve_prepared_linear(const PreparedAffineLinearProblem<Dim>& problem,
                                     KrylovWorkspace<Dim>& workspace, field_type& solution,
                                     const field_type& rhs,
                                     const KrylovControls<Dim>& controls) const {
    require_prepared_lane_(workspace.execution_lane(), "Program prepared linear solve");
    return pops::solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls);
  }

  OperatorEvaluationSnapshot operator_evaluation_snapshot(OperatorFingerprint authority,
                                                          const field_type& prototype,
                                                          OperatorFingerprint resources) const {
    if (operator_snapshot_revision_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program operator snapshot revision exhausted uint64_t");
    const OperatorFingerprint topology = operator_topology_(prototype);
    OperatorEvaluationSnapshot snapshot =
        current_operator_snapshot_(authority, topology, resources, ++operator_snapshot_revision_);
    if (!snapshot.valid())
      throw std::runtime_error("Program produced an invalid exact operator snapshot");
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
    system_->set_field_boundary_kernel(provider_slot, kernel);
  }

  void set_field_logical_timepoint(const std::string& provider_slot,
                                   const FieldLogicalTimePoint& point) const {
    system_->set_field_logical_timepoint(provider_slot, point);
  }

  void set_field_boundary_parameters(const std::string& provider_slot,
                                     const std::vector<double>& parameters) const {
    system_->set_field_boundary_parameters(provider_slot, parameters);
  }

  [[nodiscard]] SolveOutcome solve_fields() const { return system_->solve_fields(); }

  [[nodiscard]] SolveOutcome solve_fields_from_state(int program_block, field_type& stage) const {
    const int runtime_block = sys_block(program_block);
    require_program_stage_(program_block, runtime_block, stage);
    return system_->solve_fields_from_state(runtime_block, stage);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_state_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int program_block, field_type& stage) const {
    require_boundary_point_(point, "Program single-state field solve");
    if (provider_slot.empty())
      throw std::invalid_argument("Program field solve requires an exact provider slot");
    const int runtime_block = sys_block(program_block);
    require_program_stage_(program_block, runtime_block, stage);
    return system_->solve_fields_from_state_at(point, provider_slot, runtime_block, stage);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_blocks(
      const std::vector<const field_type*>& program_stages) const {
    const std::vector<int>& block_map = require_program_block_map_();
    if (program_stages.size() != block_map.size())
      throw std::invalid_argument(
          "Program simultaneous field solve requires one slot per Program block");

    std::vector<const field_type*> runtime_stages(static_cast<std::size_t>(system_->n_blocks()),
                                                  nullptr);
    std::vector<const field_type*> unique_stages;
    unique_stages.reserve(program_stages.size());
    for (std::size_t program = 0; program < program_stages.size(); ++program) {
      const int runtime_block = sys_block(static_cast<int>(program));
      const field_type* const stage = program_stages[program];
      if (stage == nullptr)
        continue;
      require_unaliased_stage_(unique_stages, *stage);
      require_program_stage_(static_cast<int>(program), runtime_block, *stage);
      if (runtime_stages[static_cast<std::size_t>(runtime_block)] != nullptr)
        throw std::invalid_argument(
            "Program simultaneous field solve maps two stages to one runtime block");
      runtime_stages[static_cast<std::size_t>(runtime_block)] = stage;
      unique_stages.push_back(stage);
    }
    return system_->solve_fields_from_blocks(runtime_stages);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_blocks_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t value_id,
      std::string_view field, std::initializer_list<FieldStageOverride> overrides) const {
    require_boundary_point_(point, "Program simultaneous field solve");
    if (value_id < 0)
      throw std::invalid_argument(
          "Program simultaneous field solve requires a non-negative IR identity");
    if (field.empty())
      throw std::invalid_argument("Program field solve requires an exact provider slot");
    if (overrides.size() == 0)
      throw std::invalid_argument(
          "Program simultaneous field solve requires at least one stage override");

    const std::vector<int>& block_map = require_program_block_map_();
    GeneratedFieldRoute candidate;
    candidate.field.assign(field.data(), field.size());
    candidate.program_to_system.assign(block_map.begin(), block_map.end());
    candidate.program_blocks.reserve(overrides.size());
    std::vector<const field_type*> runtime_stages(static_cast<std::size_t>(system_->n_blocks()),
                                                  nullptr);
    std::vector<const field_type*> unique_stages;
    unique_stages.reserve(overrides.size());
    for (const FieldStageOverride& override_value : overrides) {
      if (override_value.program_block < 0 ||
          static_cast<std::size_t>(override_value.program_block) >= block_map.size())
        throw std::out_of_range("Program simultaneous field solve Program block is out of range");
      if (override_value.state == nullptr)
        throw std::invalid_argument(
            "Program simultaneous field solve stage override cannot be null");
      if (std::find(candidate.program_blocks.begin(), candidate.program_blocks.end(),
                    override_value.program_block) != candidate.program_blocks.end())
        throw std::invalid_argument(
            "Program simultaneous field solve contains a duplicate Program block");
      require_unaliased_stage_(unique_stages, *override_value.state);
      const int runtime_block = sys_block(override_value.program_block);
      require_program_stage_(override_value.program_block, runtime_block, *override_value.state);
      if (runtime_stages[static_cast<std::size_t>(runtime_block)] != nullptr)
        throw std::invalid_argument(
            "Program simultaneous field solve maps two stages to one runtime block");
      candidate.program_blocks.push_back(override_value.program_block);
      runtime_stages[static_cast<std::size_t>(runtime_block)] = override_value.state;
      unique_stages.push_back(override_value.state);
    }

    const auto existing = generated_field_routes_.find(value_id);
    if (existing == generated_field_routes_.end()) {
      generated_field_routes_.emplace(value_id, std::move(candidate));
    } else if (existing->second != candidate) {
      throw std::logic_error(
          "Program simultaneous field solve IR identity changed its qualified route");
    }

    const std::string& provider_slot = generated_field_routes_.at(value_id).field;
    system_->prepare_named_field_publication_storage_(provider_slot);
    return system_->run_field_publication_outcome_([this, &point, &provider_slot, &runtime_stages] {
      return system_->solve_fields_from_blocks_at_in_place_(point, provider_slot, runtime_stages);
    });
  }

 private:
  enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
  using ScratchKey = std::tuple<ScratchKind, std::int64_t, int>;

  struct GeneratedFieldRoute {
    std::string field;
    std::vector<int> program_to_system;
    std::vector<int> program_blocks;

    friend bool operator==(const GeneratedFieldRoute&, const GeneratedFieldRoute&) = default;
  };

  static runtime_type* require_system_(runtime_type* system) {
    if (system == nullptr)
      throw std::invalid_argument("ProgramContext requires a non-null ranked System");
    return system;
  }

  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument("ProgramContext rate identity must be non-negative");
  }

  void require_prepared_lane_(const ExecutionLane& lane, const char* operation) const {
    const ExecutionLane& prepared = prepared_execution_lane();
    if (all_reduce_max(&lane == &prepared ? 0L : 1L, prepared) != 0)
      throw std::invalid_argument(std::string(operation) +
                                  " requires the context's authenticated execution lane");
  }

  static void require_same_field_contract_(const field_type& left, const field_type& right,
                                           const char* operation) {
    if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
        left.local_rank() != right.local_rank() || left.ncomp() != right.ncomp() ||
        left.ghosts() != right.ghosts())
      throw std::invalid_argument(std::string(operation) +
                                  " requires the same exact ranked field contract");
  }

  int resolve_prepared_program_block_(int program_block, const ExecutionLane& lane,
                                      const char* operation) const {
    int runtime_block = -1;
    std::exception_ptr local_error;
    try {
      runtime_block = sys_block(program_block);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(std::string(operation) + " failed collectively");
    }
    ExactContractBuilder contract;
    contract.text(operation)
        .scalar(std::int32_t{Dim})
        .scalar(std::int32_t{program_block})
        .scalar(std::int32_t{runtime_block})
        .text(lane.identity());
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("program-prepared-boundary-route"), contract.view()}}, lane))
      throw std::runtime_error(std::string(operation) + " differs across MPI ranks");
    return runtime_block;
  }

  const std::vector<int>& require_program_block_map_() const {
    const std::vector<int>& block_map = system_->program_block_map();
    if (block_map.empty())
      throw std::runtime_error(
          "Program simultaneous field solve requires an explicit program-to-runtime block map");
    std::vector<int> runtime_blocks;
    runtime_blocks.reserve(block_map.size());
    for (std::size_t program = 0; program < block_map.size(); ++program) {
      const int runtime_block = sys_block(static_cast<int>(program));
      if (std::find(runtime_blocks.begin(), runtime_blocks.end(), runtime_block) !=
          runtime_blocks.end())
        throw std::runtime_error(
            "Program simultaneous field solve block map contains duplicate runtime routes");
      runtime_blocks.push_back(runtime_block);
    }
    return block_map;
  }

  void require_program_stage_(int program_block, int runtime_block, const field_type& stage) const {
    const field_type& live = system_->block_state(runtime_block);
    require_same_field_contract_(stage, live, "Program field stage");
    for (int other = 0; other < system_->n_blocks(); ++other) {
      if (other != runtime_block && &stage == &system_->block_state(other))
        throw std::invalid_argument(
            "Program field stage cannot borrow another runtime block's live state");
    }
    if (sys_block(program_block) != runtime_block)
      throw std::logic_error("Program field stage route changed during validation");
  }

  static void require_unaliased_stage_(const std::vector<const field_type*>& stages,
                                       const field_type& candidate) {
    if (std::find(stages.begin(), stages.end(), &candidate) != stages.end())
      throw std::invalid_argument(
          "Program simultaneous field solve cannot alias one stage across blocks");
  }

  static void require_boundary_point_(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                      const char* operation) {
    if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
        point.stage < 0 || !std::isfinite(point.dt) || !(point.dt > 0.0) ||
        !std::isfinite(point.physical_time))
      throw std::invalid_argument(std::string(operation) +
                                  " requires a complete boundary evaluation point");
  }

  static void require_scalar_stencil_(const field_type& output, const field_type& input,
                                      int output_components, const char* operation) {
    if (output_components < 1 || output.ncomp() != output_components || input.ncomp() != 1 ||
        output.layout() != input.layout() || output.distribution() != input.distribution() ||
        output.local_rank() != input.local_rank() || output.local_size() != input.local_size())
      throw std::invalid_argument(std::string(operation) +
                                  " fields do not share the exact scalar stencil layout");
    for (int axis = 0; axis < Dim; ++axis)
      if (input.ghosts()[axis] < 1)
        throw std::invalid_argument(std::string(operation) +
                                    " input requires one ghost on every native axis");
  }

  static void require_tensor_stencil_(const field_type& input, const field_type& tensor,
                                      const char* operation) {
    if (tensor.ncomp() != Dim * Dim || input.layout() != tensor.layout() ||
        input.distribution() != tensor.distribution() ||
        input.local_rank() != tensor.local_rank() || input.local_size() != tensor.local_size())
      throw std::invalid_argument(std::string(operation) +
                                  " tensor must be one row-major Dim*Dim exact-ranked field");
    for (int axis = 0; axis < Dim; ++axis)
      if (tensor.ghosts()[axis] < 1)
        throw std::invalid_argument(std::string(operation) +
                                    " tensor requires one ghost on every native axis");
  }

  BoundaryTopology<Dim> scalar_boundary_topology_() const {
    return BoundaryTopology<Dim>::axis_periodic(system_->prepared_block_periodicity());
  }

  std::uint64_t next_scalar_boundary_generation_() const {
    if (scalar_boundary_generation_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program scalar boundary generation exhausted uint64_t");
    return ++scalar_boundary_generation_;
  }

  OperatorFingerprint operator_topology_(const field_type& prototype) const {
    OperatorFingerprint fingerprint =
        ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
    ::pops::detail::fingerprint_geometry(fingerprint, geometry());
    const auto periodicity = system_->prepared_block_periodicity();
    ::pops::detail::fingerprint_mix(fingerprint, "uniform-cartesian-topology");
    for (int axis = 0; axis < Dim; ++axis)
      ::pops::detail::fingerprint_mix(
          fingerprint, static_cast<std::uint64_t>(periodicity[static_cast<std::size_t>(axis)]));
    return fingerprint;
  }

  OperatorEvaluationSnapshot current_operator_snapshot_(OperatorFingerprint authority,
                                                        OperatorFingerprint topology,
                                                        OperatorFingerprint resources,
                                                        std::uint64_t revision) const {
    const amr::Rational evaluation_stage = logical_phase_begin_ + stage_time_ * logical_phase_span_;
    const double evaluation_time =
        physical_time() + logical_physical_time_offset_ + stage_time_.value() * current_dt_;
    return {authority,
            revision,
            static_cast<std::int64_t>(macro_step()),
            evaluation_stage.numerator,
            evaluation_stage.denominator,
            std::bit_cast<std::uint64_t>(current_dt_),
            std::bit_cast<std::uint64_t>(evaluation_time),
            std::uint64_t{1},
            topology,
            resources};
  }

  static field_type make_scratch_(const field_type& prototype, int ncomp,
                                  const Extent<Dim>& ghosts) {
    field_type result(prototype.layout(), prototype.distribution(), prototype.local_rank(), ncomp,
                      ghosts);
    result.set_val(Real(0));
    return result;
  }

  static field_type scalar_field_like_(const field_type& prototype, int ncomp, int ghost_depth) {
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument(
          "ProgramContext scalar field requires positive components and non-negative ghosts");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    return make_scratch_(prototype, ncomp, ghosts);
  }

  field_type& persistent_scratch_(ScratchKind kind, std::int64_t value_id, int subslot,
                                  const field_type& prototype, int ncomp,
                                  const Extent<Dim>& ghosts) const {
    if (value_id < 0 || subslot < 0)
      throw std::invalid_argument("ProgramContext scratch identity must be non-negative");
    const ScratchKey key{kind, value_id, subslot};
    auto [entry, inserted] = scratch_.try_emplace(key);
    field_type& result = entry->second;
    if (inserted || result.layout() != prototype.layout() ||
        result.distribution() != prototype.distribution() ||
        result.local_rank() != prototype.local_rank() || result.ncomp() != ncomp ||
        result.ghosts() != ghosts)
      result = make_scratch_(prototype, ncomp, ghosts);
    else
      result.set_val(Real(0));
    return result;
  }

  void store_history_(const std::string& name, const field_type& value,
                      std::optional<Real> dt) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end())
      throw std::out_of_range("ProgramContext history is not registered");
    require_same_field_contract_(found->second.front(), value, "ProgramContext history store");
    found->second.front() = value;
    if (!manager.initialized.at(name))
      for (std::size_t slot = 1; slot < found->second.size(); ++slot)
        found->second[slot] = value;
    manager.initialized[name] = true;
    manager.store_pending[name] = true;
    if (dt)
      manager.slot_dt[name][0] = *dt;
  }

  std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                         const std::string& clock,
                                                         const std::string& stage_identity,
                                                         int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level, 0,
                                      static_cast<std::int64_t>(macro_step()));
  }

  void count_kernel_(std::int64_t count = 1) const {
    runtime_state().profiler_.count("kernels", count);
  }

  runtime_type* system_ = nullptr;
  mutable std::uint64_t scalar_boundary_generation_ = 0;
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::optional<OperatorEvaluationSnapshot> active_operator_snapshot_;
  mutable double current_dt_ = 0.0;
  mutable amr::Rational stage_time_{0, 1};
  mutable amr::Rational logical_phase_begin_{0, 1};
  mutable amr::Rational logical_phase_span_{1, 1};
  mutable double logical_physical_time_offset_ = 0.0;
  mutable std::string primary_clock_;
  mutable ClockScheduleState clock_schedule_;
  mutable std::map<ScratchKey, field_type> scratch_;
  mutable std::map<std::int64_t, GeneratedFieldRoute> generated_field_routes_;
};

template <int Dim>
std::shared_ptr<ProgramContext<Dim>> make_program_execution_provider(System<Dim>* system) {
  return std::make_shared<ProgramContext<Dim>>(system);
}

template <int Dim>
ProgramContext<Dim> make_program_execution_view(System<Dim>* system) {
  return ProgramContext<Dim>(system);
}

}  // namespace pops::runtime::program
