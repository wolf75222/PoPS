/// @file
/// @brief Exact compile-time-ranked execution boundary for generated Uniform Programs.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/source_mask.hpp>
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
#include <variant>
#include <vector>

namespace pops::runtime::program {

template <int Dim>
class ProgramExecutionPreparationImage;

// These hooks are intentionally declared next to the public execution authority, rather than in
// System.  A generated v5 prelude can therefore only populate the image retained by its candidate;
// it has no route to the accepted field-plan registry while preparation is active.
template <int Dim>
void stage_uniform_field_boundary_kernel(const ProgramPreparationImage* image,
                                         const std::string& provider_slot,
                                         const CompiledFieldBoundaryKernel<Dim>& kernel);
template <int Dim>
void stage_uniform_field_logical_timepoint(const ProgramPreparationImage* image,
                                           const std::string& provider_slot,
                                           const FieldLogicalTimePoint& point);
template <int Dim>
void stage_uniform_field_boundary_parameters(const ProgramPreparationImage* image,
                                             const std::string& provider_slot,
                                             const std::vector<double>& parameters);
template <int Dim>
void stage_amr_field_boundary_kernel(const ProgramPreparationImage* image,
                                     const std::string& provider_slot,
                                     const CompiledFieldBoundaryKernel<Dim>& kernel);
template <int Dim>
void stage_amr_field_logical_timepoint(const ProgramPreparationImage* image,
                                       const std::string& provider_slot,
                                       const FieldLogicalTimePoint& point);
template <int Dim>
void stage_amr_field_boundary_parameters(const ProgramPreparationImage* image,
                                         const std::string& provider_slot,
                                         const std::vector<double>& parameters);

namespace detail {

/// Exact-ranked request records shared by both storage/topology adapters.
///
/// Uniform and AMR are deliberately different backends, but the generated Program dispatch has
/// one request ABI.  Keeping these records at the adapter boundary prevents the public service
/// from constructing a temporary conversion container on the hot path (and prevents the two
/// backends from silently acquiring ABI-distinct nested types).
template <int Dim>
struct ProgramFieldStageOverride final {
  using field_type = MultiFab<Dim>;

  int program_block = -1;
  const field_type* state = nullptr;
};

template <int Dim>
struct ProgramRhsGroupRequest final {
  using field_type = MultiFab<Dim>;

  ProgramRhsGroupRequest(int block_value, field_type* state_value, field_type* rhs_value,
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

/// A generated Program and its Uniform runtime have one immutable native rank.
///
/// The context never decodes a dimension tag, infers a missing axis, or substitutes a legacy
/// two-dimensional grid provider.  Every state, scratch, history and cache value is a
/// `MultiFab<Dim>` copied from an already-authenticated runtime layout.  Operations whose providers
/// have not yet crossed the exact-ranked boundary fail before touching storage.
template <int Dim>
class UniformStorageTopologyAdapter {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "ProgramExecutionServices only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using runtime_type = System<Dim>;
  using field_type = MultiFab<Dim>;
  using runtime_state_type = ProgramRuntimeState<Dim>;
  struct PreparedReadView final {
    ExecutionLane lane;
    Geometry<Dim> geometry;
    std::array<bool, Dim> periodicity{};
    Real hmin = Real(0);
    int macro_step = 0;
    Real physical_time = Real(0);
    std::vector<RuntimeParams> params;
    std::vector<NewtonOptions> newton;
    std::vector<bool> requires_boundary, has_linearization;
    /// Every mutable Uniform service reachable while a DSO prelude executes belongs to this
    /// image-private state.  It is populated from accepted value snapshots before the callback;
    /// a failed callback therefore cannot alter System's profiler, cache, or history authority.
    std::unique_ptr<runtime_state_type> runtime_state;
    /// The carrier and registry receipt are independent value images.  A prelude may inspect the
    /// carrier only when it was sealed in the accepted generation; it never retains a System-owned
    /// provider group or auxiliary-registry pointer.
    std::shared_ptr<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier;
    std::optional<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>> auxiliary_registry;
  };
  template <int Count>
  using provider_values_view_type = ProviderStorageView<Dim, Count>;
  using scalar_boundary_session_type = PreparedScalarBoundarySession<Dim>;

  /// Immutable authentication token for one generated block-boundary invocation.  It retains no
  /// closure, state field, or mutable boundary image: System remains the sole owner of the exact
  /// prepared authority installed with the block.
  class PreparedBlockBoundarySession {
   public:
    PreparedBlockBoundarySession(const PreparedBlockBoundarySession&) = default;
    PreparedBlockBoundarySession& operator=(const PreparedBlockBoundarySession&) = default;

   private:
    friend class UniformStorageTopologyAdapter;

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

  using FieldStageOverride = ProgramFieldStageOverride<Dim>;
  using RhsGroupRequest = ProgramRhsGroupRequest<Dim>;

  struct CouplingStateOverride {
    int program_block = -1;
    field_type* state = nullptr;
  };

  /// Move-only exact child interval used by generated subcycle bodies.
  class LogicalEvaluationScope {
   public:
    LogicalEvaluationScope(const UniformStorageTopologyAdapter& owner, int iteration, int count)
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
      const ::pops::amr::Rational child_fraction(iteration, count);
      const ::pops::amr::Rational child_span(1, count);
      owner_->current_dt_ = child_dt;
      owner_->stage_time_ = ::pops::amr::Rational(0, 1);
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

    const UniformStorageTopologyAdapter* owner_ = nullptr;
    double prior_dt_ = 0.0;
    ::pops::amr::Rational prior_stage_{0, 1};
    ::pops::amr::Rational prior_phase_begin_{0, 1};
    ::pops::amr::Rational prior_phase_span_{1, 1};
    double prior_physical_time_offset_ = 0.0;
  };

  /// Fixed-capacity storage for the operations that may run after begin_step().  The vectors are
  /// deliberately owned by the adapter and populated only at the accepted/preparation boundary;
  /// the generated path only changes their logical sizes and pointer values.  In particular, no
  /// commit image is constructed from a candidate during a step: every image has the exact layout
  /// of its owning accepted block and is copied with device deep copies.
  struct PreparedHotPathWorkspace {
    bool bound = false;
    std::size_t block_capacity = 0;
    std::vector<int> rhs_blocks;
    std::vector<int> rhs_rates;
    std::vector<int> rhs_flux_only;
    std::vector<field_type*> rhs_states;
    std::vector<field_type*> rhs_residuals;
    std::vector<field_type*> coupling_states;
    std::vector<const field_type*> solve_runtime_stages;
    std::vector<const field_type*> solve_unique_stages;
    std::vector<field_type*> commit_targets;
    std::vector<const field_type*> commit_sources;
    std::vector<const field_type*> commit_masks;
    std::vector<int> commit_runtime_blocks;
    std::vector<char> commit_identity;
    std::vector<field_type> commit_snapshots;

    void bind_shape(std::size_t blocks, std::size_t request_capacity) {
      const std::size_t capacity = std::max(blocks, request_capacity);
      if (bound && block_capacity != capacity)
        throw std::logic_error("Program hot-path workspace shape changed after preparation");
      if (bound)
        return;
      block_capacity = capacity;
      rhs_blocks.resize(capacity);
      rhs_rates.resize(capacity);
      rhs_flux_only.resize(capacity);
      rhs_states.resize(capacity);
      rhs_residuals.resize(capacity);
      coupling_states.resize(blocks);
      solve_runtime_stages.resize(blocks);
      solve_unique_stages.resize(capacity);
      commit_targets.resize(capacity);
      commit_sources.resize(capacity);
      commit_masks.resize(capacity);
      commit_runtime_blocks.resize(capacity);
      commit_identity.resize(capacity);
      bound = true;
    }

    template <class Getter>
    void bind_commit_images(std::size_t blocks, Getter&& getter) {
      if (!bound || blocks > block_capacity)
        throw std::logic_error("Program hot-path workspace is not bound to its resource plan");
      if (!commit_snapshots.empty()) {
        if (commit_snapshots.size() != blocks)
          throw std::logic_error("Program hot-path commit image count changed after preparation");
        return;
      }
      commit_snapshots.reserve(blocks);
      for (std::size_t block = 0; block < blocks; ++block)
        commit_snapshots.emplace_back(getter(block));
    }

    void require_bound(std::size_t count, const char* operation) const {
      if (!bound || count > block_capacity || commit_snapshots.size() != coupling_states.size())
        throw std::logic_error(std::string(operation) +
                               " requires a bind-sealed hot-path workspace");
    }
  };

  void bind_accepted_hot_path_workspace_() const {
    if (system_ == nullptr)
      return;
    const std::size_t blocks = static_cast<std::size_t>(program_n_blocks_());
    hot_path_workspace_.bind_shape(blocks, blocks);
    hot_path_workspace_.bind_commit_images(blocks, [this](std::size_t block) -> const field_type& {
      return program_state_const_(static_cast<int>(block));
    });
  }

  void bind_prepared_hot_path_workspace_() const {
    if (preparation_states_ == nullptr)
      return;
    hot_path_workspace_.bind_shape(preparation_states_->size(), preparation_states_->size());
    hot_path_workspace_.bind_commit_images(
        preparation_states_->size(),
        [this](std::size_t block) -> const field_type& { return preparation_states_->at(block); });
  }

  UniformStorageTopologyAdapter() = default;

  void bind_accepted_system(runtime_type* system) const {
    runtime_type* const accepted = require_system_(system);
    if (system_ != nullptr && system_ != accepted)
      throw std::logic_error("Program execution services cannot change their accepted System");
    system_ = accepted;
    bind_accepted_hot_path_workspace_();
  }

  /// Attach the immutable state/map snapshot retained by ProgramExecutionPreparationImage.  This
  /// is the only Uniform storage visible before collective acceptance; it is intentionally a view
  /// of image-owned vectors, never a System facade pointer.
  void bind_preparation_state_view(const std::vector<field_type>* states,
                                   const std::vector<int>* block_map) const {
    // A v5 Program with no block table is a deliberately state-free authority: it can own clocks,
    // diagnostics and transaction effects, but it cannot name a state, resource, or runtime block.
    // The two non-null empty vectors are therefore a complete detached view, not a missing one.
    // A nonempty image can legitimately bind its state prototypes before the descriptor has supplied
    // its final explicit map, so map cardinality is validated at resource-declaration bind time.
    if (states == nullptr || block_map == nullptr)
      throw std::invalid_argument("Program preparation state view is incomplete");
    if (system_ != nullptr)
      throw std::logic_error("accepted Program services cannot acquire a preparation state view");
    preparation_states_ = states;
    preparation_block_map_ = block_map;
    bind_prepared_hot_path_workspace_();
  }
  void bind_preparation_read_view(const PreparedReadView* view) const {
    if (view == nullptr || system_ != nullptr)
      throw std::invalid_argument("Program preparation read view is incomplete");
    preparation_read_view_ = view;
  }

  [[nodiscard]] bool has_system() const noexcept { return system_ != nullptr; }

  [[nodiscard]] ProgramHostDescriptor program_host_descriptor() const {
    if (system_ == nullptr)
      throw std::logic_error("Program preparation has no accepted host descriptor");
    return const_cast<runtime_type*>(system_)->program_host_descriptor_();
  }

  void begin_step(double dt) const {
    (void)prepared_execution_lane();
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramExecutionServices step requires a finite positive dt");
    current_dt_ = dt;
    stage_time_ = ::pops::amr::Rational(0, 1);
    logical_phase_begin_ = ::pops::amr::Rational(0, 1);
    logical_phase_span_ = ::pops::amr::Rational(1, 1);
    logical_physical_time_offset_ = 0.0;
    active_operator_snapshot_.reset();
  }

  void configure_primary_clock(const std::string& clock) const {
    clock_schedule_.configure_primary_clock(clock);
    primary_clock_ = clock;
  }

  /// Transfer a clock prepared in the host-owned installation image.  This only changes the
  /// execution-services object retained by the candidate DSO; it never reaches System during
  /// preparation.
  void adopt_prepared_clock(ClockScheduleState schedule, std::string primary_clock) const {
    if (primary_clock.empty())
      throw std::invalid_argument("Program prepared clock has no primary identity");
    clock_schedule_ = std::move(schedule);
    primary_clock_ = std::move(primary_clock);
  }

  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    clock_schedule_.declare_relation(parent, child, count);
  }

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (denominator <= 0 || numerator < 0 || numerator > denominator)
      throw std::invalid_argument("ProgramExecutionServices stage time is outside [0, 1]");
    stage_time_ = ::pops::amr::Rational(numerator, denominator);
    active_operator_snapshot_.reset();
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error(
          "ProgramExecutionServices boundary evaluation has no prepared clock and dt");
    const ::pops::amr::Rational evaluation_stage =
        logical_phase_begin_ + stage_time_ * logical_phase_span_;
    return {primary_clock_,
            static_cast<std::int64_t>(macro_step()),
            0,
            0,
            stage,
            evaluation_stage,
            current_dt_,
            physical_time() + logical_physical_time_offset_ + stage_time_.value() * current_dt_};
  }

  // These accessors are explicit writer-side seams.  A generated Uniform Program runs while the
  // candidate visibility writer is held, so it must never call a public accepted-state observer
  // (which would acquire an AcceptedReadLease and either block or recurse on the writer thread).
  int program_n_blocks_() const {
    if (system_ != nullptr)
      return system_->program_n_blocks_();
    if (preparation_states_ == nullptr)
      throw std::logic_error("Program preparation has no detached state snapshot");
    return static_cast<int>(preparation_states_->size());
  }
  const std::vector<int>& program_block_map_() const {
    if (system_ != nullptr)
      return system_->program_block_map_();
    if (preparation_block_map_ == nullptr)
      throw std::logic_error("Program preparation has no detached block map");
    return *preparation_block_map_;
  }
  field_type& program_state_(int runtime_block) const {
    if (system_ != nullptr)
      return system_->program_block_state_(runtime_block);
    if (preparation_states_ == nullptr || runtime_block < 0 ||
        runtime_block >= static_cast<int>(preparation_states_->size()))
      throw std::out_of_range("Program preparation state is outside its detached snapshot");
    return const_cast<field_type&>(
        preparation_states_->at(static_cast<std::size_t>(runtime_block)));
  }
  const field_type& program_state_const_(int runtime_block) const {
    return program_state_(runtime_block);
  }
  [[nodiscard]] const ExecutionLane& program_prepared_execution_lane_() const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->lane;
    return system_->program_prepared_boundary_execution_lane_();
  }
  [[nodiscard]] Geometry<Dim> program_geometry_() const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->geometry;
    return system_->program_prepared_block_geometry_();
  }
  [[nodiscard]] std::array<bool, Dim> program_periodicity_() const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->periodicity;
    return system_->program_prepared_block_periodicity_();
  }
  [[nodiscard]] const auto* program_provider_storage_groups_() const {
    if (system_ == nullptr) {
      if (preparation_read_view_ == nullptr || !preparation_read_view_->provider_carrier)
        throw std::logic_error("Program preparation has no sealed auxiliary provider carrier");
      return preparation_read_view_->provider_carrier.get();
    }
    return system_->program_prepared_block_provider_storage_groups_();
  }
  [[nodiscard]] const auto* program_auxiliary_consumer_plan_(std::string_view consumer_qid) const {
    if (system_ == nullptr)
      throw std::logic_error(
          "Program preparation cannot resolve an accepted auxiliary consumer plan before seal");
    return system_->program_prepared_auxiliary_consumer_plan_(std::string(consumer_qid));
  }
  [[nodiscard]] NewtonOptions program_newton_options_(int runtime_block) const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->newton.at(static_cast<std::size_t>(runtime_block));
    return system_->program_block_newton_options_(runtime_block);
  }
  [[nodiscard]] bool program_requires_boundary_session_(int runtime_block) const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->requires_boundary.at(static_cast<std::size_t>(runtime_block));
    return system_->program_requires_block_boundary_session_(runtime_block);
  }
  [[nodiscard]] bool program_has_boundary_linearization_(int runtime_block) const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->has_linearization.at(static_cast<std::size_t>(runtime_block));
    return system_->program_has_block_boundary_linearization_(runtime_block);
  }
  [[nodiscard]] Real program_hmin_() const {
    return system_ == nullptr && preparation_read_view_ != nullptr ? preparation_read_view_->hmin
                                                                   : system_->program_cfl_min_dx_();
  }
  [[nodiscard]] RuntimeParams program_params_(int program_block) const {
    if (system_ == nullptr && preparation_read_view_ != nullptr)
      return preparation_read_view_->params.at(static_cast<std::size_t>(program_block));
    return system_->program_params_(program_block);
  }
  [[nodiscard]] runtime::program::Profiler& program_profiler_() const {
    if (system_ == nullptr) {
      if (preparation_read_view_ == nullptr || !preparation_read_view_->runtime_state)
        throw std::logic_error("Program preparation has no detached profiler image");
      return preparation_read_view_->runtime_state->profiler_;
    }
    return const_cast<runtime_type*>(system_)->program_profiler_();
  }
  [[nodiscard]] runtime::program::CacheManager<Dim>& program_cache_() const {
    if (system_ == nullptr) {
      if (preparation_read_view_ == nullptr || !preparation_read_view_->runtime_state)
        throw std::logic_error("Program preparation has no detached cache image");
      return preparation_read_view_->runtime_state->cache_;
    }
    return const_cast<runtime_type*>(system_)->program_cache_();
  }

  int n_blocks() const { return program_n_blocks_(); }

  /// Borrow the runtime-owned lane authenticated during Uniform boundary preparation. Generated
  /// implicit reports use this same lane for every reduction, diagnostic selection, and outcome.
  [[nodiscard]] const ExecutionLane& prepared_execution_lane() const {
    return program_prepared_execution_lane_();
  }

  /// Borrow the exact runtime-owned communicator for persistent generated materialization. The
  /// generated Program never duplicates MPI_COMM_WORLD when its prepared boundary lane is owned by
  /// an embedding communicator.
  [[nodiscard]] ExecutionCommunicator prepared_execution_communicator() const {
    const ExecutionLane& lane = prepared_execution_lane();
#ifdef POPS_HAS_MPI
    return ExecutionCommunicator::borrowed(lane.identity(), lane.native_handle());
#else
    return ExecutionCommunicator::world();
#endif
  }

  int sys_block(int program_block) const {
    const std::vector<int>& map = program_block_map_();
    if (map.empty())
      throw std::runtime_error(
          "ProgramExecutionServices has no explicit program-to-runtime block map; positional "
          "identity is "
          "not supported");
    if (program_block < 0 || program_block >= static_cast<int>(map.size()))
      throw std::out_of_range("ProgramExecutionServices block is outside the authenticated map");
    const int runtime_block = map[static_cast<std::size_t>(program_block)];
    if (runtime_block < 0 || runtime_block >= program_n_blocks_())
      throw std::runtime_error(
          "ProgramExecutionServices block map targets an absent runtime block");
    return runtime_block;
  }

  field_type& state(int program_block) const { return program_state_(sys_block(program_block)); }
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
      const auto* const groups = program_provider_storage_groups_();
      const auto* const plan = program_auxiliary_consumer_plan_(consumer_qid);
      runtime::system::require_pointwise_provider_groups<Dim, Count>(
          state_field, groups, plan, "ProgramExecutionServices provider values");
      return runtime::system::bind_provider_storage_view<Dim, Count>(plan, groups, local_fab);
    }
  }

  field_type rhs_scratch_like(const field_type& prototype) const {
    return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
  }

  field_type scratch_state_like(const field_type& prototype) const {
    return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
  }

  field_type& rhs_scratch(ProgramCacheSlot slot, int subslot, const field_type& prototype) const {
    return persistent_scratch_(ScratchKind::Rhs, slot, subslot, prototype, prototype.ncomp(),
                               prototype.ghosts());
  }

  field_type& scratch_state(ProgramCacheSlot slot, int subslot, const field_type& prototype) const {
    return persistent_scratch_(ScratchKind::State, slot, subslot, prototype, prototype.ncomp(),
                               prototype.ghosts());
  }

  field_type& scalar_scratch(ProgramCacheSlot slot, int subslot, const field_type& prototype,
                             int ncomp = 1, int ghost_depth = 1) const {
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument(
          "ProgramExecutionServices scalar scratch requires positive components and non-negative "
          "ghosts");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    return persistent_scratch_(ScratchKind::Scalar, slot, subslot, prototype, ncomp, ghosts);
  }

  /// Install-time-only reservation of one finite scratch location.  The compact value id is the
  /// bind-sealed ProgramResourcePlan slot emitted by lowering; it is deliberately not a node id.
  /// After preparation, lookup is a pair of checked vector indices and cannot insert or allocate.
  void bind_prepared_scratch_slots(std::size_t slot_count) const {
    if (!scratch_.empty() && scratch_.size() != slot_count)
      throw std::logic_error("Program scratch plan changed after preparation");
    scratch_.resize(slot_count);
  }

  void prime_rhs_scratch(std::size_t slot, int subslot, const field_type& prototype) const {
    prime_persistent_scratch_(ScratchKind::Rhs, slot, subslot, prototype, prototype.ncomp(),
                              prototype.ghosts());
  }

  void prime_rhs_scratch_exact(std::size_t slot, int subslot, const field_type& prototype,
                               int ncomp, const Extent<Dim>& ghosts) const {
    prime_persistent_scratch_(ScratchKind::Rhs, slot, subslot, prototype, ncomp, ghosts);
  }

  void prime_state_scratch(std::size_t slot, int subslot, const field_type& prototype) const {
    prime_persistent_scratch_(ScratchKind::State, slot, subslot, prototype, prototype.ncomp(),
                              prototype.ghosts());
  }

  void prime_state_scratch_exact(std::size_t slot, int subslot, const field_type& prototype,
                                 int ncomp, const Extent<Dim>& ghosts) const {
    prime_persistent_scratch_(ScratchKind::State, slot, subslot, prototype, ncomp, ghosts);
  }

  void prime_scalar_scratch(std::size_t slot, int subslot, const field_type& prototype, int ncomp,
                            int ghost_depth) const {
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument("Program scalar scratch prime has invalid shape");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    prime_persistent_scratch_(ScratchKind::Scalar, slot, subslot, prototype, ncomp, ghosts);
  }

  field_type make_prepared_field_like(const field_type& prototype, int ncomp,
                                      int ghost_depth) const {
    return scalar_field_like_(prototype, ncomp, ghost_depth);
  }

  field_type alloc_scalar_field(int ncomp = 1, int ghost_depth = 1) const {
    return scalar_field_like_(state(0), ncomp, ghost_depth);
  }

  void rhs_into(int program_block, field_type& state_value, field_type& rhs, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel_();
    const auto point = boundary_evaluation_point(rate_id);
    const int runtime_block = sys_block(program_block);
    if (program_requires_boundary_session_(runtime_block)) {
      const ExecutionLane& lane = program_prepared_execution_lane_();
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

  [[nodiscard]] NewtonOptions block_newton_options(int program_block) const {
    return program_newton_options_(sys_block(program_block));
  }

  [[nodiscard]] SolveOutcome solve_source_default(int program_block, field_type& stage_state,
                                                  Real dt, const NewtonOptions& options) const {
    count_kernel_();
    return system_->solve_block_source(sys_block(program_block), stage_state, dt, options);
  }

  void publish_newton_report(int program_block, const SolveReport& solve) const {
    system_->publish_newton_report(sys_block(program_block), solve);
  }

  void apply_source_mask(field_type& rhs, std::initializer_list<int> keep) const {
    pops::runtime::program::apply_component_keep_mask(rhs,
                                                      std::vector<int>(keep.begin(), keep.end()));
    count_kernel_();
  }

  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    auto& workspace = hot_path_workspace_;
    workspace.require_bound(requests.size(), "ProgramExecutionServices RHS group");
    require_rate_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("ProgramExecutionServices RHS group cannot be empty");
    workspace.rhs_blocks.resize(requests.size());
    workspace.rhs_rates.resize(requests.size());
    workspace.rhs_flux_only.resize(requests.size());
    workspace.rhs_states.resize(requests.size());
    workspace.rhs_residuals.resize(requests.size());
    std::size_t index = 0;
    for (const RhsGroupRequest& request : requests) {
      require_rate_identity_(request.rate_id);
      if (request.rate_id == group_id || request.state == nullptr || request.rhs == nullptr ||
          (request.flux_only != 0 && request.flux_only != 1) ||
          std::find(workspace.rhs_rates.begin(), workspace.rhs_rates.begin() + index,
                    request.rate_id) != workspace.rhs_rates.begin() + index)
        throw std::invalid_argument(
            "ProgramExecutionServices RHS group contains an invalid request");
      workspace.rhs_rates[index] = request.rate_id;
      workspace.rhs_blocks[index] = sys_block(request.block);
      workspace.rhs_states[index] = request.state;
      workspace.rhs_residuals[index] = request.rhs;
      workspace.rhs_flux_only[index] = request.flux_only;
      ++index;
    }
    count_kernel_(static_cast<std::int64_t>(requests.size()));
    system_->block_rhs_group(boundary_evaluation_point(group_id), workspace.rhs_blocks,
                             workspace.rhs_states, workspace.rhs_residuals,
                             workspace.rhs_flux_only);
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
  void neg_div_named_flux_into(int program_block, field_type& stage_state, field_type& rhs,
                               const std::array<field_type*, Dim>& fluxes, int rate_id) const {
    (void)sys_block(program_block);
    require_rate_identity_(rate_id);
    require_same_field_contract_(stage_state, rhs, "ProgramExecutionServices named-flux residual");
    const Geometry<Dim> geometry = program_geometry_();
    for (int axis = 0; axis < Dim; ++axis) {
      const field_type* flux = fluxes[static_cast<std::size_t>(axis)];
      if (flux == nullptr || flux->layout() != rhs.layout() ||
          flux->distribution() != rhs.distribution() || flux->local_rank() != rhs.local_rank() ||
          flux->ncomp() != rhs.ncomp() || flux->local_size() != rhs.local_size() ||
          flux->ghosts()[axis] < 1)
        throw std::invalid_argument(
            "ProgramExecutionServices named flux does not match the exact ranked residual layout");
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
    const ExecutionLane& lane = prepared_execution_lane();
    const int runtime_block =
        resolve_prepared_program_block_(program_block, lane, "Program projection block");
    count_kernel_();
    system_->block_project(runtime_block, state_value);
  }

  Real max_wave_speed(int program_block, const field_type& state_value) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const int runtime_block =
        resolve_prepared_program_block_(program_block, lane, "Program maximum-speed block");
    return system_->block_max_speed_prepared_(runtime_block, state_value, lane);
  }

  Real hmin() const { return program_hmin_(); }
  RuntimeParams program_params(int program_block) const { return program_params_(program_block); }

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
    auto& workspace = hot_path_workspace_;
    workspace.require_bound(commits.size(), "ProgramExecutionServices commit");
    const ExecutionLane& lane = prepared_execution_lane();
    workspace.commit_targets.resize(commits.size());
    workspace.commit_sources.resize(commits.size());
    workspace.commit_runtime_blocks.resize(commits.size());
    workspace.commit_identity.resize(commits.size());
    workspace.commit_masks.resize(commits.size());
    std::exception_ptr structural_error;
    try {
      std::size_t candidate = 0;
      for (const auto& [target, source] : commits) {
        if (target == nullptr || source == nullptr)
          throw std::invalid_argument("ProgramExecutionServices commit contains null storage");
        if (std::find(workspace.commit_targets.begin(),
                      workspace.commit_targets.begin() + candidate,
                      target) != workspace.commit_targets.begin() + candidate)
          throw std::invalid_argument(
              "ProgramExecutionServices commit contains a duplicate target");
        require_same_field_contract_(*target, *source, "ProgramExecutionServices commit");
        workspace.commit_targets[candidate] = target;
        workspace.commit_sources[candidate] = source;
        ++candidate;
      }
    } catch (...) {
      structural_error = std::current_exception();
    }
    if (all_reduce_max(structural_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && structural_error)
        std::rethrow_exception(structural_error);
      throw std::runtime_error(
          "ProgramExecutionServices commit classification failed collectively");
    }

    const long commit_count = static_cast<long>(commits.size());
    if (all_reduce_min(commit_count, lane) != all_reduce_max(commit_count, lane))
      throw std::runtime_error("ProgramExecutionServices commit count differs between MPI ranks");

    std::exception_ptr classification_error;
    try {
      for (std::size_t candidate = 0; candidate < commits.size(); ++candidate) {
        field_type* const target = workspace.commit_targets[candidate];
        const field_type* const source = workspace.commit_sources[candidate];
        workspace.commit_identity[candidate] = target == source ? 1 : 0;
        workspace.commit_runtime_blocks[candidate] = -1;
        for (int block = 0; block < program_n_blocks_(); ++block) {
          if (target == &program_state_(block)) {
            workspace.commit_runtime_blocks[candidate] = block;
            break;
          }
        }
      }
    } catch (...) {
      classification_error = std::current_exception();
    }
    if (all_reduce_max(classification_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && classification_error)
        std::rethrow_exception(classification_error);
      throw std::runtime_error(
          "ProgramExecutionServices commit target classification failed collectively");
    }
    for (std::size_t candidate = 0; candidate < commits.size(); ++candidate) {
      const long owner = static_cast<long>(workspace.commit_runtime_blocks[candidate]);
      const long identity = static_cast<long>(workspace.commit_identity[candidate]);
      if (all_reduce_min(owner, lane) != all_reduce_max(owner, lane) ||
          all_reduce_min(identity, lane) != all_reduce_max(identity, lane))
        throw std::runtime_error(
            "ProgramExecutionServices commit target owner differs between MPI ranks");
    }

    for (std::size_t candidate = 0; candidate < commits.size(); ++candidate) {
      workspace.commit_masks[candidate] = nullptr;
      if (workspace.commit_runtime_blocks[candidate] < 0) {
        if (workspace.commit_identity[candidate] == 0)
          throw std::invalid_argument(
              "ProgramExecutionServices commit target is not a prepared runtime block");
        continue;
      }
      workspace.commit_masks[candidate] = system_->validate_program_state_publication_candidate_(
          workspace.commit_runtime_blocks[candidate], *workspace.commit_sources[candidate], lane);
    }
    for (std::size_t candidate = 0; candidate < commits.size(); ++candidate) {
      const long masked = workspace.commit_masks[candidate] != nullptr ? 1L : 0L;
      if (all_reduce_min(masked, lane) != all_reduce_max(masked, lane))
        throw std::runtime_error(
            "ProgramExecutionServices commit mask classification differs between ranks");
    }

    std::exception_ptr staging_error;
    try {
      for (std::size_t candidate = 0; candidate < commits.size(); ++candidate) {
        if (workspace.commit_identity[candidate] != 0)
          continue;
        const int block = workspace.commit_runtime_blocks[candidate];
        if (workspace.commit_masks[candidate] != nullptr) {
          copy_field_storage_(*workspace.commit_targets[candidate],
                              workspace.commit_snapshots[block]);
          copy_active_valid_cells_(*workspace.commit_sources[candidate],
                                   workspace.commit_snapshots[block],
                                   *workspace.commit_masks[candidate]);
        } else {
          copy_field_storage_(*workspace.commit_sources[candidate],
                              workspace.commit_snapshots[block]);
        }
      }
    } catch (...) {
      staging_error = std::current_exception();
    }
    try {
      device_fence();
    } catch (...) {
      staging_error = std::current_exception();
    }
    if (all_reduce_max(staging_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && staging_error)
        std::rethrow_exception(staging_error);
      throw std::runtime_error("ProgramExecutionServices commit staging failed collectively");
    }

    for (std::size_t candidate = 0; candidate < commits.size(); ++candidate)
      if (workspace.commit_identity[candidate] == 0)
        copy_field_storage_(workspace.commit_snapshots[workspace.commit_runtime_blocks[candidate]],
                            *workspace.commit_targets[candidate]);
  }

  void apply_coupling_operators(Real dt,
                                std::initializer_list<CouplingStateOverride> candidates) const {
    auto& workspace = hot_path_workspace_;
    workspace.require_bound(candidates.size(), "ProgramExecutionServices coupling");
    std::fill(workspace.coupling_states.begin(), workspace.coupling_states.end(), nullptr);
    for (const CouplingStateOverride& candidate : candidates) {
      const int block = sys_block(candidate.program_block);
      if (candidate.state == nullptr ||
          workspace.coupling_states[static_cast<std::size_t>(block)] != nullptr)
        throw std::invalid_argument(
            "ProgramExecutionServices coupling candidates are incomplete or aliased");
      workspace.coupling_states[static_cast<std::size_t>(block)] = candidate.state;
    }
    if (std::find(workspace.coupling_states.begin(), workspace.coupling_states.end(), nullptr) !=
        workspace.coupling_states.end())
      throw std::invalid_argument(
          "ProgramExecutionServices coupling requires every runtime block candidate");
    count_kernel_(static_cast<std::int64_t>(
        system_->apply_coupling_operators(dt, workspace.coupling_states)));
  }

  /// Return the prepared embedded-boundary mask for this pointwise block, or null for Cartesian / an
  /// inactive embedded boundary.  The route consensus intentionally uses only fixed scalars: this
  /// hot path must not build a dynamic exact-contract payload before every generated kernel.
  const field_type* pointwise_active_mask(int program_block, const field_type& field) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const int runtime_block = resolve_pointwise_program_block_(program_block, lane);
    return system_->prepared_program_block_active_mask_(runtime_block, field, lane);
  }

  /// Reduce one generated per-cell status on the same authenticated lane and layout used by its
  /// pointwise kernel.  Empty ranks contribute negative infinity, normalized to success only when
  /// the entire collective layout is empty.
  Real pointwise_status_max(int program_block, const field_type& status,
                            const field_type* active_cells, const ExecutionLane& lane) const {
    require_prepared_lane_(lane, "Program pointwise status");
    const int runtime_block = resolve_pointwise_program_block_(program_block, lane);
    const field_type* const expected =
        system_->prepared_program_block_active_mask_(runtime_block, status, lane);
    if (all_reduce_max(active_cells == expected ? 0L : 1L, lane) != 0)
      throw std::invalid_argument("Program pointwise status received a foreign active-cell mask");
    if (all_reduce_max(status.ncomp() == 1 ? 0L : 1L, lane) != 0)
      throw std::invalid_argument("Program pointwise status requires exactly one component");
    MaskedMaxLocalResult local;
    std::exception_ptr local_error;
    try {
      local = pops::reduce_masked_max_local(status, 0, expected);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program pointwise status reduction failed collectively");
    }
    if (all_reduce_max(local.has_invalid ? 1L : 0L, lane) != 0)
      return Real(3);
    if (all_reduce_max(local.has_active ? 1L : 0L, lane) == 0)
      return Real(0);
    const Real maximum =
        static_cast<Real>(all_reduce_max(static_cast<double>(local.maximum), lane));
    return std::isfinite(maximum) ? maximum : Real(3);
  }

  Real sum_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_sum(pops::reduce_sum_local(field, component), prepared_execution_lane()));
  }
  Real abs_sum_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_sum(pops::reduce_abs_sum_local(field, component), prepared_execution_lane()));
  }
  Real max_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_max(pops::reduce_max_local(field, component), prepared_execution_lane()));
  }
  Real min_component(const field_type& field, int component) const {
    return static_cast<Real>(
        all_reduce_min(pops::reduce_min_local(field, component), prepared_execution_lane()));
  }

  /// Generated reductions authenticate the Program owner and exclude inactive embedded-boundary
  /// cells.  They deliberately remain raw algebra: physical volume-fraction weighting belongs to
  /// System reduction services, not to Program control-flow scalars.
  Real sum_component(int program_block, const field_type& field, int component) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, field);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      local = pops::reduce_active_sum_local(field, component, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program sum reduction");
    return static_cast<Real>(all_reduce_sum(local, lane));
  }
  Real abs_sum_component(int program_block, const field_type& field, int component) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, field);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      local = pops::reduce_active_abs_sum_local(field, component, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program abs-sum reduction");
    return static_cast<Real>(all_reduce_sum(local, lane));
  }
  Real max_component(int program_block, const field_type& field, int component) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, field);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      local = pops::reduce_active_max_local(field, component, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program max reduction");
    return static_cast<Real>(all_reduce_max(local, lane));
  }
  Real min_component(int program_block, const field_type& field, int component) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, field);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      local = pops::reduce_active_min_local(field, component, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program min reduction");
    return static_cast<Real>(all_reduce_min(local, lane));
  }
  Real norm2(int program_block, const field_type& field) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, field);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      local = pops::dot_active_local(field, field, 0, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program norm2 reduction");
    return std::sqrt(static_cast<Real>(all_reduce_sum(local, lane)));
  }
  Real norm_inf(int program_block, const field_type& field) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, field);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      local = pops::reduce_active_norm_inf_local(field, 0, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program norm-inf reduction");
    return static_cast<Real>(all_reduce_max(local, lane));
  }
  Real dot(int program_block, const field_type& left, const field_type& right) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, left);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      require_same_field_contract_(left, right, "ProgramExecutionServices dot");
      local = pops::dot_active_local(left, right, 0, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    converge_owner_reduction_(local_error, lane, "Program dot reduction");
    return static_cast<Real>(all_reduce_sum(local, lane));
  }

  Geometry<Dim> geometry() const { return program_geometry_(); }

  field_type& assembly_target(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("ProgramExecutionServices assembly target requires an identity");
    return field;
  }

  field_type& assembly_source(field_type& field, std::string_view identity) const {
    if (identity.empty())
      throw std::invalid_argument("ProgramExecutionServices assembly source requires an identity");
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
      require_same_field_contract_(prototype, program_state_const_(runtime_block),
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
    return program_has_boundary_linearization_(sys_block(program_block));
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
      const auto tensor_operator = elliptic::nd::make_cartesian_tensor_operator<
          elliptic::nd::CartesianTensorDivergenceSign::positive_divergence>(
          value, elliptic::nd::packed_cartesian_tensor_coefficients<Dim>(coefficient), geom);
      for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        result(cell, 0) = tensor_operator.image(cell);
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
      throw std::invalid_argument(
          "ProgramExecutionServices history requires a name and positive lag");
    const int owner = program_owner < 0 ? 0 : sys_block(program_owner);
    const field_type& prototype = program_state_const_(owner);
    const int components = ncomp < 0 ? prototype.ncomp() : ncomp;
    if (components < 1)
      throw std::invalid_argument(
          "ProgramExecutionServices history component count must be positive");
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
        throw std::runtime_error(
            "ProgramExecutionServices history identity changed after registration");
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
      throw std::out_of_range("ProgramExecutionServices history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramExecutionServices history component contract differs");
    if (!manager.initialized.at(name))
      throw std::runtime_error("ProgramExecutionServices history has not been initialized");
    return result;
  }

  field_type& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(name))
      throw std::out_of_range("ProgramExecutionServices history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramExecutionServices history component contract differs");
    return result;
  }

  void store_history(const std::string& name, const field_type& value) const {
    if (std::isfinite(current_dt_) && current_dt_ > 0.0) {
      store_history_(name, value, static_cast<Real>(current_dt_));
      return;
    }
    // Preserve the direct legacy route when this context has no active generated-step interval:
    // System supplies its accepted last-dt provenance (or its zero-dt pre-step default).
    if (system_ == nullptr)
      throw std::logic_error(
          "Program preparation cannot publish history without an active candidate interval");
    system_->store_history(name, value);
  }
  void store_history(const std::string& name, const field_type& value, double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument(
          "ProgramExecutionServices history dt must be finite and positive");
    store_history_(name, value, static_cast<Real>(dt));
  }
  void rotate_histories() const { runtime_state().hist_.rotate(); }
  void rotate_histories(const std::string& clock) const { runtime_state().hist_.rotate(clock); }

  /// Reconstruct one retained value at an exact target-clock coordinate.  The native history
  /// ledger owns every bracketing interval; no current-state alias or fixed-dt inference is used.
  void interpolate_history_linear(field_type& output, const std::string& name, int max_lag,
                                  int program_owner, const std::string& source_clock,
                                  const std::string& target_clock, int target_step,
                                  Real target_offset) const {
    const int owner = sys_block(program_owner);
    if (max_lag < 1 || !std::isfinite(static_cast<double>(target_offset)))
      throw std::invalid_argument(
          "ProgramExecutionServices linear history interpolation has an invalid target");

    auto& manager = runtime_state().hist_;
    const auto found = manager.histories.find(name);
    if (found == manager.histories.end() || manager.depth.at(name) <= max_lag ||
        manager.owner.at(name) != owner || !manager.initialized.at(name))
      throw std::runtime_error(
          "ProgramExecutionServices linear history interpolation requires an initialized retained "
          "ring");
    require_same_field_contract_(output, found->second.front(),
                                 "ProgramExecutionServices linear history interpolation");

    const double source_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(source_clock));
    const double target_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(target_clock));
    const double coordinate =
        (static_cast<double>(target_step) + static_cast<double>(target_offset)) * source_ticks /
        target_ticks;
    if (!std::isfinite(coordinate) || coordinate > 0.0 ||
        coordinate < -static_cast<double>(max_lag))
      throw std::runtime_error(
          "ProgramExecutionServices linear history interpolation target lies outside retained "
          "timestamps");
    if (coordinate == 0.0) {
      lincomb(output, Real(1), found->second.front(), Real(0), found->second.front());
      return;
    }

    const int older_lag = static_cast<int>(std::ceil(-coordinate));
    if (older_lag < 1 || older_lag > max_lag)
      throw std::runtime_error(
          "ProgramExecutionServices linear history interpolation could not select bracketing "
          "slots");
    const auto dt_ledger = manager.slot_dt.find(name);
    if (dt_ledger == manager.slot_dt.end() || dt_ledger->second.size() != found->second.size())
      throw std::logic_error(
          "ProgramExecutionServices linear history interpolation dt ledger differs from its ring "
          "depth");

    double newer_time = static_cast<double>(physical_time());
    double older_time = newer_time;
    double bracket_dt = 0.0;
    for (int lag = 1; lag <= older_lag; ++lag) {
      const double interval = dt_ledger->second[static_cast<std::size_t>(lag)];
      if (!std::isfinite(interval) || !(interval > 0.0))
        throw std::runtime_error(
            "ProgramExecutionServices linear history interpolation requires positive exact slot "
            "timestamps");
      bracket_dt = interval;
      older_time = newer_time - interval;
      if (lag != older_lag)
        newer_time = older_time;
    }
    const double logical_fraction = coordinate + static_cast<double>(older_lag);
    const double target_time = older_time + logical_fraction * bracket_dt;
    const double alpha = (target_time - older_time) / (newer_time - older_time);
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0)
      throw std::runtime_error(
          "ProgramExecutionServices linear history interpolation target does not bracket retained "
          "timestamps");
    lincomb(output, Real(1) - static_cast<Real>(alpha),
            found->second[static_cast<std::size_t>(older_lag)], static_cast<Real>(alpha),
            found->second[static_cast<std::size_t>(older_lag - 1)]);
  }

  bool cache_should_update(ProgramCacheSlot slot, int every_n) const {
    const bool due = runtime_state().cache_.is_due(slot, macro_step(), every_n);
    runtime_state().profiler_.count(due ? "cache_misses" : "cache_hits");
    return due;
  }
  void cache_store_scratch(ProgramCacheSlot slot, const field_type& scratch) const {
    runtime_state().cache_.store(slot, scratch, macro_step());
  }
  void cache_restore_scratch(ProgramCacheSlot slot, field_type& scratch) const {
    runtime_state().cache_.restore_into(slot, scratch);
  }
  void cache_accumulate_dt(ProgramCacheSlot slot, Real dt) const {
    runtime_state().cache_.accumulate_dt(slot, dt);
  }
  Real cache_effective_dt(ProgramCacheSlot slot, Real dt) const {
    return runtime_state().cache_.effective_dt(slot, dt);
  }

  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
  }
  bool schedule_is_due(ProgramCacheSlot slot, int every_n, ScheduleDomainKind kind,
                       const std::string& clock, const std::string& stage_identity,
                       int level) const {
    (void)runtime_state().cache_.plan_entry(slot);
    if (every_n <= 0)
      throw std::invalid_argument("ProgramExecutionServices schedule has an invalid period");
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value % every_n == 0;
  }
  bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                         const std::string& stage_identity, int level) const {
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value == 0;
  }
  bool schedule_decision(ProgramCacheSlot slot, bool due, bool cache_backed) const {
    (void)runtime_state().cache_.plan_entry(slot);
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

  // Candidate execution already holds the Uniform transaction visibility writer lease.  These
  // private host seams read the live carrier for Program-internal scheduling without manufacturing
  // an AcceptedReadLease on the writer thread; public System readers continue to block until seal.
  int macro_step() const {
    if (system_ == nullptr) {
      if (preparation_read_view_ == nullptr)
        throw std::logic_error("Program preparation has no detached clock image");
      return preparation_read_view_->macro_step;
    }
    return system_->program_macro_step_();
  }
  Real physical_time() const {
    if (system_ == nullptr) {
      if (preparation_read_view_ == nullptr)
        throw std::logic_error("Program preparation has no detached clock image");
      return preparation_read_view_->physical_time;
    }
    return static_cast<Real>(system_->program_time_());
  }

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
    runtime_state().note_automatic_balance_capture_due(due, "ProgramExecutionServices");
  }
  void note_step_projection(const std::string& name) const {
    runtime_state().note_step_projection(name);
  }

  void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    runtime_state().profiler_.record(name, std::chrono::duration<double>(elapsed).count());
  }

  runtime_state_type& runtime_state() const {
    if (system_ == nullptr) {
      if (preparation_read_view_ == nullptr || !preparation_read_view_->runtime_state)
        throw std::logic_error("Program preparation has no detached runtime-state image");
      return *preparation_read_view_->runtime_state;
    }
    return system_->program_runtime_state_();
  }

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
    const ExecutionLane& runtime_lane = prepared_execution_lane();
    const ExecutionLane& workspace_lane =
        ::pops::detail::KrylovWorkspaceAccess::execution_lane(workspace);
    const std::string_view workspace_token =
        ::pops::detail::KrylovWorkspaceAccess::materialization_token(workspace);
    std::string lane_contract;
    std::exception_ptr local_error;
    try {
      const bool workspace_active = workspace_lane.active();
      const bool workspace_named = !workspace_lane.identity().empty();
      const bool workspace_tokened = !workspace_token.empty();
      const bool runtime_active = runtime_lane.active();
      const bool runtime_named = !runtime_lane.identity().empty();
      // This is a local communicator comparison only. Every failure is converged below on the
      // runtime-owned lane, so no rank can enter a workspace-lane collective conditionally.
      const bool lanes_congruent = workspace_lane.congruent_with(runtime_lane);
      ExactContractBuilder contract;
      contract.text("pops.prepared-linear-workspace-lane")
          .scalar(std::uint32_t{2})
          .text(workspace_token)
          .text(workspace_lane.identity())
          .presence(workspace_active)
          .presence(workspace_named)
          .presence(workspace_tokened)
          .text(runtime_lane.identity())
          .presence(runtime_active)
          .presence(runtime_named)
          .presence(lanes_congruent);
      lane_contract = std::move(contract).release();
      if (!workspace_active || !workspace_named || !workspace_tokened || !runtime_active ||
          !runtime_named || !lanes_congruent)
        throw std::invalid_argument(
            "Program prepared linear solve requires a workspace lane congruent with its "
            "runtime-authenticated lane");
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, runtime_lane) != 0) {
      if (runtime_lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program prepared linear solve lane validation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("pops.prepared-linear-workspace-lane"),
              std::string_view(lane_contract)}},
            runtime_lane))
      throw std::invalid_argument(
          "Program prepared linear solve workspace lane contract differs across MPI ranks");
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

  [[nodiscard]] SolveOutcome solve_fields() const { return system_->program_solve_fields_(); }

  [[nodiscard]] SolveOutcome solve_fields_from_state(int program_block, field_type& stage) const {
    const int runtime_block = sys_block(program_block);
    require_program_stage_(program_block, runtime_block, stage);
    return system_->program_solve_fields_from_state_(runtime_block, stage);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_state_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int program_block, field_type& stage) const {
    require_boundary_point_(point, "Program single-state field solve");
    if (provider_slot.empty())
      throw std::invalid_argument("Program field solve requires an exact provider slot");
    const int runtime_block = sys_block(program_block);
    require_program_stage_(program_block, runtime_block, stage);
    return system_->program_solve_fields_from_state_at_(point, provider_slot, runtime_block, stage);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_blocks(
      const std::vector<const field_type*>& program_stages) const {
    auto& workspace = hot_path_workspace_;
    workspace.require_bound(program_stages.size(), "Program simultaneous field solve");
    const std::vector<int>& block_map = program_block_map_();
    if (block_map.empty())
      throw std::runtime_error(
          "Program simultaneous field solve requires an explicit program-to-runtime block map");
    if (program_stages.size() != block_map.size())
      throw std::invalid_argument(
          "Program simultaneous field solve requires one slot per Program block");

    std::fill(workspace.solve_runtime_stages.begin(), workspace.solve_runtime_stages.end(),
              nullptr);
    workspace.solve_unique_stages.clear();
    for (std::size_t program = 0; program < program_stages.size(); ++program) {
      const int runtime_block = sys_block(static_cast<int>(program));
      const field_type* const stage = program_stages[program];
      if (stage == nullptr)
        continue;
      require_unaliased_stage_(workspace.solve_unique_stages, *stage);
      require_program_stage_(static_cast<int>(program), runtime_block, *stage);
      if (workspace.solve_runtime_stages[static_cast<std::size_t>(runtime_block)] != nullptr)
        throw std::invalid_argument(
            "Program simultaneous field solve maps two stages to one runtime block");
      workspace.solve_runtime_stages[static_cast<std::size_t>(runtime_block)] = stage;
      workspace.solve_unique_stages.push_back(stage);
    }
    return system_->program_solve_fields_from_blocks_(workspace.solve_runtime_stages);
  }

  [[nodiscard]] SolveOutcome solve_fields_from_blocks_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, std::uint32_t slot,
      std::initializer_list<FieldStageOverride> overrides) const {
    require_boundary_point_(point, "Program simultaneous field solve");
    if (slot >= generated_field_routes_.size() || !generated_field_routes_[slot].prepared)
      throw std::logic_error(
          "Program simultaneous field solve route was not prepared during installation");
    auto& route = generated_field_routes_[slot];
    if (overrides.size() != route.program_blocks.size())
      throw std::logic_error("Program simultaneous field solve changed its prepared route");
    std::fill(route.runtime_stages.begin(), route.runtime_stages.end(), nullptr);
    route.unique_stages.clear();
    for (std::size_t index = 0; index < route.program_blocks.size(); ++index) {
      const FieldStageOverride& override_value = *(overrides.begin() + index);
      const int program_block = route.program_blocks[index];
      if (override_value.program_block != program_block || override_value.state == nullptr)
        throw std::invalid_argument("Program simultaneous field solve differs from its route");
      const int runtime_block = route.runtime_blocks[index];
      require_unaliased_stage_(route.unique_stages, *override_value.state);
      require_program_stage_(program_block, runtime_block, *override_value.state);
      if (route.runtime_stages[static_cast<std::size_t>(runtime_block)] != nullptr)
        throw std::invalid_argument(
            "Program simultaneous field solve maps two stages to one runtime block");
      route.runtime_stages[static_cast<std::size_t>(runtime_block)] = override_value.state;
      route.unique_stages.push_back(override_value.state);
    }
    return system_->program_solve_fields_from_blocks_at_(point, route.field, route.runtime_stages);
  }

  void bind_prepared_generated_field_route_slots(std::size_t slot_count) const {
    if (!generated_field_routes_.empty() && generated_field_routes_.size() != slot_count)
      throw std::logic_error("Program field-route plan changed after preparation");
    generated_field_routes_.resize(slot_count);
  }

  void prepare_generated_field_route(std::uint32_t slot, std::string_view field,
                                     std::initializer_list<int> program_blocks) const {
    if (slot >= generated_field_routes_.size() || field.empty() || program_blocks.size() == 0)
      throw std::invalid_argument("Program generated field route is outside the sealed plan");
    auto& route = generated_field_routes_[slot];
    if (route.prepared) {
      if (route.field != field || route.program_blocks.size() != program_blocks.size() ||
          !std::equal(route.program_blocks.begin(), route.program_blocks.end(),
                      program_blocks.begin()))
        throw std::logic_error("Program generated field route changed after preparation");
      return;
    }
    route.field.assign(field.data(), field.size());
    route.program_blocks.assign(program_blocks.begin(), program_blocks.end());
    route.runtime_blocks.reserve(route.program_blocks.size());
    route.runtime_stages.assign(static_cast<std::size_t>(program_n_blocks_()), nullptr);
    route.unique_stages.reserve(route.program_blocks.size());
    for (std::size_t index = 0; index < route.program_blocks.size(); ++index) {
      const int program_block = route.program_blocks[index];
      if (std::find(route.program_blocks.begin(), route.program_blocks.begin() + index,
                    program_block) != route.program_blocks.begin() + index)
        throw std::invalid_argument("Program generated field route contains duplicate blocks");
      const int runtime_block = sys_block(program_block);
      if (std::find(route.runtime_blocks.begin(), route.runtime_blocks.end(), runtime_block) !=
          route.runtime_blocks.end())
        throw std::invalid_argument("Program generated field route maps two blocks to one owner");
      route.runtime_blocks.push_back(runtime_block);
    }
    route.prepared = true;
  }

 private:
  enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
  using scratch_slot_type = std::array<std::vector<std::optional<field_type>>, 3>;

  struct GeneratedFieldRoute {
    bool prepared = false;
    std::string field;
    std::vector<int> program_blocks;
    std::vector<int> runtime_blocks;
    std::vector<const field_type*> runtime_stages;
    std::vector<const field_type*> unique_stages;
  };

  static runtime_type* require_system_(runtime_type* system) {
    if (system == nullptr)
      throw std::invalid_argument("ProgramExecutionServices requires a non-null ranked System");
    return system;
  }

  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument("ProgramExecutionServices rate identity must be non-negative");
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

  static void require_same_layout_(const field_type& left, const field_type& right,
                                   const char* operation) {
    if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
        left.local_rank() != right.local_rank() || left.local_size() != right.local_size())
      throw std::invalid_argument(std::string(operation) +
                                  " requires the same exact ranked layout");
  }

  static void copy_field_storage_(const field_type& source, field_type& destination) {
    require_same_field_contract_(source, destination,
                                 "ProgramExecutionServices prepared field copy");
    for (std::size_t local = 0; local < source.local_size(); ++local)
      Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
  }

  static void copy_active_valid_cells_(const field_type& source, field_type& destination,
                                       const field_type& active) {
    require_same_layout_(source, destination, "ProgramExecutionServices commit active copy");
    require_same_layout_(source, active, "ProgramExecutionServices commit active mask");
    if (source.ncomp() != destination.ncomp() || active.ncomp() != 1)
      throw std::invalid_argument(
          "ProgramExecutionServices commit active copy requires matching components and a "
          "one-component "
          "mask");
    struct CopyActiveValid {
      FieldView<const Real, Dim> source{};
      FieldView<Real, Dim> destination{};
      FieldView<const Real, Dim> active{};
      int ncomp = 0;
      POPS_HD void operator()(const Index<Dim>& index) const {
        if (active(index, 0) < Real{0.5})
          return;
        for (int component = 0; component < ncomp; ++component)
          destination(index, component) = source(index, component);
      }
    };
    for (std::size_t local = 0; local < source.local_size(); ++local)
      for_each_cell(
          source.box(local),
          CopyActiveValid{std::as_const(source).fab(local).view(), destination.fab(local).view(),
                          std::as_const(active).fab(local).view(), source.ncomp()});
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

  void converge_owner_reduction_(std::exception_ptr local_error, const ExecutionLane& lane,
                                 const char* operation) const {
    if (all_reduce_max(local_error ? 1L : 0L, lane) == 0)
      return;
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(operation) + " failed collectively");
  }

  int resolve_pointwise_program_block_(int program_block, const ExecutionLane& lane) const {
    int runtime_block = -1;
    long local_error = 0;
    try {
      runtime_block = sys_block(program_block);
    } catch (...) {
      local_error = 1;
    }
    if (all_reduce_max(local_error, lane) != 0)
      throw std::runtime_error("Program pointwise block route failed collectively");
    const long program = static_cast<long>(program_block);
    const long runtime = static_cast<long>(runtime_block);
    if (all_reduce_min(program, lane) != all_reduce_max(program, lane) ||
        all_reduce_min(runtime, lane) != all_reduce_max(runtime, lane))
      throw std::runtime_error("Program pointwise block route differs across ranks");
    return runtime_block;
  }

  const std::vector<int>& require_program_block_map_() const {
    const std::vector<int>& block_map = program_block_map_();
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
    const field_type& live = program_state_const_(runtime_block);
    require_same_field_contract_(stage, live, "Program field stage");
    for (int other = 0; other < program_n_blocks_(); ++other) {
      if (other != runtime_block && &stage == &program_state_const_(other))
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
    return BoundaryTopology<Dim>::axis_periodic(program_periodicity_());
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
    const auto periodicity = program_periodicity_();
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
    const ::pops::amr::Rational evaluation_stage =
        logical_phase_begin_ + stage_time_ * logical_phase_span_;
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
          "ProgramExecutionServices scalar field requires positive components and non-negative "
          "ghosts");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    return make_scratch_(prototype, ncomp, ghosts);
  }

  void prime_persistent_scratch_(ScratchKind kind, std::size_t slot, int subslot,
                                 const field_type& prototype, int ncomp,
                                 const Extent<Dim>& ghosts) const {
    if (subslot < 0 || slot >= scratch_.size())
      throw std::out_of_range("Program scratch prime is outside the bind-sealed resource plan");
    auto& family = scratch_[slot][static_cast<std::size_t>(kind)];
    const auto required = static_cast<std::size_t>(subslot) + 1;
    if (family.size() < required)
      family.resize(required);
    auto& entry = family[static_cast<std::size_t>(subslot)];
    if (entry) {
      require_same_field_contract_(*entry, prototype, "Program scratch prime");
      if (entry->ncomp() != ncomp || entry->ghosts() != ghosts)
        throw std::logic_error("Program scratch prime changed a prepared shape");
      return;
    }
    entry.emplace(make_scratch_(prototype, ncomp, ghosts));
  }

  field_type& persistent_scratch_(ScratchKind kind, ProgramCacheSlot slot, int subslot,
                                  const field_type& prototype, int ncomp,
                                  const Extent<Dim>& ghosts) const {
    if (subslot < 0)
      throw std::invalid_argument("ProgramExecutionServices scratch identity must be non-negative");
    if (slot >= scratch_.size())
      throw std::out_of_range("Program scratch is outside the bind-sealed resource plan");
    auto& family = scratch_[slot][static_cast<std::size_t>(kind)];
    const auto index = static_cast<std::size_t>(subslot);
    if (index >= family.size() || !family[index])
      throw std::logic_error("Program scratch was not primed during installation");
    field_type& result = *family[index];
    require_same_field_contract_(result, prototype, "Program scratch");
    if (result.ncomp() != ncomp || result.ghosts() != ghosts)
      throw std::logic_error("Program scratch shape drifted after installation");
    result.set_val(Real(0));
    return result;
  }

  void store_history_(const std::string& name, const field_type& value,
                      std::optional<Real> dt) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end())
      throw std::out_of_range("ProgramExecutionServices history is not registered");
    require_same_field_contract_(found->second.front(), value,
                                 "ProgramExecutionServices history store");
    auto dt_ledger = manager.slot_dt.find(name);
    if (dt_ledger == manager.slot_dt.end() || dt_ledger->second.size() != found->second.size())
      throw std::logic_error(
          "ProgramExecutionServices history dt ledger differs from its ring depth");
    found->second.front() = value;
    if (!manager.initialized.at(name))
      for (std::size_t slot = 1; slot < found->second.size(); ++slot) {
        found->second[slot] = value;
        if (dt)
          dt_ledger->second[slot] = *dt;
      }
    manager.initialized[name] = true;
    manager.store_pending[name] = true;
    if (dt)
      dt_ledger->second.front() = *dt;
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

  mutable runtime_type* system_ = nullptr;
  mutable const std::vector<field_type>* preparation_states_ = nullptr;
  mutable const std::vector<int>* preparation_block_map_ = nullptr;
  mutable const PreparedReadView* preparation_read_view_ = nullptr;
  mutable ClockScheduleState clock_schedule_;
  mutable std::uint64_t scalar_boundary_generation_ = 0;
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::optional<OperatorEvaluationSnapshot> active_operator_snapshot_;
  mutable double current_dt_ = 0.0;
  mutable ::pops::amr::Rational stage_time_{0, 1};
  mutable ::pops::amr::Rational logical_phase_begin_{0, 1};
  mutable ::pops::amr::Rational logical_phase_span_{1, 1};
  mutable double logical_physical_time_offset_ = 0.0;
  mutable std::string primary_clock_;
  /// Dense [ProgramResourcePlan slot][kind][subslot] storage.  It is grown only by explicit
  /// preparation calls; generated step code cannot recover a map/string fallback.
  mutable std::vector<scratch_slot_type> scratch_;
  mutable std::vector<GeneratedFieldRoute> generated_field_routes_;
  /// All pointer packs and accepted-state rollback images used by the selected hot paths.  This
  /// member is bound before the first candidate step and is never resized by execution.
  mutable PreparedHotPathWorkspace hot_path_workspace_;
};

}  // namespace detail

// The AMR engine is implementation-only.  The public execution authority below is the only
// program-facing class irrespective of runtime kind.
}  // namespace pops::runtime::program

#include <pops/runtime/program/detail/program_execution_services_amr_backend.hpp>

namespace pops::runtime::program {

template <int Dim>
class ProgramExecutionPreparationImage;

template <int Dim>
class ProgramExecutionServices : private detail::UniformStorageTopologyAdapter<Dim>,
                                 private detail::AmrStorageTopologyAdapter<
                                     Dim, typename Kokkos::DefaultExecutionSpace::memory_space> {
 public:
  enum class Binding : std::uint8_t { accepted, preparation };

 private:
  // These private bases are storage/topology adapters only.  They own the exact-ranked
  // Uniform and AMR data paths respectively; ProgramExecutionServices is the sole public
  // authority and selects the active runtime kind.  No adapter owns cadence or dispatch:
  // ProgramRuntimeState::dispatch_cadence_step remains the only macro-step dispatcher.
  using uniform_backend = detail::UniformStorageTopologyAdapter<Dim>;
  using amr_backend =
      detail::AmrStorageTopologyAdapter<Dim, typename Kokkos::DefaultExecutionSpace::memory_space>;
  const ProgramPreparationImage* preparation_image_ = nullptr;
  mutable Binding binding_ = Binding::accepted;
  ProgramRuntimeKind runtime_kind_ = ProgramRuntimeKind::uniform;
  mutable ClockScheduleState preparation_clock_schedule_;
  mutable std::string preparation_primary_clock_;

  static Extent<Dim> uniform_ghosts_(int depth) {
    if (depth < 0)
      throw std::invalid_argument("Program scratch ghost depth must be non-negative");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = depth;
    return ghosts;
  }

 public:
  static constexpr int dimension = Dim;
  using field_type = typename uniform_backend::field_type;
  using UniformPreparedReadView = typename uniform_backend::PreparedReadView;
  using FieldStageOverride = detail::ProgramFieldStageOverride<Dim>;
  using RhsGroupRequest = detail::ProgramRhsGroupRequest<Dim>;
  using AmrPreparationTopologyView = typename amr_backend::PreparedAmrTopologyView;

  /// Uniform and AMR keep different private child-window state, but the generated Program has
  /// one scope ABI.  The variant owns exactly one already-constructed backend guard and adds no
  /// allocation or lookup to the execution path.
  class LogicalEvaluationScope final {
   public:
    using uniform_scope_type = typename uniform_backend::LogicalEvaluationScope;
    using amr_scope_type = typename amr_backend::LogicalEvaluationScope;

    explicit LogicalEvaluationScope(uniform_scope_type&& scope) noexcept
        : scope_(std::move(scope)) {}
    explicit LogicalEvaluationScope(amr_scope_type&& scope) noexcept : scope_(std::move(scope)) {}
    LogicalEvaluationScope(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope& operator=(const LogicalEvaluationScope&) = delete;
    LogicalEvaluationScope(LogicalEvaluationScope&&) noexcept = default;
    LogicalEvaluationScope& operator=(LogicalEvaluationScope&&) noexcept = default;

    [[nodiscard]] Real dt() const {
      return std::visit([](const auto& scope) { return scope.dt(); }, scope_);
    }

   private:
    std::variant<uniform_scope_type, amr_scope_type> scope_;
  };

  // Host-image hooks.  They are public only because generated DSO code owns the prelude; each
  // rejects accepted binding and delegates to the retained preparation image below.
  void prepare_rhs_scratch(std::size_t slot, int subslot, int program_block) const;
  void prepare_state_scratch(std::size_t slot, int subslot, int program_block) const;
  void prepare_scalar_scratch(std::size_t slot, int subslot, int program_block, int ncomp,
                              int ghost_depth) const;
  void prepare_cache_slot(std::size_t slot, int program_block) const;
  void prepare_generated_field_route(std::uint32_t slot, std::string_view field,
                                     std::initializer_list<int> program_blocks) const;

  // Image-private forwarding seams.  No generated step calls these names; they retain the dense
  // storage inside this provider while the image is still the sole owner.
  void bind_prepared_uniform_scratch_slots(std::size_t count) const {
    static_cast<const uniform_backend&>(*this).bind_prepared_scratch_slots(count);
  }
  void bind_prepared_generated_field_route_slots(std::size_t count) const {
    if (is_amr())
      static_cast<const amr_backend&>(*this).bind_prepared_generated_field_route_slots(count);
    else
      static_cast<const uniform_backend&>(*this).bind_prepared_generated_field_route_slots(count);
  }
  void bind_prepared_amr_generated_field_route_slots(std::size_t count) const {
    static_cast<const amr_backend&>(*this).bind_prepared_generated_field_route_slots(count);
  }
  void prepare_prepared_generated_field_route(std::uint32_t slot, std::string_view field,
                                              std::initializer_list<int> program_blocks) const {
    if (is_amr())
      static_cast<const amr_backend&>(*this).prepare_generated_field_route(slot, field,
                                                                           program_blocks);
    else
      static_cast<const uniform_backend&>(*this).prepare_generated_field_route(slot, field,
                                                                               program_blocks);
  }
  void bind_prepared_uniform_state_view(
      const std::vector<typename uniform_backend::field_type>* states,
      const std::vector<int>* block_map) const {
    static_cast<const uniform_backend&>(*this).bind_preparation_state_view(states, block_map);
  }
  void bind_preparation_read_view(const UniformPreparedReadView* view) const {
    static_cast<const uniform_backend&>(*this).bind_preparation_read_view(view);
  }
  void prime_prepared_uniform_rhs(std::size_t slot, int subslot,
                                  const typename uniform_backend::field_type& prototype, int ncomp,
                                  int ghost_depth) const {
    static_cast<const uniform_backend&>(*this).prime_rhs_scratch_exact(
        slot, subslot, prototype, ncomp, uniform_ghosts_(ghost_depth));
  }
  void prime_prepared_uniform_state(std::size_t slot, int subslot,
                                    const typename uniform_backend::field_type& prototype,
                                    int ncomp, int ghost_depth) const {
    static_cast<const uniform_backend&>(*this).prime_state_scratch_exact(
        slot, subslot, prototype, ncomp, uniform_ghosts_(ghost_depth));
  }
  void prime_prepared_uniform_scalar(std::size_t slot, int subslot,
                                     const typename uniform_backend::field_type& prototype,
                                     int ncomp, int ghost_depth) const {
    static_cast<const uniform_backend&>(*this).prime_scalar_scratch(slot, subslot, prototype, ncomp,
                                                                    ghost_depth);
  }
  [[nodiscard]] typename uniform_backend::field_type make_prepared_uniform_field_like(
      const typename uniform_backend::field_type& prototype, int ncomp, int ghost_depth) const {
    return static_cast<const uniform_backend&>(*this).make_prepared_field_like(prototype, ncomp,
                                                                               ghost_depth);
  }
  void bind_prepared_amr_scratch_slots(std::size_t count) const {
    amr_backend::bind_prepared_scratch_slots(count);
  }
  void prime_prepared_amr_scratch(std::uint8_t kind, std::size_t slot, int subslot,
                                  int program_block, int ncomp, int ghost_depth) const {
    amr_backend::prime_prepared_scratch(kind, slot, subslot, program_block, ncomp, ghost_depth);
  }
  class PreparedBlockBoundarySession {
   public:
    PreparedBlockBoundarySession() = default;
    PreparedBlockBoundarySession(const PreparedBlockBoundarySession&) = delete;
    PreparedBlockBoundarySession& operator=(const PreparedBlockBoundarySession&) = delete;
    PreparedBlockBoundarySession(PreparedBlockBoundarySession&&) noexcept = default;
    PreparedBlockBoundarySession& operator=(PreparedBlockBoundarySession&&) noexcept = default;

   private:
    friend class ProgramExecutionServices;
    std::shared_ptr<typename uniform_backend::block_boundary_session_type> uniform;
    std::shared_ptr<typename amr_backend::block_boundary_session_type> amr;
  };
  using block_boundary_session_type = PreparedBlockBoundarySession;

  [[nodiscard]] std::shared_ptr<block_boundary_session_type> prepare_block_boundary_session(
      int program_block, field_type& prototype,
      const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
    auto session = std::make_shared<block_boundary_session_type>();
    if (is_amr())
      session->amr =
          amr_backend::prepare_block_boundary_session(program_block, prototype, point, lane);
    else
      session->uniform =
          uniform_backend::prepare_block_boundary_session(program_block, prototype, point, lane);
    return session;
  }

  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                        int program_block, field_type& state, field_type& residual, bool flux_only,
                        const block_boundary_session_type& boundary) const {
    if (is_amr()) {
      if (!boundary.amr || boundary.uniform)
        throw std::invalid_argument("AMR Program core RHS requires its prepared boundary session");
      amr_backend::rhs_core_into_at(point, program_block, state, residual, flux_only,
                                    *boundary.amr);
      return;
    }
    if (!boundary.uniform || boundary.amr)
      throw std::invalid_argument(
          "Uniform Program core RHS requires its prepared boundary session");
    uniform_backend::rhs_core_into_at(point, program_block, state, residual, flux_only,
                                      *boundary.uniform);
  }

  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 int program_block, field_type& state, field_type& residual,
                                 const block_boundary_session_type& boundary) const {
    if (is_amr()) {
      if (!boundary.amr || boundary.uniform)
        throw std::invalid_argument(
            "AMR Program boundary residual requires its prepared boundary session");
      amr_backend::boundary_residual_into_at(point, program_block, state, residual, *boundary.amr);
      return;
    }
    if (!boundary.uniform || boundary.amr)
      throw std::invalid_argument(
          "Uniform Program boundary residual requires its prepared boundary session");
    uniform_backend::boundary_residual_into_at(point, program_block, state, residual,
                                               *boundary.uniform);
  }

  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            int program_block, field_type& state, const field_type& direction,
                            field_type& result, const block_boundary_session_type& boundary) const {
    if (is_amr()) {
      if (!boundary.amr || boundary.uniform)
        throw std::invalid_argument(
            "AMR Program boundary JVP requires its prepared boundary session");
      amr_backend::boundary_jvp_into_at(point, program_block, state, direction, result,
                                        *boundary.amr);
      return;
    }
    if (!boundary.uniform || boundary.amr)
      throw std::invalid_argument(
          "Uniform Program boundary JVP requires its prepared boundary session");
    uniform_backend::boundary_jvp_into_at(point, program_block, state, direction, result,
                                          *boundary.uniform);
  }

  SolveOutcome solve_prepared_linear(const PreparedAffineLinearProblem<Dim>& problem,
                                     KrylovWorkspace<Dim>& workspace, field_type& solution,
                                     const field_type& rhs,
                                     const KrylovControls<Dim>& controls) const {
    if (is_amr())
      return amr_backend::solve_prepared_linear(problem, workspace, solution, rhs, controls);
    return uniform_backend::solve_prepared_linear(problem, workspace, solution, rhs, controls);
  }

  // AMR-only operations are explicit public authority methods.  Uniform never reaches an
  // uninitialised hierarchy adapter: each route refuses before it can allocate or mutate state.
  template <class... Args>
  decltype(auto) advance_hierarchy(Args&&... args) const {
    return amr_only_("advance_hierarchy", [&](auto& backend) -> decltype(auto) {
      return backend.advance_hierarchy(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) advance_synchronized_hierarchy(Args&&... args) const {
    return amr_only_("advance_synchronized_hierarchy", [&](auto& backend) -> decltype(auto) {
      return backend.advance_synchronized_hierarchy(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) refresh_accepted_hierarchy(Args&&... args) const {
    return amr_only_("refresh_accepted_hierarchy", [&](auto& backend) -> decltype(auto) {
      return backend.refresh_accepted_hierarchy(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) accept_history_remap(Args&&... args) const {
    return amr_only_("accept_history_remap", [&](auto& backend) -> decltype(auto) {
      return backend.accept_history_remap(std::forward<Args>(args)...);
    });
  }
  void preflight_restart_regrid() const {
    amr_only_("preflight_restart_regrid",
              [](auto& backend) { backend.preflight_restart_regrid(); });
  }
  void restart_regrid() const {
    amr_only_("restart_regrid", [](auto& backend) { backend.restart_regrid(); });
  }
  void resync_after_restart() const {
    amr_only_("resync_after_restart", [](auto& backend) { backend.resync_after_restart(); });
  }
  decltype(auto) create_accepted_context_snapshot() const {
    return amr_only_("create_accepted_context_snapshot", [](auto& backend) -> decltype(auto) {
      return backend.create_accepted_context_snapshot();
    });
  }
  template <class... Args>
  decltype(auto) for_each_program_resource_level(Args&&... args) const {
    return amr_only_("for_each_program_resource_level", [&](auto& backend) -> decltype(auto) {
      return backend.for_each_program_resource_level(std::forward<Args>(args)...);
    });
  }
  [[nodiscard]] int level() const {
    return amr_only_("level", [](auto& backend) { return backend.level(); });
  }
  template <class... Args>
  decltype(auto) prepare_rebalance(Args&&... args) const {
    return amr_only_("prepare_rebalance", [&](auto& backend) -> decltype(auto) {
      return backend.prepare_rebalance(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) prepare_same_level_cell_temporal_execution(Args&&... args) const {
    return amr_only_(
        "prepare_same_level_cell_temporal_execution", [&](auto& backend) -> decltype(auto) {
          return backend.prepare_same_level_cell_temporal_execution(std::forward<Args>(args)...);
        });
  }
  void advance_same_level_cell_temporal(double dt) const {
    amr_only_("advance_same_level_cell_temporal",
              [dt](auto& backend) { backend.advance_same_level_cell_temporal(dt); });
  }
  template <class... Args>
  decltype(auto) publish_regrid(Args&&... args) const {
    return amr_only_("publish_regrid", [&](auto& backend) -> decltype(auto) {
      return backend.publish_regrid(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) register_hierarchy_tensor_solver_provider(Args&&... args) const {
    return amr_only_(
        "register_hierarchy_tensor_solver_provider", [&](auto& backend) -> decltype(auto) {
          return backend.register_hierarchy_tensor_solver_provider(std::forward<Args>(args)...);
        });
  }
  template <class... Args>
  decltype(auto) solve_default_field_on_coarse_level(Args&&... args) const {
    return amr_only_("solve_default_field_on_coarse_level", [&](auto& backend) -> decltype(auto) {
      return backend.solve_default_field_on_coarse_level(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) solve_hierarchy_tensor(Args&&... args) const {
    return amr_only_("solve_hierarchy_tensor", [&](auto& backend) -> decltype(auto) {
      return backend.solve_hierarchy_tensor(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) solve_source_default(Args&&... args) const {
    return amr_only_("solve_source_default", [&](auto& backend) -> decltype(auto) {
      return backend.solve_source_default(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) stage_linear_initial_guess(Args&&... args) const {
    return amr_only_("stage_linear_initial_guess", [&](auto& backend) -> decltype(auto) {
      return backend.stage_linear_initial_guess(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) configure_hierarchy_tensor_solver(Args&&... args) const {
    return amr_only_("configure_hierarchy_tensor_solver", [&](auto& backend) -> decltype(auto) {
      return backend.configure_hierarchy_tensor_solver(std::forward<Args>(args)...);
    });
  }
  [[nodiscard]] bool uses_prepared_krylov_fallback() const {
    return amr_only_("uses_prepared_krylov_fallback",
                     [](auto& backend) { return backend.uses_prepared_krylov_fallback(); });
  }
  decltype(auto) hierarchy_solution() const {
    return amr_only_("hierarchy_solution",
                     [](auto& backend) -> decltype(auto) { return backend.hierarchy_solution(); });
  }
  template <class... Args>
  decltype(auto) linear_solution(Args&&... args) const {
    return amr_only_("linear_solution", [&](auto& backend) -> decltype(auto) {
      return backend.linear_solution(std::forward<Args>(args)...);
    });
  }
  decltype(auto) program_resource_topology() const {
    return amr_only_("program_resource_topology", [](auto& backend) -> decltype(auto) {
      return backend.program_resource_topology();
    });
  }
  template <class... Args>
  decltype(auto) with_program_resource_level(Args&&... args) const {
    return amr_only_("with_program_resource_level", [&](auto& backend) -> decltype(auto) {
      return backend.with_program_resource_level(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) prepare_regrid(Args&&... args) const {
    return amr_only_("prepare_regrid", [&](auto& backend) -> decltype(auto) {
      return backend.prepare_regrid(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) copy_component_span(Args&&... args) const {
    return amr_only_("copy_component_span", [&](auto& backend) -> decltype(auto) {
      return backend.copy_component_span(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) copy_grown_component_span(Args&&... args) const {
    return amr_only_("copy_grown_component_span", [&](auto& backend) -> decltype(auto) {
      return backend.copy_grown_component_span(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) rhs_jacvec_pair_into_at(Args&&... args) const {
    return amr_only_("rhs_jacvec_pair_into_at", [&](auto& backend) -> decltype(auto) {
      return backend.rhs_jacvec_pair_into_at(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) neg_div_named_flux_into(Args&&... args) const {
    return amr_only_("neg_div_named_flux_into", [&](auto& backend) -> decltype(auto) {
      return backend.neg_div_named_flux_into(std::forward<Args>(args)...);
    });
  }
  template <class... Args>
  decltype(auto) alloc_scalar_field(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.alloc_scalar_field(std::forward<Args>(args)...);
                    })
        return amr_backend::alloc_scalar_field(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: alloc_scalar_field");
    }
    return uniform_backend::alloc_scalar_field(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) apply_projection(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.apply_projection(std::forward<Args>(args)...);
                    })
        return amr_backend::apply_projection(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: apply_projection");
    }
    return uniform_backend::apply_projection(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) apply_source_mask(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.apply_source_mask(std::forward<Args>(args)...);
                    })
        return amr_backend::apply_source_mask(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: apply_source_mask");
    }
    return uniform_backend::apply_source_mask(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) assembly_source(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.assembly_source(std::forward<Args>(args)...);
                    })
        return amr_backend::assembly_source(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: assembly_source");
    }
    return uniform_backend::assembly_source(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) assembly_target(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.assembly_target(std::forward<Args>(args)...);
                    })
        return amr_backend::assembly_target(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: assembly_target");
    }
    return uniform_backend::assembly_target(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) axpy(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.axpy(std::forward<Args>(args)...);
                    })
        return amr_backend::axpy(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: axpy");
    }
    return uniform_backend::axpy(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) boundary_evaluation_point(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.boundary_evaluation_point(std::forward<Args>(args)...);
                    })
        return amr_backend::boundary_evaluation_point(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: boundary_evaluation_point");
    }
    return uniform_backend::boundary_evaluation_point(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) cache_accumulate_dt(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.cache_accumulate_dt(std::forward<Args>(args)...);
                    })
        return amr_backend::cache_accumulate_dt(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: cache_accumulate_dt");
    }
    return uniform_backend::cache_accumulate_dt(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) cache_effective_dt(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.cache_effective_dt(std::forward<Args>(args)...);
                    })
        return amr_backend::cache_effective_dt(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: cache_effective_dt");
    }
    return uniform_backend::cache_effective_dt(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) cache_restore_scratch(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.cache_restore_scratch(std::forward<Args>(args)...);
                    })
        return amr_backend::cache_restore_scratch(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: cache_restore_scratch");
    }
    return uniform_backend::cache_restore_scratch(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) cache_should_update(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.cache_should_update(std::forward<Args>(args)...);
                    })
        return amr_backend::cache_should_update(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: cache_should_update");
    }
    return uniform_backend::cache_should_update(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) cache_store_scratch(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.cache_store_scratch(std::forward<Args>(args)...);
                    })
        return amr_backend::cache_store_scratch(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: cache_store_scratch");
    }
    return uniform_backend::cache_store_scratch(std::forward<Args>(args)...);
  }
  void commit_many(std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
    if (is_amr())
      amr_backend::commit_many(commits);
    else
      uniform_backend::commit_many(commits);
  }
  template <class... Args>
  decltype(auto) configure_program_resource_field_nullspace(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.configure_program_resource_field_nullspace(
                          std::forward<Args>(args)...);
                    })
        return amr_backend::configure_program_resource_field_nullspace(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: configure_program_resource_field_nullspace");
    }
    return uniform_backend::configure_program_resource_field_nullspace(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) divergence(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.divergence(std::forward<Args>(args)...);
                    })
        return amr_backend::divergence(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: divergence");
    }
    return uniform_backend::divergence(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) dot(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.dot(std::forward<Args>(args)...);
                    })
        return amr_backend::dot(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: dot");
    }
    return uniform_backend::dot(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) fill_boundary(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.fill_boundary(std::forward<Args>(args)...);
                    })
        return amr_backend::fill_boundary(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: fill_boundary");
    }
    return uniform_backend::fill_boundary(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) geometry(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.geometry(std::forward<Args>(args)...);
                    })
        return amr_backend::geometry(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: geometry");
    }
    return uniform_backend::geometry(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) gradient(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.gradient(std::forward<Args>(args)...);
                    })
        return amr_backend::gradient(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: gradient");
    }
    return uniform_backend::gradient(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) has_boundary_linearization(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.has_boundary_linearization(std::forward<Args>(args)...);
                    })
        return amr_backend::has_boundary_linearization(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: has_boundary_linearization");
    }
    return uniform_backend::has_boundary_linearization(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) history(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.history(std::forward<Args>(args)...);
                    })
        return amr_backend::history(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: history");
    }
    return uniform_backend::history(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) history_zero_start(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.history_zero_start(std::forward<Args>(args)...);
                    })
        return amr_backend::history_zero_start(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: history_zero_start");
    }
    return uniform_backend::history_zero_start(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) hmin(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.hmin(std::forward<Args>(args)...);
                    })
        return amr_backend::hmin(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: hmin");
    }
    return uniform_backend::hmin(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) laplacian(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.laplacian(std::forward<Args>(args)...);
                    })
        return amr_backend::laplacian(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: laplacian");
    }
    return uniform_backend::laplacian(std::forward<Args>(args)...);
  }
  [[nodiscard]] LogicalEvaluationScope logical_evaluation_scope(int iteration, int count) const {
    if (is_amr())
      return LogicalEvaluationScope(amr_backend::logical_evaluation_scope(iteration, count));
    return LogicalEvaluationScope(uniform_backend::logical_evaluation_scope(iteration, count));
  }
  template <class... Args>
  decltype(auto) macro_step(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.macro_step(std::forward<Args>(args)...);
                    })
        return amr_backend::macro_step(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: macro_step");
    }
    return uniform_backend::macro_step(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) max_component(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.max_component(std::forward<Args>(args)...);
                    })
        return amr_backend::max_component(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: max_component");
    }
    return uniform_backend::max_component(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) max_wave_speed(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.max_wave_speed(std::forward<Args>(args)...);
                    })
        return amr_backend::max_wave_speed(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: max_wave_speed");
    }
    return uniform_backend::max_wave_speed(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) min_component(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.min_component(std::forward<Args>(args)...);
                    })
        return amr_backend::min_component(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: min_component");
    }
    return uniform_backend::min_component(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) norm2(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.norm2(std::forward<Args>(args)...);
                    })
        return amr_backend::norm2(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: norm2");
    }
    return uniform_backend::norm2(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) norm_inf(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.norm_inf(std::forward<Args>(args)...);
                    })
        return amr_backend::norm_inf(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: norm_inf");
    }
    return uniform_backend::norm_inf(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) note_automatic_balance_capture_due(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.note_automatic_balance_capture_due(std::forward<Args>(args)...);
                    })
        return amr_backend::note_automatic_balance_capture_due(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: note_automatic_balance_capture_due");
    }
    return uniform_backend::note_automatic_balance_capture_due(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) note_step_projection(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.note_step_projection(std::forward<Args>(args)...);
                    })
        return amr_backend::note_step_projection(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: note_step_projection");
    }
    return uniform_backend::note_step_projection(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) neg_div_flux_default_into(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.neg_div_flux_default_into(std::forward<Args>(args)...);
                    })
        return amr_backend::neg_div_flux_default_into(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: neg_div_flux_default_into");
    }
    return uniform_backend::neg_div_flux_default_into(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) pack_vector(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.pack_vector(std::forward<Args>(args)...);
                    })
        return amr_backend::pack_vector(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: pack_vector");
    }
    return uniform_backend::pack_vector(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) physical_time(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.physical_time(std::forward<Args>(args)...);
                    })
        return amr_backend::physical_time(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: physical_time");
    }
    return uniform_backend::physical_time(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) pointwise_active_mask(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.pointwise_active_mask(std::forward<Args>(args)...);
                    })
        return amr_backend::pointwise_active_mask(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: pointwise_active_mask");
    }
    return uniform_backend::pointwise_active_mask(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) pointwise_status_max(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.pointwise_status_max(std::forward<Args>(args)...);
                    })
        return amr_backend::pointwise_status_max(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: pointwise_status_max");
    }
    return uniform_backend::pointwise_status_max(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) prepare_generated_state(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.prepare_generated_state(std::forward<Args>(args)...);
                    })
        return amr_backend::prepare_generated_state(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: prepare_generated_state");
    }
    return uniform_backend::prepare_generated_state(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) profile_record(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.profile_record(std::forward<Args>(args)...);
                    })
        return amr_backend::profile_record(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: profile_record");
    }
    return uniform_backend::profile_record(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) probe_operator_evaluation(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.probe_operator_evaluation(std::forward<Args>(args)...);
                    })
        return amr_backend::probe_operator_evaluation(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: probe_operator_evaluation");
    }
    return uniform_backend::probe_operator_evaluation(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) program_params(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.program_params(std::forward<Args>(args)...);
                    })
        return amr_backend::program_params(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: program_params");
    }
    return uniform_backend::program_params(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) program_resource_field_level(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.program_resource_field_level(std::forward<Args>(args)...);
                    })
        return amr_backend::program_resource_field_level(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: program_resource_field_level");
    }
    return uniform_backend::program_resource_field_level(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) program_resource_vector_distribution(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.program_resource_vector_distribution(std::forward<Args>(args)...);
                    })
        return amr_backend::program_resource_vector_distribution(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: program_resource_vector_distribution");
    }
    return uniform_backend::program_resource_vector_distribution(std::forward<Args>(args)...);
  }
  template <int Count, class... Args>
  decltype(auto) provider_values_view(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.template provider_values_view<Count>(std::forward<Args>(args)...);
                    })
        return amr_backend::template provider_values_view<Count>(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: provider_values_view");
    }
    return uniform_backend::template provider_values_view<Count>(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) publish_newton_report(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.publish_newton_report(std::forward<Args>(args)...);
                    })
        return amr_backend::publish_newton_report(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: publish_newton_report");
    }
    return uniform_backend::publish_newton_report(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) record_scalar(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.record_scalar(std::forward<Args>(args)...);
                    })
        return amr_backend::record_scalar(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: record_scalar");
    }
    return uniform_backend::record_scalar(std::forward<Args>(args)...);
  }
  void register_history(const std::string& name, int lag, int ncomp = -1, int program_owner = -1,
                        const std::string& state_identity = {},
                        const std::string& space_identity = {},
                        const std::string& clock_identity = {},
                        const std::string& interpolation_identity = {}) const;
  template <class... Args>
  decltype(auto) require_cartesian_generated_operator(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.require_cartesian_generated_operator(std::forward<Args>(args)...);
                    })
        return amr_backend::require_cartesian_generated_operator(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: require_cartesian_generated_operator");
    }
    return uniform_backend::require_cartesian_generated_operator(std::forward<Args>(args)...);
  }
  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    if (is_amr()) {
      amr_backend::rhs_group(group_id, requests);
      return;
    }
    uniform_backend::rhs_group(group_id, requests);
  }
  template <class... Args>
  decltype(auto) rhs_scratch_like(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.rhs_scratch_like(std::forward<Args>(args)...);
                    })
        return amr_backend::rhs_scratch_like(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: rhs_scratch_like");
    }
    return uniform_backend::rhs_scratch_like(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) schedule_decision(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.schedule_decision(std::forward<Args>(args)...);
                    })
        return amr_backend::schedule_decision(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: schedule_decision");
    }
    return uniform_backend::schedule_decision(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) scratch_state(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.scratch_state(std::forward<Args>(args)...);
                    })
        return amr_backend::scratch_state(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: scratch_state");
    }
    return uniform_backend::scratch_state(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) scratch_state_like(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.scratch_state_like(std::forward<Args>(args)...);
                    })
        return amr_backend::scratch_state_like(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: scratch_state_like");
    }
    return uniform_backend::scratch_state_like(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) operator_evaluation_snapshot(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.operator_evaluation_snapshot(std::forward<Args>(args)...);
                    })
        return amr_backend::operator_evaluation_snapshot(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: operator_evaluation_snapshot");
    }
    return uniform_backend::operator_evaluation_snapshot(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) prepared_execution_communicator(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.prepared_execution_communicator(std::forward<Args>(args)...);
                    })
        return amr_backend::prepared_execution_communicator(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: prepared_execution_communicator");
    }
    return uniform_backend::prepared_execution_communicator(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) prepared_execution_lane(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.prepared_execution_lane(std::forward<Args>(args)...);
                    })
        return amr_backend::prepared_execution_lane(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: prepared_execution_lane");
    }
    return uniform_backend::prepared_execution_lane(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) set_field_boundary_kernel(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.set_field_boundary_kernel(std::forward<Args>(args)...);
                    })
        return amr_backend::set_field_boundary_kernel(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: set_field_boundary_kernel");
    }
    return uniform_backend::set_field_boundary_kernel(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) set_field_boundary_parameters(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.set_field_boundary_parameters(std::forward<Args>(args)...);
                    })
        return amr_backend::set_field_boundary_parameters(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: set_field_boundary_parameters");
    }
    return uniform_backend::set_field_boundary_parameters(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) set_field_logical_timepoint(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.set_field_logical_timepoint(std::forward<Args>(args)...);
                    })
        return amr_backend::set_field_logical_timepoint(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: set_field_logical_timepoint");
    }
    return uniform_backend::set_field_logical_timepoint(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) solve_fields(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.solve_fields(std::forward<Args>(args)...);
                    })
        return amr_backend::solve_fields(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: solve_fields");
    }
    return uniform_backend::solve_fields(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) solve_fields_from_blocks(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.solve_fields_from_blocks(std::forward<Args>(args)...);
                    })
        return amr_backend::solve_fields_from_blocks(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: solve_fields_from_blocks");
    }
    return uniform_backend::solve_fields_from_blocks(std::forward<Args>(args)...);
  }
  [[nodiscard]] SolveOutcome solve_fields_from_blocks_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, std::uint32_t slot,
      std::initializer_list<FieldStageOverride> overrides) const {
    if (is_amr())
      return amr_backend::solve_fields_from_blocks_at(point, slot, overrides);
    return uniform_backend::solve_fields_from_blocks_at(point, slot, overrides);
  }
  template <class... Args>
  decltype(auto) solve_fields_from_state_at(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.solve_fields_from_state_at(std::forward<Args>(args)...);
                    })
        return amr_backend::solve_fields_from_state_at(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: solve_fields_from_state_at");
    }
    return uniform_backend::solve_fields_from_state_at(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) solve_fields_from_state(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.solve_fields_from_state(std::forward<Args>(args)...);
                    })
        return amr_backend::solve_fields_from_state(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: solve_fields_from_state");
    }
    return uniform_backend::solve_fields_from_state(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) evaluate_with_field_state_at(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.evaluate_with_field_state_at(std::forward<Args>(args)...);
                    })
        return amr_backend::evaluate_with_field_state_at(std::forward<Args>(args)...);
      throw std::logic_error(
          "AMR Program does not provide operation: evaluate_with_field_state_at");
    }
    throw std::logic_error(
        "Uniform Program does not support operation: evaluate_with_field_state_at");
  }
  template <class... Args>
  decltype(auto) source_default_into(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.source_default_into(std::forward<Args>(args)...);
                    })
        return amr_backend::source_default_into(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: source_default_into");
    }
    return uniform_backend::source_default_into(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) state(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.state(std::forward<Args>(args)...);
                    })
        return amr_backend::state(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: state");
    }
    return uniform_backend::state(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) store_history(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.store_history(std::forward<Args>(args)...);
                    })
        return amr_backend::store_history(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: store_history");
    }
    return uniform_backend::store_history(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) rotate_histories(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.rotate_histories(std::forward<Args>(args)...);
                    })
        return amr_backend::rotate_histories(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: rotate_histories");
    }
    return uniform_backend::rotate_histories(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) interpolate_history_linear(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.interpolate_history_linear(std::forward<Args>(args)...);
                    })
        return amr_backend::interpolate_history_linear(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: interpolate_history_linear");
    }
    return uniform_backend::interpolate_history_linear(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) subcycle_scope(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.subcycle_scope(std::forward<Args>(args)...);
                    })
        return amr_backend::subcycle_scope(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: subcycle_scope");
    }
    return uniform_backend::subcycle_scope(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) sum_component(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.sum_component(std::forward<Args>(args)...);
                    })
        return amr_backend::sum_component(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: sum_component");
    }
    return uniform_backend::sum_component(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) sys_block(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.sys_block(std::forward<Args>(args)...);
                    })
        return amr_backend::sys_block(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: sys_block");
    }
    return uniform_backend::sys_block(std::forward<Args>(args)...);
  }
  template <class... Args>
  decltype(auto) lincomb(Args&&... args) const {
    if (is_amr()) {
      if constexpr (requires(const amr_backend& backend) {
                      backend.lincomb(std::forward<Args>(args)...);
                    })
        return amr_backend::lincomb(std::forward<Args>(args)...);
      throw std::logic_error("AMR Program does not provide operation: lincomb");
    }
    return uniform_backend::lincomb(std::forward<Args>(args)...);
  }

 private:
  friend class ProgramExecutionPreparationImage<Dim>;

  explicit ProgramExecutionServices(Binding binding)
      : uniform_backend(),
        amr_backend(),
        binding_(binding),
        runtime_kind_(ProgramRuntimeKind::uniform) {}
  explicit ProgramExecutionServices(const typename amr_backend::PreparedAmrTopologyView* view)
      : uniform_backend(),
        amr_backend(view),
        binding_(Binding::preparation),
        runtime_kind_(ProgramRuntimeKind::amr) {}

 public:
  [[nodiscard]] ProgramHostDescriptor program_host_descriptor() const {
    if (!is_amr()) {
      if (!uniform_backend::has_system())
        throw std::logic_error(
            "Uniform Program preparation has no accepted System descriptor before publication");
      return uniform_backend::program_host_descriptor();
    }
    return amr_backend::program_host_descriptor();
  }

  [[nodiscard]] ProgramRuntimeKind runtime_kind() const noexcept { return runtime_kind_; }

  [[nodiscard]] ProgramRuntimeState<Dim>& runtime_state() const {
    if (!is_amr())
      return uniform_backend::runtime_state();
    return amr_backend::runtime_state();
  }

  /// Candidate-prelude-only declarations for the finite accepted-step registries. Generated code
  /// emits these from its reachable lowered graph before the host seals the preparation image;
  /// accepted execution has no mutation seam for registry composition.
  void declare_diagnostic(std::string_view name) const {
    require_preparation_authority_("declare_diagnostic");
    runtime_state().declare_diagnostic(std::string(name));
  }

  void declare_balance_route(std::string_view route) const {
    require_preparation_authority_("declare_balance_route");
    runtime_state().declare_balance_route(std::string(route));
  }

  void declare_step_projection(std::string_view identity) const {
    require_preparation_authority_("declare_step_projection");
    runtime_state().declare_step_projection(std::string(identity));
  }

  void declare_automatic_balance_term(int program_block, int level, int component,
                                      std::string_view term) const {
    require_preparation_authority_("declare_automatic_balance_term");
    if (level < 0)
      throw std::invalid_argument(
          "Program automatic balance declaration requires one prepared hierarchy level");
    const int runtime_block = is_amr() ? amr_backend::sys_block(program_block)
                                       : uniform_backend::sys_block(program_block);
    if (!is_amr() && level != 0)
      throw std::invalid_argument(
          "Uniform Program automatic balance declaration requires level zero");
    if (is_amr() && level >= amr_backend::program_resource_topology().levels)
      throw std::out_of_range(
          "AMR Program automatic balance declaration exceeds the prepared hierarchy");
    runtime_state().declare_automatic_balance_term(runtime_block, level, component, term);
  }

  /// Close the finite accepted-step registries while the provider is still detached.  The
  /// generated prelude is the sole declaration authority; activation can only expose a shape that
  /// has already crossed this boundary.
  void seal_transaction_authorities() const {
    require_preparation_authority_("seal_transaction_authorities");
    runtime_state().bind_transaction_authorities();
  }

  [[nodiscard]] std::string transaction_authority_contract(
      std::uint64_t preparation_generation) const {
    const auto& state = runtime_state();
    if (!state.transaction_authorities_bound())
      throw std::logic_error(
          "Program transaction-authority contract requires bind-sealed registries");
    ExactContractBuilder contract;
    contract.text("pops.program.transaction-authorities")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint32_t>(Dim))
        .scalar(static_cast<std::uint8_t>(runtime_kind_))
        .scalar(preparation_generation)
        .scalar(static_cast<std::uint64_t>(state.diagnostics_.size()));
    for (const auto& [name, slot] : state.diagnostics_) {
      (void)slot;
      contract.text(name);
    }
    contract.scalar(static_cast<std::uint64_t>(state.step_balance_terms_.size()));
    for (const auto& [route, slot] : state.step_balance_terms_) {
      (void)slot;
      contract.text(route);
    }
    contract.scalar(static_cast<std::uint64_t>(state.automatic_balance_terms_.size()));
    for (const auto& [key, slot] : state.automatic_balance_terms_) {
      (void)slot;
      contract.scalar(key.runtime_block)
          .scalar(key.level)
          .scalar(key.component)
          .scalar(static_cast<std::uint8_t>(key.term));
    }
    contract.scalar(static_cast<std::uint64_t>(state.step_projections_.size()));
    for (const auto& projection : state.step_projections_)
      contract.text(projection.identity);
    return std::move(contract).release();
  }

  void begin_step(double dt) const {
    if (is_amr())
      amr_backend::begin_step(dt);
    else
      uniform_backend::begin_step(dt);
  }

  void configure_primary_clock(const std::string& clock) const {
    if (binding_ == Binding::preparation) {
      preparation_clock_schedule_.configure_primary_clock(clock);
      preparation_primary_clock_ = clock;
      if (preparation_image_ != nullptr) {
        auto& image = const_cast<ProgramExecutionPreparationImage<Dim>&>(
            static_cast<const ProgramExecutionPreparationImage<Dim>&>(*preparation_image_));
        image.stage_clock_schedule(preparation_clock_schedule_, preparation_primary_clock_);
      }
    } else if (is_amr())
      amr_backend::configure_primary_clock(clock);
    else
      uniform_backend::configure_primary_clock(clock);
  }

  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    if (binding_ == Binding::preparation) {
      try {
        preparation_clock_schedule_.declare_relation(parent, child, count);
      } catch (const std::exception& error) {
        throw std::runtime_error("Program preparation clock relation '" + parent + "->" + child +
                                 "' failed: " + error.what());
      }
      if (preparation_image_ != nullptr && !preparation_primary_clock_.empty()) {
        auto& image = const_cast<ProgramExecutionPreparationImage<Dim>&>(
            static_cast<const ProgramExecutionPreparationImage<Dim>&>(*preparation_image_));
        image.stage_clock_schedule(preparation_clock_schedule_, preparation_primary_clock_);
      }
      return;
    }
    if (is_amr()) {
      amr_backend::declare_clock_relation(parent, child, count);
      return;
    }
    uniform_backend::declare_clock_relation(parent, child, count);
  }

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (is_amr())
      amr_backend::set_stage_time(numerator, denominator);
    else
      uniform_backend::set_stage_time(numerator, denominator);
  }

  [[nodiscard]] int n_blocks() const {
    return is_amr() ? amr_backend::n_blocks() : uniform_backend::n_blocks();
  }

  [[nodiscard]] typename uniform_backend::field_type& state(int program_block) const {
    return is_amr() ? amr_backend::state(program_block) : uniform_backend::state(program_block);
  }

  typename uniform_backend::field_type& rhs_scratch(
      ProgramCacheSlot slot, int subslot,
      const typename uniform_backend::field_type& prototype) const {
    return is_amr() ? amr_backend::rhs_scratch(slot, subslot, prototype)
                    : uniform_backend::rhs_scratch(slot, subslot, prototype);
  }

  typename uniform_backend::field_type scratch_state_like(
      const typename uniform_backend::field_type& prototype) const {
    return is_amr() ? amr_backend::scratch_state_like(prototype)
                    : uniform_backend::scratch_state_like(prototype);
  }

  typename uniform_backend::field_type& scratch_state(
      ProgramCacheSlot slot, int subslot,
      const typename uniform_backend::field_type& prototype) const {
    return is_amr() ? amr_backend::scratch_state(slot, subslot, prototype)
                    : uniform_backend::scratch_state(slot, subslot, prototype);
  }

  typename uniform_backend::field_type& scalar_scratch(
      ProgramCacheSlot slot, int subslot, const typename uniform_backend::field_type& prototype,
      int ncomp = 1, int ghost_depth = 1) const {
    return is_amr() ? amr_backend::scalar_scratch(slot, subslot, prototype, ncomp, ghost_depth)
                    : uniform_backend::scalar_scratch(slot, subslot, prototype, ncomp, ghost_depth);
  }

  typename uniform_backend::field_type alloc_scalar_field(int ncomp = 1,
                                                          int ghost_depth = 1) const {
    return is_amr() ? amr_backend::alloc_scalar_field(ncomp, ghost_depth)
                    : uniform_backend::alloc_scalar_field(ncomp, ghost_depth);
  }

  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return is_amr() ? amr_backend::schedule_domain_occurs(kind, clock, stage_identity, level)
                    : uniform_backend::schedule_domain_occurs(kind, clock, stage_identity, level);
  }

  bool schedule_is_due(ProgramCacheSlot slot, int every_n, ScheduleDomainKind kind,
                       const std::string& clock, const std::string& stage_identity,
                       int level) const {
    return is_amr()
               ? amr_backend::schedule_is_due(slot, every_n, kind, clock, stage_identity, level)
               : uniform_backend::schedule_is_due(slot, every_n, kind, clock, stage_identity,
                                                  level);
  }

  bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                         const std::string& stage_identity, int level) const {
    return is_amr() ? amr_backend::schedule_at_start(kind, clock, stage_identity, level)
                    : uniform_backend::schedule_at_start(kind, clock, stage_identity, level);
  }

  Real abs_sum_component(const typename uniform_backend::field_type& field, int component) const {
    if (is_amr())
      throw std::logic_error(
          "AMR Program abs_sum_component requires an explicit program block owner");
    return uniform_backend::abs_sum_component(field, component);
  }

  Real abs_sum_component(int program_block, const typename uniform_backend::field_type& field,
                         int component) const {
    if (is_amr())
      return amr_backend::abs_sum_component(program_block, field, component);
    if (program_block != 0)
      throw std::out_of_range("Uniform Program abs_sum_component has only block zero");
    return uniform_backend::abs_sum_component(program_block, field, component);
  }

  void rhs_into(int program_block, typename uniform_backend::field_type& state_value,
                typename uniform_backend::field_type& rhs, int rate_id) const {
    if (is_amr())
      amr_backend::rhs_into(program_block, state_value, rhs, rate_id);
    else
      uniform_backend::rhs_into(program_block, state_value, rhs, rate_id);
  }

  void axpy(typename uniform_backend::field_type& destination, Real factor,
            const typename uniform_backend::field_type& source) const {
    if (is_amr())
      amr_backend::axpy(destination, factor, source);
    else
      uniform_backend::axpy(destination, factor, source);
  }

  void record_balance_term(const std::string& route, const std::string& term, Real value) const {
    if (is_amr())
      amr_backend::record_balance_term(route, term, value);
    else
      uniform_backend::record_balance_term(route, term, value);
  }

  /// Candidate-only route for a generated auxiliary consumer plan.  During preparation it records
  /// the immutable plan in the typed host image; the System/AMR installer validates and publishes
  /// that detached registry after its collective preflight.  It deliberately never calls a facade
  /// mutator from DSO code.
  void stage_auxiliary_consumer_plan(
      runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) const;

 private:
  void require_preparation_authority_(std::string_view operation) const {
    if (binding_ != Binding::preparation || preparation_image_ == nullptr)
      throw std::logic_error("ProgramExecutionServices::" + std::string(operation) +
                             " is available only on a detached preparation image");
  }

  template <int>
  friend struct detail::AmrProgramHistoryRemapCollectiveTestAccess;

  [[nodiscard]] const amr_backend& amr_test_backend_() const noexcept {
    return static_cast<const amr_backend&>(*this);
  }

  template <class Function>
  decltype(auto) amr_only_(std::string_view operation, Function&& function) const {
    if (!is_amr())
      throw std::logic_error("Uniform Program does not provide AMR operation: " +
                             std::string(operation));
    return std::forward<Function>(function)(
        const_cast<amr_backend&>(static_cast<const amr_backend&>(*this)));
  }

 private:
  void bind_preparation_image(const ProgramPreparationImage* image) noexcept {
    preparation_image_ = image;
  }

 public:
  /// Seal the DSO-visible preparation state into the retained provider before publication.  The
  /// provider remains host-owned by ProgramExecutionPreparationImage, so this is a detached
  /// mutation and cannot alter an accepted System.
  void seal_uniform_preparation() const {
    if (is_amr() || binding_ != Binding::preparation)
      return;
    seal_transaction_authorities();
    if (preparation_primary_clock_.empty())
      throw std::logic_error("Uniform Program preparation did not configure a primary clock");
    static_cast<const uniform_backend&>(*this).adopt_prepared_clock(preparation_clock_schedule_,
                                                                    preparation_primary_clock_);
    binding_ = Binding::accepted;
  }

  void activate_uniform_after_collective(System<Dim>* system) const {
    if (is_amr())
      throw std::logic_error("AMR Program cannot activate through the Uniform preparation image");
    static_cast<const uniform_backend&>(*this).bind_accepted_system(system);
    if (binding_ == Binding::preparation && has_staged_uniform_clock())
      seal_uniform_preparation();
    else if (binding_ == Binding::preparation)
      binding_ = Binding::accepted;
  }

  void activate_amr_after_collective(::pops::AmrSystem<Dim>* system) const {
    if (!is_amr() || binding_ != Binding::preparation)
      throw std::logic_error("AMR Program activation requires one preparation provider");
    if (!preparation_primary_clock_.empty())
      static_cast<const amr_backend&>(*this).adopt_prepared_clock(preparation_clock_schedule_,
                                                                  preparation_primary_clock_);
    const_cast<amr_backend&>(static_cast<const amr_backend&>(*this)).bind_accepted_facade(system);
    binding_ = Binding::accepted;
  }

  [[nodiscard]] bool has_staged_uniform_clock() const noexcept {
    return !is_amr() && !preparation_primary_clock_.empty();
  }

  [[nodiscard]] bool is_amr() const noexcept { return runtime_kind() == ProgramRuntimeKind::amr; }

  void set_field_boundary_kernel(const std::string& provider_slot,
                                 const CompiledFieldBoundaryKernel<Dim>& kernel) const {
    if (is_amr() && binding_ == Binding::preparation) {
      stage_amr_field_boundary_kernel<Dim>(preparation_image_, provider_slot, kernel);
      return;
    }
    if (is_amr()) {
      amr_backend::set_field_boundary_kernel(provider_slot, kernel);
      return;
    }
    if (binding_ == Binding::preparation) {
      stage_uniform_field_boundary_kernel<Dim>(preparation_image_, provider_slot, kernel);
      return;
    }
    uniform_backend::set_field_boundary_kernel(provider_slot, kernel);
  }

  void set_field_logical_timepoint(const std::string& provider_slot,
                                   const FieldLogicalTimePoint& point) const {
    if (is_amr() && binding_ == Binding::preparation) {
      stage_amr_field_logical_timepoint<Dim>(preparation_image_, provider_slot, point);
      return;
    }
    if (is_amr()) {
      amr_backend::set_field_logical_timepoint(provider_slot, point);
      return;
    }
    if (binding_ == Binding::preparation) {
      stage_uniform_field_logical_timepoint<Dim>(preparation_image_, provider_slot, point);
      return;
    }
    uniform_backend::set_field_logical_timepoint(provider_slot, point);
  }

  void set_field_boundary_parameters(const std::string& provider_slot,
                                     const std::vector<double>& parameters) const {
    if (is_amr() && binding_ == Binding::preparation) {
      stage_amr_field_boundary_parameters<Dim>(preparation_image_, provider_slot, parameters);
      return;
    }
    if (is_amr()) {
      amr_backend::set_field_boundary_parameters(provider_slot, parameters);
      return;
    }
    if (binding_ == Binding::preparation) {
      stage_uniform_field_boundary_parameters<Dim>(preparation_image_, provider_slot, parameters);
      return;
    }
    uniform_backend::set_field_boundary_parameters(provider_slot, parameters);
  }
};

/// Typed implementation of the opaque v5 preparation image.  It owns the execution services for
/// the complete candidate lifetime and records provider-consumer declarations in a host-owned
/// vector.  No DSO object ever receives a System/AmrSystem pointer through this interface.
template <int Dim>
class ProgramExecutionPreparationImage final : public ProgramPreparationImage {
 public:
  struct HistoryRequest final {
    std::string name;
    int lag = 0;
    int components = -1;
    int program_owner = -1;
    std::string state_identity;
    std::string space_identity;
    std::string clock_identity;
    std::string interpolation_identity;
  };
  struct CacheRequest final {
    std::size_t slot = 0;
    int program_block = -1;
    MultiFab<Dim> prototype;
  };
  struct AmrFieldBoundaryRequest final {
    std::string provider_slot;
    std::optional<CompiledFieldBoundaryKernel<Dim>> kernel;
    std::optional<FieldLogicalTimePoint> point;
    std::optional<std::vector<double>> parameters;
  };
  explicit ProgramExecutionPreparationImage(System<Dim>* system, std::uint64_t generation)
      : ProgramPreparationImage(static_cast<std::uint32_t>(Dim), ProgramRuntimeKind::uniform,
                                system->program_host_descriptor().services, generation),
        uniform_activation_(system),
        uniform_prototypes_(capture_uniform_prototypes_(system)),
        uniform_read_view_(capture_uniform_read_view_(system)),
        provider_(std::shared_ptr<ProgramExecutionServices<Dim>>(new ProgramExecutionServices<Dim>(
            ProgramExecutionServices<Dim>::Binding::preparation))) {
    if (uniform_activation_ == nullptr)
      throw std::invalid_argument("Uniform Program preparation requires one System");
    provider_->bind_preparation_image(this);
    provider_->bind_prepared_uniform_state_view(&uniform_prototypes_, &uniform_block_map_);
    provider_->bind_preparation_read_view(&*uniform_read_view_);
    bind_image_services(adapter_services_(*provider_));
  }

  ProgramExecutionPreparationImage(
      ProgramHostDescriptor source,
      std::shared_ptr<const typename ProgramExecutionServices<Dim>::AmrPreparationTopologyView>
          topology,
      std::uint64_t generation)
      : ProgramPreparationImage(static_cast<std::uint32_t>(Dim), ProgramRuntimeKind::amr,
                                source.services, generation),
        amr_topology_(std::move(topology)),
        provider_(std::shared_ptr<ProgramExecutionServices<Dim>>(
            new ProgramExecutionServices<Dim>(amr_topology_.get()))) {
    if (!amr_topology_)
      throw std::invalid_argument("AMR Program preparation requires one topology image");
    provider_->bind_preparation_image(this);
    bind_image_services(adapter_services_(*provider_));
  }

  [[nodiscard]] const std::shared_ptr<ProgramExecutionServices<Dim>>& provider() const noexcept {
    return provider_;
  }

  /// Bind the symbolic declaration before the first DSO callback.  Runtime-sized rows cannot be
  /// materialized yet: their exact local MultiFab layouts are observed by prepare_* below and are
  /// collectively sealed only after every rank has completed the callback.  The declaration is
  /// nevertheless already dense and immutable, so a candidate cannot manufacture a value-id or
  /// grow a lookup table during preparation.
  void bind_uniform_resource_declaration(
      const std::vector<ProgramInstallationTables::ResourcePlan>& declaration,
      std::vector<int> program_block_map) {
    if (runtime_kind() != ProgramRuntimeKind::uniform)
      throw std::logic_error("Uniform resource declaration cannot bind an AMR preparation image");
    const bool state_free = uniform_prototypes_.empty();
    if (program_block_map.empty()) {
      if (!state_free)
        throw std::invalid_argument("Uniform Program preparation has no explicit block map");
      for (const auto& row : declaration)
        if (row.runtime_sized() || row.owner != "global" || row.level != -1)
          throw std::invalid_argument(
              "state-free Uniform Program can declare only exact global, level-independent "
              "resources");
    } else if (state_free) {
      throw std::invalid_argument(
          "state-free Uniform Program cannot bind a block-owned resource map");
    }
    uniform_resource_declaration_ = declaration;
    for (std::size_t slot = 0; slot < uniform_resource_declaration_.size(); ++slot)
      if (uniform_resource_declaration_[slot].slot != slot)
        throw std::invalid_argument("Uniform Program resource-plan slots are not dense");
    uniform_block_map_ = std::move(program_block_map);
    provider_->bind_prepared_uniform_scratch_slots(uniform_resource_declaration_.size());
    provider_->bind_prepared_generated_field_route_slots(uniform_resource_declaration_.size());
  }

  void bind_amr_resource_declaration(
      const std::vector<ProgramInstallationTables::ResourcePlan>& declaration) {
    if (runtime_kind() != ProgramRuntimeKind::amr)
      throw std::logic_error("AMR resource declaration cannot bind a Uniform image");
    amr_resource_declaration_ = declaration;
    for (std::size_t slot = 0; slot < amr_resource_declaration_.size(); ++slot)
      if (amr_resource_declaration_[slot].slot != slot)
        throw std::invalid_argument("AMR Program resource-plan slots are not dense");
    provider_->bind_prepared_amr_scratch_slots(amr_resource_declaration_.size());
    provider_->bind_prepared_generated_field_route_slots(amr_resource_declaration_.size());
  }

  void prepare_amr_scratch(std::uint8_t kind, std::size_t slot, int subslot, int program_block,
                           int ncomp, int ghost_depth) {
    if (runtime_kind() != ProgramRuntimeKind::amr || slot >= amr_resource_declaration_.size())
      throw std::logic_error("AMR Program scratch preparation has no sealed declaration");
    const auto& row = amr_resource_declaration_[slot];
    const auto& view = *amr_topology_;
    const int runtime_block = view.program_block_map.at(static_cast<std::size_t>(program_block));
    if (row.level < 0 &&
        view.block_prototypes.at(static_cast<std::size_t>(runtime_block)).size() != 1)
      throw std::invalid_argument(
          "AMR runtime-sized resource must declare one exact hierarchy level");
    const std::size_t level = row.level < 0 ? 0U : static_cast<std::size_t>(row.level);
    const auto& prototype =
        view.block_prototypes.at(static_cast<std::size_t>(runtime_block)).at(level);
    if (kind > 2 || row.components == 0)
      throw std::invalid_argument("AMR Program scratch preparation has an invalid shape");
    if (kind == 2 &&
        (ncomp != static_cast<int>(row.components) || ghost_depth != static_cast<int>(row.ghosts)))
      throw std::invalid_argument(
          "AMR Program scalar scratch request differs from its resource declaration");
    ncomp = static_cast<int>(row.components);
    ghost_depth = static_cast<int>(row.ghosts);
    provider_->prime_prepared_amr_scratch(kind, slot, subslot, program_block, ncomp, ghost_depth);
    const auto key = std::tuple{kind, slot, subslot};
    if (std::find(amr_resource_prototype_keys_.begin(), amr_resource_prototype_keys_.end(), key) !=
        amr_resource_prototype_keys_.end())
      return;
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    MultiFab<Dim> exact_prototype(prototype.layout(), prototype.distribution(),
                                  prototype.local_rank(), ncomp, ghosts);
    exact_prototype.set_val(Real(0));
    const std::uint64_t cells = resource_cells_(exact_prototype, "AMR scratch");
    validate_resource_layout_(row, cells, ncomp, ghost_depth, "AMR scratch");
    if (!row.runtime_sized())
      return;
    using kind_type = ProgramInstallationTables::ResourcePrototypeKind;
    const std::array<kind_type, 3> kinds{kind_type::rhs, kind_type::state, kind_type::scalar};
    amr_resource_prototype_keys_.push_back(key);
    amr_resource_prototypes_.push_back(
        {static_cast<std::uint32_t>(slot),
         subslot,
         {cells, sizeof(Real), static_cast<std::uint32_t>(ncomp),
          static_cast<std::uint32_t>(ghost_depth), std::nullopt, std::nullopt},
         kinds.at(kind)});
  }

  [[nodiscard]] std::vector<ProgramInstallationTables::ResourcePrototype>
  take_amr_resource_prototypes() {
    return std::exchange(amr_resource_prototypes_, {});
  }

  void prepare_amr_cache(std::size_t slot, int program_block) {
    if (runtime_kind() != ProgramRuntimeKind::amr || slot >= amr_resource_declaration_.size() ||
        program_block < 0 ||
        static_cast<std::size_t>(program_block) >= amr_topology_->program_block_map.size())
      throw std::logic_error("AMR Program cache preparation has no sealed declaration");
    const auto& row = amr_resource_declaration_[slot];
    if (row.lifetime != "persistent" && row.lifetime != "persistent_schedule")
      throw std::invalid_argument("AMR cache preparation requires a persistent schedule row");
    const auto& view = *amr_topology_;
    const int runtime_block = view.program_block_map.at(static_cast<std::size_t>(program_block));
    if (row.level < 0 &&
        view.block_prototypes.at(static_cast<std::size_t>(runtime_block)).size() != 1)
      throw std::invalid_argument("AMR runtime-sized cache must declare one exact hierarchy level");
    const std::size_t level = row.level < 0 ? 0U : static_cast<std::size_t>(row.level);
    const auto& prototype =
        view.block_prototypes.at(static_cast<std::size_t>(runtime_block)).at(level);
    if (row.components == 0)
      throw std::invalid_argument("AMR Program cache declaration has no component shape");
    const int ncomp = static_cast<int>(row.components);
    const int ghost_depth = static_cast<int>(row.ghosts);
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    MultiFab<Dim> cache_prototype(prototype.layout(), prototype.distribution(),
                                  prototype.local_rank(), ncomp, ghosts);
    cache_prototype.set_val(Real(0));
    const auto existing =
        std::find_if(amr_cache_requests_.begin(), amr_cache_requests_.end(),
                     [slot](const CacheRequest& request) { return request.slot == slot; });
    if (existing != amr_cache_requests_.end()) {
      if (existing->program_block != program_block || existing->prototype.ncomp() != ncomp ||
          existing->prototype.ghosts() != cache_prototype.ghosts())
        throw std::logic_error("AMR Program cache preparation changed a prepared shape");
      return;
    }
    const std::uint64_t cells = resource_cells_(cache_prototype, "AMR cache");
    validate_resource_layout_(row, cells, ncomp, ghost_depth, "AMR cache");
    const auto key = std::tuple{std::uint8_t{3}, slot, 0};
    if (row.runtime_sized()) {
      if (std::find(amr_resource_prototype_keys_.begin(), amr_resource_prototype_keys_.end(),
                    key) != amr_resource_prototype_keys_.end())
        throw std::logic_error("AMR Program cache prototype was staged twice");
      amr_resource_prototype_keys_.push_back(key);
      amr_resource_prototypes_.push_back(
          {static_cast<std::uint32_t>(slot),
           0,
           {cells, sizeof(Real), static_cast<std::uint32_t>(ncomp),
            static_cast<std::uint32_t>(ghost_depth), std::nullopt, std::nullopt},
           ProgramInstallationTables::ResourcePrototypeKind::persistent_schedule});
    }
    amr_cache_requests_.push_back({slot, program_block, std::move(cache_prototype)});
  }

  [[nodiscard]] std::vector<CacheRequest> take_amr_cache_requests() {
    return std::exchange(amr_cache_requests_, {});
  }

  void prepare_uniform_scratch(std::uint8_t kind, std::size_t slot, int subslot, int program_block,
                               int ncomp, int ghost_depth) {
    if (runtime_kind() != ProgramRuntimeKind::uniform ||
        slot >= uniform_resource_declaration_.size())
      throw std::logic_error("Uniform Program scratch preparation has no sealed declaration");
    if (subslot < 0 || program_block < 0 ||
        program_block >= static_cast<int>(uniform_block_map_.size()))
      throw std::out_of_range("Program scratch preparation is outside its sealed declaration");
    const auto& row = uniform_resource_declaration_[slot];
    const int runtime_block = uniform_block_map_[static_cast<std::size_t>(program_block)];
    if (runtime_block < 0 || runtime_block >= static_cast<int>(uniform_prototypes_.size()))
      throw std::logic_error("Program scratch preparation targets an absent Uniform prototype");
    const auto& prototype = uniform_prototypes_[static_cast<std::size_t>(runtime_block)];
    if (kind > 2 || row.components == 0)
      throw std::invalid_argument("Uniform Program scratch preparation has an invalid shape");
    if (kind == 2 &&
        (ncomp != static_cast<int>(row.components) || ghost_depth != static_cast<int>(row.ghosts)))
      throw std::invalid_argument(
          "Uniform Program scalar scratch request differs from its resource declaration");
    ncomp = static_cast<int>(row.components);
    ghost_depth = static_cast<int>(row.ghosts);
    switch (kind) {
      case 0:
        provider_->prime_prepared_uniform_rhs(slot, subslot, prototype, ncomp, ghost_depth);
        break;
      case 1:
        provider_->prime_prepared_uniform_state(slot, subslot, prototype, ncomp, ghost_depth);
        break;
      case 2:
        provider_->prime_prepared_uniform_scalar(slot, subslot, prototype, ncomp, ghost_depth);
        break;
      default:
        throw std::invalid_argument("Program scratch preparation has an unknown kind");
    }
    MultiFab<Dim> exact_prototype =
        provider_->make_prepared_uniform_field_like(prototype, ncomp, ghost_depth);
    const std::uint64_t cells = resource_cells_(exact_prototype, "Uniform scratch");
    validate_resource_layout_(row, cells, ncomp, ghost_depth, "Uniform scratch");
    if (!row.runtime_sized())
      return;
    const auto key = std::tuple{kind, slot, subslot};
    if (std::find(uniform_resource_prototype_keys_.begin(), uniform_resource_prototype_keys_.end(),
                  key) != uniform_resource_prototype_keys_.end())
      return;
    using prototype_type = ProgramInstallationTables::ResourcePrototype;
    using kind_type = ProgramInstallationTables::ResourcePrototypeKind;
    const std::array<kind_type, 3> kinds{kind_type::rhs, kind_type::state, kind_type::scalar};
    uniform_resource_prototype_keys_.push_back(key);
    uniform_resource_prototypes_.push_back(
        {static_cast<std::uint32_t>(slot),
         subslot,
         {cells, sizeof(Real), static_cast<std::uint32_t>(ncomp),
          static_cast<std::uint32_t>(ghost_depth), std::nullopt, std::nullopt},
         kinds.at(kind)});
  }

  [[nodiscard]] std::vector<ProgramInstallationTables::ResourcePrototype>
  take_uniform_resource_prototypes() {
    return std::exchange(uniform_resource_prototypes_, {});
  }

  void prepare_uniform_cache(std::size_t slot, int program_block) {
    if (runtime_kind() != ProgramRuntimeKind::uniform ||
        slot >= uniform_resource_declaration_.size() || program_block < 0 ||
        program_block >= static_cast<int>(uniform_block_map_.size()))
      throw std::logic_error("Uniform Program cache preparation has no sealed declaration");
    const auto& row = uniform_resource_declaration_[slot];
    if (row.lifetime != "persistent" && row.lifetime != "persistent_schedule")
      throw std::invalid_argument("Program cache preparation requires a persistent schedule row");
    const int runtime_block = uniform_block_map_.at(static_cast<std::size_t>(program_block));
    if (runtime_block < 0 || runtime_block >= static_cast<int>(uniform_prototypes_.size()))
      throw std::logic_error("Uniform Program cache preparation targets an absent prototype");
    if (row.components == 0)
      throw std::invalid_argument("Uniform Program cache declaration has no component shape");
    const auto& source = uniform_prototypes_[static_cast<std::size_t>(runtime_block)];
    const int ncomp = static_cast<int>(row.components);
    const int ghost_depth = static_cast<int>(row.ghosts);
    MultiFab<Dim> prototype =
        provider_->make_prepared_uniform_field_like(source, ncomp, ghost_depth);
    const auto existing =
        std::find_if(uniform_cache_requests_.begin(), uniform_cache_requests_.end(),
                     [slot](const CacheRequest& request) { return request.slot == slot; });
    if (existing != uniform_cache_requests_.end()) {
      if (existing->program_block != program_block || existing->prototype.ncomp() != ncomp ||
          existing->prototype.ghosts() != prototype.ghosts())
        throw std::logic_error("Uniform Program cache preparation changed a prepared shape");
      return;
    }
    const std::uint64_t cells = resource_cells_(prototype, "Uniform cache");
    validate_resource_layout_(row, cells, ncomp, ghost_depth, "Uniform cache");
    const auto key = std::tuple{std::uint8_t{3}, slot, 0};
    if (row.runtime_sized()) {
      if (std::find(uniform_resource_prototype_keys_.begin(),
                    uniform_resource_prototype_keys_.end(),
                    key) != uniform_resource_prototype_keys_.end())
        throw std::logic_error("Uniform Program cache prototype was staged twice");
      uniform_resource_prototype_keys_.push_back(key);
      uniform_resource_prototypes_.push_back(
          {static_cast<std::uint32_t>(slot),
           0,
           {cells, sizeof(Real), static_cast<std::uint32_t>(ncomp),
            static_cast<std::uint32_t>(ghost_depth), std::nullopt, std::nullopt},
           ProgramInstallationTables::ResourcePrototypeKind::persistent_schedule});
    }
    uniform_cache_requests_.push_back({slot, program_block, std::move(prototype)});
  }

  void prepare_generated_field_route(std::uint32_t slot, std::string_view field,
                                     std::initializer_list<int> program_blocks) {
    if (slot >= (runtime_kind() == ProgramRuntimeKind::uniform
                     ? uniform_resource_declaration_.size()
                     : amr_resource_declaration_.size()))
      throw std::out_of_range("Program generated field route is outside its sealed plan");
    provider_->prepare_prepared_generated_field_route(slot, field, program_blocks);
  }

  [[nodiscard]] std::vector<CacheRequest> take_uniform_cache_requests() {
    return std::exchange(uniform_cache_requests_, {});
  }

  void seed_uniform_field_boundaries(ArtifactFieldBoundaryAuthorityRegistry<Dim> authorities) {
    if (runtime_kind() != ProgramRuntimeKind::uniform)
      throw std::logic_error("Uniform field boundaries cannot seed an AMR preparation image");
    if (uniform_field_boundaries_)
      throw std::logic_error("Uniform field-boundary stage was seeded more than once");
    uniform_field_boundaries_.emplace();
    uniform_field_boundaries_->authorities = std::move(authorities);
  }

  void stage_uniform_field_boundary_kernel(const std::string& provider_slot,
                                           const CompiledFieldBoundaryKernel<Dim>& kernel) {
    kernel.validate();
    auto& authority = uniform_boundary_authority_(provider_slot);
    if (authority.kernel)
      throw std::logic_error("Program field boundary kernel was staged more than once");
    authority.kernel = kernel;
  }

  void stage_uniform_field_logical_timepoint(const std::string& provider_slot,
                                             const FieldLogicalTimePoint& point) {
    auto& authority = uniform_boundary_authority_(provider_slot);
    if (authority.point)
      throw std::logic_error("Program field logical timepoint was staged more than once");
    authority.point = point;
  }

  void stage_uniform_field_boundary_parameters(const std::string& provider_slot,
                                               const std::vector<double>& parameters) {
    if (!std::all_of(parameters.begin(), parameters.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("Program field boundary parameters must be finite");
    auto& authority = uniform_boundary_authority_(provider_slot);
    if (!authority.parameters.empty())
      throw std::logic_error("Program field boundary parameters were staged more than once");
    authority.parameters = parameters;
  }

  [[nodiscard]] ArtifactFieldBoundaryAuthorityRegistry<Dim> take_uniform_field_boundaries() {
    if (!uniform_field_boundaries_)
      throw std::logic_error("Uniform field-boundary stage was not seeded");
    return std::move(uniform_field_boundaries_->authorities);
  }

  void stage_amr_field_boundary_kernel(const std::string& provider_slot,
                                       const CompiledFieldBoundaryKernel<Dim>& kernel) {
    kernel.validate();
    auto& request = amr_boundary_request_(provider_slot);
    if (request.kernel)
      throw std::logic_error("AMR Program field boundary kernel was staged more than once");
    request.kernel = kernel;
  }

  void stage_amr_field_logical_timepoint(const std::string& provider_slot,
                                         const FieldLogicalTimePoint& point) {
    auto& request = amr_boundary_request_(provider_slot);
    if (request.point)
      throw std::logic_error("AMR Program field logical timepoint was staged more than once");
    request.point = point;
  }

  void stage_amr_field_boundary_parameters(const std::string& provider_slot,
                                           const std::vector<double>& parameters) {
    if (!std::all_of(parameters.begin(), parameters.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("AMR Program field boundary parameters must be finite");
    auto& request = amr_boundary_request_(provider_slot);
    if (request.parameters)
      throw std::logic_error("AMR Program field boundary parameters were staged more than once");
    request.parameters = parameters;
  }

  [[nodiscard]] std::vector<AmrFieldBoundaryRequest> take_amr_field_boundaries() {
    if (runtime_kind() != ProgramRuntimeKind::amr)
      throw std::logic_error("AMR field-boundary stage requires an AMR preparation image");
    return std::exchange(amr_field_boundaries_, {});
  }

  void activate_uniform_after_collective() const {
    if (runtime_kind() != ProgramRuntimeKind::uniform || uniform_activation_ == nullptr)
      throw std::logic_error("Program preparation image has no Uniform activation authority");
    provider_->activate_uniform_after_collective(uniform_activation_);
  }

  void activate_amr_after_collective(::pops::AmrSystem<Dim>* system) const {
    if (runtime_kind() != ProgramRuntimeKind::amr)
      throw std::logic_error("Program preparation image has no AMR activation authority");
    provider_->activate_amr_after_collective(system);
  }

  void stage_auxiliary_consumer_plan(runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) {
    plan.validate();
    provider_plans_.push_back(std::move(plan));
  }

  void stage_history(HistoryRequest request) {
    if (request.name.empty() || request.lag < 1)
      throw std::invalid_argument("Program staged history has an invalid identity or lag");
    for (const auto& existing : histories_) {
      if (existing.name == request.name) {
        if (existing.lag != request.lag || existing.components != request.components ||
            existing.program_owner != request.program_owner ||
            existing.state_identity != request.state_identity ||
            existing.space_identity != request.space_identity ||
            existing.clock_identity != request.clock_identity ||
            existing.interpolation_identity != request.interpolation_identity)
          throw std::invalid_argument("Program staged history has conflicting declarations");
        return;
      }
    }
    histories_.push_back(std::move(request));
  }

  /// Materialize every authored AMR history in the detached candidate state before collective
  /// sealing.  The adapter walks the frozen topology view, so this never consults an accepted
  /// facade or allocates after the installation boundary.
  void materialize_amr_staged_histories() {
    if (runtime_kind() != ProgramRuntimeKind::amr)
      throw std::logic_error("AMR history materialization requires an AMR preparation image");
    provider_->amr_only_("materialize_amr_staged_histories", [&](auto& backend) {
      if (has_staged_clock()) {
        try {
          backend.adopt_prepared_clock(staged_clock_schedule_, staged_primary_clock_);
        } catch (const std::exception& error) {
          throw std::runtime_error("AMR staged clock adoption failed: " +
                                   std::string(error.what()));
        }
      }
      backend.for_each_program_resource_level([&](int) {
        for (const auto& request : histories_) {
          try {
            backend.register_history(request.name, request.lag, request.components,
                                     request.program_owner, request.state_identity,
                                     request.space_identity, request.clock_identity,
                                     request.interpolation_identity);
          } catch (const std::exception& error) {
            throw std::runtime_error("AMR staged history '" + request.name +
                                     "' materialization failed: " + error.what());
          }
        }
      });
    });
  }

  // A native candidate may declare clocks while its image is still private.  In particular this
  // prevents AMR's prepare callback from mutating the accepted hierarchy schedule before the
  // all-rank installation decision.
  void stage_clock_schedule(ClockScheduleState schedule, std::string primary_clock) {
    if (primary_clock.empty())
      throw std::invalid_argument("Program staged clock has no primary identity");
    staged_clock_schedule_ = std::move(schedule);
    staged_primary_clock_ = std::move(primary_clock);
  }

  [[nodiscard]] bool has_staged_clock() const noexcept { return !staged_primary_clock_.empty(); }
  [[nodiscard]] const ClockScheduleState& staged_clock_schedule() const {
    if (!has_staged_clock())
      throw std::logic_error("Program preparation image has no staged clock");
    return staged_clock_schedule_;
  }
  [[nodiscard]] const std::string& staged_primary_clock() const {
    if (!has_staged_clock())
      throw std::logic_error("Program preparation image has no staged primary clock");
    return staged_primary_clock_;
  }

  [[nodiscard]] std::vector<runtime::system::AuxiliaryConsumerProviderPlan<Dim>>
  take_auxiliary_consumer_plans() {
    return std::exchange(provider_plans_, {});
  }

  [[nodiscard]] std::vector<HistoryRequest> take_histories() {
    return std::exchange(histories_, {});
  }

  void seal_transaction_authorities() const { provider_->seal_transaction_authorities(); }

  [[nodiscard]] std::string transaction_authority_contract() const {
    return provider_->transaction_authority_contract(generation());
  }

  [[nodiscard]] ProgramRuntimeState<Dim> take_uniform_transaction_authority_state() {
    if (runtime_kind() != ProgramRuntimeKind::uniform || !uniform_read_view_ ||
        !uniform_read_view_->runtime_state)
      throw std::logic_error(
          "Uniform Program transaction-authority image was already consumed or is absent");
    if (!uniform_read_view_->runtime_state->transaction_authorities_bound())
      throw std::logic_error("Uniform Program transaction-authority image was not bind-sealed");
    ProgramRuntimeState<Dim> result = std::move(*uniform_read_view_->runtime_state);
    uniform_read_view_->runtime_state.reset();
    return result;
  }

 private:
  static std::vector<MultiFab<Dim>> capture_uniform_prototypes_(System<Dim>* system) {
    if (system == nullptr)
      throw std::invalid_argument("Uniform Program preparation requires one System");
    const int count = system->program_n_blocks_();
    if (count < 0)
      throw std::logic_error("Uniform Program preparation reported a negative block count");
    std::vector<MultiFab<Dim>> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int block = 0; block < count; ++block)
      result.emplace_back(system->program_block_state_(block));
    return result;
  }

  static typename ProgramExecutionServices<Dim>::UniformPreparedReadView capture_uniform_read_view_(
      System<Dim>* system) {
    if (system == nullptr)
      throw std::invalid_argument("Uniform Program preparation requires one System");
    const ExecutionLane& accepted = system->program_prepared_boundary_execution_lane_();
    // The image owns a distinct communicator before the DSO runs.  It never borrows the
    // accepted lane, so a failed prepare cannot retain a facade-owned MPI object.
#ifdef POPS_HAS_MPI
    auto preparation_lane = ExecutionLane::duplicate_collectively(
        ExecutionCommunicator::borrowed(accepted.identity(), accepted.native_handle()),
        "program-preparation");
#else
    auto preparation_lane = ExecutionLane::duplicate_world_collectively("program-preparation");
#endif
    typename ProgramExecutionServices<Dim>::UniformPreparedReadView view{
        std::move(preparation_lane),
        system->program_prepared_block_geometry_(),
        system->program_prepared_block_periodicity_(),
        system->program_cfl_min_dx_(),
        system->macro_step(),
        static_cast<Real>(system->time())};
    view.runtime_state = std::make_unique<ProgramRuntimeState<Dim>>();
    // Public accepted observers return values or lease-owned views.  Copy both into the image now:
    // after this function returns, no candidate callback has a route back to the live cache or
    // profiler even if it fails before collective acceptance.
    view.runtime_state->profiler_ = system->profiler();
    auto accepted_cache = system->program_cache();
    if (!accepted_cache || accepted_cache.get() == nullptr)
      throw std::logic_error("Uniform Program preparation has no accepted cache image");
    view.runtime_state->cache_ = *accepted_cache.get();
    // Auxiliary storage may legitimately be absent before any provider graph is sealed.  Capture
    // it only after the accepted registry has crossed its explicit seal boundary.  Once sealed,
    // every failure is a malformed/unstable accepted image and must propagate; a broad
    // logic_error catch here would silently turn corrupted storage into "no providers".
    if (system->program_auxiliary_registry_sealed_()) {
      view.auxiliary_registry = system->capture_auxiliary_checkpoint_accepted_state();
      const auto carrier = system->prepared_block_provider_storage_owner();
      if (carrier)
        view.provider_carrier =
            std::make_shared<runtime::system::AuxiliaryStorageGroups<Dim>>(*carrier);
    }
    if (!view.provider_carrier)
      view.provider_carrier = std::make_shared<runtime::system::AuxiliaryStorageGroups<Dim>>();
    const int blocks = system->program_n_blocks_();
    view.params.reserve(static_cast<std::size_t>(blocks));
    view.newton.reserve(static_cast<std::size_t>(blocks));
    view.requires_boundary.reserve(static_cast<std::size_t>(blocks));
    view.has_linearization.reserve(static_cast<std::size_t>(blocks));
    for (int block = 0; block < blocks; ++block) {
      view.params.push_back(system->program_params_(block));
      view.newton.push_back(system->program_block_newton_options_(block));
      view.requires_boundary.push_back(system->program_requires_block_boundary_session_(block));
      view.has_linearization.push_back(system->program_has_block_boundary_linearization_(block));
    }
    return view;
  }

  static int uniform_ghost_depth_(const Extent<Dim>& ghosts) {
    const int depth = ghosts[0];
    if (depth < 0)
      throw std::invalid_argument("Uniform Program scratch has a negative ghost depth");
    for (int axis = 1; axis < Dim; ++axis)
      if (ghosts[axis] != depth)
        throw std::invalid_argument(
            "Uniform Program scratch requires one exact isotropic ghost depth");
    return depth;
  }

  static std::uint64_t resource_cells_(const MultiFab<Dim>& prototype, std::string_view what) {
    std::uint64_t cells = 0;
    for (std::size_t local = 0; local < prototype.local_size(); ++local) {
      const auto points = prototype.fab(local).grown_box().numPts();
      if (points < 0 ||
          static_cast<std::uint64_t>(points) > std::numeric_limits<std::uint64_t>::max() - cells)
        throw std::overflow_error(std::string(what) + " prototype cell count overflows uint64");
      cells += static_cast<std::uint64_t>(points);
    }
    return cells;
  }

  static void validate_resource_layout_(const ProgramInstallationTables::ResourcePlan& row,
                                        std::uint64_t cells, int components, int ghosts,
                                        std::string_view what) {
    if (components < 1 || ghosts < 0 || row.components != static_cast<std::uint32_t>(components) ||
        row.ghosts != static_cast<std::uint32_t>(ghosts))
      throw std::invalid_argument(std::string(what) +
                                  " shape differs from its resource declaration");
    if (row.runtime_sized())
      return;
    if (!row.cells || !row.itemsize || !row.bytes || !row.maximum_bytes || *row.cells != cells ||
        *row.itemsize != sizeof(Real) ||
        cells > std::numeric_limits<std::uint64_t>::max() / sizeof(Real) /
                    static_cast<std::uint64_t>(components))
      throw std::invalid_argument(std::string(what) +
                                  " exact layout differs from its resource declaration");
    const auto bytes = cells * sizeof(Real) * static_cast<std::uint64_t>(components);
    if (*row.bytes != bytes || *row.maximum_bytes < bytes)
      throw std::invalid_argument(std::string(what) +
                                  " exact byte layout differs from its resource declaration");
  }

  ArtifactFieldBoundaryAuthority<Dim>& uniform_boundary_authority_(
      const std::string& provider_slot) {
    if (!uniform_field_boundaries_)
      throw std::logic_error("Uniform field-boundary stage was not seeded");
    const auto found = uniform_field_boundaries_->authorities.find(provider_slot);
    if (found == uniform_field_boundaries_->authorities.end())
      throw std::out_of_range("Program field boundary names an unknown provider slot");
    return found->second;
  }

  AmrFieldBoundaryRequest& amr_boundary_request_(const std::string& provider_slot) {
    if (runtime_kind() != ProgramRuntimeKind::amr || provider_slot.empty())
      throw std::invalid_argument("AMR Program field boundary has no provider slot");
    const auto found = std::find_if(amr_field_boundaries_.begin(), amr_field_boundaries_.end(),
                                    [&](const AmrFieldBoundaryRequest& request) {
                                      return request.provider_slot == provider_slot;
                                    });
    if (found != amr_field_boundaries_.end())
      return *found;
    amr_field_boundaries_.push_back({.provider_slot = provider_slot});
    return amr_field_boundaries_.back();
  }

  static ProgramExecutionServicesRef adapter_services_(
      ProgramExecutionServices<Dim>& provider) noexcept {
    // The C ABI intentionally carries opaque identities only. Every required service resolves
    // through the one retained provider/image object; no System/AmrSystem address crosses prepare.
    void* const adapter = static_cast<void*>(std::addressof(provider));
    return {adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter};
  }

  System<Dim>* uniform_activation_ = nullptr;
  std::shared_ptr<const typename ProgramExecutionServices<Dim>::AmrPreparationTopologyView>
      amr_topology_;
  std::shared_ptr<ProgramExecutionServices<Dim>> provider_;
  std::vector<ProgramInstallationTables::ResourcePlan> uniform_resource_declaration_;
  std::vector<int> uniform_block_map_;
  std::vector<MultiFab<Dim>> uniform_prototypes_;
  // AMR images do not own a Uniform lane.  Keeping this optional avoids constructing a dummy
  // communicator solely to satisfy an unrelated aggregate member.
  std::optional<typename ProgramExecutionServices<Dim>::UniformPreparedReadView> uniform_read_view_;
  std::vector<std::tuple<std::uint8_t, std::size_t, int>> uniform_resource_prototype_keys_;
  std::vector<ProgramInstallationTables::ResourcePrototype> uniform_resource_prototypes_;
  std::vector<ProgramInstallationTables::ResourcePlan> amr_resource_declaration_;
  std::vector<std::tuple<std::uint8_t, std::size_t, int>> amr_resource_prototype_keys_;
  std::vector<ProgramInstallationTables::ResourcePrototype> amr_resource_prototypes_;
  std::vector<CacheRequest> amr_cache_requests_;
  std::vector<CacheRequest> uniform_cache_requests_;
  std::optional<ArtifactFieldBoundaryStage<Dim>> uniform_field_boundaries_;
  std::vector<AmrFieldBoundaryRequest> amr_field_boundaries_;
  std::vector<runtime::system::AuxiliaryConsumerProviderPlan<Dim>> provider_plans_;
  std::vector<HistoryRequest> histories_;
  ClockScheduleState staged_clock_schedule_;
  std::string staged_primary_clock_;
};

/// Host-only v5 image used for the descriptor callback.  Its service words are distinct static
/// sentinels, never a System/AmrSystem address and never a callable adapter.  The final typed
/// image replaces it before the candidate receives `prepare`.
class ProgramInspectionPreparationImage final : public ProgramPreparationImage {
 public:
  ProgramInspectionPreparationImage(std::uint32_t native_dimension, ProgramRuntimeKind runtime_kind,
                                    std::uint64_t generation)
      : ProgramPreparationImage(native_dimension, runtime_kind, inspection_services(), generation,
                                /*execution_ready=*/false) {
    bind_image_services(inspection_services());
  }

 private:
  [[nodiscard]] static ProgramExecutionServicesRef inspection_services() noexcept {
    return {&tokens_[0], &tokens_[1], &tokens_[2], &tokens_[3], &tokens_[4],
            &tokens_[5], &tokens_[6], &tokens_[7], &tokens_[8]};
  }

  inline static std::array<std::uint8_t, 9> tokens_{};
};

template <int Dim>
[[nodiscard]] std::shared_ptr<ProgramPreparationImage> make_program_preparation_image(
    System<Dim>* system, std::uint64_t generation) {
  if (system == nullptr)
    throw std::invalid_argument("Uniform Program preparation requires one System");
  return std::make_shared<ProgramExecutionPreparationImage<Dim>>(system, generation);
}

template <int Dim>
[[nodiscard]] std::shared_ptr<ProgramPreparationImage> make_program_inspection_image(
    ProgramRuntimeKind runtime_kind, std::uint64_t generation) {
  if (runtime_kind != ProgramRuntimeKind::uniform && runtime_kind != ProgramRuntimeKind::amr)
    throw std::invalid_argument("Program inspection image has an invalid runtime kind");
  return std::make_shared<ProgramInspectionPreparationImage>(static_cast<std::uint32_t>(Dim),
                                                             runtime_kind, generation);
}

template <int Dim>
[[nodiscard]] std::shared_ptr<ProgramPreparationImage> make_program_preparation_image(
    ProgramHostDescriptor source,
    std::shared_ptr<const typename ProgramExecutionServices<Dim>::AmrPreparationTopologyView>
        topology,
    std::uint64_t generation) {
  if (source.runtime_kind != ProgramRuntimeKind::amr)
    throw std::invalid_argument("AMR Program preparation requires an AMR host descriptor");
  return std::make_shared<ProgramExecutionPreparationImage<Dim>>(source, std::move(topology),
                                                                 generation);
}

template <int Dim>
std::shared_ptr<ProgramExecutionServices<Dim>> make_program_execution_provider(
    const ProgramPreparationHostRef& preparation) {
  const auto& base =
      require_program_preparation_image(preparation, static_cast<std::uint32_t>(Dim));
  if (!base.execution_ready())
    throw std::logic_error(
        "Program execution provider is unavailable during descriptor-only inspection");
  const auto& image = static_cast<const ProgramExecutionPreparationImage<Dim>&>(base);
  return image.provider();
}

template <int Dim>
void bind_staged_uniform_program_resource_declaration(
    const std::shared_ptr<ProgramPreparationImage>& base,
    const std::vector<ProgramInstallationTables::ResourcePlan>& declaration,
    std::vector<int> program_block_map) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Uniform Program resource declaration requires its preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).bind_uniform_resource_declaration(
      declaration, std::move(program_block_map));
}

template <int Dim>
[[nodiscard]] std::vector<ProgramInstallationTables::ResourcePrototype>
take_staged_uniform_resource_prototypes(const std::shared_ptr<ProgramPreparationImage>& base) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Uniform Program resource prototypes require its preparation image");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*base)
      .take_uniform_resource_prototypes();
}

template <int Dim>
void bind_staged_amr_program_resource_declaration(
    const std::shared_ptr<ProgramPreparationImage>& base,
    const std::vector<ProgramInstallationTables::ResourcePlan>& declaration) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("AMR Program resource declaration requires its preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).bind_amr_resource_declaration(
      declaration);
}

template <int Dim>
[[nodiscard]] std::vector<ProgramInstallationTables::ResourcePrototype>
take_staged_amr_resource_prototypes(const std::shared_ptr<ProgramPreparationImage>& base) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("AMR Program resource prototypes require their preparation image");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).take_amr_resource_prototypes();
}

template <int Dim>
[[nodiscard]] std::vector<typename ProgramExecutionPreparationImage<Dim>::CacheRequest>
take_staged_amr_cache_requests(const std::shared_ptr<ProgramPreparationImage>& base) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("AMR Program cache requests require their preparation image");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).take_amr_cache_requests();
}

template <int Dim>
[[nodiscard]] std::vector<typename ProgramExecutionPreparationImage<Dim>::AmrFieldBoundaryRequest>
take_staged_amr_field_boundaries(const std::shared_ptr<ProgramPreparationImage>& base) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("AMR field-boundary stage requires its preparation image");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).take_amr_field_boundaries();
}

template <int Dim>
void seed_staged_uniform_field_boundaries(const std::shared_ptr<ProgramPreparationImage>& base,
                                          ArtifactFieldBoundaryAuthorityRegistry<Dim> authorities) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Uniform field-boundary stage requires its preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).seed_uniform_field_boundaries(
      std::move(authorities));
}

template <int Dim>
[[nodiscard]] ArtifactFieldBoundaryAuthorityRegistry<Dim> take_staged_uniform_field_boundaries(
    const std::shared_ptr<ProgramPreparationImage>& base) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Uniform field-boundary stage requires its preparation image");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).take_uniform_field_boundaries();
}

template <int Dim>
[[nodiscard]] std::vector<typename ProgramExecutionPreparationImage<Dim>::CacheRequest>
take_staged_uniform_cache_requests(const std::shared_ptr<ProgramPreparationImage>& base) {
  if (!base || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Uniform cache stage requires its preparation image");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*base).take_uniform_cache_requests();
}

template <int Dim>
void ProgramExecutionServices<Dim>::prepare_rhs_scratch(std::size_t slot, int subslot,
                                                        int program_block) const {
  if (binding_ != Binding::preparation || preparation_image_ == nullptr)
    throw std::logic_error("Program RHS scratch preparation requires one preparation image");
  if (is_amr()) {
    static_cast<ProgramExecutionPreparationImage<Dim>&>(
        *const_cast<ProgramPreparationImage*>(preparation_image_))
        .prepare_amr_scratch(0, slot, subslot, program_block,
                             amr_backend::state(program_block).ncomp(),
                             amr_backend::state(program_block).ghosts()[0]);
    return;
  }
  static_cast<ProgramExecutionPreparationImage<Dim>&>(
      *const_cast<ProgramPreparationImage*>(preparation_image_))
      .prepare_uniform_scratch(0, slot, subslot, program_block, -1, -1);
}

template <int Dim>
void ProgramExecutionServices<Dim>::prepare_state_scratch(std::size_t slot, int subslot,
                                                          int program_block) const {
  if (binding_ != Binding::preparation || preparation_image_ == nullptr)
    throw std::logic_error("Program state scratch preparation requires one preparation image");
  if (is_amr()) {
    static_cast<ProgramExecutionPreparationImage<Dim>&>(
        *const_cast<ProgramPreparationImage*>(preparation_image_))
        .prepare_amr_scratch(1, slot, subslot, program_block,
                             amr_backend::state(program_block).ncomp(),
                             amr_backend::state(program_block).ghosts()[0]);
    return;
  }
  static_cast<ProgramExecutionPreparationImage<Dim>&>(
      *const_cast<ProgramPreparationImage*>(preparation_image_))
      .prepare_uniform_scratch(1, slot, subslot, program_block, -1, -1);
}

template <int Dim>
void ProgramExecutionServices<Dim>::prepare_scalar_scratch(std::size_t slot, int subslot,
                                                           int program_block, int ncomp,
                                                           int ghost_depth) const {
  if (binding_ != Binding::preparation || preparation_image_ == nullptr)
    throw std::logic_error("Program scalar scratch preparation requires one preparation image");
  if (is_amr()) {
    static_cast<ProgramExecutionPreparationImage<Dim>&>(
        *const_cast<ProgramPreparationImage*>(preparation_image_))
        .prepare_amr_scratch(2, slot, subslot, program_block, ncomp, ghost_depth);
    return;
  }
  static_cast<ProgramExecutionPreparationImage<Dim>&>(
      *const_cast<ProgramPreparationImage*>(preparation_image_))
      .prepare_uniform_scratch(2, slot, subslot, program_block, ncomp, ghost_depth);
}

template <int Dim>
void ProgramExecutionServices<Dim>::prepare_cache_slot(std::size_t slot, int program_block) const {
  if (binding_ != Binding::preparation || preparation_image_ == nullptr)
    throw std::logic_error("Program cache preparation requires one preparation image");
  if (is_amr()) {
    static_cast<ProgramExecutionPreparationImage<Dim>&>(
        *const_cast<ProgramPreparationImage*>(preparation_image_))
        .prepare_amr_cache(slot, program_block);
    return;
  }
  static_cast<ProgramExecutionPreparationImage<Dim>&>(
      *const_cast<ProgramPreparationImage*>(preparation_image_))
      .prepare_uniform_cache(slot, program_block);
}

template <int Dim>
void ProgramExecutionServices<Dim>::prepare_generated_field_route(
    std::uint32_t slot, std::string_view field, std::initializer_list<int> program_blocks) const {
  if (binding_ != Binding::preparation || preparation_image_ == nullptr)
    throw std::logic_error("Program field-route preparation requires one preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(
      *const_cast<ProgramPreparationImage*>(preparation_image_))
      .prepare_generated_field_route(slot, field, program_blocks);
}

template <int Dim>
void stage_uniform_field_boundary_kernel(const ProgramPreparationImage* base,
                                         const std::string& provider_slot,
                                         const CompiledFieldBoundaryKernel<Dim>& kernel) {
  if (base == nullptr || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Program field boundary kernel requires the Uniform preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*const_cast<ProgramPreparationImage*>(base))
      .stage_uniform_field_boundary_kernel(provider_slot, kernel);
}

template <int Dim>
void stage_uniform_field_logical_timepoint(const ProgramPreparationImage* base,
                                           const std::string& provider_slot,
                                           const FieldLogicalTimePoint& point) {
  if (base == nullptr || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Program field timepoint requires the Uniform preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*const_cast<ProgramPreparationImage*>(base))
      .stage_uniform_field_logical_timepoint(provider_slot, point);
}

template <int Dim>
void stage_uniform_field_boundary_parameters(const ProgramPreparationImage* base,
                                             const std::string& provider_slot,
                                             const std::vector<double>& parameters) {
  if (base == nullptr || base->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::logic_error("Program field parameters require the Uniform preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*const_cast<ProgramPreparationImage*>(base))
      .stage_uniform_field_boundary_parameters(provider_slot, parameters);
}

template <int Dim>
void stage_amr_field_boundary_kernel(const ProgramPreparationImage* base,
                                     const std::string& provider_slot,
                                     const CompiledFieldBoundaryKernel<Dim>& kernel) {
  if (base == nullptr || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("Program field boundary kernel requires the AMR preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*const_cast<ProgramPreparationImage*>(base))
      .stage_amr_field_boundary_kernel(provider_slot, kernel);
}

template <int Dim>
void stage_amr_field_logical_timepoint(const ProgramPreparationImage* base,
                                       const std::string& provider_slot,
                                       const FieldLogicalTimePoint& point) {
  if (base == nullptr || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("Program field timepoint requires the AMR preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*const_cast<ProgramPreparationImage*>(base))
      .stage_amr_field_logical_timepoint(provider_slot, point);
}

template <int Dim>
void stage_amr_field_boundary_parameters(const ProgramPreparationImage* base,
                                         const std::string& provider_slot,
                                         const std::vector<double>& parameters) {
  if (base == nullptr || base->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::logic_error("Program field parameters require the AMR preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*const_cast<ProgramPreparationImage*>(base))
      .stage_amr_field_boundary_parameters(provider_slot, parameters);
}

template <int Dim>
void ProgramExecutionServices<Dim>::stage_auxiliary_consumer_plan(
    runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) const {
  if (preparation_image_ == nullptr)
    throw std::logic_error("Program auxiliary consumer plan requires a preparation image");
  const auto& image =
      static_cast<const ProgramExecutionPreparationImage<Dim>&>(*preparation_image_);
  const_cast<ProgramExecutionPreparationImage<Dim>&>(image).stage_auxiliary_consumer_plan(plan);
}

template <int Dim>
void ProgramExecutionServices<Dim>::register_history(
    const std::string& name, int lag, int ncomp, int program_owner,
    const std::string& state_identity, const std::string& space_identity,
    const std::string& clock_identity, const std::string& interpolation_identity) const {
  if (preparation_image_ != nullptr && binding_ == Binding::preparation) {
    if (!is_amr() && uniform_backend::n_blocks() == 0)
      throw std::logic_error("state-free Uniform Program cannot stage block-owned histories");
    auto& image = const_cast<ProgramExecutionPreparationImage<Dim>&>(
        static_cast<const ProgramExecutionPreparationImage<Dim>&>(*preparation_image_));
    image.stage_history({name, lag, ncomp, program_owner, state_identity, space_identity,
                         clock_identity, interpolation_identity});
    return;
  }
  if (is_amr())
    amr_backend::register_history(name, lag, ncomp, program_owner, state_identity, space_identity,
                                  clock_identity, interpolation_identity);
  else
    uniform_backend::register_history(name, lag, ncomp, program_owner, state_identity,
                                      space_identity, clock_identity, interpolation_identity);
}

template <int Dim>
[[nodiscard]] std::vector<runtime::system::AuxiliaryConsumerProviderPlan<Dim>>
take_staged_auxiliary_consumer_plans(const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim))
    throw std::invalid_argument("Program preparation image has the wrong native dimension");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*image)
      .take_auxiliary_consumer_plans();
}

template <int Dim>
[[nodiscard]] std::vector<typename ProgramExecutionPreparationImage<Dim>::HistoryRequest>
take_staged_histories(const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim))
    throw std::invalid_argument("Program preparation image has the wrong native dimension");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*image).take_histories();
}

template <int Dim>
void materialize_staged_amr_histories(const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim) ||
      image->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::invalid_argument("AMR staged histories require their preparation image");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*image).materialize_amr_staged_histories();
}

/// Finalize Uniform provider declarations while the image remains private to the installer. AMR
/// retains its dedicated level-qualified preparation graph and does not use this route.
template <int Dim>
void seal_staged_uniform_program_execution_services(
    const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim) ||
      image->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::invalid_argument("Uniform Program preparation image has the wrong authority");
  const auto& provider = static_cast<ProgramExecutionPreparationImage<Dim>&>(*image).provider();
  if (provider->has_staged_uniform_clock())
    provider->seal_uniform_preparation();
  else
    provider->seal_transaction_authorities();
}

template <int Dim>
void seal_staged_program_transaction_authorities(
    const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim))
    throw std::invalid_argument("Program transaction-authority image has the wrong dimension");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*image).seal_transaction_authorities();
}

template <int Dim>
[[nodiscard]] std::string staged_program_transaction_authority_contract(
    const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim))
    throw std::invalid_argument("Program transaction-authority image has the wrong dimension");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*image)
      .transaction_authority_contract();
}

template <int Dim>
[[nodiscard]] ProgramRuntimeState<Dim> take_staged_uniform_transaction_authority_state(
    const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim) ||
      image->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::invalid_argument("Uniform transaction-authority image has the wrong authority");
  return static_cast<ProgramExecutionPreparationImage<Dim>&>(*image)
      .take_uniform_transaction_authority_state();
}

template <int Dim>
void activate_staged_uniform_program_execution_services(
    const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim) ||
      image->runtime_kind() != ProgramRuntimeKind::uniform)
    throw std::invalid_argument("Uniform Program preparation image has the wrong authority");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*image).activate_uniform_after_collective();
}

template <int Dim>
void activate_staged_amr_program_execution_services(
    const std::shared_ptr<ProgramPreparationImage>& image, ::pops::AmrSystem<Dim>* system) {
  if (!image || image->native_dimension() != static_cast<std::uint32_t>(Dim) ||
      image->runtime_kind() != ProgramRuntimeKind::amr)
    throw std::invalid_argument("AMR Program preparation image has the wrong authority");
  static_cast<ProgramExecutionPreparationImage<Dim>&>(*image).activate_amr_after_collective(system);
}

}  // namespace pops::runtime::program
