/// @file
/// @brief Exact compile-time-ranked execution boundary for generated Uniform Programs.

#pragma once
#include <pops/runtime/program/field_transform_support.hpp>
#include <pops/runtime/program/field_read_transform.hpp>

#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/program/distributed_array.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/swept_interval.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/consensus_words.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>
#include <pops/numerics/spatial/nd/face_frequency.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>
#include <pops/runtime/program/prepared_integral_capture.hpp>
#include <pops/runtime/program/program_value_authority.hpp>
#include <pops/runtime/program/field_exchange_expression.hpp>
#include <pops/runtime/program/spatial_direct_interaction.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/spatial_interaction_history_source.hpp>
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
#include <set>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {

namespace program_context_detail {
// Launch functors are namespace-visible: nvcc emits host launch stubs outside
// ProgramContext and cannot name function-local or private nested types there.
struct FiniteReal {
  POPS_HD bool operator()(Real value) const {
    return value == value && value <= std::numeric_limits<Real>::max() &&
           value >= -std::numeric_limits<Real>::max();
  }
};

template <int Dim>
struct LaplacianKernel {
  FieldView<Real, Dim> result{};
  FieldView<const Real, Dim> value{};
  Geometry<Dim> geometry{};
  int components = 0;

  POPS_HD void operator()(const Index<Dim>& cell) const {
    for (int component = 0; component < components; ++component) {
      Real image = Real(0);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell;
        Index<Dim> upper = cell;
        --lower[axis];
        ++upper[axis];
        const Real spacing = geometry.spacing(axis);
        image +=
            (value(upper, component) - Real(2) * value(cell, component) + value(lower, component)) /
            (spacing * spacing);
      }
      result(cell, component) = image;
    }
  }
};

template <int Dim>
struct GradientKernel {
  FieldView<Real, Dim> result{};
  FieldView<const Real, Dim> value{};
  Geometry<Dim> geometry{};

  POPS_HD void operator()(const Index<Dim>& cell) const {
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = cell;
      Index<Dim> upper = cell;
      --lower[axis];
      ++upper[axis];
      result(cell, axis) = (value(upper, 0) - value(lower, 0)) / (Real(2) * geometry.spacing(axis));
    }
  }
};

template <int Dim, class Operator>
struct ScalarOperatorKernel {
  FieldView<Real, Dim> result{};
  Operator op;

  POPS_HD void operator()(const Index<Dim>& cell) const { result(cell, 0) = op.image(cell); }
};

template <int Dim>
struct CopyActiveValidKernel {
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
}  // namespace program_context_detail

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

#include <pops/runtime/program/program_context_value_authority.inc>

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

  /// A scalar potential keeps the topology and physical laws of its exact installed Field slot.
  /// Transport coefficients and fluxes deliberately retain their separate mesh-boundary session.
  class PreparedFieldBoundarySession {
   public:
    const ExecutionLane& lane() const noexcept { return *lane_; }
    const Geometry<Dim>& geometry() const noexcept { return halo_->geometry(); }
    const PhysicalBoundaryConditions<Dim>& conditions() const noexcept {
      return physical_.conditions();
    }
    void fill(field_type& potential) const {
      halo_->fill_halo(potential);
      fill_physical_boundary(potential, physical_);
    }

   private:
    friend class ProgramContext;
    PreparedFieldBoundarySession(const runtime_type* system, int program_block, int runtime_block,
                                 std::string provider_slot, const ExecutionLane& lane,
                                 std::shared_ptr<scalar_boundary_session_type> halo,
                                 PreparedPhysicalBoundary<Dim> physical)
        : system_(system),
          program_block_(program_block),
          runtime_block_(runtime_block),
          provider_slot_(std::move(provider_slot)),
          lane_(&lane),
          halo_(std::move(halo)),
          physical_(std::move(physical)) {}
    const runtime_type* system_;
    int program_block_;
    int runtime_block_;
    std::string provider_slot_;
    const ExecutionLane* lane_;
    std::shared_ptr<scalar_boundary_session_type> halo_;
    PreparedPhysicalBoundary<Dim> physical_;
  };
  using field_boundary_session_type = PreparedFieldBoundarySession;

  struct FieldStageOverride {
    int program_block = -1;
    const field_type* state = nullptr;
  };

  struct RhsGroupRequest {
    RhsGroupRequest(int block_value, field_type* state_value, field_type* rhs_value,
                    int rate_id_value, int flux_only_value,
                    std::vector<nd::FaceField<Dim>>* retained_faces_value = nullptr)
        : block(block_value),
          state(state_value),
          rhs(rhs_value),
          rate_id(rate_id_value),
          flux_only(flux_only_value),
          retained_faces(retained_faces_value) {}

    int block = -1;
    field_type* state = nullptr;
    field_type* rhs = nullptr;
    int rate_id = -1;
    int flux_only = 0;
    std::vector<nd::FaceField<Dim>>* retained_faces = nullptr;
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

    const ProgramContext* owner_ = nullptr;
    double prior_dt_ = 0.0;
    ::pops::amr::Rational prior_stage_{0, 1};
    ::pops::amr::Rational prior_phase_begin_{0, 1};
    ::pops::amr::Rational prior_phase_span_{1, 1};
    double prior_physical_time_offset_ = 0.0;
  };

  explicit ProgramContext(runtime_type* system) : system_(require_system_(system)) {}

  void install(std::function<void(double)> step) const {
    auto resources = program_resource_lifetime_();
    std::weak_ptr<FieldExchangeExpressions> expressions = field_exchanges_;
    auto lifetime = [resources = std::move(resources), expressions](PreparedResourceAction action) {
      if (action != PreparedResourceAction::drain)
        if (auto live = expressions.lock())
          live->clear();
      resources(action);
    };
    std::weak_ptr<AlgebraLifetime> algebra = algebra_lifetime_;
    std::weak_ptr<ContinuationFieldInstances> fields = continuation_field_instances_;
    auto restart = [algebra, fields](
                       const FiniteDomainRealmRef& keep,
                       const std::shared_ptr<distributed::Registry>& restored) noexcept {
      if (auto live = fields.lock())
        live->values.clear();
      if (auto live = algebra.lock())
        live->restart_committed(keep, restored);
    };
    typename ContinuationCodec<Dim>::Prototypes checkpoint_prototypes =
        [system = system_, providers = continuation_field_providers_](
            const ContinuationType& type,
            const ContinuationFieldSupport::Ref& support) -> const field_type& {
      return continuation_field_prototype_(system, providers, type, support);
    };
    system_->install_program_step(std::move(step));
    runtime_state().install_resource_lifetime(std::move(lifetime));
    runtime_state().install_restart_lifetime(std::move(restart));
    runtime_state().install_continuation_checkpoint_prototypes(std::move(checkpoint_prototypes));
  }

  void reached_duration(double duration) const {
    runtime_state().set_computed_duration(duration);
    // Reserved output sinks are emitted only by the authenticated frontier IR node. The ordinary
    // accepted-state transaction snapshots these diagnostics alongside the numerical candidate.
    record_scalar("pops.frontier.duration", static_cast<Real>(duration));
  }

  void suspend_map(std::string identity, bool target, field_type& field,
                   std::function<void()> continuation) const {
    runtime_state().suspend_program_map(std::move(identity), target, {&field},
                                        std::move(continuation));
  }

  void begin_step(double dt) const {
    (void)prepared_execution_lane();
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramContext step requires a finite positive dt");
    program_values_.revoke_();
    field_exchanges_->clear();
    (void)prepared_resources_.begin_attempt(prepared_execution_lane());
    if (runtime_state().continuation_.configured()) {
      continuation_local_phase([&] { algebra_lifetime_->continuation.begin(resource_attempt()); },
                               "continuation attempt issuance");
      continuation_control_consensus();
    }
    drain_program_value_writes_();
    current_dt_ = dt;
    stage_time_ = ::pops::amr::Rational(0, 1);
    logical_phase_begin_ = ::pops::amr::Rational(0, 1);
    logical_phase_span_ = ::pops::amr::Rational(1, 1);
    logical_physical_time_offset_ = 0.0;
    active_operator_snapshot_.reset();
    auxiliary_evaluation_sequence_ = 0;
  }

  void configure_primary_clock(const std::string& clock) const {
    const auto& manifest = runtime_state().checkpoint_metadata_.uniform_auxiliary_clocks;
    if (!manifest.owner_identity.empty()) {
      manifest.require_owned(clock);
      if (clock != manifest.primary_clock_identity)
        throw std::invalid_argument(
            "Program context primary clock differs from installed ownership");
    }
    clock_schedule_.configure_primary_clock(clock);
    primary_clock_ = clock;
  }

  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    const auto& manifest = runtime_state().checkpoint_metadata_.uniform_auxiliary_clocks;
    if (!manifest.owner_identity.empty()) {
      manifest.require_owned(parent);
      manifest.require_owned(child);
    }
    clock_schedule_.declare_relation(parent, child, count);
  }

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (denominator <= 0 || numerator < 0 || numerator > denominator)
      throw std::invalid_argument("ProgramContext stage time is outside [0, 1]");
    stage_time_ = ::pops::amr::Rational(numerator, denominator);
    active_operator_snapshot_.reset();
  }

  class StageEvaluationScope {
   public:
    StageEvaluationScope(const ProgramContext& owner, std::int64_t numerator,
                         std::int64_t denominator)
        : owner_(&owner), prior_(owner.stage_time_) {
      owner.set_stage_time(numerator, denominator);
    }
    StageEvaluationScope(const StageEvaluationScope&) = delete;
    StageEvaluationScope& operator=(const StageEvaluationScope&) = delete;
    StageEvaluationScope(StageEvaluationScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), prior_(other.prior_) {}
    ~StageEvaluationScope() {
      if (owner_) {
        owner_->stage_time_ = prior_;
        owner_->active_operator_snapshot_.reset();
      }
    }

   private:
    const ProgramContext* owner_;
    ::pops::amr::Rational prior_;
  };
  [[nodiscard]] StageEvaluationScope stage_evaluation_scope(std::int64_t numerator,
                                                            std::int64_t denominator) const {
    return StageEvaluationScope(*this, numerator, denominator);
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("ProgramContext boundary evaluation has no prepared clock and dt");
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

  int n_blocks() const { return system_->n_blocks(); }

  /// Borrow the runtime-owned lane authenticated during Uniform boundary preparation. Generated
  /// implicit reports use this same lane for every reduction, diagnostic selection, and outcome.
  // Live domain issuance is outside accepted snapshots: rollback never reuses an ordinal.
  [[nodiscard]] distributed::Registry& distributed_registry() const {
    const auto& lane = prepared_execution_lane();
    const long present = algebra_lifetime_->distributed ? 1L : 0L;
    if (all_reduce_min(present, lane) != all_reduce_max(present, lane))
      throw std::logic_error("distributed registry lifecycle differs across execution ranks");
    if (!algebra_lifetime_->distributed)
      algebra_lifetime_->distributed = std::make_shared<distributed::Registry>(lane);
    return *algebra_lifetime_->distributed;
  }
  void revoke_distributed_registry_on_restart() const noexcept {
    algebra_lifetime_->distributed.reset();
  }
  [[nodiscard]] distributed::Registry* distributed_registry_if_present() const noexcept {
    return algebra_lifetime_->distributed.get();
  }

  [[nodiscard]] const ExecutionLane& prepared_execution_lane() const {
    return system_->prepared_boundary_execution_lane();
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

  void consume_pointwise_evaluation_status(int program_block, int evaluation_id, Real status,
                                           const char* operation_identity,
                                           std::uint32_t reason_code = 0) const {
    consume_native_evaluation_status(prepared_execution_lane(), program_block, evaluation_id,
                                     static_cast<double>(status), operation_identity, reason_code);
  }

  using ProgramFieldComponent = typename runtime_type::ProgramFieldComponent;
  void publish_field_components(int evaluation_id, const std::string& publication_identity,
                                std::initializer_list<ProgramFieldComponent> components) const {
    const auto& lane = prepared_execution_lane();
    runtime::multiblock::BoundaryEvaluationPoint point;
    std::vector<ProgramFieldComponent> prepared;
    std::exception_ptr error;
    try {
      point = boundary_evaluation_point(evaluation_id);
      prepared.assign(components.begin(), components.end());
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("Program field publication preparation failed collectively");
    }
    system_->publish_program_field_components(point, publication_identity, prepared);
  }

  void stage_exchange(ExchangeRecord record) const {
    stage_exchange_batch([&](auto&& stage) { stage(std::move(record)); });
  }

  void declare_integral_state(const std::string& identity, Real initial) const {
    system_->declare_program_integral(identity, initial);
  }
  Real integral_state(const std::string& identity) const {
    return system_->program_integral(identity);
  }
  PreparedIntegralCapture capture_integral_candidate(const std::string& identity,
                                                     const std::string& units) const {
    return PreparedIntegralCapture::prepare_(
        this, [&] { return resource_attempt(); }, [&] { return boundary_evaluation_point(0); },
        identity, units,
        [&] {
          const auto& ledger = runtime_state().accepted_exchanges_;
          return PreparedIntegralCapture::ReadImage{ledger.integral(identity),
                                                    ledger.checkpoint(true)};
        },
        prepared_execution_lane());
  }
  Real integral_candidate_value(const PreparedIntegralCapture& capture, const std::string& identity,
                                const std::string& units) const {
    return static_cast<Real>(capture.consume_(
        this, [&] { return resource_attempt(); }, [&] { return boundary_evaluation_point(0); },
        identity, units,
        [&] {
          const auto& ledger = runtime_state().accepted_exchanges_;
          return PreparedIntegralCapture::ReadImage{ledger.integral(identity),
                                                    ledger.checkpoint(true)};
        },
        prepared_execution_lane()));
  }
  Real consume_external_trace(const std::string& integral_identity,
                              const AcceptedExchangeLedger::TraceSelection& selection,
                              Real scale) const {
    return system_->consume_program_external_trace(integral_identity, selection, scale);
  }
  bool is_external_trace_face(int axis, int side, const Index<Dim>& cell) const {
    if (axis < 0 || axis >= Dim || (side != 0 && side != 1))
      throw std::invalid_argument("external trace face has invalid axis or side");
    const auto boundary = side == 0 ? BoundarySide::lower : BoundarySide::upper;
    return !scalar_boundary_topology_().is_periodic(Face<Dim>{axis, boundary}) &&
           cell[axis] == (side == 0 ? geometry().domain().lo[axis] : geometry().domain().hi[axis]);
  }

  /// Prepare collectively before a rank-local exchange producer. The returned
  /// predicate owns its domain/topology and never calls back into the facade.
  auto prepare_external_trace_face_predicate() const {
    const auto domain = geometry().domain();
    const auto topology = scalar_boundary_topology_();
    return [domain, topology](int axis, int side, const Index<Dim>& cell) {
      if (axis < 0 || axis >= Dim || (side != 0 && side != 1))
        throw std::invalid_argument("external trace face has invalid axis or side");
      const auto boundary = side == 0 ? BoundarySide::lower : BoundarySide::upper;
      return !topology.is_periodic(Face<Dim>{axis, boundary}) &&
             cell[axis] == (side == 0 ? domain.lo[axis] : domain.hi[axis]);
    };
  }

  template <class Producer>
  void stage_exchange_batch(Producer&& producer) const {
    auto records = prepare_exchange_batch(
        std::forward<Producer>(producer),
        [&](ExchangeRecord& record) { record.qualify_runtime_point(boundary_evaluation_point(0)); },
        prepared_execution_lane());
    system_->stage_program_exchanges(records);
  }

#include <pops/runtime/program/program_context_field_exchanges.inc>

  field_type& state(int program_block) const {
    return system_->block_state(sys_block(program_block));
  }

#include <pops/runtime/program/program_context_moving_interval.inc>
#include <pops/runtime/program/program_context_moving_projection.inc>
  /// One collective prerequisite publication before a generated consumer traverses local Fabs.
  /// The supplied SSA state is authenticated without substituting the accepted block state.
  /// Native DerivedAux launchers consume their declared auxiliary dependencies; state-dependent
  /// formulas continue to read this SSA value in the generated kernel itself.
  void prepare_provider_values(std::string_view consumer_qid, int program_block,
                               const field_type& stage_state, int evaluation_id) const {
    if (prepare_provider_values_for_solve(consumer_qid, program_block, stage_state,
                                          evaluation_id) ==
        runtime::system::AuxiliaryPublicationStatus::nonfinite_candidate)
      throw std::runtime_error(
          "System auxiliary publication rejected: candidate valid/ghost image contains non-finite "
          "values");
  }
  [[nodiscard]] runtime::system::AuxiliaryPublicationStatus prepare_provider_values_for_solve(
      std::string_view consumer_qid, int program_block, const field_type& stage_state,
      int evaluation_id) const {
    if (auxiliary_evaluation_sequence_ == std::numeric_limits<int>::max())
      throw std::overflow_error("Program auxiliary evaluation sequence exceeds its exact range");
    return system_->prepare_program_auxiliary_consumer_for_solve(
        boundary_evaluation_point(evaluation_id), std::string(consumer_qid),
        sys_block(program_block), stage_state, auxiliary_evaluation_sequence_++);
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

  double path_rhs_courant() const { return numerical_face_courant(); }

  /// The authored numerical face bound is checked at its explicit consumer. A
  /// FixedDt invocation has no CFL proposal, so it uses the unit incident-face
  /// budget; step_cfl supplies its caller-authored Courant instead.
  double numerical_face_courant() const {
    const double active = system_->active_program_step_courant();
    return std::isfinite(active) && active > 0.0 ? active : 1.0;
  }

  void path_rhs_into(int program_block, field_type& input, field_type& output, int rate_id,
                     std::string_view temporal_family, double courant,
                     Real* evaluated_frequency = nullptr) const {
    const ExecutionLane& lane = prepared_execution_lane();
    int runtime_block = -1;
    runtime::multiblock::BoundaryEvaluationPoint point;
    std::string contract;
    std::exception_ptr error;
    try {
      runtime_block = sys_block(program_block);
      require_rate_identity_(rate_id);
      point = boundary_evaluation_point(rate_id);
      const double authored_courant = numerical_face_courant();
      const auto destination =
          scratch_.find(ScratchKey{ScratchKind::Rhs, runtime_block, rate_id, 0});
      if (rate_id < 0 || temporal_family.empty() || &input == &output ||
          input.shares_storage_with(output) || destination == scratch_.end() ||
          &destination->second != &output || !std::isfinite(courant) || !(courant > 0.0) ||
          courant != authored_courant)
        throw std::invalid_argument(
            "Uniform path RHS lost its exact staged input, output or Courant");
      require_same_field_contract_(input, output, "Uniform path RHS output");
      ExactContractBuilder identity;
      identity.text("pops.uniform-path-rhs-stage.v2")
          .scalar(runtime_block)
          .scalar(rate_id)
          .text(temporal_family)
          .scalar(point.tick)
          .scalar(point.stage)
          .scalar(point.stage_fraction.numerator)
          .scalar(point.stage_fraction.denominator)
          .scalar(point.dt)
          .scalar(point.physical_time)
          .scalar(courant)
          .scalar(evaluated_frequency != nullptr);
      contract = std::move(identity).release();
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("Uniform path RHS staging failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"uniform-path-rhs-stage", contract}}, lane))
      throw std::invalid_argument("Uniform path RHS stage differs between ranks");
    auto boundary = prepare_block_boundary_session(program_block, input, point, lane);
    count_kernel_();
    system_->block_path_rhs_into_at(point, runtime_block, input, output, static_cast<Real>(courant),
                                    boundary->system(), boundary->runtime_block(),
                                    boundary->point(), boundary->lane(), boundary->transport(),
                                    evaluated_frequency);
  }

  template <class Resource, class Matches, class... Args>
  Resource& prepared_resource(std::int64_t node, int block, Matches&& matches,
                              Args&&... args) const {
    return prepared_resources_.template acquire<Resource>(node, block, 0, prepared_execution_lane(),
                                                          std::forward<Matches>(matches),
                                                          std::forward<Args>(args)...);
  }

  template <class Resource, class Matches, class... Args>
  PreparedResourceLease<Resource> prepared_resource_lease(std::int64_t node, int block,
                                                          Matches&& matches, Args&&... args) const {
    return prepared_resources_.template acquire_lease<Resource>(
        node, block, 0, prepared_execution_lane(), std::forward<Matches>(matches),
        std::forward<Args>(args)...);
  }

  PreparedResourceAttempt resource_attempt() const { return prepared_resources_.current_attempt(); }

  template <class Executor, class Functor, class... Resources>
  PreparedResourceTask submit_prepared_for(
      const PreparedResourceLease<Executor>& executor, std::size_t lane, const char* label,
      std::int64_t count, Functor functor,
      const PreparedResourceLease<Resources>&... resources) const {
    return executor.get().submit_for(prepared_resources_, resource_attempt(), executor, lane, label,
                                     count, std::move(functor), resources...);
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

  /// Allocate scalar storage on one exact Program block, never the first same-shaped block.
  field_type alloc_scalar_field_for_block(int program_block, int ncomp = 1,
                                          int ghost_depth = 1) const {
    const ExecutionLane& lane = prepared_execution_lane();
    require_prepared_lane_(lane, "Program scalar storage owner");
    std::optional<field_type> candidate;
    std::exception_ptr local_error;
    try {
      candidate.emplace(scalar_field_like_(state(program_block), ncomp, ghost_depth));
    } catch (...) {
      local_error = std::current_exception();
    }
    // Owner lookup, host/device allocation and zero-initialization are rank-local. Converge their
    // failures before peers may enter the next scalar allocation or provider/layout collective.
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program scalar storage allocation failed on another rank");
    }
    return std::move(*candidate);
  }

  /// Retain both the holder and its field inside the same collective allocation boundary.
  std::shared_ptr<field_type> alloc_shared_scalar_field_for_block(int program_block, int ncomp = 1,
                                                                  int ghost_depth = 1) const {
    const ExecutionLane& lane = prepared_execution_lane();
    require_prepared_lane_(lane, "Program shared scalar storage owner");
    std::shared_ptr<field_type> candidate;
    std::exception_ptr local_error;
    try {
      candidate = std::make_shared<field_type>(
          scalar_field_like_(state(program_block), ncomp, ghost_depth));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program shared scalar storage allocation failed on another rank");
    }
    return candidate;
  }

  /// Finish all rank-local session snapshots before a provider may enter boundary collectives.
  /// The operation must contain local construction/copies only, never an MPI collective.
  template <class Operation>
  void prepare_operator_session_storage(Operation&& operation) const {
    const ExecutionLane& lane = prepared_execution_lane();
    require_prepared_lane_(lane, "Program operator session storage");
    std::exception_ptr local_error;
    try {
      std::forward<Operation>(operation)();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_exception(local_error, lane,
                                   "Program operator session storage allocation failed");
  }

  /// Authenticate source/output against the installed owner before a scalar sampling kernel.
  void require_scalar_sampling_layout(int program_block, const field_type& source,
                                      const field_type& output, int component = 0) const {
    const ExecutionLane& lane = prepared_execution_lane();
    std::exception_ptr local_error;
    try {
      const field_type& owner = state(program_block);
      if (output.ncomp() != 1 || component < 0 || component >= source.ncomp())
        throw std::invalid_argument("Program scalar sample component contract is invalid");
      for (const field_type* candidate : {&source, &output})
        if (candidate->layout() != owner.layout() ||
            candidate->distribution() != owner.distribution() ||
            candidate->local_rank() != owner.local_rank() ||
            candidate->local_size() != owner.local_size())
          throw std::invalid_argument("Program scalar sample layout/rank differs from its owner");
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program scalar sample layout failed collectively");
    }
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
    const auto point = boundary_evaluation_point(rate_id);
    const int runtime_block = sys_block(program_block);
    if (system_->requires_block_boundary_session(runtime_block)) {
      const ExecutionLane& lane = system_->prepared_boundary_execution_lane();
      auto boundary = prepare_block_boundary_session(program_block, state_value, point, lane);
      system_->block_neg_div_flux_into_at_prepared(
          point, runtime_block, state_value, rhs, boundary->system(), boundary->runtime_block(),
          boundary->point(), boundary->lane(), boundary->transport());
      return;
    }
    system_->block_neg_div_flux_into_at(point, runtime_block, state_value, rhs);
  }

  /// Retain the exact native faces used by one flux-only residual. These are provisional
  /// numerical observations; only the accepted Program quadrature may stage exchanges.
  void neg_div_flux_default_with_faces_into(int program_block, field_type& state_value,
                                            field_type& rhs, int rate_id,
                                            std::vector<nd::FaceField<Dim>>& faces,
                                            Real* transport_frequency = nullptr) const {
    require_rate_identity_(rate_id);
    count_kernel_();
    const auto point = boundary_evaluation_point(rate_id);
    const auto& lane = prepared_execution_lane();
    if (all_reduce_min(transport_frequency ? 1L : 0L, lane) !=
        all_reduce_max(transport_frequency ? 1L : 0L, lane))
      throw std::invalid_argument("transport face-frequency request differs across ranks");
    auto boundary = prepare_block_boundary_session(program_block, state_value, point, lane);
    system_->block_neg_div_flux_into_at_prepared(
        point, sys_block(program_block), state_value, rhs, boundary->system(),
        boundary->runtime_block(), boundary->point(), boundary->lane(), boundary->transport());
    // FaceField owns Fab values with deep-copy semantics: subsequent residual evaluations
    // cannot overwrite a prior stage's accepted quadrature data.
    Real local_frequency = Real(0);
    std::exception_ptr error;
    try {
      boundary->transport().with_boundary_scratch(state_value, [&](auto& scratch) {
        if (transport_frequency)
          local_frequency = nd::maximum_incident_face_frequency(
              state_value, scratch.cartesian_operator.face_speeds(), geometry(),
              scratch.generated_face_frequency);
        faces = scratch.generated_faces;
      });
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("transport face observation failed collectively");
    }
    if (transport_frequency) {
      const Real frequency = static_cast<Real>(all_reduce_max(local_frequency, lane));
      if (!std::isfinite(frequency) || frequency < Real(0))
        throw std::runtime_error("transport face frequency is nonfinite or negative");
      *transport_frequency = frequency;
    }
  }

  /// A partition guard may be emitted after grouped RHS evaluation, where no
  /// individual face carrier is retained. Re-evaluate the same prepared default
  /// numerical flux at its exact stage, but publish neither RHS nor exchanges.
  Real evaluated_transport_frequency(int program_block, field_type& state_value,
                                     int rate_id) const {
    const auto& lane = prepared_execution_lane();
    std::optional<field_type> provisional;
    std::exception_ptr error;
    try {
      provisional.emplace(state_value.layout(), state_value.distribution(),
                          state_value.local_rank(), state_value.ncomp(), state_value.ghosts());
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("transport frequency scratch allocation failed collectively");
    }
    std::vector<nd::FaceField<Dim>> provisional_faces;
    Real frequency = Real(0);
    neg_div_flux_default_with_faces_into(program_block, state_value, *provisional, rate_id,
                                         provisional_faces, &frequency);
    return frequency;
  }

  void source_default_into(int program_block, field_type& state_value, field_type& rhs) const {
    count_kernel_();
    system_->block_source_into(sys_block(program_block), state_value, rhs);
  }

  [[nodiscard]] NewtonOptions block_newton_options(int program_block) const {
    return system_->block_newton_options(sys_block(program_block));
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
    require_rate_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("ProgramContext RHS group cannot be empty");
    std::vector<int> blocks;
    std::vector<field_type*> states;
    std::vector<field_type*> residuals;
    std::vector<int> flux_only;
    std::vector<std::vector<nd::FaceField<Dim>>*> retained_faces;
    retained_faces.reserve(requests.size());
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
      retained_faces.push_back(request.retained_faces);
    }
    count_kernel_(static_cast<std::int64_t>(requests.size()));
    system_->block_rhs_group(boundary_evaluation_point(group_id), blocks, states, residuals,
                             flux_only, retained_faces);
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
    require_same_field_contract_(stage_state, rhs, "ProgramContext named-flux residual");
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

  Real hmin() const { return system_->cfl_min_dx(); }
  RuntimeParams program_params(int program_block) const {
    return system_->program_params(program_block);
  }

  /// Rank-local control-flow copy; its caller converges any numerical failure.
  void copy_with_provenance(field_type& destination, const field_type& source) const {
    field_exchanges_->copy(&destination, &source);
    PureFieldAlgebra::copy(destination, source);
    count_kernel_();
  }

  struct UniformAffineTerm {
    const field_type* source;
    Real coefficient;
    bool rate;
  };

  // The generated equation has already written destination; retain the proved expression of
  // actual evaluated faces through dynamic branches and loops, not a static source expansion.
  void attach_affine_flux_expression(field_type& destination,
                                     std::span<const UniformAffineTerm> terms) const {
    std::exception_ptr error;
    try {
      std::vector<FieldExchangeExpressions::WeightedField> expression;
      expression.reserve(terms.size());
      for (const auto& term : terms) {
        if (term.source == nullptr || !std::isfinite(term.coefficient))
          throw std::invalid_argument("Program affine field coefficient is invalid");
        require_same_field_contract_(destination, *term.source, "Program affine field");
        expression.push_back({term.source, term.coefficient});
      }
      field_exchanges_->combine(&destination, expression);
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, prepared_execution_lane(),
                                   "Program affine field metadata");
  }

  void axpy(field_type& destination, Real factor, const field_type& source) const {
    const std::array<FieldExchangeExpressions::WeightedField, 2> terms{
        {{&destination, 1}, {&source, factor}}};
    field_exchanges_->combine(&destination, terms);
    count_kernel_();
    pops::saxpy(destination, factor, source);
  }

  void axpy(field_type& destination, Real factor, const field_type& source, Real,
            std::initializer_list<ExactCoefficientTerm>) const {
    axpy(destination, factor, source);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right) const {
    const std::array<FieldExchangeExpressions::WeightedField, 2> terms{
        {{&left, left_factor}, {&right, right_factor}}};
    field_exchanges_->combine(&destination, terms);
    count_kernel_();
    pops::lincomb(destination, left_factor, left, right_factor, right);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
               const field_type& right, Real, std::initializer_list<ExactCoefficientTerm>,
               std::initializer_list<ExactCoefficientTerm>) const {
    lincomb(destination, left_factor, left, right_factor, right);
  }

  void commit_many(std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
    commit_many(std::span<const std::pair<field_type*, const field_type*>>(commits.begin(),
                                                                           commits.size()));
  }

  void commit_many(std::span<const std::pair<field_type*, const field_type*>> commits) const {
    const ExecutionLane& lane = prepared_execution_lane();
    std::vector<field_type*> targets;
    std::vector<const field_type*> sources;
    std::exception_ptr structural_error;
    try {
      targets.reserve(commits.size());
      sources.reserve(commits.size());
      for (const auto& [target, source] : commits) {
        if (target == nullptr || source == nullptr)
          throw std::invalid_argument("ProgramContext commit contains null storage");
        if (std::find(targets.begin(), targets.end(), target) != targets.end())
          throw std::invalid_argument("ProgramContext commit contains a duplicate target");
        require_same_field_contract_(*target, *source, "ProgramContext commit");
        for (const auto& [identity, moving] : runtime_state().moving_interval_geometry_)
          if (target == &system_->block_state(moving.runtime_block))
            throw std::invalid_argument(
                "Program state " + identity +
                " owns moving geometry and requires a coupled state/geometry commit");
        targets.push_back(target);
        sources.push_back(source);
      }
    } catch (...) {
      structural_error = std::current_exception();
    }
    if (all_reduce_max(structural_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && structural_error)
        std::rethrow_exception(structural_error);
      throw std::runtime_error("ProgramContext commit classification failed collectively");
    }

    const long commit_count = static_cast<long>(targets.size());
    if (all_reduce_min(commit_count, lane) != all_reduce_max(commit_count, lane))
      throw std::runtime_error("ProgramContext commit count differs between MPI ranks");

    std::vector<int> runtime_blocks;
    std::vector<char> identity_commits;
    std::vector<const field_type*> masks;
    std::exception_ptr classification_error;
    try {
      runtime_blocks.assign(targets.size(), -1);
      identity_commits.assign(targets.size(), 0);
      masks.assign(targets.size(), nullptr);
      for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
        identity_commits[candidate] = targets[candidate] == sources[candidate] ? 1 : 0;
        for (int block = 0; block < system_->n_blocks(); ++block) {
          if (targets[candidate] == &system_->block_state(block)) {
            runtime_blocks[candidate] = block;
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
      throw std::runtime_error("ProgramContext commit target classification failed collectively");
    }
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
      const long owner = static_cast<long>(runtime_blocks[candidate]);
      const long identity = static_cast<long>(identity_commits[candidate]);
      if (all_reduce_min(owner, lane) != all_reduce_max(owner, lane) ||
          all_reduce_min(identity, lane) != all_reduce_max(identity, lane))
        throw std::runtime_error("ProgramContext commit target owner differs between MPI ranks");
    }

    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
      if (runtime_blocks[candidate] < 0)
        continue;
      masks[candidate] = system_->validate_program_state_publication_candidate_(
          runtime_blocks[candidate], *sources[candidate], lane);
    }
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
      const long masked = masks[candidate] != nullptr ? 1L : 0L;
      if (all_reduce_min(masked, lane) != all_reduce_max(masked, lane))
        throw std::runtime_error("ProgramContext commit mask classification differs between ranks");
    }

    std::vector<std::optional<field_type>> snapshots;
    std::exception_ptr staging_error;
    try {
      snapshots.resize(targets.size());
      for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
        if (identity_commits[candidate] != 0)
          continue;
        if (masks[candidate] != nullptr) {
          snapshots[candidate].emplace(*targets[candidate]);
          copy_active_valid_cells_(*sources[candidate], *snapshots[candidate], *masks[candidate]);
        } else {
          snapshots[candidate].emplace(*sources[candidate]);
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
      throw std::runtime_error("ProgramContext commit staging failed collectively");
    }

    // Continuation publication projects structurally unpacked candidates directly through this
    // native path. The immutable expression identity prevents a second append if generated
    // end-of-program publication already staged the same candidate for this runtime block.
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
      if (runtime_blocks[candidate] >= 0)
        publish_field_exchanges_(runtime_blocks[candidate], *sources[candidate]);
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
      if (snapshots[candidate])
        *targets[candidate] = std::move(*snapshots[candidate]);
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

  /// Return the prepared embedded-boundary mask for this pointwise block, or null for Cartesian / an
  /// inactive embedded boundary.  The route consensus intentionally uses only fixed scalars: this
  /// hot path must not build a dynamic exact-contract payload before every generated kernel. The
  /// route and the mask query are decided by one collective (spec 0.3 SV20, PF8).
  const field_type* pointwise_active_mask(int program_block, const field_type& field) const {
    const ExecutionLane& lane = prepared_execution_lane();
    ConsensusWords<kPointwiseRouteWords + runtime::system::kPreparedActiveMaskWords> words;
    const PointwiseRouteSlots route = add_pointwise_route_consensus_(words, program_block);
    runtime::system::PreparedActiveMaskLocal<field_type> local;
    if (route.runtime_block >= 0)
      local = system_->prepared_program_block_active_mask_local_(route.runtime_block, field, lane);
    const auto mask = runtime::system::add_active_mask_consensus(words, local);
    words.reduce(lane);
    require_pointwise_route_consensus_(words, route);
    runtime::system::require_active_mask_consensus(words, mask);
    return local.mask;
  }

  /// Uniform faces have no coarse coverage to exclude. Accepted-exchange producers
  /// query this separately from the embedded-boundary mask so the AMR route can
  /// enforce both physical activity and finest-owner coverage.
  const field_type* pointwise_exchange_coverage_mask(int, const field_type&) const noexcept {
    return nullptr;
  }

  /// IR18: retain lag >= 1 independently of a pending write to slot zero.
  template <class Kernel>
  field_type spatial_interaction_history(int program_block, const field_type& source,
                                         std::span<const int> components, std::uint64_t max_bytes,
                                         std::string_view identity, std::string_view source_clock,
                                         bool accepted_composite, const std::string& name, int lag,
                                         std::string_view seed_state, std::string_view seed_space,
                                         std::string_view interpolation, Kernel kernel) const {
    const auto& lane = prepared_execution_lane();
    const int owner = resolve_pointwise_program_block_(program_block, lane);
    const auto& history = runtime_state().hist_;
    const field_type* selected = nullptr;
    std::string snapshot;
    auto proof = [&] {
      ExactContractBuilder exact;
      const bool cold = interaction_history_selected(
          history, name, lag, owner, seed_state, seed_space, source_clock, interpolation, exact);
      if (accepted_composite || &source != &history.histories.at(name).at(lag))
        throw std::invalid_argument("interaction History@2 has no exact issued keeper carrier");
      selected = cold ? &state(program_block) : &source;
      if (cold && history.initialized.at(name))
        interaction_history_cold_equal(source, *selected);
      return std::move(exact).release();
    };
    interaction_phase(lane, [&] { snapshot = proof(); });
    std::string authority;
    interaction_phase(lane, [&] {
      ExactContractBuilder exact;
      exact.text(identity).text("pops.spatial-interaction-history-source@1").text(snapshot);
      authority = std::move(exact).release();
    });
    auto result = spatial_interaction(program_block, *selected, components, max_bytes, authority,
                                      source_clock, false, kernel);
    interaction_phase(lane, [&] {
      if (snapshot != proof())
        throw std::logic_error("interaction selected history lifecycle changed during snapshot");
    });
    return result;
  }

  template <class Kernel>
  field_type spatial_interaction(int program_block, const field_type& source,
                                 std::span<const int> components, std::uint64_t max_bytes,
                                 std::string_view identity, std::string_view source_clock,
                                 bool accepted_composite, Kernel kernel) const {
    const auto& lane = prepared_execution_lane();
    PreparedResourceAttempt attempt;
    runtime::multiblock::BoundaryEvaluationPoint point;
    interaction_phase(lane, [&] {
      point = boundary_evaluation_point(0);
      if (point.clock != source_clock)
        throw std::invalid_argument(
            "direct interaction source clock differs from its issued frame");
      attempt = resource_attempt();
      if (!attempt.visible())
        throw std::logic_error("direct interaction requires a live Program attempt");
      if (accepted_composite && &source != &state(program_block))
        throw std::invalid_argument(
            "accepted interaction source is not the accepted State carrier");
    });
    const auto* active = pointwise_active_mask(program_block, source);
    const int owner = resolve_pointwise_program_block_(program_block, lane);
    const auto* kappa = system_->prepared_program_block_volume_fraction_(owner, source, lane);
    const auto geom = geometry();
    const field_type* output_prototype = nullptr;
    interaction_phase(lane, [&] {
      output_prototype = &system_->block_state(owner);
      if (!runtime_state().moving_interval_geometry_.empty())
        throw std::invalid_argument(
            "direct cell-midpoint interaction requires a prepared Cartesian physical map");
    });
    const std::array<InteractionLevelView<Dim, typename field_type::memory_space>, 1> levels{
        InteractionLevelView<Dim, typename field_type::memory_space>{&source, active, nullptr,
                                                                     kappa, geom}};
    std::string authority;
    interaction_phase(lane, [&] {
      ExactContractBuilder exact;
      exact.text(identity).scalar(attempt.ordinal());
      interaction_append_point(exact, point);
      authority = std::move(exact).release();
    });
    auto result = direct_spatial_interaction<Dim, typename field_type::memory_space>(
        levels, 0, components, max_bytes, authority, lane, kernel, output_prototype);
    interaction_phase(lane, [&] {
      if (!attempt.visible() || !attempt.same_attempt(resource_attempt()) ||
          point != boundary_evaluation_point(0))
        throw std::logic_error("direct interaction source attempt/frame was revoked");
    });
    return result;
  }

  /// Reduce one generated per-cell status on the same authenticated lane and layout used by its
  /// pointwise kernel.  Empty ranks contribute negative infinity, normalized to success only when
  /// the entire collective layout is empty. The lane, route, mask and storage checks, the local
  /// failure and the status maximum are decided by one collective on the prepared lane, tested in
  /// the order of the separate collectives they replace (spec 0.3 SV20, PF8).
  Real pointwise_status_max(int program_block, const field_type& status,
                            const field_type* active_cells, const ExecutionLane& lane) const {
    const ExecutionLane& prepared = prepared_execution_lane();
    ConsensusWords<kPointwiseRouteWords + runtime::system::kPreparedActiveMaskWords + 7> words;
    const std::size_t foreign_lane = words.failure(&lane != &prepared);
    const PointwiseRouteSlots route = add_pointwise_route_consensus_(words, program_block);
    runtime::system::PreparedActiveMaskLocal<field_type> expected;
    if (route.runtime_block >= 0)
      expected =
          system_->prepared_program_block_active_mask_local_(route.runtime_block, status, lane);
    const auto mask = runtime::system::add_active_mask_consensus(words, expected);
    const std::size_t foreign_mask = words.failure(active_cells != expected.mask);
    const std::size_t components = words.failure(status.ncomp() != 1);
    MaskedMaxLocalResult local;
    std::exception_ptr local_error;
    try {
      local = pops::reduce_masked_max_local(status, 0, expected.mask);
    } catch (...) {
      local_error = std::current_exception();
    }
    const std::size_t failed = words.failure(static_cast<bool>(local_error));
    const std::size_t invalid = words.failure(local.has_invalid);
    const std::size_t active = words.failure(local.has_active);
    const std::size_t maximum = words.maximum(static_cast<double>(local.maximum));
    words.reduce(prepared);
    if (words.failed(foreign_lane))
      throw std::invalid_argument(
          "Program pointwise status requires the context's authenticated execution lane");
    require_pointwise_route_consensus_(words, route);
    runtime::system::require_active_mask_consensus(words, mask);
    if (words.failed(foreign_mask))
      throw std::invalid_argument("Program pointwise status received a foreign active-cell mask");
    if (words.failed(components))
      throw std::invalid_argument("Program pointwise status requires exactly one component");
    if (words.failed(failed)) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program pointwise status reduction failed collectively");
    }
    if (words.failed(invalid))
      return Real(3);
    if (!words.failed(active))
      return Real(0);
    const Real reduced = static_cast<Real>(words.value(maximum));
    return std::isfinite(reduced) ? reduced : Real(3);
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
    return reduce_owner_(local, local_error, lane, OwnerReduction::sum, "Program sum reduction");
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
    return reduce_owner_(local, local_error, lane, OwnerReduction::sum,
                         "Program abs-sum reduction");
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
    return reduce_owner_(local, local_error, lane, OwnerReduction::max, "Program max reduction");
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
    return reduce_owner_(local, local_error, lane, OwnerReduction::min, "Program min reduction");
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
    return std::sqrt(
        reduce_owner_(local, local_error, lane, OwnerReduction::sum, "Program norm2 reduction"));
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
    return reduce_owner_(local, local_error, lane, OwnerReduction::max,
                         "Program norm-inf reduction");
  }
  Real dot(int program_block, const field_type& left, const field_type& right) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, left);
    std::exception_ptr local_error;
    Real local = Real(0);
    try {
      require_same_field_contract_(left, right, "ProgramContext dot");
      local = pops::dot_active_local(left, right, 0, active);
    } catch (...) {
      local_error = std::current_exception();
    }
    return reduce_owner_(local, local_error, lane, OwnerReduction::sum, "Program dot reduction");
  }

  /// pops.program.dot-all@1: raw algebra on every component of the owned active domain.
  /// The historical dot overload remains component-zero algebra.
  Real dot_all(int program_block, const field_type& left, const field_type& right) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const field_type* const active = pointwise_active_mask(program_block, left);
    std::exception_ptr local_error;
    FiniteCompensatedSum local;
    try {
      require_same_field_contract_(left, right, "ProgramContext dot_all");
      const auto checked = pops::dot_owned_active_all_finite_sum_local(left, right, active);
      // Every physical replica is validated; exactly one contributes to the lane pairing.
      local =
          left.distribution().replicated() && lane.rank() != 0 ? FiniteCompensatedSum{} : checked;
    } catch (...) {
      local_error = std::current_exception();
    }
    return pops::collective_finite_compensated_sum(local, local_error, lane.communicator(),
                                                   "Program dot-all reduction");
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

  PhysicalBoundaryConditions<Dim> field_boundary_conditions(
      int program_block, std::string_view provider_slot = {}) const {
    const ExecutionLane& lane = prepared_execution_lane();
    const int runtime_block =
        resolve_prepared_program_block_(program_block, lane, "Program field boundary block");
    std::optional<PhysicalBoundaryConditions<Dim>> conditions;
    std::exception_ptr error;
    try {
      conditions.emplace(system_->prepared_field_boundary_conditions(runtime_block, provider_slot));
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane,
                                   "Program field boundary authority failed collectively");
    return std::move(*conditions);
  }

  std::shared_ptr<field_boundary_session_type> prepare_field_boundary_session(
      int program_block, field_type& prototype, const ExecutionLane& lane,
      std::string_view provider_slot = {}) const {
    require_prepared_lane_(lane, "Program field boundary preparation");
    const auto conditions = field_boundary_conditions(program_block, provider_slot);
    int runtime_block = -1;
    std::string contract;
    std::exception_ptr error;
    try {
      runtime_block = sys_block(program_block);
      require_same_layout_(prototype, system_->block_state(runtime_block),
                           "Program field boundary prototype");
      if (prototype.ncomp() != 1)
        throw std::invalid_argument("Program field potential must have one scalar component");
      for (int axis = 0; axis < Dim; ++axis)
        if (prototype.ghosts()[axis] < 1)
          throw std::invalid_argument("Program field potential requires a ghost on every axis");
      ExactContractBuilder proof;
      proof.text("pops.program.prepared-field-boundary-session@1")
          .scalar(std::int32_t{Dim})
          .scalar(std::int32_t{program_block})
          .scalar(std::int32_t{runtime_block})
          .text(provider_slot)
          .text(lane.identity());
      for (int axis = 0; axis < Dim; ++axis) {
        proof.scalar(conditions.spacing()[axis]);
        for (const auto side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          const auto& law = conditions.at(face);
          proof.presence(conditions.topology().is_periodic(face))
              .scalar(static_cast<std::uint8_t>(law.kind))
              .scalar(law.alpha)
              .scalar(law.beta)
              .scalar(law.value);
        }
      }
      contract = std::move(proof).release();
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane,
                                   "Program field boundary prototype failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"program-prepared-field-boundary-session", contract}}, lane))
      throw std::invalid_argument("Program field boundary identity differs across ranks");
    auto halo = scalar_boundary_session_type::prepare(geometry(), conditions.topology(), prototype,
                                                      lane, next_scalar_boundary_generation_());
    std::shared_ptr<field_boundary_session_type> session;
    error = nullptr;
    try {
      auto physical = prepare_physical_boundary(
          geometry().domain(), prototype.ghosts(), conditions,
          BoundaryScheduleBudget{scalar_boundary_detail::boundary_entry_budget(
              geometry(), prototype.ghosts(), conditions.topology())});
      session = std::shared_ptr<field_boundary_session_type>(new field_boundary_session_type(
          system_, program_block, runtime_block, std::string(provider_slot), lane, std::move(halo),
          std::move(physical)));
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane,
                                   "Program field boundary allocation failed collectively");
    return session;
  }

  void fill_field_potential(int program_block, field_type& potential,
                            const field_boundary_session_type& boundary) const {
    require_field_boundary_session_(boundary, "Program field potential fill");
    const int runtime_block = resolve_prepared_program_block_(program_block, boundary.lane(),
                                                              "Program field potential owner");
    if (all_reduce_max(runtime_block == boundary.runtime_block_ ? 0L : 1L, boundary.lane()) != 0)
      throw std::invalid_argument("Program field potential fill lost its exact block owner");
    boundary.fill(potential);
  }

  void fill_field_potential(int program_block, field_type& potential,
                            std::string_view provider_slot = {}) const {
    auto boundary = prepare_field_boundary_session(program_block, potential,
                                                   prepared_execution_lane(), provider_slot);
    fill_field_potential(program_block, potential, *boundary);
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
    laplacian_with_boundary_(output, input, boundary);
  }

  void laplacian(field_type& output, field_type& input,
                 const field_boundary_session_type& boundary) const {
    require_field_boundary_session_(boundary, "Program field Laplacian boundary");
    laplacian_with_boundary_(output, input, boundary);
  }

 private:
  template <class Boundary>
  void laplacian_with_boundary_(field_type& output, field_type& input,
                                const Boundary& boundary) const {
    require_componentwise_stencil_(output, input, "Program Laplacian");
    boundary.fill(input);
    const Geometry<Dim> geom = boundary.geometry();
    const int components = input.ncomp();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<Real, Dim> result = output.fab(local).view();
      const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
      for_each_cell(output.box(local),
                    program_context_detail::LaplacianKernel<Dim>{result, value, geom, components});
    }
    count_kernel_();
  }

 public:
  void laplacian(field_type& output, field_type& input,
                 const scalar_boundary_session_type& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program Laplacian");
    laplacian(output, input, boundary);
  }

  void laplacian(field_type& output, field_type& input, const field_boundary_session_type& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program field Laplacian");
    laplacian(output, input, boundary);
  }

  void laplacian_field(field_type& output, field_type& input,
                       const field_boundary_session_type& boundary) const {
    laplacian(output, input, boundary);
  }

  void gradient(field_type& output, field_type& input) const {
    auto boundary = prepare_mesh_boundary_session(input, prepared_execution_lane());
    gradient(output, input, *boundary);
  }

  void gradient(field_type& output, field_type& input,
                const scalar_boundary_session_type& boundary) const {
    require_prepared_lane_(boundary.lane(), "Program gradient boundary");
    gradient_with_boundary_(output, input, boundary);
  }

  void gradient(field_type& output, field_type& input,
                const field_boundary_session_type& boundary) const {
    require_field_boundary_session_(boundary, "Program field gradient boundary");
    gradient_with_boundary_(output, input, boundary);
  }

 private:
  template <class Boundary>
  void gradient_with_boundary_(field_type& output, field_type& input,
                               const Boundary& boundary) const {
    require_scalar_stencil_(output, input, Dim, "Program gradient");
    boundary.fill(input);
    const Geometry<Dim> geom = boundary.geometry();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const FieldView<Real, Dim> result = output.fab(local).view();
      const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
      for_each_cell(output.box(local),
                    program_context_detail::GradientKernel<Dim>{result, value, geom});
    }
    count_kernel_();
  }

 public:
  void gradient(field_type& output, field_type& input, const scalar_boundary_session_type& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program gradient");
    gradient(output, input, boundary);
  }

  void gradient(field_type& output, field_type& input, const field_boundary_session_type& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program field gradient");
    gradient(output, input, boundary);
  }

  void gradient_field(field_type& output, field_type& input,
                      const field_boundary_session_type& boundary) const {
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
    tensor_laplacian_with_boundary_(output, input, tensor, boundary);
  }

  void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                        const field_boundary_session_type& boundary) const {
    require_field_boundary_session_(boundary, "Program field tensor Laplacian boundary");
    tensor_laplacian_with_boundary_(output, input, tensor, boundary);
  }

 private:
  template <class Boundary>
  void tensor_laplacian_with_boundary_(field_type& output, field_type& input,
                                       const field_type& tensor, const Boundary& boundary) const {
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
      for_each_cell(output.box(local),
                    program_context_detail::ScalarOperatorKernel<Dim, decltype(tensor_operator)>{
                        result, tensor_operator});
    }
    count_kernel_();
  }

 public:
  void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                        const scalar_boundary_session_type& boundary,
                        const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program tensor Laplacian");
    tensor_laplacian(output, input, tensor, boundary);
  }

  void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                        const field_boundary_session_type& boundary,
                        const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_boundary_point_(point, "Program field tensor Laplacian");
    tensor_laplacian(output, input, tensor, boundary);
  }

  void tensor_laplacian_field(field_type& output, field_type& input, const field_type& tensor,
                              const field_boundary_session_type& boundary) const {
    tensor_laplacian(output, input, tensor, boundary);
  }

  void configure_history_field_metadata(std::uint64_t retained_bytes,
                                        std::uint64_t workspace_bytes) const {
    HistoryFieldMetadataBudget budget{retained_bytes, workspace_bytes};
    budget.validate();
    if (!budget.enabled())
      throw std::invalid_argument("History Field metadata requires positive explicit capacities");
    auto& manager = runtime_state().hist_;
    if (manager.field_metadata_budget.enabled() && manager.field_metadata_budget != budget)
      throw std::invalid_argument("History Field metadata capacity changed after installation");
    manager.field_metadata_budget = budget;
  }

  void register_history(const std::string& name, int lag, int ncomp = -1, int program_owner = -1,
                        const std::string& state_identity = {},
                        const std::string& space_identity = {},
                        const std::string& clock_identity = {},
                        const std::string& interpolation_identity = {},
                        int ghost_depth = -1) const {
    if (name.empty() || lag < 1 || ghost_depth < -1)
      throw std::invalid_argument(
          "ProgramContext history requires a name, positive lag and valid declared halo");
    const int owner = program_owner < 0 ? 0 : sys_block(program_owner);
    const field_type& prototype = system_->block_state(owner);
    const int components = ncomp < 0 ? prototype.ncomp() : ncomp;
    if (components < 1)
      throw std::invalid_argument("ProgramContext history component count must be positive");
    Extent<Dim> ghosts = prototype.ghosts();
    if (ghost_depth >= 0)
      for (int axis = 0; axis < Dim; ++axis)
        ghosts[axis] = ghost_depth;
    auto& history = runtime_state().hist_;
    const int ring_depth = lag + 1;
    const auto existing = history.histories.find(name);
    if (existing != history.histories.end()) {
      if (history.depth.at(name) != ring_depth || history.owner.at(name) != owner ||
          existing->second.front().ncomp() != components ||
          existing->second.front().ghosts() != ghosts ||
          history.ghost_depth.at(name) != ghost_depth ||
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
      ring.push_back(make_scratch_(prototype, components, ghosts));
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
    history.ghost_depth[name] = ghost_depth;
    history.slot_dt[name] = std::vector<Real>(static_cast<std::size_t>(ring_depth), Real(0));
    history.slot_sample[name].assign(static_cast<std::size_t>(ring_depth),
                                     HistorySampleIdentity::zero_start());
    history.slot_exchanges[name].resize(static_cast<std::size_t>(ring_depth));
    history.slot_fields[name].resize(static_cast<std::size_t>(ring_depth));
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
    if (auto metadata = manager.slot_exchanges.find(name); metadata != manager.slot_exchanges.end())
      field_exchanges_->restore(&result, metadata->second.at(static_cast<std::size_t>(lag)));
    return result;
  }

  field_type& history_observation(const std::string& name, int lag, int components,
                                  int program_owner, const std::string& descriptor) const {
    continuation_local_phase(
        [&] {
          const auto& manager = runtime_state().hist_;
          const auto found = manager.histories.find(name);
          if (found == manager.histories.end() || lag < 1 || lag >= manager.depth.at(name) ||
              manager.state_identity.at(name) != descriptor ||
              manager.owner.at(name) != sys_block(program_owner))
            throw std::invalid_argument("independent Field history has foreign storage authority");
          runtime_state().require_frozen_output_history_projection(
              name, manager.owner.at(name), descriptor, manager.space_identity.at(name),
              manager.clock_identity.at(name), manager.interpolation_identity.at(name),
              manager.depth.at(name), components, program_owner, manager.ghost_depth.at(name));
          const auto& metadata = manager.slot_fields.at(name).at(static_cast<std::size_t>(lag));
          if (!manager.initialized.at(name) || !metadata.support ||
              metadata.support->program_identity != runtime_state().installed_hash_)
            throw std::invalid_argument("independent Field history lacks its retained producer");
        },
        "read retained independent Field history");
    return history(name, lag, components);
  }

  field_type& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(name))
      throw std::out_of_range("ProgramContext history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramContext history component contract differs");
    if (auto metadata = manager.slot_exchanges.find(name); metadata != manager.slot_exchanges.end())
      field_exchanges_->restore(&result, metadata->second.at(static_cast<std::size_t>(lag)));
    return result;
  }

  void store_history(const std::string& name, const field_type& value) const {
    if (std::isfinite(current_dt_) && current_dt_ > 0.0) {
      store_history_(name, value, current_dt_);
      return;
    }
    // Preserve the direct legacy route when this context has no active generated-step interval:
    // System supplies its accepted last-dt provenance (or its zero-dt pre-step default).
    const auto instance = continuation_field_instance_(value);
    if (instance && (instance->metadata.support || instance->metadata.representation))
      throw std::invalid_argument(
          "retained Field history store requires an explicit positive interval");
    revoke_program_history_values_();
    system_->store_history(name, value);
  }
  void store_history(const std::string& name, const field_type& value, double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramContext history dt must be finite and positive");
    store_history_(name, value, dt);
  }
  void rotate_histories() const {
    revoke_program_history_values_();
    runtime_state().hist_.rotate();
  }
  void rotate_histories(const std::string& clock) const {
    revoke_program_history_values_();
    runtime_state().hist_.rotate(clock);
  }

  // Select the retained numerical slots before allocating an owned result.
  history_field_detail::LinearBracket history_linear_bracket_(
      const std::string& name, int max_lag, int program_owner, const std::string& source_clock,
      const std::string& target_clock, int target_step, Real target_offset) const {
    const auto& manager = runtime_state().hist_;
    const auto found = manager.histories.find(name);
    if (found == manager.histories.end() || manager.depth.at(name) <= max_lag ||
        manager.owner.at(name) != sys_block(program_owner) || !manager.initialized.at(name) ||
        manager.clock_identity.at(name) != source_clock)
      throw std::runtime_error("history interpolation requires its initialized owner/clock ring");
    const double coordinate = (static_cast<double>(target_step) + target_offset) *
                              static_cast<double>(clock_schedule_.ticks_per_macro(source_clock)) /
                              static_cast<double>(clock_schedule_.ticks_per_macro(target_clock));
    return history_field_detail::linear_bracket(coordinate, physical_time(),
                                                manager.slot_dt.at(name), max_lag);
  }

  std::shared_ptr<field_type> interpolate_history_linear_owned(
      const std::string& name, int max_lag, int program_owner, const std::string& source_clock,
      const std::string& target_clock, int target_step, Real target_offset) const {
    const auto bracket = history_linear_bracket_(name, max_lag, program_owner, source_clock,
                                                 target_clock, target_step, target_offset);
    if (bracket.endpoint())
      return own_structural_field(history(name, static_cast<int>(bracket.selected())));
    const auto& older = history(name, static_cast<int>(bracket.older));
    const auto& newer = history(name, static_cast<int>(bracket.newer));
    const auto& manager = runtime_state().hist_;
    const auto metadata = history_field_detail::interpolated_metadata(
        manager.slot_fields.at(name).at(bracket.older),
        manager.slot_fields.at(name).at(bracket.newer), bracket.alpha, bracket.target_time,
        manager.field_metadata_budget);
    auto result = own_structural_field(older);
    lincomb(*result, Real(1) - bracket.alpha, older, bracket.alpha, newer);
    register_continuation_field_copy_(
        *result, result,
        {field_exchanges_->expression(result.get()), metadata.support, metadata.representation});
    return result;
  }

  // Legacy borrowed-output callers have no lifetime to retain Field descriptions.
  // The generated public API always uses the owned route above.
  void interpolate_history_linear(field_type& output, const std::string& name, int max_lag,
                                  int program_owner, const std::string& source_clock,
                                  const std::string& target_clock, int target_step,
                                  Real target_offset) const {
    const auto bracket = history_linear_bracket_(name, max_lag, program_owner, source_clock,
                                                 target_clock, target_step, target_offset);
    const auto& manager = runtime_state().hist_;
    if (const auto fields = manager.slot_fields.find(name); fields != manager.slot_fields.end())
      if (!fields->second.at(bracket.older).empty() || !fields->second.at(bracket.newer).empty())
        throw std::invalid_argument(
            "qualified history interpolation requires an owned Field result");
    const auto& older = history(name, static_cast<int>(bracket.older));
    const auto& newer = history(name, static_cast<int>(bracket.newer));
    require_same_field_contract_(output, older, "history interpolation");
    lincomb(output, Real(1) - bracket.alpha, older, bracket.alpha, newer);
  }

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

#include <pops/runtime/program/program_context_continuation_fields.inc>
#include <pops/runtime/program/program_context_field_representation_obligation.inc>

 public:
  /// Own opaque structural storage without performing a numerical Field operation.
  /// The copy constructor retains the complete valid/ghost storage. Its caller
  /// converges allocation/copy errors before publishing any structural slot.
  std::shared_ptr<field_type> own_structural_field(const field_type& source) const {
    (void)continuation_field_instance_(source);
    auto owned = std::make_shared<field_type>(source);
    continuation_bind_field_provenance(source, *owned, owned);
    return owned;
  }

  void continuation_bind_field_provenance(const field_type& source,
                                          const field_type& destination) const {
    field_exchanges_->copy(&destination, &source);
  }
  template <class Lifetime>
  FieldExchangeExpressions::Ref continuation_bind_field_provenance(const field_type& source,
                                                                   const field_type& destination,
                                                                   const Lifetime& lifetime) const {
    const auto instance = continuation_field_instance_(source);
    field_exchanges_->copy_retained(&destination, &source, lifetime);
    const auto expression = field_exchanges_->expression(&destination);
    if (instance) {
      auto metadata = instance->metadata;
      metadata.expression = expression;
      register_continuation_field_copy_(destination, lifetime, std::move(metadata));
    }
    return expression;
  }
  template <class Lifetime>
  ContinuationFieldMetadata continuation_bind_field_provenance(const field_type& source,
                                                               const field_type& destination,
                                                               const Lifetime& lifetime,
                                                               const ContinuationType& type) const {
    continuation_preflight_field(source, type);
    const auto instance = continuation_field_instance_(source);
    const auto expression = continuation_bind_field_provenance(source, destination, lifetime);
    return {expression, instance ? instance->metadata.support : ContinuationFieldSupport::Ref{},
            instance ? instance->metadata.representation : FieldRepresentationObligation::Ref{}};
  }
  template <class Lifetime>
  void continuation_bind_field_provenance(const field_type& source, const field_type& destination,
                                          const Lifetime& lifetime,
                                          const ContinuationFieldMetadata& metadata) const {
    const auto instance = continuation_field_instance_(source);
    if ((metadata.support || metadata.representation) &&
        (!instance || instance->metadata.support != metadata.support ||
         instance->metadata.expression != metadata.expression ||
         instance->metadata.representation != metadata.representation))
      throw std::invalid_argument(
          "Uniform Field restoration requires an explicit support transfer");
    register_continuation_field_copy_(destination, lifetime, metadata);
    continuation_bind_field_provenance(source, destination, lifetime, metadata.expression);
  }
  template <class Lifetime>
  FieldExchangeExpressions::Ref continuation_bind_field_provenance(
      const field_type&, const field_type& destination, const Lifetime& lifetime,
      FieldExchangeExpressions::Ref expression) const {
    field_exchanges_->restore_retained(&destination, expression, lifetime);
    return expression;
  }
#include <pops/runtime/program/program_context_continuation.inc>

  const PreparedVectorDistribution<Dim>& program_resource_vector_distribution() const {
    return PreparedVectorDistribution<Dim>::Distributed;
  }

  int program_resource_field_level() const noexcept { return 0; }

  std::int64_t invocation_level() const {
    if (!resource_attempt().visible() || !std::isfinite(current_dt_) || !(current_dt_ > 0.0))
      throw std::logic_error("invocation level requires a prepared Program attempt");
    return 0;
  }

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
    system_->require_solve_outcome_creation_(3);
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
    return system_->track_solve_outcome(
        pops::solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls));
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

  void publish_field_warm_start(const std::string& provider_slot, int program_block,
                                const field_type& potential, int evaluation_id,
                                const std::string& publication_identity) const {
    const auto& lane = prepared_execution_lane();
    const auto point = boundary_evaluation_point(evaluation_id);
    int runtime_block = -1;
    std::exception_ptr error;
    try {
      require_boundary_point_(point, "Program field warm start");
      runtime_block = sys_block(program_block);
      if (point != boundary_evaluation_point(point.stage) || publication_identity.empty())
        throw std::invalid_argument("Program field warm start lost its context-owned point");
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane,
                                   "Program field warm-start owner failed collectively");
    system_->publish_program_field_warm_start(point, provider_slot, runtime_block, potential,
                                              publication_identity);
  }

  void set_field_boundary_parameters(const std::string& provider_slot,
                                     const std::vector<double>& parameters) const {
    system_->set_field_boundary_parameters(provider_slot, parameters);
  }

  [[nodiscard]] SolveOutcome solve_fields() const { return system_->solve_fields(); }

  [[nodiscard]] SolveOutcome solve_fields_from_program_values_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
      std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
    using Request = runtime::system::FieldSolveRequest<Dim>;
    const auto& lane = prepared_execution_lane();
    std::optional<Request> request;
    std::exception_ptr error;
    try {
      require_boundary_point_(point, "Program SSA Field solve");
      require_program_value_identity_();
      if (point.level != 0 || field_node < 0 || field.empty())
        throw std::invalid_argument("Program SSA Field solve has an invalid point or identity");
      if (point != boundary_evaluation_point(point.stage))
        throw std::invalid_argument(
            "Program SSA Field solve lost its context-owned consumer point");
      std::vector<typename Request::Source> sources;
      std::vector<typename Request::LegacySource> legacy;
      std::set<int> seen;
      for (const auto& item : overrides) {
        if (!item.state || !seen.insert(item.program_block).second)
          throw std::invalid_argument("Program SSA Field solve has a null or duplicate stage");
        const int owner = sys_block(item.program_block);
        require_program_stage_(item.program_block, owner, *item.state);
        if (item.source_ssa >= 0) {
          auto value = program_value(item.source_ssa, item.program_block, *item.state);
          auto validate =
              require_program_field_value(value, field_node, item.program_block, *item.state);
          sources.push_back({std::move(value), std::move(validate)});
        } else {
          if (program_values_.state_->edges.contains({field_node, item.program_block}))
            throw std::invalid_argument("V2 Field source cannot enter the legacy source route");
          legacy.push_back({owner, 0, std::make_shared<field_type>(*item.state)});
        }
      }
      for (const auto& edge : program_values_.state_->plan.fields) {
        if (edge.field_node != field_node || seen.contains(edge.program_block))
          continue;
        if (!edge.accepted_only)
          throw std::invalid_argument("Program SSA Field solve lacks its declared stage override");
        const int owner = sys_block(edge.program_block);
        const auto& accepted = system_->block_state(owner);
        auto value =
            program_values_.accepted_(field_node, edge.program_block,
                                      typename ProgramValueAuthority<Dim>::Slot{
                                          ProgramValueStorage::State, 0, owner, -1, 0, {}, 0},
                                      accepted, resource_attempt(), point);
        auto validate = program_values_.require_accepted_field_(
            value, field_node, edge.program_block, 0, accepted, resource_attempt());
        sources.push_back({std::move(value), std::move(validate)});
      }
      request = Request(std::string(field), point, std::move(sources), std::move(legacy));
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane, "Program SSA Field request failed collectively");
    system_->prepare_named_field_publication_storage_(request->field());
    return system_->run_field_publication_outcome_(
        [this, &request] { return system_->solve_fields_from_request_in_place_(*request); });
  }

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
  using ScratchKey = std::tuple<ScratchKind, int, std::int64_t, int>;

#include <pops/runtime/program/program_context_value_authority_private.inc>

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

  void require_field_boundary_session_(const field_boundary_session_type& boundary,
                                       const char* operation) const {
    require_prepared_lane_(boundary.lane(), operation);
    std::exception_ptr error;
    try {
      if (boundary.system_ != system_ ||
          sys_block(boundary.program_block_) != boundary.runtime_block_)
        throw std::invalid_argument(std::string(operation) +
                                    " lost its installed System and exact block authority");
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, boundary.lane(),
                                   "Program prepared field session validation failed collectively");
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

  static void copy_active_valid_cells_(const field_type& source, field_type& destination,
                                       const field_type& active) {
    require_same_layout_(source, destination, "ProgramContext commit active copy");
    require_same_layout_(source, active, "ProgramContext commit active mask");
    if (source.ncomp() != destination.ncomp() || active.ncomp() != 1)
      throw std::invalid_argument(
          "ProgramContext commit active copy requires matching components and a one-component "
          "mask");
    for (std::size_t local = 0; local < source.local_size(); ++local)
      for_each_cell(source.box(local),
                    program_context_detail::CopyActiveValidKernel<Dim>{
                        std::as_const(source).fab(local).view(), destination.fab(local).view(),
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

  enum class OwnerReduction { sum, max, min };

  /// One collective per generated reduction (spec 0.3 SV20, duties 2 and 3). The local failure
  /// travels as a second word of the same reduction, combined by the same operator (a negated
  /// flag for min), instead of a separate vote before it. A single rank rethrows its own failure.
  Real reduce_owner_(Real local, std::exception_ptr local_error, const ExecutionLane& lane,
                     OwnerReduction reduction, const char* operation) const {
    if (lane.size() == 1) {
      if (local_error)
        std::rethrow_exception(local_error);
      return local;
    }
    const Real failed = local_error ? Real(1) : Real(0);
    std::array<Real, 2> words{local, reduction == OwnerReduction::min ? -failed : failed};
    switch (reduction) {
      case OwnerReduction::sum:
        all_reduce_sum_inplace(words.data(), 2, lane);
        break;
      case OwnerReduction::max:
        all_reduce_max_inplace(words.data(), 2, lane);
        break;
      case OwnerReduction::min:
        all_reduce_min_inplace(words.data(), 2, lane);
        break;
    }
    if (words[1] != Real(0))
      throw std::runtime_error(std::string(operation) + " failed collectively");
    return words[0];
  }

  static constexpr std::size_t kPointwiseRouteWords = 5;

  /// The slots of one pointwise route in a ConsensusWords: its local failure, then the program and
  /// runtime blocks, which every rank must hold equal.
  struct PointwiseRouteSlots {
    int runtime_block = -1;
    std::size_t failed = 0;
    std::size_t program = 0;
    std::size_t runtime = 0;
  };

  template <std::size_t Capacity>
  PointwiseRouteSlots add_pointwise_route_consensus_(ConsensusWords<Capacity>& words,
                                                     int program_block) const {
    PointwiseRouteSlots route;
    bool failed = false;
    try {
      route.runtime_block = sys_block(program_block);
    } catch (...) {
      route.runtime_block = -1;
      failed = true;
    }
    route.failed = words.failure(failed);
    route.program = words.agreement(program_block);
    route.runtime = words.agreement(route.runtime_block);
    return route;
  }

  template <std::size_t Capacity>
  static void require_pointwise_route_consensus_(const ConsensusWords<Capacity>& words,
                                                 const PointwiseRouteSlots& route) {
    if (words.failed(route.failed))
      throw std::runtime_error("Program pointwise block route failed collectively");
    if (words.differs(route.program) || words.differs(route.runtime))
      throw std::runtime_error("Program pointwise block route differs across ranks");
  }

  int resolve_pointwise_program_block_(int program_block, const ExecutionLane& lane) const {
    ConsensusWords<kPointwiseRouteWords> words;
    const PointwiseRouteSlots route = add_pointwise_route_consensus_(words, program_block);
    words.reduce(lane);
    require_pointwise_route_consensus_(words, route);
    return route.runtime_block;
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

  static void require_componentwise_stencil_(const field_type& output, const field_type& input,
                                             const char* operation) {
    if (input.ncomp() < 1 || output.ncomp() != input.ncomp() || output.layout() != input.layout() ||
        output.distribution() != input.distribution() ||
        output.local_rank() != input.local_rank() || output.local_size() != input.local_size())
      throw std::invalid_argument(std::string(operation) +
                                  " fields do not share the exact componentwise stencil layout");
    for (int axis = 0; axis < Dim; ++axis)
      if (input.ghosts()[axis] < 1)
        throw std::invalid_argument(std::string(operation) +
                                    " input requires one ghost on every native axis");
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
    int runtime_owner = -1;
    try {
      runtime_owner = scratch_prototype_owner_(prototype);
    } catch (const std::invalid_argument&) {
      if (program_values_.state_->installed)
        throw;
      // Preserve the direct legacy scratch API's detached-prototype rebinding. This branch does
      // not issue SSA authority. A V2 plan always requires an actual prototype runtime owner.
      bool found_owner = false;
      for (const auto& [existing_key, existing_field] : scratch_) {
        (void)existing_field;
        if (std::get<0>(existing_key) != kind || std::get<2>(existing_key) != value_id ||
            std::get<3>(existing_key) != subslot)
          continue;
        if (found_owner && runtime_owner != std::get<1>(existing_key))
          throw std::invalid_argument("legacy scratch rebind has ambiguous runtime owners");
        runtime_owner = std::get<1>(existing_key);
        found_owner = true;
      }
    }
    const ScratchKey key{kind, runtime_owner, value_id, subslot};
    drain_program_value_writes_();
    auto [entry, inserted] = scratch_.try_emplace(key);
    field_type& result = entry->second;
    program_values_.revoke_buffer_(result);
    if (inserted || result.layout() != prototype.layout() ||
        result.distribution() != prototype.distribution() ||
        result.local_rank() != prototype.local_rank() || result.ncomp() != ncomp ||
        result.ghosts() != ghosts)
      result = make_scratch_(prototype, ncomp, ghosts);
    else
      result.set_val(Real(0));
    field_exchanges_->reset(&result);
    return result;
  }

  void store_history_(const std::string& name, const field_type& value, double dt) const {
    auto& manager = runtime_state().hist_;
    std::vector<field_type> prepared_fields;
    std::vector<Real> prepared_dts;
    std::vector<HistorySampleIdentity> prepared_samples;
    std::vector<FieldExchangeExpressions::Ref> prepared_exchanges;
    std::vector<HistoryFieldMetadata> prepared_metadata;
    std::string contract;
    std::exception_ptr error;
    try {
      const auto found = manager.histories.find(name);
      if (found == manager.histories.end() || found->second.empty())
        throw std::out_of_range("ProgramContext history is not registered");
      require_same_field_contract_(found->second.front(), value, "ProgramContext history store");
      const auto instance = continuation_field_instance_(value);
      const HistoryFieldMetadata incoming =
          instance
              ? HistoryFieldMetadata{instance->metadata.support, instance->metadata.representation}
              : HistoryFieldMetadata{};
      if (!incoming.empty() && !manager.field_metadata_budget.enabled())
        throw std::length_error(
            "retaining Field metadata requires "
            "Program(history_field_metadata_budget=(retained_bytes, workspace_bytes))");
      if (manager.field_metadata_budget.enabled()) {
        std::uint64_t required = 0;
        history_field_detail::charge(required, found->second.size(), sizeof(HistoryFieldMetadata),
                                     manager.field_metadata_budget.workspace_bytes);
      }
      prepared_metadata = manager.slot_fields.at(name);
      prepared_dts = manager.slot_dt.at(name);
      prepared_samples =
          manager.prepare_sample_store(name, system_->time() + logical_physical_time_offset_, dt);
      prepared_exchanges = manager.slot_exchanges.at(name);
      const auto expression = field_exchanges_->expression(&value);
      FieldExchangeExpressions::require_expression(expression, field_exchanges_->limits());
      const std::size_t count = manager.initialized.at(name) ? 1 : found->second.size();
      prepared_fields.reserve(count);
      for (std::size_t slot = 0; slot < count; ++slot) {
        prepared_fields.push_back(value);
        prepared_dts[slot] = static_cast<Real>(dt);
        prepared_exchanges[slot] = expression;
        prepared_metadata[slot] = incoming;
      }
      if (manager.field_metadata_budget.enabled()) {
        auto budget = manager.field_metadata_budget;
        std::uint64_t staging_bytes = 0;
        history_field_detail::charge(staging_bytes, prepared_metadata.size(),
                                     sizeof(HistoryFieldMetadata), budget.workspace_bytes);
        budget.workspace_bytes -= staging_bytes;
        history_field_detail::Tables tables(budget.workspace_bytes);
        tables.collect(manager.slot_fields, budget, name, prepared_metadata);
      }
      if (field_exchanges_->limits().max_bases) {
        std::vector<FieldExchangeExpressions::Ref> roots;
        for (const auto& [ring_name, metadata] : manager.slot_exchanges) {
          const auto& selected = ring_name == name ? prepared_exchanges : metadata;
          roots.insert(roots.end(), selected.begin(), selected.end());
        }
        (void)FieldExchangeExpressions::encode_roots(roots, field_exchanges_->limits());
      }
      ExactContractBuilder proof;
      const auto clock = manager.clock_identity.find(name);
      proof.text("pops.program.history-publication")
          .text(name)
          .text(clock == manager.clock_identity.end() ? std::string_view{} : clock->second)
          .scalar(count);
      if (field_exchanges_->limits().max_bases)
        proof.text(
            FieldExchangeExpressions::collective_identity(expression, field_exchanges_->limits()));
      proof.presence(incoming.support != nullptr).presence(incoming.representation != nullptr);
      if (incoming.support) {
        const auto& source = *incoming.support;
        proof.text(source.program_identity)
            .text(source.origin_point)
            .scalar(source.program_owner)
            .scalar(source.producer_level)
            .scalar(source.topology_epoch)
            .scalar(source.materialization_generation)
            .text(source.field_provider);
      }
      if (incoming.representation)
        proof.text(incoming.representation->contract).text(incoming.representation->origin);
      for (std::size_t slot = 0; slot < prepared_samples.size(); ++slot) {
        const auto& identity = prepared_samples[slot];
        proof.scalar(identity.kind)
            .scalar(identity.start_bits)
            .scalar(identity.interval_bits)
            .scalar(identity.ordinal)
            .scalar(prepared_dts[slot]);
      }
      contract = std::move(proof).release();
    } catch (...) {
      error = std::current_exception();
    }
    const auto& lane = prepared_execution_lane();
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("Program history publication preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"pops.program.history", contract}}, lane))
      throw std::invalid_argument("Program history publication identity differs between ranks");
    static_assert(std::is_nothrow_swappable_v<field_type>);
    auto& ring = manager.histories.at(name);
    for (const auto& slot : ring)
      program_values_.revoke_buffer_(slot);
    for (std::size_t slot = 0; slot < prepared_fields.size(); ++slot)
      std::swap(ring[slot], prepared_fields[slot]);
    manager.slot_dt.at(name).swap(prepared_dts);
    manager.slot_sample.at(name).swap(prepared_samples);
    manager.slot_exchanges.at(name).swap(prepared_exchanges);
    manager.slot_fields.at(name).swap(prepared_metadata);
    manager.initialized.at(name) = true;
    manager.store_pending.at(name) = true;
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
  mutable ::pops::amr::Rational stage_time_{0, 1};
  mutable int auxiliary_evaluation_sequence_ = 0;
  mutable ::pops::amr::Rational logical_phase_begin_{0, 1};
  mutable ::pops::amr::Rational logical_phase_span_{1, 1};
  mutable double logical_physical_time_offset_ = 0.0;
  mutable std::string primary_clock_;
  mutable ClockScheduleState clock_schedule_;
  mutable std::map<ScratchKey, field_type> scratch_;
  mutable PreparedResourceCache prepared_resources_;
  mutable ProgramValueAuthority<Dim> program_values_;
  std::shared_ptr<FieldExchangeExpressions> field_exchanges_ =
      std::make_shared<FieldExchangeExpressions>();
  // This live authority object is never part of accepted-state snapshots. A weak callback
  // survives artifact rollback without borrowing a Context address (stack views are valid).
  struct AlgebraLifetime {
    ContinuationAuthority<Dim> continuation;
    FiniteDomainRegistry finite;
    FiniteDomainRegistry local_finite{false};
    ContinuationTransportSession<Dim> transport;
    std::shared_ptr<distributed::Registry> distributed;
    void restart_committed(const FiniteDomainRealmRef& keep,
                           const std::shared_ptr<distributed::Registry>& restored) noexcept {
      continuation.revoke();
      finite.revoke_on_restart(keep);
      local_finite.revoke_on_restart();
      // The prepared Store already validated the restored budget and ownership.
      // Replacing this shared owner releases the old issuer, never adopts it.
      distributed = restored;
      transport.clear();
    }
  };
  const std::shared_ptr<AlgebraLifetime> algebra_lifetime_ = std::make_shared<AlgebraLifetime>();
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
