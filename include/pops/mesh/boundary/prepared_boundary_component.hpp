#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {
namespace boundary_phase_detail {

struct StepRejectionEnvelope {
  SolveStatus status = SolveStatus::kInvalidEvaluation;
  StepAttemptDisposition disposition = StepAttemptDisposition::kReject;
  std::uint32_t reason_code = 0;
  std::string phase;
  std::string detail;
};

inline void append_u64(std::string& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
}

inline std::uint64_t read_u64(std::string_view bytes, std::size_t& cursor) {
  if (cursor > bytes.size() || bytes.size() - cursor < 8)
    throw std::runtime_error("collective boundary step rejection envelope is truncated");
  std::uint64_t value = 0;
  for (int byte = 0; byte < 8; ++byte)
    value = (value << 8u) | static_cast<unsigned char>(bytes[cursor++]);
  return value;
}

inline void append_text(std::string& bytes, std::string_view value) {
  append_u64(bytes, static_cast<std::uint64_t>(value.size()));
  bytes.append(value.data(), value.size());
}

inline std::string read_text(std::string_view bytes, std::size_t& cursor) {
  const std::uint64_t encoded_size = read_u64(bytes, cursor);
  if (encoded_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("collective boundary step rejection text exceeds size_t");
  const std::size_t size = static_cast<std::size_t>(encoded_size);
  if (cursor > bytes.size() || size > bytes.size() - cursor)
    throw std::runtime_error("collective boundary step rejection text is truncated");
  std::string value(bytes.substr(cursor, size));
  cursor += size;
  return value;
}

inline std::string encode_step_rejection(const StepAttemptRejected& rejected) {
  std::string bytes("pops.boundary-step-rejection.v1");
  append_u64(bytes, static_cast<std::uint64_t>(rejected.status()));
  append_u64(bytes, static_cast<std::uint64_t>(rejected.disposition()));
  append_u64(bytes, rejected.reason_code());
  append_text(bytes, rejected.phase());
  append_text(bytes, rejected.detail());
  return bytes;
}

inline StepRejectionEnvelope decode_step_rejection(std::string_view bytes) {
  constexpr std::string_view prefix = "pops.boundary-step-rejection.v1";
  if (!bytes.starts_with(prefix))
    throw std::runtime_error("collective boundary step rejection has another schema");
  std::size_t cursor = prefix.size();
  const std::uint64_t status = read_u64(bytes, cursor);
  const std::uint64_t disposition = read_u64(bytes, cursor);
  const std::uint64_t reason_code = read_u64(bytes, cursor);
  if (status > static_cast<std::uint64_t>(SolveStatus::kSafeguardFailure) ||
      disposition > static_cast<std::uint64_t>(StepAttemptDisposition::kReject) ||
      reason_code > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("collective boundary step rejection has invalid typed fields");
  StepRejectionEnvelope result;
  result.status = static_cast<SolveStatus>(status);
  result.disposition = static_cast<StepAttemptDisposition>(disposition);
  result.reason_code = static_cast<std::uint32_t>(reason_code);
  result.phase = read_text(bytes, cursor);
  result.detail = read_text(bytes, cursor);
  if (cursor != bytes.size() || result.phase.empty())
    throw std::runtime_error("collective boundary step rejection envelope is incomplete");
  return result;
}

[[noreturn]] inline void throw_collective_step_rejection(const ExecutionLane& lane,
                                                         const std::string& local_payload,
                                                         long locally_rejected) {
  std::string selected_payload;
  if (all_reduce_min(locally_rejected, lane) != 0) {
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"boundary-step-rejection", local_payload}},
                                                  lane))
      throw std::runtime_error("collective boundary step rejection fields differ between ranks");
    selected_payload = local_payload;
  } else {
    const long local_root =
        locally_rejected != 0 ? static_cast<long>(lane.rank()) : static_cast<long>(lane.size());
    const long root = all_reduce_min(local_root, lane);
    if (root < 0 || root >= static_cast<long>(lane.size()))
      throw std::runtime_error("collective boundary step rejection lost its typed envelope");
    const bool authoritative = lane.rank() == root;
    const long invalid_length =
        authoritative &&
                local_payload.size() > static_cast<std::size_t>(std::numeric_limits<long>::max())
            ? 1L
            : 0L;
    if (all_reduce_max(invalid_length, lane) != 0)
      throw std::length_error("collective boundary step rejection exceeds long capacity");
    const long encoded_length =
        all_reduce_max(authoritative ? static_cast<long>(local_payload.size()) : 0L, lane);
    if (encoded_length <= 0)
      throw std::runtime_error("collective boundary step rejection envelope is empty");
    long allocation_failed = 0;
    try {
      if (authoritative)
        selected_payload = local_payload;
      selected_payload.resize(static_cast<std::size_t>(encoded_length));
    } catch (...) {
      allocation_failed = 1;
    }
    if (all_reduce_max(allocation_failed, lane) != 0)
      throw std::bad_alloc();
    broadcast_bytes_inplace(selected_payload.data(), selected_payload.size(), lane,
                            static_cast<int>(root));
    const long mismatch = locally_rejected != 0 && local_payload != selected_payload ? 1L : 0L;
    if (all_reduce_max(mismatch, lane) != 0)
      throw std::runtime_error(
          "collective boundary step rejection fields differ between rejecting ranks");
  }
  const StepRejectionEnvelope envelope = decode_step_rejection(selected_payload);
  throw StepAttemptRejected(envelope.status, envelope.disposition, envelope.reason_code,
                            envelope.phase, envelope.detail);
}

}  // namespace boundary_phase_detail

/// Run one rank-symmetric boundary-provider phase on its exact prepared lane. Native component
/// callbacks receive only their noncollective patch authority; this outer gate converges failures
/// before another provider is allowed to enter dependency collectives. Typed retry/reject control is
/// reproduced on every rank so the enclosing transaction can roll back and rethrow it unchanged.
template <class Operation>
void collective_boundary_provider_phase(const ExecutionLane& lane, std::string_view failure_message,
                                        Operation&& operation) {
  enum class ExceptionKind : long { none = 0, step_rejected = 1, ordinary = 2 };
  ExceptionKind kind = ExceptionKind::none;
  std::string rejection_payload;
  std::exception_ptr local_error;
  try {
    std::forward<Operation>(operation)();
    ::pops::device_fence();
  } catch (const StepAttemptRejected& rejected) {
    try {
      rejection_payload = boundary_phase_detail::encode_step_rejection(rejected);
      kind = ExceptionKind::step_rejected;
    } catch (...) {
      kind = ExceptionKind::ordinary;
      local_error = std::current_exception();
    }
  } catch (...) {
    kind = ExceptionKind::ordinary;
    local_error = std::current_exception();
  }

  const long ordinary = kind == ExceptionKind::ordinary ? 1L : 0L;
  const long rejected = kind == ExceptionKind::step_rejected ? 1L : 0L;
  if (all_reduce_max(ordinary, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(failure_message));
  }
  if (all_reduce_max(rejected, lane) == 0)
    return;
  boundary_phase_detail::throw_collective_step_rejection(lane, rejection_payload, rejected);
}

}  // namespace pops::runtime::program

namespace pops {

struct PreparedBoundaryRegion {
  PopsBoundaryRegionKindV1 kind = POPS_BOUNDARY_FACE_V1;
  int dimension = 0;
  int codimension = 1;
  std::vector<std::int32_t> axes;
  std::vector<std::int32_t> sides;
  std::string identity;

  PopsBoundaryRegionV1 view() const {
    return {sizeof(PopsBoundaryRegionV1),
            kind,
            dimension,
            codimension,
            axes.size(),
            axes.data(),
            sides.data(),
            identity.c_str()};
  }
};

struct PreparedBoundaryComponentSpec {
  std::string target_identity;
  std::string component_id;
  std::string manifest_identity;
  std::uint32_t interface_version = 1;
  std::string producer_identity;
  std::string state_identity;
  std::string ghost_identity;
  std::string layout_identity;
  PreparedBoundaryRegion region;
  std::vector<std::string> states;
  std::vector<std::string> directions;
  std::vector<std::string> fields;
  std::vector<std::string> parameter_ids;
  std::vector<double> parameter_values;
  std::vector<std::string> outputs;
  std::string rate;
  std::string nonlinear_iterate;
  std::string parameters_json;
  std::string target_json;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

/// Append the rank-symmetric authored component contract. Native handles are intentionally absent:
/// execution identity, backend/precision and communicator names are collective facts, while MPI and
/// stream handles are process-local representations of those authenticated resources.
inline void append_prepared_boundary_component_contract(ExactContractBuilder& contract,
                                                        const PreparedBoundaryComponentSpec& spec) {
  if (!spec.execution)
    throw std::logic_error("prepared boundary provider has no execution authority");
  contract.text(spec.target_identity)
      .text(spec.component_id)
      .text(spec.manifest_identity)
      .scalar(spec.interface_version)
      .text(spec.producer_identity)
      .text(spec.state_identity)
      .text(spec.ghost_identity)
      .text(spec.layout_identity)
      .scalar(static_cast<std::int32_t>(spec.region.kind))
      .scalar(std::int32_t{spec.region.dimension})
      .scalar(std::int32_t{spec.region.codimension})
      .sequence(spec.region.axes)
      .sequence(spec.region.sides)
      .text(spec.region.identity)
      .sequence(spec.states,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(spec.directions,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(spec.fields,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(spec.parameter_ids,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(spec.parameter_values)
      .sequence(spec.outputs,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .text(spec.rate)
      .text(spec.nonlinear_iterate)
      .text(spec.parameters_json)
      .text(spec.target_json);
  const PopsExecutionContextV1 execution = spec.execution->view();
  contract.text(spec.execution->identity())
      .scalar(execution.context_version)
      .scalar(execution.memory_space)
      .text(execution.backend_identity)
      .text(execution.device_identity)
      .scalar(execution.scalar_type)
      .scalar(execution.storage_precision)
      .scalar(execution.compute_precision)
      .scalar(execution.accumulation_precision)
      .scalar(execution.reduction_precision)
      .text(execution.stream_identity)
      .text(execution.communicator_identity)
      .text(execution.communicator_datatype_identity);
}

enum class PreparedBoundaryOperation { GhostRegion, FluxTransform, FieldResidual, FieldJvp };

/// One statically typed prepared component invocation.  The operation is a template argument, never
/// a production string branch: installation chooses one typed entry point and scientific calls retain
/// its direct ABI table, state and exact execution context.
template <PreparedBoundaryOperation Operation>
class PreparedBoundaryComponent final {
 private:
  template <int Dim>
  struct InvocationScratch;

 public:
  class Session;

  /// Purely local, fully allocated session image.  Construction authenticates the lane projection
  /// and allocates every host/device scratch table, but deliberately does not enter the component's
  /// prepare callback.  The owner converges local construction failures on the exact lane before it
  /// calls ``finish_session`` in canonical provider order.
  template <int Dim>
  class LocalSessionCandidate final {
   public:
    LocalSessionCandidate(const LocalSessionCandidate&) = delete;
    LocalSessionCandidate& operator=(const LocalSessionCandidate&) = delete;
    LocalSessionCandidate(LocalSessionCandidate&&) noexcept = default;
    LocalSessionCandidate& operator=(LocalSessionCandidate&&) noexcept = default;
    ~LocalSessionCandidate() = default;

   private:
    friend class PreparedBoundaryComponent;
    friend class Session;

    LocalSessionCandidate(PreparedBoundaryComponentSpec spec,
                          std::shared_ptr<component::LoadedComponent> component,
                          std::shared_ptr<const component::PreparedExecutionContextV1> execution,
                          component::PreparedExecutionContextV1 patch_execution,
                          std::unique_ptr<std::mutex> invocation_mutex,
                          std::shared_ptr<void> scratch,
                          component::LoadedComponent::PreparedStateRequest prepare_request)
        : spec_(std::move(spec)),
          component_(std::move(component)),
          execution_(std::move(execution)),
          patch_execution_(std::move(patch_execution)),
          invocation_mutex_(std::move(invocation_mutex)),
          invocation_scratch_(std::move(scratch)),
          prepare_request_(std::move(prepare_request)) {}

    PreparedBoundaryComponentSpec spec_;
    std::shared_ptr<component::LoadedComponent> component_;
    std::shared_ptr<const component::PreparedExecutionContextV1> execution_;
    component::PreparedExecutionContextV1 patch_execution_;
    std::unique_ptr<std::mutex> invocation_mutex_;
    std::shared_ptr<void> invocation_scratch_;
    component::LoadedComponent::PreparedStateRequest prepare_request_;
  };

  /// One lane-bound invocation session with an independently prepared component state.
  class Session final {
   public:
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) noexcept = default;
    Session& operator=(Session&&) noexcept = default;
    ~Session() = default;

    [[nodiscard]] const PreparedBoundaryComponentSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] void* state() const noexcept { return state_.get(); }
    [[nodiscard]] const component::PreparedExecutionContextV1& execution() const noexcept {
      return *execution_;
    }
    [[nodiscard]] const component::PreparedExecutionContextV1& patch_execution() const noexcept {
      return patch_execution_;
    }
    /// The operation owner holds this non-recursive guard across dependency materialization and
    /// the typed callback.  Keeping both phases under one guard prevents a concurrent invocation
    /// from retargeting the preallocated pointer rows while native code borrows them.
    [[nodiscard]] std::mutex& invocation_mutex() noexcept { return *invocation_mutex_; }

    [[nodiscard]] const PopsGhostBoundaryApiV1& ghost_api() const {
      static_assert(Operation == PreparedBoundaryOperation::GhostRegion);
      return component_->table<PopsGhostBoundaryApiV1>(POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,
                                                       spec_.interface_version);
    }

    [[nodiscard]] const PopsBoundaryFluxApiV1& boundary_flux_api() const {
      static_assert(Operation == PreparedBoundaryOperation::FluxTransform);
      return component_->table<PopsBoundaryFluxApiV1>(POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1,
                                                      spec_.interface_version);
    }

    [[nodiscard]] const PopsFieldBoundaryClosureApiV1& field_api() const {
      static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                    Operation == PreparedBoundaryOperation::FieldJvp);
      return component_->table<PopsFieldBoundaryClosureApiV1>(
          POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1, spec_.interface_version);
    }

   private:
    friend class PreparedBoundaryComponent;

    template <int Dim>
    Session(LocalSessionCandidate<Dim>&& candidate,
            component::LoadedComponent::PreparedState state) noexcept
        : spec_(std::move(candidate.spec_)),
          component_(std::move(candidate.component_)),
          execution_(std::move(candidate.execution_)),
          patch_execution_(std::move(candidate.patch_execution_)),
          invocation_mutex_(std::move(candidate.invocation_mutex_)),
          invocation_scratch_(std::move(candidate.invocation_scratch_)),
          scratch_dimension_(Dim),
          state_(std::move(state)) {}

    PreparedBoundaryComponentSpec spec_;
    // Declaration order is intentional: state_ is destroyed before the execution strings and the
    // LoadedComponent that keeps its destroy callback's dynamic library resident.
    std::shared_ptr<component::LoadedComponent> component_;
    std::shared_ptr<const component::PreparedExecutionContextV1> execution_;
    component::PreparedExecutionContextV1 patch_execution_;
    std::unique_ptr<std::mutex> invocation_mutex_;
    std::shared_ptr<void> invocation_scratch_;
    int scratch_dimension_ = 0;
    component::LoadedComponent::PreparedState state_;
  };

  PreparedBoundaryComponent(PreparedBoundaryComponentSpec spec,
                            std::shared_ptr<component::LoadedComponent> component)
      : spec_(std::move(spec)), component_(std::move(component)) {
    validate();
  }

  PreparedBoundaryComponent(const PreparedBoundaryComponent&) = delete;
  PreparedBoundaryComponent& operator=(const PreparedBoundaryComponent&) = delete;

  ~PreparedBoundaryComponent() = default;

  const PreparedBoundaryComponentSpec& spec() const { return spec_; }
  /// Stable identity of the authenticated loaded package retained by this prepared operation.
  /// Residual/JVP pair installation uses it to reject two independently loaded providers that
  /// merely advertise colliding textual identities.
  [[nodiscard]] const component::LoadedComponent* package_owner_identity() const noexcept {
    return component_.get();
  }

  /// Allocate and authenticate every host-owned session resource without entering provider code or
  /// any collective.  Callers must exact-lane-converge failure before invoking ``finish_session``.
  template <int Dim>
  [[nodiscard]] LocalSessionCandidate<Dim> make_local_session_candidate(
      const ExecutionLane& lane, const MultiFab<Dim>& prototype, const Geometry<Dim>& geometry,
      const FieldBoundaryExecutionContext<Dim>& context) const {
    auto execution = std::make_shared<const component::PreparedExecutionContextV1>(
        spec_.execution->for_lane(lane));
    if (!execution->matches_lane(lane))
      throw std::invalid_argument(
          "prepared boundary component execution authority differs from its lane");
    if (execution->view().memory_space != POPS_MEMORY_SPACE_HOST_V1)
      throw std::invalid_argument(
          "prepared boundary component uses the host-batch ABI but its exact ExecutionContext "
          "requires a non-host memory space; install a device-native boundary provider instead");
    auto scratch = std::make_shared<InvocationScratch<Dim>>(spec_, prototype, geometry, context);
    PreparedBoundaryComponentSpec candidate_spec = spec_;
    candidate_spec.execution = execution;
    auto patch_execution = execution->without_collective_authority();
    auto invocation_mutex = std::make_unique<std::mutex>();
    auto prepare_request = component_->prepare_state_request(
        native_interface_id_(), spec_.interface_version, execution->view(),
        candidate_spec.parameters_json, candidate_spec.target_json);
    return LocalSessionCandidate<Dim>(std::move(candidate_spec), component_, std::move(execution),
                                      std::move(patch_execution), std::move(invocation_mutex),
                                      std::move(scratch), std::move(prepare_request));
  }

  /// Enter exactly one provider prepare callback after the owner has converged every local
  /// candidate allocation.  No host allocation follows the callback: Session construction only
  /// moves the already-owned image, so the caller can immediately converge the provider result
  /// before advancing to another provider.
  template <int Dim>
  [[nodiscard]] Session finish_session(LocalSessionCandidate<Dim>&& candidate) const {
    auto state =
        candidate.component_->execute_prepared_state(std::move(candidate.prepare_request_));
    return Session(std::move(candidate), std::move(state));
  }

  const PopsGhostBoundaryApiV1& ghost_api() const {
    static_assert(Operation == PreparedBoundaryOperation::GhostRegion);
    return component_->table<PopsGhostBoundaryApiV1>(POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,
                                                     spec_.interface_version);
  }

  const PopsBoundaryFluxApiV1& boundary_flux_api() const {
    static_assert(Operation == PreparedBoundaryOperation::FluxTransform);
    return component_->table<PopsBoundaryFluxApiV1>(POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1,
                                                    spec_.interface_version);
  }

  const PopsFieldBoundaryClosureApiV1& field_api() const {
    static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                  Operation == PreparedBoundaryOperation::FieldJvp);
    return component_->table<PopsFieldBoundaryClosureApiV1>(
        POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1, spec_.interface_version);
  }

 private:
  static std::size_t checked_buffer_size_(std::size_t points, std::size_t components) {
    if (components != 0 && points > std::numeric_limits<std::size_t>::max() / components)
      throw std::length_error("prepared boundary component scratch size overflow");
    return points * components;
  }

  template <int Dim, class Function>
  static void for_each_host_index_(const Box<Dim>& box, Function&& function) {
    const std::int64_t signed_points = box.numPts();
    if (signed_points < 0)
      throw std::overflow_error("prepared boundary component region size overflow");
    const std::size_t points = static_cast<std::size_t>(signed_points);
    for (std::size_t linear = 0; linear < points; ++linear) {
      std::size_t remainder = linear;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t extent = static_cast<std::size_t>(box.length(axis));
        index[axis] = box.lo[axis] + static_cast<int>(remainder % extent);
        remainder /= extent;
      }
      function(index, linear);
    }
  }

  template <int Dim>
  struct InvocationScratch {
    using host_type = typename Fab<Dim>::raw_host_mirror_type;

    struct Patch {
      Patch() = default;

      Patch(const Fab<Dim>& prototype, std::string identity, std::size_t points,
            std::size_t components, std::span<const MultiFab<Dim>* const> dependencies,
            std::size_t global_patch)
          : patch_identity(std::move(identity)),
            input_host("pops_boundary_input_host", prototype.size()),
            output_host("pops_boundary_output_host",
                        Operation == PreparedBoundaryOperation::GhostRegion ||
                                Operation == PreparedBoundaryOperation::FieldResidual ||
                                Operation == PreparedBoundaryOperation::FieldJvp
                            ? prototype.size()
                            : 0),
            direction_host("pops_boundary_direction_host",
                           Operation == PreparedBoundaryOperation::FieldJvp ? prototype.size() : 0),
            interior(Operation == PreparedBoundaryOperation::GhostRegion
                         ? checked_buffer_size_(points, components)
                         : 0),
            output(Operation == PreparedBoundaryOperation::GhostRegion
                       ? checked_buffer_size_(points, components)
                       : 0),
            coordinates(checked_buffer_size_(points, static_cast<std::size_t>(Dim))),
            offsets(checked_buffer_size_(points, components)),
            base(Operation == PreparedBoundaryOperation::FluxTransform
                     ? checked_buffer_size_(points, components)
                     : 0),
            transformed(Operation == PreparedBoundaryOperation::FluxTransform
                            ? checked_buffer_size_(points, components)
                            : 0),
            normals(Operation == PreparedBoundaryOperation::FluxTransform
                        ? checked_buffer_size_(points, static_cast<std::size_t>(Dim))
                        : 0),
            measures(Operation == PreparedBoundaryOperation::FluxTransform ? points : 0),
            state_values(Operation == PreparedBoundaryOperation::FluxTransform ||
                                 Operation == PreparedBoundaryOperation::FieldResidual ||
                                 Operation == PreparedBoundaryOperation::FieldJvp
                             ? checked_buffer_size_(points, components)
                             : 0),
            direction_values(Operation == PreparedBoundaryOperation::FieldJvp
                                 ? checked_buffer_size_(points, components)
                                 : 0),
            contribution(Operation == PreparedBoundaryOperation::FieldResidual ||
                                 Operation == PreparedBoundaryOperation::FieldJvp
                             ? checked_buffer_size_(points, components)
                             : 0),
            actions(Operation == PreparedBoundaryOperation::FluxTransform ? points : 0),
            dependency_hosts(dependencies.size()),
            dependency_values(dependencies.size()),
            dependency_rows(Operation == PreparedBoundaryOperation::GhostRegion ||
                                    Operation == PreparedBoundaryOperation::FluxTransform
                                ? dependencies.size()
                                : 0) {
        for (std::size_t dependency = 0; dependency < dependencies.size(); ++dependency) {
          const MultiFab<Dim>* field = dependencies[dependency];
          if (field != nullptr && !field->contains_local(global_patch))
            throw std::invalid_argument(
                "prepared boundary scratch dependency lacks its exact local patch");
          // InvocationState deliberately has no prepared image: it is supplied by the exact
          // state/iterate argument at callback time.  Its slot remains fully preallocated here
          // from the owner prototype so row ordering never changes on the hot path.
          const Fab<Dim>& fab =
              field == nullptr ? prototype : field->fab(field->local_index_of(global_patch));
          dependency_hosts[dependency] = host_type("pops_boundary_dependency_host", fab.size());
          dependency_values[dependency].resize(
              checked_buffer_size_(points, static_cast<std::size_t>(fab.ncomp())));
        }
      }

      std::string patch_identity;
      host_type input_host;
      host_type output_host;
      host_type direction_host;
      std::array<host_type, Dim> face_hosts{};
      std::vector<double> interior;
      std::vector<double> output;
      std::vector<double> coordinates;
      std::vector<std::size_t> offsets;
      std::vector<double> base;
      std::vector<double> transformed;
      std::vector<double> normals;
      std::vector<double> measures;
      std::vector<double> state_values;
      std::vector<double> direction_values;
      std::vector<double> contribution;
      std::vector<PopsComponentActionV1> actions;
      std::vector<host_type> dependency_hosts;
      std::vector<std::vector<double>> dependency_values;
      std::vector<PopsQualifiedConstFieldV1> dependency_rows;
      std::vector<PopsQualifiedConstFieldV1> state_rows;
      std::vector<PopsQualifiedConstFieldV1> field_rows;
      std::array<std::size_t, 3> extents{1, 1, 1};
      std::array<std::ptrdiff_t, 3> strides{0, 0, 0};
      std::size_t points = 0;
      bool active = false;
    };

    InvocationScratch(const PreparedBoundaryComponentSpec& spec, const MultiFab<Dim>& prototype,
                      const Geometry<Dim>& geometry,
                      const FieldBoundaryExecutionContext<Dim>& context)
        : layout(prototype.layout()),
          distribution(prototype.distribution()),
          local_rank(prototype.local_rank()),
          ncomp(prototype.ncomp()),
          ghosts(prototype.ghosts()),
          prepared_geometry(geometry) {
      parameters.reserve(spec.parameter_ids.size());
      for (std::size_t index = 0; index < spec.parameter_ids.size(); ++index)
        parameters.push_back({sizeof(PopsQualifiedScalarV1), spec.parameter_ids[index].c_str(),
                              spec.parameter_values[index]});

      dependencies.reserve(spec.states.size() + spec.fields.size());
      dependency_components.reserve(spec.states.size() + spec.fields.size());
      auto append = [&](const std::string& identity, bool field) {
        const MultiFab<Dim>* dependency = nullptr;
        const int count = field ? context.field_count : context.state_count;
        const MultiFab<Dim>* const* values = field ? context.fields : context.states;
        const std::string* identities = field ? context.field_identities : context.state_identities;
        if (count < 0 || (count != 0 && (values == nullptr || identities == nullptr)))
          throw std::invalid_argument("prepared boundary scratch dependency table is incomplete");
        for (int index = 0; index < count; ++index)
          if (identities[index] == identity) {
            dependency = values[index];
            break;
          }
        const bool invocation_state = !field && identity == spec.state_identity;
        if ((!invocation_state && dependency == nullptr) ||
            (dependency != nullptr && dependency->ncomp() < 1))
          throw std::invalid_argument("prepared boundary scratch dependency is unavailable");
        dependencies.push_back(dependency);
        dependency_components.push_back(dependency == nullptr ? prototype.ncomp()
                                                              : dependency->ncomp());
      };
      for (const std::string& identity : spec.states)
        append(identity, false);
      for (const std::string& identity : spec.fields)
        append(identity, true);

      patches.resize(prototype.local_size());
      for (std::size_t local = 0; local < prototype.local_size(); ++local) {
        const Fab<Dim>& fab = prototype.fab(local);
        const Box<Dim>& valid = fab.box();
        bool touches = true;
        for (std::size_t boundary_axis = 0; boundary_axis < spec.region.axes.size();
             ++boundary_axis) {
          const int axis = spec.region.axes[boundary_axis];
          touches = touches && (spec.region.sides[boundary_axis] < 0
                                    ? valid.lo[axis] == geometry.domain().lo[axis]
                                    : valid.hi[axis] == geometry.domain().hi[axis]);
        }
        if (!touches)
          continue;
        Box<Dim> operation_region = valid;
        for (std::size_t boundary_axis = 0; boundary_axis < spec.region.axes.size();
             ++boundary_axis) {
          const int axis = spec.region.axes[boundary_axis];
          if constexpr (Operation == PreparedBoundaryOperation::GhostRegion) {
            const int depth = prototype.ghosts()[axis];
            if (depth < 1)
              throw std::invalid_argument(
                  "prepared ghost boundary scratch requires positive physical depth");
            if (spec.region.sides[boundary_axis] < 0) {
              operation_region.lo[axis] = valid.lo[axis] - depth;
              operation_region.hi[axis] = valid.lo[axis] - 1;
            } else {
              operation_region.lo[axis] = valid.hi[axis] + 1;
              operation_region.hi[axis] = valid.hi[axis] + depth;
            }
          } else {
            operation_region.lo[axis] = operation_region.hi[axis] =
                spec.region.sides[boundary_axis] < 0 ? valid.lo[axis] : valid.hi[axis];
          }
        }
        const std::int64_t signed_points = operation_region.numPts();
        if (signed_points < 1)
          throw std::invalid_argument("prepared boundary scratch patch is empty");
        const std::size_t points = static_cast<std::size_t>(signed_points);
        patches[local] = Patch(
            fab, spec.region.identity + "::patch:" + std::to_string(prototype.global_index(local)),
            points, static_cast<std::size_t>(prototype.ncomp()), dependencies,
            prototype.global_index(local));
        Patch& patch = patches[local];
        if constexpr (Operation == PreparedBoundaryOperation::FieldResidual ||
                      Operation == PreparedBoundaryOperation::FieldJvp) {
          patch.state_rows.resize(spec.states.size());
          patch.field_rows.resize(spec.fields.size());
        }
        if constexpr (Operation == PreparedBoundaryOperation::FluxTransform) {
          patch.face_hosts[0] = host_type(
              "pops_boundary_face_host",
              checked_buffer_size_(static_cast<std::size_t>(nd::face_box<0>(fab.box()).numPts()),
                                   static_cast<std::size_t>(prototype.ncomp())));
        }
        if constexpr (Operation == PreparedBoundaryOperation::FluxTransform && Dim >= 2)
          patch.face_hosts[1] = host_type(
              "pops_boundary_face_host",
              checked_buffer_size_(static_cast<std::size_t>(nd::face_box<1>(fab.box()).numPts()),
                                   static_cast<std::size_t>(prototype.ncomp())));
        if constexpr (Operation == PreparedBoundaryOperation::FluxTransform && Dim >= 3)
          patch.face_hosts[2] = host_type(
              "pops_boundary_face_host",
              checked_buffer_size_(static_cast<std::size_t>(nd::face_box<2>(fab.box()).numPts()),
                                   static_cast<std::size_t>(prototype.ncomp())));
      }
    }

    std::vector<Patch> patches;
    std::vector<PopsQualifiedScalarV1> parameters;
    std::vector<const MultiFab<Dim>*> dependencies;
    std::vector<int> dependency_components;
    typename MultiFab<Dim>::layout_type layout;
    typename MultiFab<Dim>::distribution_type distribution;
    typename MultiFab<Dim>::rank_type local_rank{};
    int ncomp = 0;
    typename MultiFab<Dim>::ghost_type ghosts{};
    Geometry<Dim> prepared_geometry;
  };

  template <int Dim>
  static InvocationScratch<Dim>& invocation_scratch_(Session& session, const MultiFab<Dim>& field,
                                                     const Geometry<Dim>& geometry) {
    if (session.scratch_dimension_ != Dim || !session.invocation_scratch_)
      throw std::logic_error("prepared boundary component lacks exact-ranked invocation scratch");
    auto& scratch = *static_cast<InvocationScratch<Dim>*>(session.invocation_scratch_.get());
    if (scratch.patches.size() != field.local_size() || scratch.layout != field.layout() ||
        scratch.distribution != field.distribution() || scratch.local_rank != field.local_rank() ||
        scratch.ncomp != field.ncomp() || scratch.ghosts != field.ghosts() ||
        scratch.prepared_geometry != geometry)
      throw std::invalid_argument("prepared boundary component scratch layout changed");
    return scratch;
  }

 public:
  /// Execute one exact-ranked physical ghost region through the generated ABI. All component
  /// calls write host scratch first; the runtime field is published only after every local patch
  /// succeeds. The enclosing System boundary transaction supplies collective rollback if a later
  /// compiled-boundary operation fails.
  template <int Dim>
  void apply_ghost_region(Session& session,
                          const runtime::multiblock::BoundaryEvaluationPoint& point,
                          MultiFab<Dim>& state, const Geometry<Dim>& geometry,
                          const ExecutionLane& lane,
                          const FieldBoundaryExecutionContext<Dim>& context) const
    requires(Operation == PreparedBoundaryOperation::GhostRegion)
  {
    if (session.component_.get() != component_.get())
      throw std::invalid_argument("prepared ghost boundary session belongs to another provider");
    static_assert(Dim >= 1 && Dim <= 3);
    static_assert(sizeof(Real) == sizeof(double),
                  "GhostBoundary ABI v1 requires the binary64 PoPS backend");
    if (spec_.region.dimension != Dim || state.ncomp() < 1 || point.clock.empty() ||
        point.tick < 0 || point.level < 0 || point.substep < 0 || point.stage < 0 ||
        point.stage_fraction.denominator <= 0 || point.stage_fraction.numerator < 0 ||
        point.stage_fraction.numerator > point.stage_fraction.denominator || !(point.dt > 0.0) ||
        !std::isfinite(point.dt) || !std::isfinite(point.physical_time) ||
        context.point.level != point.level || context.point.step != point.tick ||
        context.clock_identity == nullptr || *context.clock_identity != point.clock ||
        !session.execution().matches_lane(lane) ||
        lane.size() != static_cast<int>(state.rank_space().size()) ||
        lane.rank() != static_cast<int>(state.rank_space().linear_rank(state.local_rank())))
      throw std::invalid_argument(
          "prepared ghost boundary received an incomplete exact-ranked invocation");
    if (!spec_.directions.empty() || spec_.outputs.size() != 1 ||
        spec_.outputs.front() != spec_.state_identity)
      throw std::invalid_argument(
          "GhostBoundary requires one primary-state output and no direction dependencies");

    auto& scratch = invocation_scratch_(session, state, geometry);
    for (auto& patch : scratch.patches)
      patch.active = false;
    const PopsLogicalTimeV1 logical_time{sizeof(PopsLogicalTimeV1),
                                         point.clock.c_str(),
                                         point.tick,
                                         point.level,
                                         point.substep,
                                         point.stage,
                                         point.stage_fraction.numerator,
                                         point.stage_fraction.denominator,
                                         point.dt,
                                         point.physical_time};
    for (std::size_t local = 0; local < state.local_size(); ++local) {
      Fab<Dim>& fab = state.fab(local);
      auto& patch = scratch.patches[local];
      const Box<Dim>& valid = fab.box();
      bool touches = true;
      for (std::size_t boundary_axis = 0; boundary_axis < spec_.region.axes.size();
           ++boundary_axis) {
        const int axis = spec_.region.axes[boundary_axis];
        const int side = spec_.region.sides[boundary_axis];
        touches = touches && (side < 0 ? valid.lo[axis] == geometry.domain().lo[axis]
                                       : valid.hi[axis] == geometry.domain().hi[axis]);
      }
      if (!touches)
        continue;

      Box<Dim> region = valid;
      for (std::size_t boundary_axis = 0; boundary_axis < spec_.region.axes.size();
           ++boundary_axis) {
        const int axis = spec_.region.axes[boundary_axis];
        const int side = spec_.region.sides[boundary_axis];
        const int depth = state.ghosts()[axis];
        if (depth < 1)
          throw std::invalid_argument(
              "prepared ghost boundary region requires positive allocated depth");
        if (side < 0) {
          region.lo[axis] = valid.lo[axis] - depth;
          region.hi[axis] = valid.lo[axis] - 1;
        } else {
          region.lo[axis] = valid.hi[axis] + 1;
          region.hi[axis] = valid.hi[axis] + depth;
        }
      }
      if (!fab.grown_box().contains(region))
        throw std::invalid_argument("prepared ghost boundary exact region exceeds its Fab storage");

      patch.extents = {1, 1, 1};
      patch.strides = {0, 0, 0};
      std::size_t points = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        patch.extents[axis] = static_cast<std::size_t>(region.length(axis));
        patch.strides[axis] = static_cast<std::ptrdiff_t>(points);
        points *= patch.extents[axis];
      }
      const std::size_t components = static_cast<std::size_t>(state.ncomp());
      const std::size_t values = checked_buffer_size_(points, components);
      if (patch.interior.size() < values || patch.output.size() < values ||
          patch.offsets.size() < values ||
          patch.coordinates.size() < checked_buffer_size_(points, Dim))
        throw std::invalid_argument("prepared ghost boundary exceeds its prepared scratch");
      Kokkos::deep_copy(patch.input_host, fab.storage());
      Kokkos::deep_copy(patch.output_host, fab.storage());
      const FieldView<const Real, Dim> storage = std::as_const(fab).view();
      for_each_host_index_(region, [&](const Index<Dim>& index, std::size_t point_index) {
        Index<Dim> source = index;
        for (std::size_t boundary_axis = 0; boundary_axis < spec_.region.axes.size();
             ++boundary_axis) {
          const int selected = spec_.region.axes[boundary_axis];
          source[selected] =
              spec_.region.sides[boundary_axis] < 0 ? valid.lo[selected] : valid.hi[selected];
        }
        std::size_t ghost_offset = 0;
        std::size_t source_offset = 0;
        for (int selected = 0; selected < Dim; ++selected) {
          ghost_offset += static_cast<std::size_t>(index[selected] - storage.origin[selected]) *
                          static_cast<std::size_t>(storage.strides[selected]);
          source_offset += static_cast<std::size_t>(source[selected] - storage.origin[selected]) *
                           static_cast<std::size_t>(storage.strides[selected]);
          patch.coordinates[point_index + static_cast<std::size_t>(selected) * points] =
              static_cast<double>(geometry.cell_coordinate(selected, index[selected]));
        }
        for (std::size_t component = 0; component < components; ++component) {
          const std::size_t packed = point_index + component * points;
          const std::size_t ghost = ghost_offset + component * storage.component_stride;
          const std::size_t source_value = source_offset + component * storage.component_stride;
          patch.interior[packed] = static_cast<double>(patch.input_host(source_value));
          patch.output[packed] = static_cast<double>(patch.input_host(ghost));
          patch.offsets[packed] = ghost;
        }
      });

      auto const_view = [&](const void* data, std::size_t component_count) {
        return PopsConstFieldViewV1{sizeof(PopsConstFieldViewV1),
                                    data,
                                    static_cast<std::uint32_t>(Dim),
                                    {patch.extents[0], patch.extents[1], patch.extents[2]},
                                    {patch.strides[0], patch.strides[1], patch.strides[2]},
                                    component_count,
                                    static_cast<std::ptrdiff_t>(points),
                                    POPS_FIELD_CENTERING_CELL_V1,
                                    0,
                                    {0, 0, 0},
                                    {0, 0, 0},
                                    POPS_SCALAR_FLOAT64_V1,
                                    POPS_MEMORY_SPACE_HOST_V1,
                                    spec_.layout_identity.c_str(),
                                    patch.patch_identity.c_str(),
                                    POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
      };
      const PopsConstFieldViewV1 interior_view = const_view(patch.interior.data(), components);
      const PopsConstFieldViewV1 coordinate_view =
          const_view(patch.coordinates.data(), static_cast<std::size_t>(Dim));
      std::size_t dependency_index = 0;
      auto append_dependency = [&](const std::string& identity, bool field) {
        const MultiFab<Dim>& dependency = !field && identity == spec_.state_identity
                                              ? state
                                              : *scratch.dependencies[dependency_index];
        pack_dependency_region(dependency, state.global_index(local), region,
                               patch.dependency_hosts[dependency_index],
                               patch.dependency_values[dependency_index]);
        patch.dependency_rows[dependency_index] = {
            sizeof(PopsQualifiedConstFieldV1), 1, identity.c_str(),
            const_view(patch.dependency_values[dependency_index].data(),
                       static_cast<std::size_t>(dependency.ncomp()))};
        ++dependency_index;
      };
      for (const std::string& identity : spec_.states)
        append_dependency(identity, false);
      for (const std::string& identity : spec_.fields)
        append_dependency(identity, true);
      PopsFieldViewV1 ghost_view{sizeof(PopsFieldViewV1),
                                 patch.output.data(),
                                 interior_view.dimension,
                                 {patch.extents[0], patch.extents[1], patch.extents[2]},
                                 {patch.strides[0], patch.strides[1], patch.strides[2]},
                                 components,
                                 static_cast<std::ptrdiff_t>(points),
                                 POPS_FIELD_CENTERING_CELL_V1,
                                 0,
                                 {0, 0, 0},
                                 {0, 0, 0},
                                 POPS_SCALAR_FLOAT64_V1,
                                 POPS_MEMORY_SPACE_HOST_V1,
                                 spec_.layout_identity.c_str(),
                                 patch.patch_identity.c_str(),
                                 POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
      PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                                   nullptr};
      const PopsGhostBoundaryRequestV1 request{sizeof(PopsGhostBoundaryRequestV1),
                                               spec_.producer_identity.c_str(),
                                               spec_.state_identity.c_str(),
                                               spec_.ghost_identity.c_str(),
                                               interior_view,
                                               ghost_view,
                                               coordinate_view,
                                               spec_.region.view(),
                                               patch.dependency_rows.size(),
                                               patch.dependency_rows.data(),
                                               scratch.parameters.size(),
                                               scratch.parameters.data(),
                                               logical_time,
                                               session.patch_execution().view()};
      const int code =
          component::apply_ghost_boundary(session.ghost_api(), session.state(), request, status);
      require_success(code, status, "apply_region_batch");
      if (std::any_of(patch.output.begin(), patch.output.begin() + values,
                      [](double value) { return !std::isfinite(value); }))
        throw std::runtime_error("native boundary component published a non-finite ghost value");
      patch.points = points;
      patch.active = true;
    }

    for (std::size_t local = 0; local < scratch.patches.size(); ++local) {
      auto& patch = scratch.patches[local];
      if (!patch.active)
        continue;
      const std::size_t values =
          checked_buffer_size_(patch.points, static_cast<std::size_t>(state.ncomp()));
      for (std::size_t value = 0; value < values; ++value)
        patch.output_host(patch.offsets[value]) = static_cast<Real>(patch.output[value]);
      Kokkos::deep_copy(state.fab(local).storage(), patch.output_host);
    }
  }

  /// Transform the already-materialized numerical flux on one physical face.  The generated
  /// operator remains the sole Riemann engine: this executor only converts its oriented face
  /// values to outward-normal form, invokes the typed component, validates a detached candidate,
  /// and publishes the transformed values back to the same face storage.
  template <int Dim>
  void transform_boundary_flux(Session& session,
                               const runtime::multiblock::BoundaryEvaluationPoint& point,
                               const MultiFab<Dim>& state, std::vector<nd::FaceField<Dim>>& faces,
                               const Geometry<Dim>& geometry, const ExecutionLane& lane,
                               const FieldBoundaryExecutionContext<Dim>& context) const
    requires(Operation == PreparedBoundaryOperation::FluxTransform)
  {
    if (session.component_.get() != component_.get())
      throw std::invalid_argument("prepared BoundaryFlux session belongs to another provider");
    static_assert(Dim >= 1 && Dim <= 3);
    static_assert(sizeof(Real) == sizeof(double));
    if (spec_.region.dimension != Dim || spec_.region.axes.size() != 1 ||
        spec_.region.sides.size() != 1 || faces.size() != state.local_size() ||
        !spec_.directions.empty() || point.clock.empty() || point.tick < 0 || point.level < 0 ||
        point.substep < 0 || point.stage < 0 || point.stage_fraction.denominator <= 0 ||
        point.stage_fraction.numerator < 0 ||
        point.stage_fraction.numerator > point.stage_fraction.denominator || !(point.dt > 0.0) ||
        !std::isfinite(point.dt) || !std::isfinite(point.physical_time) ||
        context.point.level != point.level || context.point.step != point.tick ||
        context.clock_identity == nullptr || *context.clock_identity != point.clock ||
        !session.execution().matches_lane(lane) ||
        lane.size() != static_cast<int>(state.rank_space().size()) ||
        lane.rank() != static_cast<int>(state.rank_space().linear_rank(state.local_rank())))
      throw std::invalid_argument(
          "prepared BoundaryFlux received an incomplete exact-ranked invocation");
    const int selected_axis = spec_.region.axes.front();
    const int side = spec_.region.sides.front();
    if (selected_axis < 0 || selected_axis >= Dim || (side != -1 && side != 1))
      throw std::invalid_argument("prepared BoundaryFlux has an invalid oriented face");
    auto& scratch = invocation_scratch_(session, state, geometry);
    for (auto& patch : scratch.patches)
      patch.active = false;

    const PopsLogicalTimeV1 logical_time{sizeof(PopsLogicalTimeV1),
                                         point.clock.c_str(),
                                         point.tick,
                                         point.level,
                                         point.substep,
                                         point.stage,
                                         point.stage_fraction.numerator,
                                         point.stage_fraction.denominator,
                                         point.dt,
                                         point.physical_time};
    for (std::size_t local = 0; local < faces.size(); ++local) {
      auto& patch = scratch.patches[local];
      const Box<Dim>& cells = faces[local].cell_box();
      if (side < 0 ? cells.lo[selected_axis] != geometry.domain().lo[selected_axis]
                   : cells.hi[selected_axis] != geometry.domain().hi[selected_axis])
        continue;

      auto invoke_axis = [&]<int Axis>() {
        if (selected_axis != Axis)
          return;
        auto& face_fab = faces[local].template field<Axis>();
        Box<Dim> region = face_fab.box();
        region.lo[Axis] = region.hi[Axis] =
            side < 0 ? geometry.domain().lo[Axis] : geometry.domain().hi[Axis] + 1;
        patch.extents = {1, 1, 1};
        patch.strides = {0, 0, 0};
        std::size_t points = 1;
        for (int axis = 0; axis < Dim; ++axis) {
          patch.extents[axis] = static_cast<std::size_t>(region.length(axis));
          patch.strides[axis] = static_cast<std::ptrdiff_t>(points);
          points *= patch.extents[axis];
        }
        const std::size_t components = static_cast<std::size_t>(face_fab.ncomp());
        const std::size_t values = checked_buffer_size_(points, components);
        if (patch.base.size() < values || patch.transformed.size() < values ||
            patch.offsets.size() < values || patch.measures.size() < points ||
            patch.coordinates.size() < checked_buffer_size_(points, Dim) ||
            patch.normals.size() < checked_buffer_size_(points, Dim) ||
            patch.state_values.size() <
                checked_buffer_size_(points, static_cast<std::size_t>(state.ncomp())))
          throw std::invalid_argument("prepared BoundaryFlux exceeds its prepared scratch");
        auto& face_host = patch.face_hosts[Axis];
        Kokkos::deep_copy(face_host, face_fab.storage());
        Kokkos::deep_copy(patch.input_host, state.fab(local).storage());
        const auto face_view = std::as_const(face_fab).view();
        const auto state_view = std::as_const(state).fab(local).view();
        const double face_measure = [&] {
          double value = 1.0;
          for (int axis = 0; axis < Dim; ++axis)
            if (axis != Axis)
              value *= static_cast<double>(geometry.spacing(axis));
          return value;
        }();

        std::fill_n(patch.normals.begin(), checked_buffer_size_(points, Dim), 0.0);
        std::fill_n(patch.measures.begin(), points, 1.0);
        for_each_host_index_(region, [&](const Index<Dim>& index, std::size_t packed_point) {
          Index<Dim> cell = index;
          cell[Axis] = side < 0 ? cells.lo[Axis] : cells.hi[Axis];
          std::size_t face_offset = 0;
          std::size_t state_offset = 0;
          for (int selected = 0; selected < Dim; ++selected) {
            face_offset += static_cast<std::size_t>(index[selected] - face_view.origin[selected]) *
                           static_cast<std::size_t>(face_view.strides[selected]);
            state_offset += static_cast<std::size_t>(cell[selected] - state_view.origin[selected]) *
                            static_cast<std::size_t>(state_view.strides[selected]);
            patch.coordinates[packed_point + static_cast<std::size_t>(selected) * points] =
                selected == Axis
                    ? static_cast<double>(geometry.lower()[selected]) +
                          static_cast<double>(index[selected] - geometry.domain().lo[selected]) *
                              static_cast<double>(geometry.spacing(selected))
                    : static_cast<double>(geometry.cell_coordinate(selected, index[selected]));
            patch.normals[packed_point + static_cast<std::size_t>(selected) * points] =
                selected == Axis ? static_cast<double>(side) : 0.0;
          }
          patch.measures[packed_point] = face_measure;
          for (std::size_t component = 0; component < components; ++component) {
            const std::size_t packed = packed_point + component * points;
            const std::size_t offset =
                face_offset + component * static_cast<std::size_t>(face_view.component_stride);
            patch.base[packed] = static_cast<double>(side) * static_cast<double>(face_host(offset));
            patch.transformed[packed] = patch.base[packed];
            patch.offsets[packed] = offset;
          }
          for (int component = 0; component < state.ncomp(); ++component)
            patch.state_values[packed_point + static_cast<std::size_t>(component) * points] =
                static_cast<double>(patch.input_host(
                    state_offset + static_cast<std::size_t>(component) *
                                       static_cast<std::size_t>(state_view.component_stride)));
        });

        auto const_view = [&](const void* data, std::size_t component_count,
                              PopsFieldCenteringV1 centering, std::uint32_t centering_axes) {
          return PopsConstFieldViewV1{sizeof(PopsConstFieldViewV1),
                                      data,
                                      static_cast<std::uint32_t>(Dim),
                                      {patch.extents[0], patch.extents[1], patch.extents[2]},
                                      {patch.strides[0], patch.strides[1], patch.strides[2]},
                                      component_count,
                                      static_cast<std::ptrdiff_t>(points),
                                      centering,
                                      centering_axes,
                                      {0, 0, 0},
                                      {0, 0, 0},
                                      POPS_SCALAR_FLOAT64_V1,
                                      POPS_MEMORY_SPACE_HOST_V1,
                                      spec_.layout_identity.c_str(),
                                      patch.patch_identity.c_str(),
                                      POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
        };
        const std::uint32_t centering_axes = 1u << static_cast<unsigned>(Axis);
        const PopsConstFieldViewV1 base_view =
            const_view(patch.base.data(), components, POPS_FIELD_CENTERING_FACE_V1, centering_axes);
        const PopsConstFieldViewV1 coordinate_view =
            const_view(patch.coordinates.data(), Dim, POPS_FIELD_CENTERING_FACE_V1, centering_axes);
        const PopsConstFieldViewV1 normal_view =
            const_view(patch.normals.data(), Dim, POPS_FIELD_CENTERING_FACE_V1, centering_axes);
        Box<Dim> dependency_region = region;
        dependency_region.lo[Axis] = dependency_region.hi[Axis] =
            side < 0 ? cells.lo[Axis] : cells.hi[Axis];
        std::size_t dependency_index = 0;
        auto append_dependency = [&](const std::string& identity, bool field) {
          const MultiFab<Dim>& dependency = !field && identity == spec_.state_identity
                                                ? state
                                                : *scratch.dependencies[dependency_index];
          if (!field && identity == spec_.state_identity)
            patch.dependency_rows[dependency_index] = {
                sizeof(PopsQualifiedConstFieldV1), 1, identity.c_str(),
                const_view(patch.state_values.data(), static_cast<std::size_t>(dependency.ncomp()),
                           POPS_FIELD_CENTERING_CELL_V1, 0)};
          else {
            pack_dependency_region(dependency, state.global_index(local), dependency_region,
                                   patch.dependency_hosts[dependency_index],
                                   patch.dependency_values[dependency_index]);
            patch.dependency_rows[dependency_index] = {
                sizeof(PopsQualifiedConstFieldV1), 1, identity.c_str(),
                const_view(patch.dependency_values[dependency_index].data(),
                           static_cast<std::size_t>(dependency.ncomp()),
                           POPS_FIELD_CENTERING_CELL_V1, 0)};
          }
          ++dependency_index;
        };
        for (const std::string& identity : spec_.states)
          append_dependency(identity, false);
        for (const std::string& identity : spec_.fields)
          append_dependency(identity, true);
        PopsFieldViewV1 output_view{sizeof(PopsFieldViewV1),
                                    patch.transformed.data(),
                                    static_cast<std::uint32_t>(Dim),
                                    {patch.extents[0], patch.extents[1], patch.extents[2]},
                                    {patch.strides[0], patch.strides[1], patch.strides[2]},
                                    components,
                                    static_cast<std::ptrdiff_t>(points),
                                    POPS_FIELD_CENTERING_FACE_V1,
                                    centering_axes,
                                    {0, 0, 0},
                                    {0, 0, 0},
                                    POPS_SCALAR_FLOAT64_V1,
                                    POPS_MEMORY_SPACE_HOST_V1,
                                    spec_.layout_identity.c_str(),
                                    patch.patch_identity.c_str(),
                                    POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
        std::fill_n(patch.actions.begin(), points, POPS_COMPONENT_CONTINUE_V1);
        PopsBoundaryFluxResultV1 result{sizeof(PopsBoundaryFluxResultV1), output_view,
                                        patch.actions.data(),
                                        component::unwritten_component_status()};
        const PopsBoundaryFluxRequestV1 request{sizeof(PopsBoundaryFluxRequestV1),
                                                spec_.producer_identity.c_str(),
                                                spec_.state_identity.c_str(),
                                                base_view,
                                                coordinate_view,
                                                normal_view,
                                                patch.measures.data(),
                                                spec_.region.view(),
                                                patch.dependency_rows.size(),
                                                patch.dependency_rows.data(),
                                                scratch.parameters.size(),
                                                scratch.parameters.data(),
                                                logical_time,
                                                session.patch_execution().view()};
        const int code = component::transform_boundary_flux(session.boundary_flux_api(),
                                                            session.state(), request, result);
        require_success(code, result.status, "transform_faces");
        if (std::any_of(patch.actions.begin(), patch.actions.begin() + points,
                        [](PopsComponentActionV1 action) {
                          return action != POPS_COMPONENT_CONTINUE_V1;
                        }) ||
            std::any_of(patch.transformed.begin(), patch.transformed.begin() + values,
                        [](double value) { return !std::isfinite(value); }))
          throw std::runtime_error(
              "native BoundaryFlux returned a non-continuing or non-finite candidate");
        for (std::size_t value = 0; value < values; ++value)
          face_host(patch.offsets[value]) = static_cast<Real>(side) * patch.transformed[value];
        patch.points = points;
        patch.active = true;
      };
      invoke_axis.template operator()<0>();
      if constexpr (Dim >= 2)
        invoke_axis.template operator()<1>();
      if constexpr (Dim >= 3)
        invoke_axis.template operator()<2>();
    }
    for (std::size_t local = 0; local < scratch.patches.size(); ++local) {
      auto& patch = scratch.patches[local];
      if (!patch.active)
        continue;
      auto publish_axis = [&]<int Axis>() {
        if (selected_axis == Axis)
          Kokkos::deep_copy(faces[local].template field<Axis>().storage(), patch.face_hosts[Axis]);
      };
      publish_axis.template operator()<0>();
      if constexpr (Dim >= 2)
        publish_axis.template operator()<1>();
      if constexpr (Dim >= 3)
        publish_axis.template operator()<2>();
    }
  }

  /// Add one external physical-boundary residual or JVP contribution. The retained Session owns
  /// the provider state and its lane-qualified execution strings for the complete prepared-block
  /// lifetime. Each patch is evaluated into detached host scratch and published additively only
  /// after the typed ABI call returns one finite candidate.
  template <int Dim>
  void evaluate_field_boundary_face(Session& session, int face, const MultiFab<Dim>& iterate,
                                    const MultiFab<Dim>* direction, MultiFab<Dim>& output,
                                    const Geometry<Dim>& geometry,
                                    const FieldBoundaryExecutionContext<Dim>& context) const
    requires(Operation == PreparedBoundaryOperation::FieldResidual ||
             Operation == PreparedBoundaryOperation::FieldJvp)
  {
    if (session.component_.get() != component_.get())
      throw std::invalid_argument(
          "prepared FieldBoundaryClosure session belongs to another provider");
    static_assert(Dim >= 1 && Dim <= 3);
    static_assert(sizeof(Real) == sizeof(double));
    constexpr bool jvp = Operation == PreparedBoundaryOperation::FieldJvp;
    if (face < 0 || face >= 2 * Dim || spec_.region.dimension != Dim ||
        spec_.region.axes.size() != 1 || spec_.region.sides.size() != 1 ||
        spec_.region.axes.front() != face / 2 ||
        spec_.region.sides.front() != (face % 2 == 0 ? -1 : 1) || spec_.states.empty() ||
        std::find(spec_.states.begin(), spec_.states.end(), spec_.state_identity) ==
            spec_.states.end() ||
        (jvp ? (direction == nullptr || spec_.directions.size() != 1 ||
                spec_.directions.front() != spec_.state_identity)
             : (direction != nullptr || !spec_.directions.empty())) ||
        (!spec_.nonlinear_iterate.empty() && spec_.nonlinear_iterate != spec_.state_identity) ||
        spec_.outputs.size() != 1 || iterate.local_size() != output.local_size() ||
        iterate.ncomp() != output.ncomp() ||
        (direction != nullptr && (direction->local_size() != iterate.local_size() ||
                                  direction->ncomp() != iterate.ncomp())) ||
        context.point.level < 0 || context.clock_identity == nullptr ||
        context.clock_identity->empty())
      throw std::invalid_argument(
          "prepared FieldBoundaryClosure received an incomplete exact routed invocation");
    auto& scratch = invocation_scratch_(session, iterate, geometry);
    for (auto& patch : scratch.patches)
      patch.active = false;

    const PopsLogicalTimeV1 logical_time{sizeof(PopsLogicalTimeV1),
                                         context.clock_identity->c_str(),
                                         context.point.step,
                                         context.point.level,
                                         context.point.substep,
                                         context.point.stage_slot,
                                         context.point.stage_fraction_numerator,
                                         context.point.stage_fraction_denominator,
                                         static_cast<double>(context.point.dt),
                                         static_cast<double>(context.point.time)};
    const int axis = face / 2;
    const int side = face % 2 == 0 ? -1 : 1;
    for (std::size_t local = 0; local < iterate.local_size(); ++local) {
      const Fab<Dim>& iterate_fab = iterate.fab(local);
      Fab<Dim>& output_fab = output.fab(local);
      auto& patch = scratch.patches[local];
      const Box<Dim>& valid = iterate_fab.box();
      if (side < 0 ? valid.lo[axis] != geometry.domain().lo[axis]
                   : valid.hi[axis] != geometry.domain().hi[axis])
        continue;
      if (iterate.global_index(local) != output.global_index(local) || valid != output_fab.box() ||
          (direction != nullptr && (direction->global_index(local) != iterate.global_index(local) ||
                                    direction->fab(local).box() != valid)))
        throw std::invalid_argument(
            "prepared FieldBoundaryClosure patch ownership differs across its invocation");

      Box<Dim> region = valid;
      region.lo[axis] = region.hi[axis] = side < 0 ? valid.lo[axis] : valid.hi[axis];
      patch.extents = {1, 1, 1};
      patch.strides = {0, 0, 0};
      std::size_t points = 1;
      for (int selected = 0; selected < Dim; ++selected) {
        patch.extents[selected] = static_cast<std::size_t>(region.length(selected));
        patch.strides[selected] = static_cast<std::ptrdiff_t>(points);
        points *= patch.extents[selected];
      }
      const std::size_t components = static_cast<std::size_t>(iterate.ncomp());
      const std::size_t values = checked_buffer_size_(points, components);
      if (patch.state_values.size() < values || patch.contribution.size() < values ||
          patch.offsets.size() < values ||
          patch.coordinates.size() < checked_buffer_size_(points, Dim) ||
          patch.input_host.extent(0) != iterate_fab.storage().extent(0) ||
          patch.output_host.extent(0) != output_fab.storage().extent(0))
        throw std::invalid_argument("prepared FieldBoundaryClosure exceeds its prepared scratch");
      if constexpr (jvp)
        if (patch.direction_values.size() < values ||
            patch.direction_host.extent(0) != direction->fab(local).storage().extent(0))
          throw std::invalid_argument(
              "prepared FieldBoundaryClosure JVP exceeds its prepared scratch");
      Kokkos::deep_copy(patch.input_host, iterate_fab.storage());
      Kokkos::deep_copy(patch.output_host, output_fab.storage());
      if constexpr (jvp)
        Kokkos::deep_copy(patch.direction_host, direction->fab(local).storage());
      std::fill_n(patch.contribution.begin(), values, 0.0);
      const auto iterate_view = iterate_fab.view();
      const auto output_view = output_fab.view();
      const auto direction_view = jvp ? direction->fab(local).view() : iterate_view;

      for_each_host_index_(region, [&](const Index<Dim>& index, std::size_t packed_point) {
        std::size_t iterate_offset = 0;
        std::size_t output_offset = 0;
        std::size_t direction_offset = 0;
        for (int coordinate = 0; coordinate < Dim; ++coordinate) {
          iterate_offset +=
              static_cast<std::size_t>(index[coordinate] - iterate_view.origin[coordinate]) *
              static_cast<std::size_t>(iterate_view.strides[coordinate]);
          output_offset +=
              static_cast<std::size_t>(index[coordinate] - output_view.origin[coordinate]) *
              static_cast<std::size_t>(output_view.strides[coordinate]);
          direction_offset +=
              static_cast<std::size_t>(index[coordinate] - direction_view.origin[coordinate]) *
              static_cast<std::size_t>(direction_view.strides[coordinate]);
          patch.coordinates[packed_point + static_cast<std::size_t>(coordinate) * points] =
              static_cast<double>(geometry.cell_coordinate(coordinate, index[coordinate]));
        }
        for (std::size_t component = 0; component < components; ++component) {
          const std::size_t packed = packed_point + component * points;
          patch.state_values[packed] = static_cast<double>(
              patch.input_host(iterate_offset + component * iterate_view.component_stride));
          if constexpr (jvp)
            patch.direction_values[packed] = static_cast<double>(patch.direction_host(
                direction_offset + component * direction_view.component_stride));
          patch.offsets[packed] = output_offset + component * output_view.component_stride;
        }
      });

      auto const_view = [&](const void* values, std::size_t component_count) {
        return PopsConstFieldViewV1{sizeof(PopsConstFieldViewV1),
                                    values,
                                    static_cast<std::uint32_t>(Dim),
                                    {patch.extents[0], patch.extents[1], patch.extents[2]},
                                    {patch.strides[0], patch.strides[1], patch.strides[2]},
                                    component_count,
                                    static_cast<std::ptrdiff_t>(points),
                                    POPS_FIELD_CENTERING_CELL_V1,
                                    0,
                                    {0, 0, 0},
                                    {0, 0, 0},
                                    POPS_SCALAR_FLOAT64_V1,
                                    POPS_MEMORY_SPACE_HOST_V1,
                                    spec_.layout_identity.c_str(),
                                    patch.patch_identity.c_str(),
                                    POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
      };
      const PopsConstFieldViewV1 state_view = const_view(patch.state_values.data(), components);
      const PopsConstFieldViewV1 coordinate_view =
          const_view(patch.coordinates.data(), static_cast<std::size_t>(Dim));
      const PopsConstFieldViewV1 direction_field_view =
          const_view(patch.direction_values.data(), components);
      std::size_t dependency_index = 0;
      for (std::size_t state_index = 0; state_index < spec_.states.size(); ++state_index) {
        const std::string& identity = spec_.states[state_index];
        const MultiFab<Dim>& dependency =
            identity == spec_.state_identity ? iterate : *scratch.dependencies[dependency_index];
        if (identity == spec_.state_identity)
          patch.state_rows[state_index] = {sizeof(PopsQualifiedConstFieldV1), 1, identity.c_str(),
                                           state_view};
        else {
          pack_dependency_region(dependency, iterate.global_index(local), region,
                                 patch.dependency_hosts[dependency_index],
                                 patch.dependency_values[dependency_index]);
          patch.state_rows[state_index] = {
              sizeof(PopsQualifiedConstFieldV1), 1, identity.c_str(),
              const_view(patch.dependency_values[dependency_index].data(),
                         static_cast<std::size_t>(dependency.ncomp()))};
        }
        ++dependency_index;
      }
      for (std::size_t field_index = 0; field_index < spec_.fields.size(); ++field_index) {
        const std::string& identity = spec_.fields[field_index];
        const MultiFab<Dim>& dependency = *scratch.dependencies[dependency_index];
        pack_dependency_region(dependency, iterate.global_index(local), region,
                               patch.dependency_hosts[dependency_index],
                               patch.dependency_values[dependency_index]);
        patch.field_rows[field_index] = {
            sizeof(PopsQualifiedConstFieldV1), 1, identity.c_str(),
            const_view(patch.dependency_values[dependency_index].data(),
                       static_cast<std::size_t>(dependency.ncomp()))};
        ++dependency_index;
      }
      PopsFieldViewV1 contribution_view{sizeof(PopsFieldViewV1),
                                        patch.contribution.data(),
                                        static_cast<std::uint32_t>(Dim),
                                        {patch.extents[0], patch.extents[1], patch.extents[2]},
                                        {patch.strides[0], patch.strides[1], patch.strides[2]},
                                        components,
                                        static_cast<std::ptrdiff_t>(points),
                                        POPS_FIELD_CENTERING_CELL_V1,
                                        0,
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        POPS_SCALAR_FLOAT64_V1,
                                        POPS_MEMORY_SPACE_HOST_V1,
                                        spec_.layout_identity.c_str(),
                                        patch.patch_identity.c_str(),
                                        POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
      PopsQualifiedConstFieldV1 direction_row{sizeof(PopsQualifiedConstFieldV1), 1,
                                              spec_.state_identity.c_str(), direction_field_view};
      PopsQualifiedFieldV1 output_row{sizeof(PopsQualifiedFieldV1), spec_.outputs.front().c_str(),
                                      contribution_view};
      const PopsQualifiedConstFieldV1 absent{sizeof(PopsQualifiedConstFieldV1), 0, nullptr, {}};
      PopsQualifiedConstFieldV1 rate = absent;
      if (!spec_.rate.empty()) {
        const auto state_rate = std::find_if(
            patch.state_rows.begin(), patch.state_rows.end(),
            [&](const auto& row) { return std::string_view(row.qualified_id) == spec_.rate; });
        const auto field_rate = std::find_if(
            patch.field_rows.begin(), patch.field_rows.end(),
            [&](const auto& row) { return std::string_view(row.qualified_id) == spec_.rate; });
        if (state_rate != patch.state_rows.end())
          rate = *state_rate;
        else if (field_rate != patch.field_rows.end())
          rate = *field_rate;
        else
          throw std::invalid_argument(
              "prepared FieldBoundaryClosure rate is absent from its routed dependencies");
      }
      const PopsQualifiedConstFieldV1 nonlinear{
          sizeof(PopsQualifiedConstFieldV1), spec_.nonlinear_iterate.empty() ? 0u : 1u,
          spec_.nonlinear_iterate.empty() ? nullptr : spec_.nonlinear_iterate.c_str(), state_view};
      const PopsFieldBoundaryRequestV1 request{sizeof(PopsFieldBoundaryRequestV1),
                                               spec_.producer_identity.c_str(),
                                               spec_.region.view(),
                                               coordinate_view,
                                               patch.state_rows.size(),
                                               patch.state_rows.data(),
                                               jvp ? 1u : 0u,
                                               jvp ? &direction_row : nullptr,
                                               patch.field_rows.size(),
                                               patch.field_rows.data(),
                                               scratch.parameters.size(),
                                               scratch.parameters.data(),
                                               1,
                                               &output_row,
                                               rate,
                                               nonlinear,
                                               context.point.level,
                                               logical_time,
                                               session.patch_execution().view()};
      PopsComponentStatusV1 status = component::unwritten_component_status();
      const int code = component::evaluate_field_boundary(session.field_api(), session.state(),
                                                          request, status, jvp);
      require_success(code, status, jvp ? "jvp" : "residual");
      if (std::any_of(patch.contribution.begin(), patch.contribution.begin() + values,
                      [](double value) { return !std::isfinite(value); }))
        throw std::runtime_error("native FieldBoundaryClosure returned a non-finite candidate");
      for (std::size_t value = 0; value < values; ++value)
        patch.output_host(patch.offsets[value]) += static_cast<Real>(patch.contribution[value]);
      patch.points = points;
      patch.active = true;
    }
    for (std::size_t local = 0; local < scratch.patches.size(); ++local)
      if (scratch.patches[local].active)
        Kokkos::deep_copy(output.fab(local).storage(), scratch.patches[local].output_host);
  }

  template <int Dim>
  static void pack_dependency_region(const MultiFab<Dim>& dependency, std::size_t global_patch,
                                     const Box<Dim>& region,
                                     typename Fab<Dim>::raw_host_mirror_type& host,
                                     std::vector<double>& result) {
    if (!dependency.contains_local(global_patch))
      throw std::invalid_argument(
          "prepared boundary dependency does not materialize the owning local patch");
    const Fab<Dim>& fab = dependency.fab(dependency.local_index_of(global_patch));
    const Box<Dim>& storage = fab.grown_box();
    for (int axis = 0; axis < Dim; ++axis)
      if (region.lo[axis] < storage.lo[axis] || region.hi[axis] > storage.hi[axis])
        throw std::invalid_argument(
            "prepared boundary dependency lacks the exact physical-region halo");
    if (fab.ncomp() < 1)
      throw std::invalid_argument("prepared boundary dependency has no components");
    const std::size_t points = static_cast<std::size_t>(region.numPts());
    if (result.size() < checked_buffer_size_(points, static_cast<std::size_t>(fab.ncomp())) ||
        host.extent(0) != fab.storage().extent(0))
      throw std::invalid_argument(
          "prepared boundary dependency differs from its prepared host scratch");
    Kokkos::deep_copy(host, fab.storage());
    const auto view = fab.view();
    for_each_host_index_(region, [&](const Index<Dim>& index, std::size_t packed) {
      std::size_t offset = 0;
      for (int selected = 0; selected < Dim; ++selected)
        offset += static_cast<std::size_t>(index[selected] - view.origin[selected]) *
                  static_cast<std::size_t>(view.strides[selected]);
      for (int component = 0; component < fab.ncomp(); ++component)
        result[packed + static_cast<std::size_t>(component) * points] = static_cast<double>(
            host(offset + static_cast<std::size_t>(component) * view.component_stride));
    });
  }

  static void require_success(int code, const PopsComponentStatusV1& status,
                              const char* operation) {
    if (!component::component_status_is_well_formed(status) || code != 0)
      throw std::runtime_error(
          std::string("native boundary component ") + operation +
          " transport failed: " + (status.reason == nullptr ? "no reason" : status.reason));
    if (status.action == POPS_COMPONENT_CONTINUE_V1) {
      if (status.code != 0)
        throw std::runtime_error(std::string("native boundary component ") + operation +
                                 " returned a contradictory continue status");
      return;
    }
    if (status.action == POPS_COMPONENT_RETRY_STEP_V1 ||
        status.action == POPS_COMPONENT_REJECT_STEP_V1) {
      if (status.code <= 0 || status.reason == nullptr || *status.reason == '\0')
        throw std::runtime_error(std::string("native boundary component ") + operation +
                                 " returned an incomplete step rejection");
      const auto disposition = status.action == POPS_COMPONENT_RETRY_STEP_V1
                                   ? runtime::program::StepAttemptDisposition::kRetry
                                   : runtime::program::StepAttemptDisposition::kReject;
      throw runtime::program::StepAttemptRejected(SolveStatus::kInvalidEvaluation, disposition,
                                                  static_cast<std::uint32_t>(status.code),
                                                  operation, status.reason);
    }
    throw std::runtime_error(
        std::string("native boundary component ") + operation +
        " aborted the run: " + (status.reason == nullptr ? "no reason" : status.reason));
  }

 private:
  static constexpr PopsNativeInterfaceIdV1 native_interface_id_() {
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion)
      return POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1;
    else if constexpr (Operation == PreparedBoundaryOperation::FluxTransform)
      return POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1;
    else
      return POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1;
  }

  const PopsComponentTableHeaderV1& table_header() const {
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion)
      return ghost_api().header;
    else if constexpr (Operation == PreparedBoundaryOperation::FluxTransform)
      return boundary_flux_api().header;
    else
      return field_api().header;
  }

  void validate() const {
    if (!component_ || spec_.target_identity.empty() || spec_.component_id.empty() ||
        spec_.manifest_identity.empty() || spec_.producer_identity.empty() ||
        spec_.state_identity.empty() || spec_.ghost_identity.empty() ||
        spec_.layout_identity.empty() || spec_.region.identity.empty() ||
        spec_.parameter_ids.size() != spec_.parameter_values.size() ||
        std::any_of(spec_.parameter_values.begin(), spec_.parameter_values.end(),
                    [](double value) { return !std::isfinite(value); }))
      throw std::invalid_argument("prepared boundary component identity/tables are incomplete");
    if (spec_.interface_version != 1)
      throw std::invalid_argument("prepared boundary component requires interface version 1");
    if (!spec_.execution)
      throw std::invalid_argument("prepared boundary component lacks ExecutionContext authority");
    component::validate_execution_context(spec_.execution->view());
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion) {
      component::require_operation(ghost_api().apply_region_batch != nullptr, "apply_region_batch");
      if (spec_.outputs.size() != 1 || spec_.outputs.front() != spec_.state_identity ||
          !spec_.directions.empty() || !spec_.rate.empty() || !spec_.nonlinear_iterate.empty())
        throw std::invalid_argument(
            "GhostBoundary requires one primary-state output and no direction dependencies");
    } else if constexpr (Operation == PreparedBoundaryOperation::FluxTransform) {
      component::require_operation(boundary_flux_api().transform_faces != nullptr,
                                   "transform_faces");
      if (spec_.region.kind != POPS_BOUNDARY_FACE_V1 || spec_.region.codimension != 1 ||
          spec_.outputs.size() != 1 || spec_.outputs.front() != spec_.state_identity ||
          !spec_.directions.empty() || !spec_.rate.empty() || !spec_.nonlinear_iterate.empty())
        throw std::invalid_argument(
            "BoundaryFlux requires one oriented face, one state output and no direction table");
    } else {
      component::require_operation(
          Operation == PreparedBoundaryOperation::FieldResidual ? field_api().residual != nullptr
                                                                : field_api().jvp != nullptr,
          Operation == PreparedBoundaryOperation::FieldResidual ? "residual" : "jvp");
      if (spec_.region.kind != POPS_BOUNDARY_FACE_V1 || spec_.region.codimension != 1 ||
          spec_.states.empty() ||
          std::find(spec_.states.begin(), spec_.states.end(), spec_.state_identity) ==
              spec_.states.end() ||
          spec_.outputs.size() != 1 ||
          (!spec_.rate.empty() &&
           std::find(spec_.states.begin(), spec_.states.end(), spec_.rate) == spec_.states.end() &&
           std::find(spec_.fields.begin(), spec_.fields.end(), spec_.rate) == spec_.fields.end()) ||
          (!spec_.nonlinear_iterate.empty() && spec_.nonlinear_iterate != spec_.state_identity) ||
          (Operation == PreparedBoundaryOperation::FieldResidual && !spec_.directions.empty()) ||
          (Operation == PreparedBoundaryOperation::FieldJvp &&
           (spec_.directions.size() != 1 || spec_.directions.front() != spec_.state_identity)))
        throw std::invalid_argument(
            "FieldBoundaryClosure requires one exact routed residual/JVP face contract");
    }
    const auto& api = component_->api();
    if (api.component_id == nullptr || api.manifest_identity == nullptr ||
        spec_.component_id != api.component_id || spec_.manifest_identity != api.manifest_identity)
      throw std::invalid_argument("prepared boundary component changed native identity");
    component::validate_boundary_region(spec_.region.view());
  }

  PreparedBoundaryComponentSpec spec_;
  std::shared_ptr<component::LoadedComponent> component_;
};

using PreparedGhostBoundaryComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::GhostRegion>;
using PreparedBoundaryFluxComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FluxTransform>;
using PreparedFieldBoundaryResidualComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldResidual>;
using PreparedFieldBoundaryJvpComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldJvp>;

}  // namespace pops
