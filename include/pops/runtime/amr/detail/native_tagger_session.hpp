/// @file
/// @brief Native execution and storage session for a prepared AMR Tagger.

#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops::runtime::amr::detail {

/// Native execution, storage, and ABI binding for one prepared Tagger component.
template <int Dim, class MemorySpace, class Spec>
class NativeTaggerSession {
 public:
  using Program = PreparedTaggingProgram<Dim>;
  using Field = PreparedTaggingField<Dim, MemorySpace>;
  using Candidates = PreparedTaggerCandidates<Dim>;
  using mask_type = ::pops::amr::tagging::TagMask<Dim>;

  NativeTaggerSession() = default;
  NativeTaggerSession(const NativeTaggerSession&) = delete;
  NativeTaggerSession& operator=(const NativeTaggerSession&) = delete;
  NativeTaggerSession(NativeTaggerSession&&) = delete;
  NativeTaggerSession& operator=(NativeTaggerSession&&) = delete;

  static std::unique_ptr<NativeTaggerSession> prepare(
      std::shared_ptr<component::LoadedComponent> component, Spec spec, const Program& program,
      const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      std::uint32_t periodic_axes, const ExecutionLane& lane) {
    const CommunicatorView communicator = lane.communicator();
    std::optional<std::unique_ptr<NativeTaggerSession>> candidate;
    enum class ExceptionKind : long { None = 0, StepRejected = 1, Ordinary = 2 };
    ExceptionKind exception_kind = ExceptionKind::None;
    std::string rejection_payload;
    std::exception_ptr local_error;
    try {
      if (comm_active() && !communicator.active())
        throw std::invalid_argument(
            "native AMR Tagger requires an explicit active execution communicator");
      candidate.emplace(prepare_local_(std::move(component), std::move(spec), program,
                                       fields_by_level, layouts, budgets, topology_generation,
                                       periodic_axes, lane));
    } catch (const ::pops::runtime::program::StepAttemptRejected& rejected) {
      try {
        rejection_payload = encode_step_rejection_(rejected);
        exception_kind = ExceptionKind::StepRejected;
      } catch (...) {
        exception_kind = ExceptionKind::Ordinary;
        local_error = std::current_exception();
      }
    } catch (...) {
      exception_kind = ExceptionKind::Ordinary;
      local_error = std::current_exception();
    }
    const long ordinary = exception_kind == ExceptionKind::Ordinary ? 1L : 0L;
    const long rejected = exception_kind == ExceptionKind::StepRejected ? 1L : 0L;
    if (all_reduce_max(ordinary, communicator) != 0) {
      if (communicator.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("native AMR Tagger preparation failed on another rank");
    }
    if (all_reduce_max(rejected, communicator) != 0)
      throw_collective_rejection_(rejection_payload, rejected, communicator);
    std::string rank_budget_contract;
    local_error = nullptr;
    try {
      rank_budget_contract =
          tagging_detail::exact_rank_ordered_budget_contract(budgets, communicator);
      ExactContractBuilder collective;
      collective.text("pops.amr.native-tagger-collective")
          .scalar(std::uint32_t{1})
          .bytes((*candidate)->storage_->collective_contract)
          .bytes(rank_budget_contract);
      (*candidate)->storage_->collective_contract = std::move(collective).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
      if (communicator.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("native AMR Tagger budget authentication failed on another rank");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"native-amr-tagger", (*candidate)->storage_->collective_contract}}, communicator))
      throw std::invalid_argument("native AMR Tagger prepared contracts differ between ranks");
    (*candidate)->storage_->prepared = true;
    return std::move(*candidate);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return storage_ != nullptr && storage_->prepared;
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return storage_ == nullptr ? std::string_view{} : storage_->collective_contract;
  }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return storage_ == nullptr ? 0 : storage_->topology_generation;
  }

  const Candidates& execute(std::size_t level_index,
                            const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
                            const std::array<Real, Dim>& spacing, std::uint64_t topology_generation,
                            std::int64_t tick, double physical_time) {
    long preflight_failure = storage_ == nullptr || !storage_->prepared ||
                                     level_index >= storage_->levels.size() ||
                                     topology_generation != storage_->topology_generation ||
                                     tick < 0 || !std::isfinite(physical_time)
                                 ? 1L
                                 : 0L;
    for (int axis = 0; axis < Dim; ++axis)
      if (!(spacing[axis] > Real(0)) || !std::isfinite(static_cast<double>(spacing[axis])))
        preflight_failure = 1;
    if (preflight_failure == 0 && storage_->levels[level_index].identity != layout.exact_identity())
      preflight_failure = 1;
    if (all_reduce_max(preflight_failure, storage_->communicator) != 0)
      throw std::runtime_error("native AMR Tagger collective execution preflight failed");

    Level& level = storage_->levels[level_index];
    enum class ExceptionKind : long { None = 0, StepRejected = 1, Ordinary = 2 };
    ExceptionKind exception_kind = ExceptionKind::None;
    std::string rejection_payload;
    std::exception_ptr local_error;
    try {
      for (Patch& patch : level.patches)
        execute_patch_(*storage_, patch, layout.domain(), spacing, level_index, tick,
                       physical_time);
      device_fence();
    } catch (const ::pops::runtime::program::StepAttemptRejected& rejected) {
      try {
        rejection_payload = encode_step_rejection_(rejected);
        exception_kind = ExceptionKind::StepRejected;
      } catch (...) {
        exception_kind = ExceptionKind::Ordinary;
        local_error = std::current_exception();
      }
    } catch (...) {
      exception_kind = ExceptionKind::Ordinary;
      local_error = std::current_exception();
    }
    const long ordinary = exception_kind == ExceptionKind::Ordinary ? 1L : 0L;
    const long rejected = exception_kind == ExceptionKind::StepRejected ? 1L : 0L;
    if (all_reduce_max(ordinary, storage_->communicator) != 0) {
      if (storage_->communicator.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("native AMR Tagger failed on another rank");
    }
    if (all_reduce_max(rejected, storage_->communicator) != 0)
      throw_collective_rejection_(rejection_payload, rejected, storage_->communicator);

    if (level.replicated) {
      std::size_t offset = 0;
      for (const Patch& patch : level.patches) {
        for (std::size_t point = 0; point < patch.host_outputs[0].extent(0); ++point) {
          std::uint8_t packed = 0;
          for (std::size_t output = 0; output < patch.host_outputs.size(); ++output) {
            packed = static_cast<std::uint8_t>(
                packed | static_cast<std::uint8_t>(patch.host_outputs[output](point) << output));
          }
          level.replica_min[offset] = static_cast<char>(packed);
          level.replica_max[offset] = level.replica_min[offset];
          ++offset;
        }
      }
      all_reduce_min_inplace(level.replica_min.data(), level.replica_min.size(),
                             storage_->communicator);
      all_reduce_max_inplace(level.replica_max.data(), level.replica_max.size(),
                             storage_->communicator);
      if (!std::equal(level.replica_min.begin(), level.replica_min.end(),
                      level.replica_max.begin()))
        throw std::runtime_error(
            "native AMR Tagger replicated fields produced different masks between ranks");
    }

    for (const Patch& patch : level.patches)
      for_each_host_index_(patch.box, [&](const Index<Dim>& index, std::size_t ordinal) {
        level.candidates.refine.set(patch.global_patch, index, patch.host_outputs[0](ordinal) != 0);
        level.candidates.coarsen.set(patch.global_patch, index,
                                     patch.host_outputs[1](ordinal) != 0);
        level.candidates.refine_equalities.set(patch.global_patch, index,
                                               patch.host_outputs[2](ordinal) != 0);
        level.candidates.coarsen_equalities.set(patch.global_patch, index,
                                                patch.host_outputs[3](ordinal) != 0);
      });
    return level.candidates;
  }

 private:
  template <class T>
  using CommunicationVector = std::vector<T, comm_allocator<T>>;
  using DeviceRealView = typename Fab<Dim, MemorySpace>::storage_type;
  using HostRealView = typename Fab<Dim, MemorySpace>::raw_host_mirror_type;
  using DeviceMaskView = Kokkos::View<std::uint8_t*, MemorySpace>;
  using HostMaskView = Kokkos::View<std::uint8_t*, Kokkos::HostSpace>;

  struct Patch {
    Box<Dim> box{};
    std::size_t global_patch = 0;
    std::string patch_identity{};
    std::vector<std::string> state_identities{};
    std::vector<PopsQualifiedConstFieldV1> states{};
    std::vector<DeviceRealView> source_state_values{};
    std::vector<HostRealView> host_state_values{};
    std::vector<PopsQualifiedConstFieldV1> host_states{};
    std::array<DeviceMaskView, 4> device_outputs{};
    std::array<HostMaskView, 4> host_outputs{};
  };

  struct Level {
    ::pops::amr::hierarchy::LevelLayoutIdentity<Dim> identity{};
    std::vector<Patch> patches{};
    bool replicated = false;
    CommunicationVector<char> replica_min{};
    CommunicationVector<char> replica_max{};
    Candidates candidates;

    Level(const ::pops::amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& local_rank,
          const PreparedTaggingExecutionBudget& budget, std::size_t local_cells)
        : identity(layout.exact_identity()),
          replicated(layout.distribution().replicated()),
          replica_min(replicated ? local_cells : 0, char{0}),
          replica_max(replicated ? local_cells : 0, char{0}),
          candidates{mask_type(layout, local_rank, budget.candidate_mask),
                     mask_type(layout, local_rank, budget.candidate_mask),
                     mask_type(layout, local_rank, budget.candidate_mask),
                     mask_type(layout, local_rank, budget.candidate_mask)} {}
  };

  struct ComponentState {
    void* value = nullptr;
    PopsComponentDestroyFnV1 destroy = nullptr;

    ComponentState() = default;
    ComponentState(void* prepared, PopsComponentDestroyFnV1 destructor) noexcept
        : value(prepared), destroy(destructor) {}
    ComponentState(const ComponentState&) = delete;
    ComponentState& operator=(const ComponentState&) = delete;
    ComponentState(ComponentState&& other) noexcept
        : value(std::exchange(other.value, nullptr)),
          destroy(std::exchange(other.destroy, nullptr)) {}
    ComponentState& operator=(ComponentState&& other) noexcept {
      if (this != &other) {
        reset();
        value = std::exchange(other.value, nullptr);
        destroy = std::exchange(other.destroy, nullptr);
      }
      return *this;
    }
    ~ComponentState() { reset(); }

    [[nodiscard]] void* get() const noexcept { return value; }

   private:
    void reset() noexcept {
      if (destroy != nullptr)
        destroy(value);
      value = nullptr;
      destroy = nullptr;
    }
  };

  struct Storage {
    // LoadedComponent must outlive the session state that calls its table-owned destroy function.
    std::shared_ptr<component::LoadedComponent> component{};
    ComponentState state{};
    Spec spec{};
    Program program{};
    std::vector<PopsTaggingLeafV1> abi_leaves{};
    std::vector<std::array<PopsTaggingAxisStencilV1, Dim>> abi_axes{};
    std::vector<PopsTaggingStencilV1> abi_stencils{};
    std::vector<Level> levels{};
    std::uint64_t topology_generation = 0;
    std::uint32_t periodic_axes = 0;
    CommunicatorView communicator{};
    std::string collective_contract{};
    bool prepared = false;
  };

  struct StepRejectionEnvelope {
    SolveStatus status = SolveStatus::kInvalidEvaluation;
    ::pops::runtime::program::StepAttemptDisposition disposition =
        ::pops::runtime::program::StepAttemptDisposition::kReject;
    std::uint32_t reason_code = 0;
    std::string phase{};
    std::string detail{};
  };

  static void append_u64_(std::string& bytes, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
  }

  static std::uint64_t read_u64_(std::string_view bytes, std::size_t& cursor) {
    if (cursor > bytes.size() || bytes.size() - cursor < 8)
      throw std::runtime_error("collective AMR Tagger rejection envelope is truncated");
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte)
      value = (value << 8u) | static_cast<unsigned char>(bytes[cursor++]);
    return value;
  }

  static void append_text_(std::string& bytes, std::string_view value) {
    append_u64_(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.append(value.data(), value.size());
  }

  static std::string read_text_(std::string_view bytes, std::size_t& cursor) {
    const std::uint64_t encoded_size = read_u64_(bytes, cursor);
    if (encoded_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("collective AMR Tagger rejection text exceeds size_t");
    const std::size_t size = static_cast<std::size_t>(encoded_size);
    if (cursor > bytes.size() || size > bytes.size() - cursor)
      throw std::runtime_error("collective AMR Tagger rejection text is truncated");
    std::string result(bytes.substr(cursor, size));
    cursor += size;
    return result;
  }

  static std::string encode_step_rejection_(
      const ::pops::runtime::program::StepAttemptRejected& rejected) {
    std::string bytes("pops.amr-tagger.step-rejection.v1");
    append_u64_(bytes, static_cast<std::uint64_t>(rejected.status()));
    append_u64_(bytes, static_cast<std::uint64_t>(rejected.disposition()));
    append_u64_(bytes, rejected.reason_code());
    append_text_(bytes, rejected.phase());
    append_text_(bytes, rejected.detail());
    return bytes;
  }

  static StepRejectionEnvelope decode_step_rejection_(std::string_view bytes) {
    constexpr std::string_view prefix = "pops.amr-tagger.step-rejection.v1";
    if (!bytes.starts_with(prefix))
      throw std::runtime_error("collective AMR Tagger rejection has another schema");
    std::size_t cursor = prefix.size();
    const std::uint64_t status = read_u64_(bytes, cursor);
    const std::uint64_t disposition = read_u64_(bytes, cursor);
    const std::uint64_t reason_code = read_u64_(bytes, cursor);
    if (status > static_cast<std::uint64_t>(SolveStatus::kSafeguardFailure) ||
        disposition >
            static_cast<std::uint64_t>(::pops::runtime::program::StepAttemptDisposition::kReject) ||
        reason_code > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("collective AMR Tagger rejection has invalid typed fields");
    StepRejectionEnvelope result;
    result.status = static_cast<SolveStatus>(status);
    result.disposition = static_cast<::pops::runtime::program::StepAttemptDisposition>(disposition);
    result.reason_code = static_cast<std::uint32_t>(reason_code);
    result.phase = read_text_(bytes, cursor);
    result.detail = read_text_(bytes, cursor);
    if (cursor != bytes.size() || result.phase.empty() || result.detail.empty())
      throw std::runtime_error("collective AMR Tagger rejection envelope is incomplete");
    return result;
  }

  [[noreturn]] static void throw_collective_rejection_(const std::string& rejection_payload,
                                                       long rejected,
                                                       const CommunicatorView& communicator) {
    std::string selected_payload;
    if (all_reduce_min(rejected, communicator) != 0) {
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{"amr-tagger-step-rejection", rejection_payload}}, communicator))
        throw std::runtime_error("collective AMR Tagger rejection fields differ between ranks");
      selected_payload = rejection_payload;
    } else {
      const long local_root = rejected != 0 ? static_cast<long>(communicator.rank())
                                            : static_cast<long>(communicator.size());
      const long root = all_reduce_min(local_root, communicator);
      if (root < 0 || root >= static_cast<long>(communicator.size()))
        throw std::runtime_error("collective AMR Tagger rejection lost its typed envelope");
      const bool authoritative = communicator.rank() == root;
      const long invalid_length =
          authoritative && rejection_payload.size() >
                               static_cast<std::size_t>(std::numeric_limits<long>::max())
              ? 1L
              : 0L;
      if (all_reduce_max(invalid_length, communicator) != 0)
        throw std::length_error("collective AMR Tagger rejection exceeds long capacity");
      const long encoded_length = all_reduce_max(
          authoritative ? static_cast<long>(rejection_payload.size()) : 0L, communicator);
      if (encoded_length <= 0)
        throw std::runtime_error("collective AMR Tagger rejection envelope is empty");
      long allocation_failed = 0;
      try {
        if (authoritative)
          selected_payload = rejection_payload;
        selected_payload.resize(static_cast<std::size_t>(encoded_length));
      } catch (...) {
        allocation_failed = 1;
      }
      if (all_reduce_max(allocation_failed, communicator) != 0)
        throw std::bad_alloc();
      broadcast_bytes_inplace(selected_payload.data(), selected_payload.size(),
                              static_cast<int>(root), communicator);
      const long mismatch = rejected != 0 && rejection_payload != selected_payload ? 1L : 0L;
      if (all_reduce_max(mismatch, communicator) != 0)
        throw std::runtime_error(
            "collective AMR Tagger rejection fields differ between rejecting ranks");
    }
    const StepRejectionEnvelope envelope = decode_step_rejection_(selected_payload);
    throw ::pops::runtime::program::StepAttemptRejected(envelope.status, envelope.disposition,
                                                        envelope.reason_code, envelope.phase,
                                                        envelope.detail);
  }

  static void require_component_status_(int transport_code, const PopsComponentStatusV1& status,
                                        std::string_view phase) {
    if (!component::component_status_is_well_formed(status) || transport_code != 0)
      throw std::runtime_error(status.reason == nullptr ? "native AMR Tagger transport failed"
                                                        : status.reason);
    if (status.action == POPS_COMPONENT_CONTINUE_V1) {
      if (status.code != 0)
        throw std::runtime_error("native AMR Tagger returned a contradictory continue status");
      return;
    }
    if (status.action == POPS_COMPONENT_RETRY_STEP_V1 ||
        status.action == POPS_COMPONENT_REJECT_STEP_V1) {
      if (status.code <= 0 || status.reason == nullptr || *status.reason == '\0')
        throw std::runtime_error("native AMR Tagger returned an incomplete step rejection");
      const auto disposition = status.action == POPS_COMPONENT_RETRY_STEP_V1
                                   ? ::pops::runtime::program::StepAttemptDisposition::kRetry
                                   : ::pops::runtime::program::StepAttemptDisposition::kReject;
      throw ::pops::runtime::program::StepAttemptRejected(
          SolveStatus::kInvalidEvaluation, disposition, static_cast<std::uint32_t>(status.code),
          std::string(phase), status.reason);
    }
    throw std::runtime_error(status.reason == nullptr ? "native AMR Tagger aborted the run"
                                                      : status.reason);
  }

  static component::PreparedExecutionContextV1 host_staging_execution_(
      const component::PreparedExecutionContextV1& lane_execution) {
    const PopsExecutionContextV1 source = lane_execution.view();
    return component::PreparedExecutionContextV1(
        std::string(source.execution_identity) + "/host-staging", source.context_version,
        POPS_MEMORY_SPACE_HOST_V1, source.backend_identity, "host", source.scalar_type,
        source.storage_precision, source.compute_precision, source.accumulation_precision,
        source.reduction_precision, 0, "host::synchronous", source.communicator_f_handle,
        source.communicator_datatype_f_handle, source.communicator_identity,
        source.communicator_datatype_identity);
  }

  static ComponentState prepare_component_state_(Storage& storage) {
    const auto& api = storage.component->template table<PopsTaggerApiV2>(
        POPS_NATIVE_INTERFACE_TAGGER_V2, storage.spec.interface_version);
    if (api.header.prepare == nullptr)
      return {};
    void* state = nullptr;
    PopsComponentStatusV1 status = component::unwritten_component_status();
    const PopsComponentPrepareRequestV1 request{
        sizeof(PopsComponentPrepareRequestV1), storage.spec.parameters_json.c_str(),
        storage.spec.target_json.c_str(), storage.spec.execution->view()};
    const int code = api.header.prepare(&request, &state, &status);
    try {
      require_component_status_(code, status, "amr_tagger_prepare");
    } catch (...) {
      if (state != nullptr && api.header.destroy != nullptr)
        api.header.destroy(state);
      throw;
    }
    return ComponentState(state, api.header.destroy);
  }

  static std::unique_ptr<NativeTaggerSession> prepare_local_(
      std::shared_ptr<component::LoadedComponent> component, Spec spec, const Program& program,
      const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      std::uint32_t periodic_axes, const ExecutionLane& lane) {
    const CommunicatorView communicator = lane.communicator();
    if (!component || !spec.execution || spec.component_id.empty() ||
        spec.manifest_identity.empty() || spec.provider_identity.empty() ||
        spec.tagging_graph_identity.empty() || spec.layout_identity.empty() ||
        spec.clock_identity.empty() || spec.interface_version != 2 || !program.prepared ||
        program.clock_identity != spec.clock_identity || program.leaves.empty() ||
        fields_by_level.empty() || fields_by_level.size() != layouts.size() ||
        layouts.size() != budgets.size())
      throw std::invalid_argument("native AMR Tagger preparation authority is incomplete");
    if (std::any_of(program.leaves.begin(), program.leaves.end(), [](const auto& leaf) {
          return leaf.opcode == POPS_TAGGING_PRESCRIBED_WINDOW_V1;
        }))
      throw std::invalid_argument(
          "native AMR Tagger V2 cannot receive prescribed windows through PopsTaggingLeafV1");
    const PopsComponentApiV1& component_api = component->api();
    if (component_api.component_id == nullptr || component_api.manifest_identity == nullptr ||
        spec.component_id != component_api.component_id ||
        spec.manifest_identity != component_api.manifest_identity)
      throw std::invalid_argument("native AMR Tagger component identity changed after load");
    (void)component->table<PopsTaggerApiV2>(POPS_NATIVE_INTERFACE_TAGGER_V2,
                                            spec.interface_version);
    const auto lane_execution = spec.execution->for_lane(lane);
    if (!lane_execution.matches_lane(lane))
      throw std::invalid_argument("native AMR Tagger execution lane is unauthenticated");
    const bool host_staging = spec.execution_mode == POPS_TAGGER_EXECUTION_HOST_V2;
    if (!host_staging && spec.execution_mode != POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2)
      throw std::invalid_argument("native AMR Tagger execution or memory authority is inexact");
    const auto callback_execution =
        host_staging ? host_staging_execution_(lane_execution) : lane_execution;
    const auto execution = callback_execution.without_collective_authority();
    if ((host_staging && execution.view().memory_space != POPS_MEMORY_SPACE_HOST_V1) ||
        (!host_staging && !execution_memory_matches_(execution.view().memory_space)))
      throw std::invalid_argument("native AMR Tagger execution or memory authority is inexact");

    auto result = std::make_unique<NativeTaggerSession>();
    Storage* storage = result->storage_;
    storage->component = std::move(component);
    storage->spec = std::move(spec);
    storage->spec.execution =
        std::make_shared<const component::PreparedExecutionContextV1>(execution);
    storage->program = program;
    storage->topology_generation = topology_generation;
    storage->periodic_axes = periodic_axes;
    storage->communicator = communicator;

    ExactContractBuilder contract;
    contract.text("pops.amr.native-tagger-session")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(storage->spec.component_id)
        .text(storage->spec.manifest_identity)
        .text(storage->spec.provider_identity)
        .text(storage->spec.tagging_graph_identity)
        .text(storage->spec.layout_identity)
        .text(storage->spec.clock_identity)
        .scalar(storage->spec.interface_version)
        .scalar(storage->spec.execution_mode)
        .text(host_staging ? "kokkos-host-staging-v1" : "native-backend-direct-v1")
        .text(storage->spec.execution->identity())
        .text(lane.identity())
        .text(storage->spec.parameters_json)
        .text(storage->spec.target_json)
        .scalar(topology_generation)
        .scalar(periodic_axes)
        .scalar(static_cast<std::uint64_t>(layouts.size()));

    std::vector<std::vector<tagging_detail::PreparedTaggingFieldContract<Dim>>> field_contracts;
    field_contracts.reserve(layouts.size());
    storage->levels.reserve(layouts.size());
    for (std::size_t level_index = 0; level_index < layouts.size(); ++level_index) {
      const auto& layout = layouts[level_index];
      const auto& fields = fields_by_level[level_index];
      if (fields.empty() || fields.front().values == nullptr)
        throw std::invalid_argument("native AMR Tagger level has no qualified field authority");
      const auto& reference = *fields.front().values;
      if (reference.layout() != layout.patches() ||
          reference.distribution() != layout.distribution())
        throw std::invalid_argument("native AMR Tagger field layout is unauthenticated");
      const std::size_t rank_count = layout.distribution().rank_space().size();
      if (communicator.size() < 1 || communicator.rank() < 0 ||
          static_cast<std::size_t>(communicator.size()) != rank_count ||
          layout.distribution().rank_space().linear_rank(reference.local_rank()) !=
              static_cast<std::size_t>(communicator.rank()))
        throw std::invalid_argument(
            "native AMR Tagger rank coordinate differs from its communicator rank space");
      std::vector<tagging_detail::PreparedTaggingFieldContract<Dim>> level_field_contracts;
      level_field_contracts.reserve(fields.size());
      for (const Field& field : fields) {
        if (field.values == nullptr || field.qualified_identity.empty() ||
            field.values->layout() != reference.layout() ||
            field.values->distribution() != reference.distribution() ||
            field.values->local_rank() != reference.local_rank() ||
            field.values->local_global_indices() != reference.local_global_indices())
          throw std::invalid_argument(
              "native AMR Tagger fields do not share one exact qualified layout");
        if (std::any_of(level_field_contracts.begin(), level_field_contracts.end(),
                        [&](const auto& existing) {
                          return existing.qualified_identity == field.qualified_identity;
                        }))
          throw std::invalid_argument("native AMR Tagger field identity is not unique");
        level_field_contracts.push_back(
            {field.qualified_identity, field.values->ncomp(), field.values->ghosts()});
      }
      field_contracts.push_back(std::move(level_field_contracts));
      std::size_t local_cells = 0;
      for (std::size_t local = 0; local < reference.local_size(); ++local)
        local_cells = checked_sum_(local_cells, checked_cell_count_(reference.box(local)));
      std::size_t required_scratch = checked_product_(local_cells, std::size_t{8});
      if (host_staging)
        for (const Field& field : fields)
          for (std::size_t local = 0; local < field.values->local_size(); ++local)
            required_scratch = checked_sum_(
                required_scratch, checked_product_(field.values->fab(local).size(), sizeof(Real)));
      if (required_scratch > budgets[level_index].scratch_bytes)
        throw std::length_error("native AMR Tagger exceeds its explicit scratch budget");
      const std::size_t consensus_bytes =
          layout.distribution().replicated() ? checked_product_(local_cells, 2u) : 0u;
      if (consensus_bytes > budgets[level_index].replicated_consensus_bytes)
        throw std::length_error(
            "native AMR Tagger exceeds its explicit replicated-consensus budget");
      storage->levels.emplace_back(layout, reference.local_rank(), budgets[level_index],
                                   local_cells);
      Level& level = storage->levels.back();
      level.patches.reserve(reference.local_size());
      for (std::size_t local = 0; local < reference.local_size(); ++local) {
        const std::size_t global = reference.global_index(local);
        level.patches.emplace_back();
        Patch& patch = level.patches.back();
        patch.box = reference.box(local);
        patch.global_patch = global;
        if (!layout.domain().contains(patch.box) || patch.box != layout.patches()[global])
          throw std::invalid_argument(
              "native AMR Tagger local patch differs from its authenticated level");
        patch.patch_identity = storage->spec.layout_identity + "/level/" +
                               std::to_string(level_index) + "/patch/" + std::to_string(global);
        patch.state_identities.reserve(fields.size());
        for (const Field& field : fields)
          patch.state_identities.push_back(field.qualified_identity);
        patch.states.reserve(fields.size());
        patch.source_state_values.reserve(fields.size());
        patch.host_state_values.reserve(host_staging ? fields.size() : 0);
        patch.host_states.reserve(host_staging ? fields.size() : 0);
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
          const auto& fab = fields[field_index].values->fab(local);
          patch.source_state_values.emplace_back(fab.storage());
          patch.states.push_back(qualified_field_view_(
              *fields[field_index].values, local, patch.state_identities[field_index],
              storage->spec.layout_identity, patch.patch_identity,
              storage->spec.execution->view().memory_space));
          if (host_staging) {
            patch.host_state_values.push_back(Kokkos::create_mirror_view(fab.storage()));
            patch.host_states.push_back(qualified_field_view_(
                *fields[field_index].values, local, patch.state_identities[field_index],
                storage->spec.layout_identity, patch.patch_identity, POPS_MEMORY_SPACE_HOST_V1,
                patch.host_state_values.back().data()));
          }
        }
        const std::size_t cells = checked_cell_count_(patch.box);
        for (std::size_t output = 0; output < patch.device_outputs.size(); ++output) {
          patch.device_outputs[output] = DeviceMaskView("pops_amr_tagger_device_mask", cells);
          patch.host_outputs[output] = HostMaskView("pops_amr_tagger_host_mask", cells);
        }
      }
    }
    prepare_program_views_(*storage);
    storage->state = prepare_component_state_(*storage);
    contract.bytes(tagging_detail::exact_program_contract(program, field_contracts, layouts,
                                                          topology_generation));
    storage->collective_contract = std::move(contract).release();
    return result;
  }

  static void prepare_program_views_(Storage& storage) {
    storage.abi_leaves.reserve(storage.program.leaves.size());
    for (const auto& leaf : storage.program.leaves)
      storage.abi_leaves.push_back({sizeof(PopsTaggingLeafV1), leaf.state_index, leaf.component,
                                    leaf.opcode, leaf.threshold, leaf.stencil_index});
    storage.abi_axes.resize(storage.program.stencils.size());
    storage.abi_stencils.reserve(storage.program.stencils.size());
    for (std::size_t stencil_index = 0; stencil_index < storage.program.stencils.size();
         ++stencil_index) {
      const auto& stencil = storage.program.stencils[stencil_index];
      auto& axes = storage.abi_axes[stencil_index];
      for (int axis = 0; axis < Dim; ++axis) {
        const auto& source = stencil.axes[static_cast<std::size_t>(axis)];
        axes[static_cast<std::size_t>(axis)] = {sizeof(PopsTaggingAxisStencilV1),
                                                source.axis,
                                                source.derivative_order,
                                                source.formal_order,
                                                source.ghost_lower,
                                                source.ghost_upper,
                                                source.offsets.size(),
                                                source.offsets.data(),
                                                source.coefficients.data()};
      }
      storage.abi_stencils.push_back({sizeof(PopsTaggingStencilV1), stencil.identity.c_str(),
                                      stencil.route.c_str(), stencil.norm.c_str(),
                                      stencil.scale.c_str(), stencil.boundary_mode.c_str(), Dim,
                                      static_cast<std::size_t>(Dim), axes.data()});
    }
  }

  static PopsTaggingProgramV1 program_view_(const Storage& storage) {
    return {sizeof(PopsTaggingProgramV1),       storage.program.provider_identity.c_str(),
            storage.abi_stencils.size(),        storage.abi_stencils.data(),
            storage.abi_leaves.size(),          storage.abi_leaves.data(),
            storage.program.refine_ops.size(),  storage.program.refine_ops.data(),
            storage.program.refine_args.data(), storage.program.coarsen_ops.size(),
            storage.program.coarsen_ops.data(), storage.program.coarsen_args.data(),
            storage.program.minimum_cycles,     storage.program.equality_policy,
            storage.program.conflict_policy,    storage.program.non_finite_policy};
  }

  static PopsQualifiedConstFieldV1 qualified_field_view_(const MultiFab<Dim, MemorySpace>& field,
                                                         std::size_t local,
                                                         const std::string& qualified_identity,
                                                         const std::string& layout_identity,
                                                         const std::string& patch_identity,
                                                         PopsMemorySpaceV1 memory_space,
                                                         const Real* staged_data = nullptr) {
    const auto view = field.fab(local).view();
    PopsConstFieldViewV1 values{};
    values.struct_size = sizeof(PopsConstFieldViewV1);
    values.data = staged_data == nullptr ? view.data : staged_data;
    values.dimension = Dim;
    for (int axis = 0; axis < 3; ++axis) {
      values.extents[axis] = 1;
      values.axis_strides[axis] = 0;
      values.ghost_lower[axis] = 0;
      values.ghost_upper[axis] = 0;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      values.extents[axis] = static_cast<std::size_t>(view.extents[axis]);
      values.axis_strides[axis] = view.strides[axis];
      values.ghost_lower[axis] = static_cast<std::size_t>(field.ghosts()[axis]);
      values.ghost_upper[axis] = static_cast<std::size_t>(field.ghosts()[axis]);
    }
    values.component_count = static_cast<std::size_t>(view.ncomp);
    values.component_stride = view.component_stride;
    values.centering = POPS_FIELD_CENTERING_CELL_V1;
    values.scalar_type = POPS_SCALAR_FLOAT64_V1;
    values.memory_space = memory_space;
    values.layout_identity = layout_identity.c_str();
    values.patch_identity = patch_identity.c_str();
    values.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
    return {sizeof(PopsQualifiedConstFieldV1), 1, qualified_identity.c_str(), values};
  }

  template <class View>
  static PopsTaggerMaskViewV2 mask_view_(View& values, PopsMemorySpaceV1 memory_space) {
    return {sizeof(PopsTaggerMaskViewV2), values.data(), values.extent(0), memory_space,
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  static void execute_patch_(Storage& storage, Patch& patch, const Box<Dim>& domain,
                             const std::array<Real, Dim>& spacing, std::size_t level,
                             std::int64_t tick, double physical_time) {
    const bool host_staging = storage.spec.execution_mode == POPS_TAGGER_EXECUTION_HOST_V2;
    if (host_staging) {
      if (patch.host_state_values.size() != patch.source_state_values.size() ||
          patch.host_states.size() != patch.states.size())
        throw std::logic_error("native AMR Tagger host staging plan is incomplete");
      for (std::size_t state = 0; state < patch.source_state_values.size(); ++state)
        Kokkos::deep_copy(patch.host_state_values[state], patch.source_state_values[state]);
      for (auto& output : patch.host_outputs)
        Kokkos::deep_copy(output, std::uint8_t{0});
    } else {
      for (auto& output : patch.device_outputs)
        Kokkos::deep_copy(output, std::uint8_t{0});
    }
    PopsTaggerRequestV2 request{};
    request.struct_size = sizeof(PopsTaggerRequestV2);
    request.execution_mode = storage.spec.execution_mode;
    request.collective_scope = POPS_TAGGER_COLLECTIVE_NONE_V2;
    request.state_count = host_staging ? patch.host_states.size() : patch.states.size();
    request.states = host_staging ? patch.host_states.data() : patch.states.data();
    request.program = program_view_(storage);
    for (int axis = 0; axis < 3; ++axis) {
      request.patch_lower[axis] = 0;
      request.domain_lower[axis] = 0;
      request.domain_upper[axis] = 0;
      request.cell_size[axis] = 0.0;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      request.patch_lower[axis] = patch.box.lo[axis];
      request.domain_lower[axis] = domain.lo[axis];
      request.domain_upper[axis] = domain.hi[axis];
      request.cell_size[axis] = static_cast<double>(spacing[axis]);
    }
    request.periodic_axes = storage.periodic_axes;
    const PopsMemorySpaceV1 memory_space = storage.spec.execution->view().memory_space;
    request.refine_candidates = host_staging ? mask_view_(patch.host_outputs[0], memory_space)
                                             : mask_view_(patch.device_outputs[0], memory_space);
    request.coarsen_candidates = host_staging ? mask_view_(patch.host_outputs[1], memory_space)
                                              : mask_view_(patch.device_outputs[1], memory_space);
    request.refine_equalities = host_staging ? mask_view_(patch.host_outputs[2], memory_space)
                                             : mask_view_(patch.device_outputs[2], memory_space);
    request.coarsen_equalities = host_staging ? mask_view_(patch.host_outputs[3], memory_space)
                                              : mask_view_(patch.device_outputs[3], memory_space);
    request.logical_time = {sizeof(PopsLogicalTimeV1),
                            storage.spec.clock_identity.c_str(),
                            tick,
                            static_cast<std::int32_t>(level),
                            0,
                            0,
                            0,
                            1,
                            0.0,
                            physical_time};
    request.execution = storage.spec.execution->view();
    PopsComponentStatusV1 status = component::unwritten_component_status();
    const auto& api = storage.component->template table<PopsTaggerApiV2>(
        POPS_NATIVE_INTERFACE_TAGGER_V2, storage.spec.interface_version);
    const int code = component::tag_batch(api, storage.state.get(), request, status);
    require_component_status_(code, status, "amr_tagger");
    if (!host_staging) {
      device_fence();
      for (std::size_t output = 0; output < patch.device_outputs.size(); ++output)
        Kokkos::deep_copy(patch.host_outputs[output], patch.device_outputs[output]);
    }
    for (const auto& output : patch.host_outputs)
      for (std::size_t point = 0; point < output.extent(0); ++point)
        if (output(point) > std::uint8_t{1})
          throw std::runtime_error("native AMR Tagger returned a non-binary candidate mask");
    if (host_staging) {
      for (std::size_t output = 0; output < patch.device_outputs.size(); ++output)
        Kokkos::deep_copy(patch.device_outputs[output], patch.host_outputs[output]);
      device_fence();
    }
  }

  static std::size_t checked_cell_count_(const Box<Dim>& box) {
    const std::int64_t cells = box.numPts();
    if (cells < 0 || static_cast<std::uint64_t>(cells) >
                         static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::length_error("native AMR Tagger patch cell count exceeds size_t");
    return static_cast<std::size_t>(cells);
  }

  static std::size_t checked_sum_(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
      throw std::length_error("native AMR Tagger scratch count exceeds size_t");
    return left + right;
  }

  static std::size_t checked_product_(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::length_error("native AMR Tagger consensus count exceeds size_t");
    return left * right;
  }

  template <class Function>
  static void for_each_host_index_(const Box<Dim>& box, Function&& function) {
    const std::size_t cells = checked_cell_count_(box);
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      Index<Dim> index{};
      std::size_t quotient = ordinal;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t length = static_cast<std::size_t>(box.length(axis));
        index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                       static_cast<std::int64_t>(quotient % length));
        quotient /= length;
      }
      function(index, ordinal);
    }
  }

  static constexpr bool execution_memory_matches_(PopsMemorySpaceV1 claimed) {
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
      return claimed == POPS_MEMORY_SPACE_HOST_V1;
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible)
      return claimed == POPS_MEMORY_SPACE_MANAGED_V1;
    return claimed == POPS_MEMORY_SPACE_DEVICE_V1;
  }

  // Storage is direct to retain one allocation: the facade only allocates this session.
  Storage storage{};
  Storage* storage_ = &storage;
};

}  // namespace pops::runtime::amr::detail
