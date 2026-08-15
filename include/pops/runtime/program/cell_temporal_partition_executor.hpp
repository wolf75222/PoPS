#pragma once

/// @file
/// @brief Prepared rung-batched execution for one transactional cell-local temporal partition.

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#define POPS_CELL_TEMPORAL_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION
#else
#define POPS_CELL_TEMPORAL_INLINE_FUNCTION inline
#endif

namespace pops::runtime::program {

/// Compile-time proof that one provider owns both operations required by a cell-local stage.
///
/// The executor deliberately accepts no independent ``has_stage`` or ``has_ledger`` flags.  A
/// provider must expose this exact tag and one combined device call, so a clock cannot advance after
/// evaluating a local stage without also invoking the provider-owned space-time flux transaction.
struct PreparedCellTemporalStageFluxContractV1 {};

struct CellTemporalAttemptDescriptor {
  std::uint64_t topology_epoch = 0;
  std::int64_t begin_tick = 0;
  std::int64_t target_tick = 0;
  std::int64_t tick_denominator = 1;
  std::size_t cell_count = 0;
};

/// Exact local time passed to the prepared numerical provider.
struct CellTemporalStagePoint {
  std::size_t record_index = 0;
  int level = 0;
  std::uint64_t cell = 0;
  int rung = 0;
  std::int64_t begin_tick = 0;
  std::int64_t end_tick = 0;
  std::int64_t tick_denominator = 1;
};

/// Host-side identity of one prepared same-rung launch.
///
/// A numerical provider that needs a coherent read-only stage image (for example a finite-volume
/// residual assembled from neighbouring cells) may use this descriptor to materialize that image
/// before the combined per-cell stage/flux operation.  It carries no rank-local pointers and is
/// therefore also part of the reviewable provider protocol rather than an executor side channel.
struct CellTemporalRungBatchDescriptor {
  int rung = 0;
  std::int64_t begin_tick = 0;
  std::int64_t end_tick = 0;
  std::int64_t tick_denominator = 1;
  std::size_t cell_count = 0;
  std::size_t local_cell_count = 0;
};

enum class CellTemporalStageDisposition : std::uint32_t {
  Accepted = 0,
  Rejected = 1,
  Failed = 2,
};

/// Result of the combined local-stage and space-time-flux operation.
///
/// ``Accepted`` means that the provider evaluated the stage at the exact rational time in
/// ``CellTemporalStagePoint`` and recorded its attempt-local, time-integrated interface flux.
/// Rejections and failures must carry a stable non-zero provider-owned reason code.
struct CellTemporalStageOutcome {
  CellTemporalStageDisposition disposition = CellTemporalStageDisposition::Accepted;
  std::uint32_t reason_code = 0;

  [[nodiscard]] POPS_CELL_TEMPORAL_INLINE_FUNCTION static constexpr CellTemporalStageOutcome
  accepted() noexcept {
    return {};
  }
  [[nodiscard]] POPS_CELL_TEMPORAL_INLINE_FUNCTION static constexpr CellTemporalStageOutcome
  rejected(std::uint32_t reason) noexcept {
    return {CellTemporalStageDisposition::Rejected, reason};
  }
  [[nodiscard]] POPS_CELL_TEMPORAL_INLINE_FUNCTION static constexpr CellTemporalStageOutcome failed(
      std::uint32_t reason) noexcept {
    return {CellTemporalStageDisposition::Failed, reason};
  }
};

class CellTemporalStageFailure : public std::runtime_error {
 public:
  CellTemporalStageFailure(CellTemporalStageDisposition disposition, std::uint32_t reason_code)
      : std::runtime_error(message_(disposition, reason_code)),
        disposition_(disposition),
        reason_code_(reason_code) {}

  [[nodiscard]] CellTemporalStageDisposition disposition() const noexcept { return disposition_; }
  [[nodiscard]] std::uint32_t reason_code() const noexcept { return reason_code_; }

 private:
  static std::string message_(CellTemporalStageDisposition disposition, std::uint32_t reason_code) {
    const char* kind =
        disposition == CellTemporalStageDisposition::Rejected ? "rejected" : "failed";
    return "cell-local temporal stage " + std::string(kind) + " with provider reason code " +
           std::to_string(reason_code);
  }

  CellTemporalStageDisposition disposition_;
  std::uint32_t reason_code_;
};

template <class DeviceView>
concept CellTemporalStageFluxDeviceView =
    std::is_trivially_copyable_v<DeviceView> &&
    requires(const DeviceView& view, CellTemporalStagePoint point) {
      {
        view.evaluate_local_stage_and_record_space_time_flux(point)
      } noexcept -> std::same_as<CellTemporalStageOutcome>;
    };

template <class Provider>
using CellTemporalStageFluxDeviceViewType = decltype(std::declval<const Provider&>().device_view());

/// Host/device contract consumed by ``PreparedBatchedCellTemporalExecutor``.
///
/// ``begin_attempt`` binds provider-owned scratch prepared before the hot rung loop.  Device calls
/// may mutate only that scratch. ``prepare_commit_attempt`` re-authenticates every external
/// topology/storage authority after the final device fence and before accepted publication.
/// ``commit_attempt`` publishes only after that support decision and every local clock reaches the
/// barrier; ``rollback_attempt`` discards scratch after any rejection or exception.
template <class Provider>
concept CellTemporalStageFluxProvider = requires(Provider& provider, const Provider& const_provider,
                                                 ExactContractBuilder& contract,
                                                 CellTemporalAttemptDescriptor attempt) {
  { Provider::provider_identity() } noexcept -> std::same_as<PreparedProviderIdentity>;
  {
    Provider::stage_flux_contract()
  } noexcept -> std::same_as<PreparedCellTemporalStageFluxContractV1>;
  { const_provider.serialize_exact_parameters(contract) } -> std::same_as<void>;
  { provider.begin_attempt(attempt) } noexcept -> std::same_as<PreparedProviderSupport>;
  { provider.prepare_commit_attempt() } noexcept -> std::same_as<PreparedProviderSupport>;
  { provider.commit_attempt() } noexcept -> std::same_as<void>;
  { provider.rollback_attempt() } noexcept -> std::same_as<void>;
  { const_provider.device_view() } noexcept;
} && CellTemporalStageFluxDeviceView<CellTemporalStageFluxDeviceViewType<Provider>>;

/// Optional lifecycle for providers whose stage uses neighbouring cells.
///
/// Both hooks are required together. ``begin_rung_batch`` may assemble a provider-owned immutable
/// stage image and may throw before the device launch. ``complete_rung_batch`` only rotates already
/// prepared attempt-local storage and must not publish accepted state. Publication remains solely in
/// ``commit_attempt`` after the synchronization barrier.
template <class Provider>
concept CellTemporalRungBatchLifecycle =
    requires(Provider& provider, CellTemporalRungBatchDescriptor batch) {
      { provider.begin_rung_batch(batch) } -> std::same_as<void>;
      { provider.complete_rung_batch(batch) } noexcept -> std::same_as<void>;
    };

/// Optional accepted-boundary resynchronization used when an outer Program transaction restores
/// native state and accepted checkpoint bytes after this executor had already committed locally.
///
/// The executor validates the complete immutable prepared layout before invoking this hook.  The
/// provider therefore only rebinds its accepted logical clock; it must not allocate, touch the live
/// numerical state or publish fluxes.
template <class Provider>
concept CellTemporalAcceptedBoundaryLifecycle =
    requires(Provider& provider, const CellTemporalPartitionAcceptedState& accepted) {
      { provider.restore_accepted_boundary(accepted) } noexcept -> std::same_as<void>;
    };

/// Optional distributed ownership protocol. The accepted partition remains identical on every
/// rank, while only the canonical record indices returned here enter the local device kernel.
/// Collectives always use the provider's borrowed runtime-owned communicator.
template <class Provider>
concept DistributedCellTemporalStageFluxProvider = requires(const Provider& provider) {
  { provider.communicator() } noexcept -> std::same_as<CommunicatorView>;
  { provider.local_record_indices() } noexcept -> std::same_as<std::span<const std::size_t>>;
};

struct CellTemporalExecutionStats {
  /// Number of combined stage/ledger kernels (or host batches without Kokkos), never per-cell.
  std::uint64_t rung_batch_launches = 0;
  std::uint64_t stage_evaluations = 0;
};

namespace cell_temporal_detail {

inline std::string canonical_provider_identity(PreparedProviderIdentity identity) {
  if (identity.name.empty() || identity.version == 0)
    throw std::invalid_argument(
        "cell-local temporal provider requires a non-empty name and non-zero version");
  return std::string(identity.name) + "@" + std::to_string(identity.version);
}

inline std::string exact_execution_contract(const CellTemporalPartitionAcceptedState& state,
                                            const auto& provider) {
  ExactContractBuilder provider_parameters;
  provider.serialize_exact_parameters(provider_parameters);
  ExactContractBuilder contract;
  contract.text("pops.cell-temporal-partition-executor")
      .scalar(std::uint32_t{1})
      .text(state.provider_identity)
      .scalar(state.topology_epoch)
      .scalar(state.synchronization_tick)
      .scalar(state.tick_denominator)
      .sequence(state.cells,
                [](ExactContractBuilder& item, const CellTemporalPartitionRecord& cell) {
                  item.scalar(std::int32_t{cell.level})
                      .scalar(cell.cell)
                      .scalar(std::int32_t{cell.rung})
                      .scalar(cell.accepted_tick);
                })
      .bytes(provider_parameters.view());
  return std::move(contract).release();
}

struct DeviceCellTemporalRecord {
  int level = 0;
  std::uint64_t cell = 0;
  int rung = 0;
};

inline constexpr std::uint32_t kMalformedOutcomeReason = std::numeric_limits<std::uint32_t>::max();

template <CellTemporalStageFluxDeviceView DeviceView>
struct EvaluateRungBatch {
  const DeviceCellTemporalRecord* records = nullptr;
  const std::size_t* record_indices = nullptr;
  std::int64_t* pending_ticks = nullptr;
  std::size_t batch_offset = 0;
  std::int64_t begin_tick = 0;
  std::int64_t end_tick = 0;
  std::int64_t tick_denominator = 1;
  DeviceView provider;

  [[nodiscard]] POPS_CELL_TEMPORAL_INLINE_FUNCTION static constexpr std::uint64_t encode_outcome(
      CellTemporalStageOutcome outcome) noexcept {
    const std::uint32_t disposition = static_cast<std::uint32_t>(outcome.disposition);
    const bool malformed =
        disposition > static_cast<std::uint32_t>(CellTemporalStageDisposition::Failed) ||
        ((disposition == 0) != (outcome.reason_code == 0));
    const std::uint32_t encoded_disposition =
        malformed ? static_cast<std::uint32_t>(CellTemporalStageDisposition::Failed) : disposition;
    const std::uint32_t encoded_reason = malformed ? kMalformedOutcomeReason : outcome.reason_code;
    return (static_cast<std::uint64_t>(encoded_disposition) << 32u) | encoded_reason;
  }

  POPS_CELL_TEMPORAL_INLINE_FUNCTION void operator()(std::int64_t local_index,
                                                     std::uint64_t& aggregate) const noexcept {
    const std::size_t record_index =
        record_indices[batch_offset + static_cast<std::size_t>(local_index)];
    const DeviceCellTemporalRecord& record = records[record_index];
    const CellTemporalStagePoint point{record_index, record.level, record.cell,     record.rung,
                                       begin_tick,   end_tick,     tick_denominator};
    const CellTemporalStageOutcome outcome =
        provider.evaluate_local_stage_and_record_space_time_flux(point);
    const std::uint64_t encoded = encode_outcome(outcome);
    if (encoded == 0)
      pending_ticks[record_index] = end_tick;
    if (encoded > aggregate)
      aggregate = encoded;
  }
};

}  // namespace cell_temporal_detail

/// Prepared executor for bounded cell-local rungs.
///
/// Preparation groups canonical cell records into compact device-accessible rung arrays.  One
/// combined stage/ledger kernel is launched for each active rung event, independently of the number
/// of cells in that rung.  All clocks and provider flux records remain attempt-local until
/// ``commit``; any provider rejection automatically rolls back the complete attempt.
template <CellTemporalStageFluxProvider Provider>
class PreparedBatchedCellTemporalExecutor {
 public:
  PreparedBatchedCellTemporalExecutor(CellTemporalPartitionAcceptedState accepted,
                                      Provider provider)
      : provider_(std::move(provider)),
        partition_(std::move(accepted)),
        provider_identity_(
            cell_temporal_detail::canonical_provider_identity(Provider::provider_identity())),
        exact_contract_(
            cell_temporal_detail::exact_execution_contract(partition_.accepted_state(), provider_)),
        records_(partition_.accepted_state().cells.size()),
        pending_ticks_(partition_.accepted_state().cells.size()) {
    partition_.require_prepared_execution_route(provider_identity_);
    prepare_batches_();
  }

  PreparedBatchedCellTemporalExecutor(const PreparedBatchedCellTemporalExecutor&) = delete;
  PreparedBatchedCellTemporalExecutor& operator=(const PreparedBatchedCellTemporalExecutor&) =
      delete;
  PreparedBatchedCellTemporalExecutor(PreparedBatchedCellTemporalExecutor&&) = delete;
  PreparedBatchedCellTemporalExecutor& operator=(PreparedBatchedCellTemporalExecutor&&) = delete;

  ~PreparedBatchedCellTemporalExecutor() { rollback(); }

  [[nodiscard]] const std::string& provider_identity() const noexcept { return provider_identity_; }
  [[nodiscard]] const std::string& exact_contract() const noexcept { return exact_contract_; }
  [[nodiscard]] const CellTemporalPartitionAcceptedState& accepted_state() const noexcept {
    return partition_.accepted_state();
  }
  [[nodiscard]] CellTemporalPartitionAcceptedState checkpoint() const {
    return partition_.checkpoint();
  }
  [[nodiscard]] bool attempt_active() const noexcept { return attempt_active_; }
  [[nodiscard]] std::size_t prepared_rung_count() const noexcept { return batches_.size(); }
  [[nodiscard]] const CellTemporalExecutionStats& stats() const noexcept { return stats_; }
  [[nodiscard]] static constexpr bool uses_kokkos() noexcept {
#if defined(POPS_HAS_KOKKOS)
    return true;
#else
    return false;
#endif
  }

  /// Resynchronize this prepared executor with an exact accepted barrier restored by its owner.
  ///
  /// Preparation (cell identities, rungs, topology, denominator and provider) is immutable.  A
  /// rollback may only move the common accepted tick backwards or forwards within that authority.
  /// Providers without the explicit lifecycle hook cannot be safely retained and fail closed.
  void restore_accepted_boundary(CellTemporalPartitionAcceptedState accepted) {
    if (attempt_active_)
      throw std::logic_error("cell-local temporal executor cannot restore an active attempt");
    validate_cell_temporal_partition_state(accepted);
    BatchedCellTemporalPartition candidate(accepted);
    candidate.require_prepared_execution_route(provider_identity_);
    const CellTemporalPartitionAcceptedState& current = partition_.accepted_state();
    if (accepted.kind != current.kind || accepted.provider_identity != current.provider_identity ||
        accepted.topology_epoch != current.topology_epoch ||
        accepted.tick_denominator != current.tick_denominator ||
        accepted.cells.size() != current.cells.size())
      throw std::invalid_argument(
          "cell-local temporal executor restore targets another prepared authority");
    for (std::size_t index = 0; index < accepted.cells.size(); ++index) {
      const CellTemporalPartitionRecord& next = accepted.cells[index];
      const CellTemporalPartitionRecord& prepared = current.cells[index];
      if (next.level != prepared.level || next.cell != prepared.cell || next.rung != prepared.rung)
        throw std::invalid_argument(
            "cell-local temporal executor restore changes a prepared cell or rung");
    }
    if constexpr (!CellTemporalAcceptedBoundaryLifecycle<Provider>) {
      throw std::logic_error(
          "cell-local temporal provider cannot resynchronize an accepted rollback boundary");
    } else {
      std::string restored_contract =
          cell_temporal_detail::exact_execution_contract(accepted, provider_);
      provider_.restore_accepted_boundary(accepted);
      partition_.restore(std::move(accepted));
      for (RungBatch& batch : batches_)
        batch.current_tick = partition_.accepted_state().synchronization_tick;
      for (std::size_t index = 0; index < partition_.accepted_state().cells.size(); ++index)
        pending_ticks_[index] = partition_.accepted_state().cells[index].accepted_tick;
      target_tick_ = 0;
      exact_contract_ = std::move(restored_contract);
    }
  }

  void begin_attempt(std::int64_t target_tick) {
    const bool had_active_attempt = attempt_active_;
    std::exception_ptr partition_error;
    if (had_active_attempt) {
      partition_error = std::make_exception_ptr(
          std::logic_error("cell-local temporal executor attempt is already active"));
    } else {
      try {
        partition_.begin_attempt(target_tick);
      } catch (...) {
        partition_error = std::current_exception();
      }
    }
    try {
      require_phase_consensus_("attempt clock preparation", partition_error);
    } catch (...) {
      if (had_active_attempt)
        abort_attempt_();
      else
        partition_.rollback();
      throw;
    }
    const CellTemporalPartitionAcceptedState& accepted = partition_.accepted_state();
    for (std::size_t index = 0; index < accepted.cells.size(); ++index)
      pending_ticks_[index] = accepted.cells[index].accepted_tick;
    for (RungBatch& batch : batches_)
      batch.current_tick = accepted.synchronization_tick;

    const CellTemporalAttemptDescriptor descriptor{
        accepted.topology_epoch, accepted.synchronization_tick, target_tick,
        accepted.tick_denominator, accepted.cells.size()};
    const PreparedProviderSupport support = provider_.begin_attempt(descriptor);
    const long local_refusal = !support.well_formed() || !support.accepted() ? 1L : 0L;
    if (all_reduce_max(local_refusal, communicator_()) != 0) {
      provider_.rollback_attempt();
      partition_.rollback();
      const std::string reason =
          local_refusal == 0
              ? "another rank refused the prepared attempt"
              : (!support.well_formed() ? "malformed prepared-provider support decision"
                                        : std::string(support.reason));
      throw std::runtime_error("cell-local temporal provider refused attempt preparation: " +
                               reason);
    }
    target_tick_ = target_tick;
    attempt_active_ = true;
  }

  /// Execute every local rung event needed to reach the declared synchronization barrier.
  void advance_to_barrier() {
    if (!attempt_active_)
      throw std::logic_error("cell-local temporal execution requires an active attempt");
    while (RungBatch* batch = next_batch_())
      execute_batch_(*batch);
    std::exception_ptr barrier_error;
    try {
      partition_.require_barrier("cell-local temporal executor");
    } catch (...) {
      barrier_error = std::current_exception();
    }
    try {
      require_phase_consensus_("synchronization-barrier validation", barrier_error);
    } catch (...) {
      abort_attempt_();
      throw;
    }
  }

  void commit() {
    if (!attempt_active_)
      throw std::logic_error("cell-local temporal commit requires an active attempt");
    try {
      CellTemporalPartitionAcceptedState next;
      std::string next_exact_contract;
      std::exception_ptr publication_preflight_error;
      try {
        partition_.require_barrier("cell-local temporal provider commit");
        next = partition_.accepted_state();
        next.synchronization_tick = target_tick_;
        for (CellTemporalPartitionRecord& cell : next.cells)
          cell.accepted_tick = target_tick_;
        next_exact_contract = cell_temporal_detail::exact_execution_contract(next, provider_);
      } catch (...) {
        publication_preflight_error = std::current_exception();
      }
      require_phase_consensus_("accepted-publication contract preparation",
                               publication_preflight_error);
      const PreparedProviderSupport support = provider_.prepare_commit_attempt();
      const long local_refusal = !support.well_formed() || !support.accepted() ? 1L : 0L;
      if (all_reduce_max(local_refusal, communicator_()) != 0) {
        const std::string reason =
            local_refusal == 0
                ? "another rank refused accepted publication"
                : (!support.well_formed() ? "malformed prepared-provider support decision"
                                          : std::string(support.reason));
        throw std::runtime_error("cell-local temporal provider refused accepted publication: " +
                                 reason);
      }
      provider_.commit_attempt();
      partition_.commit();
      exact_contract_ = std::move(next_exact_contract);
      target_tick_ = 0;
      attempt_active_ = false;
    } catch (...) {
      abort_attempt_();
      throw;
    }
  }

  void rollback() noexcept {
    if (!attempt_active_)
      return;
    provider_.rollback_attempt();
    partition_.rollback();
    target_tick_ = 0;
    attempt_active_ = false;
  }

 private:
  struct RungBatch {
    int rung = 0;
    std::int64_t stride = 1;
    std::int64_t current_tick = 0;
    std::size_t offset = 0;
    std::vector<std::size_t> indices;
    std::vector<std::size_t> local_indices;
  };

  [[nodiscard]] CommunicatorView communicator_() const noexcept {
    if constexpr (DistributedCellTemporalStageFluxProvider<Provider>)
      return provider_.communicator();
    return CommunicatorView{};
  }

  void require_phase_consensus_(std::string_view phase,
                                const std::exception_ptr& local_error) const {
    const long local_failure = local_error ? 1L : 0L;
    if (all_reduce_max(local_failure, communicator_()) == 0)
      return;
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("cell-local temporal " + std::string(phase) +
                             " failed on another rank");
  }

  void prepare_batches_() {
    const auto& cells = partition_.accepted_state().cells;
    std::map<int, std::vector<std::size_t>> grouped;
    for (std::size_t index = 0; index < cells.size(); ++index) {
      records_[index] = {cells[index].level, cells[index].cell, cells[index].rung};
      pending_ticks_[index] = cells[index].accepted_tick;
      grouped[cells[index].rung].push_back(index);
    }
    record_indices_.reserve(cells.size());
    batches_.reserve(grouped.size());
    std::span<const std::size_t> owned;
    if constexpr (DistributedCellTemporalStageFluxProvider<Provider>) {
      owned = provider_.local_record_indices();
      if (!std::is_sorted(owned.begin(), owned.end()) ||
          std::adjacent_find(owned.begin(), owned.end()) != owned.end() ||
          (!owned.empty() && owned.back() >= cells.size()))
        throw std::invalid_argument(
            "distributed cell-local provider returned invalid canonical local records");
    }
    for (auto& [rung, indices] : grouped) {
      const std::size_t offset = record_indices_.size();
      std::vector<std::size_t> local_indices;
      if constexpr (DistributedCellTemporalStageFluxProvider<Provider>) {
        local_indices.reserve(indices.size());
        std::set_intersection(indices.begin(), indices.end(), owned.begin(), owned.end(),
                              std::back_inserter(local_indices));
      } else {
        local_indices = indices;
      }
      record_indices_.insert(record_indices_.end(), local_indices.begin(), local_indices.end());
      batches_.push_back({rung, std::int64_t{1} << rung,
                          partition_.accepted_state().synchronization_tick, offset,
                          std::move(indices), std::move(local_indices)});
    }
  }

  [[nodiscard]] RungBatch* next_batch_() noexcept {
    RungBatch* selected = nullptr;
    std::int64_t selected_end = std::numeric_limits<std::int64_t>::max();
    for (RungBatch& batch : batches_) {
      if (batch.current_tick >= target_tick_)
        continue;
      const std::int64_t end_tick = batch.current_tick + batch.stride;
      if (end_tick < selected_end ||
          (end_tick == selected_end && (selected == nullptr || batch.rung < selected->rung))) {
        selected = &batch;
        selected_end = end_tick;
      }
    }
    return selected;
  }

  void abort_attempt_() noexcept {
    provider_.rollback_attempt();
    partition_.rollback();
    target_tick_ = 0;
    attempt_active_ = false;
  }

  void execute_batch_(RungBatch& batch) {
    const std::int64_t begin_tick = batch.current_tick;
    const std::int64_t end_tick = begin_tick + batch.stride;
    if (end_tick > target_tick_) {
      abort_attempt_();
      throw std::logic_error("prepared cell-local rung crosses its synchronization barrier");
    }

    const CellTemporalRungBatchDescriptor descriptor{batch.rung,
                                                     begin_tick,
                                                     end_tick,
                                                     partition_.accepted_state().tick_denominator,
                                                     batch.indices.size(),
                                                     batch.local_indices.size()};
    std::uint64_t aggregate = 0;
    try {
      std::exception_ptr rung_preparation_error;
      try {
        if constexpr (CellTemporalRungBatchLifecycle<Provider>)
          provider_.begin_rung_batch(descriptor);
      } catch (...) {
        rung_preparation_error = std::current_exception();
      }
      require_phase_consensus_("rung-batch preparation", rung_preparation_error);

      using DeviceView = CellTemporalStageFluxDeviceViewType<Provider>;
      const DeviceView view = provider_.device_view();
      const cell_temporal_detail::EvaluateRungBatch<DeviceView> kernel{
          records_.data(),
          record_indices_.data(),
          pending_ticks_.data(),
          batch.offset,
          begin_tick,
          end_tick,
          partition_.accepted_state().tick_denominator,
          view};
      std::exception_ptr kernel_error;
      try {
#if defined(POPS_HAS_KOKKOS)
        using Policy =
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>;
        Kokkos::parallel_reduce("pops_cell_temporal_stage_flux_batch",
                                Policy(0, static_cast<std::int64_t>(batch.local_indices.size())),
                                kernel, Kokkos::Max<std::uint64_t>(aggregate));
        device_fence();
#else
        for (std::size_t index = 0; index < batch.local_indices.size(); ++index)
          kernel(static_cast<std::int64_t>(index), aggregate);
#endif
      } catch (...) {
        kernel_error = std::current_exception();
      }
      require_phase_consensus_("rung-batch device execution", kernel_error);
      ++stats_.rung_batch_launches;
      stats_.stage_evaluations += static_cast<std::uint64_t>(batch.local_indices.size());
      aggregate = all_reduce_max(aggregate, communicator_());
      if (aggregate != 0) {
        const auto disposition =
            static_cast<CellTemporalStageDisposition>(static_cast<std::uint32_t>(aggregate >> 32u));
        const std::uint32_t reason = static_cast<std::uint32_t>(aggregate);
        throw CellTemporalStageFailure(disposition, reason);
      }
      if constexpr (CellTemporalRungBatchLifecycle<Provider>)
        provider_.complete_rung_batch(descriptor);
    } catch (...) {
      abort_attempt_();
      throw;
    }
    std::exception_ptr clock_advance_error;
    try {
      partition_.advance_batch(batch.rung, batch.indices, end_tick);
    } catch (...) {
      clock_advance_error = std::current_exception();
    }
    try {
      require_phase_consensus_("rung-batch clock advance", clock_advance_error);
    } catch (...) {
      abort_attempt_();
      throw;
    }
    batch.current_tick = end_tick;
  }

  Provider provider_;
  BatchedCellTemporalPartition partition_;
  std::string provider_identity_;
  std::string exact_contract_;
  std::vector<cell_temporal_detail::DeviceCellTemporalRecord,
              fab_allocator<cell_temporal_detail::DeviceCellTemporalRecord>>
      records_;
  std::vector<std::size_t, fab_allocator<std::size_t>> record_indices_;
  std::vector<std::int64_t, fab_allocator<std::int64_t>> pending_ticks_;
  std::vector<RungBatch> batches_;
  CellTemporalExecutionStats stats_;
  std::int64_t target_tick_ = 0;
  bool attempt_active_ = false;
};

}  // namespace pops::runtime::program

#undef POPS_CELL_TEMPORAL_INLINE_FUNCTION
