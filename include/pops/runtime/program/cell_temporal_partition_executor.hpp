#pragma once

/// @file
/// @brief Prepared rung-batched execution for one transactional cell-local temporal partition.

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <optional>
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
/// evaluating a local stage without also producing its provider-owned space-time face diagnostics.
/// Conservation remains part of the state candidate; this tag does not authorize a second ledger.
struct PreparedCellTemporalStageFluxContractV1 {};

struct CellTemporalAttemptDescriptor {
  std::uint64_t topology_epoch = 0;
  std::int64_t begin_tick = 0;
  std::int64_t target_tick = 0;
  std::int64_t tick_denominator = 1;
  std::size_t cell_count = 0;
  std::size_t local_cell_count = 0;
};

/// Exact local time passed to the prepared numerical provider.
struct CellTemporalStagePoint {
  std::size_t record_index = 0;
  std::size_t local_record_index = 0;
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
/// ``CellTemporalStagePoint`` and recorded its attempt-local, time-integrated face diagnostics.
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

/// Distributed ownership and exact communicator contract required by the executor.
///
/// The provider exposes canonical indices into the rank-independent accepted record sequence.  The
/// indices name only records owned by this lane rank; an empty span is valid and still participates
/// in every preparation, kernel-outcome and publication phase.
template <class Provider>
concept DistributedCellTemporalStageFluxProvider =
    CellTemporalStageFluxProvider<Provider> && std::is_nothrow_move_constructible_v<Provider> &&
    requires(const Provider& provider) {
      { provider.execution_lane() } noexcept -> std::same_as<const ExecutionLane&>;
      { provider.local_record_indices() } noexcept -> std::same_as<std::span<const std::size_t>>;
    };

/// Optional lifecycle for providers whose stage uses neighbouring cells.
///
/// All hooks are required together. ``prepare_rung_batch_local`` performs only rank-local candidate
/// preparation. The executor reaches lane consensus before
/// ``materialize_rung_batch_snapshot`` enters any halo collective and seals the immutable stage
/// image consumed by the kernel. ``complete_rung_batch`` only rotates already-prepared attempt-local
/// storage and must not publish accepted state. Publication remains solely in ``commit_attempt``
/// after the synchronization barrier.
template <class Provider>
concept CellTemporalRungBatchLifecycle =
    requires(Provider& provider, CellTemporalRungBatchDescriptor batch) {
      { provider.prepare_rung_batch_local(batch) } -> std::same_as<void>;
      { provider.materialize_rung_batch_snapshot(batch) } -> std::same_as<void>;
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
                                            const auto& provider, const ExecutionLane& lane) {
  ExactContractBuilder provider_parameters;
  provider.serialize_exact_parameters(provider_parameters);
  ExactContractBuilder contract;
  contract.text("pops.cell-temporal-partition-executor")
      .scalar(std::uint32_t{2})
      .text(lane.identity())
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

inline std::string sha256_exact_bytes(std::string_view exact_bytes) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(exact_bytes.size());
  for (const char byte : exact_bytes)
    bytes.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
  return identity::sha256_hex(bytes);
}

struct FixedCollectiveEnvelope {
  static constexpr std::size_t digest_size = 64;
  static_assert(std::numeric_limits<std::size_t>::digits <=
                std::numeric_limits<std::uint64_t>::digits);
  static_assert(std::numeric_limits<long>::digits >= 32);
  std::array<long, 11> scalars{};
  std::array<char, digest_size> digest{};
};

inline void encode_u64(std::array<long, 11>& scalars, std::size_t offset,
                       std::uint64_t value) noexcept {
  scalars[offset] = static_cast<long>(value >> 32u);
  scalars[offset + 1] = static_cast<long>(value & 0xffffffffu);
}

inline FixedCollectiveEnvelope fixed_collective_envelope(std::uint32_t schema,
                                                         std::size_t record_count,
                                                         std::uint64_t topology_epoch,
                                                         std::int64_t synchronization_tick,
                                                         std::int64_t tick_denominator,
                                                         std::string_view exact_digest) {
  if (exact_digest.size() != FixedCollectiveEnvelope::digest_size || synchronization_tick < 0 ||
      tick_denominator <= 0)
    throw std::invalid_argument("cell-local temporal fixed envelope is malformed");
  FixedCollectiveEnvelope result;
  result.scalars[0] = static_cast<long>(schema);
  encode_u64(result.scalars, 1, static_cast<std::uint64_t>(record_count));
  encode_u64(result.scalars, 3, topology_epoch);
  encode_u64(result.scalars, 5, static_cast<std::uint64_t>(synchronization_tick));
  encode_u64(result.scalars, 7, static_cast<std::uint64_t>(tick_denominator));
  result.scalars[9] = 0x43454c4cL;
  result.scalars[10] = 0x54494d45L;
  std::copy(exact_digest.begin(), exact_digest.end(), result.digest.begin());
  return result;
}

inline bool fixed_collective_envelope_agrees(const FixedCollectiveEnvelope& envelope,
                                             const ExecutionLane& lane) {
  auto minimum_scalars = envelope.scalars;
  auto maximum_scalars = envelope.scalars;
  auto minimum_digest = envelope.digest;
  auto maximum_digest = envelope.digest;
  all_reduce_min_inplace(minimum_scalars.data(), minimum_scalars.size(), lane);
  all_reduce_max_inplace(maximum_scalars.data(), maximum_scalars.size(), lane);
  all_reduce_min_inplace(minimum_digest.data(), minimum_digest.size(), lane);
  all_reduce_max_inplace(maximum_digest.data(), maximum_digest.size(), lane);
  return minimum_scalars == maximum_scalars && minimum_digest == maximum_digest;
}

struct DeviceCellTemporalRecord {
  std::size_t global_record_index = 0;
  std::size_t provider_local_index = 0;
  int level = 0;
  std::uint64_t cell = 0;
  int rung = 0;
};

static_assert(std::is_trivially_copyable_v<DeviceCellTemporalRecord>);

inline constexpr std::uint32_t kMalformedOutcomeReason = std::numeric_limits<std::uint32_t>::max();

template <CellTemporalStageFluxDeviceView DeviceView>
struct EvaluateRungBatch {
  const DeviceCellTemporalRecord* records = nullptr;
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
    const std::size_t packed_index = batch_offset + static_cast<std::size_t>(local_index);
    const DeviceCellTemporalRecord& record = records[packed_index];
    const CellTemporalStagePoint point{record.global_record_index,
                                       record.provider_local_index,
                                       record.level,
                                       record.cell,
                                       record.rung,
                                       begin_tick,
                                       end_tick,
                                       tick_denominator};
    const CellTemporalStageOutcome outcome =
        provider.evaluate_local_stage_and_record_space_time_flux(point);
    const std::uint64_t encoded = encode_outcome(outcome);
    if (encoded == 0)
      pending_ticks[record.global_record_index] = end_tick;
    if (encoded > aggregate)
      aggregate = encoded;
  }
};

}  // namespace cell_temporal_detail

/// Prepared executor for bounded cell-local rungs.
///
/// Preparation groups canonical cell records into compact device-accessible rung arrays.  One
/// combined stage/diagnostic kernel is launched for each active rung event, independently of the
/// number of cells in that rung.  All clocks and provider diagnostics remain attempt-local until
/// ``commit``; any provider rejection automatically rolls back the complete attempt.
template <DistributedCellTemporalStageFluxProvider Provider>
class PreparedBatchedCellTemporalExecutor {
 public:
  PreparedBatchedCellTemporalExecutor(CellTemporalPartitionAcceptedState accepted,
                                      Provider provider, const ExecutionLane& lane)
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        provider_(std::move(provider)),
        partition_() {
    require_exact_lane_();
    std::optional<BatchedCellTemporalPartition> prepared_partition;
    std::optional<cell_temporal_detail::FixedCollectiveEnvelope> plan_envelope;
    std::exception_ptr local_error;
    try {
      prepared_partition.emplace(std::move(accepted));
      provider_identity_ =
          cell_temporal_detail::canonical_provider_identity(Provider::provider_identity());
      prepared_partition->require_prepared_execution_route(provider_identity_);
      exact_contract_ = cell_temporal_detail::exact_execution_contract(
          prepared_partition->accepted_state(), provider_, *lane_);
      exact_contract_digest_ = cell_temporal_detail::sha256_exact_bytes(exact_contract_);
      plan_envelope.emplace(cell_temporal_detail::fixed_collective_envelope(
          3, prepared_partition->accepted_state().cells.size(),
          prepared_partition->accepted_state().topology_epoch,
          prepared_partition->accepted_state().synchronization_tick,
          prepared_partition->accepted_state().tick_denominator, exact_contract_digest_));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0)
      rethrow_collective_failure_(
          local_error, "cell-local temporal executor preparation failed on another rank");
    if (!cell_temporal_detail::fixed_collective_envelope_agrees(*plan_envelope, *lane_))
      throw std::invalid_argument(
          "cell-local temporal executor fixed plan envelope differs between execution-lane ranks");
    static_assert(std::is_nothrow_move_assignable_v<BatchedCellTemporalPartition>);
    partition_ = std::move(*prepared_partition);
    try {
      prepare_batches_();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0)
      rethrow_collective_failure_(
          local_error, "cell-local temporal ownership preparation failed on another rank");
    require_unique_collective_ownership_();
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
  [[nodiscard]] const ExecutionLane& execution_lane() const noexcept { return *lane_; }
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
    std::optional<BatchedCellTemporalPartition> candidate;
    std::optional<cell_temporal_detail::FixedCollectiveEnvelope> restore_envelope;
    std::string restored_contract;
    std::string restored_digest;
    std::exception_ptr local_error;
    try {
      if (attempt_active_)
        throw std::logic_error("cell-local temporal executor cannot restore an active attempt");
      if constexpr (!CellTemporalAcceptedBoundaryLifecycle<Provider>)
        throw std::logic_error(
            "cell-local temporal provider cannot resynchronize an accepted rollback boundary");
      candidate.emplace(std::move(accepted));
      candidate->require_prepared_execution_route(provider_identity_);
      const CellTemporalPartitionAcceptedState& next = candidate->accepted_state();
      const CellTemporalPartitionAcceptedState& current = partition_.accepted_state();
      if (next.kind != current.kind || next.provider_identity != current.provider_identity ||
          next.topology_epoch != current.topology_epoch ||
          next.tick_denominator != current.tick_denominator ||
          next.cells.size() != current.cells.size())
        throw std::invalid_argument(
            "cell-local temporal executor restore targets another prepared authority");
      for (std::size_t index = 0; index < next.cells.size(); ++index) {
        const CellTemporalPartitionRecord& restored = next.cells[index];
        const CellTemporalPartitionRecord& prepared = current.cells[index];
        if (restored.level != prepared.level || restored.cell != prepared.cell ||
            restored.rung != prepared.rung)
          throw std::invalid_argument(
              "cell-local temporal executor restore changes a prepared cell or rung");
      }
      restored_contract = cell_temporal_detail::exact_execution_contract(next, provider_, *lane_);
      restored_digest = cell_temporal_detail::sha256_exact_bytes(restored_contract);
      restore_envelope.emplace(cell_temporal_detail::fixed_collective_envelope(
          4, next.cells.size(), next.topology_epoch, next.synchronization_tick,
          next.tick_denominator, restored_digest));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      rollback();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal restore preparation failed on another rank");
    }
    if (!cell_temporal_detail::fixed_collective_envelope_agrees(*restore_envelope, *lane_))
      throw std::invalid_argument(
          "cell-local temporal restore target differs between execution-lane ranks");

    static_assert(std::is_nothrow_move_assignable_v<BatchedCellTemporalPartition>);
    static_assert(std::is_nothrow_move_assignable_v<std::string>);
    const CellTemporalPartitionAcceptedState& next = candidate->accepted_state();
    if constexpr (CellTemporalAcceptedBoundaryLifecycle<Provider>)
      provider_.restore_accepted_boundary(next);
    else
      std::terminate();
    partition_ = std::move(*candidate);
    const CellTemporalPartitionAcceptedState& published = partition_.accepted_state();
    for (RungBatch& batch : batches_)
      batch.current_tick = published.synchronization_tick;
    for (std::size_t index = 0; index < published.cells.size(); ++index)
      pending_ticks_[index] = published.cells[index].accepted_tick;
    target_tick_ = 0;
    exact_contract_ = std::move(restored_contract);
    exact_contract_digest_ = std::move(restored_digest);
  }

  void begin_attempt(std::int64_t target_tick) {
    const long active_local = attempt_active_ ? 1L : 0L;
    const long active_minimum = all_reduce_min(active_local, *lane_);
    const long active_maximum = all_reduce_max(active_local, *lane_);
    if (active_maximum != 0) {
      if (active_minimum != active_maximum)
        rollback();
      throw std::logic_error("cell-local temporal attempt is already active");
    }
    require_collective_target_(target_tick);
    std::exception_ptr local_error;
    try {
      partition_.begin_attempt(target_tick);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      partition_.rollback();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal clock preparation failed on another rank");
    }
    const CellTemporalPartitionAcceptedState& accepted = partition_.accepted_state();
    for (std::size_t index = 0; index < accepted.cells.size(); ++index)
      pending_ticks_[index] = accepted.cells[index].accepted_tick;
    for (RungBatch& batch : batches_)
      batch.current_tick = accepted.synchronization_tick;

    const CellTemporalAttemptDescriptor descriptor{
        accepted.topology_epoch,   accepted.synchronization_tick, target_tick,
        accepted.tick_denominator, accepted.cells.size(),         records_.size()};
    const PreparedProviderSupport support = provider_.begin_attempt(descriptor);
    const long refused_local = !support.well_formed() || !support.accepted() ? 1L : 0L;
    if (all_reduce_max(refused_local, *lane_) != 0) {
      provider_.rollback_attempt();
      partition_.rollback();
      if (refused_local != 0) {
        const std::string reason = !support.well_formed()
                                       ? "malformed prepared-provider support decision"
                                       : std::string(support.reason);
        throw std::runtime_error("cell-local temporal provider refused attempt preparation: " +
                                 reason);
      }
      throw std::runtime_error(
          "cell-local temporal provider refused attempt preparation on another rank");
    }
    target_tick_ = target_tick;
    attempt_active_ = true;
  }

  /// Execute every local rung event needed to reach the declared synchronization barrier.
  void advance_to_barrier() {
    if (all_reduce_min(attempt_active_ ? 1L : 0L, *lane_) != 1) {
      rollback();
      throw std::logic_error("cell-local temporal execution requires an active attempt");
    }
    while (RungBatch* batch = next_batch_())
      execute_batch_(*batch);
    std::exception_ptr local_error;
    try {
      partition_.require_barrier("cell-local temporal executor");
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      abort_attempt_();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal barrier validation failed on another rank");
    }
  }

  void commit() {
    std::optional<BatchedCellTemporalPartition> next_partition;
    std::optional<cell_temporal_detail::FixedCollectiveEnvelope> next_envelope;
    CellTemporalPartitionAcceptedState next;
    std::string next_exact_contract;
    std::string next_exact_digest;
    std::exception_ptr local_error;
    try {
      if (!attempt_active_)
        throw std::logic_error("cell-local temporal commit requires an active attempt");
      partition_.require_barrier("cell-local temporal provider commit");
      next = partition_.accepted_state();
      next.synchronization_tick = target_tick_;
      for (CellTemporalPartitionRecord& cell : next.cells)
        cell.accepted_tick = target_tick_;
      next_exact_contract = cell_temporal_detail::exact_execution_contract(next, provider_, *lane_);
      next_exact_digest = cell_temporal_detail::sha256_exact_bytes(next_exact_contract);
      next_partition.emplace(std::move(next));
      const CellTemporalPartitionAcceptedState& prepared = next_partition->accepted_state();
      next_envelope.emplace(cell_temporal_detail::fixed_collective_envelope(
          5, prepared.cells.size(), prepared.topology_epoch, prepared.synchronization_tick,
          prepared.tick_denominator, next_exact_digest));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      abort_attempt_();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal commit preparation failed on another rank");
    }
    if (!cell_temporal_detail::fixed_collective_envelope_agrees(*next_envelope, *lane_)) {
      abort_attempt_();
      throw std::invalid_argument(
          "cell-local temporal commit candidate differs between execution-lane ranks");
    }
    const PreparedProviderSupport support = provider_.prepare_commit_attempt();
    const long refused_local = !support.well_formed() || !support.accepted() ? 1L : 0L;
    if (all_reduce_max(refused_local, *lane_) != 0) {
      abort_attempt_();
      if (refused_local == 0)
        throw std::runtime_error(
            "cell-local temporal provider refused accepted publication on another rank");
      const std::string reason = !support.well_formed()
                                     ? "malformed prepared-provider support decision"
                                     : std::string(support.reason);
      throw std::runtime_error("cell-local temporal provider refused accepted publication: " +
                               reason);
    }
    provider_.commit_attempt();
    static_assert(std::is_nothrow_move_assignable_v<BatchedCellTemporalPartition>);
    static_assert(std::is_nothrow_move_assignable_v<std::string>);
    partition_ = std::move(*next_partition);
    exact_contract_ = std::move(next_exact_contract);
    exact_contract_digest_ = std::move(next_exact_digest);
    target_tick_ = 0;
    attempt_active_ = false;
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
    std::size_t local_offset = 0;
    std::size_t local_count = 0;
    std::vector<std::size_t> global_indices;
  };

  [[noreturn]] static void rethrow_collective_failure_(std::exception_ptr local_error,
                                                       const char* remote_message) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(remote_message);
  }

  void require_exact_lane_() const {
    const ExecutionLane& provider_lane = provider_.execution_lane();
    long local_failure = 0;
    try {
      if (provider_lane.identity() != lane_->identity() || !provider_lane.congruent_with(*lane_))
        local_failure = 1;
    } catch (...) {
      local_failure = 1;
    }
    if (all_reduce_max(local_failure, *lane_) != 0)
      throw std::invalid_argument(
          "cell-local temporal provider and executor require the same exact execution lane");
  }

  void require_unique_collective_ownership_() const {
    std::vector<double> ownership;
    std::exception_ptr local_error;
    try {
      ownership.assign(partition_.accepted_state().cells.size(), 0.0);
      for (const auto& record : records_)
        ownership[record.global_record_index] = 1.0;
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0)
      rethrow_collective_failure_(
          local_error, "cell-local temporal ownership preparation failed on another rank");
    all_reduce_sum_inplace(ownership.data(), ownership.size(), *lane_);
    if (std::any_of(ownership.begin(), ownership.end(),
                    [](double owners) { return owners != 1.0; }))
      throw std::invalid_argument(
          "cell-local temporal records require exactly one execution-lane owner");
  }

  void require_collective_target_(std::int64_t target_tick) const {
    std::string bytes;
    std::exception_ptr local_error;
    try {
      ExactContractBuilder target;
      target.text("pops.cell-temporal-executor-target")
          .scalar(std::uint32_t{1})
          .scalar(target_tick);
      bytes = std::move(target).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0)
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal target preparation failed on another rank");
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"pops.cell-temporal-executor-target", bytes}},
                                                  *lane_))
      throw std::invalid_argument(
          "cell-local temporal target differs between execution-lane ranks");
  }

  void prepare_batches_() {
    const auto& cells = partition_.accepted_state().cells;
    pending_ticks_.assign(cells.size(), 0);
    records_.clear();
    ordered_records_.clear();
    batches_.clear();
    std::map<int, std::vector<std::size_t>> grouped;
    for (std::size_t index = 0; index < cells.size(); ++index) {
      pending_ticks_[index] = cells[index].accepted_tick;
      grouped[cells[index].rung].push_back(index);
    }
    const std::span<const std::size_t> local = provider_.local_record_indices();
    records_.reserve(local.size());
    std::size_t previous = std::numeric_limits<std::size_t>::max();
    for (std::size_t provider_local = 0; provider_local < local.size(); ++provider_local) {
      const std::size_t global = local[provider_local];
      if (global >= cells.size() ||
          (previous != std::numeric_limits<std::size_t>::max() && global <= previous))
        throw std::invalid_argument(
            "cell-local temporal provider ownership must be a canonical record subset");
      const CellTemporalPartitionRecord& cell = cells[global];
      records_.push_back({global, provider_local, cell.level, cell.cell, cell.rung});
      previous = global;
    }
    batches_.reserve(grouped.size());
    ordered_records_.reserve(records_.size());
    for (auto& [rung, indices] : grouped) {
      const std::size_t offset = ordered_records_.size();
      for (const auto& record : records_)
        if (record.rung == rung)
          ordered_records_.push_back(record);
      batches_.push_back({rung, std::int64_t{1} << rung,
                          partition_.accepted_state().synchronization_tick, offset,
                          ordered_records_.size() - offset, std::move(indices)});
    }
    records_.swap(ordered_records_);
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

  void abort_attempt_() noexcept { rollback(); }

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
                                                     batch.global_indices.size(),
                                                     batch.local_count};
    std::uint64_t aggregate = 0;
    std::exception_ptr local_error;
    try {
      if constexpr (CellTemporalRungBatchLifecycle<Provider>)
        provider_.prepare_rung_batch_local(descriptor);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      abort_attempt_();
      rethrow_collective_failure_(
          local_error, "cell-local temporal rung-batch preparation failed on another rank");
    }
    try {
      if constexpr (CellTemporalRungBatchLifecycle<Provider>)
        provider_.materialize_rung_batch_snapshot(descriptor);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      abort_attempt_();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal rung-batch snapshot failed on another rank");
    }
    try {
      using DeviceView = CellTemporalStageFluxDeviceViewType<Provider>;
      const DeviceView view = provider_.device_view();
      const cell_temporal_detail::EvaluateRungBatch<DeviceView> kernel{
          records_.data(),
          pending_ticks_.data(),
          batch.local_offset,
          begin_tick,
          end_tick,
          partition_.accepted_state().tick_denominator,
          view};
#if defined(POPS_HAS_KOKKOS)
      using Policy =
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>;
      Kokkos::parallel_reduce("pops_cell_temporal_stage_flux_batch",
                              Policy(0, static_cast<std::int64_t>(batch.local_count)), kernel,
                              Kokkos::Max<std::uint64_t>(aggregate));
      device_fence();
#else
      for (std::size_t index = 0; index < batch.local_count; ++index)
        kernel(static_cast<std::int64_t>(index), aggregate);
#endif
      ++stats_.rung_batch_launches;
      stats_.stage_evaluations += static_cast<std::uint64_t>(batch.local_count);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      abort_attempt_();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal rung-batch kernel failed on another rank");
    }
    aggregate = all_reduce_max(aggregate, lane_->communicator());
    if (aggregate != 0) {
      abort_attempt_();
      const auto disposition =
          static_cast<CellTemporalStageDisposition>(static_cast<std::uint32_t>(aggregate >> 32u));
      const std::uint32_t reason = static_cast<std::uint32_t>(aggregate);
      throw CellTemporalStageFailure(disposition, reason);
    }
    try {
      partition_.advance_batch(batch.rung, batch.global_indices, end_tick);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      abort_attempt_();
      rethrow_collective_failure_(local_error,
                                  "cell-local temporal clock advance failed on another rank");
    }
    if constexpr (CellTemporalRungBatchLifecycle<Provider>)
      provider_.complete_rung_batch(descriptor);
    batch.current_tick = end_tick;
  }

  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  Provider provider_;
  BatchedCellTemporalPartition partition_;
  std::string provider_identity_;
  std::string exact_contract_;
  std::string exact_contract_digest_;
  std::vector<cell_temporal_detail::DeviceCellTemporalRecord,
              fab_allocator<cell_temporal_detail::DeviceCellTemporalRecord>>
      records_;
  std::vector<cell_temporal_detail::DeviceCellTemporalRecord,
              fab_allocator<cell_temporal_detail::DeviceCellTemporalRecord>>
      ordered_records_;
  std::vector<std::int64_t, fab_allocator<std::int64_t>> pending_ticks_;
  std::vector<RungBatch> batches_;
  CellTemporalExecutionStats stats_;
  std::int64_t target_tick_ = 0;
  bool attempt_active_ = false;
};

}  // namespace pops::runtime::program

#undef POPS_CELL_TEMPORAL_INLINE_FUNCTION
