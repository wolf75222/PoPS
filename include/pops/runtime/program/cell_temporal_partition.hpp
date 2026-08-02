#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {

inline constexpr const char* kGlobalTemporalPartitionProvider = "pops.temporal-partition.global@1";

enum class TemporalPartitionKind : std::uint8_t { Global = 0, CellLocal = 1 };

/// Rank-independent identity and accepted logical clock of one cell.
///
/// ``cell`` is a provider-owned canonical cell id within ``level``. It deliberately does not carry
/// an MPI rank or patch-local address, so ownership migration can rematerialize device storage from
/// the same accepted scientific image.
struct CellTemporalPartitionRecord {
  int level = 0;
  std::uint64_t cell = 0;
  int rung = 0;
  std::int64_t accepted_tick = 0;

  friend bool operator==(const CellTemporalPartitionRecord&,
                         const CellTemporalPartitionRecord&) = default;
};

/// Compact accepted-boundary image for a prepared temporal-partition provider.
///
/// Physical time is represented by integer ticks over one provider-owned denominator. Accepted
/// checkpoints are synchronization barriers: every record must be at ``synchronization_tick``.
/// Attempt-local clocks live in ``BatchedCellTemporalPartition`` and can never leak into this image.
struct CellTemporalPartitionAcceptedState {
  TemporalPartitionKind kind = TemporalPartitionKind::Global;
  std::string provider_identity = kGlobalTemporalPartitionProvider;
  std::uint64_t topology_epoch = 0;
  std::int64_t synchronization_tick = 0;
  std::int64_t tick_denominator = 1;
  std::vector<CellTemporalPartitionRecord> cells;

  friend bool operator==(const CellTemporalPartitionAcceptedState&,
                         const CellTemporalPartitionAcceptedState&) = default;
};

inline void validate_cell_temporal_partition_state(
    const CellTemporalPartitionAcceptedState& state) {
  if (state.provider_identity.empty())
    throw std::invalid_argument("temporal partition provider identity cannot be empty");
  if (state.synchronization_tick < 0 || state.tick_denominator <= 0)
    throw std::invalid_argument(
        "temporal partition accepted clock requires non-negative ticks and a positive denominator");
  if (state.kind == TemporalPartitionKind::Global) {
    if (state.provider_identity != kGlobalTemporalPartitionProvider || state.topology_epoch != 0 ||
        !state.cells.empty())
      throw std::invalid_argument(
          "global temporal partition state cannot carry topology or cell-local clocks");
    return;
  }
  if (state.kind != TemporalPartitionKind::CellLocal)
    throw std::invalid_argument("temporal partition state has an unsupported kind");
  if (state.provider_identity == kGlobalTemporalPartitionProvider || state.cells.empty())
    throw std::invalid_argument(
        "cell-local temporal partition state requires its prepared provider and cell clocks");

  std::tuple<int, std::uint64_t> previous{-1, 0};
  bool first = true;
  for (const CellTemporalPartitionRecord& cell : state.cells) {
    if (cell.level < 0 || cell.rung < 0 || cell.rung > 30 || cell.accepted_tick < 0)
      throw std::invalid_argument("temporal partition cell record is outside its bounded domain");
    const std::tuple<int, std::uint64_t> identity{cell.level, cell.cell};
    if (!first && !(previous < identity))
      throw std::invalid_argument(
          "temporal partition cell records must be unique and canonically ordered");
    if (cell.accepted_tick != state.synchronization_tick)
      throw std::invalid_argument(
          "temporal partition accepted checkpoint is not at a synchronization barrier");
    const std::int64_t stride = std::int64_t{1} << cell.rung;
    if (cell.accepted_tick % stride != 0)
      throw std::invalid_argument(
          "temporal partition accepted tick is not aligned to its cell rung");
    previous = identity;
    first = false;
  }
}

/// Host authority for one bounded batch schedule and its transactional clocks.
///
/// This class owns no numerical field and launches no per-cell task. A future Kokkos execution
/// provider consumes the canonical records in rung batches; this authority supplies exact attempt,
/// rollback, barrier, checkpoint and diagnostic semantics independently of the execution space.
class BatchedCellTemporalPartition {
 public:
  explicit BatchedCellTemporalPartition(
      CellTemporalPartitionAcceptedState accepted = CellTemporalPartitionAcceptedState{})
      : accepted_(std::move(accepted)) {
    validate_cell_temporal_partition_state(accepted_);
  }

  const CellTemporalPartitionAcceptedState& accepted_state() const noexcept { return accepted_; }
  bool attempt_active() const noexcept { return attempt_active_; }

  void begin_attempt(std::int64_t target_tick) {
    if (attempt_active_)
      throw std::logic_error("temporal partition attempt is already active");
    if (target_tick <= accepted_.synchronization_tick)
      throw std::invalid_argument("temporal partition attempt target must advance accepted time");
    pending_ticks_.clear();
    pending_ticks_.reserve(accepted_.cells.size());
    for (const CellTemporalPartitionRecord& cell : accepted_.cells) {
      const std::int64_t stride = std::int64_t{1} << cell.rung;
      if ((target_tick - cell.accepted_tick) % stride != 0)
        throw std::invalid_argument(
            "temporal partition attempt target is unreachable for one prepared rung");
      pending_ticks_.push_back(cell.accepted_tick);
    }
    target_tick_ = target_tick;
    attempt_active_ = true;
  }

  /// Advance a canonically ordered batch of record indices belonging to exactly one rung.
  void advance_batch(int rung, const std::vector<std::size_t>& indices, std::int64_t target_tick) {
    if (!attempt_active_)
      throw std::logic_error("temporal partition batch requires an active attempt");
    if (indices.empty())
      throw std::invalid_argument("temporal partition batch cannot be empty");
    if (target_tick > target_tick_)
      throw std::invalid_argument("temporal partition batch crosses its synchronization target");
    std::size_t previous = std::numeric_limits<std::size_t>::max();
    for (std::size_t index : indices) {
      if (index >= accepted_.cells.size() ||
          (previous != std::numeric_limits<std::size_t>::max() && index <= previous))
        throw std::invalid_argument(
            "temporal partition batch indices must be unique and canonically ordered");
      const CellTemporalPartitionRecord& cell = accepted_.cells[index];
      if (cell.rung != rung)
        throw std::invalid_argument("temporal partition batch mixes prepared rungs");
      const std::int64_t stride = std::int64_t{1} << cell.rung;
      if (target_tick <= pending_ticks_[index] ||
          (target_tick - pending_ticks_[index]) % stride != 0)
        throw std::invalid_argument(
            "temporal partition batch target is not a forward rung-aligned tick");
      previous = index;
    }
    for (std::size_t index : indices)
      pending_ticks_[index] = target_tick;
  }

  void require_barrier(const std::string& operation) const {
    if (!attempt_active_)
      return;
    if (std::any_of(pending_ticks_.begin(), pending_ticks_.end(),
                    [this](std::int64_t tick) { return tick != target_tick_; }))
      throw std::logic_error(operation +
                             " requires every cell-local clock at the synchronization barrier");
  }

  void commit() {
    if (!attempt_active_)
      throw std::logic_error("temporal partition commit requires an active attempt");
    require_barrier("temporal partition commit");
    for (std::size_t index = 0; index < accepted_.cells.size(); ++index)
      accepted_.cells[index].accepted_tick = pending_ticks_[index];
    accepted_.synchronization_tick = target_tick_;
    clear_attempt_();
  }

  void rollback() noexcept { clear_attempt_(); }

  CellTemporalPartitionAcceptedState checkpoint() const {
    if (attempt_active_)
      throw std::logic_error(
          "temporal partition checkpoint requires an accepted synchronization barrier");
    return accepted_;
  }

  void restore(CellTemporalPartitionAcceptedState accepted) {
    if (attempt_active_)
      throw std::logic_error("temporal partition restore cannot replace an active attempt");
    validate_cell_temporal_partition_state(accepted);
    accepted_ = std::move(accepted);
  }

  void require_global_execution_route() const {
    if (accepted_.kind != TemporalPartitionKind::Global)
      throw std::logic_error(
          "cell-local temporal partition requires its prepared batched executor; the global AMR "
          "step cannot silently replace it");
  }

  std::vector<std::vector<std::string>> manifest() const {
    std::map<int, std::size_t> rung_counts;
    for (const CellTemporalPartitionRecord& cell : accepted_.cells)
      ++rung_counts[cell.rung];
    std::vector<std::vector<std::string>> rows;
    rows.push_back(
        {"summary", accepted_.kind == TemporalPartitionKind::Global ? "global" : "cell_local",
         accepted_.provider_identity, std::to_string(accepted_.topology_epoch),
         std::to_string(accepted_.synchronization_tick), std::to_string(accepted_.tick_denominator),
         std::to_string(accepted_.cells.size())});
    for (const auto& [rung, count] : rung_counts)
      rows.push_back({"rung", std::to_string(rung), std::to_string(count)});
    return rows;
  }

 private:
  void clear_attempt_() noexcept {
    pending_ticks_.clear();
    target_tick_ = 0;
    attempt_active_ = false;
  }

  CellTemporalPartitionAcceptedState accepted_;
  std::vector<std::int64_t> pending_ticks_;
  std::int64_t target_tick_ = 0;
  bool attempt_active_ = false;
};

}  // namespace pops::runtime::program
