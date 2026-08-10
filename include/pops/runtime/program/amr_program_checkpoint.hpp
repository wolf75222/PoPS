/// @file
/// @brief Exact-ranked accepted checkpoint for an AMR Program face-flux ledger.

#pragma once

#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

namespace pops::runtime::program {

namespace amr_reflux = ::pops::amr::reflux;

using AmrProgramFacePayload = std::vector<Real>;

/// Logical history identity retained independently from its per-level native storage keys.  The
/// checkpoint carries the complete authored contract so a fresh exact-ranked hierarchy can
/// materialize rings without executing physics merely to trigger Program prelude allocation.
struct AmrProgramHistoryDescriptor {
  std::string name;
  int program_owner = -1;
  std::string state_identity;
  std::string space_identity;
  std::string clock_identity;
  std::string interpolation_identity;
  int depth = 0;
  int components = 0;
};

/// Rank-independent accepted image of one exact native AMR Program.
///
/// Face entries are the published side of the canonical transactional ledger. Pending fragments,
/// patch-local addresses and MPI ranks are deliberately absent, so an accepted checkpoint can
/// never serialize an incomplete attempt or a stale local buffer.
template <int Dim>
struct AmrProgramAcceptedState {
  static_assert(Dim >= 1 && Dim <= 3, "AMR Program checkpoints support dimensions 1..3");

  std::string spatial_contract;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  std::vector<::pops::amr::ClockStamp> level_clocks;
  std::map<std::string, std::int64_t> logical_clock_ticks;
  std::vector<AmrProgramHistoryDescriptor> histories;
  CellTemporalPartitionAcceptedState temporal_partition;
  std::vector<std::uint8_t> tagging_hysteresis_state;
  std::array<std::vector<amr_reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
      accepted_face_flux;
};

namespace checkpoint_detail {

inline constexpr std::array<std::uint8_t, 8> kMagic{'P', 'O', 'P', 'S', 'A', 'N', 'D', '2'};

class Writer {
 public:
  void raw(std::span<const std::uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }

  void u64(std::uint64_t value) {
    for (int shift = 0; shift != 64; shift += 8)
      bytes_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }

  void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }
  void i32(int value) { i64(static_cast<std::int64_t>(value)); }

  void real(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }

  void size(std::size_t value) { u64(static_cast<std::uint64_t>(value)); }

  void string(std::string_view value) {
    size(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void bytes(std::span<const std::uint8_t> value) {
    size(value.size());
    raw(value);
  }

  std::vector<std::uint8_t> take() && { return std::move(bytes_); }

 private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
 public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  void expect_raw(std::span<const std::uint8_t> expected) {
    require_(expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      if (bytes_[cursor_ + index] != expected[index])
        fail_("unsupported magic/version");
    cursor_ += expected.size();
  }

  std::uint64_t u64() {
    require_(8);
    std::uint64_t value = 0;
    for (int shift = 0; shift != 64; shift += 8)
      value |= static_cast<std::uint64_t>(bytes_[cursor_++]) << shift;
    return value;
  }

  std::int64_t i64() { return static_cast<std::int64_t>(u64()); }

  int i32() {
    const std::int64_t value = i64();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
      fail_("integer is outside the native int range");
    return static_cast<int>(value);
  }

  double real() {
    const std::uint64_t bits = u64();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  std::size_t size() {
    const std::uint64_t value = u64();
    constexpr std::uint64_t kMaxElements = std::uint64_t{1} << 30;
    if (value > kMaxElements || value > bytes_.size())
      fail_("container length is not credible for this payload");
    return static_cast<std::size_t>(value);
  }

  std::string string() {
    const std::size_t count = size();
    require_(count);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + cursor_), count);
    cursor_ += count;
    return value;
  }

  std::vector<std::uint8_t> bytes() {
    const std::size_t count = size();
    require_(count);
    std::vector<std::uint8_t> value(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                                    bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_ + count));
    cursor_ += count;
    return value;
  }

  void finish() const {
    if (cursor_ != bytes_.size())
      fail_("trailing bytes after the accepted-state image");
  }

 private:
  [[noreturn]] static void fail_(std::string_view reason) {
    throw std::runtime_error("invalid exact AMR Program checkpoint: " + std::string(reason));
  }

  void require_(std::size_t count) const {
    if (count > bytes_.size() - cursor_)
      fail_("truncated payload");
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t cursor_ = 0;
};

inline void write_rational(Writer& out, const ::pops::amr::Rational& value) {
  out.i64(value.numerator);
  out.i64(value.denominator);
}

inline ::pops::amr::Rational read_rational(Reader& in) {
  return {in.i64(), in.i64()};
}

inline void write_clock(Writer& out, const ::pops::amr::ClockStamp& value) {
  out.i32(value.level);
  out.i64(value.macro_step);
  write_rational(out, value.phase);
  out.real(value.physical_time);
}

inline ::pops::amr::ClockStamp read_clock(Reader& in) {
  return {in.i32(), in.i64(), read_rational(in), in.real()};
}

template <int Dim>
void write_index(Writer& out, const Index<Dim>& value) {
  for (int axis = 0; axis < Dim; ++axis)
    out.i32(value[axis]);
}

template <int Dim>
Index<Dim> read_index(Reader& in) {
  Index<Dim> value{};
  for (int axis = 0; axis < Dim; ++axis)
    value[axis] = in.i32();
  return value;
}

inline void write_temporal_partition(Writer& out, const CellTemporalPartitionAcceptedState& value) {
  validate_cell_temporal_partition_state(value);
  out.u64(static_cast<std::uint64_t>(value.kind));
  out.string(value.provider_identity);
  out.u64(value.topology_epoch);
  out.i64(value.synchronization_tick);
  out.i64(value.tick_denominator);
  out.size(value.cells.size());
  for (const CellTemporalPartitionRecord& cell : value.cells) {
    out.i32(cell.level);
    out.u64(cell.cell);
    out.i32(cell.rung);
    out.i64(cell.accepted_tick);
  }
}

inline CellTemporalPartitionAcceptedState read_temporal_partition(Reader& in) {
  CellTemporalPartitionAcceptedState value;
  const std::uint64_t kind = in.u64();
  if (kind > static_cast<std::uint64_t>(TemporalPartitionKind::CellLocal))
    throw std::runtime_error("invalid exact AMR Program checkpoint: temporal partition kind");
  value.kind = static_cast<TemporalPartitionKind>(kind);
  value.provider_identity = in.string();
  value.topology_epoch = in.u64();
  value.synchronization_tick = in.i64();
  value.tick_denominator = in.i64();
  value.cells.resize(in.size());
  for (CellTemporalPartitionRecord& cell : value.cells) {
    cell.level = in.i32();
    cell.cell = in.u64();
    cell.rung = in.i32();
    cell.accepted_tick = in.i64();
  }
  validate_cell_temporal_partition_state(value);
  return value;
}

template <int Dim>
void write_face_fragment(Writer& out,
                         const amr_reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>& fragment) {
  amr_reflux::validate_face_flux_fragment(fragment.key, fragment.measure);
  out.string(fragment.key.owner);
  out.string(fragment.key.state);
  out.i32(fragment.key.levels.coarse);
  out.i32(fragment.key.levels.fine);
  out.u64(static_cast<std::uint64_t>(fragment.key.centering));
  out.i32(fragment.key.axis);
  write_index(out, fragment.key.face);
  write_index(out, fragment.key.coarse_face);
  write_clock(out, fragment.key.clock);
  out.string(fragment.key.stage);
  out.u64(fragment.key.attempt);
  out.u64(static_cast<std::uint64_t>(fragment.key.role));
  out.u64(static_cast<std::uint64_t>(fragment.key.contribution));
  write_rational(out, fragment.measure.stage_weight);
  write_rational(out, fragment.measure.substep_begin);
  write_rational(out, fragment.measure.substep_end);
  out.real(fragment.measure.substep_duration);
  out.real(fragment.measure.face_measure);
  out.size(fragment.payload.size());
  for (Real component : fragment.payload)
    out.real(static_cast<double>(component));
}

template <int Dim>
amr_reflux::FaceFluxFragment<Dim, AmrProgramFacePayload> read_face_fragment(Reader& in) {
  amr_reflux::FaceFluxFragment<Dim, AmrProgramFacePayload> fragment;
  fragment.key.owner = in.string();
  fragment.key.state = in.string();
  fragment.key.levels.coarse = in.i32();
  fragment.key.levels.fine = in.i32();
  const std::uint64_t centering = in.u64();
  if (centering > static_cast<std::uint64_t>(amr_reflux::FaceLedgerCentering::Cell))
    throw std::runtime_error("invalid exact AMR Program checkpoint: face centering");
  fragment.key.centering = static_cast<amr_reflux::FaceLedgerCentering>(centering);
  fragment.key.axis = in.i32();
  fragment.key.face = read_index<Dim>(in);
  fragment.key.coarse_face = read_index<Dim>(in);
  fragment.key.clock = read_clock(in);
  fragment.key.stage = in.string();
  fragment.key.attempt = in.u64();
  const std::uint64_t role = in.u64();
  const std::uint64_t contribution = in.u64();
  if (role > static_cast<std::uint64_t>(amr_reflux::FaceLedgerRole::Fine) ||
      contribution > static_cast<std::uint64_t>(amr_reflux::FaceLedgerContribution::Source))
    throw std::runtime_error("invalid exact AMR Program checkpoint: face role/contribution");
  fragment.key.role = static_cast<amr_reflux::FaceLedgerRole>(role);
  fragment.key.contribution = static_cast<amr_reflux::FaceLedgerContribution>(contribution);
  fragment.measure.stage_weight = read_rational(in);
  fragment.measure.substep_begin = read_rational(in);
  fragment.measure.substep_end = read_rational(in);
  fragment.measure.substep_duration = in.real();
  fragment.measure.face_measure = in.real();
  fragment.payload.resize(in.size());
  for (Real& component : fragment.payload)
    component = static_cast<Real>(in.real());
  amr_reflux::validate_face_flux_fragment(fragment.key, fragment.measure);
  return fragment;
}

template <int Dim>
void validate_state(const AmrProgramAcceptedState<Dim>& state) {
  if (state.spatial_contract.empty())
    throw std::invalid_argument("exact AMR Program checkpoint requires its spatial contract");
  if (state.level_clocks.empty())
    throw std::invalid_argument("exact AMR Program checkpoint requires at least one level clock");
  for (std::size_t level = 0; level < state.level_clocks.size(); ++level) {
    const auto& clock = state.level_clocks[level];
    if (clock.level != static_cast<int>(level) || clock.macro_step < 0 ||
        clock.phase.denominator <= 0 ||
        ::pops::amr::Rational(clock.phase.numerator, clock.phase.denominator) != clock.phase ||
        !std::isfinite(clock.physical_time))
      throw std::invalid_argument("exact AMR Program checkpoint has an invalid level clock");
  }
  validate_cell_temporal_partition_state(state.temporal_partition);
  if (state.temporal_partition.kind == TemporalPartitionKind::CellLocal &&
      state.temporal_partition.topology_epoch != state.topology_epoch)
    throw std::invalid_argument(
        "exact AMR Program checkpoint temporal partition names another topology");

  std::string previous_history;
  for (const AmrProgramHistoryDescriptor& history : state.histories) {
    if (history.name.empty() || history.program_owner < 0 || history.state_identity.empty() ||
        history.space_identity.empty() || history.clock_identity.empty() ||
        history.interpolation_identity.empty() || history.depth < 2 || history.components < 1)
      throw std::invalid_argument(
          "exact AMR Program checkpoint has an incomplete history descriptor");
    if (!previous_history.empty() && previous_history >= history.name)
      throw std::invalid_argument(
          "exact AMR Program checkpoint history descriptors must be uniquely ordered");
    previous_history = history.name;
  }

  for (int axis = 0; axis < Dim; ++axis) {
    const auto& fragments = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    std::optional<amr_reflux::FaceFluxFragmentKey<Dim>> previous;
    for (const auto& fragment : fragments) {
      if (fragment.key.axis != axis)
        throw std::invalid_argument(
            "exact AMR Program checkpoint stores a face under another axis");
      amr_reflux::validate_face_flux_fragment(fragment.key, fragment.measure);
      if (fragment.payload.empty())
        throw std::invalid_argument("exact AMR Program checkpoint face payload cannot be empty");
      for (Real component : fragment.payload)
        if (!std::isfinite(static_cast<double>(component)))
          throw std::invalid_argument("exact AMR Program checkpoint face payload must be finite");
      if (previous && !(previous.value() < fragment.key))
        throw std::invalid_argument(
            "exact AMR Program checkpoint face fragments must be uniquely ordered");
      previous = fragment.key;
    }
  }
}

}  // namespace checkpoint_detail

template <int Dim>
AmrProgramAcceptedState<Dim> accepted_amr_program_state(
    std::string spatial_contract, std::uint64_t topology_epoch,
    std::uint64_t materialization_generation, std::vector<::pops::amr::ClockStamp> level_clocks,
    CellTemporalPartitionAcceptedState temporal_partition,
    const amr_reflux::TransactionalFaceFluxLedger<Dim, AmrProgramFacePayload>& ledger) {
  if (ledger.in_transaction())
    throw std::logic_error(
        "exact AMR Program checkpoint cannot observe an active face-flux transaction");
  AmrProgramAcceptedState<Dim> state;
  state.spatial_contract = std::move(spatial_contract);
  state.topology_epoch = topology_epoch;
  state.materialization_generation = materialization_generation;
  state.level_clocks = std::move(level_clocks);
  state.temporal_partition = std::move(temporal_partition);
  for (int axis = 0; axis < Dim; ++axis) {
    auto& destination = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    destination = ledger.published_entries(axis);
    std::sort(destination.begin(), destination.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
  }
  checkpoint_detail::validate_state(state);
  return state;
}

template <int Dim>
std::vector<std::uint8_t> serialize_amr_program_accepted_state(
    const AmrProgramAcceptedState<Dim>& state) {
  checkpoint_detail::validate_state(state);
  checkpoint_detail::Writer out;
  out.raw(checkpoint_detail::kMagic);
  out.i32(Dim);
  out.string(state.spatial_contract);
  out.u64(state.topology_epoch);
  out.u64(state.materialization_generation);
  out.size(state.level_clocks.size());
  for (const auto& clock : state.level_clocks)
    checkpoint_detail::write_clock(out, clock);
  out.size(state.logical_clock_ticks.size());
  for (const auto& [identity, tick] : state.logical_clock_ticks) {
    out.string(identity);
    out.i64(tick);
  }
  out.size(state.histories.size());
  for (const AmrProgramHistoryDescriptor& history : state.histories) {
    out.string(history.name);
    out.i32(history.program_owner);
    out.string(history.state_identity);
    out.string(history.space_identity);
    out.string(history.clock_identity);
    out.string(history.interpolation_identity);
    out.i32(history.depth);
    out.i32(history.components);
  }
  checkpoint_detail::write_temporal_partition(out, state.temporal_partition);
  out.bytes(state.tagging_hysteresis_state);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto& fragments = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    out.size(fragments.size());
    for (const auto& fragment : fragments)
      checkpoint_detail::write_face_fragment(out, fragment);
  }
  return std::move(out).take();
}

template <int Dim>
AmrProgramAcceptedState<Dim> deserialize_amr_program_accepted_state(
    std::span<const std::uint8_t> bytes) {
  checkpoint_detail::Reader in(bytes);
  in.expect_raw(checkpoint_detail::kMagic);
  if (in.i32() != Dim)
    throw std::runtime_error(
        "invalid exact AMR Program checkpoint: native dimension does not match the artifact");
  AmrProgramAcceptedState<Dim> state;
  state.spatial_contract = in.string();
  state.topology_epoch = in.u64();
  state.materialization_generation = in.u64();
  state.level_clocks.resize(in.size());
  for (auto& clock : state.level_clocks)
    clock = checkpoint_detail::read_clock(in);
  const std::size_t logical_clock_count = in.size();
  for (std::size_t index = 0; index < logical_clock_count; ++index) {
    std::string identity = in.string();
    const std::int64_t tick = in.i64();
    if (!state.logical_clock_ticks.emplace(std::move(identity), tick).second)
      throw std::runtime_error(
          "invalid exact AMR Program checkpoint: duplicate logical clock identity");
  }
  state.histories.resize(in.size());
  for (AmrProgramHistoryDescriptor& history : state.histories) {
    history.name = in.string();
    history.program_owner = in.i32();
    history.state_identity = in.string();
    history.space_identity = in.string();
    history.clock_identity = in.string();
    history.interpolation_identity = in.string();
    history.depth = in.i32();
    history.components = in.i32();
  }
  state.temporal_partition = checkpoint_detail::read_temporal_partition(in);
  state.tagging_hysteresis_state = in.bytes();
  for (int axis = 0; axis < Dim; ++axis) {
    auto& fragments = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    fragments.resize(in.size());
    for (auto& fragment : fragments)
      fragment = checkpoint_detail::read_face_fragment<Dim>(in);
  }
  in.finish();
  checkpoint_detail::validate_state(state);
  return state;
}

template <int Dim>
amr_reflux::TransactionalFaceFluxLedger<Dim, AmrProgramFacePayload>
restore_amr_program_face_flux_ledger(const AmrProgramAcceptedState<Dim>& state,
                                     amr_reflux::FaceFluxLedgerBudget budget) {
  checkpoint_detail::validate_state(state);
  using Fragment = amr_reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>;
  std::map<std::uint64_t, std::vector<Fragment>> attempts;
  for (const auto& axis : state.accepted_face_flux)
    for (const Fragment& fragment : axis)
      attempts[fragment.key.attempt].push_back(fragment);

  amr_reflux::TransactionalFaceFluxLedger<Dim, AmrProgramFacePayload> ledger(budget);
  for (auto& [attempt, fragments] : attempts) {
    ledger.begin(attempt);
    try {
      for (Fragment& fragment : fragments)
        ledger.accumulate(std::move(fragment.key), fragment.measure, std::move(fragment.payload));
      ledger.commit();
    } catch (...) {
      if (ledger.in_transaction())
        ledger.rollback();
      throw;
    }
  }
  return ledger;
}

template <int Dim, class MemorySpace>
void require_live_amr_program_checkpoint(
    const AmrProgramAcceptedState<Dim>& state,
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime) {
  checkpoint_detail::validate_state(state);
  if (state.spatial_contract != runtime.spatial_contract() ||
      state.topology_epoch != runtime.topology_epoch() ||
      state.materialization_generation != runtime.materialization_generation() ||
      state.level_clocks.size() != runtime.hierarchy().num_levels())
    throw std::invalid_argument(
        "exact AMR Program checkpoint does not authenticate the live hierarchy");
}

/// Collective fail-closed preflight used before any rank publishes a restored accepted state.
template <int Dim>
void require_collective_amr_program_checkpoint_consensus(
    const AmrProgramAcceptedState<Dim>& state, const ExecutionLane& lane = ExecutionLane::world()) {
  const std::vector<std::uint8_t> bytes = serialize_amr_program_accepted_state(state);
  const std::string_view payload(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.amr-program-checkpoint"), payload}}, lane))
    throw std::runtime_error("exact AMR Program checkpoint differs between communicator ranks");
}

}  // namespace pops::runtime::program
