/// @file
/// @brief Exact-ranked accepted checkpoint for an AMR Program face-flux ledger.

#pragma once

#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>
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
#include <tuple>
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

  friend bool operator==(const AmrProgramHistoryDescriptor&,
                         const AmrProgramHistoryDescriptor&) = default;
};

/// Accepted per-level/per-slot provenance retained with a dense history value.  In particular the
/// outgoing interval is the authority needed to replay an AB2/BDF2 ring under variable dt; it must
/// never be inferred from the facade's last step.
struct AmrProgramHistorySlotProvenance {
  std::string name;
  int level = -1;
  int slot = -1;
  double outgoing_dt = 0.0;
  bool initialized = false;
  int fill_count = 0;

  friend bool operator==(const AmrProgramHistorySlotProvenance&,
                         const AmrProgramHistorySlotProvenance&) = default;
};

/// One deferred, direct-child variable-step AB2 read.  It is accepted state, not an attempt-local
/// scratch: a checkpoint immediately after regrid must reproduce the same first child lag.
struct AmrProgramPendingHistoryRemap {
  std::string key;
  int parent_level = -1;
  int child_level = -1;
  std::uint64_t prior_topology_epoch = 0;
  std::uint64_t prior_materialization_generation = 0;
  std::uint64_t published_topology_epoch = 0;
  std::uint64_t published_materialization_generation = 0;
  std::int64_t accepted_macro_step = -1;
  /// Exact IntegralOnly child steps per parent source interval.  This is wire authority, never
  /// inferred from binary dt values.
  std::int64_t temporal_numerator = 0;
  std::int64_t temporal_denominator = 0;
  double source_dt = 0.0;
  double target_dt = 0.0;
  bool consumed = false;

  friend bool operator==(const AmrProgramPendingHistoryRemap&,
                         const AmrProgramPendingHistoryRemap&) = default;
};

struct AmrProgramSynchronizationEvent {
  int parent_level = -1;
  int child_level = -1;
  int runtime_block = -1;
  std::string phase;
  ::pops::amr::ClockStamp clock;
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
  std::vector<AmrProgramHistorySlotProvenance> history_slots;
  std::vector<AmrProgramPendingHistoryRemap> pending_history_remaps;
  /// Opaque at the facade boundary but structured by AmrProgramContext: every level-qualified
  /// history slot's exact FluxBasis samples and rational coefficients.  This deliberately lives
  /// beside (not inside) the numerical history image, whose MPI rematerialization is independent.
  std::vector<std::uint8_t> history_flux_payload;
  CellTemporalPartitionAcceptedState temporal_partition;
  std::vector<std::uint8_t> tagging_hysteresis_state;
  /// Exact prepared authorities which bounded and coupled the accepted face ledgers.
  std::string flux_budget_contract;
  std::string coupling_contract;
  std::array<std::vector<amr_reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
      accepted_face_flux;
  std::vector<::pops::amr::InterfaceFluxFragment<AmrProgramFacePayload>> accepted_interface_flux;
  std::vector<AmrProgramSynchronizationEvent> synchronization_events;
};

namespace checkpoint_detail {

inline constexpr std::array<std::uint8_t, 8> kMagic{'P', 'O', 'P', 'S', 'A', 'N', 'D', '4'};

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

/// Allocation-free twin of Writer used by the artifact checkpoint-capacity preflight.  Keeping the
/// primitive surface identical lets the binary encoder itself remain the only wire-schema
/// authority: a field added to POPSAND4 changes both serialization and capacity accounting in the
/// same function.
class CountingWriter {
 public:
  void raw(std::span<const std::uint8_t> bytes) { add_(bytes.size()); }

  void u64(std::uint64_t) { add_(sizeof(std::uint64_t)); }
  void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }
  void i32(int value) { i64(static_cast<std::int64_t>(value)); }
  void real(double) { add_(sizeof(double)); }
  void size(std::size_t value) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
      if (value > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
        throw std::length_error("AMR Program checkpoint length exceeds uint64_t");
    u64(static_cast<std::uint64_t>(value));
  }
  void string(std::string_view value) {
    size(value.size());
    add_(value.size());
  }
  void bytes(std::span<const std::uint8_t> value) {
    size(value.size());
    add_(value.size());
  }
  void string_size(std::size_t characters) {
    size(characters);
    add_(characters);
  }
  void bytes_size(std::size_t count) {
    size(count);
    add_(count);
  }
  void repeated_bytes(std::size_t count, std::size_t bytes_per_value) {
    if (bytes_per_value != 0 && count > std::numeric_limits<std::size_t>::max() / bytes_per_value)
      throw std::length_error("AMR Program checkpoint repeated capacity exceeds size_t");
    add_(count * bytes_per_value);
  }

  [[nodiscard]] std::size_t count() const noexcept { return count_; }

 private:
  void add_(std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - count_)
      throw std::length_error("AMR Program checkpoint capacity exceeds size_t");
    count_ += value;
  }

  std::size_t count_ = 0;
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

  std::size_t size(std::size_t element_bytes = 1) {
    const std::uint64_t value = u64();
    constexpr std::uint64_t kMaxElements = std::uint64_t{1} << 30;
    if (element_bytes == 0 || value > kMaxElements ||
        value > static_cast<std::uint64_t>((bytes_.size() - cursor_) / element_bytes))
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

inline constexpr std::size_t kEncodedScalarBytes = 8;
inline constexpr std::size_t kEncodedClockBytes = 5 * kEncodedScalarBytes;
// String minima below include their eight-byte length prefixes; zero-length strings remain a
// credible binary shape even when semantic validation later rejects them.
inline constexpr std::size_t kMinLogicalClockBytes = 2 * kEncodedScalarBytes;
inline constexpr std::size_t kMinHistoryDescriptorBytes = 8 * kEncodedScalarBytes;
inline constexpr std::size_t kMinHistorySlotBytes = 6 * kEncodedScalarBytes;
inline constexpr std::size_t kMinTemporalPartitionRecordBytes = 4 * kEncodedScalarBytes;
inline constexpr std::size_t kMinInterfaceFragmentBytes = 32 * kEncodedScalarBytes;
inline constexpr std::size_t kMinSynchronizationEventBytes = 9 * kEncodedScalarBytes;
// Pending remap: key length plus two i32 words (encoded as i64), four u64 words, three i64
// words, two reals, and its consumed tag.  Keep this in wire units, not sizeof(int).
inline constexpr std::size_t kMinPendingHistoryRemapBytes = 13 * kEncodedScalarBytes;

template <int Dim>
inline constexpr std::size_t kMinFaceFragmentBytes =
    (24 + 2 * static_cast<std::size_t>(Dim)) * kEncodedScalarBytes;

template <class Output>
inline void write_rational(Output& out, const ::pops::amr::Rational& value) {
  out.i64(value.numerator);
  out.i64(value.denominator);
}

inline ::pops::amr::Rational read_rational(Reader& in) {
  return {in.i64(), in.i64()};
}

template <class Output>
inline void write_clock(Output& out, const ::pops::amr::ClockStamp& value) {
  out.i32(value.level);
  out.i64(value.macro_step);
  write_rational(out, value.phase);
  out.real(value.physical_time);
}

inline ::pops::amr::ClockStamp read_clock(Reader& in) {
  return {in.i32(), in.i64(), read_rational(in), in.real()};
}

template <class Output>
inline void write_clock_window(Output& out, const ::pops::amr::ClockWindow& value) {
  write_clock(out, value.begin);
  write_clock(out, value.end);
}

inline ::pops::amr::ClockWindow read_clock_window(Reader& in) {
  return {read_clock(in), read_clock(in)};
}

template <int Dim, class Output>
void write_index(Output& out, const Index<Dim>& value) {
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

template <class Output>
inline void write_temporal_partition(Output& out, const CellTemporalPartitionAcceptedState& value) {
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
  value.cells.resize(in.size(kMinTemporalPartitionRecordBytes));
  for (CellTemporalPartitionRecord& cell : value.cells) {
    cell.level = in.i32();
    cell.cell = in.u64();
    cell.rung = in.i32();
    cell.accepted_tick = in.i64();
  }
  validate_cell_temporal_partition_state(value);
  return value;
}

template <int Dim, class Output>
void write_face_fragment(Output& out,
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
  fragment.payload.resize(in.size(sizeof(double)));
  for (Real& component : fragment.payload)
    component = static_cast<Real>(in.real());
  amr_reflux::validate_face_flux_fragment(fragment.key, fragment.measure);
  return fragment;
}

template <class Output>
inline void write_interface_fragment(
    Output& out, const ::pops::amr::InterfaceFluxFragment<AmrProgramFacePayload>& fragment) {
  ::pops::amr::validate_interface_flux_fragment(fragment.key, fragment.measure,
                                                fragment.key.topology_epoch);
  out.string(fragment.key.interface_identity);
  out.u64(fragment.key.topology_epoch);
  out.i32(fragment.key.coarse_level);
  out.i32(fragment.key.fine_level);
  write_clock(out, fragment.key.clock);
  out.string(fragment.key.stage_identity);
  out.string(fragment.key.graph_identity);
  out.string(fragment.key.rate_identity);
  out.string(fragment.key.application_identity);
  write_clock_window(out, fragment.key.interval);
  out.u64(static_cast<std::uint64_t>(fragment.key.orientation));
  out.u64(static_cast<std::uint64_t>(fragment.key.left_block));
  out.u64(static_cast<std::uint64_t>(fragment.key.right_block));
  write_rational(out, fragment.measure.stage_weight);
  out.real(fragment.measure.face_measure);
  out.real(fragment.measure.substep_duration);
  out.u64(fragment.measure.stage_weight_resolved ? 1U : 0U);
  out.size(fragment.payload.size());
  for (Real component : fragment.payload)
    out.real(static_cast<double>(component));
}

inline ::pops::amr::InterfaceFluxFragment<AmrProgramFacePayload> read_interface_fragment(
    Reader& in, std::size_t* remaining_payload_terms = nullptr) {
  ::pops::amr::InterfaceFluxFragment<AmrProgramFacePayload> fragment;
  fragment.key.interface_identity = in.string();
  fragment.key.topology_epoch = in.u64();
  fragment.key.coarse_level = in.i32();
  fragment.key.fine_level = in.i32();
  fragment.key.clock = read_clock(in);
  fragment.key.stage_identity = in.string();
  fragment.key.graph_identity = in.string();
  fragment.key.rate_identity = in.string();
  fragment.key.application_identity = in.string();
  fragment.key.interval = read_clock_window(in);
  const std::uint64_t orientation = in.u64();
  if (orientation > static_cast<std::uint64_t>(::pops::amr::InterfaceFluxOrientation::FineOutward))
    throw std::runtime_error("invalid exact AMR Program checkpoint: interface orientation");
  fragment.key.orientation = static_cast<::pops::amr::InterfaceFluxOrientation>(orientation);
  const std::uint64_t left_block = in.u64();
  const std::uint64_t right_block = in.u64();
  if (left_block > std::numeric_limits<std::size_t>::max() ||
      right_block > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("invalid exact AMR Program checkpoint: interface block identity");
  fragment.key.left_block = static_cast<std::size_t>(left_block);
  fragment.key.right_block = static_cast<std::size_t>(right_block);
  fragment.measure.stage_weight = read_rational(in);
  fragment.measure.face_measure = in.real();
  fragment.measure.substep_duration = in.real();
  const std::uint64_t resolved = in.u64();
  if (resolved > 1U)
    throw std::runtime_error("invalid exact AMR Program checkpoint: interface stage-weight tag");
  fragment.measure.stage_weight_resolved = resolved != 0;
  const std::size_t payload_terms = in.size(sizeof(double));
  if (remaining_payload_terms != nullptr) {
    if (payload_terms > *remaining_payload_terms)
      throw std::length_error(
          "invalid exact AMR Program checkpoint: interface payload exceeds artifact budget");
    *remaining_payload_terms -= payload_terms;
  }
  fragment.payload.resize(payload_terms);
  for (Real& component : fragment.payload)
    component = static_cast<Real>(in.real());
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

  std::tuple<std::string, int, int> previous_slot{"", -1, -1};
  bool first_slot = true;
  for (const AmrProgramHistorySlotProvenance& slot : state.history_slots) {
    const auto descriptor = std::find_if(
        state.histories.begin(), state.histories.end(),
        [&](const AmrProgramHistoryDescriptor& history) { return history.name == slot.name; });
    if (descriptor == state.histories.end() || slot.level < 0 ||
        static_cast<std::size_t>(slot.level) >= state.level_clocks.size() || slot.slot < 0 ||
        slot.slot >= descriptor->depth || !std::isfinite(slot.outgoing_dt) ||
        slot.outgoing_dt < 0.0 || slot.fill_count < 0 || slot.fill_count > descriptor->depth)
      throw std::invalid_argument(
          "exact AMR Program checkpoint has invalid history-slot provenance");
    if (slot.initialized != (slot.fill_count > 0) ||
        (slot.fill_count == 0 ? slot.outgoing_dt != 0.0 : !(slot.outgoing_dt > 0.0)))
      throw std::invalid_argument(
          "exact AMR Program checkpoint history dt differs from its publication metadata");
    const std::tuple<std::string, int, int> identity{slot.name, slot.level, slot.slot};
    if (!first_slot && !(previous_slot < identity))
      throw std::invalid_argument(
          "exact AMR Program checkpoint history-slot provenance is not uniquely ordered");
    previous_slot = identity;
    first_slot = false;
  }
  std::size_t slot_index = 0;
  for (const AmrProgramHistoryDescriptor& history : state.histories) {
    for (std::size_t level = 0; level < state.level_clocks.size(); ++level) {
      std::optional<std::pair<bool, int>> level_publication;
      for (int slot = 0; slot < history.depth; ++slot) {
        if (slot_index >= state.history_slots.size())
          throw std::invalid_argument("exact AMR Program checkpoint omits history-slot provenance");
        const auto& provenance = state.history_slots[slot_index++];
        if (provenance.name != history.name || provenance.level != static_cast<int>(level) ||
            provenance.slot != slot)
          throw std::invalid_argument(
              "exact AMR Program checkpoint does not cover every history level and slot");
        const std::pair<bool, int> publication{provenance.initialized, provenance.fill_count};
        if (level_publication && *level_publication != publication)
          throw std::invalid_argument(
              "exact AMR Program checkpoint history publication differs between slots");
        level_publication = publication;
      }
    }
  }
  if (slot_index != state.history_slots.size())
    throw std::invalid_argument("exact AMR Program checkpoint has foreign history-slot provenance");
  std::string previous_pending;
  for (const auto& pending : state.pending_history_remaps) {
    constexpr std::string_view history_prefix = "pops.amr.level-history.v1/";
    const auto parse_pending_key = [&]() -> std::pair<int, std::string_view> {
      if (!std::string_view(pending.key).starts_with(history_prefix))
        throw std::invalid_argument(
            "exact AMR Program checkpoint pending history remap has a foreign key");
      const std::string_view suffix = std::string_view(pending.key).substr(history_prefix.size());
      const std::size_t slash = suffix.find('/');
      const std::size_t colon = suffix.find(':', slash == std::string_view::npos ? 0 : slash);
      if (slash == std::string_view::npos || colon == std::string_view::npos)
        throw std::invalid_argument(
            "exact AMR Program checkpoint pending history remap key is malformed");
      const auto parse_decimal = [](std::string_view text) -> std::uint64_t {
        if (text.empty())
          throw std::invalid_argument(
              "exact AMR Program checkpoint pending history remap key has an empty decimal field");
        std::uint64_t result = 0;
        for (const char character : text) {
          if (character < '0' || character > '9' ||
              result > (std::numeric_limits<std::uint64_t>::max() -
                        static_cast<std::uint64_t>(character - '0')) /
                           10)
            throw std::invalid_argument(
                "exact AMR Program checkpoint pending history remap key has an invalid decimal "
                "field");
          result = result * 10 + static_cast<std::uint64_t>(character - '0');
        }
        return result;
      };
      const std::uint64_t level = parse_decimal(suffix.substr(0, slash));
      const std::uint64_t name_size = parse_decimal(suffix.substr(slash + 1, colon - slash - 1));
      const std::string_view name = suffix.substr(colon + 1);
      if (level > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          name_size != name.size())
        throw std::invalid_argument(
            "exact AMR Program checkpoint pending history remap key is not canonical");
      return {static_cast<int>(level), name};
    };
    const auto [key_level, key_name] = parse_pending_key();
    const auto history =
        std::find_if(state.histories.begin(), state.histories.end(),
                     [&](const AmrProgramHistoryDescriptor& row) { return row.name == key_name; });
    const auto lag_slot = std::find_if(state.history_slots.begin(), state.history_slots.end(),
                                       [&](const AmrProgramHistorySlotProvenance& slot) {
                                         return slot.name == key_name &&
                                                slot.level == pending.child_level && slot.slot == 1;
                                       });
    if (pending.key.empty() || (!previous_pending.empty() && previous_pending >= pending.key) ||
        pending.parent_level < 0 || pending.parent_level == std::numeric_limits<int>::max() ||
        pending.child_level != pending.parent_level + 1 ||
        pending.child_level >= static_cast<int>(state.level_clocks.size()) || pending.consumed ||
        key_level != pending.child_level || history == state.histories.end() ||
        history->depth != 2 || lag_slot == state.history_slots.end() || !lag_slot->initialized ||
        lag_slot->fill_count != 2 || lag_slot->outgoing_dt != pending.source_dt ||
        pending.accepted_macro_step !=
            state.level_clocks[static_cast<std::size_t>(pending.child_level)].macro_step ||
        pending.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        pending.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
        pending.prior_topology_epoch + 1 != pending.published_topology_epoch ||
        pending.prior_materialization_generation + 1 !=
            pending.published_materialization_generation ||
        pending.published_topology_epoch != state.topology_epoch ||
        pending.published_materialization_generation != state.materialization_generation ||
        pending.accepted_macro_step < 0 || pending.temporal_denominator != 1 ||
        (pending.temporal_numerator != 1 && pending.temporal_numerator != 2) ||
        !std::isfinite(pending.source_dt) || !std::isfinite(pending.target_dt) ||
        !(pending.source_dt > 0.0) || !(pending.target_dt > 0.0) ||
        pending.target_dt != pending.source_dt / static_cast<double>(pending.temporal_numerator))
      throw std::invalid_argument(
          "exact AMR Program checkpoint has an invalid pending history remap");
    previous_pending = pending.key;
  }
  if (!state.history_flux_payload.empty() &&
      state.history_flux_payload.size() < sizeof(std::uint64_t))
    throw std::invalid_argument(
        "exact AMR Program checkpoint has a truncated history-flux payload");

  for (int axis = 0; axis < Dim; ++axis) {
    const auto validate_fragments = [&](const auto& fragments, std::string_view family) {
      std::optional<amr_reflux::FaceFluxFragmentKey<Dim>> previous;
      for (const auto& fragment : fragments) {
        if (fragment.key.axis != axis)
          throw std::invalid_argument("exact AMR Program checkpoint stores a " +
                                      std::string(family) + " face under another axis");
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
    };
    validate_fragments(state.accepted_face_flux[static_cast<std::size_t>(axis)], "flux");
  }
  std::optional<::pops::amr::InterfaceFluxFragmentKey> previous_interface;
  for (const auto& fragment : state.accepted_interface_flux) {
    ::pops::amr::validate_interface_flux_fragment(fragment.key, fragment.measure,
                                                  state.topology_epoch);
    if (!fragment.measure.stage_weight_resolved || fragment.payload.empty())
      throw std::invalid_argument(
          "exact AMR Program checkpoint interface fragment is not accepted");
    for (Real component : fragment.payload)
      if (!std::isfinite(static_cast<double>(component)))
        throw std::invalid_argument(
            "exact AMR Program checkpoint interface payload must be finite");
    if (previous_interface && !(previous_interface.value() < fragment.key))
      throw std::invalid_argument(
          "exact AMR Program checkpoint interface fragments must be uniquely ordered");
    previous_interface = fragment.key;
  }
  for (const AmrProgramSynchronizationEvent& event : state.synchronization_events) {
    if (event.parent_level < 0 || event.child_level != event.parent_level + 1 ||
        static_cast<std::size_t>(event.child_level) >= state.level_clocks.size() ||
        event.runtime_block < 0 || (event.phase != "reflux" && event.phase != "average_down") ||
        event.clock.level != event.parent_level || event.clock.macro_step < 0 ||
        !std::isfinite(event.clock.physical_time))
      throw std::invalid_argument(
          "exact AMR Program checkpoint has an invalid synchronization event");
  }
}

template <int Dim, class Output>
void write_state(Output& out, const AmrProgramAcceptedState<Dim>& state) {
  out.raw(kMagic);
  out.i32(Dim);
  out.string(state.spatial_contract);
  out.u64(state.topology_epoch);
  out.u64(state.materialization_generation);
  out.size(state.level_clocks.size());
  for (const auto& clock : state.level_clocks)
    write_clock(out, clock);
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
  out.size(state.history_slots.size());
  for (const AmrProgramHistorySlotProvenance& slot : state.history_slots) {
    out.string(slot.name);
    out.i32(slot.level);
    out.i32(slot.slot);
    out.real(slot.outgoing_dt);
    out.u64(slot.initialized ? 1U : 0U);
    out.i32(slot.fill_count);
  }
  out.size(state.pending_history_remaps.size());
  for (const auto& pending : state.pending_history_remaps) {
    out.string(pending.key);
    out.i32(pending.parent_level);
    out.i32(pending.child_level);
    out.u64(pending.prior_topology_epoch);
    out.u64(pending.prior_materialization_generation);
    out.u64(pending.published_topology_epoch);
    out.u64(pending.published_materialization_generation);
    out.i64(pending.accepted_macro_step);
    out.i64(pending.temporal_numerator);
    out.i64(pending.temporal_denominator);
    out.real(pending.source_dt);
    out.real(pending.target_dt);
    out.u64(pending.consumed ? 1U : 0U);
  }
  out.bytes(state.history_flux_payload);
  write_temporal_partition(out, state.temporal_partition);
  out.bytes(state.tagging_hysteresis_state);
  out.string(state.flux_budget_contract);
  out.string(state.coupling_contract);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto& fragments = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    out.size(fragments.size());
    for (const auto& fragment : fragments)
      write_face_fragment(out, fragment);
  }
  out.size(state.accepted_interface_flux.size());
  for (const auto& fragment : state.accepted_interface_flux)
    write_interface_fragment(out, fragment);
  out.size(state.synchronization_events.size());
  for (const AmrProgramSynchronizationEvent& event : state.synchronization_events) {
    out.i32(event.parent_level);
    out.i32(event.child_level);
    out.i32(event.runtime_block);
    out.string(event.phase);
    write_clock(out, event.clock);
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
  checkpoint_detail::write_state(out, state);
  return std::move(out).take();
}

template <int Dim>
std::size_t serialized_amr_program_accepted_state_size(const AmrProgramAcceptedState<Dim>& state) {
  checkpoint_detail::validate_state(state);
  checkpoint_detail::CountingWriter out;
  checkpoint_detail::write_state(out, state);
  return out.count();
}

/// Artifact-derived maximum POPSAND4 shape.  It carries character and term counts only: computing a
/// resource ceiling must never first allocate the potentially large scientific vectors it is meant
/// to bound.
template <int Dim>
struct AmrProgramAcceptedStateCapacity {
  std::size_t spatial_contract_characters = 0;
  std::size_t level_count = 0;
  std::vector<std::string> logical_clock_identities;
  std::vector<AmrProgramHistoryDescriptor> histories;
  std::string temporal_provider_identity;
  std::size_t temporal_cell_count = 0;
  std::size_t tagging_hysteresis_bytes = 0;
  std::size_t history_flux_payload_bytes = 0;
  std::size_t pending_history_remap_count = 0;
  std::size_t pending_history_remap_key_characters = 0;
  std::size_t flux_budget_contract_characters = 0;
  std::size_t coupling_contract_characters = 0;
  std::array<std::size_t, Dim> face_fragment_counts{};
  std::size_t face_owner_characters = 0;
  std::size_t face_state_characters = 0;
  std::size_t face_stage_characters = 0;
  std::size_t face_payload_terms = 0;
  std::size_t interface_fragment_count = 0;
  std::size_t interface_identity_characters = 0;
  std::size_t interface_program_identity_characters = 0;
  std::size_t interface_stage_characters = 0;
  std::size_t interface_payload_terms = 0;
  std::size_t synchronization_event_count = 0;
  std::size_t synchronization_phase_characters = 0;
};

template <int Dim>
std::size_t serialized_amr_program_accepted_state_capacity(
    const AmrProgramAcceptedStateCapacity<Dim>& capacity) {
  if (capacity.spatial_contract_characters == 0 || capacity.level_count == 0 ||
      capacity.logical_clock_identities.empty() || capacity.temporal_provider_identity.empty() ||
      capacity.flux_budget_contract_characters == 0 || capacity.coupling_contract_characters == 0)
    throw std::invalid_argument("AMR Program checkpoint capacity has incomplete frozen metadata");
  checkpoint_detail::CountingWriter out;
  out.raw(checkpoint_detail::kMagic);
  out.i32(Dim);
  out.string_size(capacity.spatial_contract_characters);
  out.u64(0);
  out.u64(0);
  out.size(capacity.level_count);
  out.repeated_bytes(capacity.level_count, checkpoint_detail::kEncodedClockBytes);
  out.size(capacity.logical_clock_identities.size());
  for (const std::string& identity : capacity.logical_clock_identities) {
    if (identity.empty())
      throw std::invalid_argument("AMR Program checkpoint capacity has an empty logical clock");
    out.string_size(identity.size());
    out.i64(0);
  }
  out.size(capacity.histories.size());
  std::size_t history_slot_count = 0;
  for (const AmrProgramHistoryDescriptor& history : capacity.histories) {
    if (history.name.empty() || history.program_owner < 0 || history.state_identity.empty() ||
        history.space_identity.empty() || history.clock_identity.empty() ||
        history.interpolation_identity.empty() || history.depth < 2 || history.components < 1)
      throw std::invalid_argument("AMR Program checkpoint capacity has an invalid history row");
    out.string_size(history.name.size());
    out.i32(history.program_owner);
    out.string_size(history.state_identity.size());
    out.string_size(history.space_identity.size());
    out.string_size(history.clock_identity.size());
    out.string_size(history.interpolation_identity.size());
    out.i32(history.depth);
    out.i32(history.components);
    const std::size_t depth = static_cast<std::size_t>(history.depth);
    if (capacity.level_count > std::numeric_limits<std::size_t>::max() / depth ||
        capacity.level_count * depth > std::numeric_limits<std::size_t>::max() - history_slot_count)
      throw std::length_error("AMR Program checkpoint history-slot capacity exceeds size_t");
    history_slot_count += capacity.level_count * depth;
  }
  out.size(history_slot_count);
  for (const AmrProgramHistoryDescriptor& history : capacity.histories) {
    const std::size_t count = capacity.level_count * static_cast<std::size_t>(history.depth);
    if (history.name.size() >
        std::numeric_limits<std::size_t>::max() - checkpoint_detail::kMinHistorySlotBytes)
      throw std::length_error("AMR Program checkpoint history name capacity exceeds size_t");
    out.repeated_bytes(count, checkpoint_detail::kMinHistorySlotBytes + history.name.size());
  }
  out.size(capacity.pending_history_remap_count);
  if (capacity.pending_history_remap_count != 0) {
    if (capacity.pending_history_remap_key_characters == 0)
      throw std::invalid_argument("AMR Program checkpoint capacity has empty pending-remap keys");
    constexpr std::size_t fixed = 2 * checkpoint_detail::kEncodedScalarBytes +
                                  4 * sizeof(std::uint64_t) + 3 * sizeof(std::int64_t) +
                                  2 * sizeof(double) + sizeof(std::uint64_t);
    out.repeated_bytes(capacity.pending_history_remap_count,
                       checkpoint_detail::kEncodedScalarBytes +
                           capacity.pending_history_remap_key_characters + fixed);
  }
  out.bytes_size(capacity.history_flux_payload_bytes);
  out.u64(0);
  out.string_size(capacity.temporal_provider_identity.size());
  out.u64(0);
  out.i64(0);
  out.i64(1);
  out.size(capacity.temporal_cell_count);
  out.repeated_bytes(capacity.temporal_cell_count,
                     checkpoint_detail::kMinTemporalPartitionRecordBytes);
  out.bytes_size(capacity.tagging_hysteresis_bytes);
  out.string_size(capacity.flux_budget_contract_characters);
  out.string_size(capacity.coupling_contract_characters);
  std::size_t face_fragment_count = 0;
  for (const std::size_t axis_count : capacity.face_fragment_counts) {
    out.size(axis_count);
    if (axis_count > std::numeric_limits<std::size_t>::max() - face_fragment_count)
      throw std::length_error("AMR Program face-fragment capacity exceeds size_t");
    face_fragment_count += axis_count;
    out.repeated_bytes(axis_count, checkpoint_detail::kMinFaceFragmentBytes<Dim>);
  }
  std::size_t face_characters = capacity.face_owner_characters;
  if (capacity.face_state_characters > std::numeric_limits<std::size_t>::max() - face_characters)
    throw std::length_error("AMR Program face identity capacity exceeds size_t");
  face_characters += capacity.face_state_characters;
  if (capacity.face_stage_characters > std::numeric_limits<std::size_t>::max() - face_characters)
    throw std::length_error("AMR Program face identity capacity exceeds size_t");
  face_characters += capacity.face_stage_characters;
  out.repeated_bytes(face_fragment_count, face_characters);
  out.repeated_bytes(capacity.face_payload_terms, sizeof(double));

  out.size(capacity.interface_fragment_count);
  out.repeated_bytes(capacity.interface_fragment_count,
                     checkpoint_detail::kMinInterfaceFragmentBytes);
  std::size_t interface_characters = capacity.interface_identity_characters;
  for (const std::size_t additional :
       {capacity.interface_program_identity_characters, capacity.interface_stage_characters}) {
    if (additional > std::numeric_limits<std::size_t>::max() - interface_characters)
      throw std::length_error("AMR Program interface identity capacity exceeds size_t");
    interface_characters += additional;
  }
  out.repeated_bytes(capacity.interface_fragment_count, interface_characters);
  out.repeated_bytes(capacity.interface_payload_terms, sizeof(double));

  out.size(capacity.synchronization_event_count);
  out.repeated_bytes(capacity.synchronization_event_count,
                     checkpoint_detail::kMinSynchronizationEventBytes);
  out.repeated_bytes(capacity.synchronization_event_count,
                     capacity.synchronization_phase_characters);
  return out.count();
}

template <int Dim>
AmrProgramAcceptedState<Dim> deserialize_amr_program_accepted_state(
    std::span<const std::uint8_t> bytes,
    const ::pops::amr::InterfaceFluxLedgerBudget* interface_budget = nullptr) {
  checkpoint_detail::Reader in(bytes);
  in.expect_raw(checkpoint_detail::kMagic);
  if (in.i32() != Dim)
    throw std::runtime_error(
        "invalid exact AMR Program checkpoint: native dimension does not match the artifact");
  AmrProgramAcceptedState<Dim> state;
  state.spatial_contract = in.string();
  state.topology_epoch = in.u64();
  state.materialization_generation = in.u64();
  state.level_clocks.resize(in.size(checkpoint_detail::kEncodedClockBytes));
  for (auto& clock : state.level_clocks)
    clock = checkpoint_detail::read_clock(in);
  const std::size_t logical_clock_count = in.size(checkpoint_detail::kMinLogicalClockBytes);
  for (std::size_t index = 0; index < logical_clock_count; ++index) {
    std::string identity = in.string();
    const std::int64_t tick = in.i64();
    if (!state.logical_clock_ticks.emplace(std::move(identity), tick).second)
      throw std::runtime_error(
          "invalid exact AMR Program checkpoint: duplicate logical clock identity");
  }
  state.histories.resize(in.size(checkpoint_detail::kMinHistoryDescriptorBytes));
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
  state.history_slots.resize(in.size(checkpoint_detail::kMinHistorySlotBytes));
  for (AmrProgramHistorySlotProvenance& slot : state.history_slots) {
    slot.name = in.string();
    slot.level = in.i32();
    slot.slot = in.i32();
    slot.outgoing_dt = in.real();
    const std::uint64_t initialized = in.u64();
    if (initialized > 1U)
      throw std::runtime_error(
          "invalid exact AMR Program checkpoint: invalid history initialized tag");
    slot.initialized = initialized != 0;
    slot.fill_count = in.i32();
  }
  state.pending_history_remaps.resize(in.size(checkpoint_detail::kMinPendingHistoryRemapBytes));
  for (auto& pending : state.pending_history_remaps) {
    pending.key = in.string();
    pending.parent_level = in.i32();
    pending.child_level = in.i32();
    pending.prior_topology_epoch = in.u64();
    pending.prior_materialization_generation = in.u64();
    pending.published_topology_epoch = in.u64();
    pending.published_materialization_generation = in.u64();
    pending.accepted_macro_step = in.i64();
    pending.temporal_numerator = in.i64();
    pending.temporal_denominator = in.i64();
    pending.source_dt = in.real();
    pending.target_dt = in.real();
    const std::uint64_t consumed = in.u64();
    if (consumed > 1U)
      throw std::runtime_error("invalid exact AMR Program checkpoint: invalid pending history tag");
    pending.consumed = consumed != 0;
  }
  state.history_flux_payload = in.bytes();
  state.temporal_partition = checkpoint_detail::read_temporal_partition(in);
  state.tagging_hysteresis_state = in.bytes();
  state.flux_budget_contract = in.string();
  state.coupling_contract = in.string();
  for (int axis = 0; axis < Dim; ++axis) {
    auto& fragments = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    fragments.resize(in.size(checkpoint_detail::kMinFaceFragmentBytes<Dim>));
    for (auto& fragment : fragments)
      fragment = checkpoint_detail::read_face_fragment<Dim>(in);
  }
  const std::size_t interface_count = in.size(checkpoint_detail::kMinInterfaceFragmentBytes);
  if (interface_budget != nullptr && interface_count > interface_budget->max_fragments_per_window)
    throw std::length_error(
        "invalid exact AMR Program checkpoint: interface fragments exceed artifact budget");
  state.accepted_interface_flux.resize(interface_count);
  std::size_t remaining_interface_terms = interface_budget == nullptr
                                              ? std::numeric_limits<std::size_t>::max()
                                              : interface_budget->max_payload_terms_per_window;
  for (auto& fragment : state.accepted_interface_flux)
    fragment = checkpoint_detail::read_interface_fragment(
        in, interface_budget == nullptr ? nullptr : &remaining_interface_terms);
  state.synchronization_events.resize(in.size(checkpoint_detail::kMinSynchronizationEventBytes));
  for (AmrProgramSynchronizationEvent& event : state.synchronization_events) {
    event.parent_level = in.i32();
    event.child_level = in.i32();
    event.runtime_block = in.i32();
    event.phase = in.string();
    event.clock = checkpoint_detail::read_clock(in);
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

template <int Dim>
::pops::amr::TransactionalInterfaceFluxLedger<AmrProgramFacePayload>
restore_amr_program_interface_flux_ledger(const AmrProgramAcceptedState<Dim>& state,
                                          ::pops::amr::InterfaceFluxLedgerBudget budget) {
  checkpoint_detail::validate_state(state);
  ::pops::amr::TransactionalInterfaceFluxLedger<AmrProgramFacePayload> ledger(state.topology_epoch,
                                                                              std::move(budget));
  if (state.accepted_interface_flux.empty())
    return ledger;
  ledger.begin();
  try {
    for (const auto& fragment : state.accepted_interface_flux)
      ledger.accumulate(fragment.key, fragment.measure, fragment.payload);
    ledger.commit();
  } catch (...) {
    if (ledger.in_transaction())
      ledger.rollback();
    throw;
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
