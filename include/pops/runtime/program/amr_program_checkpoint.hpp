#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_ledger.hpp>
#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>
#include <pops/runtime/amr/amr_program_reflux.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

namespace pops::runtime::program {

namespace amr = ::pops::amr;

struct AmrProgramFluxContribution {
  int rate_id = -1;
  amr::Rational weight{1, 1};
  int dt_power = 0;
  double duration = 0.0;
  amr::ClockStamp evaluation_clock;
  EdgeFlux payload;
};

struct AmrProgramFluxAuditEntry {
  amr::FluxLedgerKey key;
  amr::FluxMeasure measure;
};

struct AmrProgramInterfaceFluxAuditEntry {
  amr::InterfaceFluxFragmentKey key;
  amr::InterfaceFluxFragmentMeasure measure;
};

struct AmrProgramSyncEvent {
  int parent_level = 0;
  int child_level = 0;
  int block = 0;
  int phase = 0;
  amr::ClockStamp clock;
};

/// Complete accepted state owned by the compiled AMR Program context.  Engine-owned history values
/// remain in the regular checkpoint arrays; this image carries the semantic clock/identity authority
/// and the lagged effective-flux strips which cannot be reconstructed from state buffers alone.
struct AmrProgramAcceptedState {
  std::vector<amr::ClockStamp> level_clocks;
  std::map<std::string, std::int64_t> logical_clock_ticks;
  CellTemporalPartitionAcceptedState temporal_partition;
  /// Rank-independent canonical image of the runtime-owned AMR tagging hysteresis.
  std::vector<std::uint8_t> tagging_hysteresis_state;
  std::map<std::string, int> history_owners;
  std::map<std::string, std::string> history_states;
  std::map<std::string, std::string> history_spaces;
  std::map<std::string, std::string> history_clocks;
  std::map<std::string, std::string> history_interpolations;
  std::map<std::string, std::vector<std::vector<amr::ClockStamp>>> ring_clocks;
  std::map<std::string, std::vector<std::vector<std::optional<amr::HistoryIdentity>>>>
      ring_identities;
  std::map<std::string, std::vector<std::vector<EdgeFlux>>> ring_flux;
  std::map<std::string, std::vector<std::vector<std::vector<AmrProgramFluxContribution>>>>
      ring_flux_contributions;
  std::map<std::string, std::vector<char>> ring_flux_initialized;
  std::vector<AmrProgramFluxAuditEntry> accepted_flux_ledger;
  std::vector<AmrProgramInterfaceFluxAuditEntry> accepted_interface_flux_ledger;
  std::vector<AmrProgramSyncEvent> accepted_sync;
};

/// Rank ownership of the exact recorded global patch order at every active AMR level.
///
/// This is deliberately separate from the accepted-state byte protocol: changing only MPI
/// cardinality must not change the scientific checkpoint image.  A caller that rematerializes a
/// checkpoint supplies both the recorded source ownership and the already prepared target ownership;
/// this layer never invents a rank mapping.
struct AmrProgramRankOwnership {
  int rank_count = 0;
  std::vector<std::vector<int>> level_patch_owners;
};

namespace checkpoint_detail {

class Writer {
 public:
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
  void string(const std::string& value) {
    size(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void bytes(const std::vector<std::uint8_t>& value) {
    size(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void size(std::size_t value) { u64(static_cast<std::uint64_t>(value)); }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

 private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

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
  [[noreturn]] static void fail_(const std::string& why) {
    throw std::runtime_error("invalid AMR Program accepted-state payload: " + why);
  }
  void require_(std::size_t count) const {
    if (count > bytes_.size() - cursor_)
      fail_("truncated payload");
  }
  const std::vector<std::uint8_t>& bytes_;
  std::size_t cursor_ = 0;
};

inline void write_clock(Writer& out, const amr::ClockStamp& value) {
  out.i32(value.level);
  out.i64(value.macro_step);
  out.i64(value.phase.numerator);
  out.i64(value.phase.denominator);
  out.real(value.physical_time);
}

inline amr::ClockStamp read_clock(Reader& in) {
  const int level = in.i32();
  const std::int64_t macro_step = in.i64();
  const std::int64_t numerator = in.i64();
  const std::int64_t denominator = in.i64();
  const double physical_time = in.real();
  return {level, macro_step, amr::Rational(numerator, denominator), physical_time};
}

inline void write_rational(Writer& out, const amr::Rational& value) {
  out.i64(value.numerator);
  out.i64(value.denominator);
}

inline amr::Rational read_rational(Reader& in) {
  return {in.i64(), in.i64()};
}

inline void write_identity(Writer& out, const amr::HistoryIdentity& value) {
  out.string(value.owner);
  out.string(value.state);
  out.string(value.space);
  out.i32(value.level);
  write_clock(out, value.clock);
}

inline amr::HistoryIdentity read_identity(Reader& in) {
  amr::HistoryIdentity value;
  value.owner = in.string();
  value.state = in.string();
  value.space = in.string();
  value.level = in.i32();
  value.clock = read_clock(in);
  return value;
}

template <class Allocator>
inline void write_reals(Writer& out, const std::vector<Real, Allocator>& values) {
  out.size(values.size());
  for (Real value : values)
    out.real(static_cast<double>(value));
}

inline RefluxStorage<Real> read_reflux_reals(Reader& in) {
  RefluxStorage<Real> values(in.size());
  for (Real& value : values)
    value = static_cast<Real>(in.real());
  return values;
}

inline void write_strip(Writer& out, const EdgeStrip& value) {
  out.i32(value.I0);
  out.i32(value.I1);
  out.i32(value.J0);
  out.i32(value.J1);
  write_reals(out, value.cL);
  write_reals(out, value.cR);
  write_reals(out, value.cB);
  write_reals(out, value.cT);
  write_reals(out, value.fL);
  write_reals(out, value.fR);
  write_reals(out, value.fB);
  write_reals(out, value.fT);
}

inline EdgeStrip read_strip(Reader& in) {
  EdgeStrip value;
  value.I0 = in.i32();
  value.I1 = in.i32();
  value.J0 = in.i32();
  value.J1 = in.i32();
  value.cL = read_reflux_reals(in);
  value.cR = read_reflux_reals(in);
  value.cB = read_reflux_reals(in);
  value.cT = read_reflux_reals(in);
  value.fL = read_reflux_reals(in);
  value.fR = read_reflux_reals(in);
  value.fB = read_reflux_reals(in);
  value.fT = read_reflux_reals(in);
  return value;
}

inline void write_flux(Writer& out, const EdgeFlux& value) {
  out.size(value.coarse.size());
  for (const EdgeStrip& strip : value.coarse)
    write_strip(out, strip);
  out.size(value.fine.size());
  for (const EdgeStrip& strip : value.fine)
    write_strip(out, strip);
}

inline EdgeFlux read_flux(Reader& in) {
  EdgeFlux value;
  value.coarse.resize(in.size());
  for (EdgeStrip& strip : value.coarse)
    strip = read_strip(in);
  value.fine.resize(in.size());
  for (EdgeStrip& strip : value.fine)
    strip = read_strip(in);
  return value;
}

inline void write_contribution(Writer& out, const AmrProgramFluxContribution& value) {
  out.i32(value.rate_id);
  write_rational(out, value.weight);
  out.i32(value.dt_power);
  out.real(value.duration);
  write_clock(out, value.evaluation_clock);
  write_flux(out, value.payload);
}

inline AmrProgramFluxContribution read_contribution(Reader& in) {
  AmrProgramFluxContribution value;
  value.rate_id = in.i32();
  value.weight = read_rational(in);
  value.dt_power = in.i32();
  value.duration = in.real();
  value.evaluation_clock = read_clock(in);
  value.payload = read_flux(in);
  return value;
}

inline void write_flux_audit(Writer& out, const AmrProgramFluxAuditEntry& value) {
  out.string(value.key.owner);
  out.string(value.key.state);
  out.string(value.key.rate);
  out.string(value.key.flux);
  out.i32(value.key.level);
  write_clock(out, value.key.clock);
  write_rational(out, value.measure.stage_weight);
  out.i32(static_cast<int>(value.measure.orientation));
  out.real(value.measure.face_measure);
  out.real(value.measure.substep_duration);
}

inline AmrProgramFluxAuditEntry read_flux_audit(Reader& in) {
  AmrProgramFluxAuditEntry value;
  value.key.owner = in.string();
  value.key.state = in.string();
  value.key.rate = in.string();
  value.key.flux = in.string();
  value.key.level = in.i32();
  value.key.clock = read_clock(in);
  value.measure.stage_weight = read_rational(in);
  const int orientation = in.i32();
  if (orientation < static_cast<int>(amr::FluxOrientation::XMinus) ||
      orientation > static_cast<int>(amr::FluxOrientation::YPlus))
    throw std::runtime_error(
        "invalid AMR Program accepted-state payload: invalid flux orientation");
  value.measure.orientation = static_cast<amr::FluxOrientation>(orientation);
  value.measure.face_measure = in.real();
  value.measure.substep_duration = in.real();
  return value;
}

inline void write_interface_flux_audit(Writer& out,
                                       const AmrProgramInterfaceFluxAuditEntry& value) {
  out.string(value.key.interface_identity);
  out.u64(value.key.topology_epoch);
  out.i32(value.key.coarse_level);
  out.i32(value.key.fine_level);
  write_clock(out, value.key.clock);
  out.string(value.key.stage_identity);
  write_clock(out, value.key.interval.begin);
  write_clock(out, value.key.interval.end);
  out.i32(static_cast<int>(value.key.orientation));
  out.u64(static_cast<std::uint64_t>(value.key.left_block));
  out.u64(static_cast<std::uint64_t>(value.key.right_block));
  write_rational(out, value.measure.stage_weight);
  out.real(value.measure.face_measure);
  out.real(value.measure.substep_duration);
  out.u64(value.measure.stage_weight_resolved ? 1 : 0);
}

inline AmrProgramInterfaceFluxAuditEntry read_interface_flux_audit(Reader& in) {
  AmrProgramInterfaceFluxAuditEntry value;
  value.key.interface_identity = in.string();
  value.key.topology_epoch = in.u64();
  value.key.coarse_level = in.i32();
  value.key.fine_level = in.i32();
  value.key.clock = read_clock(in);
  value.key.stage_identity = in.string();
  value.key.interval.begin = read_clock(in);
  value.key.interval.end = read_clock(in);
  const int orientation = in.i32();
  if (orientation < static_cast<int>(amr::InterfaceFluxOrientation::CoarseOutward) ||
      orientation > static_cast<int>(amr::InterfaceFluxOrientation::FineOutward))
    throw std::runtime_error(
        "invalid AMR Program accepted-state payload: invalid interface-flux orientation");
  value.key.orientation = static_cast<amr::InterfaceFluxOrientation>(orientation);
  const std::uint64_t left_block = in.u64();
  const std::uint64_t right_block = in.u64();
  if (left_block > std::numeric_limits<std::size_t>::max() ||
      right_block > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error(
        "invalid AMR Program accepted-state payload: interface-flux block index overflows size_t");
  value.key.left_block = static_cast<std::size_t>(left_block);
  value.key.right_block = static_cast<std::size_t>(right_block);
  value.measure.stage_weight = read_rational(in);
  value.measure.face_measure = in.real();
  value.measure.substep_duration = in.real();
  const std::uint64_t resolved = in.u64();
  if (resolved > 1)
    throw std::runtime_error(
        "invalid AMR Program accepted-state payload: invalid interface-flux resolved flag");
  value.measure.stage_weight_resolved = resolved != 0;
  try {
    amr::validate_interface_flux_fragment(value.key, value.measure, value.key.topology_epoch);
  } catch (const std::exception& error) {
    throw std::runtime_error(
        std::string("invalid AMR Program accepted-state payload: invalid interface flux: ") +
        error.what());
  }
  if (!value.measure.stage_weight_resolved)
    throw std::runtime_error(
        "invalid AMR Program accepted-state payload: accepted interface flux has unresolved "
        "Program stage weight");
  return value;
}

inline void write_sync(Writer& out, const AmrProgramSyncEvent& value) {
  out.i32(value.parent_level);
  out.i32(value.child_level);
  out.i32(value.block);
  out.i32(value.phase);
  write_clock(out, value.clock);
}

inline AmrProgramSyncEvent read_sync(Reader& in) {
  AmrProgramSyncEvent value;
  value.parent_level = in.i32();
  value.child_level = in.i32();
  value.block = in.i32();
  value.phase = in.i32();
  value.clock = read_clock(in);
  return value;
}

template <class Map, class WriteValue>
void write_map(Writer& out, const Map& values, WriteValue&& write_value) {
  out.size(values.size());
  for (const auto& [name, value] : values) {
    out.string(name);
    write_value(out, value);
  }
}

template <class Map, class ReadValue>
Map read_map(Reader& in, ReadValue&& read_value) {
  Map values;
  const std::size_t count = in.size();
  for (std::size_t index = 0; index < count; ++index) {
    std::string name = in.string();
    if (!values.emplace(std::move(name), read_value(in)).second)
      throw std::runtime_error("invalid AMR Program accepted-state payload: duplicate map key");
  }
  return values;
}

}  // namespace checkpoint_detail

inline std::vector<std::uint8_t> serialize_amr_program_accepted_state(
    const AmrProgramAcceptedState& state) {
  using namespace checkpoint_detail;
  validate_cell_temporal_partition_state(state.temporal_partition);
  Writer out;
  out.u64(0x3554534153504f50ULL);  // "POPSAST5", little-endian bytes
  out.size(state.level_clocks.size());
  for (const auto& clock : state.level_clocks)
    write_clock(out, clock);
  write_map(out, state.logical_clock_ticks, [](Writer& w, std::int64_t value) { w.i64(value); });
  out.u64(static_cast<std::uint64_t>(state.temporal_partition.kind));
  out.string(state.temporal_partition.provider_identity);
  out.u64(state.temporal_partition.topology_epoch);
  out.i64(state.temporal_partition.synchronization_tick);
  out.i64(state.temporal_partition.tick_denominator);
  out.size(state.temporal_partition.cells.size());
  for (const CellTemporalPartitionRecord& cell : state.temporal_partition.cells) {
    out.i32(cell.level);
    out.u64(cell.cell);
    out.i32(cell.rung);
    out.i64(cell.accepted_tick);
  }
  out.bytes(state.tagging_hysteresis_state);
  write_map(out, state.history_owners, [](Writer& w, int v) { w.i32(v); });
  write_map(out, state.history_states, [](Writer& w, const std::string& v) { w.string(v); });
  write_map(out, state.history_spaces, [](Writer& w, const std::string& v) { w.string(v); });
  write_map(out, state.history_clocks, [](Writer& w, const std::string& v) { w.string(v); });
  write_map(out, state.history_interpolations,
            [](Writer& w, const std::string& v) { w.string(v); });
  write_map(out, state.ring_clocks, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const auto& clock : slot)
        write_clock(w, clock);
    }
  });
  write_map(out, state.ring_identities, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const auto& identity : slot) {
        w.u64(identity ? 1 : 0);
        if (identity)
          write_identity(w, *identity);
      }
    }
  });
  write_map(out, state.ring_flux, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const EdgeFlux& flux : slot)
        write_flux(w, flux);
    }
  });
  write_map(out, state.ring_flux_contributions, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const auto& level : slot) {
        w.size(level.size());
        for (const auto& contribution : level)
          write_contribution(w, contribution);
      }
    }
  });
  write_map(out, state.ring_flux_initialized, [](Writer& w, const auto& values) {
    w.size(values.size());
    for (char value : values)
      w.u64(value ? 1 : 0);
  });
  out.size(state.accepted_flux_ledger.size());
  for (const auto& entry : state.accepted_flux_ledger)
    write_flux_audit(out, entry);
  out.size(state.accepted_interface_flux_ledger.size());
  for (const auto& entry : state.accepted_interface_flux_ledger)
    write_interface_flux_audit(out, entry);
  out.size(state.accepted_sync.size());
  for (const auto& event : state.accepted_sync)
    write_sync(out, event);
  return out.take();
}

inline AmrProgramAcceptedState deserialize_amr_program_accepted_state(
    const std::vector<std::uint8_t>& bytes) {
  using namespace checkpoint_detail;
  Reader in(bytes);
  const std::uint64_t magic = in.u64();
  const bool carries_temporal_partition = magic == 0x3554534153504f50ULL;
  if (!carries_temporal_partition && magic != 0x3454534153504f50ULL)
    throw std::runtime_error(
        "invalid AMR Program accepted-state payload: unsupported magic/version");
  AmrProgramAcceptedState state;
  state.level_clocks.resize(in.size());
  for (auto& clock : state.level_clocks)
    clock = read_clock(in);
  state.logical_clock_ticks =
      read_map<decltype(state.logical_clock_ticks)>(in, [](Reader& r) { return r.i64(); });
  if (carries_temporal_partition) {
    const std::uint64_t kind = in.u64();
    if (kind > static_cast<std::uint64_t>(TemporalPartitionKind::CellLocal))
      throw std::runtime_error(
          "invalid AMR Program accepted-state payload: unsupported temporal partition kind");
    state.temporal_partition.kind = static_cast<TemporalPartitionKind>(kind);
    state.temporal_partition.provider_identity = in.string();
    state.temporal_partition.topology_epoch = in.u64();
    state.temporal_partition.synchronization_tick = in.i64();
    state.temporal_partition.tick_denominator = in.i64();
    state.temporal_partition.cells.resize(in.size());
    for (CellTemporalPartitionRecord& cell : state.temporal_partition.cells) {
      cell.level = in.i32();
      cell.cell = in.u64();
      cell.rung = in.i32();
      cell.accepted_tick = in.i64();
    }
    try {
      validate_cell_temporal_partition_state(state.temporal_partition);
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string("invalid AMR Program accepted-state payload: ") +
                               error.what());
    }
  }
  state.tagging_hysteresis_state = in.bytes();
  state.history_owners =
      read_map<std::map<std::string, int>>(in, [](Reader& r) { return r.i32(); });
  state.history_states =
      read_map<std::map<std::string, std::string>>(in, [](Reader& r) { return r.string(); });
  state.history_spaces =
      read_map<std::map<std::string, std::string>>(in, [](Reader& r) { return r.string(); });
  state.history_clocks =
      read_map<decltype(state.history_clocks)>(in, [](Reader& r) { return r.string(); });
  state.history_interpolations =
      read_map<decltype(state.history_interpolations)>(in, [](Reader& r) { return r.string(); });
  state.ring_clocks = read_map<decltype(state.ring_clocks)>(in, [](Reader& r) {
    std::vector<std::vector<amr::ClockStamp>> ring(r.size());
    for (auto& slot : ring) {
      slot.resize(r.size());
      for (auto& clock : slot)
        clock = read_clock(r);
    }
    return ring;
  });
  state.ring_identities = read_map<decltype(state.ring_identities)>(in, [](Reader& r) {
    std::vector<std::vector<std::optional<amr::HistoryIdentity>>> ring(r.size());
    for (auto& slot : ring) {
      slot.resize(r.size());
      for (auto& identity : slot) {
        const std::uint64_t present = r.u64();
        if (present > 1)
          throw std::runtime_error(
              "invalid AMR Program accepted-state payload: invalid optional flag");
        if (present)
          identity = read_identity(r);
      }
    }
    return ring;
  });
  state.ring_flux = read_map<decltype(state.ring_flux)>(in, [](Reader& r) {
    std::vector<std::vector<EdgeFlux>> ring(r.size());
    for (auto& slot : ring) {
      slot.resize(r.size());
      for (auto& flux : slot)
        flux = read_flux(r);
    }
    return ring;
  });
  state.ring_flux_contributions =
      read_map<decltype(state.ring_flux_contributions)>(in, [](Reader& r) {
        std::vector<std::vector<std::vector<AmrProgramFluxContribution>>> ring(r.size());
        for (auto& slot : ring) {
          slot.resize(r.size());
          for (auto& level : slot) {
            level.resize(r.size());
            for (auto& contribution : level)
              contribution = read_contribution(r);
          }
        }
        return ring;
      });
  state.ring_flux_initialized = read_map<decltype(state.ring_flux_initialized)>(in, [](Reader& r) {
    std::vector<char> values(r.size());
    for (char& value : values) {
      const std::uint64_t flag = r.u64();
      if (flag > 1)
        throw std::runtime_error("invalid AMR Program accepted-state payload: invalid flag");
      value = flag ? 1 : 0;
    }
    return values;
  });
  state.accepted_flux_ledger.resize(in.size());
  for (auto& entry : state.accepted_flux_ledger)
    entry = read_flux_audit(in);
  state.accepted_interface_flux_ledger.resize(in.size());
  for (auto& entry : state.accepted_interface_flux_ledger)
    entry = read_interface_flux_audit(in);
  state.accepted_sync.resize(in.size());
  for (auto& event : state.accepted_sync)
    event = read_sync(in);
  in.finish();
  return state;
}

namespace rematerialization_detail {

inline constexpr const char* kErrorPrefix = "AMR Program accepted-state rematerialization: ";

[[noreturn]] inline void fail(const std::string& why) {
  throw std::runtime_error(std::string(kErrorPrefix) + why);
}

inline void validate_ownership(const AmrProgramRankOwnership& ownership,
                               std::size_t expected_levels, const char* role) {
  if (ownership.rank_count <= 0)
    fail(std::string(role) + " rank count must be positive");
  if (ownership.level_patch_owners.size() != expected_levels)
    fail(std::string(role) + " ownership level count differs from the accepted hierarchy");
  for (std::size_t level = 0; level < ownership.level_patch_owners.size(); ++level) {
    if (ownership.level_patch_owners[level].empty())
      fail(std::string(role) + " ownership has no active patch at level " + std::to_string(level));
    for (int owner : ownership.level_patch_owners[level])
      if (owner < 0 || owner >= ownership.rank_count)
        fail(std::string(role) + " ownership contains an out-of-range rank at level " +
             std::to_string(level));
  }
}

inline AmrProgramAcceptedState without_rank_payloads(AmrProgramAcceptedState state) {
  for (auto& [name, ring] : state.ring_flux) {
    (void)name;
    for (auto& slot : ring)
      for (EdgeFlux& flux : slot)
        flux = EdgeFlux{};
  }
  for (auto& [name, ring] : state.ring_flux_contributions) {
    (void)name;
    for (auto& slot : ring)
      for (auto& level : slot)
        for (AmrProgramFluxContribution& contribution : level)
          contribution.payload = EdgeFlux{};
  }
  return state;
}

using StripRole = std::vector<EdgeStrip>;

inline void rematerialize_strip_role(const std::vector<const StripRole*>& source_roles,
                                     const std::vector<StripRole*>& target_roles,
                                     const std::vector<int>& source_owners,
                                     const std::vector<int>& target_owners,
                                     const std::vector<int>& target_owner_to_slot,
                                     const std::string& context) {
  if (source_owners.size() != target_owners.size())
    fail(context + " source/target global patch counts differ");
  const std::size_t patch_count = source_owners.size();
  const std::size_t recorded_axis_size = source_roles.front()->size();
  bool materialized_axis = false;
  for (std::size_t rank = 0; rank < source_roles.size(); ++rank) {
    const std::size_t size = source_roles[rank]->size();
    if (size != 0 && size != patch_count)
      fail(context + " rank " + std::to_string(rank) +
           " strip axis differs from the recorded global patch count");
    if (size != recorded_axis_size)
      fail(context + " strip axis differs across source ranks");
    materialized_axis = materialized_axis || size != 0;
  }
  for (StripRole* role : target_roles) {
    role->clear();
    if (materialized_axis)
      role->resize(patch_count);
  }
  if (!materialized_axis)
    return;

  for (std::size_t patch = 0; patch < patch_count; ++patch) {
    const EdgeStrip* active = nullptr;
    int active_rank = -1;
    for (std::size_t rank = 0; rank < source_roles.size(); ++rank) {
      const StripRole& role = *source_roles[rank];
      if (role.empty() || !::pops::detail::edge_strip_has_storage(role[patch]))
        continue;
      if (active != nullptr)
        fail(context + " patch " + std::to_string(patch) +
             " has duplicate active payloads on source ranks " + std::to_string(active_rank) +
             " and " + std::to_string(rank));
      active = &role[patch];
      active_rank = static_cast<int>(rank);
    }
    if (active == nullptr)
      continue;
    if (source_owners[patch] != active_rank)
      fail(context + " patch " + std::to_string(patch) + " is active on source rank " +
           std::to_string(active_rank) + " but its recorded owner is rank " +
           std::to_string(source_owners[patch]));
    const int target_slot = target_owner_to_slot[static_cast<std::size_t>(target_owners[patch])];
    if (target_slot >= 0)
      (*target_roles[static_cast<std::size_t>(target_slot)])[patch] = *active;
  }
}

inline void rematerialize_edge_flux(const std::vector<const EdgeFlux*>& source_fluxes,
                                    const std::vector<EdgeFlux*>& target_fluxes, std::size_t level,
                                    const AmrProgramRankOwnership& source_ownership,
                                    const AmrProgramRankOwnership& target_ownership,
                                    const std::vector<int>& target_owner_to_slot,
                                    const std::string& context) {
  static const std::vector<int> empty_owners;
  const std::size_t level_count = source_ownership.level_patch_owners.size();
  if (level >= level_count)
    fail(context + " level axis exceeds the accepted hierarchy");
  if (source_fluxes.size() != static_cast<std::size_t>(source_ownership.rank_count))
    fail(context + " source payload count differs from the recorded rank count");
  const std::vector<int>& source_coarse_owners =
      level + 1 < level_count ? source_ownership.level_patch_owners[level + 1] : empty_owners;
  const std::vector<int>& target_coarse_owners =
      level + 1 < level_count ? target_ownership.level_patch_owners[level + 1] : empty_owners;
  const std::vector<int>& source_fine_owners =
      level > 0 ? source_ownership.level_patch_owners[level] : empty_owners;
  const std::vector<int>& target_fine_owners =
      level > 0 ? target_ownership.level_patch_owners[level] : empty_owners;

  std::vector<const StripRole*> source_roles;
  std::vector<StripRole*> target_roles;
  source_roles.reserve(source_fluxes.size());
  target_roles.reserve(target_fluxes.size());
  for (const EdgeFlux* flux : source_fluxes)
    source_roles.push_back(&flux->coarse);
  for (EdgeFlux* flux : target_fluxes)
    target_roles.push_back(&flux->coarse);
  rematerialize_strip_role(source_roles, target_roles, source_coarse_owners, target_coarse_owners,
                           target_owner_to_slot, context + " coarse role");

  source_roles.clear();
  target_roles.clear();
  for (const EdgeFlux* flux : source_fluxes)
    source_roles.push_back(&flux->fine);
  for (EdgeFlux* flux : target_fluxes)
    target_roles.push_back(&flux->fine);
  rematerialize_strip_role(source_roles, target_roles, source_fine_owners, target_fine_owners,
                           target_owner_to_slot, context + " fine role");
}

inline std::vector<AmrProgramAcceptedState> rematerialize_selected_target_ranks(
    const std::vector<AmrProgramAcceptedState>& source_rank_states,
    const AmrProgramRankOwnership& source_ownership,
    const AmrProgramRankOwnership& target_ownership, const std::vector<int>& target_ranks) {
  if (source_rank_states.empty())
    fail("at least one source rank state is required");
  if (source_ownership.rank_count != static_cast<int>(source_rank_states.size()))
    fail("source state count differs from the recorded source rank count");

  const std::size_t level_count = source_rank_states.front().level_clocks.size();
  if (level_count == 0)
    fail("accepted hierarchy has no active level");
  validate_ownership(source_ownership, level_count, "source");
  validate_ownership(target_ownership, level_count, "target");
  for (std::size_t level = 0; level < level_count; ++level)
    if (source_ownership.level_patch_owners[level].size() !=
        target_ownership.level_patch_owners[level].size())
      fail("source/target ownership differs from the recorded patch count at level " +
           std::to_string(level));
  if (target_ranks.empty())
    fail("at least one target rank must be selected");
  std::vector<int> target_owner_to_slot(static_cast<std::size_t>(target_ownership.rank_count), -1);
  for (std::size_t slot = 0; slot < target_ranks.size(); ++slot) {
    const int rank = target_ranks[slot];
    if (rank < 0 || rank >= target_ownership.rank_count)
      fail("selected target rank is out of range");
    int& existing = target_owner_to_slot[static_cast<std::size_t>(rank)];
    if (existing >= 0)
      fail("selected target rank is duplicated");
    existing = static_cast<int>(slot);
  }

  AmrProgramAcceptedState common = without_rank_payloads(source_rank_states.front());
  const std::vector<std::uint8_t> common_image = serialize_amr_program_accepted_state(common);
  for (std::size_t rank = 1; rank < source_rank_states.size(); ++rank)
    if (serialize_amr_program_accepted_state(without_rank_payloads(source_rank_states[rank])) !=
        common_image)
      fail("source rank " + std::to_string(rank) +
           " disagrees on common clocks, tagging hysteresis, history metadata or "
           "accepted reports");

  std::vector<AmrProgramAcceptedState> result(target_ranks.size(), common);
  std::vector<const EdgeFlux*> source_fluxes;
  std::vector<EdgeFlux*> target_fluxes;
  source_fluxes.reserve(source_rank_states.size());
  target_fluxes.reserve(result.size());

  for (const auto& [name, common_ring] : common.ring_flux)
    for (std::size_t slot = 0; slot < common_ring.size(); ++slot)
      for (std::size_t level = 0; level < common_ring[slot].size(); ++level) {
        source_fluxes.clear();
        target_fluxes.clear();
        for (const AmrProgramAcceptedState& state : source_rank_states)
          source_fluxes.push_back(&state.ring_flux.at(name)[slot][level]);
        for (AmrProgramAcceptedState& state : result)
          target_fluxes.push_back(&state.ring_flux.at(name)[slot][level]);
        rematerialize_edge_flux(source_fluxes, target_fluxes, level, source_ownership,
                                target_ownership, target_owner_to_slot,
                                "history '" + name + "' slot " + std::to_string(slot) + " level " +
                                    std::to_string(level));
      }

  for (const auto& [name, common_ring] : common.ring_flux_contributions)
    for (std::size_t slot = 0; slot < common_ring.size(); ++slot)
      for (std::size_t level = 0; level < common_ring[slot].size(); ++level)
        for (std::size_t contribution = 0; contribution < common_ring[slot][level].size();
             ++contribution) {
          source_fluxes.clear();
          target_fluxes.clear();
          for (const AmrProgramAcceptedState& state : source_rank_states)
            source_fluxes.push_back(
                &state.ring_flux_contributions.at(name)[slot][level][contribution].payload);
          for (AmrProgramAcceptedState& state : result)
            target_fluxes.push_back(
                &state.ring_flux_contributions.at(name)[slot][level][contribution].payload);
          rematerialize_edge_flux(source_fluxes, target_fluxes, level, source_ownership,
                                  target_ownership, target_owner_to_slot,
                                  "history '" + name + "' slot " + std::to_string(slot) +
                                      " level " + std::to_string(level) + " contribution " +
                                      std::to_string(contribution));
        }
  return result;
}

}  // namespace rematerialization_detail

/// Merge rank-local accepted Program images and materialize one exact image per target rank.
///
/// Every source state must carry byte-identical clocks, qualified history metadata, contribution
/// metadata and accepted audit reports.  Only the compact EdgeFlux payloads may differ by rank.
/// Active strips are accepted exclusively from their explicit recorded owner and are copied only to
/// their explicit target owner.  Missing strips remain missing (cold/flat histories); duplicates,
/// non-owner payloads and topology-axis mismatches fail closed.  Element `r` of the source vector is
/// the image recorded by source rank `r`.
inline std::vector<AmrProgramAcceptedState> rematerialize_amr_program_accepted_states(
    const std::vector<AmrProgramAcceptedState>& source_rank_states,
    const AmrProgramRankOwnership& source_ownership,
    const AmrProgramRankOwnership& target_ownership) {
  if (target_ownership.rank_count <= 0)
    rematerialization_detail::fail("target rank count must be positive");
  std::vector<int> target_ranks(static_cast<std::size_t>(target_ownership.rank_count));
  for (int rank = 0; rank < target_ownership.rank_count; ++rank)
    target_ranks[static_cast<std::size_t>(rank)] = rank;
  return rematerialization_detail::rematerialize_selected_target_ranks(
      source_rank_states, source_ownership, target_ownership, target_ranks);
}

/// Byte-protocol convenience seam for a single target rank.
///
/// Native facades keep accepted Program state opaque, so restart orchestration can use this wrapper
/// without duplicating either the checkpoint decoder or the ownership proof.  The selected rank is
/// explicit and range-checked.  All source images and target ownership are validated together, while
/// only that rank's filtered image is materialized.
inline std::vector<std::uint8_t> rematerialize_amr_program_accepted_state_bytes(
    const std::vector<std::vector<std::uint8_t>>& source_rank_payloads,
    const AmrProgramRankOwnership& source_ownership,
    const AmrProgramRankOwnership& target_ownership, int target_rank) {
  std::vector<AmrProgramAcceptedState> source_states;
  source_states.reserve(source_rank_payloads.size());
  for (const std::vector<std::uint8_t>& payload : source_rank_payloads)
    source_states.push_back(deserialize_amr_program_accepted_state(payload));
  std::vector<AmrProgramAcceptedState> target_states =
      rematerialization_detail::rematerialize_selected_target_ranks(
          source_states, source_ownership, target_ownership, {target_rank});
  return serialize_amr_program_accepted_state(target_states.front());
}

}  // namespace pops::runtime::program
