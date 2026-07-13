#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/runtime/amr/amr_program_reflux.hpp>

namespace pops::runtime::program {

namespace amr = ::pops::amr;

/// Complete accepted state owned by the compiled AMR Program context.  Engine-owned history values
/// remain in the regular checkpoint arrays; this image carries the semantic clock/identity authority
/// and the lagged effective-flux strips which cannot be reconstructed from state buffers alone.
struct AmrProgramAcceptedState {
  std::vector<amr::ClockStamp> level_clocks;
  std::map<std::string, int> history_owners;
  std::map<std::string, std::string> history_states;
  std::map<std::string, std::string> history_spaces;
  std::map<std::string, std::vector<std::vector<amr::ClockStamp>>> ring_clocks;
  std::map<std::string,
           std::vector<std::vector<std::optional<amr::HistoryIdentity>>>>
      ring_identities;
  std::map<std::string, std::vector<std::vector<EdgeFlux>>> ring_flux;
  std::map<std::string, std::vector<char>> ring_flux_initialized;
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

inline void write_reals(Writer& out, const std::vector<Real>& values) {
  out.size(values.size());
  for (Real value : values)
    out.real(static_cast<double>(value));
}

inline std::vector<Real> read_reals(Reader& in) {
  std::vector<Real> values(in.size());
  for (Real& value : values)
    value = static_cast<Real>(in.real());
  return values;
}

inline void write_strip(Writer& out, const EdgeStrip& value) {
  out.i32(value.I0); out.i32(value.I1); out.i32(value.J0); out.i32(value.J1);
  write_reals(out, value.cL); write_reals(out, value.cR);
  write_reals(out, value.cB); write_reals(out, value.cT);
  write_reals(out, value.fL); write_reals(out, value.fR);
  write_reals(out, value.fB); write_reals(out, value.fT);
}

inline EdgeStrip read_strip(Reader& in) {
  EdgeStrip value;
  value.I0 = in.i32(); value.I1 = in.i32(); value.J0 = in.i32(); value.J1 = in.i32();
  value.cL = read_reals(in); value.cR = read_reals(in);
  value.cB = read_reals(in); value.cT = read_reals(in);
  value.fL = read_reals(in); value.fR = read_reals(in);
  value.fB = read_reals(in); value.fT = read_reals(in);
  return value;
}

inline void write_flux(Writer& out, const EdgeFlux& value) {
  out.size(value.coarse.size());
  for (const EdgeStrip& strip : value.coarse) write_strip(out, strip);
  out.size(value.fine.size());
  for (const EdgeStrip& strip : value.fine) write_strip(out, strip);
}

inline EdgeFlux read_flux(Reader& in) {
  EdgeFlux value;
  value.coarse.resize(in.size());
  for (EdgeStrip& strip : value.coarse) strip = read_strip(in);
  value.fine.resize(in.size());
  for (EdgeStrip& strip : value.fine) strip = read_strip(in);
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
  Writer out;
  out.u64(0x3154534153504f50ULL);  // "POPSAST1", little-endian bytes
  out.size(state.level_clocks.size());
  for (const auto& clock : state.level_clocks) write_clock(out, clock);
  write_map(out, state.history_owners, [](Writer& w, int v) { w.i32(v); });
  write_map(out, state.history_states, [](Writer& w, const std::string& v) { w.string(v); });
  write_map(out, state.history_spaces, [](Writer& w, const std::string& v) { w.string(v); });
  write_map(out, state.ring_clocks, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const auto& clock : slot) write_clock(w, clock);
    }
  });
  write_map(out, state.ring_identities, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const auto& identity : slot) {
        w.u64(identity ? 1 : 0);
        if (identity) write_identity(w, *identity);
      }
    }
  });
  write_map(out, state.ring_flux, [](Writer& w, const auto& ring) {
    w.size(ring.size());
    for (const auto& slot : ring) {
      w.size(slot.size());
      for (const EdgeFlux& flux : slot) write_flux(w, flux);
    }
  });
  write_map(out, state.ring_flux_initialized, [](Writer& w, const auto& values) {
    w.size(values.size());
    for (char value : values) w.u64(value ? 1 : 0);
  });
  return out.take();
}

inline AmrProgramAcceptedState deserialize_amr_program_accepted_state(
    const std::vector<std::uint8_t>& bytes) {
  using namespace checkpoint_detail;
  Reader in(bytes);
  if (in.u64() != 0x3154534153504f50ULL)
    throw std::runtime_error("invalid AMR Program accepted-state payload: unsupported magic/version");
  AmrProgramAcceptedState state;
  state.level_clocks.resize(in.size());
  for (auto& clock : state.level_clocks) clock = read_clock(in);
  state.history_owners = read_map<std::map<std::string, int>>(in, [](Reader& r) { return r.i32(); });
  state.history_states = read_map<std::map<std::string, std::string>>(
      in, [](Reader& r) { return r.string(); });
  state.history_spaces = read_map<std::map<std::string, std::string>>(
      in, [](Reader& r) { return r.string(); });
  state.ring_clocks = read_map<decltype(state.ring_clocks)>(in, [](Reader& r) {
    std::vector<std::vector<amr::ClockStamp>> ring(r.size());
    for (auto& slot : ring) {
      slot.resize(r.size());
      for (auto& clock : slot) clock = read_clock(r);
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
          throw std::runtime_error("invalid AMR Program accepted-state payload: invalid optional flag");
        if (present) identity = read_identity(r);
      }
    }
    return ring;
  });
  state.ring_flux = read_map<decltype(state.ring_flux)>(in, [](Reader& r) {
    std::vector<std::vector<EdgeFlux>> ring(r.size());
    for (auto& slot : ring) {
      slot.resize(r.size());
      for (auto& flux : slot) flux = read_flux(r);
    }
    return ring;
  });
  state.ring_flux_initialized = read_map<decltype(state.ring_flux_initialized)>(
      in, [](Reader& r) {
        std::vector<char> values(r.size());
        for (char& value : values) {
          const std::uint64_t flag = r.u64();
          if (flag > 1)
            throw std::runtime_error("invalid AMR Program accepted-state payload: invalid flag");
          value = flag ? 1 : 0;
        }
        return values;
      });
  in.finish();
  return state;
}

}  // namespace pops::runtime::program
