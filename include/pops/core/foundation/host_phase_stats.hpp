/// @file
/// @brief Image-local counters of the array bytes that cross between the target memory space and
///        the host (spec 0.3 DM8 and PF2).
///
/// Two kinds of crossings are counted:
///   - mesh arrays: every whole-Fab host mirror copy (`Fab::copy_to_host`, `Fab::copy_from_host`).
///     On a host build the mirror aliases the storage and nothing moves, but the count still marks
///     a host access that a device build turns into a copy, so a CPU run predicts what a GPU run
///     moves. No step may move one (DM8).
///   - communication staging: the halo bytes that a halo transport copies through pinned host
///     buffers because its route is `host_staged` (MPI not device-aware, or GPU isolation that
///     forbids device IPC). This is the admissible host phase of PF2, reported with its bytes.
///
/// The staging counters cover the halo exchange (`HaloExchange`) and the AMR parent ghost
/// transport only. The other transports that stage through pinned host memory (`RegionTransport`,
/// `TranslationExchange`, the traces of the multiblock interface, `comm_allocator` buffers) are
/// seen only by their Kokkos label "pops_host_staging:"; these counters are not the total staging.
///
/// Like `allocation_event_stats`, each independently loaded hidden-visibility image owns its own
/// counters; a probe that spans images reads the Kokkos Tools labels instead
/// ("pops_mesh_array:", "pops_host_staging:", and "pops_host_phase:" for host buffers that carry
/// host work and are no admissible phase of PF2).

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace pops {

struct HostPhaseStats {
  std::uint64_t mesh_to_host_calls = 0;
  std::uint64_t mesh_to_host_bytes = 0;
  std::uint64_t mesh_from_host_calls = 0;
  std::uint64_t mesh_from_host_bytes = 0;
  std::uint64_t staging_to_host_calls = 0;
  std::uint64_t staging_to_host_bytes = 0;
  std::uint64_t staging_from_host_calls = 0;
  std::uint64_t staging_from_host_bytes = 0;

  friend bool operator==(const HostPhaseStats&, const HostPhaseStats&) = default;

  /// Counts made after `before` (counters only grow).
  [[nodiscard]] HostPhaseStats since(const HostPhaseStats& before) const noexcept {
    return {mesh_to_host_calls - before.mesh_to_host_calls,
            mesh_to_host_bytes - before.mesh_to_host_bytes,
            mesh_from_host_calls - before.mesh_from_host_calls,
            mesh_from_host_bytes - before.mesh_from_host_bytes,
            staging_to_host_calls - before.staging_to_host_calls,
            staging_to_host_bytes - before.staging_to_host_bytes,
            staging_from_host_calls - before.staging_from_host_calls,
            staging_from_host_bytes - before.staging_from_host_bytes};
  }
};

namespace detail {

enum class HostPhaseCounter : unsigned char {
  mesh_to_host_calls,
  mesh_to_host_bytes,
  mesh_from_host_calls,
  mesh_from_host_bytes,
  staging_to_host_calls,
  staging_to_host_bytes,
  staging_from_host_calls,
  staging_from_host_bytes,
  count,
};

inline std::atomic<std::uint64_t>& host_phase_counter(HostPhaseCounter counter) noexcept {
  static std::atomic<std::uint64_t> values[static_cast<int>(HostPhaseCounter::count)]{};
  return values[static_cast<int>(counter)];
}

inline void add_host_phase(HostPhaseCounter calls, HostPhaseCounter bytes,
                           std::size_t amount) noexcept {
  host_phase_counter(calls).fetch_add(1, std::memory_order_relaxed);
  host_phase_counter(bytes).fetch_add(static_cast<std::uint64_t>(amount),
                                      std::memory_order_relaxed);
}

inline void record_mesh_to_host(std::size_t bytes) noexcept {
  add_host_phase(HostPhaseCounter::mesh_to_host_calls, HostPhaseCounter::mesh_to_host_bytes, bytes);
}
inline void record_mesh_from_host(std::size_t bytes) noexcept {
  add_host_phase(HostPhaseCounter::mesh_from_host_calls, HostPhaseCounter::mesh_from_host_bytes,
                 bytes);
}
inline void record_staging_to_host(std::size_t bytes) noexcept {
  add_host_phase(HostPhaseCounter::staging_to_host_calls, HostPhaseCounter::staging_to_host_bytes,
                 bytes);
}
inline void record_staging_from_host(std::size_t bytes) noexcept {
  add_host_phase(HostPhaseCounter::staging_from_host_calls,
                 HostPhaseCounter::staging_from_host_bytes, bytes);
}

}  // namespace detail

inline HostPhaseStats host_phase_stats() noexcept {
  using detail::HostPhaseCounter;
  const auto read = [](HostPhaseCounter counter) {
    return detail::host_phase_counter(counter).load(std::memory_order_relaxed);
  };
  return {read(HostPhaseCounter::mesh_to_host_calls),
          read(HostPhaseCounter::mesh_to_host_bytes),
          read(HostPhaseCounter::mesh_from_host_calls),
          read(HostPhaseCounter::mesh_from_host_bytes),
          read(HostPhaseCounter::staging_to_host_calls),
          read(HostPhaseCounter::staging_to_host_bytes),
          read(HostPhaseCounter::staging_from_host_calls),
          read(HostPhaseCounter::staging_from_host_bytes)};
}

}  // namespace pops
