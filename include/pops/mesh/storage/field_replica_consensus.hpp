#pragma once

/// @file
/// @brief Exact native consensus for fields whose physical distribution is replicated.

#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>

namespace pops::detail {

/// Eight MiB keeps exact validation bounded while a 1024x1024 scalar field needs one canonical
/// comparison instead of 128 tiny chunks. Both buffers are persistent provider scratch.
inline constexpr std::size_t kFieldReplicaConsensusChunkBytes = 8u * 1024u * 1024u;

constexpr std::size_t field_replica_consensus_storage_size() noexcept {
  return 2u * kFieldReplicaConsensusChunkBytes;
}

template <class Value>
inline void append_exact_contract_value(std::string& bytes, const Value& value) {
  static_assert(std::is_trivially_copyable_v<Value>);
  bytes.append(reinterpret_cast<const char*>(&value), sizeof(Value));
}

/// Canonical structural identity of one field distribution. Distributed mappings preserve their
/// exact rank owners. Replicated mappings canonicalize every local owner to zero after the caller
/// has proved that every global box is materialized on the current rank.
inline std::string field_distribution_layout_contract(const MultiFab& field,
                                                      FieldDistribution distribution) {
  std::string bytes;
  const auto& boxes = field.box_array().boxes();
  const auto& ranks = field.dmap().ranks();
  bytes.reserve(3u * sizeof(int) + sizeof(std::uint64_t) + boxes.size() * (5u * sizeof(int)));
  append_exact_contract_value(bytes, static_cast<std::uint8_t>(distribution));
  append_exact_contract_value(bytes, field.ncomp());
  append_exact_contract_value(bytes, field.n_grow());
  append_exact_contract_value(bytes, static_cast<std::uint64_t>(boxes.size()));
  append_exact_contract_value(bytes, static_cast<std::uint64_t>(ranks.size()));
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const Box2D& box = boxes[index];
    append_exact_contract_value(bytes, box.lo[0]);
    append_exact_contract_value(bytes, box.lo[1]);
    append_exact_contract_value(bytes, box.hi[0]);
    append_exact_contract_value(bytes, box.hi[1]);
    const int owner = distribution == FieldDistribution::Replicated
                          ? 0
                          : (index < ranks.size() ? ranks[index] : -1);
    append_exact_contract_value(bytes, owner);
  }
  return bytes;
}

inline bool field_distribution_layout_matches(const MultiFab& field,
                                              FieldDistribution distribution) {
  const auto& boxes = field.box_array().boxes();
  const auto& ranks = field.dmap().ranks();
  if (field.ncomp() <= 0 || field.n_grow() < 0 || boxes.empty() || boxes.size() != ranks.size())
    return false;
  if (distribution == FieldDistribution::Distributed)
    return true;
  if (distribution != FieldDistribution::Replicated)
    return false;
  const int rank = my_rank();
  return std::all_of(ranks.begin(), ranks.end(), [rank](int owner) { return owner == rank; });
}

/// Authenticate an unauthenticated public descriptor before any scientific collective. This is
/// exact rather than hash-based, so a rank-local full replica mislabeled Distributed is rejected
/// instead of being silently counted once per rank.
inline void require_collective_field_distribution_layout(const MultiFab& field,
                                                         FieldDistribution distribution,
                                                         const char* where) {
  const long raw = static_cast<long>(distribution);
  const long minimum = all_reduce_min(raw);
  const long maximum = all_reduce_max(raw);
  const bool valid_distribution = field_distribution_is_valid(distribution);
  const long invalid_distribution = all_reduce_max(valid_distribution ? 0L : 1L);
  const bool valid_layout =
      valid_distribution && field_distribution_layout_matches(field, distribution);
  const long invalid_layout = all_reduce_max(valid_layout ? 0L : 1L);

  const FieldDistribution canonical_distribution =
      valid_distribution ? distribution : FieldDistribution::Distributed;
  std::string contract;
  long contract_failure_local = 0;
  try {
    contract = field_distribution_layout_contract(field, canonical_distribution);
  } catch (...) {
    contract_failure_local = 1;
  }
  const long contract_failure = all_reduce_max(contract_failure_local);
  if (contract_failure != 0)
    throw std::runtime_error(
        std::string(where) +
        ": field layout contract materialization failed on at least one communicator rank");
  const bool contract_agrees = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("field-distribution-layout"), std::string_view(contract)}});

  if (minimum != maximum)
    throw std::invalid_argument(std::string(where) +
                                ": field distribution differs between communicator ranks");
  if (invalid_distribution != 0)
    throw std::invalid_argument(std::string(where) + ": invalid field distribution");
  if (invalid_layout != 0)
    throw std::invalid_argument(std::string(where) +
                                ": field layout does not realize its declared distribution");
  if (!contract_agrees)
    throw std::invalid_argument(std::string(where) +
                                ": field layout differs between communicator ranks");
}

/// Compare every valid cell bit-for-bit in canonical global-box/component/j/i order. Kokkos packs
/// directly from the execution-space-visible field into persistent shared scratch; rank zero then
/// broadcasts the canonical bytes and every rank performs an exact device comparison. This avoids
/// a host scan and remains collision-free. The caller supplies two fixed-size chunks, so memory is
/// bounded independently of field size.
inline void require_exact_replicated_field_values_prevalidated(
    const MultiFab& field, char* storage, std::size_t storage_size, const char* where,
    const CommunicatorView& communicator) {
  if (communicator.size() == 1)
    return;
  const long invalid_storage = all_reduce_max(
      storage == nullptr || storage_size < field_replica_consensus_storage_size() ? 1L : 0L,
      communicator);
  if (invalid_storage != 0)
    throw std::logic_error(std::string(where) + ": replica consensus storage is incoherent");
  long missing_box = 0;
  for (int global = 0; global < field.box_array().size(); ++global)
    missing_box = std::max(missing_box, field.local_index_of(global) < 0 ? 1L : 0L);
  if (all_reduce_max(missing_box, communicator) != 0)
    throw std::logic_error(std::string(where) + ": prevalidated replica is missing a global box");

  static_assert(kFieldReplicaConsensusChunkBytes % sizeof(Real) == 0);
  constexpr std::size_t chunk_value_capacity = kFieldReplicaConsensusChunkBytes / sizeof(Real);
  char* local_bytes = storage;
  char* canonical_bytes = storage + kFieldReplicaConsensusChunkBytes;

  std::size_t total_values = 0;
  for (const Box2D& box : field.box_array().boxes()) {
    const std::size_t nx = static_cast<std::size_t>(box.nx());
    const std::size_t ny = static_cast<std::size_t>(box.ny());
    if (ny != 0 && nx > std::numeric_limits<std::size_t>::max() / ny)
      throw std::overflow_error(std::string(where) + ": replica size overflows size_t");
    const std::size_t cells = nx * ny;
    if (cells != 0 &&
        static_cast<std::size_t>(field.ncomp()) > std::numeric_limits<std::size_t>::max() / cells)
      throw std::overflow_error(std::string(where) + ": replica component size overflows size_t");
    const std::size_t box_values = cells * static_cast<std::size_t>(field.ncomp());
    if (box_values > std::numeric_limits<std::size_t>::max() - total_values)
      throw std::overflow_error(std::string(where) + ": replica size overflows size_t");
    total_values += box_values;
  }

  field.sync_device();
  for (std::size_t chunk_begin = 0; chunk_begin < total_values;
       chunk_begin += std::min(chunk_value_capacity, total_values - chunk_begin)) {
    const std::size_t chunk_values = std::min(chunk_value_capacity, total_values - chunk_begin);
    const std::size_t chunk_end = chunk_begin + chunk_values;
    std::size_t box_begin = 0;
    for (int global = 0; global < field.box_array().size(); ++global) {
      const int local = field.local_index_of(global);
      const Box2D& box = field.box(local);
      const std::size_t nx = static_cast<std::size_t>(box.nx());
      const std::size_t ny = static_cast<std::size_t>(box.ny());
      const std::size_t plane = nx * ny;
      const std::size_t box_values = plane * static_cast<std::size_t>(field.ncomp());
      const std::size_t box_end = box_begin + box_values;
      const std::size_t overlap_begin = std::max(chunk_begin, box_begin);
      const std::size_t overlap_end = std::min(chunk_end, box_end);
      if (overlap_begin < overlap_end) {
        const ConstArray4 values = field.fab(local).const_array();
        const std::size_t source_begin = overlap_begin - box_begin;
        const std::size_t destination_begin = overlap_begin - chunk_begin;
        const std::size_t count = overlap_end - overlap_begin;
        Kokkos::parallel_for(
            "pops_pack_exact_field_replica",
            Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, count),
            KOKKOS_LAMBDA(const std::size_t offset) {
              const std::size_t source = source_begin + offset;
              const int component = static_cast<int>(source / plane);
              const std::size_t cell = source - static_cast<std::size_t>(component) * plane;
              const int j = box.lo[1] + static_cast<int>(cell / nx);
              const int i = box.lo[0] + static_cast<int>(cell % nx);
              const std::uint64_t bits = Kokkos::bit_cast<std::uint64_t>(values(i, j, component));
              const std::size_t destination = (destination_begin + offset) * sizeof(Real);
              for (std::size_t byte = 0; byte < sizeof(Real); ++byte)
                local_bytes[destination + byte] =
                    static_cast<char>((bits >> (byte * 8u)) & std::uint64_t{0xff});
            });
      }
      box_begin = box_end;
    }
    device_fence();

    const std::size_t chunk_bytes = chunk_values * sizeof(Real);
    if (communicator.rank() == 0) {
      Kokkos::parallel_for(
          "pops_copy_canonical_field_replica",
          Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, chunk_bytes),
          KOKKOS_LAMBDA(const std::size_t index) { canonical_bytes[index] = local_bytes[index]; });
      device_fence();
    }
    broadcast_bytes_inplace(canonical_bytes, chunk_bytes, 0, communicator);
    // MPI writes the pinned host buffer outside the Kokkos execution space. Establish the
    // host-to-execution-space visibility boundary before the exact device comparison.
    device_fence();

    long local_mismatch = 0;
    Kokkos::parallel_reduce(
        "pops_compare_exact_field_replica",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, chunk_bytes),
        KOKKOS_LAMBDA(const std::size_t index, long& mismatch) {
          const long differs = local_bytes[index] != canonical_bytes[index] ? 1L : 0L;
          mismatch = mismatch < differs ? differs : mismatch;
        },
        Kokkos::Max<long>(local_mismatch));
    if (all_reduce_max(local_mismatch, communicator) != 0)
      throw std::runtime_error(std::string(where) +
                               ": replicated field values differ between communicator ranks");
  }
}

inline void require_exact_replicated_field_values_prevalidated(const MultiFab& field, char* storage,
                                                               std::size_t storage_size,
                                                               const char* where) {
  require_exact_replicated_field_values_prevalidated(field, storage, storage_size, where,
                                                     world_communicator_view());
}

inline void require_exact_field_replica(const MultiFab& field, FieldDistribution distribution,
                                        char* storage, std::size_t storage_size,
                                        const char* where) {
  require_collective_field_distribution_layout(field, distribution, where);
  if (distribution == FieldDistribution::Replicated)
    require_exact_replicated_field_values_prevalidated(field, storage, storage_size, where);
}

}  // namespace pops::detail
