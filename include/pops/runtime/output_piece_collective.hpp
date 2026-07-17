#pragma once

/// @file
/// @brief Native MPI root gather for scientific-output pieces.
///
/// Local providers are evaluated on every rank under an all-rank error consensus.  Metadata and
/// IEEE-754 values are framed in a versioned, endian-stable native wire payload and transferred by
/// WorldCommunicator's chunked MPI_Gatherv transport.  Only rank zero materializes the global piece
/// vector; Python never gathers NumPy arrays or executes an MPI collective.

#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/output_piece.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace pops {

namespace detail {

inline constexpr std::string_view kOutputPieceWireMagic = "POPSOP01";

inline void output_wire_append_u64(std::string& payload, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    payload.push_back(static_cast<char>((value >> shift) & 0xffU));
}

inline void output_wire_append_i64(std::string& payload, std::int64_t value) {
  output_wire_append_u64(payload, std::bit_cast<std::uint64_t>(value));
}

inline std::uint64_t output_wire_read_u64(std::string_view payload, std::size_t& offset,
                                          std::string_view field) {
  if (payload.size() - std::min(payload.size(), offset) < sizeof(std::uint64_t))
    throw std::runtime_error("truncated native output-piece field: " + std::string(field));
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << shift;
  return value;
}

inline std::int64_t output_wire_read_i64(std::string_view payload, std::size_t& offset,
                                         std::string_view field) {
  return std::bit_cast<std::int64_t>(output_wire_read_u64(payload, offset, field));
}

inline int output_wire_int(std::int64_t value, std::string_view field) {
  if (value < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("native output-piece field exceeds int: " + std::string(field));
  return static_cast<int>(value);
}

inline std::size_t output_piece_value_count(const OutputPiece& piece) {
  if (piece.box.level < 0 || piece.box.ihi < piece.box.ilo || piece.box.jhi < piece.box.jlo)
    throw std::runtime_error("native output piece has invalid bounds");
  if (piece.global_box_index < 0 || piece.owner_rank < 0 || piece.ncomp < 1)
    throw std::runtime_error("native output piece has invalid ownership/components");
  const std::uint64_t nx_u64 = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(piece.box.ihi) - static_cast<std::int64_t>(piece.box.ilo) + 1);
  const std::uint64_t ny_u64 = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(piece.box.jhi) - static_cast<std::int64_t>(piece.box.jlo) + 1);
  if (nx_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      ny_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("native output piece extent exceeds size_t");
  const std::size_t nx = static_cast<std::size_t>(nx_u64);
  const std::size_t ny = static_cast<std::size_t>(ny_u64);
  const std::size_t ncomp = static_cast<std::size_t>(piece.ncomp);
  if (ny > std::numeric_limits<std::size_t>::max() / nx ||
      ncomp > std::numeric_limits<std::size_t>::max() / (ny * nx))
    throw std::overflow_error("native output piece compact shape overflows size_t");
  return ncomp * ny * nx;
}

inline std::string serialize_output_pieces(const std::vector<OutputPiece>& pieces) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559,
                "output-piece wire requires IEEE-754 binary64");
  std::string payload(kOutputPieceWireMagic);
  output_wire_append_u64(payload, static_cast<std::uint64_t>(pieces.size()));
  for (const OutputPiece& piece : pieces) {
    const std::size_t expected = output_piece_value_count(piece);
    if (piece.values.size() != expected)
      throw std::runtime_error("native output piece has an inconsistent compact shape");
    output_wire_append_i64(payload, piece.box.level);
    output_wire_append_i64(payload, piece.box.ilo);
    output_wire_append_i64(payload, piece.box.jlo);
    output_wire_append_i64(payload, piece.box.ihi);
    output_wire_append_i64(payload, piece.box.jhi);
    output_wire_append_i64(payload, piece.global_box_index);
    output_wire_append_i64(payload, piece.owner_rank);
    output_wire_append_u64(payload, piece.replicated ? 1U : 0U);
    output_wire_append_i64(payload, piece.ncomp);
    output_wire_append_u64(payload, static_cast<std::uint64_t>(piece.values.size()));
    for (double value : piece.values)
      output_wire_append_u64(payload, std::bit_cast<std::uint64_t>(value));
  }
  return payload;
}

inline std::vector<OutputPiece> deserialize_output_pieces(std::string_view payload,
                                                          int source_rank) {
  if (!payload.starts_with(kOutputPieceWireMagic))
    throw std::runtime_error("native output-piece payload has an invalid wire identity");
  std::size_t offset = kOutputPieceWireMagic.size();
  const std::uint64_t count = output_wire_read_u64(payload, offset, "piece_count");
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("native output-piece count exceeds size_t");
  std::vector<OutputPiece> result;
  result.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    OutputPiece piece;
    piece.box.level =
        output_wire_int(output_wire_read_i64(payload, offset, "box.level"), "box.level");
    piece.box.ilo = output_wire_int(output_wire_read_i64(payload, offset, "box.ilo"), "box.ilo");
    piece.box.jlo = output_wire_int(output_wire_read_i64(payload, offset, "box.jlo"), "box.jlo");
    piece.box.ihi = output_wire_int(output_wire_read_i64(payload, offset, "box.ihi"), "box.ihi");
    piece.box.jhi = output_wire_int(output_wire_read_i64(payload, offset, "box.jhi"), "box.jhi");
    piece.global_box_index = output_wire_int(
        output_wire_read_i64(payload, offset, "global_box_index"), "global_box_index");
    piece.owner_rank =
        output_wire_int(output_wire_read_i64(payload, offset, "owner_rank"), "owner_rank");
    const std::uint64_t replicated = output_wire_read_u64(payload, offset, "replicated");
    if (replicated > 1U)
      throw std::runtime_error("native output-piece replicated flag is not boolean");
    piece.replicated = replicated == 1U;
    piece.ncomp = output_wire_int(output_wire_read_i64(payload, offset, "ncomp"), "ncomp");
    const std::uint64_t values_count = output_wire_read_u64(payload, offset, "values_count");
    if (values_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("native output-piece values exceed size_t");
    const std::size_t values_size = static_cast<std::size_t>(values_count);
    if (values_size > (payload.size() - std::min(payload.size(), offset)) / sizeof(std::uint64_t))
      throw std::runtime_error("truncated native output-piece values");
    piece.values.resize(values_size);
    for (double& value : piece.values)
      value = std::bit_cast<double>(output_wire_read_u64(payload, offset, "value"));
    if (piece.owner_rank != source_rank)
      throw std::runtime_error("native output-piece owner differs from its source MPI rank");
    if (piece.values.size() != output_piece_value_count(piece))
      throw std::runtime_error("native output-piece wire shape is inconsistent");
    result.push_back(std::move(piece));
  }
  if (offset != payload.size())
    throw std::runtime_error("native output-piece payload has trailing bytes");
  return result;
}

inline std::string output_collective_identity(std::string_view engine, std::string_view family,
                                              std::string_view selector, int level) {
  std::string result("pops.output-piece-root.v1");
  const auto append_text = [&result](std::string_view value) {
    output_wire_append_u64(result, static_cast<std::uint64_t>(value.size()));
    result.append(value);
  };
  append_text(engine);
  append_text(family);
  append_text(selector);
  output_wire_append_i64(result, level);
  return result;
}

inline std::string current_exception_text() {
  try {
    throw;
  } catch (const std::exception& error) {
    return std::string(typeid(error).name()) + ": " + error.what();
  } catch (...) {
    return "unknown native exception";
  }
}

}  // namespace detail

/// Evaluate a local OutputPiece provider and gather its exact result onto MPI rank zero.
template <typename Provider>
std::vector<OutputPiece> output_pieces_to_root(const WorldCommunicator& world,
                                               std::string operation_identity,
                                               Provider&& provider) {
  world.require_active_mpi_world();
  const int rank = world.rank();

  const std::vector<std::string> operations = world.allgather_bytes(operation_identity);
  if (!std::all_of(operations.begin(), operations.end(),
                   [&](const std::string& value) { return value == operation_identity; }))
    throw std::invalid_argument("output-piece root gather arguments differ across MPI ranks");

  std::vector<OutputPiece> local;
  std::string packed;
  std::string local_error;
  try {
    local = std::forward<Provider>(provider)();
    // Replicated AMR coarse boxes have one canonical root contributor in ROOT mode.
    local.erase(
        std::remove_if(local.begin(), local.end(),
                       [&](const OutputPiece& piece) { return piece.replicated && rank != 0; }),
        local.end());
    for (const OutputPiece& piece : local) {
      if (piece.owner_rank != rank)
        throw std::runtime_error("rank-local output piece is owned by another MPI rank");
    }
    packed = detail::serialize_output_pieces(local);
  } catch (...) {
    local_error = detail::current_exception_text();
  }

  const std::vector<std::string> errors = world.allgather_bytes(local_error);
  for (std::size_t source = 0; source < errors.size(); ++source) {
    if (!errors[source].empty())
      throw std::runtime_error("native output-piece provider failed on rank " +
                               std::to_string(source) + ": " + errors[source]);
  }

  const std::optional<std::vector<std::string>> gathered = world.gather_bytes(packed, 0);
  std::vector<OutputPiece> result;
  std::string root_error;
  if (rank == 0) {
    try {
      if (!gathered || gathered->size() != static_cast<std::size_t>(world.size()))
        throw std::runtime_error("native output-piece root gather has invalid rank cardinality");
      for (std::size_t source = 0; source < gathered->size(); ++source) {
        std::vector<OutputPiece> decoded =
            detail::deserialize_output_pieces((*gathered)[source], static_cast<int>(source));
        result.insert(result.end(), std::make_move_iterator(decoded.begin()),
                      std::make_move_iterator(decoded.end()));
      }
      std::sort(result.begin(), result.end(),
                [](const OutputPiece& left, const OutputPiece& right) {
                  return left.global_box_index < right.global_box_index;
                });
      if (std::adjacent_find(result.begin(), result.end(),
                             [](const OutputPiece& left, const OutputPiece& right) {
                               return left.global_box_index == right.global_box_index;
                             }) != result.end())
        throw std::runtime_error("native output-piece root gather contains duplicate boxes");
    } catch (...) {
      root_error = detail::current_exception_text();
    }
  }
  root_error = world.broadcast_bytes(std::move(root_error), 0);
  if (!root_error.empty())
    throw std::runtime_error("native output-piece reconstruction failed: " + root_error);
  return result;
}

}  // namespace pops
