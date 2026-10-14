#pragma once

#include <pops/runtime/program/continuation_image.hpp>
#include <pops/runtime/program/continuation_closure.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/parallel/collective_exception.hpp>

#include <exception>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace pops::runtime::program {

// Diagnostic resource policy, independent of physical equations. Exceeding either bound is an
// explicit fatal transport failure: never silently discard or truncate a typed error value.
struct TypedFailureCapacity {
  std::uint64_t per_rank = 64ULL * 1024 * 1024;
  std::uint64_t aggregate = 256ULL * 1024 * 1024;
};

namespace typed_failure_detail {
inline constexpr std::string_view magic = "POPSFAIL1";
inline void word(std::string& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & 255));
}
inline std::uint64_t word(std::string_view bytes) {
  if (bytes.size() != 8) throw std::invalid_argument("invalid typed failure size frame");
  std::uint64_t result = 0;
  for (unsigned char byte : bytes) result = (result << 8) | byte;
  return result;
}
// ExecutionLane has no allocating allgather member. Use its exact communicator and an
// ordered sequence of bounded broadcasts, like ObserverMpiLane, without selecting the world.
// Each allocation and capacity check is a local phase converged before the next collective.
inline std::vector<std::string> allgather_bytes(const ExecutionLane& lane,
                                                const std::string& local,
                                                TypedFailureCapacity capacity) {
  const auto communicator = lane.communicator();
  const int ranks = communicator.size(), me = communicator.rank();
  std::vector<std::string> result;
  std::exception_ptr error;
  try {
    if (ranks <= 0 || local.size() > capacity.per_rank ||
        local.size() > static_cast<std::size_t>(std::numeric_limits<long>::max()))
      throw std::length_error("typed failure byte shard exceeds native transport capacity");
    result.resize(static_cast<std::size_t>(ranks));
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, communicator, "typed failure gather preparation failed");
  std::uint64_t total = 0;
  for (int root = 0; root < ranks; ++root) {
    const long length = all_reduce_max(me == root ? static_cast<long>(local.size()) : 0L,
                                       communicator);
    error = {};
    try {
      const auto size = static_cast<std::uint64_t>(length);
      if (length < 0 || size > capacity.per_rank || total > capacity.aggregate ||
          size > capacity.aggregate - total)
        throw std::length_error("typed failure gathered bytes exceed diagnostic capacity");
      total += size;
      auto& destination = result[static_cast<std::size_t>(root)];
      if (me == root) destination = local;
      else destination.resize(static_cast<std::size_t>(size));
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, communicator, "typed failure gather allocation failed");
    auto& destination = result[static_cast<std::size_t>(root)];
    broadcast_bytes_inplace(destination.data(), destination.size(), root, communicator);
  }
  return result;
}

template<int Dim>
void describe(const ContinuationType& type, const ContinuationValueRef<Dim>& value,
               std::string path, std::string& text, unsigned& remaining) {
  if (!remaining) return;
  using K = ContinuationKind;
  if (type.kind == K::Product) {
    for (std::size_t n = 0; n < type.children.size(); ++n)
      describe(type.children[n], value->members().at(n), path + type.names[n] + ".", text, remaining);
  } else if (type.kind == K::Variant) {
    const auto n = value->arm();
    describe(type.children.at(n), value->members().at(0), path + type.names.at(n) + ".", text, remaining);
  } else {
    --remaining;
    if (!path.empty()) path.pop_back();
    if (path.size() > 160) path = path.substr(0, 160) + "...";
    text += "; " + path + "=";
    if (type.kind == K::Int64) text += std::to_string(value->integer());
    else if (type.kind == K::Float64) text += std::to_string(value->real());
    else if (type.kind == K::Boolean) text += value->boolean() ? "true" : "false";
    else text += "<owned typed payload>";
  }
}
}  // namespace typed_failure_detail

// Preparation is a generated, rank-local expression: evaluate the closed structural value,
// copy any field shards and converge, close distributed dependencies, then encode each shard.
// All ranks receive the same ordered aggregate. No field shard is replaced with rank-zero data.
template<int Dim, class Prepare, class Prototypes>
[[noreturn]] void reject_typed_failure(const ExecutionLane& lane, Prepare&& prepare,
                                        Prototypes&& prototypes, TypedFailureCapacity capacity = {},
                                        distributed::Registry* registry = nullptr) {
  std::string local, schema, detail;
  ContinuationType type; ContinuationValueRef<Dim> value;
  std::exception_ptr error;
  try {
    auto [prepared_type, prepared_value, scientific_schema] = prepare();
    type=std::move(prepared_type); value=std::move(prepared_value); schema=std::move(scientific_schema);
  } catch (...) { error = std::current_exception(); }
  try { Kokkos::fence(); } catch (...) { if (!error) error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "typed failure value preparation failed");

  // This closure is a read-only authenticated snapshot of all reachable versions.
  // It includes every rank's shard, even if only one rank carries a reference.
  // It never adopts a restored registry or grants a continuation publication.
  const auto closure=continuation_collective_closure<Dim>(value,registry,lane,capacity.per_rank);
  error={};
  try {
    ContinuationCodec<Dim> codec(type, schema, capacity.per_rank,
        std::forward<Prototypes>(prototypes), &lane, registry, closure.get());
    auto bytes = codec.encode(value, 0);
    local.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    detail = "unhandled typed Result failure";
    unsigned remaining = 8;
    typed_failure_detail::describe(type, value, "", detail, remaining);
  } catch (...) { error = std::current_exception(); }
  try { Kokkos::fence(); } catch (...) { if (!error) error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "typed failure local preparation failed");

  std::string size_frame;
  error = {};
  try { typed_failure_detail::word(size_frame, local.size()); }
  catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "typed failure size framing failed");
  const auto sizes = typed_failure_detail::allgather_bytes(lane, size_frame, capacity);
  error = {};
  try {
    // Fixed magic plus rank count and one length word per rank; all additions are bounded.
    std::uint64_t total = typed_failure_detail::magic.size() + 8;
    for (const auto& frame : sizes) {
      const auto size = typed_failure_detail::word(frame);
      if (total > capacity.aggregate || capacity.aggregate - total < 8 ||
          size > capacity.aggregate - total - 8)
        throw std::length_error("typed failure exceeds aggregate diagnostic capacity");
      total += 8 + size;
    }
    if (total > std::numeric_limits<std::size_t>::max())
      throw std::length_error("typed failure exceeds local address capacity");
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "typed failure aggregate capacity failed");

  const auto schemas = typed_failure_detail::allgather_bytes(lane, schema, capacity);
  const auto shards = typed_failure_detail::allgather_bytes(lane, local, capacity);
  const auto details = typed_failure_detail::allgather_bytes(lane, detail, capacity);
  std::string aggregate;
  std::exception_ptr rejected;
  error = {};
  try {
    for (const auto& other : schemas)
      if (other != schema) throw std::invalid_argument("typed failure schemas differ between ranks");
    aggregate.assign(typed_failure_detail::magic.data(), typed_failure_detail::magic.size());
    typed_failure_detail::word(aggregate, shards.size());
    for (const auto& shard : shards) {
      typed_failure_detail::word(aggregate, shard.size()); aggregate += shard;
    }
    // The diagnostic is abbreviated; the canonical payload retains every rank in full.
    throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, StepAttemptDisposition::kReject,
                              0, "structured_control", details.front(), std::move(schema),
                              std::move(aggregate));
  } catch (const StepAttemptRejected&) { rejected = std::current_exception(); }
  catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "typed failure collective publication failed");
  std::rethrow_exception(rejected);
}

}  // namespace pops::runtime::program
