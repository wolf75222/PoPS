#pragma once
// Bounded contiguous-domain execution with explicit collective publication boundaries.
#include <pops/runtime/program/distributed_domain.hpp>
#include <pops/runtime/program/continuation_image.hpp>
#include <pops/runtime/program/distributed_gc.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::runtime::program::distributed {
// Kept only inside one local region; the generated capture supplies the static
// error-union injection. It never crosses an MPI phase as a C++ exception.
template <int Dim>
struct LocalPayloadFailure {
  ContinuationValueRef<Dim> payload;
  std::string schema;
};
template <int Dim>
struct CollectivePayloadFailure {
  ContinuationValueRef<Dim> payload;
};
template <int Dim>
Bytes encode_payload(const ContinuationType& type, ContinuationValueRef<Dim> value,
                     std::uint64_t capacity, ContinuationTransportSession<Dim>* session) {
  return session ? encode_continuation_transport<Dim>(type, std::move(value), capacity, *session)
                 : encode_continuation_transport<Dim>(type, std::move(value), capacity);
}
template <int Dim>
ContinuationValueRef<Dim> decode_payload(const ContinuationType& type,
                                         std::span<const std::uint8_t> bytes,
                                         std::uint64_t capacity,
                                         ContinuationTransportSession<Dim>* session) {
  return session ? decode_continuation_transport<Dim>(type, bytes, capacity, *session)
                 : decode_continuation_transport<Dim>(type, bytes, capacity);
}
template <int Dim>
using Array = ContinuationDistributedArray<Dim>;
template <int Dim>
using ArrayRef = std::shared_ptr<const Array<Dim>>;
template <int Dim>
ArrayRef<Dim> resolve_array(Registry& registry, Identity version, const DomainRef& domain,
                            const ContinuationType& element) {
  registry.authenticate(domain);
  auto value = std::static_pointer_cast<const Array<Dim>>(
      registry.resolve_value(version, array_type_tag<Dim>()));
  local_require(value && value->domain == domain && value->element == element &&
                    value->version.issuer == version.issuer &&
                    value->version.ordinal == version.ordinal &&
                    value->values.size() == std::size_t(domain->local_size()),
                Code::domain);
  return value;
}
template <int Dim>
ArrayRef<Dim> publish_array(Registry& registry, std::shared_ptr<Array<Dim>> pending,
                            std::uint64_t capacity, ContinuationTransportSession<Dim>* transport) {
  std::uint64_t charge = 0;
  phase(registry.lane(), [&] {
    local_require(pending && pending->domain && pending->version.ordinal == 0 &&
                      pending->values.size() == std::size_t(pending->domain->local_size()),
                  Code::domain);
    registry.authenticate(pending->domain);
  });
  if (pending->element.kind == ContinuationKind::DistributedIndex) {
    DomainRef witness = pending->index_domain;
    phase(registry.lane(), [&] {
      for (const auto& value : pending->values) {
        local_require(value && value->kind() == ContinuationKind::DistributedIndex, Code::index);
        const auto& domain = value->distributed_domain_ref();
        if (transport)
          transport->authenticate_distributed_domain(domain);
        else
          registry.authenticate(domain);
        if (!witness)
          witness = domain;
        local_require(witness == domain, Code::domain);
      }
    });
    const auto owner = common_failure_rank(bool(witness), registry.lane());
    if (owner != registry.lane().size()) {
      std::array<std::uint64_t, 3> words{};
      if (registry.lane().rank() == owner) {
        auto id = witness->identity();
        words = {id.issuer[0], id.issuer[1], id.ordinal};
      }
#ifdef POPS_HAS_MPI
      mpi_status(
          MPI_Bcast(words.data(), 3, MPI_UINT64_T, int(owner), registry.lane().native_handle()));
#endif
      phase(registry.lane(), [&] {
        const Identity id{{words[0], words[1]}, words[2]};
        auto common =
            transport ? transport->resolve_distributed_domain(id) : registry.resolve_existing(id);
        local_require((!witness || witness == common) &&
                          common->family() == pending->element.family &&
                          common->capacity() == pending->element.capacity,
                      Code::domain);
        pending->index_domain = std::move(common);
      });
    }
  }
  std::string type;
  phase(registry.lane(), [&] {
    type = array_type_tag<Dim>();
    // One canonical charge policy for ordinary publication and checkpoint restore,
    // independent of whether the caller needed a persistent transport session.
    charge = continuation_distributed_array_charge<Dim>(pending, registry, capacity, {}, transport);
  });
  pending->version = registry.publish_value(pending, std::move(type), charge);
  return std::move(pending);
}
template <int Dim>
ContinuationValueRef<Dim> at_local(const ArrayRef<Dim>& a, const Index& i) {
  local_require(a && a->domain && i.domain == a->domain && a->domain->owns(i.gid) &&
                    a->values.size() == std::size_t(a->domain->local_size()),
                Code::index, i.gid);
  return a->values[std::size_t(i.gid - a->domain->begin())];
}
template <int Dim>
struct Outcome {
  ArrayRef<Dim> value;
  std::int64_t rank = -1, gid = -1;
  Code code = Code::internal;
  ContinuationValueRef<Dim> payload;  // exact user E only when code==user
  bool ok() const noexcept { return bool(value); }
};
// Callback returns a declared two-arm Result: arm 0=err, arm 1=ok (canonical sorted tags).
// It MUST contain only local operations. The frontend and lowering enforce that finite effect set.
template <int Dim, class F>
Outcome<Dim> map(Registry& registry, DomainRef domain, const ContinuationType& element,
                 const ContinuationType& error, std::uint64_t byte_capacity, F&& body,
                 ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto& lane = registry.lane();
  Outcome<Dim> result;
  try {
    registry.common(domain);
    phase(lane, [&] {
      registry.authenticate(domain);
      local_require(valid_budget(byte_capacity), Code::capacity);
    });
    std::shared_ptr<Array<Dim>> pending;
    phase(lane, [&] {
      pending = std::make_shared<Array<Dim>>(Array<Dim>{domain, element, {}});
      pending->values.resize(std::size_t(domain->local_size()));
    });
    ContinuationValueRef<Dim> local_error;
    std::optional<Failure> local_fault;
    std::int64_t bad_gid = -1;
    // NO collective anywhere in this loop. Stop only the local loop on failure.
    for (std::int64_t gid = domain->begin(); gid < domain->end(); ++gid) {
      try {
        auto r = body(Index{domain, gid});
        local_require(
            r && r->kind() == ContinuationKind::Variant && r->members().size() == 1 && r->arm() < 2,
            Code::wire, gid);
        if (r->arm() == 0) {
          (void)encode_payload<Dim>(error, r->members()[0], byte_capacity, transport);
          local_error = r->members()[0];
          bad_gid = gid;
          break;
        }
        (void)encode_payload<Dim>(element, r->members()[0], byte_capacity, transport);
        pending->values[std::size_t(gid - domain->begin())] = r->members()[0];
      } catch (const Failure& f) {
        local_fault = f;
        bad_gid = gid;
        break;
      } catch (const std::bad_alloc&) {
        local_fault.emplace(Code::allocation);
        bad_gid = gid;
        break;
      } catch (...) {
        local_fault.emplace(Code::arithmetic);
        bad_gid = gid;
        break;
      }
    }
    const auto owner = common_failure_rank(bool(local_error) || bool(local_fault), lane);
    if (owner == lane.size()) {
      result.value = publish_array<Dim>(registry, std::move(pending), byte_capacity, transport);
      return result;
    }
    std::array<std::int64_t, 2> info{};
    if (lane.rank() == owner)
      info = {std::int64_t(local_fault ? local_fault->code : Code::user), bad_gid};
#ifdef POPS_HAS_MPI
    mpi_status(MPI_Bcast(info.data(), 2, MPI_INT64_T, int(owner), lane.native_handle()));
#endif
    result.code = Code(info[0]);
    result.rank = owner;
    result.gid = info[1];
    if (result.code == Code::user) {
      Bytes bytes;
      std::uint64_t length = 0;
      phase(lane, [&] {
        if (lane.rank() == owner) {
          bytes = encode_payload<Dim>(error, local_error, byte_capacity, transport);
          length = bytes.size();
        }
      });
#ifdef POPS_HAS_MPI
      mpi_status(MPI_Bcast(&length, 1, MPI_UINT64_T, int(owner), lane.native_handle()));
#endif
      phase(lane, [&] {
        local_require(length <= byte_capacity, Code::capacity);
        bytes.resize(byte_size(length));
      });
      broadcast(bytes, int(owner), lane);
      phase(lane,
            [&] { result.payload = decode_payload<Dim>(error, bytes, byte_capacity, transport); });
    }
    return result;  // pending dies without publication on all ranks
  } catch (const Failure& f) {
    result.code = f.code;
    result.rank = f.rank;
    result.gid = f.gid;
    return result;
  }
}
// Rank-indexed byte buckets. The common count exchange precedes all payload allocations.
inline std::vector<Bytes> exchange(const ExecutionLane& lane, const std::vector<Bytes>& send,
                                   std::uint64_t capacity, std::uint64_t chunk_limit = INT_MAX) {
  std::vector<std::uint64_t> sc, rc;
  std::vector<Bytes> result;
  phase(lane, [&] {
    local_require(valid_budget(capacity) && send.size() == std::size_t(lane.size()),
                  Code::capacity);
    (void)ByteChunks(0, chunk_limit);
    sc.resize(lane.size());
    rc.resize(lane.size());
    std::uint64_t total = 0;
    for (int r = 0; r < lane.size(); ++r) {
      local_require(send[r].size() <= capacity - total, Code::capacity);
      sc[r] = send[r].size();
      total += sc[r];
    }
  });
  agree_i64(std::int64_t(chunk_limit), lane);
#ifdef POPS_HAS_MPI
  mpi_status(
      MPI_Alltoall(sc.data(), 1, MPI_UINT64_T, rc.data(), 1, MPI_UINT64_T, lane.native_handle()));
#else
  rc = sc;
#endif
  // All destination storage is allocated before any payload moves. A rejected
  // phase cannot mutate a source bucket or publish a partially received value.
  phase(lane, [&] {
    std::uint64_t total = 0;
    result.resize(lane.size());
    for (int r = 0; r < lane.size(); ++r) {
      local_require(rc[r] <= capacity - total, Code::capacity);
      total += rc[r];
      result[r].resize(byte_size(rc[r]));
    }
  });
#ifdef POPS_HAS_MPI
  bool counts_match = true;
  for (int step = 0; step < lane.size(); ++step) {
    const int target = int((std::int64_t(lane.rank()) + step) % lane.size());
    const int source = int((std::int64_t(lane.rank()) - step + lane.size()) % lane.size());
    const ByteChunks sending(sc[target], chunk_limit), receiving(rc[source], chunk_limit);
    std::uint64_t local_rounds = std::max(sending.rounds(), receiving.rounds()), rounds = 0;
    mpi_status(
        MPI_Allreduce(&local_rounds, &rounds, 1, MPI_UINT64_T, MPI_MAX, lane.native_handle()));
    for (std::uint64_t round = 0; round < rounds; ++round) {
      const int sent = sending.count(round), received = receiving.count(round);
      const auto* from = sent ? send[target].data() + std::size_t(sending.offset(round)) : nullptr;
      auto* to = received ? result[source].data() + std::size_t(receiving.offset(round)) : nullptr;
      MPI_Status status;
      mpi_status(MPI_Sendrecv(from, sent, MPI_BYTE, target, 0, to, received, MPI_BYTE, source, 0,
                              lane.native_handle(), &status));
      int actual = 0;
      mpi_status(MPI_Get_count(&status, MPI_BYTE, &actual));
      counts_match = counts_match && actual == received;
    }
  }
  phase(lane, [&] { local_require(counts_match, Code::wire); });
#else
  std::copy(send[0].begin(), send[0].end(), result[0].begin());
#endif
  return result;
}

struct Requests {
  DomainRef domain, target;
  std::vector<Index> indices;
  std::shared_ptr<const void> value;
  int dimension = 0;
};
template <int Dim>
ArrayRef<Dim> requests_array(const Requests& requests) {
  local_require(requests.value && requests.dimension == Dim, Code::domain);
  auto array = std::static_pointer_cast<const Array<Dim>>(requests.value);
  local_require(array && array->domain == requests.domain, Code::domain);
  array->domain->require_live();
  return array;
}
template <int Dim>
Requests requests_from_array(const ArrayRef<Dim>& array) {
  local_require(array && array->domain &&
                    array->element.kind == ContinuationKind::DistributedIndex &&
                    array->values.size() == std::size_t(array->domain->local_size()),
                Code::domain);
  array->domain->require_live();
  Requests result{array->domain, array->index_domain, {}, array, Dim};
  result.indices.reserve(array->values.size());
  for (const auto& value : array->values) {
    local_require(value && value->kind() == ContinuationKind::DistributedIndex, Code::index);
    auto target = value->distributed_domain_ref();
    if (!result.target)
      result.target = target;
    local_require(result.target == target, Code::domain);
    result.indices.push_back(index(target, value->distributed_index_offset()));
  }
  return result;
}
template <int Dim>
void publish_requests(Registry& registry, Requests& requests, std::uint64_t capacity,
                      ContinuationTransportSession<Dim>* transport) {
  std::shared_ptr<Array<Dim>> pending;
  phase(registry.lane(), [&] {
    local_require(requests.target && requests.domain, Code::domain);
    ContinuationType element{ContinuationKind::DistributedIndex,
                             {},
                             {},
                             requests.target->family(),
                             requests.target->capacity()};
    pending = std::make_shared<Array<Dim>>(Array<Dim>{requests.domain, std::move(element), {}});
    pending->index_domain = requests.target;
    pending->values.reserve(requests.indices.size());
    for (const auto& value : requests.indices)
      pending->values.push_back(ContinuationValue<Dim>::distributed_index(value.domain, value.gid));
  });
  requests.value = publish_array<Dim>(registry, std::move(pending), capacity, transport);
  requests.dimension = Dim;
}
template <class F>
Requests requests(Registry& registry, DomainRef domain, DomainRef target, F&& body) {
  registry.common(domain);
  registry.common(target);
  Requests result{domain, target, {}};
  phase(registry.lane(), [&] {
    result.indices.reserve(std::size_t(domain->local_size()));
    for (std::int64_t gid = domain->begin(); gid < domain->end(); ++gid) {
      auto i = body(Index{domain, gid});
      local_require(i.domain == target && i.gid >= 0 && i.gid < target->size(), Code::index, i.gid);
      result.indices.push_back(std::move(i));
    }
  });
  return result;
}
// Duplicates retain distinct request positions; empty owners/requesters participate in both exchanges.
template <int Dim>
ArrayRef<Dim> import_values(Registry& registry, const ArrayRef<Dim>& source,
                            const Requests& requests, std::uint64_t byte_capacity,
                            ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto& lane = registry.lane();
  std::vector<Bytes> request_send;
  phase(lane, [&] { local_require(bool(source), Code::domain); });
  registry.common(source->domain);
  registry.common(requests.domain);
  phase(lane, [&] {
    local_require(source && source->domain, Code::domain);
    registry.authenticate(source->domain);
    registry.authenticate(requests.domain);
    if (requests.target)
      registry.authenticate(requests.target);
    local_require((!requests.target || requests.target == source->domain) &&
                      source->values.size() == std::size_t(source->domain->local_size()) &&
                      requests.indices.size() == std::size_t(requests.domain->local_size()),
                  Code::domain);
    request_send.resize(lane.size());
    for (std::size_t pos = 0; pos < requests.indices.size(); ++pos) {
      const auto& i = requests.indices[pos];
      local_require(i.domain == source->domain && i.gid >= 0 && i.gid < source->domain->size(),
                    Code::index, i.gid);
      auto& b = request_send[source->domain->owner(i.gid)];
      write_word(b, std::bit_cast<std::uint64_t>(i.gid));
      write_word(b, pos);
      local_require(b.size() <= byte_capacity, Code::capacity);
    }
  });
  auto incoming = exchange(lane, request_send, byte_capacity);
  std::vector<Bytes> reply_send;
  phase(lane, [&] {
    reply_send.resize(lane.size());
    for (int rank = 0; rank < lane.size(); ++rank) {
      std::size_t at = 0;
      auto& answer = reply_send[rank];
      while (at < incoming[rank].size()) {
        const auto gid = std::bit_cast<std::int64_t>(read_word(incoming[rank], at));
        auto pos = read_word(incoming[rank], at);
        local_require(source->domain->owns(gid), Code::index, gid);
        auto bytes = encode_payload<Dim>(source->element, at_local(source, {source->domain, gid}),
                                         byte_capacity, transport);
        local_require(bytes.size() <= byte_capacity && answer.size() <= byte_capacity &&
                          byte_capacity - answer.size() >= 16 &&
                          bytes.size() <= byte_capacity - answer.size() - 16,
                      Code::capacity);
        write_word(answer, pos);
        write_word(answer, bytes.size());
        answer.insert(answer.end(), bytes.begin(), bytes.end());
      }
    }
  });
  auto replies = exchange(lane, reply_send, byte_capacity);
  std::shared_ptr<Array<Dim>> pending;
  phase(lane, [&] {
    pending = std::make_shared<Array<Dim>>(Array<Dim>{requests.domain, source->element, {}});
    pending->values.resize(requests.indices.size());
    for (int owner = 0; owner < lane.size(); ++owner) {
      std::size_t at = 0;
      while (at < replies[owner].size()) {
        auto pos = read_word(replies[owner], at), n = read_word(replies[owner], at);
        local_require(pos < pending->values.size() && !pending->values[pos] &&
                          n <= replies[owner].size() - at,
                      Code::wire);
        local_require(source->domain->owner(requests.indices[pos].gid) == owner, Code::wire);
        pending->values[pos] = decode_payload<Dim>(
            source->element,
            std::span<const std::uint8_t>(replies[owner]).subspan(at, std::size_t(n)),
            byte_capacity, transport);
        at += std::size_t(n);
      }
    }
    local_require(std::all_of(pending->values.begin(), pending->values.end(),
                              [](const auto& v) { return bool(v); }),
                  Code::wire);
  });
  return publish_array<Dim>(registry, std::move(pending), byte_capacity, transport);
}

inline void agree_bytes(const Bytes& bytes, const ExecutionLane& lane) {
  phase(lane, [&] { local_require(bytes.size() <= maximum_budget, Code::capacity); });
  agree_i64(std::int64_t(bytes.size()), lane);
  bool same = true;
  std::array<std::uint8_t, 4096> chunk{};
  for (std::size_t offset = 0; offset < bytes.size(); offset += chunk.size()) {
    const auto n = std::min(chunk.size(), bytes.size() - offset);
    if (lane.rank() == 0)
      std::copy_n(bytes.begin() + offset, n, chunk.begin());
    broadcast(std::span<std::uint8_t>(chunk).first(n), 0, lane);
    same = same && std::equal(chunk.begin(), chunk.begin() + n, bytes.begin() + offset);
  }
  phase(lane, [&] { local_require(same, Code::domain); });
}

template <int Dim>
void common_initial(const ContinuationType& type, const ContinuationValueRef<Dim>& value,
                    std::uint64_t capacity, const ExecutionLane& lane,
                    ContinuationTransportSession<Dim>* transport = nullptr) {
  Bytes bytes;
  phase(lane, [&] {
    local_require(valid_budget(capacity), Code::capacity);
    bytes = encode_payload<Dim>(type, value, capacity, transport);
  });
  agree_bytes(bytes, lane);
}

template <int Dim>
void converge_payload_failure(const ExecutionLane& lane, ContinuationValueRef<Dim> payload,
                              const ContinuationType& type, std::uint64_t capacity,
                              ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto owner = common_failure_rank(bool(payload), lane);
  if (owner == lane.size())
    return;
  Bytes bytes;
  std::int64_t length = 0;
  phase(lane, [&] {
    if (lane.rank() == owner) {
      bytes = encode_payload<Dim>(type, payload, capacity, transport);
      length = std::int64_t(bytes.size());
    }
  });
#ifdef POPS_HAS_MPI
  mpi_status(MPI_Bcast(&length, 1, MPI_INT64_T, int(owner), lane.native_handle()));
#endif
  phase(lane, [&] {
    local_require(length >= 0 && std::uint64_t(length) <= capacity, Code::capacity);
    if (lane.rank() != owner)
      bytes.resize(byte_size(length));
  });
  broadcast(bytes, int(owner), lane);
  phase(lane, [&] {
    if (lane.rank() != owner)
      payload = decode_payload<Dim>(type, bytes, capacity, transport);
  });
  throw CollectivePayloadFailure<Dim>{std::move(payload)};
}

// Collect the declared global sequence exactly once per consumer. The explicit
// byte budget bounds local messages, each exchange, and the received sequence.
template <int Dim>
std::vector<ContinuationValueRef<Dim>> ordered_values(
    Registry& registry, const ArrayRef<Dim>& source, std::uint64_t capacity,
    ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto& lane = registry.lane();
  phase(lane, [&] { local_require(bool(source), Code::domain); });
  registry.common(source->domain);
  std::vector<Bytes> send;
  phase(lane, [&] {
    local_require(source->values.size() == std::size_t(source->domain->local_size()), Code::domain);
    Bytes local;
    for (std::int64_t gid = source->domain->begin(); gid < source->domain->end(); ++gid) {
      auto bytes = encode_payload<Dim>(source->element, at_local(source, {source->domain, gid}),
                                       capacity, transport);
      local_require(local.size() <= capacity && capacity - local.size() >= 16 &&
                        bytes.size() <= capacity - local.size() - 16,
                    Code::capacity);
      write_word(local, std::bit_cast<std::uint64_t>(gid));
      write_word(local, bytes.size());
      local.insert(local.end(), bytes.begin(), bytes.end());
    }
    local_require(local.size() <= capacity / std::uint64_t(lane.size()), Code::capacity);
    send.assign(std::size_t(lane.size()), local);
  });
  auto received = exchange(lane, send, capacity);
  std::vector<ContinuationValueRef<Dim>> ordered;
  phase(lane, [&] {
    ordered.resize(std::size_t(source->domain->size()));
    for (int rank = 0; rank < lane.size(); ++rank) {
      std::size_t at = 0;
      while (at < received[rank].size()) {
        const auto gid = std::bit_cast<std::int64_t>(read_word(received[rank], at));
        auto n = read_word(received[rank], at);
        local_require(gid >= 0 && gid < source->domain->size() &&
                          source->domain->owner(gid) == rank && !ordered[std::size_t(gid)] &&
                          n <= received[rank].size() - at,
                      Code::wire, gid);
        ordered[std::size_t(gid)] = decode_payload<Dim>(
            source->element,
            std::span<const std::uint8_t>(received[rank]).subspan(at, std::size_t(n)), capacity,
            transport);
        at += std::size_t(n);
      }
    }
    local_require(
        std::all_of(ordered.begin(), ordered.end(), [](const auto& x) { return bool(x); }),
        Code::wire);
  });
  return ordered;
}

// Ordered fold: every rank consumes the same global sequence, never MPI_SUM.
template <int Dim, class F>
ContinuationValueRef<Dim> reduce(Registry& registry, const ArrayRef<Dim>& source,
                                 const ContinuationType& state, ContinuationValueRef<Dim> initial,
                                 const ContinuationType& error, std::uint64_t capacity, F&& body,
                                 ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto& lane = registry.lane();
  common_initial<Dim>(state, initial, capacity, lane, transport);
  auto ordered = ordered_values<Dim>(registry, source, capacity, transport);
  ContinuationValueRef<Dim> result, local_error;
  phase(lane, [&] {
    try {
      auto carry = initial;
      for (const auto& value : ordered) {
        carry = body(carry, value);
        (void)encode_payload<Dim>(state, carry, capacity, transport);
      }
      result = std::move(carry);
    } catch (const LocalPayloadFailure<Dim>& failure) {
      local_error = failure.payload;
    }
  });
  converge_payload_failure<Dim>(lane, std::move(local_error), error, capacity, transport);
  common_initial<Dim>(state, result, capacity, lane, transport);
  return result;
}

template <int Dim>
struct Scan {
  ContinuationValueRef<Dim> state;
  ArrayRef<Dim> values;
};
template <int Dim, class F>
Scan<Dim> scan(Registry& registry, const ArrayRef<Dim>& source, const ContinuationType& state,
               ContinuationValueRef<Dim> initial, const ContinuationType& error,
               const ContinuationType& element, std::uint64_t capacity, F&& body,
               ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto& lane = registry.lane();
  common_initial<Dim>(state, initial, capacity, lane, transport);
  auto ordered = ordered_values<Dim>(registry, source, capacity, transport);
  Scan<Dim> result;
  std::shared_ptr<Array<Dim>> pending;
  ContinuationValueRef<Dim> local_error;
  phase(lane, [&] {
    try {
      pending = std::make_shared<Array<Dim>>(Array<Dim>{source->domain, element, {}});
      pending->values.reserve(std::size_t(source->domain->local_size()));
      auto carry = initial;
      for (std::size_t gid = 0; gid < ordered.size(); ++gid) {
        auto [next, value] = body(carry, ordered[gid]);
        (void)encode_payload<Dim>(state, next, capacity, transport);
        (void)encode_payload<Dim>(element, value, capacity, transport);
        if (source->domain->owns(std::int64_t(gid)))
          pending->values.push_back(std::move(value));
        carry = std::move(next);
      }
      result.state = std::move(carry);
    } catch (const LocalPayloadFailure<Dim>& failure) {
      local_error = failure.payload;
    }
  });
  converge_payload_failure<Dim>(lane, std::move(local_error), error, capacity, transport);
  common_initial<Dim>(state, result.state, capacity, lane, transport);
  result.values = publish_array<Dim>(registry, std::move(pending), capacity, transport);
  return result;
}

// Each incident source contributes exactly once, and repeated target indices
// retain their multiplicity. Destination-local folds follow source global IDs.
template <int Dim, class F>
ArrayRef<Dim> scatter_reduce(Registry& registry, const ArrayRef<Dim>& source,
                             const Requests& indices, DomainRef target,
                             const ContinuationType& state, ContinuationValueRef<Dim> initial,
                             const ContinuationType& error, std::uint64_t capacity, F&& body,
                             ContinuationTransportSession<Dim>* transport = nullptr) {
  const auto& lane = registry.lane();
  phase(lane, [&] { local_require(bool(source), Code::domain); });
  registry.common(source->domain);
  registry.common(indices.domain);
  registry.common(target);
  common_initial<Dim>(state, initial, capacity, lane, transport);
  std::vector<Bytes> send;
  phase(lane, [&] {
    local_require(indices.domain == source->domain &&
                      (!indices.target || indices.target == target) &&
                      source->values.size() == std::size_t(source->domain->local_size()) &&
                      indices.indices.size() == source->values.size(),
                  Code::domain);
    send.resize(std::size_t(lane.size()));
    for (std::size_t pos = 0; pos < source->values.size(); ++pos) {
      const auto& index = indices.indices[pos];
      const auto gid = source->domain->begin() + std::int64_t(pos);
      local_require(index.domain == target && index.gid >= 0 && index.gid < target->size(),
                    Code::index, index.gid);
      auto bytes = encode_payload<Dim>(source->element, source->values[pos], capacity, transport);
      auto& message = send[std::size_t(target->owner(index.gid))];
      local_require(message.size() <= capacity && capacity - message.size() >= 24 &&
                        bytes.size() <= capacity - message.size() - 24,
                    Code::capacity);
      write_word(message, std::bit_cast<std::uint64_t>(index.gid));
      write_word(message, std::bit_cast<std::uint64_t>(gid));
      write_word(message, bytes.size());
      message.insert(message.end(), bytes.begin(), bytes.end());
    }
  });
  auto received = exchange(lane, send, capacity);
  std::shared_ptr<Array<Dim>> pending;
  ContinuationValueRef<Dim> local_error;
  phase(lane, [&] {
    try {
      struct Entry {
        std::int64_t target, gid;
        ContinuationValueRef<Dim> value;
      };
      std::vector<Entry> entries;
      for (int rank = 0; rank < lane.size(); ++rank) {
        std::size_t at = 0;
        while (at < received[rank].size()) {
          const auto destination = std::bit_cast<std::int64_t>(read_word(received[rank], at));
          const auto gid = std::bit_cast<std::int64_t>(read_word(received[rank], at));
          const auto n = read_word(received[rank], at);
          local_require(target->owns(destination) && gid >= 0 && gid < source->domain->size() &&
                            source->domain->owner(gid) == rank && n <= received[rank].size() - at,
                        Code::wire, gid);
          auto value = decode_payload<Dim>(
              source->element,
              std::span<const std::uint8_t>(received[rank]).subspan(at, std::size_t(n)), capacity,
              transport);
          at += std::size_t(n);
          entries.push_back({destination, gid, std::move(value)});
        }
      }
      std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return a.gid < b.gid; });
      pending = std::make_shared<Array<Dim>>(Array<Dim>{target, state, {}});
      pending->values.assign(std::size_t(target->local_size()), initial);
      std::int64_t previous = -1;
      for (const auto& entry : entries) {
        local_require(entry.gid > previous, Code::wire, entry.gid);
        previous = entry.gid;
        auto& carry = pending->values[std::size_t(entry.target - target->begin())];
        carry = body(carry, entry.value);
        (void)encode_payload<Dim>(state, carry, capacity, transport);
      }
    } catch (const LocalPayloadFailure<Dim>& failure) {
      local_error = failure.payload;
    }
  });
  converge_payload_failure<Dim>(lane, std::move(local_error), error, capacity, transport);
  return publish_array<Dim>(registry, std::move(pending), capacity, transport);
}
struct CSR {
  DomainRef rows, target;
  Requests columns;
  std::vector<std::int64_t> offsets;
};
inline CSR csr(Registry& registry, DomainRef rows, DomainRef target, const Requests& columns,
               const std::vector<std::int64_t>& offsets, std::uint64_t nnz_capacity) {
  registry.common(rows);
  registry.common(target);
  registry.common(columns.domain);
  CSR result;
  phase(registry.lane(), [&] {
    local_require((!columns.target || columns.target == target) &&
                      columns.indices.size() <= nnz_capacity &&
                      offsets.size() == std::size_t(rows->local_size() + 1) &&
                      offsets.front() == 0 && std::is_sorted(offsets.begin(), offsets.end()) &&
                      offsets.back() == std::int64_t(columns.indices.size()) &&
                      columns.indices.size() == std::size_t(columns.domain->local_size()),
                  Code::csr);
    for (const auto& i : columns.indices)
      local_require(i.domain == target && i.gid >= 0 && i.gid < target->size(), Code::index, i.gid);
    result = {rows, target, columns, offsets};
    result.columns.target = target;
  });
  return result;
}
template <class F>
CSR csr_offsets(Registry& registry, DomainRef rows, DomainRef target, const Requests& columns,
                std::uint64_t capacity, F&& offset_body) {
  registry.common(rows);
  registry.common(target);
  registry.common(columns.domain);
  std::vector<std::int64_t> offsets;
  Requests owned;
  phase(registry.lane(), [&] {
    owned = columns;
    offsets.reserve(std::size_t(rows->local_size() + 1));
    for (std::int64_t i = 0; i <= rows->local_size(); ++i)
      offsets.push_back(offset_body(i));
  });
  return csr(registry, std::move(rows), std::move(target), std::move(owned), std::move(offsets),
             capacity);
}
template <int Dim>
ArrayRef<Dim> import_csr(Registry& registry, const ArrayRef<Dim>& source, const CSR& incidence,
                         std::uint64_t capacity,
                         ContinuationTransportSession<Dim>* transport = nullptr) {
  phase(registry.lane(),
        [&] { local_require(source && source->domain == incidence.target, Code::domain); });
  return import_values(registry, source, incidence.columns, capacity, transport);
}
}  // namespace pops::runtime::program::distributed
