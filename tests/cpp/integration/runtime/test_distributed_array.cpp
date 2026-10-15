// PREPARED ONLY. Native PoPS execution-lane/codec test; compile and run under MPI2 and MPI3.
// This is not a replacement runtime and does not qualify the pending full Program dispatch bridge.
#include <pops/runtime/program/distributed_array.hpp>
#include <pops/runtime/program/continuation_closure.hpp>
#include <pops/runtime/program/finite_kokkos.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
namespace d = pops::runtime::program::distributed;
namespace p = pops::runtime::program;
using V = p::ContinuationValue<2>;
using T = p::ContinuationType;
using K = p::ContinuationKind;
static constexpr std::int64_t large = (std::int64_t(1) << 60) + 17;
void check(const pops::ExecutionLane& lane, bool valid) {
  d::phase(lane, [&] { d::local_require(valid, d::Code::internal); });
}
template <class F>
void rejects(const pops::ExecutionLane& lane, d::Code expected, F&& operation) {
  bool caught = false;
  try {
    operation();
  } catch (const d::Failure& e) {
    caught = e.code == expected;
  }
  check(lane, caught);
}
int run() {
  auto lane = pops::ExecutionLane::world();
  const int n = lane.size(), rank = lane.rank();
  check(lane, n == 2 || n == 3);
  d::Registry registry(lane);
  std::vector<std::int64_t> owners(n + 1, 1), requested(n + 1, 2);
  owners.front() = requested.front() = 0;
  owners.back() = 2;
  requested.back() = 3;
  auto D = registry.issue(10, 4, owners), Q = registry.issue(11, 4, requested);
  T integer{K::Int64}, real{K::Float64}, boolean{K::Boolean};
  T tag, element, error;
  d::phase(lane, [&] {
    tag = T{K::Variant, {boolean, integer}, {"boolean", "integer"}};
    element = T{K::Product, {integer, tag}, {"id", "tag"}};
    error = T{K::Product, {integer, real}, {"code", "defect"}};
  });
  auto value = [](std::int64_t gid) {
    return V::product({V::integer(large + 10 * gid),
                       V::variant(unsigned(gid % 2), gid % 2 ? V::integer(99) : V::boolean(true))});
  };
  auto made = d::map<2>(registry, D, element, error, 4096,
                        [&](d::Index i) { return V::variant(1, value(i.gid)); });
  check(lane, made.ok());
  const auto source = made.value;
  std::vector<std::vector<std::uint8_t>> before;
  d::phase(lane, [&] {
    for (const auto& x : source->values)
      before.push_back(p::encode_continuation_transport<2>(element, x, 4096));
  });
  auto req =
      d::requests(registry, Q, D, [&](d::Index i) { return d::index(D, i.gid < 2 ? 1 : 0); });
  auto imported = d::import_values(registry, source, req, 65536);
  bool exact = imported->values.size() == std::size_t(Q->local_size());
  for (std::size_t j = 0; j < imported->values.size(); ++j) {
    const auto wanted = (Q->begin() + std::int64_t(j)) < 2 ? 1 : 0;
    const auto& x = imported->values[j];
    exact = exact && x->members()[0]->integer() == large + 10 * wanted;
    exact = exact && x->members()[1]->arm() == unsigned(wanted % 2);
    exact = exact && (wanted % 2 ? x->members()[1]->members()[0]->integer() == 99
                                 : x->members()[1]->members()[0]->boolean());
  }
  check(lane, exact);  // independent global formula, exact Int64, duplicate positions, empty rank

  auto C = d::csr_offsets(registry, D, D, req, 4,
                          [&](std::int64_t row) { return row == 0 ? 0 : Q->local_size(); });
  auto via_csr = d::import_csr(registry, source, C, 65536);
  check(lane, via_csr->values.size() == imported->values.size());
  for (std::size_t j = 0; j < via_csr->values.size(); ++j)
    exact = exact && via_csr->values[j]->members()[0]->integer() ==
                         imported->values[j]->members()[0]->integer();
  check(lane, exact);

  // Full E is selected at the first failure of the lowest failing rank, including >2^53.
  auto failed = d::map<2>(registry, Q, element, error, 4096, [&](d::Index i) {
    if (i.gid == 1)
      return V::variant(0, V::product({V::integer(large), V::real(.25)}));
    return V::variant(1, value(i.gid));
  });
  check(lane, !failed.ok() && failed.rank == 0 && failed.gid == 1 && failed.code == d::Code::user &&
                  failed.payload->members()[0]->integer() == large &&
                  failed.payload->members()[1]->real() == .25);
  auto arithmetic = d::map<2>(registry, D, element, error, 4096, [&](d::Index i) {
    if (i.gid == 1)
      (void)d::integer(3, 1, 0);
    return V::variant(1, value(i.gid));
  });
  check(lane, !arithmetic.ok() && arithmetic.code == d::Code::arithmetic &&
                  arithmetic.rank == n - 1 && arithmetic.gid == 1);

  // Error only on one rank must be observed before any peer enters the next communication.
  rejects(lane, d::Code::index, [&] {
    auto bad = req;
    if (rank == n - 1)
      bad.indices[0].gid = 100;
    (void)d::import_values(registry, source, bad, 65536);
  });
  rejects(lane, d::Code::capacity,
          [&] { (void)d::import_values(registry, source, req, rank == 0 ? 1 : 65536); });
  rejects(lane, d::Code::csr, [&] {
    (void)d::csr_offsets(registry, D, D, req, 4, [&](std::int64_t row) {
      return row == 0 ? 0 : Q->local_size() + (rank == n - 1 ? 1 : 0);
    });
  });
  rejects(lane, d::Code::allocation, [&] {
    d::phase(lane, [&] {
      if (rank == (n == 3 ? 1 : 0))
        throw std::bad_alloc();
    });
  });

  // A new same-family/same-shape instance cannot authenticate old indices.
  const auto issued = registry.issued();
  auto fresh = registry.issue(10, 4, owners);
  check(lane, registry.issued() == issued + 1 && fresh->identity().ordinal > D->identity().ordinal);
  rejects(lane, d::Code::index, [&] {
    auto forged = req;
    if (rank == n - 1)
      forged.indices[0].domain = fresh;
    (void)d::import_values(registry, source, forged, 65536);
  });
  d::Registry restarted(lane);
  rejects(lane, d::Code::domain, [&] { d::phase(lane, [&] { restarted.authenticate(D); }); });
  rejects(lane, d::Code::domain, [&] {
    auto divergent = owners;
    if (rank == n - 1)
      divergent.back() = 3;
    (void)registry.issue(19, 4, divergent);
  });

  // No attempted failure changed an already published immutable array.
  exact = source == made.value;
  d::phase(lane, [&] {
    for (std::size_t j = 0; j < before.size(); ++j)
      exact = exact &&
              before[j] == p::encode_continuation_transport<2>(element, source->values[j], 4096);
  });
  check(lane, exact);
  // All-zero sizes still enter the count exchanges and produce a complete empty result.
  std::vector<std::int64_t> zeros(n + 1, 0);
  auto Z = registry.issue(20, 1, zeros);
  auto z = d::map<2>(registry, Z, element, error, 4096,
                     [&](d::Index i) { return V::variant(1, value(i.gid)); });
  auto zr = d::requests(registry, Z, Z, [&](d::Index i) { return i; });
  auto zi = d::import_values(registry, z.value, zr, 4096);
  check(lane, zi->values.empty());
  if (rank == 0)
    std::cout << "distributed MPI protocol prepared checks passed on " << n << " ranks\n";
  return 0;
}
TEST(DistributedArray, ExactNativeProtocolWithAsymmetricOwnership) {
  const auto lane = pops::ExecutionLane::world();
  if (lane.size() != 2 && lane.size() != 3)
    GTEST_SKIP() << "requires MPI2 or MPI3";
  EXPECT_EQ(run(), 0);
}

TEST(DistributedArray, RetainedIssuerIsRevokedAndRankLocalRevocationConverges) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1);
  offsets.front() = 0;
  auto domain = registry.issue(61, 2, offsets);
  const auto retained_index = d::index(domain, 0);
  auto local = std::make_shared<d::Array<2>>();
  local->domain = domain;
  local->element = T{K::Int64};
  local->values.resize(std::size_t(domain->local_size()), V::integer(3));
  d::ArrayRef<2> array = local;
  if (lane.rank() == 0)
    registry.revoke();
  // The revocation is deliberately inconsistent here. Registry.common must converge the error
  // before any peer performs a domain-sized operation or communication.
  rejects(lane, d::Code::domain, [&] { registry.common(domain); });
  registry.revoke();
  EXPECT_FALSE(domain->visible());
  EXPECT_THROW(domain->size(), d::Failure);
  EXPECT_THROW(d::index(domain, 0), d::Failure);
  EXPECT_THROW(d::at_local<2>(array, retained_index), d::Failure);
  rejects(lane, d::Code::domain, [&] { (void)registry.issue(61, 2, offsets); });
}

TEST(DistributedArray, DestroyingRegistryRevokesSurvivingDomainWitness) {
  auto lane = pops::ExecutionLane::world();
  d::DomainRef old;
  {
    d::Registry registry(lane);
    std::vector<std::int64_t> offsets(lane.size() + 1, 1);
    offsets.front() = 0;
    old = registry.issue(62, 2, offsets);
  }
  EXPECT_FALSE(old->visible());
  EXPECT_THROW(old->local_size(), d::Failure);
}

TEST(DistributedArray, Int64DomainCapacityDoesNotReserveItsCardinality) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  const auto capacity = std::numeric_limits<std::int64_t>::max();
  std::vector<std::int64_t> offsets(lane.size() + 1, 2);
  offsets.front() = 0;
  auto domain = registry.issue(63, capacity, offsets);
  EXPECT_EQ(domain->size(), 2);
  EXPECT_EQ(domain->capacity(), capacity);
  EXPECT_LT(registry.retained_bytes(), 4096);
  const T integer{K::Int64};
  auto made = d::map<2>(registry, domain, integer, integer, 4096,
                        [](d::Index i) { return V::variant(1, V::integer(i.gid + 17)); });
  check(lane, made.ok() && made.value->values.size() == std::size_t(domain->local_size()));
  for (std::size_t i = 0; i < made.value->values.size(); ++i)
    EXPECT_EQ(made.value->values[i]->integer(), domain->begin() + std::int64_t(i) + 17);

  // Checkpoint reconstruction preserves large capacity and exact issuer/offset authority.
  auto restored = d::Registry::restore_local(lane, registry.issuer(), 65536, 8);
  auto copy = restored->restore_domain_local(domain->identity(), 63, capacity, offsets);
  EXPECT_EQ(copy->capacity(), capacity);
  EXPECT_EQ(copy->partition_offsets(), offsets);
  EXPECT_THROW(restored->restore_domain_local(domain->identity(), 63, capacity, offsets),
               d::Failure);

  std::vector<std::int64_t> empty(lane.size() + 1, 0);
  auto zero = registry.issue(64, 0, empty);
  auto large_empty = registry.issue(65, capacity, empty);
  EXPECT_EQ(zero->size(), 0);
  auto unused = d::map<2>(registry, large_empty, integer, integer, 4096,
                          [](d::Index) -> p::ContinuationValueRef<2> {
                            throw std::logic_error("empty map body executed");
                          });
  check(lane, unused.ok() && unused.value->values.empty());
  rejects(lane, d::Code::domain, [&] { (void)registry.issue(66, -1, empty); });
  rejects(lane, d::Code::domain, [&] { (void)registry.issue(66, 1, offsets); });
  if (lane.size() > 1)
    rejects(lane, d::Code::domain,
            [&] { (void)registry.issue(66, capacity - (lane.rank() == 0 ? 1 : 0), empty); });
}

TEST(DistributedArray, PortableKokkosIntegerChecksPreserveFloorAndOverflowSemantics) {
  pops::detail::ensure_kokkos_initialized();
  namespace k = pops::runtime::program::finite_kokkos;
  constexpr auto lo = std::numeric_limits<std::int64_t>::min();
  constexpr auto hi = std::numeric_limits<std::int64_t>::max();
  struct Input {
    unsigned op;
    std::int64_t a, b, expected;
    bool valid;
  };
  const Input cases[] = {{0, hi, 1, 0, false}, {0, lo, -1, 0, false}, {0, hi, 0, hi, true},
                         {1, lo, 1, 0, false}, {1, hi, -1, 0, false}, {2, lo, -1, 0, false},
                         {2, lo, 1, lo, true}, {2, -7, -3, 21, true}, {3, -7, 3, -3, true},
                         {4, -7, 3, 2, true},  {4, 7, -3, -2, true},  {3, lo, -1, 0, false},
                         {3, 1, 0, 0, false}};
  const auto n = sizeof(cases) / sizeof(cases[0]);
  Kokkos::View<Input*> inputs("finite_integer_cases", n);
  auto host = Kokkos::create_mirror_view(inputs);
  for (std::size_t i = 0; i < n; ++i)
    host(i) = cases[i];
  Kokkos::deep_copy(inputs, host);
  auto outputs = k::allocate<std::int64_t>(std::int64_t(n), "finite_integer_results");
  auto valid = k::allocate<unsigned char>(std::int64_t(n), "finite_integer_validity");
  Kokkos::parallel_for(
      "finite_checked_integer_reception", n, KOKKOS_LAMBDA(const int i) {
        std::int64_t result = 0;
        valid(i) = k::integer(inputs(i).op, inputs(i).a, inputs(i).b, result);
        outputs(i) = result;
      });
  Kokkos::fence();
  auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outputs);
  auto actual_valid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, valid);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(bool(actual_valid(i)), cases[i].valid);
    if (cases[i].valid)
      EXPECT_EQ(actual(i), cases[i].expected);
  }
  EXPECT_THROW(k::allocate<double>(-1, "negative"), std::length_error);
  EXPECT_THROW(k::allocate<double>(hi, "unaddressable"), std::length_error);
}

TEST(DistributedArray, FailureDiagnosticRetainsExactFieldsWithoutAllocatingStorage) {
  static_assert(noexcept(d::Failure(d::Code::capacity)));
  const auto lo = std::numeric_limits<std::int64_t>::min();
  const auto hi = std::numeric_limits<std::int64_t>::max();
  const d::Failure original(d::Code::capacity, lo, hi);
  const d::Failure copied = original;
  const std::exception& erased = copied;
  EXPECT_EQ(copied.code, d::Code::capacity);
  EXPECT_EQ(copied.rank, lo);
  EXPECT_EQ(copied.gid, hi);
  EXPECT_EQ(std::string(erased.what()),
            "distributed finite phase rejected [code=2 rank=-9223372036854775808 "
            "gid=9223372036854775807]");
  d::Failure assigned(d::Code::internal);
  assigned = copied;
  EXPECT_EQ(std::string(assigned.what()), std::string(original.what()));
  const d::Failure default_location(d::Code::arithmetic);
  EXPECT_EQ(std::string(default_location.what()),
            "distributed finite phase rejected [code=6 rank=-1 gid=-1]");
}

TEST(DistributedArray, ByteChunksPlansInt64ExtentsWithoutAllocatingPayloads) {
  const auto max = std::uint64_t(std::numeric_limits<std::int64_t>::max());
  const std::array<std::uint64_t, 8> sizes{
      0, 1, 6, 7, 8, std::uint64_t(INT_MAX), std::uint64_t(INT_MAX) + 1, max};
  for (const auto size : sizes)
    for (const auto chunk : std::array<std::uint64_t, 3>{1, 7, std::uint64_t(INT_MAX)}) {
      const d::ByteChunks plan(size, chunk);
      EXPECT_EQ(plan.rounds(), size == 0 ? 0 : 1 + (size - 1) / chunk);
      EXPECT_EQ(plan.count(0), int(std::min(size, chunk)));
      EXPECT_EQ(plan.offset(std::numeric_limits<std::uint64_t>::max()), size);
      EXPECT_EQ(plan.count(std::numeric_limits<std::uint64_t>::max()), 0);
      if (plan.rounds()) {
        const auto last = plan.rounds() - 1;
        EXPECT_LE(plan.offset(last), size);
        EXPECT_GT(plan.count(last), 0);
        EXPECT_EQ(plan.offset(last) + std::uint64_t(plan.count(last)), size);
      }
    }
  EXPECT_THROW((void)d::ByteChunks(max + 1), d::Failure);
  EXPECT_THROW((void)d::ByteChunks(1, 0), d::Failure);
  EXPECT_THROW((void)d::ByteChunks(1, std::uint64_t(INT_MAX) + 1), d::Failure);
}

TEST(DistributedArray, ChunkedBroadcastPreservesBytesAndRejectsBeforeMutation) {
  const auto lane = pops::ExecutionLane::world();
  for (const int root : std::array<int, 2>{0, lane.size() - 1}) {
    d::Bytes bytes(29, 0);
    if (lane.rank() == root)
      for (std::size_t j = 0; j < bytes.size(); ++j)
        bytes[j] = std::uint8_t(root * 19 + j * 3);
    d::broadcast(bytes, root, lane, 7);
    for (std::size_t j = 0; j < bytes.size(); ++j)
      EXPECT_EQ(bytes[j], std::uint8_t(root * 19 + j * 3));
    d::Bytes empty;
    d::broadcast(empty, root, lane, 7);
    EXPECT_TRUE(empty.empty());
  }
  d::Bytes bytes(17, std::uint8_t(lane.rank() + 7));
  const auto before = bytes;
  rejects(lane, d::Code::capacity, [&] { d::broadcast(bytes, 0, lane, lane.rank() == 0 ? 0 : 7); });
  EXPECT_EQ(bytes, before);
  if (lane.size() > 1) {
    rejects(lane, d::Code::domain, [&] { d::broadcast(bytes, 0, lane, lane.rank() == 0 ? 1 : 7); });
    EXPECT_EQ(bytes, before);
    if (lane.rank() == lane.size() - 1)
      bytes.pop_back();
    const auto unequal = bytes;
    rejects(lane, d::Code::domain, [&] { d::broadcast(bytes, 0, lane, 7); });
    EXPECT_EQ(bytes, unequal);
  }
}

TEST(DistributedArray, ChunkedExchangeUsesActualAsymmetricAndEmptyBuckets) {
  const auto lane = pops::ExecutionLane::world();
  const int n = lane.size(), rank = lane.rank();
  const auto capacity = std::uint64_t(std::numeric_limits<std::int64_t>::max());
  auto length = [&](int source, int target, int pattern) -> std::size_t {
    if (pattern == 2 || (n > 1 && source == 0))
      return 0;
    if (pattern == 1)
      return target == n - 1 ? std::size_t(23 + source * 3) : 0;
    return std::size_t(1 + (source + 1) * (target + 3) % 29);
  };
  auto byte = [](int source, int target, std::size_t j) {
    return std::uint8_t(source * 31 + target * 7 + j * 3);
  };
  for (int pattern = 0; pattern < 3; ++pattern) {
    std::vector<d::Bytes> send(n);
    for (int target = 0; target < n; ++target) {
      send[target].resize(length(rank, target, pattern));
      for (std::size_t j = 0; j < send[target].size(); ++j)
        send[target][j] = byte(rank, target, j);
    }
    const auto before = send;
    const auto received = d::exchange(lane, send, capacity, 7);
    EXPECT_EQ(send, before);
    check(lane, received.size() == std::size_t(n));
    for (int source = 0; source < n; ++source) {
      EXPECT_EQ(received[source].size(), length(source, rank, pattern));
      for (std::size_t j = 0; j < received[source].size(); ++j)
        EXPECT_EQ(received[source][j], byte(source, rank, j));
    }
    rejects(lane, d::Code::capacity,
            [&] { (void)d::exchange(lane, send, capacity, rank == 0 ? 0 : 7); });
    EXPECT_EQ(send, before);
    if (pattern == 0) {
      rejects(lane, d::Code::capacity,
              [&] { (void)d::exchange(lane, send, rank == n - 1 ? 1 : capacity, 7); });
      EXPECT_EQ(send, before);
      if (n > 1) {
        // Rank0 sends nothing, so its insufficient RECEIVE budget fails only
        // after the count exchange, before any data transfer/publication.
        rejects(lane, d::Code::capacity,
                [&] { (void)d::exchange(lane, send, rank == 0 ? 1 : capacity, 7); });
        EXPECT_EQ(send, before);
        rejects(lane, d::Code::domain,
                [&] { (void)d::exchange(lane, send, capacity, rank == 0 ? 1 : 7); });
      }
    }
  }
}

TEST(DistributedArray, WideBudgetsChargeActualStorageAndTransportSmallPayloads) {
  const auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  const auto budget = std::uint64_t(std::numeric_limits<std::int64_t>::max());
  registry.configure(budget, budget);
  std::vector<std::int64_t> offsets(lane.size() + 1, 2);
  offsets.front() = 0;
  const auto domain = registry.issue(71, 2, offsets);
  const T integer{K::Int64};
  auto made = d::map<2>(registry, domain, integer, integer, budget,
                        [](d::Index i) { return V::variant(1, V::integer(i.gid + 11)); });
  check(lane, made.ok());
  EXPECT_LT(registry.retained_bytes(), 65536);
  auto requests = d::requests(registry, domain, domain,
                              [&](d::Index i) { return d::index(domain, 1 - i.gid); });
  auto imported = d::import_values(registry, made.value, requests, budget);
  for (std::size_t i = 0; i < imported->values.size(); ++i)
    EXPECT_EQ(imported->values[i]->integer(), 12 - domain->begin() - std::int64_t(i));
  const auto before_bytes = registry.retained_bytes(), before_values = registry.retained_values();
  rejects(lane, d::Code::capacity,
          [&] { (void)d::import_values(registry, made.value, requests, 1); });
  EXPECT_EQ(registry.retained_bytes(), before_bytes);
  EXPECT_EQ(registry.retained_values(), before_values);
  const auto root = V::distributed_array(imported);
  auto closure = p::continuation_collective_closure<2>(root, &registry, lane, budget);
  EXPECT_FALSE(closure->arrays.empty());
  auto restored = d::Registry::restore_local(lane, registry.issuer(), budget, budget);
  EXPECT_EQ(restored->storage_bytes(), budget);
  EXPECT_EQ(restored->storage_versions(), budget);
  rejects(lane, d::Code::capacity, [&] {
    d::Registry invalid(lane);
    invalid.configure(budget + 1, 8);
  });
}

namespace {
void bind_registry(p::ContinuationTransportSession<2>& session, d::Registry& registry) {
  session.bind_distributed_registry(
      registry, [&registry](d::Identity id, d::DomainRef domain, const T& element) {
        return d::resolve_array<2>(registry, id, domain, element);
      });
}
}  // namespace
TEST(DistributedArray, MultiIssuerContinuationClosureAndTransportKeepExactAuthorities) {
  const auto lane = pops::ExecutionLane::world();
  auto first = std::make_shared<d::Registry>(lane);
  auto second = std::make_shared<d::Registry>(lane);
  std::vector<std::int64_t> offsets(lane.size() + 1, 2);
  offsets.front() = 0;  // Empty shards on all ranks except the first.
  auto a = first->issue(71, 2, offsets), b = second->issue(71, 2, offsets);
  const T integer{K::Int64};
  const auto make = [&](auto& registry, auto domain, std::int64_t offset) {
    return d::map<2>(registry, domain, integer, integer, 65536,
                     [&](d::Index i) { return V::variant(1, V::integer(offset + i.gid)); })
        .value;
  };
  auto av = make(*first, a, 11), bv = make(*second, b, 101);
  T array{K::DistributedArray, {integer}};
  array.family = 71;
  array.capacity = 2;
  const T type{K::Product, {array, array}, {"first", "second"}};
  const auto value = V::product({V::distributed_array(av), V::distributed_array(bv)});
  const std::vector<std::shared_ptr<d::Registry>> authorities{first, second};
  auto closure = p::continuation_collective_closure<2>(value, first, lane, 65536, {}, authorities);
  EXPECT_EQ(closure->registries.size(), 2u);
  EXPECT_EQ(closure->domains.size(), 2u);
  EXPECT_EQ(closure->arrays.size(), 2u);
  const auto before_first = first->retained_bytes(), before_second = second->retained_bytes();
  EXPECT_THROW(p::continuation_collective_closure<2>(value, first, lane, 65536), d::Failure);
  EXPECT_EQ(first->retained_bytes(), before_first);
  EXPECT_EQ(second->retained_bytes(), before_second);
  p::ContinuationTransportSession<2> session;
  bind_registry(session, *first);
  EXPECT_THROW(session.encode(type, value, 65536), std::logic_error);
  bind_registry(session, *second);
  const auto bytes = session.encode(type, value, 65536);
  const auto decoded = session.decode(type, bytes, 65536);
  EXPECT_EQ(decoded->members()[0]->distributed_array_ref(), av);
  EXPECT_EQ(decoded->members()[1]->distributed_array_ref(), bv);
  auto duplicate = d::Registry::restore_local(lane, first->issuer(), first->storage_bytes(),
                                              first->storage_versions());
  EXPECT_THROW(bind_registry(session, *duplicate), std::logic_error);
  auto conflicting = authorities;
  conflicting.push_back(duplicate);
  EXPECT_THROW(p::continuation_collective_closure<2>(value, first, lane, 65536, {}, conflicting),
               d::Failure);
  second->revoke();
  EXPECT_THROW(session.decode(type, bytes, 65536), d::Failure);
}
TEST(DistributedArray, MultiIssuerCheckpointRetainsAllocatorAndRankLocalShards) {
  const auto lane = pops::ExecutionLane::world();
  auto first = std::make_shared<d::Registry>(lane);
  auto second = std::make_shared<d::Registry>(lane);
  std::vector<std::int64_t> offsets(lane.size() + 1, 2);
  offsets.front() = 0;
  const T integer{K::Int64};
  auto domain_a = first->issue(83, 2, offsets), domain_b = second->issue(83, 2, offsets);
  const auto make = [&](auto& registry, auto domain, std::int64_t offset) {
    return d::map<2>(registry, domain, integer, integer, 65536,
                     [&](d::Index i) { return V::variant(1, V::integer(offset + i.gid)); })
        .value;
  };
  auto a = make(*first, domain_a, 31), b = make(*second, domain_b, 71);
  T array{K::DistributedArray, {integer}};
  array.family = 83;
  array.capacity = 2;
  const T type{K::Product, {array, array}, {"first", "second"}};
  const auto root = V::product({V::distributed_array(a), V::distributed_array(b)});
  const std::vector<std::shared_ptr<d::Registry>> authorities{first, second};
  auto closure = p::continuation_collective_closure<2>(root, first, lane, 65536, {}, authorities);
  const auto prototypes = [](int) -> const pops::MultiFab<2>& {
    throw std::logic_error("no field in test schema");
  };
  p::ContinuationCodec<2> codec(type, "multi-issuer", 65536, prototypes, &lane, first.get(),
                                closure.get());
  const auto bytes = codec.encode(root, 19);
  p::continuation_detail::Reader header(bytes);
  EXPECT_EQ(header.text(), "POPSCONT3");
  auto decoded = codec.decode(bytes);
  EXPECT_EQ(decoded.publication, 19u);
  ASSERT_TRUE(decoded.distributed_registry);
  EXPECT_EQ(decoded.distributed_registry->issuer(), first->issuer());
  ASSERT_EQ(decoded.closure->registries.size(), 2u);
  for (std::size_t k = 0; k != 2; ++k) {
    const auto restored = decoded.root->members()[k]->distributed_array_ref();
    EXPECT_NE(restored, k ? b : a);
    EXPECT_EQ(restored->version.issuer, (k ? b : a)->version.issuer);
    for (std::size_t i = 0; i < restored->values.size(); ++i)
      EXPECT_EQ(restored->values[i]->integer(),
                (k ? 71 : 31) + domain_a->begin() + std::int64_t(i));
  }
  auto verified = p::continuation_collective_closure<2>(
      decoded.root, decoded.distributed_registry, lane, 65536, {}, decoded.closure->registries);
  EXPECT_EQ(verified->arrays, decoded.closure->arrays);
  p::ContinuationCodec<2> recode(type, "multi-issuer", 65536, prototypes, &lane,
                                 decoded.distributed_registry.get(), decoded.closure.get());
  EXPECT_EQ(recode.encode(decoded.root, decoded.publication), bytes);
  p::ContinuationCodec<2> tiny(type, "multi-issuer", 64, prototypes, &lane, first.get(),
                               closure.get());
  EXPECT_THROW(tiny.encode(root, 19), std::length_error);
  EXPECT_EQ(codec.encode(root, 19), bytes);
  // Retaining a child does not make that child the parent's allocator.
  p::ContinuationCodec<2> borrowed(type, "multi-issuer", 65536, prototypes, &lane, nullptr,
                                   closure.get());
  EXPECT_FALSE(borrowed.decode(borrowed.encode(root, 19)).distributed_registry);
}

TEST(DistributedArray, CrossIssuerArrayDependenciesSurvivePublicationGcAndCheckpoint) {
  const auto lane = pops::ExecutionLane::world();
  auto first = std::make_shared<d::Registry>(lane), second = std::make_shared<d::Registry>(lane);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1);
  offsets.front() = 0;
  auto da = first->issue(91, 1, offsets), db = second->issue(92, 1, offsets);
  const T integer{K::Int64};
  auto child = d::map<2>(*second, db, integer, integer, 65536, [](d::Index) {
                 return V::variant(1, V::integer(313));
               }).value;
  T inner{K::DistributedArray, {integer}};
  inner.family = 92;
  inner.capacity = 1;
  p::ContinuationTransportSession<2> transport;
  bind_registry(transport, *first);
  bind_registry(transport, *second);
  auto parent =
      d::map<2>(
          *first, da, inner, integer, 65536,
          [&](d::Index) { return V::variant(1, V::distributed_array(child)); }, &transport)
          .value;
  ASSERT_TRUE(parent);
  child.reset();
  d::collect_garbage<2>(*first);
  d::collect_garbage<2>(*second);
  EXPECT_EQ(first->retained_values(), 1u);
  EXPECT_EQ(second->retained_values(), 1u);  // Reference exists on rank0 only.
  const auto root = V::distributed_array(parent);
  const std::vector<std::shared_ptr<d::Registry>> registries{first, second};
  auto closure = p::continuation_collective_closure<2>(root, first, lane, 65536, {}, registries);
  EXPECT_EQ(closure->arrays.size(), 2u);
  T outer{K::DistributedArray, {inner}};
  outer.family = 91;
  outer.capacity = 1;
  auto prototypes = [](int) -> const pops::MultiFab<2>& { throw std::logic_error("no field"); };
  p::ContinuationCodec<2> codec(outer, "cross-issuer", 65536, prototypes, &lane, first.get(),
                                closure.get());
  auto decoded = codec.decode(codec.encode(root, 3));
  const auto restored = decoded.root->distributed_array_ref();
  if (!restored->values.empty())
    EXPECT_EQ(restored->values[0]->distributed_array_ref()->values[0]->integer(), 313);
  auto checked = p::continuation_collective_closure<2>(
      decoded.root, decoded.distributed_registry, lane, 65536, {}, decoded.closure->registries);
  EXPECT_EQ(checked->arrays, decoded.closure->arrays);
}

TEST(DistributedArray, BalancedPartitionExactArithmeticAndBudget) {
  EXPECT_EQ(d::partition_balanced(0, 3, 32), (std::vector<std::int64_t>{0, 0, 0, 0}));
  EXPECT_EQ(d::partition_balanced(2, 3, 32), (std::vector<std::int64_t>{0, 1, 2, 2}));
  EXPECT_EQ(d::partition_balanced(8, 3, 32), (std::vector<std::int64_t>{0, 3, 6, 8}));
  const auto large_size = std::numeric_limits<std::int64_t>::max();
  const auto offsets = d::partition_balanced(large_size, 3, 32);
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), large_size);
  EXPECT_EQ(offsets[1], large_size / 3 + 1);
  EXPECT_EQ(offsets[2], 2 * (large_size / 3) + 1);
  EXPECT_THROW(d::partition_balanced(2, 3, 31), d::Failure);
  EXPECT_THROW(d::partition_balanced(-1, 3, 32), d::Failure);
  EXPECT_THROW(d::partition_balanced(2, 0, 32), d::Failure);
  EXPECT_THROW(d::partition_balanced(2, -1, 32), d::Failure);
}

TEST(DistributedArray, BalancedDomainUsesActualLaneAndRejectsBeforePublication) {
  const auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  for (const std::int64_t size :
       {std::int64_t(0), std::int64_t(1), std::int64_t(2 * lane.size() + 1)}) {
    auto domain = registry.issue_balanced(1, 100, size);
    EXPECT_EQ(domain->size(), size);
    EXPECT_EQ(domain->local_size(), size / lane.size() + (lane.rank() < size % lane.size()));
    const T integer{K::Int64};
    auto values = d::map<2>(registry, domain, integer, integer, 65536,
                            [](d::Index i) { return V::variant(1, V::integer(large + i.gid)); });
    check(lane, values.ok());
    for (std::size_t j = 0; j < values.value->values.size(); ++j)
      EXPECT_EQ(values.value->values[j]->integer(), large + domain->begin() + std::int64_t(j));
  }
  const auto before_domains = registry.retained_domains(), before_bytes = registry.retained_bytes();
  rejects(lane, d::Code::domain, [&] { registry.issue_balanced(1, 2, 3); });
  rejects(lane, d::Code::domain, [&] { registry.issue_balanced(1, 2, lane.rank() ? 1 : -1); });
  if (lane.size() > 1)
    rejects(lane, d::Code::domain, [&] { registry.issue_balanced(1, 2, lane.rank() ? 1 : 0); });
  EXPECT_EQ(registry.retained_domains(), before_domains);
  EXPECT_EQ(registry.retained_bytes(), before_bytes);
  d::Registry small(lane);
  small.configure(d::Registry::domain_charge(std::size_t(lane.size()) + 1) - 1, 10);
  rejects(lane, d::Code::capacity, [&] { small.issue_balanced(1, 2, 1); });
  EXPECT_EQ(small.retained_domains(), 0u);
  EXPECT_EQ(small.retained_bytes(), 0u);
}
