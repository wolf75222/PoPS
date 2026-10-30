#include <pops/runtime/program/distributed_array.hpp>
#include <pops/runtime/program/typed_failure.hpp>
#include <gtest/gtest.h>

namespace {
namespace d = pops::runtime::program::distributed;
namespace p = pops::runtime::program;
using V = p::ContinuationValue<2>;
using T = p::ContinuationType;
using K = p::ContinuationKind;

TEST(DistributedAlgebraV14, IssuedDomainRemainsResolvableWithoutLocalUserReference) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1);
  offsets.front() = 0;
  auto domain = registry.issue(714, 1, offsets);
  const auto id = domain->identity();
  std::weak_ptr<const d::Domain> witness = domain;
  domain.reset();
  EXPECT_FALSE(witness.expired());
  auto resolved = registry.resolve_domain(id);
  EXPECT_EQ(resolved, witness.lock());
  EXPECT_EQ(registry.retained_domains(), 1);
  EXPECT_EQ(registry.retained_bytes(), d::Registry::domain_charge(offsets.size()));
  auto restored = d::Registry::restore_local(lane, registry.issuer(), 65536, 4);
  const auto fresh = restored->restore_domain_local(id, 714, 1, offsets);
  EXPECT_NE(fresh, resolved);
  EXPECT_EQ(restored->retained_bytes(), registry.retained_bytes());
  EXPECT_THROW(restored->authenticate(resolved), d::Failure);
  registry.revoke();
  EXPECT_THROW(registry.resolve_domain(id), d::Failure);
  EXPECT_TRUE(fresh->visible());
}

TEST(DistributedAlgebraV14, VersionCapacityFailureRetainsPublishedImmutableData) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 1);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1);
  offsets.front() = 0;
  const auto domain = registry.issue(715, 1, offsets);
  const auto scalar = T{K::Int64};
  auto first = d::map<2>(registry, domain, scalar, scalar, 4096,
                         [](d::Index i) { return V::variant(1, V::integer(80 + i.gid)); });
  d::phase(lane, [&] { d::local_require(first.ok(), d::Code::internal); });
  const auto version = first.value->version;
  const auto bytes = registry.retained_bytes();
  auto refused = d::map<2>(registry, domain, scalar, scalar, 4096,
                           [](d::Index i) { return V::variant(1, V::integer(900 + i.gid)); });
  EXPECT_FALSE(refused.ok());
  EXPECT_EQ(refused.code, d::Code::capacity);
  EXPECT_EQ(registry.retained_values(), 1);
  EXPECT_EQ(registry.retained_bytes(), bytes);
  const auto resolved = d::resolve_array<2>(registry, version, domain, scalar);
  EXPECT_EQ(resolved, first.value);
  for (const auto& value : resolved->values)
    EXPECT_EQ(value->integer(), 80);
}

TEST(DistributedAlgebraV14, TerminalClosureIncludesShardsWithNoLocalRootReference) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1);
  offsets.front() = 0;
  offsets.back() = 4;
  auto domain = registry.issue(716, 4, offsets);
  T integer{K::Int64};
  auto made = d::map<2>(registry, domain, integer, integer, 4096,
                        [](d::Index i) { return V::variant(1, V::integer(100 + i.gid)); });
  d::phase(lane, [&] { d::local_require(made.ok(), d::Code::internal); });
  T array_type{K::DistributedArray, {integer}, {}, 716, 4};
  T failure{K::Variant, {integer, array_type}, {"empty", "refs"}};
  std::string bytes;
  bool caught = false;
  try {
    p::reject_typed_failure<2>(
        lane,
        [&] {
          auto value = lane.rank() == lane.size() - 1
                           ? V::variant(1, V::distributed_array(made.value))
                           : V::variant(0, V::integer(77));
          return std::make_tuple(failure, value,
                                 std::string("distributed-terminal-asymmetric-v14"));
        },
        [](int) -> const pops::MultiFab<2>& { throw std::logic_error("no field prototype"); }, {},
        &registry);
  } catch (const p::StepAttemptRejected& error) {
    caught = true;
    bytes = error.typed_failure_payload();
  }
  d::phase(lane, [&] { d::local_require(caught, d::Code::internal); });
  EXPECT_EQ(bytes.substr(0, 9), "POPSFAIL1");
  std::size_t at = 9;
  auto outer_word = [&]() {
    auto out = p::typed_failure_detail::word(std::string_view(bytes).substr(at, 8));
    at += 8;
    return out;
  };
  EXPECT_EQ(outer_word(), std::uint64_t(lane.size()));
  for (int rank = 0; rank < lane.size(); ++rank) {
    const auto n = outer_word();
    ASSERT_LE(n, bytes.size() - at);
    const auto shard = std::string_view(bytes).substr(at, n);
    at += n;
    std::size_t pos = 0;
    auto word = [&]() {
      if (shard.size() - pos < 8)
        throw std::logic_error("truncated test shard");
      std::uint64_t out = 0;
      for (unsigned i = 0; i < 8; ++i)
        out |= std::uint64_t(static_cast<unsigned char>(shard[pos++])) << (8 * i);
      return out;
    };
    auto text = [&]() {
      const auto count = word();
      if (count > shard.size() - pos)
        throw std::logic_error("truncated test text");
      const auto out = shard.substr(pos, count);
      pos += count;
      return out;
    };
    EXPECT_EQ(text(), "POPSCONT2");
    EXPECT_EQ(text(), "distributed-terminal-asymmetric-v14");
    EXPECT_EQ(word(), 2);
    EXPECT_EQ(word(), 0);
    EXPECT_EQ(word(), 0);
    EXPECT_EQ(word(), 0);
    EXPECT_EQ(word(), 1);
    for (int k = 0; k < 4; ++k)
      (void)word();  // issuer and storage bounds
    EXPECT_EQ(word(), domain->identity().ordinal);
    EXPECT_EQ(word(), 716);
    EXPECT_EQ(word(), 4);
    EXPECT_EQ(word(), std::uint64_t(lane.size() + 1));
    for (auto offset : offsets)
      EXPECT_EQ(word(), offset);
    EXPECT_EQ(word(), 1);
    EXPECT_EQ(word(), 1);
    (void)text();
    EXPECT_EQ(word(), 0);
    EXPECT_EQ(word(), 0);
    EXPECT_GT(word(), 0);
    EXPECT_EQ(word(), std::uint64_t(offsets[rank + 1] - offsets[rank]));
    for (auto gid = offsets[rank]; gid < offsets[rank + 1]; ++gid) {
      EXPECT_EQ(word(), 1);
      EXPECT_EQ(word(), 100 + gid);
    }
    EXPECT_EQ(word(), 4);
    EXPECT_EQ(word(), std::uint64_t(rank == lane.size() - 1));
    EXPECT_EQ(word(), 1);
    if (rank == lane.size() - 1) {
      EXPECT_EQ(word(), 11);
      EXPECT_EQ(word(), 0);
    } else {
      EXPECT_EQ(word(), 1);
      EXPECT_EQ(word(), 77);
    }
    EXPECT_EQ(pos, shard.size());
  }
  EXPECT_EQ(at, bytes.size());
}
}  // namespace
