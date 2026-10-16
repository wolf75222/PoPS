#include <pops/runtime/program/continuation_distributed_migration.hpp>
#include <pops/runtime/program/continuation_closure.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
namespace p = pops::runtime::program;
namespace d = p::distributed;
using Codec = p::ContinuationCodec<2>;
using Value = p::ContinuationValue<2>;
using Type = p::ContinuationType;
using Kind = p::ContinuationKind;
constexpr std::uint64_t capacity = 1 << 20;
constexpr std::int64_t cardinality = 7;
constexpr std::int64_t exact = (std::int64_t{1} << 60) + 17;

Type element_type() {
  Type index{Kind::DistributedIndex};
  index.family = 801;
  index.capacity = cardinality;
  return {Kind::Product,
          {Type{Kind::Int64}, Type{Kind::Float64}, index},
          {"integer", "signed_zero", "index"}};
}
Type root_type() {
  Type array{Kind::DistributedArray, {element_type()}};
  array.family = 801;
  array.capacity = cardinality;
  Type domain{Kind::DistributedDomain};
  domain.family = 801;
  domain.capacity = cardinality;
  return {Kind::Product, {domain, array, array}, {"domain", "first", "alias"}};
}
auto prototypes() {
  return [](int) -> const pops::MultiFab<2>& { throw std::logic_error("field-free archive test"); };
}
std::vector<std::uint8_t> make_source(const pops::ExecutionLane& lane) {
  auto registry = std::make_shared<d::Registry>(lane);
  registry->configure(capacity, 1000);
  std::vector<std::int64_t> offsets(lane.size() + 1, cardinality);
  offsets.front() = 0;  // all GIDs initially on rank zero: target balanced placement is different
  const auto domain = registry->issue(801, cardinality, offsets);
  p::ContinuationTransportSession<2> transport;
  transport.bind_distributed_registry(*registry,
                                      [&](d::Identity id, d::DomainRef owner, const Type& type) {
                                        return d::resolve_array<2>(*registry, id, owner, type);
                                      });
  const auto array =
      d::map<2>(
          *registry, domain, element_type(), Type{Kind::Int64}, capacity,
          [&](d::Index i) {
            return Value::variant(
                1,
                Value::product({Value::integer(exact + i.gid), Value::real(i.gid % 2 ? -0.0 : 0.0),
                                Value::distributed_index(domain, cardinality - 1 - i.gid)}));
          },
          &transport)
          .value;
  const auto root =
      Value::product({Value::distributed_domain(domain), Value::distributed_array(array),
                      Value::distributed_array(array)});
  auto closure = p::continuation_collective_closure<2>(root, registry, lane, capacity);
  Codec codec(root_type(), "distributed-migration-native", capacity, prototypes(), &lane,
              registry.get(), closure.get());
  return codec.encode(root, 9);
}
std::vector<std::vector<std::uint8_t>> gather(const std::vector<std::uint8_t>& local,
                                              const pops::ExecutionLane& lane) {
  std::vector<std::vector<std::uint8_t>> all;
  d::phase(lane, [&] { all.resize(lane.size()); });
  for (int rank = 0; rank < lane.size(); ++rank) {
    std::uint64_t bytes = lane.rank() == rank ? local.size() : 0;
#ifdef POPS_HAS_MPI
    d::mpi_status(MPI_Bcast(&bytes, 1, MPI_UINT64_T, rank, lane.native_handle()));
#endif
    d::phase(lane, [&] {
      if (bytes > capacity)
        throw std::length_error("test archive source exceeds installed capacity");
      all[rank].resize(bytes);
      if (rank == lane.rank())
        all[rank] = local;
    });
    d::broadcast(all[rank], rank, lane);
  }
  return all;
}
struct Parsed {
  p::ContinuationMigrationBudget budget{128 * capacity};
  Codec::DecodeContext context{128 * capacity};
  Codec codec{root_type(), "distributed-migration-native",
              capacity,    prototypes(),
              nullptr,     nullptr,
              nullptr,     &context};
  std::vector<Codec::ArchiveImage> images;
  std::vector<p::ContinuationDistributedMigration<2>::Input> inputs;
  void read(const std::vector<std::vector<std::uint8_t>>& all) {
    images.reserve(all.size());
    inputs.reserve(all.size());
    for (std::size_t rank = 0; rank < all.size(); ++rank) {
      images.push_back(codec.decode_archive(all[rank], static_cast<int>(rank),
                                            static_cast<int>(all.size()),
                                            [&](std::uint64_t bytes) { budget.charge(bytes); }));
      inputs.push_back({&images.back(), &codec, capacity, capacity, 1000});
    }
  }
};
void verify(const Value::Ref& root, const pops::ExecutionLane& lane) {
  const auto& domain = root->members()[0]->distributed_domain_ref();
  const auto& array = root->members()[1]->distributed_array_ref();
  ASSERT_EQ(array, root->members()[2]->distributed_array_ref());
  ASSERT_EQ(domain, array->domain);
  EXPECT_EQ(domain->partition_offsets(), d::partition_balanced(cardinality, lane.size(), capacity));
  for (std::int64_t gid = domain->begin(); gid < domain->end(); ++gid) {
    const auto& value = array->values.at(static_cast<std::size_t>(gid - domain->begin()));
    EXPECT_EQ(value->members()[0]->integer(), exact + gid);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value->members()[1]->real()),
              std::bit_cast<std::uint64_t>(gid % 2 ? -0.0 : 0.0));
    EXPECT_EQ(value->members()[2]->distributed_domain_ref(), domain);
    EXPECT_EQ(value->members()[2]->distributed_index_offset(), cardinality - 1 - gid);
  }
}

TEST(ContinuationDistributedMigration, InertArchiveMatchesNativeChargesAndCommonIdentity) {
  const auto lane = pops::ExecutionLane::world();
  const auto bytes = make_source(lane);
  Parsed parsed;
  auto image = parsed.codec.decode_archive(bytes, lane.rank(), lane.size(),
                                           [&](std::uint64_t n) { parsed.budget.charge(n); });
  ASSERT_EQ(image.arrays.size(), 1u);
  ASSERT_EQ(image.registries.size(), 1u);
  // Exact native decoder on the real source lane is an independent authority/charge oracle.
  Codec native(root_type(), "distributed-migration-native", capacity, prototypes(), &lane);
  auto decoded = native.decode(bytes);
  EXPECT_EQ(image.arrays[0].charge,
            decoded.distributed_registry->value_charge(decoded.closure->arrays[0]->version,
                                                       d::array_type_tag<2>()));
  EXPECT_EQ(parsed.codec.archive_collective_identity(
                image, capacity, [&](std::uint64_t n) { parsed.budget.charge(n); }),
            native.collective_identity(decoded.root, decoded.publication));
  image.arrays[0].charge += 1;
  EXPECT_THROW(parsed.codec.validate_archive_charges(
                   image, [&](std::uint64_t n) { parsed.budget.charge(n); }),
               std::invalid_argument);
}

TEST(ContinuationDistributedMigration, ReissuesEveryAliasAndPreservesGlobalGidsAndBits) {
  const auto lane = pops::ExecutionLane::world();
  const auto bytes = make_source(lane);
  const auto all = gather(bytes, lane);
  Parsed parsed;
  d::phase(lane, [&] { parsed.read(all); });
  p::ContinuationDistributedMigration<2> plan(parsed.budget);
  d::phase(lane, [&] { plan.prepare(parsed.inputs); });
  plan.materialize(lane);
  auto root = plan.value(parsed.images.front(), parsed.images.front().root);
  verify(root, lane);
  auto registry = plan.primary(parsed.images.front());
  ASSERT_TRUE(registry);
  EXPECT_NE(registry->issuer(), parsed.images.front().registries[0].issuer);
  EXPECT_THROW(registry->resolve_existing(parsed.images.front().domains[0].identity), d::Failure);
  for (const auto& image : parsed.images) {
    const auto alias = plan.value(image, image.root);
    EXPECT_EQ(alias->members()[1]->distributed_array_ref(),
              root->members()[1]->distributed_array_ref());
  }
  auto closure =
      p::continuation_collective_closure<2>(root, registry, lane, capacity, {}, plan.authorities());
  Codec target(root_type(), "distributed-migration-native", capacity, prototypes(), &lane,
               registry.get(), closure.get());
  const auto checkpoint = target.encode(root, 9);
  const auto restarted = target.decode(checkpoint);
  verify(restarted.root, lane);
  EXPECT_EQ(restarted.publication, 9u);
}

TEST(ContinuationDistributedMigration, RejectsMissingShardsAndWorkspaceBeforeTargetPublication) {
  const auto lane = pops::ExecutionLane::world();
  const auto bytes = make_source(lane);
  Parsed parsed;
  parsed.read(gather(bytes, lane));
  p::ContinuationMigrationBudget tiny(1);
  p::ContinuationDistributedMigration<2> no_space(tiny);
  EXPECT_THROW(no_space.prepare(parsed.inputs), std::length_error);
  EXPECT_TRUE(no_space.authorities().empty());
  if (lane.size() > 1) {
    p::ContinuationDistributedMigration<2> incomplete(parsed.budget);
    EXPECT_THROW(incomplete.prepare(std::span(parsed.inputs).first(parsed.inputs.size() - 1)),
                 std::invalid_argument);
    EXPECT_TRUE(incomplete.authorities().empty());
  }
  EXPECT_EQ(make_source(lane).size(), bytes.size());
}
}  // namespace
