#include <chrono>
#include <gtest/gtest.h>
#include <pops/runtime/program/distributed_array.hpp>

namespace {
namespace d = pops::runtime::program::distributed;
namespace p = pops::runtime::program;
using V = p::ContinuationValue<2>;
using Ref = p::ContinuationValueRef<2>;
using A = d::Array<2>;
using AR = d::ArrayRef<2>;
using T = p::ContinuationType;
using K = p::ContinuationKind;
std::vector<std::int64_t> offsets(const pops::ExecutionLane& lane) {
  std::vector<std::int64_t> out(lane.size() + 1, 1);
  out.front() = 0;
  out.back() = 4;
  return out;
}
AR make(d::Registry& registry, const d::DomainRef& domain, std::int64_t shift = 0) {
  T integer{K::Int64};
  auto result = d::map<2>(registry, domain, integer, integer, 4096,
                          [&](d::Index i) { return V::variant(1, V::integer(shift + i.gid)); });
  d::phase(registry.lane(), [&] { d::local_require(result.ok(), d::Code::internal); });
  return std::move(result.value);
}

TEST(DistributedGcV14, StructuralMemoChecksRequestedBytesBeforeAllocation) {
  struct Upstream final : std::pmr::memory_resource {
    int allocations = 0;
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
      ++allocations;
      return std::pmr::new_delete_resource()->allocate(bytes, alignment);
    }
    void do_deallocate(void* data, std::size_t bytes, std::size_t alignment) override {
      std::pmr::new_delete_resource()->deallocate(data, bytes, alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  } upstream;
  p::continuation_detail::StructuralWorkspaceResource workspace(16, &upstream);
  EXPECT_THROW((void)workspace.allocate(17), std::length_error);
  EXPECT_EQ(upstream.allocations, 0);
  EXPECT_EQ(workspace.used(), 0);
  auto* data = workspace.allocate(16);
  EXPECT_EQ(workspace.used(), 16);
  EXPECT_THROW((void)workspace.allocate(1), std::length_error);
  EXPECT_EQ(upstream.allocations, 1);
  EXPECT_EQ(workspace.used(), 16);
  workspace.deallocate(data, 16);
  EXPECT_EQ(workspace.used(), 0);
  {
    std::pmr::unordered_set<const void*> memo(&workspace);
    EXPECT_THROW(memo.insert(&upstream), std::length_error);
    EXPECT_TRUE(memo.empty());
  }
  EXPECT_EQ(workspace.used(), 0);
}

TEST(DistributedGcV14, RepeatedStepsReuseBudgetAndProtectRollbackAndFailure) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 3);
  auto domain = registry.issue(741, 4, offsets(lane));
  AR accepted;
  Ref failure;
  std::uint64_t previous = 0;
  for (int step = 0; step < 32; ++step) {
    d::collect_garbage<2>(registry);
    {
      auto rollback = accepted;
      auto next = make(registry, domain, 100 * step);
      EXPECT_GT(next->version.ordinal, previous);
      previous = next->version.ordinal;
      accepted = next;
      if (step == 0)
        failure = V::distributed_array(accepted);
      if (step == 8)
        failure.reset();
      // Collect with the rollback snapshot and new value simultaneously live.
      d::collect_garbage<2>(registry);
      if (rollback)
        for (std::size_t i = 0; i < rollback->values.size(); ++i)
          EXPECT_EQ(rollback->values[i]->integer(),
                    100 * (step - 1) + domain->begin() + std::int64_t(i));
    }
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_values(), std::uint64_t(step > 0 && step < 8 ? 2 : 1));
    const auto expected =
        d::Registry::domain_charge(domain->partition_offsets().size()) +
        registry.value_charge(accepted->version, d::array_type_tag<2>()) +
        (step > 0 && step < 8 ? registry.value_charge(failure->distributed_array_ref()->version,
                                                      d::array_type_tag<2>())
                              : 0);
    EXPECT_EQ(registry.retained_bytes(), expected);
  }
  accepted.reset();
  failure.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 0);
  EXPECT_EQ(registry.retained_domains(), 1);
  domain.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_domains(), 0);
  EXPECT_EQ(registry.retained_bytes(), 0);
}

TEST(DistributedGcV14, RemoteOnlyRootAndFailureRetainEveryShard) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  auto domain = registry.issue(742, 4, offsets(lane));
  auto source = make(registry, domain, 30);
  const auto source_id = source->version;
  T integer{K::Int64};
  T array{K::DistributedArray, {integer}, {}, 742, 4};
  T element{K::Variant, {integer, array}, {"empty", "reference"}};
  p::ContinuationTransportSession<2> session;
  session.bind_distributed_registry(registry, [&](auto id, const auto& dom, const auto& t) {
    return d::resolve_array<2>(registry, id, dom, t);
  });
  auto made = d::map<2>(
      registry, domain, element, integer, 4096,
      [&](d::Index i) {
        return V::variant(1, i.gid == 3 ? V::variant(1, V::distributed_array(source))
                                        : V::variant(0, V::integer(i.gid)));
      },
      &session);
  d::phase(lane, [&] { d::local_require(made.ok(), d::Code::internal); });
  Ref root = lane.rank() == 0 ? V::distributed_array(made.value) : Ref{};
  made.value.reset();
  source.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 2);
  Ref failure;
  if (lane.rank() == lane.size() - 1)
    failure = V::distributed_array(d::resolve_array<2>(registry, source_id, domain, integer));
  root.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 1);
  {
    const auto value = d::resolve_array<2>(registry, source_id, domain, integer);
    for (std::size_t i = 0; i < value->values.size(); ++i)
      EXPECT_EQ(value->values[i]->integer(), 30 + domain->begin() + std::int64_t(i));
  }
  failure.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 0);
}

TEST(DistributedGcV14, SharedProductEdgesAreCountedOncePerOwnershipEdge) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  auto domain = registry.issue(743, 4, offsets(lane));
  auto source = make(registry, domain, 40);
  T integer{K::Int64};
  T array{K::DistributedArray, {integer}, {}, 743, 4};
  T product{K::Product, {array, array}, {"left", "right"}};
  auto shared = V::distributed_array(source);
  auto tuple = V::product({shared, shared});
  p::ContinuationTransportSession<2> session;
  session.bind_distributed_registry(registry, [&](auto id, const auto& dom, const auto& t) {
    return d::resolve_array<2>(registry, id, dom, t);
  });
  auto made = d::map<2>(
      registry, domain, product, integer, 4096, [&](d::Index) { return V::variant(1, tuple); },
      &session);
  d::phase(lane, [&] { d::local_require(made.ok(), d::Code::internal); });
  source.reset();
  shared.reset();
  made.value.reset();
  if (lane.rank() != 0)
    tuple.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 1);
  if (tuple)
    EXPECT_EQ(tuple->members()[0], tuple->members()[1]);
  tuple.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 0);
}

TEST(DistributedGcV14, CorruptNativeCycleIsRejectedBeforeAnySweep) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  auto domain = registry.issue(744, 4, offsets(lane));
  auto pending = std::make_shared<A>(A{domain, T{K::Int64}, {}});
  for (auto gid = domain->begin(); gid < domain->end(); ++gid)
    pending->values.push_back(V::integer(gid));
  // Deliberately bypass immutable publication through the raw native Registry.
  pending->version = registry.publish_value(pending, d::array_type_tag<2>(), 4096);
  AR published = pending;
  EXPECT_FALSE(published->has_immutable_scalar_values());
  // A prior successful closure must never cache a verdict into this collection.
  auto root = V::distributed_array(published);
  EXPECT_NO_THROW(p::continuation_collective_closure<2>(root, &registry, lane, 65536));
  // A mutable raw native entry must be inspected again even after a successful
  // closure. Its declared scalar element type cannot certify the actual graph.
  if (!pending->values.empty())
    pending->values[0] = V::distributed_array(published);
  const auto bytes = registry.retained_bytes();
  EXPECT_THROW(d::collect_garbage<2>(registry), d::Failure);
  EXPECT_EQ(registry.retained_values(), 1);
  EXPECT_EQ(registry.retained_bytes(), bytes);
  if (!pending->values.empty())
    pending->values[0] = V::integer(domain->begin());
  pending.reset();
  published.reset();
  root.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 0);
}

TEST(DistributedGcV14, ImmutablePublicationIsolatesMutablePendingAndDropsCopiedProof) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  auto domain = registry.issue(948, 4, offsets(lane));
  auto pending = std::make_shared<A>(A{domain, T{K::Int64}, {}});
  for (auto gid = domain->begin(); gid < domain->end(); ++gid)
    pending->values.push_back(V::integer(gid));
  auto published = d::publish_array<2>(registry, pending, 4096, nullptr);
  EXPECT_NE(published.get(), pending.get());
  EXPECT_TRUE(published->has_immutable_scalar_values());
  EXPECT_EQ(pending->version.ordinal, 0u);
  const auto bytes = registry.retained_bytes();
  if (!pending->values.empty()) {
    // Neither retained vector references nor the pending object can mutate the
    // immutable object. A distributed edge added here is not a published cycle.
    auto& old_element = pending->values.front();
    old_element = V::distributed_array(published);
    EXPECT_EQ(published->values.front()->integer(), domain->begin());
    EXPECT_NE(published->values.data(), pending->values.data());
  }
  A copied = *published;
  A moved = std::move(copied);
  A assigned;
  assigned = *published;
  A move_assigned;
  move_assigned = std::move(assigned);
  EXPECT_FALSE(copied.has_immutable_scalar_values());
  EXPECT_FALSE(moved.has_immutable_scalar_values());
  EXPECT_FALSE(assigned.has_immutable_scalar_values());
  EXPECT_FALSE(move_assigned.has_immutable_scalar_values());
  // Even directly copying/moving the opaque proof cannot install it on staging.
  pending->publication_proof = published->publication_proof;
  EXPECT_FALSE(pending->has_immutable_scalar_values());
  EXPECT_NO_THROW(d::collect_garbage<2>(registry));
  EXPECT_EQ(registry.retained_values(), 1u);
  EXPECT_EQ(registry.retained_bytes(), bytes);
  if (!pending->values.empty())
    pending->values.front() = V::integer(domain->begin());
  pending.reset();
  published.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 0u);
}

TEST(DistributedGcV14, ImmutableSummaryUsesActualLeavesAndKeepsResourceCharges) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  std::vector<std::int64_t> partition(static_cast<std::size_t>(lane.size()) + 1);
  for (int rank = 0; rank <= lane.size(); ++rank)
    partition[static_cast<std::size_t>(rank)] = 256 * rank;
  auto domain = registry.issue(949, partition.back(), partition);
  A staging{domain, T{K::Int64}, {}};
  staging.values.assign(256, V::integer(1));
  auto sealed = A::immutable_publication(staging, {registry.issuer(), 1}, 4096);
  EXPECT_TRUE(sealed->has_immutable_scalar_values());
  EXPECT_THROW(A::immutable_publication(staging, {registry.issuer(), 1},
                                        staging.values.size() * sizeof(Ref) - 1),
               std::length_error);
  staging.values.front() = V::distributed_domain(domain);
  auto opaque = A::immutable_publication(staging, {registry.issuer(), 1}, 4096);
  EXPECT_FALSE(opaque->has_immutable_scalar_values());
  staging.values.front().reset();
  EXPECT_FALSE(A::immutable_publication(staging, {registry.issuer(), 1}, 4096)
                   ->has_immutable_scalar_values());
  // Skipping payload visits never skips their declared work-resource charge.
  d::gc_detail::Census<2> bounded(128, registry.issuer());
  bounded.register_domain(domain->identity().ordinal, domain);
  bounded.register_array(1, sealed, d::array_type_tag<2>());
  EXPECT_THROW(bounded.complete(), d::Failure);
}

TEST(DistributedGcV14, ImmutableFactoryFailureIsCollectiveAndDoesNotPublish) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  auto domain = registry.issue(950, 4, offsets(lane));
  auto accepted = make(registry, domain);
  const auto bytes = registry.retained_bytes();
  EXPECT_THROW(registry.publish_value_with_identity(
                   [&](d::Identity identity) -> std::shared_ptr<const void> {
                     if (lane.rank() == 0)
                       throw d::Failure(d::Code::capacity);
                     return A::immutable_publication(*accepted, identity, 4096);
                   },
                   d::array_type_tag<2>(), 1024),
               d::Failure);
  EXPECT_EQ(registry.retained_values(), 1u);
  EXPECT_EQ(registry.retained_bytes(), bytes);
  auto next = make(registry, domain);
  EXPECT_EQ(next->version.ordinal, accepted->version.ordinal + 2);
}

TEST(DistributedGcV14, OneRankPreparationFailureLeavesEveryVaultUnchanged) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 4);
  auto domain = registry.issue(745, 4, offsets(lane));
  auto dead = make(registry, domain, 50);
  dead.reset();
  auto pending = std::make_shared<A>(A{domain, T{K::Int64}, {}});
  for (auto gid = domain->begin(); gid < domain->end(); ++gid)
    pending->values.push_back(V::integer(gid));
  pending->version = registry.publish_value(
      pending, lane.rank() == 0 ? "unsupported-test-value" : d::array_type_tag<2>(), 1024);
  const auto bytes = registry.retained_bytes();
  const auto count = registry.retained_values();
  EXPECT_THROW(d::collect_garbage<2>(registry), d::Failure);
  EXPECT_EQ(registry.retained_values(), count);
  EXPECT_EQ(registry.retained_bytes(), bytes);
}
TEST(DistributedGcV14, DeepRemoteRootAndSharedEdgesSurviveClosureAndCollectExactly) {
  constexpr std::size_t depth = 1024;
  constexpr std::uint64_t capacity = 4 * 1024 * 1024;
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(32 * 1024 * 1024, 8);
  auto domain = registry.issue(946, 4, offsets(lane));
  // Keep a real distributed edge at the deep leaf. A scalar-only immutable
  // Product is now proved irrelevant to registry ownership at construction.
  T type{K::DistributedIndex, {}, {}, 946, 4};
  Ref payload = V::distributed_index(domain, 3);
  for (std::size_t i = 0; i < depth; ++i) {
    T parent;
    parent.names = {"value"};
    parent.children.push_back(std::move(type));
    type = std::move(parent);
    payload = V::product({std::move(payload)});
  }
  // Each local slot owns the same immutable DAG; these are internal edges, not
  // independent external roots. Empty ranks still share the global liveness.
  auto pending = std::make_shared<A>();
  pending->domain = domain;
  pending->element = type;
  pending->values.assign(static_cast<std::size_t>(domain->local_size()), payload);
  auto published = d::publish_array<2>(registry, pending, capacity, nullptr);
  Ref root = lane.rank() == 0 ? V::distributed_array(published) : Ref{};
  const auto version = published->version;
  pending.reset();
  published.reset();
  payload.reset();
  const auto bytes = registry.retained_bytes();
  {
    d::gc_detail::Census<2> small(256, registry.issuer());
    const auto held = registry.resolve_value(version, d::array_type_tag<2>());
    small.register_domain(domain->identity().ordinal, domain);
    small.register_array(version.ordinal, held, d::array_type_tag<2>());
    if (domain->local_size() != 0) {
      try {
        small.complete();
        FAIL() << "deep census unexpectedly fit a 256-unit resource budget";
      } catch (const d::Failure& rejected) {
        EXPECT_EQ(rejected.code, d::Code::capacity);
      }
    }
  }
  EXPECT_EQ(registry.retained_bytes(), bytes);
  EXPECT_EQ(registry.retained_values(), 1u);
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 1u);
  EXPECT_EQ(registry.retained_bytes(), bytes);
  {
    const auto held = d::resolve_array<2>(registry, version, domain, type);
    for (auto value : held->values) {
      for (std::size_t i = 0; i < depth; ++i)
        value = value->members().at(0);
      EXPECT_EQ(value->distributed_index_offset(), 3);
    }
  }
  root.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(), 0u);
  domain.reset();
  d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_domains(), 0u);
  EXPECT_EQ(registry.retained_bytes(), 0u);
}

TEST(DistributedGcV14, ImmutableScalarProductCensusDoesNotVisitElementPayloads) {
  auto lane = pops::ExecutionLane::world();
  constexpr std::uint64_t storage = 64 * 1024 * 1024;
  constexpr int versions = 4;
  const T scalar{K::Float64};
  const T element{K::Product, {scalar, scalar, scalar}, {"x", "y", "z"}};
  const T error{K::Int64};
  for (const std::int64_t count : {4096, 8192}) {
    d::Registry registry(lane);
    registry.configure(storage, versions + 1);
    std::vector<std::int64_t> partition(static_cast<std::size_t>(lane.size()) + 1);
    for (int rank = 0; rank <= lane.size(); ++rank)
      partition[static_cast<std::size_t>(rank)] = count * rank / lane.size();
    auto domain = registry.issue(947, count, partition);
    std::vector<AR> retained;
    for (int version = 0; version < versions; ++version) {
      auto made = d::map<2>(registry, domain, element, error, storage, [&](d::Index index) {
        return V::variant(1, V::product({V::real(index.gid + version), V::real(2), V::real(3)}));
      });
      d::phase(lane, [&] { d::local_require(made.ok(), d::Code::internal); });
      retained.push_back(std::move(made.value));
    }
    std::uint64_t inspected = 0, scalar_values = 0, ownership_nodes = 0;
    d::phase(lane, [&] {
      d::gc_detail::Census<2> census(registry.storage_bytes(), registry.issuer());
      // These are exactly the one Domain and four Arrays published above.
      // This census measures visited edges only; real root discovery/sweep is
      // exercised below by collect_garbage, with the registry's borrowed
      // handles.
      census.register_domain(domain->identity().ordinal, domain);
      for (const auto& array : retained)
        census.register_array(array->version.ordinal, array, d::array_type_tag<2>());
      census.complete();
      inspected = census.inspected_array_values();
      scalar_values = census.scalar_values();
      ownership_nodes = census.ownership_value_nodes();
    });
    EXPECT_EQ(inspected, 0u);
    EXPECT_EQ(scalar_values, static_cast<std::uint64_t>(domain->local_size()) * versions);
    EXPECT_EQ(ownership_nodes, 0u);
    const auto bytes = registry.retained_bytes();
    const auto begin = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 4; ++repeat)
      d::collect_garbage<2>(registry);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    RecordProperty("gc_seconds_n" + std::to_string(count), std::to_string(elapsed));
    RecordProperty("census_edges_n" + std::to_string(count), std::to_string(inspected));
    EXPECT_EQ(registry.retained_values(), versions);
    EXPECT_EQ(registry.retained_bytes(), bytes);
    // A remote-only root still retains every shard; scalar pruning cannot
    // replace collective reachability with rank-local reference counting.
    if (lane.rank() != 0)
      retained.clear();
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_values(), versions);
    retained.clear();
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_values(), 0u);
    domain.reset();
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_bytes(), 0u);
  }
}

TEST(DistributedGcV14, ScalarProofIsClosedOverActualValuesAndKeepsOpaqueKinds) {
  auto scalar = V::product({V::integer(1), V::variant(0, V::real(2)), V::boolean(true)});
  EXPECT_TRUE(scalar->is_scalar_structure());
  EXPECT_FALSE(V::product({scalar, Ref{}})->is_scalar_structure());
  EXPECT_FALSE(V::variant(1, Ref{})->is_scalar_structure());
  p::FiniteDomainRegistry finite;
  auto domain = finite.issue(1, 1, 1);
  EXPECT_FALSE(V::domain(domain)->is_scalar_structure());
  EXPECT_FALSE(V::index(domain, 0)->is_scalar_structure());
  EXPECT_FALSE(V::array(domain, {scalar})->is_scalar_structure());
  EXPECT_FALSE(V::product({scalar, V::index(domain, 0)})->is_scalar_structure());
}

TEST(DistributedGcV14, LeafTraversalPreservesCallbacksWithoutFrameStorage) {
  const int leaf = 7;
  std::size_t entered = 0, exited = 0, counted = 0;
  p::continuation_detail::StructuralBudget one_node(1);
  p::continuation_detail::walk_structure(
      &leaf,
      [&](const int&) {
        ++counted;
        return std::size_t(0);
      },
      [](const int&, std::size_t) -> const int* { throw std::logic_error("leaf child"); },
      [&](const int&) {
        ++entered;
        return true;
      },
      [&](const int&) { ++exited; }, one_node);
  EXPECT_EQ(entered, 1u);
  EXPECT_EQ(exited, 1u);
  EXPECT_EQ(counted, 1u);
  p::continuation_detail::StructuralBudget no_work(0);
  EXPECT_THROW(p::continuation_detail::walk_structure(
                   &leaf, [](const int&) { return std::size_t(0); },
                   [](const int&, std::size_t) { return static_cast<const int*>(nullptr); },
                   [](const int&) { return true; }, [](const int&) {}, no_work),
               std::length_error);
}

TEST(DistributedGcV14, SharedIndexDagFitsUniqueEdgeBudgetAndRechecksRevocation) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  registry.configure(65536, 8);
  auto domain = registry.issue(948, 4, offsets(lane));
  auto index = V::distributed_index(domain, 3);
  auto row = V::product(std::vector<Ref>(256, index));
  auto root = V::product(std::vector<Ref>(256, row));
  // 513 unique pointer edges fit; expanding the repeated Product takes 65793
  // visits and fails this unchanged 4096-unit traversal budget.
  const auto closure = p::continuation_collective_closure<2>(root, &registry, lane, 4096);
  ASSERT_EQ(closure->domains.size(), 1u);
  EXPECT_EQ(closure->domains[0], domain);
  EXPECT_TRUE(closure->arrays.empty());
  registry.revoke();
  EXPECT_THROW(p::continuation_collective_closure<2>(root, &registry, lane, 4096), d::Failure);
}

TEST(DistributedGcV14, RequestsKeepIndexOwnershipAndRemoteRootsWithoutLeafFrames) {
  auto lane = pops::ExecutionLane::world();
  constexpr std::uint64_t storage = 64 * 1024 * 1024;
  constexpr int versions = 4;
  for (const std::int64_t count : {4096, 8192}) {
    d::Registry registry(lane);
    registry.configure(storage, versions + 1);
    auto domain = registry.issue_balanced(949, count, count);
    std::vector<d::Requests> retained;
    for (int version = 0; version < versions; ++version) {
      auto requests = d::requests(registry, domain, domain, [&](d::Index index) {
        return d::index(domain, (index.gid + version) % count);
      });
      d::publish_requests<2>(registry, requests, storage, nullptr);
      retained.push_back(std::move(requests));
    }
    d::phase(lane, [&] {
      d::gc_detail::Census<2> census(storage, registry.issuer());
      census.register_domain(domain->identity().ordinal, domain);
      for (const auto& requests : retained) {
        const auto array = d::requests_array<2>(requests);
        census.register_array(array->version.ordinal, array, d::array_type_tag<2>());
      }
      census.complete();
      const auto expected = static_cast<std::uint64_t>(domain->local_size()) * versions;
      EXPECT_EQ(census.inspected_array_values(), expected);
      EXPECT_EQ(census.scalar_values(), 0u);
      EXPECT_EQ(census.ownership_value_nodes(), expected);
    });
    const auto bytes = registry.retained_bytes();
    const auto begin = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 4; ++repeat)
      d::collect_garbage<2>(registry);
    RecordProperty(
        "requests_gc_seconds_n" + std::to_string(count),
        std::to_string(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()));
    EXPECT_EQ(registry.retained_values(), versions);
    EXPECT_EQ(registry.retained_bytes(), bytes);
    if (lane.rank() != 0)
      retained.clear();
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_values(), versions);
    retained.clear();
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_values(), 0u);
    domain.reset();
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_bytes(), 0u);
  }
}

}  // namespace
