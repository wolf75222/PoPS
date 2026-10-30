#include <gtest/gtest.h>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/program/continuation_authority.hpp>
#include <pops/runtime/program/distributed_array.hpp>

using namespace pops;
using namespace pops::runtime::program;
namespace {
constexpr int D = kNativeDimension;
using V = ContinuationValue<D>;
using R = ContinuationValueRef<D>;
using Codec = ContinuationCodec<D>;
ContinuationType primitive(ContinuationKind kind) {
  ContinuationType value;
  value.kind = kind;
  return value;
}
ContinuationType record(std::vector<std::string> names, std::vector<ContinuationType> members) {
  ContinuationType value;
  value.names = std::move(names);
  value.children = std::move(members);
  return value;
}
ContinuationType domain() {
  auto value = primitive(ContinuationKind::Domain);
  value.family = 4;
  value.capacity = 8;
  return value;
}
ContinuationType array(ContinuationType element) {
  auto value = domain();
  value.kind = ContinuationKind::Array;
  value.children = {std::move(element)};
  return value;
}
ContinuationType bundle_type() {
  auto index = domain();
  index.kind = ContinuationKind::Index;
  auto values = array(primitive(ContinuationKind::Float64));
  auto out = record(
      {"instance", "fields", "geometry", "maps", "histories"},
      {domain(), record({"q"}, {values}), record({"volume"}, {values}),
       record({"source", "parent"}, {domain(), array(index)}), record({"previous"}, {values})});
  out.dependent_bundle = true;
  out.domain_bindings = {{{0}, {0}},       {{1, 0}, {0}}, {{2, 0}, {0}},
                         {{3, 0}, {3, 0}}, {{3, 1}, {0}}, {{3, 1, continuation_each}, {3, 0}},
                         {{4, 0}, {0}}};
  return out;
}
Codec codec(ContinuationType type = bundle_type(), std::uint64_t capacity = 4096) {
  return Codec(std::move(type), "bundle/test/v14", capacity,
               [](int) -> const MultiFab<D>& { throw std::logic_error("no legacy field"); });
}
R bundle(FiniteDomainRef current, FiniteDomainRef source, FiniteDomainRef history = {}) {
  if (!history)
    history = current;
  std::vector<R> state, volumes, parent, previous;
  for (std::int64_t i = 0; i < current->size; ++i) {
    state.push_back(V::real(3.));
    volumes.push_back(V::real(1. / current->size));
    parent.push_back(V::index(source, 0));
  }
  for (std::int64_t i = 0; i < history->size; ++i)
    previous.push_back(V::real(2.));
  return V::product({V::domain(current), V::product({V::array(current, std::move(state))}),
                     V::product({V::array(current, std::move(volumes))}),
                     V::product({V::domain(source), V::array(current, std::move(parent))}),
                     V::product({V::array(history, std::move(previous))})});
}
}  // namespace

TEST(BundleV14, SameCardinalityDoesNotRetargetAHistory) {
  FiniteDomainRegistry registry;
  auto before = registry.issue(2, 8, 4), after = registry.issue(2, 8, 4);
  auto validator = codec();
  EXPECT_NO_THROW(validator.validate(bundle(after, before)));
  EXPECT_THROW(validator.validate(bundle(after, before, before)), std::invalid_argument);
}

TEST(BundleV14, MissingDependencyIsNotACompleteBundleType) {
  auto type = bundle_type();
  type.domain_bindings.pop_back();
  EXPECT_THROW(codec(type), std::invalid_argument);
  type = bundle_type();
  type.domain_bindings.push_back(type.domain_bindings.front());
  EXPECT_THROW(codec(type), std::invalid_argument);
}

TEST(BundleV14, RemappedGeometryMapsAndHistoryKeepExactAliasesAfterCheckpoint) {
  FiniteDomainRegistry registry;
  auto before = registry.issue(1, 8, 4), after = registry.issue(2, 8, 4);
  auto validator = codec();
  auto restored = validator.decode(validator.encode(bundle(after, before), 17));
  const auto& root = restored.root;
  const auto& current = root->members()[0]->domain_ref();
  const auto& maps = root->members()[3]->members();
  const auto& source = maps[0]->domain_ref();
  EXPECT_NE(current, source);
  EXPECT_EQ(current, root->members()[1]->members()[0]->domain_ref());
  EXPECT_EQ(current, root->members()[2]->members()[0]->domain_ref());
  EXPECT_EQ(current, root->members()[4]->members()[0]->domain_ref());
  EXPECT_EQ(current, maps[1]->domain_ref());
  for (const auto& parent : maps[1]->members())
    EXPECT_EQ(parent->domain_ref(), source);
  EXPECT_EQ(restored.publication, 17u);
}

TEST(BundleV14, RejectedCapacityCannotReplaceThePreviousOwnedImage) {
  FiniteDomainRegistry registry;
  auto before = registry.issue(1, 8, 4), after = registry.issue(8, 8, 4);
  const auto original = bundle(before, before);
  const auto saved = codec().encode(original, 1);
  EXPECT_THROW(codec(bundle_type(), 64).encode(bundle(after, before), 2), std::length_error);
  EXPECT_EQ(codec().encode(original, 1), saved);
}

TEST(BundleV14, TransportInternsAcrossMessagesAndReturnsTheOriginalOwnerAlias) {
  FiniteDomainRegistry owner(false), unrelated(false);
  auto first = owner.issue(2, 8, 4), same_size = unrelated.issue(2, 8, 4);
  auto type = record({"domain", "values"}, {domain(), array(domain())});
  auto packet =
      V::product({V::domain(first), V::array(first, {V::domain(first), V::domain(same_size)})});
  ContinuationTransportSession<D> sender, receiver;
  const auto wire = sender.encode(type, packet, 4096);
  const auto received = receiver.decode(type, wire, 4096);
  const auto again = receiver.decode(type, wire, 4096);
  const auto foreign = received->members()[0]->domain_ref();
  EXPECT_NE(foreign, first);
  EXPECT_EQ(foreign, received->members()[1]->domain_ref());
  EXPECT_EQ(foreign, received->members()[1]->members()[0]->domain_ref());
  EXPECT_NE(foreign, received->members()[1]->members()[1]->domain_ref());
  EXPECT_EQ(foreign, again->members()[0]->domain_ref());
  const auto returned = sender.decode(type, receiver.encode(type, received, 4096), 4096);
  EXPECT_EQ(first, returned->members()[0]->domain_ref());
  EXPECT_EQ(same_size, returned->members()[1]->members()[1]->domain_ref());
  receiver.clear();
  EXPECT_THROW(foreign->require_live(), std::logic_error);
  EXPECT_NO_THROW(first->require_live());
}

TEST(BundleV14, FiniteCheckpointRetainsMultipleLocalIssuersWithoutVersionCollision) {
  FiniteDomainRegistry a(false), b(false);
  const auto left = a.issue(2, 8, 4), right = b.issue(2, 8, 4);
  ASSERT_EQ(left->version, right->version);
  auto type = record({"left", "right"}, {domain(), domain()});
  auto validator = codec(type);
  auto decoded =
      validator.decode(validator.encode(V::product({V::domain(left), V::domain(right)}), 5));
  EXPECT_NE(decoded.root->members()[0]->domain_ref(), decoded.root->members()[1]->domain_ref());
  EXPECT_NE(decoded.root->members()[0]->domain_ref()->realm(),
            decoded.root->members()[1]->domain_ref()->realm());
  EXPECT_EQ(decoded.realms.size(), 2u);
  EXPECT_FALSE(decoded.realm);
}

TEST(BundleV14, RestoredArrayCannotUnderstateItsNativeRetainedByteCharge) {
  namespace dist = pops::runtime::program::distributed;
  const auto lane = ExecutionLane::world();
  dist::Registry registry(lane);
  registry.configure(65536, 8);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1);
  offsets.front() = 0;
  offsets.back() = 4;
  auto domain = registry.issue(741, 4, offsets);
  ContinuationType scalar{ContinuationKind::Int64};
  auto made = dist::map<D>(registry, domain, scalar, scalar, 4096, [](dist::Index index) {
    return V::variant(1, V::integer(10 + index.gid));
  });
  dist::phase(lane, [&] { dist::local_require(made.ok(), dist::Code::internal); });
  auto root = V::distributed_array(made.value);
  const auto closure = continuation_collective_closure<D>(root, &registry, lane, 65536);
  ContinuationType type{ContinuationKind::DistributedArray, {scalar}, {}, 741, 4};
  Codec validator(
      type, "distributed-charge-v14", 65536,
      [](int) -> const MultiFab<D>& { throw std::logic_error("no field prototype"); }, &lane,
      &registry, closure.get());
  const auto bytes = validator.encode(root, 7);
  EXPECT_NO_THROW(validator.decode(bytes));
  const auto retained = registry.retained_bytes();
  auto damaged = bytes;
  continuation_detail::Reader in(damaged);
  EXPECT_EQ(in.text(), "POPSCONT2");
  (void)in.text();
  (void)in.word();
  (void)in.word();
  const auto realms = in.word();
  for (std::uint64_t i = 0; i < 3 * realms; ++i)
    (void)in.word();
  const auto finite = in.word();
  for (std::uint64_t i = 0; i < 5 * finite; ++i)
    (void)in.word();
  const auto domains = in.word();
  if (domains)
    for (int i = 0; i < 4; ++i)
      (void)in.word();
  for (std::uint64_t i = 0; i < domains; ++i) {
    for (int j = 0; j < 3; ++j)
      (void)in.word();
    const auto n = in.word();
    for (std::uint64_t j = 0; j < n; ++j)
      (void)in.word();
  }
  EXPECT_EQ(in.word(), 1u);
  (void)in.word();
  (void)in.text();
  (void)in.word();
  (void)in.word();
  const auto position = in.at;
  const auto charge = in.word();
  for (unsigned byte = 0; byte < 8; ++byte)
    damaged[position + byte] = std::uint8_t((charge - 1) >> (8 * byte));
  bool rejected = false;
  try {
    (void)validator.decode(damaged);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("retained charge differs") != std::string::npos;
  }
  EXPECT_TRUE(rejected);
  EXPECT_EQ(registry.retained_bytes(), retained);
  EXPECT_EQ(validator.encode(root, 7), bytes);
}
