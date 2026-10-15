#include <gtest/gtest.h>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/program/continuation_authority.hpp>
#include <pops/runtime/program/amr_finite_value_lifetime.hpp>
#include <pops/runtime/program/finite_structural_storage.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>

using namespace pops;
using namespace pops::runtime::program;
namespace {
constexpr int D = kNativeDimension;
using V = ContinuationValue<D>;
using R = ContinuationValueRef<D>;
using Codec = ContinuationCodec<D>;
using Authority = ContinuationAuthority<D>;
using Store = ContinuationStore<D>;
ContinuationType scalar(ContinuationKind kind) {
  ContinuationType t;
  t.kind = kind;
  return t;
}
ContinuationType product(std::vector<std::string> names, std::vector<ContinuationType> children) {
  ContinuationType t;
  t.names = std::move(names);
  t.children = std::move(children);
  return t;
}
Codec::Prototypes none() {
  return [](int) -> const MultiFab<D>& { throw std::logic_error("test schema has no field"); };
}
ContinuationConfiguration configuration() {
  ContinuationConfiguration c;
  c.type = product({"time", "step", "counter"},
                   {scalar(ContinuationKind::Float64), scalar(ContinuationKind::Int64),
                    scalar(ContinuationKind::Int64)});
  c.identity = "test/continuation/structural";
  c.byte_capacity = 4096;
  c.time_path = {0};
  c.step_path = {1};
  return c;
}
R root(double time = 0, std::int64_t step = 0, std::int64_t counter = 0) {
  return V::product({V::real(time), V::integer(step), V::integer(counter)});
}
R envelope(R next, std::int64_t increment = 7) {
  return V::product({std::move(next), V::integer(increment), V::product({}), V::boolean(true)});
}
Store initial_store() {
  Store store;
  store.configure(configuration());
  store.publish(store.prepare(root(), none(), true));
  return store;
}
}  // namespace

TEST(ContinuationImage, ExactInt64NanPayloadAndVariantRoundTrip) {
  auto sum = product({"integer", "bits"},
                     {scalar(ContinuationKind::Int64), scalar(ContinuationKind::Float64)});
  sum.kind = ContinuationKind::Variant;
  const auto type = product({"sum", "huge"}, {sum, scalar(ContinuationKind::Int64)});
  Codec codec(type, "test/typed-bits", 4096, none());
  const auto nan = std::bit_cast<double>(UINT64_C(0x7ff8000000000005));
  auto value = V::product({V::variant(1, V::real(nan)), V::integer(INT64_C(9007199254740997))});
  const auto restored = codec.decode(codec.encode(value, 41));
  EXPECT_EQ(restored.publication, 41u);
  EXPECT_EQ(restored.root->members()[1]->integer(), INT64_C(9007199254740997));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(restored.root->members()[0]->members()[0]->real()),
            UINT64_C(0x7ff8000000000005));
}

TEST(ContinuationImage, DomainAliasesSurviveOneTableAndRealmIsRenewed) {
  FiniteDomainRegistry registry;
  auto domain = registry.issue(2, 8, 5);
  auto dt = scalar(ContinuationKind::Domain);
  dt.family = 5;
  dt.capacity = 8;
  auto it = dt;
  it.kind = ContinuationKind::Index;
  auto at = dt;
  at.kind = ContinuationKind::Array;
  at.children = {scalar(ContinuationKind::Int64)};
  Codec codec(product({"domain", "index", "array"}, {dt, it, at}), "test/aliases", 4096, none());
  auto image = V::product(
      {V::domain(domain), V::index(domain, 1), V::array(domain, {V::integer(3), V::integer(9)})});
  auto restored = codec.decode(codec.encode(image, 2));
  const auto a = restored.root->members()[0]->domain_ref();
  EXPECT_EQ(a, restored.root->members()[1]->domain_ref());
  EXPECT_EQ(a, restored.root->members()[2]->domain_ref());
  EXPECT_NE(a, domain);
  EXPECT_NE(a->realm(), domain->realm());
  EXPECT_EQ(a->version, domain->version);
  registry.import_high_water(restored.max_domain_version);
  EXPECT_GT(registry.issue(0, 8, 5, restored.realm)->version, a->version);
}

TEST(ContinuationImage, SynchronizedReplicatedRealmRetainsItsCollectiveIdentity) {
  FiniteDomainRegistry registry;
  registry.synchronize_identity(registry.realm(), {71, 93});
  const auto domain = registry.issue(2, 8, 17);
  auto domain_type = scalar(ContinuationKind::Domain);
  domain_type.family = 17;
  domain_type.capacity = 8;
  auto index_type = domain_type;
  index_type.kind = ContinuationKind::Index;
  const auto type = product({"domain", "index"}, {domain_type, index_type});
  Codec codec(type, "test/replicated-origin", 4096, none());
  const auto value = V::product({V::domain(domain), V::index(domain, 1)});
  const auto identity = codec.collective_identity(value, 7);
  const auto image = codec.encode(value, 7);
  continuation_detail::Reader header(image);
  EXPECT_EQ(header.text(), "POPSCONT2");
  const auto restored = codec.decode(image);
  EXPECT_EQ(restored.realm->transport_identity(), domain->realm()->transport_identity());
  EXPECT_TRUE(restored.realm->synchronized());
  EXPECT_NE(restored.realm, domain->realm());
  EXPECT_EQ(codec.collective_identity(restored.root, restored.publication), identity);
  EXPECT_EQ(codec.encode(restored.root, restored.publication), image);
}

TEST(ContinuationImage, SharedDecodeContextPreservesFiniteAliasesAndRejectsDescriptorChanges) {
  FiniteDomainRegistry first, foreign;
  const auto domain = first.issue(2, 4, 71);
  const auto changed = foreign.issue(3, 4, 71);
  first.synchronize_identity(domain->realm(), {871, 931});
  foreign.synchronize_identity(changed->realm(), {871, 931});
  auto type = scalar(ContinuationKind::Domain);
  type.family = 71;
  type.capacity = 4;
  Codec encoder(type, "shared/finite", 65536, none());
  const auto bytes = encoder.encode(V::domain(domain), 5);
  Codec::DecodeContext shared(65536);
  Codec decoder(type, "shared/finite", 65536, none(), nullptr, nullptr, nullptr, &shared);
  const auto one = decoder.decode(bytes);
  const auto two = decoder.decode(bytes);
  EXPECT_EQ(one.root->domain_ref(), two.root->domain_ref());
  EXPECT_EQ(one.realm, two.realm);
  EXPECT_GT(shared.workspace_used(), 0U);
  EXPECT_THROW(decoder.decode(encoder.encode(V::domain(changed), 5)), std::invalid_argument);
  EXPECT_TRUE(one.root->domain_ref()->visible());
  EXPECT_EQ(encoder.encode(one.root, one.publication), bytes);
}

TEST(ContinuationImage, SharedDecodeContextRejectsRegistryPolicyAndWorkspaceChanges) {
  auto lane = ExecutionLane::duplicate_world_collectively("test.continuation.shared-policy");
  Codec::DecodeContext empty(0);
  EXPECT_THROW(empty.registry(lane, {121, 173}, 65536, 32), std::length_error);
  Codec::DecodeContext shared(65536);
  const auto registry = shared.registry(lane, {121, 173}, 65536, 32);
  EXPECT_EQ(shared.registry(lane, {121, 173}, 65536, 32), registry);
  EXPECT_THROW(shared.registry(lane, {121, 173}, 131072, 32), std::invalid_argument);
  EXPECT_THROW(shared.registry(lane, {121, 173}, 65536, 64), std::invalid_argument);
  EXPECT_EQ(registry->retained_domains(), 0U);
  EXPECT_EQ(registry->retained_values(), 0U);
}

TEST(ContinuationImage, SharedDecodeContextPreservesCrossImageArraysAndRejectsChangedBodies) {
  namespace d = distributed;
  auto lane = ExecutionLane::duplicate_world_collectively("test.continuation.shared-array");
  constexpr std::uint64_t capacity = 1024 * 1024;
  auto registry = d::Registry::restore_local(lane, {181, 193}, capacity, 32);
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(lane.size()) + 1);
  for (std::size_t rank = 0; rank < offsets.size(); ++rank)
    offsets[rank] = static_cast<std::int64_t>(2 * rank);
  const auto domain =
      registry->restore_domain_local({registry->issuer(), 1}, 97, offsets.back(), offsets);
  auto array = std::make_shared<ContinuationDistributedArray<D>>();
  array->domain = domain;
  array->element = scalar(ContinuationKind::Int64);
  array->version = {registry->issuer(), 1};
  const auto marker = INT64_C(0x123456789abcde0) + lane.rank();
  array->values = {V::integer(marker), V::integer(marker + 17)};
  const auto charge = continuation_distributed_array_charge<D>(array, *registry, capacity);
  registry->restore_value_local(array->version, array, d::array_type_tag<D>(), charge);
  ContinuationType array_type{
      ContinuationKind::DistributedArray, {array->element}, {}, 97, domain->capacity()};
  const auto child = V::distributed_array(array);
  const auto parent = V::product({child, V::integer(19)});
  const auto parent_type =
      product({"child", "global"}, {array_type, scalar(ContinuationKind::Int64)});
  const auto closure = continuation_collective_closure<D>(child, registry, lane, capacity);
  Codec parent_encoder(parent_type, "shared/parent", capacity, none(), &lane, registry.get(),
                       closure.get());
  Codec child_encoder(array_type, "shared/child", capacity, none(), &lane, registry.get(),
                      closure.get());
  const auto parent_bytes = parent_encoder.encode(parent, 7);
  const auto child_bytes = child_encoder.encode(child, 11);
  Codec::DecodeContext shared(capacity);
  Codec parent_decoder(parent_type, "shared/parent", capacity, none(), &lane, nullptr, nullptr,
                       &shared);
  Codec child_decoder(array_type, "shared/child", capacity, none(), &lane, nullptr, nullptr,
                      &shared);
  const auto restored_parent = parent_decoder.decode(parent_bytes);
  const auto restored_child = child_decoder.decode(child_bytes);
  EXPECT_EQ(restored_parent.distributed_registry, restored_child.distributed_registry);
  EXPECT_EQ(restored_parent.root->members()[0]->distributed_array_ref(),
            restored_child.root->distributed_array_ref());
  EXPECT_EQ(restored_parent.closure->domains, restored_child.closure->domains);
  EXPECT_EQ(restored_child.distributed_registry->retained_values(), 1U);

  // Change a valid integer payload while preserving its version, type and wire
  // framing. The second image must not redefine an already authenticated Array.
  std::array<std::uint8_t, 8> pattern{};
  for (std::size_t i = 0; i < pattern.size(); ++i)
    pattern[i] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(marker) >> (8 * i));
  auto changed = child_bytes;
  const auto found = std::search(changed.begin(), changed.end(), pattern.begin(), pattern.end());
  ASSERT_NE(found, changed.end());
  ASSERT_EQ(std::search(found + 1, changed.end(), pattern.begin(), pattern.end()), changed.end());
  *found ^= 1;
  EXPECT_THROW(child_decoder.decode(changed), std::invalid_argument);
  EXPECT_EQ(restored_child.root->distributed_array_ref()->values[0]->integer(), marker);
  EXPECT_EQ(restored_child.distributed_registry->retained_values(), 1U);
}

TEST(ContinuationImage, RetainsDistinctRealmAuthoritiesAndRejectsBoundsTypeAndCorruption) {
  FiniteDomainRegistry first, second;
  auto dt = scalar(ContinuationKind::Domain);
  dt.family = 3;
  dt.capacity = 4;
  Codec codec(product({"a", "b"}, {dt, dt}), "test/realm", 4096, none());
  const auto first_domain = first.issue(0, 4, 3), second_domain = second.issue(0, 4, 3);
  const auto mixed = V::product({V::domain(first_domain), V::domain(second_domain)});
  const auto restored = codec.decode(codec.encode(mixed, 0));
  EXPECT_NE(restored.root->members()[0]->domain_ref()->realm(),
            restored.root->members()[1]->domain_ref()->realm());
  EXPECT_EQ(restored.root->members()[0]->domain_ref()->realm()->transport_identity(),
            first_domain->realm()->transport_identity());
  EXPECT_EQ(restored.root->members()[1]->domain_ref()->realm()->transport_identity(),
            second_domain->realm()->transport_identity());
  auto store = initial_store();
  auto bytes = store.checkpoint(none());
  bytes.push_back(0);
  EXPECT_THROW(store.prepare_restart(bytes, none()), std::invalid_argument);
  bytes.pop_back();
  bytes.pop_back();
  EXPECT_THROW(store.prepare_restart(bytes, none()), std::invalid_argument);
  EXPECT_THROW(store.prepare(V::integer(1), none()), std::invalid_argument);
  Codec tiny(configuration().type, "test/continuation/structural", 32, none());
  EXPECT_THROW(tiny.encode(root(), 0), std::length_error);
}

TEST(ContinuationImage, ExtendedLocalRealmsRoundTripAndAmbiguousIdentitiesAreRejected) {
  FiniteDomainRegistry replicated, local_a(false), local_b(false);
  auto dt = scalar(ContinuationKind::Domain);
  dt.family = 3;
  dt.capacity = 4;
  const auto a = local_a.issue(1, 4, 3), b = local_b.issue(2, 4, 3);
  Codec codec(product({"global", "local_a", "local_b", "alias_a"}, {dt, dt, dt, dt}),
              "test/local-realms", 4096, none());
  auto original =
      V::product({V::domain(replicated.issue(0, 4, 3)), V::domain(a), V::domain(b), V::domain(a)});
  const auto restored = codec.decode(codec.encode(original, 7));
  ASSERT_EQ(restored.realms.size(), 3u);
  ASSERT_EQ(restored.root->members().size(), 4u);
  const auto global = restored.root->members()[0]->domain_ref();
  const auto ra = restored.root->members()[1]->domain_ref();
  const auto rb = restored.root->members()[2]->domain_ref();
  EXPECT_EQ(restored.publication, 7u);
  EXPECT_EQ(restored.realm, global->realm());
  EXPECT_TRUE(global->realm()->replicated());
  EXPECT_FALSE(ra->realm()->replicated());
  EXPECT_FALSE(rb->realm()->replicated());
  EXPECT_NE(ra->realm(), rb->realm());
  EXPECT_NE(ra->realm(), a->realm());
  EXPECT_EQ(ra, restored.root->members()[3]->domain_ref());
  EXPECT_EQ(ra->version, rb->version);  // equal ordinals from distinct issuers remain distinct
  EXPECT_EQ(ra->size, 1);
  EXPECT_EQ(rb->size, 2);

  Codec pair(product({"a", "b"}, {dt, dt}), "test/extended-authority-rejection", 4096, none());
  ContinuationTransportSession<D> sender, first_receiver, second_receiver;
  const auto wire = sender.encode(dt, V::domain(a), 4096);
  const auto first = first_receiver.decode(dt, wire, 4096);
  const auto second = second_receiver.decode(dt, wire, 4096);
  ASSERT_NE(first->domain_ref()->realm(), second->domain_ref()->realm());
  ASSERT_EQ(first->domain_ref()->realm()->unexported_identity(),
            second->domain_ref()->realm()->unexported_identity());
  EXPECT_THROW(pair.encode(V::product({first, second}), 0), std::invalid_argument);

  FiniteDomainRegistry another_local(false);
  const auto repeated = another_local.issue(1, 4, 3, a->realm());
  ASSERT_NE(a, repeated);
  ASSERT_EQ(a->realm(), repeated->realm());
  ASSERT_EQ(a->version, repeated->version);
  EXPECT_THROW(pair.encode(V::product({V::domain(a), V::domain(repeated)}), 0),
               std::invalid_argument);
}

TEST(ContinuationImage, StoreSnapshotRollbackAndRestartFrontier) {
  auto store = initial_store();
  auto before = store;
  store.publish(store.prepare(root(.25, 1, 4), none()));
  store.require_frontier(.25, 1);
  EXPECT_THROW(store.require_frontier(0, 0), std::logic_error);
  store.swap(before);
  store.require_frontier(0, 0);
  auto archived = before.checkpoint(none());
  store.publish(store.prepare_restart(archived, none()));
  store.require_frontier(.25, 1);
  EXPECT_EQ(store.read()->members()[2]->integer(), 4);
}

TEST(ContinuationAuthority, RepeatedProducerCannotReuseAssessmentAndTransformInvalidates) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto first = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 1)));
  auto assessment = authority.attest(first, V::boolean(true), true, "graph/check");
  auto second = authority.freeze(base, V::real(.2), envelope(root(.2, 1, 2)));
  EXPECT_THROW(authority.accept(second, assessment), std::logic_error);
  auto accepted = authority.accept(first, assessment);
  authority.supersede(first);
  EXPECT_THROW(accepted.value(), std::logic_error);
  EXPECT_THROW(assessment.accepted(), std::logic_error);
  EXPECT_NO_THROW(second.value());
}

TEST(ContinuationAuthority, ChildCannotRevokeParentAndFailureRevokesChild) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto parent = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 1)));
  Authority::Candidate child;
  {
    auto scope = authority.begin_scope();
    EXPECT_THROW(authority.supersede(parent), std::logic_error);
    child = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 2)));
  }
  EXPECT_THROW(child.value(), std::logic_error);
  EXPECT_NO_THROW(parent.value());
}

TEST(ContinuationAuthority, PromotionKeepsWholeCandidateAndConsumesDeltaOnce) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto scope = authority.begin_scope();
  auto candidate = authority.freeze(base, V::real(.2), envelope(root(.2, 1, 3), 11));
  auto assessment = authority.attest(candidate, V::boolean(true), true, "graph/check");
  auto accepted = authority.promote(std::move(scope), authority.accept(candidate, assessment));
  EXPECT_EQ(accepted.value(), candidate.value());
  EXPECT_TRUE(assessment.accepted());
  EXPECT_EQ(accepted.value()->members().size(), 4u);
  EXPECT_EQ(authority.consume_increment(accepted)->integer(), 11);
  EXPECT_THROW(authority.consume_increment(accepted), std::logic_error);
  EXPECT_EQ(authority.require_root_publication(accepted, store), accepted.value()->members()[0]);
  auto provisional = authority.provisional_base(accepted);
  EXPECT_EQ(provisional.root(), accepted.value()->members()[0]);
  authority.supersede(candidate);
  EXPECT_THROW(provisional.root(), std::logic_error);
}

TEST(ContinuationAuthority, NewAttemptRevokesAllOldCapabilities) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto candidate = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 1)));
  auto accepted = authority.accept(
      candidate, authority.attest(candidate, V::boolean(true), true, "graph/check"));
  authority.begin(cache.begin_attempt());
  EXPECT_THROW(base.root(), std::logic_error);
  EXPECT_THROW(accepted.value(), std::logic_error);
  EXPECT_NO_THROW(authority.read(store));
}

TEST(ContinuationImage, FieldSnapshotIsOwnedAndRestoresExactLayout) {
  Extent<D> cells{}, ghosts{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 4;
    ghosts[d] = 1;
  }
  const auto box = Box<D>::from_extents(cells);
  const auto layout = mesh::BoxArray<D>::from_domain(box, cells);
  Extent<D> ranks_extent{};
  for (int d = 0; d < D; ++d)
    ranks_extent[d] = 1;
  const mesh::RankSpace<D> ranks(Index<D>{}, ranks_extent);
  const auto distribution = mesh::Distribution<D>::replicated(layout, ranks);
  MultiFab<D> field(layout, distribution, Index<D>{}, 2, ghosts);
  field.set_val(Real(7));
  auto snapshot = V::field(field, 0, "role/read-write/owner0");
  field.set_val(Real(9));
  auto type = scalar(ContinuationKind::Field);
  type.field_owner = 0;
  type.field_components = 2;
  type.scientific_contract = "role/read-write/owner0";
  Codec codec(type, "test/owned-field", 1024 * 1024, [&](int owner) -> const MultiFab<D>& {
    if (owner != 0)
      throw std::logic_error("foreign owner");
    return field;
  });
  auto decoded = codec.decode(codec.encode(snapshot, 0));
  auto host = decoded.root->field().fab(0).create_host_mirror();
  decoded.root->field().fab(0).copy_to_host(host);
  for (std::size_t i = 0; i < host.size(); ++i)
    EXPECT_EQ(host(i), Real(7));
  EXPECT_THROW(require_continuation_projection<D>(decoded.root->field(), field), std::logic_error);
  field.set_val(Real(7));
  EXPECT_NO_THROW(require_continuation_projection<D>(decoded.root->field(), field));
  constexpr std::uint64_t budget = 1024 * 1024;
  const finite_structural::Limits limits{budget, budget, budget};
  auto flat = finite_structural::unpack<D>(type, snapshot, limits);
  auto copied = finite_structural::pack<D>(type, flat, limits);
  EXPECT_EQ(codec.encode(copied, 0), codec.encode(snapshot, 0));
  auto foreign = type;
  foreign.field_owner = 1;
  EXPECT_THROW(finite_structural::unpack<D>(foreign, snapshot, limits), std::invalid_argument);
  foreign = type;
  foreign.scientific_contract = "different/role";
  EXPECT_THROW(finite_structural::unpack<D>(foreign, snapshot, limits), std::invalid_argument);
}

TEST(ContinuationImage, FieldExpressionCodecPreservesSharedBasesAndRejectsMissingAlias) {
  Extent<D> cells{}, ranks_extent{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 2;
    ranks_extent[d] = 1;
  }
  const auto layout = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), cells);
  const auto distribution =
      mesh::Distribution<D>::replicated(layout, mesh::RankSpace<D>(Index<D>{}, ranks_extent));
  MultiFab<D> field(layout, distribution, Index<D>{}, 1, Extent<D>{});
  field.set_val(Real(3));
  auto basis = std::make_shared<const FieldExchangeExpressions::Basis>(
      FieldExchangeExpressions::Basis{"native/evaluation/7", 0, {}, {}});
  auto expression = std::make_shared<const FieldExchangeExpressions::Expression>(
      FieldExchangeExpressions::Expression{true, {{basis, 1.}}});
  auto weighted = std::make_shared<const FieldExchangeExpressions::Expression>(
      FieldExchangeExpressions::Expression{true, {{basis, -.25}}});
  auto snapshot = V::field(field, 0, "retained/source", expression);
  auto alternate = V::with_field_exchange_expression(snapshot, weighted);
  EXPECT_EQ(snapshot->field_storage_owner(), alternate->field_storage_owner());
  auto type = scalar(ContinuationKind::Field);
  type.field_owner = 0;
  type.field_components = 1;
  type.scientific_contract = "retained/source";
  auto schema = product({"a", "alias", "weighted"}, {type, type, type});
  Codec codec(schema, "test/field-expression", 1ULL << 20, [&](int owner) -> const MultiFab<D>& {
    if (owner != 0)
      throw std::invalid_argument("foreign owner");
    return field;
  });
  auto root = V::product({snapshot, snapshot, alternate});
  const auto encoded = codec.encode(root, 9);
  continuation_detail::Reader framing(encoded);
  EXPECT_EQ(framing.text(), "POPSCONT4");
  const auto decoded = codec.decode(encoded);
  const auto& a = decoded.root->members()[0]->field_exchange_expression();
  const auto& alias = decoded.root->members()[1]->field_exchange_expression();
  const auto& b = decoded.root->members()[2]->field_exchange_expression();
  ASSERT_TRUE(a && alias && b);
  EXPECT_EQ(a, alias);
  EXPECT_EQ(a->terms.at(0).basis, b->terms.at(0).basis);
  EXPECT_EQ(b->terms.at(0).coefficient, -.25);
  EXPECT_EQ(codec.encode(decoded.root, 9), encoded);
  EXPECT_EQ(codec.collective_identity(decoded.root, 9), codec.collective_identity(root, 9));
  EXPECT_NE(codec.collective_identity(root, 9),
            codec.collective_identity(V::product({snapshot, snapshot, snapshot}), 9));
  auto corrupt = encoded;
  std::fill(corrupt.end() - 8, corrupt.end(), 0xff);
  EXPECT_THROW(codec.decode(corrupt), std::invalid_argument);
  std::uint64_t used = 0;
  const auto archived = codec.decode_archive(encoded, 0, 1, [&](std::uint64_t bytes) {
    if (bytes > (1ULL << 22) - used)
      throw std::length_error("workspace");
    used += bytes;
  });
  EXPECT_EQ(codec.archive_collective_identity(archived, 1ULL << 20, [](std::uint64_t) {}),
            codec.collective_identity(root, 9));
  EXPECT_THROW(codec.decode_archive(encoded, 0, 1,
                                    [](std::uint64_t) { throw std::length_error("workspace"); }),
               std::length_error);
}

TEST(ContinuationImage, ProducerSupportKeepsHistoricalPointAndExactPrototypeThroughCodec) {
  Extent<D> cells{}, ranks_extent{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 2;
    ranks_extent[d] = 1;
  }
  const auto layout = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), cells);
  const auto distribution =
      mesh::Distribution<D>::replicated(layout, mesh::RankSpace<D>(Index<D>{}, ranks_extent));
  MultiFab<D> field(layout, distribution, Index<D>{}, 1, Extent<D>{});
  field.set_val(Real(7));
  auto support = std::make_shared<const ContinuationFieldSupport>(
      ContinuationFieldSupport{"same-program", "history/time=0.125/point=stage-2", 0, 2, 7, 19});
  auto basis = std::make_shared<const FieldExchangeExpressions::Basis>(
      FieldExchangeExpressions::Basis{"historical-native/evaluation", 0, {}, {}});
  auto expression = std::make_shared<const FieldExchangeExpressions::Expression>(
      FieldExchangeExpressions::Expression{true, {{basis, .5}}});
  auto type = scalar(ContinuationKind::Field);
  type.field_owner = 0;
  type.field_components = 1;
  type.scientific_contract = "history/type-and-point";
  int exact_requests = 0;
  auto provider = [&](int owner,
                      const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
    if (owner != 0 || !origin || *origin != *support)
      throw std::invalid_argument("unattested producer support");
    ++exact_requests;
    return field;
  };
  Codec codec(type, "test/producer-support", 1ULL << 20, provider);
  for (bool with_expression : {false, true}) {
    auto value = V::field(field, 0, type.scientific_contract,
                          with_expression ? expression : FieldExchangeExpressions::Ref{}, support);
    const auto bytes = codec.encode(value, 4);
    continuation_detail::Reader reader(bytes);
    EXPECT_EQ(reader.text(), "POPSCONT5");
    const auto decoded = codec.decode(bytes);
    ASSERT_TRUE(decoded.root->field_support());
    EXPECT_EQ(*decoded.root->field_support(), *support);
    EXPECT_EQ(bool(decoded.root->field_exchange_expression()), with_expression);
    EXPECT_EQ(codec.encode(decoded.root, 4), bytes);
    EXPECT_EQ(codec.collective_identity(value, 4), codec.collective_identity(decoded.root, 4));
    const auto archived = codec.decode_archive(bytes, 0, 1, [](std::uint64_t) {});
    EXPECT_EQ(codec.archive_collective_identity(archived, 1ULL << 20, [](std::uint64_t) {}),
              codec.collective_identity(value, 4));
    auto legacy = Codec(type, "test/producer-support", 1ULL << 20,
                        [&](int) -> const MultiFab<D>& { return field; });
    EXPECT_THROW(legacy.decode(bytes), std::invalid_argument);
    auto truncated = bytes;
    truncated.resize(truncated.size() - 1);
    EXPECT_THROW(codec.decode(truncated), std::invalid_argument);
    const auto shared = V::with_field_metadata(value, {expression, support});
    EXPECT_EQ(value->field_storage_owner(), shared->field_storage_owner());
    EXPECT_EQ(V::with_field_exchange_expression(shared, {})->field_support(), support);
    const finite_structural::Limits limits{1ULL << 20, 1ULL << 20, 1ULL << 20};
    ContinuationFieldMetadata retained;
    auto flat = finite_structural::unpack<D>(
        type, value, limits,
        [&](const auto&, auto&, const auto&, const ContinuationFieldMetadata& metadata) {
          retained = metadata;
        });
    auto repacked = finite_structural::pack<D>(
        type, flat, limits, [&](const auto&, auto&, const auto&, const ContinuationType& schema) {
          EXPECT_EQ(schema.scientific_contract, type.scientific_contract);
          return retained;
        });
    EXPECT_EQ(codec.encode(repacked, 4), bytes);
    EXPECT_THROW(finite_structural::unpack<D>(type, value, limits), std::invalid_argument);
  }
  EXPECT_GT(exact_requests, 0);
  auto foreign = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      support->program_identity, support->origin_point, support->program_owner, 1,
      support->topology_epoch, support->materialization_generation, support->field_provider});
  auto changed = V::field(field, 0, type.scientific_contract, {}, foreign);
  EXPECT_THROW(codec.encode(changed, 4), std::invalid_argument);
  foreign = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      support->program_identity, support->origin_point, 1, 1, support->topology_epoch,
      support->materialization_generation, support->field_provider});
  EXPECT_THROW(V::field(field, 0, type.scientific_contract, {}, foreign), std::invalid_argument);
}

TEST(ContinuationAuthority, ParentRejectionRevokesPromotedDescendants) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  Authority::Accepted accepted;
  Authority::Base derived;
  {
    auto parent = authority.begin_scope();
    auto child = authority.begin_scope();
    auto candidate = authority.freeze(base, V::real(.2), envelope(root(.2, 1, 3)));
    accepted = authority.promote(
        std::move(child), authority.accept(candidate, authority.attest(candidate, V::boolean(true),
                                                                       true, "graph/check")));
    derived = authority.provisional_base(accepted);
    EXPECT_NO_THROW(derived.root());
  }
  EXPECT_THROW(accepted.value(), std::logic_error);
  EXPECT_THROW(derived.root(), std::logic_error);
  EXPECT_NO_THROW(base.root());
}

TEST(ContinuationImage, ExactRankFieldShardsIncludeEmptyRankWithoutRankZeroSubstitution) {
  Extent<D> cells{}, ghosts{}, ranks_extent{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 2;
    ghosts[d] = 0;
    ranks_extent[d] = 1;
  }
  ranks_extent[0] = 2;
  const auto layout = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), cells);
  const mesh::RankSpace<D> ranks(Index<D>{}, ranks_extent);
  const auto distribution = mesh::Distribution<D>::partitioned(layout, ranks, {Index<D>{}});
  Index<D> rank1{};
  rank1[0] = 1;
  MultiFab<D> owned(layout, distribution, Index<D>{}, 1, ghosts),
      empty(layout, distribution, rank1, 1, ghosts);
  owned.set_val(Real(3));
  EXPECT_EQ(empty.local_size(), 0u);
  auto type = scalar(ContinuationKind::Field);
  type.field_owner = 0;
  type.field_components = 1;
  type.scientific_contract = "field/role";
  Codec owner_codec(type, "test/shards", 65536, [&](int) -> const MultiFab<D>& { return owned; });
  Codec empty_codec(type, "test/shards", 65536, [&](int) -> const MultiFab<D>& { return empty; });
  auto owner_bytes = owner_codec.encode(V::field(owned, 0, "field/role"), 0);
  auto empty_bytes = empty_codec.encode(V::field(empty, 0, "field/role"), 0);
  EXPECT_EQ(empty_codec.decode(empty_bytes).root->field().local_size(), 0u);
  EXPECT_THROW(empty_codec.decode(owner_bytes), std::invalid_argument);
  EXPECT_THROW(owner_codec.decode(empty_bytes), std::invalid_argument);
}

TEST(ContinuationImage, RestartRevokesOldAliasesButPreservesDecodedRealm) {
  FiniteDomainRegistry registry;
  const auto old = registry.issue(2, 4, 7);
  auto dt = scalar(ContinuationKind::Domain);
  dt.family = 7;
  dt.capacity = 4;
  auto it = dt;
  it.kind = ContinuationKind::Index;
  auto at = dt;
  at.kind = ContinuationKind::Array;
  at.children = {scalar(ContinuationKind::Int64)};
  Codec codec(product({"d", "i", "a"}, {dt, it, at}), "restart/realm", 4096, none());
  const auto old_index = V::index(old, 1);
  const auto old_array = V::array(old, {V::integer(3), V::integer(8)});
  const auto original = V::product({V::domain(old), old_index, old_array});
  auto decoded = codec.decode(codec.encode(original, 1));
  const auto restored = decoded.root->members()[0]->domain_ref();
  registry.adopt(decoded.realm);
  registry.revoke_on_restart(decoded.realm);
  EXPECT_FALSE(old->visible());
  EXPECT_TRUE(restored->visible());
  EXPECT_THROW(old->require_live(), std::logic_error);
  EXPECT_THROW(old_index->index_offset(), std::logic_error);
  EXPECT_THROW(old_array->members(), std::logic_error);
  EXPECT_THROW(V::domain(old), std::logic_error);
  EXPECT_THROW(codec.encode(original, 1), std::logic_error);
  EXPECT_NO_THROW(codec.encode(decoded.root, 1));
  EXPECT_EQ(restored, decoded.root->members()[1]->domain_ref());
  EXPECT_EQ(restored, decoded.root->members()[2]->domain_ref());
  // A second successful restart must also revoke a realm imported by the first restart.
  auto again = codec.decode(codec.encode(decoded.root, 2));
  registry.revoke_on_restart(again.realm);
  EXPECT_FALSE(restored->visible());
  EXPECT_TRUE(again.root->members()[0]->domain_ref()->visible());
}

TEST(ContinuationImage, RestartWithoutImageCreatesNextRealmLazilyAndBurnsOrdinals) {
  FiniteDomainRegistry registry;
  auto old = registry.issue(1, 4, 2);
  const auto version = old->version;
  static_assert(noexcept(registry.revoke_on_restart()));
  registry.revoke_on_restart();
  EXPECT_FALSE(old->visible());
  EXPECT_FALSE(registry.realm());
  auto next = registry.issue(1, 4, 2);
  EXPECT_TRUE(next->visible());
  EXPECT_NE(next->realm(), old->realm());
  EXPECT_GT(next->version, version);
  EXPECT_THROW(registry.issue(1, 4, 2, old->realm()), std::logic_error);
}

TEST(ContinuationImage, RejectingTrialDoesNotRevokeAcceptedDomainData) {
  FiniteDomainRegistry registry;
  auto domain = registry.issue(1, 2, 1);
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto child = authority.begin_scope();
  auto candidate = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 1)));
  authority.revoke();
  EXPECT_THROW(candidate.value(), std::logic_error);
  EXPECT_TRUE(domain->visible());
  EXPECT_EQ(V::index(domain, 0)->index_offset(), 0);
}

namespace {
std::pair<ContinuationType, R> deep_structure(std::size_t depth, const FiniteDomainRef& domain) {
  auto type = scalar(ContinuationKind::Int64);
  R value = V::integer(INT64_C(9007199254740997));
  for (std::size_t i = 0; i < depth; ++i) {
    ContinuationType parent;
    if (i % 3 == 0) {
      parent.kind = ContinuationKind::Product;
      parent.names = {"member"};
      parent.children.push_back(std::move(type));
      value = V::product({std::move(value)});
    } else if (i % 3 == 1) {
      parent.kind = ContinuationKind::Variant;
      parent.names = {"none", "some"};
      parent.children.push_back(scalar(ContinuationKind::Int64));
      parent.children.push_back(std::move(type));
      value = V::variant(1, std::move(value));
    } else {
      parent.kind = ContinuationKind::Array;
      parent.family = domain->family;
      parent.capacity = domain->capacity;
      parent.children.push_back(std::move(type));
      value = V::array(domain, {std::move(value)});
    }
    type = std::move(parent);
  }
  return {std::move(type), std::move(value)};
}

void expect_deep_leaf(R value, std::size_t depth) {
  for (std::size_t i = 0; i < depth; ++i) {
    ASSERT_EQ(value->members().size(), 1u);
    value = value->members()[0];
  }
  EXPECT_EQ(value->integer(), INT64_C(9007199254740997));
}
}  // namespace

TEST(ContinuationImage, DeepStructuresRoundTripLegacyExtendedAndTransportWithoutDepthLimit) {
  constexpr std::size_t depth = 1024;
  constexpr std::uint64_t capacity = 4 * 1024 * 1024;
  for (bool replicated : {true, false}) {
    FiniteDomainRegistry domains(replicated);
    auto domain = domains.issue(1, 1, 931);
    auto [type, value] = deep_structure(depth, domain);
    auto copied = type;
    EXPECT_EQ(copied, type);
    EXPECT_EQ(continuation_transport_type_identity(copied, true, capacity),
              continuation_transport_type_identity(type, true, capacity));
    Codec codec(type, "test/deep-structural-codec", capacity, none());
    const auto image = codec.encode(value, 17);
    const auto restored = codec.decode(image);
    EXPECT_EQ(restored.publication, 17u);
    expect_deep_leaf(restored.root, depth);
    EXPECT_EQ(codec.encode(restored.root, 17), image);
    EXPECT_EQ(codec.collective_identity(value, 17), codec.collective_identity(restored.root, 17));
    ContinuationTransportSession<D> sender, receiver;
    const auto message = sender.encode(type, value, capacity);
    const auto transported = receiver.decode(type, message, capacity);
    expect_deep_leaf(transported, depth);
    EXPECT_EQ(receiver.encode(type, transported, capacity), message);
  }
}

TEST(ContinuationImage, DeepStorePrepareRollbackAndRestartRetainPublishedValue) {
  constexpr std::size_t depth = 1024;
  FiniteDomainRegistry domains(false);
  auto [type, value] = deep_structure(depth, domains.issue(1, 1, 932));
  auto config = configuration();
  config.type.names.push_back("deep");
  config.type.children.push_back(std::move(type));
  config.identity = "test/deep-store";
  config.byte_capacity = 4 * 1024 * 1024;
  Store store;
  store.configure(std::move(config));
  store.publish(
      store.prepare(V::product({V::real(0), V::integer(0), V::integer(1), value}), none(), true));
  auto rollback = store;
  auto image = store.checkpoint(none());
  store.publish(
      store.prepare(V::product({V::real(.25), V::integer(1), V::integer(2), value}), none()));
  store.swap(rollback);
  store.require_frontier(0, 0);
  expect_deep_leaf(store.read()->members()[3], depth);
  store.publish(store.prepare_restart(image, none()));
  store.require_frontier(0, 0);
  expect_deep_leaf(store.read()->members()[3], depth);
  // Rejection leaves the accepted publication and exact authority unchanged.
  auto rejected = image;
  rejected.pop_back();
  const auto accepted = store.read();
  EXPECT_THROW(store.prepare_restart(rejected, none()), std::invalid_argument);
  EXPECT_EQ(store.read(), accepted);
}

TEST(ContinuationImage, IterativeRetirementPreservesLiveAliasesAndTypeCopies) {
  constexpr std::size_t depth = 20000;
  auto leaf = V::integer(11);
  R nested = leaf;
  ContinuationType type = scalar(ContinuationKind::Int64);
  for (std::size_t i = 0; i < depth; ++i) {
    nested = V::product({std::move(nested)});
    ContinuationType parent;
    parent.names = {"value"};
    parent.children.push_back(std::move(type));
    type = std::move(parent);
  }
  std::weak_ptr<const V> retired = nested;
  auto copy = type;
  EXPECT_EQ(type, copy);
  type = scalar(ContinuationKind::Boolean);
  EXPECT_NE(type, copy);
  nested.reset();
  EXPECT_TRUE(retired.expired());
  EXPECT_EQ(leaf.use_count(), 1);
  EXPECT_EQ(leaf->integer(), 11);
}

TEST(ContinuationImage, DeepStructuralBudgetFailureDoesNotMutateTheSource) {
  FiniteDomainRegistry domains;
  auto [type, value] = deep_structure(1024, domains.issue(1, 1, 933));
  EXPECT_THROW(type.validate(256), std::length_error);
  EXPECT_THROW(continuation_transport_type_identity(type, true, 256), std::length_error);
  expect_deep_leaf(value, 1024);
  EXPECT_NO_THROW(type.validate(4 * 1024 * 1024));
}

TEST(ContinuationImage, DeepBundleBindingsPreserveExactDomainWitness) {
  constexpr std::size_t depth = 1024;
  constexpr std::uint64_t capacity = 16 * 1024 * 1024;
  FiniteDomainRegistry domains;
  const auto domain = domains.issue(1, 1, 934);
  auto [nested_type, nested] = deep_structure(depth, domain);
  ContinuationType domain_type{ContinuationKind::Domain, {}, {}, 934, 1};
  ContinuationType fields;
  fields.names = {"deep"};
  fields.children.push_back(std::move(nested_type));
  ContinuationType bundle;
  bundle.names = {"instance", "fields", "geometry", "maps", "histories"};
  bundle.children.push_back(domain_type);
  bundle.children.push_back(std::move(fields));
  for (int i = 0; i < 3; ++i)
    bundle.children.emplace_back();
  bundle.dependent_bundle = true;
  bundle.domain_bindings.push_back({{0}, {0}});
  const auto* current = &bundle.children[1].children[0];
  std::vector<std::size_t> path{1, 0};
  for (std::size_t i = 0; i < depth; ++i) {
    if (current->kind == ContinuationKind::Array) {
      bundle.domain_bindings.push_back({path, {0}});
      path.push_back(continuation_each);
      current = &current->children[0];
    } else {
      const std::size_t member = current->kind == ContinuationKind::Variant ? 1 : 0;
      path.push_back(member);
      current = &current->children[member];
    }
  }
  Codec codec(bundle, "test/deep-dependent-bundle", capacity, none());
  auto pack = [&](R value) {
    return V::product({V::domain(domain), V::product({std::move(value)}), V::product({}),
                       V::product({}), V::product({})});
  };
  const auto original = pack(nested);
  const auto restored = codec.decode(codec.encode(original, 9));
  expect_deep_leaf(restored.root->members()[1]->members()[0], depth);
  const auto other_domain = domains.issue(1, 1, 934);
  auto [other_type, other] = deep_structure(depth, other_domain);
  EXPECT_EQ(bundle.children[1].children[0], other_type);
  EXPECT_THROW(codec.validate(pack(other)), std::invalid_argument);
  EXPECT_NO_THROW(codec.validate(original));
}

TEST(ContinuationImage, FlatStorageMarshallingPreservesDeepMixedStructureAndAliases) {
  namespace fs = finite_structural;
  constexpr std::uint64_t capacity = 16 * 1024 * 1024;
  constexpr std::size_t depth = 2048;
  FiniteDomainRegistry domains;
  const auto domain = domains.issue(1, 1, 935);
  auto [type, original] = deep_structure(depth, domain);
  Codec codec(type, "test/flat-storage", capacity, none());
  const auto expected = codec.encode(original, 29);
  auto flat = fs::unpack<D>(type, original, {capacity, capacity, capacity});
  const auto restored = fs::pack<D>(type, flat, {capacity, capacity, capacity});
  expect_deep_leaf(restored, depth);
  EXPECT_EQ(codec.encode(restored, 29), expected);
  auto cursor = restored;
  while (!cursor->members().empty()) {
    if (cursor->kind() == ContinuationKind::Array)
      EXPECT_EQ(cursor->domain_ref(), domain);
    cursor = cursor->members()[0];
  }
  for (const auto limits : {fs::Limits{1, capacity, capacity}, fs::Limits{capacity, 1, capacity},
                            fs::Limits{capacity, capacity, 1}}) {
    EXPECT_THROW(fs::pack<D>(type, flat, limits), std::length_error);
    EXPECT_THROW(fs::unpack<D>(type, original, limits), std::length_error);
    EXPECT_EQ(codec.encode(original, 29), expected);
  }
}

TEST(ContinuationImage, FlatStorageSelectsOnlyActiveVariantSlots) {
  namespace fs = finite_structural;
  constexpr std::uint64_t capacity = 4096;
  auto type = product({"unused", "active"},
                      {scalar(ContinuationKind::Int64), scalar(ContinuationKind::Float64)});
  type.kind = ContinuationKind::Variant;
  fs::Record record{{std::int64_t(1), std::any{}, Real(-0.0)}};
  const auto owned = fs::pack<D>(type, std::any(record), {capacity, capacity, capacity});
  EXPECT_EQ(owned->arm(), 1u);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(owned->members()[0]->real()),
            UINT64_C(0x8000000000000000));
  auto restored =
      std::any_cast<fs::Record>(fs::unpack<D>(type, owned, {capacity, capacity, capacity}));
  EXPECT_FALSE(restored.values[1].has_value());
  record.values[0] = std::int64_t(2);
  EXPECT_THROW(fs::pack<D>(type, std::any(record), {capacity, capacity, capacity}),
               std::invalid_argument);
}

TEST(ContinuationImage, FlatArrayRetirementIsIterativeAndKeepsIndependentAliases) {
  namespace fs = finite_structural;
  FiniteDomainRegistry domains;
  const auto domain = domains.issue(1, 1, 936);
  auto leaf = fs::make_values(std::vector<std::any>{std::int64_t(9007199254740997)});
  std::any value = fs::Array{domain, leaf};
  for (std::size_t depth = 0; depth < 20000; ++depth) {
    std::vector<std::any> elements;
    elements.push_back(std::move(value));
    value = fs::Array{domain, fs::make_values(std::move(elements))};
  }
  value.reset();
  ASSERT_EQ(leaf->size(), 1u);
  EXPECT_EQ(std::any_cast<std::int64_t>((*leaf)[0]), INT64_C(9007199254740997));
  EXPECT_TRUE(domain->visible());
}

TEST(ContinuationImage, RequestPreparationBorrowsLvaluesOwnsTemporariesAndBudgetsIndices) {
  namespace f = finite_structural;
  namespace d = distributed;
  auto lane = ExecutionLane::duplicate_world_collectively("test.requests.borrowed-preparation");
  d::Registry registry(lane);
  registry.configure(1024 * 1024, 8);
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(lane.size()) + 1);
  for (std::size_t rank = 0; rank < offsets.size(); ++rank)
    offsets[rank] = 2 * rank;
  const auto domain = registry.issue(901, offsets.back(), offsets);
  const auto target = registry.issue(902, offsets.back(), offsets);
  auto requests =
      d::requests(registry, domain, target, [&](d::Index i) { return d::index(target, i.gid); });
  d::publish_requests<D>(registry, requests, 65536, nullptr);
  const auto published = d::requests_array<D>(requests);
  const auto original_indices = requests.indices.data();
  const auto borrowed = f::source_cell(requests);
  EXPECT_EQ(&f::held<d::Requests>(borrowed), &requests);
  EXPECT_EQ(f::held<d::Requests>(borrowed).indices.data(), original_indices);
  const d::Index* moved_indices = nullptr;
  const auto owned = [&] {
    auto temporary = d::requests_from_array(published);
    moved_indices = temporary.indices.data();
    return f::source_cell(std::move(temporary));
  }();
  EXPECT_EQ(f::held<d::Requests>(owned).indices.data(), moved_indices);
  const auto type = ContinuationType{ContinuationKind::DistributedArray,
                                     {published->element},
                                     {},
                                     domain->family(),
                                     domain->capacity()};
  const std::uint64_t exact_source = sizeof(std::any) + requests.indices.size() * sizeof(d::Index);
  const f::Limits ample{exact_source, 65536, 65536};
  EXPECT_EQ(f::pack<D>(type, borrowed, ample)->distributed_array_ref(), published);
  EXPECT_EQ(f::pack<D>(type, owned, ample)->distributed_array_ref(), published);
  EXPECT_THROW(f::pack<D>(type, borrowed, {exact_source - 1, 65536, 65536}), std::length_error);
  EXPECT_EQ(requests.indices.data(), original_indices);
  EXPECT_EQ(d::requests_array<D>(requests), published);
  EXPECT_EQ(registry.retained_values(), 1u);
}

TEST(ContinuationImage, RetainedDomainDoesNotKeepTemporaryAliasesAlive) {
  FiniteDomainRegistry registry;
  auto accepted = registry.issue(2, 4, 7);
  auto temporary = registry.issue(2, 4, 7, accepted->realm());
  ASSERT_EQ(accepted->realm(), temporary->realm());
  const auto index = V::index(temporary, 1);
  const auto array = V::array(temporary, {V::integer(1), V::integer(2)});
  const std::array<FiniteDomainRef, 1> keep{accepted};
  registry.revoke_unretained(keep);
  EXPECT_TRUE(accepted->visible());
  EXPECT_FALSE(temporary->visible());
  EXPECT_THROW(index->index_offset(), std::logic_error);
  EXPECT_THROW(array->members(), std::logic_error);
  auto later = registry.issue(2, 4, 7, accepted->realm());
  EXPECT_GT(later->version, temporary->version);
  registry.revoke_unretained(keep);
  EXPECT_TRUE(accepted->visible());
  EXPECT_FALSE(later->visible());
}

TEST(ContinuationImage, AmrFiniteInvocationBurnsVersionsAndRevokesLocalAndReplicatedDomains) {
  AmrFiniteValueLifetime<D> lifetime;
  lifetime.begin();
  auto replicated = lifetime.finite.issue(2, 4, 7);
  auto local = lifetime.local_finite.issue(1, 4, 7);
  EXPECT_THROW(lifetime.begin(), std::logic_error);
  lifetime.finish();
  EXPECT_FALSE(replicated->visible());
  EXPECT_FALSE(local->visible());
  lifetime.begin();
  auto next = lifetime.finite.issue(2, 4, 7);
  EXPECT_GT(next->version, replicated->version);
  EXPECT_NE(next->realm(), replicated->realm());
  lifetime.retained_domains = {next};  // represents an already-attested Store publication
  lifetime.finish();
  EXPECT_TRUE(next->visible());
  lifetime.begin();
  auto trial = lifetime.finite.issue(2, 4, 7, next->realm());
  lifetime.finish();
  EXPECT_TRUE(next->visible());
  EXPECT_FALSE(trial->visible());
}

TEST(ContinuationImage, AdoptedAcceptedDomainsParticipateInSelectiveRevocation) {
  FiniteDomainRegistry source;
  const auto original = source.issue(2, 4, 7);
  auto dt = scalar(ContinuationKind::Domain);
  dt.family = 7;
  dt.capacity = 4;
  Codec codec(dt, "selective/decoded", 4096, none());
  const auto decoded = codec.decode(codec.encode(V::domain(original), 1));
  const auto restored = decoded.root->domain_ref();
  FiniteDomainRegistry owner;
  const std::array<FiniteDomainRef, 1> keep{restored};
  owner.adopt_domains(keep);
  owner.revoke_unretained(keep);
  EXPECT_TRUE(restored->visible());
  const auto later = owner.issue(2, 4, 7, restored->realm());
  EXPECT_GT(later->version, restored->version);
  owner.revoke_unretained();
  EXPECT_FALSE(restored->visible());
  EXPECT_FALSE(later->visible());
  EXPECT_TRUE(original->visible());
}

TEST(ContinuationAuthority, ParentDependsOnExactAcceptedChildrenAcrossIssuers) {
  auto parent_store = initial_store();
  auto child_store = initial_store();
  PreparedResourceCache cache;
  const auto attempt = cache.begin_attempt();
  Authority parent, child;
  parent.begin(attempt);
  child.begin(attempt);
  const auto cb = child.read(child_store);
  const auto cc = child.freeze(cb, V::real(.1), envelope(root(.1, 1, 5)));
  const auto ca = child.accept(cc, child.attest(cc, V::boolean(true), true, "child/equations"));
  const std::array<Authority::Accepted, 1> children{ca};
  const auto pc =
      parent.freeze(parent.read(parent_store), V::real(.1), envelope(root(.1, 1, 7)), children);
  const auto assessment = parent.attest(pc, V::boolean(true), true, "parent/composite");
  const auto accepted = parent.accept(pc, assessment);
  EXPECT_NO_THROW(parent.require_root_publication(accepted, parent_store));
  child.supersede(cc);
  EXPECT_THROW(pc.value(), std::logic_error);
  EXPECT_THROW(assessment.accepted(), std::logic_error);
  EXPECT_THROW(parent.require_root_publication(accepted, parent_store), std::logic_error);
  EXPECT_NO_THROW(parent_store.require_frontier(0., 0));
}

TEST(ContinuationAuthority, EqualChildPayloadCannotReplaceFrozenDependency) {
  auto store = initial_store();
  PreparedResourceCache cache;
  const auto attempt = cache.begin_attempt();
  Authority parent, child;
  parent.begin(attempt);
  child.begin(attempt);
  const auto base = child.read(store);
  const auto first = child.freeze(base, V::real(.1), envelope(root(.1, 1, 4)));
  const auto accepted_first =
      child.accept(first, child.attest(first, V::boolean(true), true, "child/check"));
  const std::array<Authority::Accepted, 1> dependencies{accepted_first};
  const auto candidate =
      parent.freeze(parent.read(store), V::real(.1), envelope(root(.1, 1, 9)), dependencies);
  const auto second = child.freeze(base, V::real(.1), envelope(root(.1, 1, 4)));
  const auto accepted_second =
      child.accept(second, child.attest(second, V::boolean(true), true, "child/check"));
  child.supersede(first);
  EXPECT_NO_THROW(accepted_second.value());
  EXPECT_THROW(parent.attest(candidate, V::boolean(true), true, "parent/check"), std::logic_error);
  const std::array<Authority::Accepted, 2> duplicates{accepted_second, accepted_second};
  EXPECT_THROW(
      parent.freeze(parent.read(store), V::real(.1), envelope(root(.1, 1, 10)), duplicates),
      std::invalid_argument);
}

TEST(ContinuationAuthority, EveryChildScopeMustSurviveTheParentVote) {
  auto store = initial_store();
  PreparedResourceCache cache;
  const auto attempt = cache.begin_attempt();
  Authority parent, a, b;
  parent.begin(attempt);
  a.begin(attempt);
  b.begin(attempt);
  auto accept_child = [&](Authority& issuer) {
    const auto c = issuer.freeze(issuer.read(store), V::real(.1), envelope(root(.1, 1, 3)));
    return issuer.accept(c, issuer.attest(c, V::boolean(true), true, "child/check"));
  };
  const std::array<Authority::Accepted, 2> children{accept_child(a), accept_child(b)};
  const auto p = parent.freeze(parent.read(store), V::real(.1), envelope(root(.1, 1, 4)), children);
  const auto accepted = parent.accept(p, parent.attest(p, V::boolean(true), true, "parent/check"));
  b.revoke();
  EXPECT_NO_THROW(children[0].value());
  EXPECT_THROW(accepted.value(), std::logic_error);
  EXPECT_THROW(parent.require_root_publication(accepted, store), std::logic_error);
}

TEST(ContinuationImage, PreparedDomainAdoptionDoesNotMutateBeforePublication) {
  FiniteDomainRegistry child, parent;
  const auto sample = child.issue(2, 4, 9);
  const std::array<FiniteDomainRef, 1> keep{sample};
  auto adoption = parent.prepare_adoption(keep);
  parent.revoke_unretained();
  EXPECT_TRUE(sample->visible());
  parent.install_adoption(std::move(adoption));
  parent.revoke_unretained(keep);
  child.revoke_unretained(keep);
  EXPECT_TRUE(sample->visible());
  auto temporary = child.issue(2, 4, 9);
  child.revoke_unretained(keep);
  EXPECT_FALSE(temporary->visible());
  EXPECT_TRUE(sample->visible());
}

TEST(ContinuationImage, RollbackAdoptionRevokesNewAliasesWithoutReusingOrdinals) {
  FiniteDomainRegistry registry;
  const auto base = registry.issue(1, 4, 9);
  const std::array<FiniteDomainRef, 1> keep{base};
  auto rollback = registry.prepare_adoption(keep, base->realm());
  const auto provisional = registry.issue(2, 4, 9);
  registry.revoke_unretained(keep);
  registry.install_adoption(std::move(rollback));
  EXPECT_TRUE(base->visible());
  EXPECT_FALSE(provisional->visible());
  const auto next = registry.issue(3, 4, 9);
  EXPECT_GT(next->version, provisional->version);
}

TEST(ContinuationImage, ParentRetainsChildRealmsWithoutAdoptingTheirAllocator) {
  FiniteDomainRegistry parent, first, second;
  parent.synchronize_identity(parent.realm(), {301, 401});
  first.synchronize_identity(first.realm(), {302, 402});
  second.synchronize_identity(second.realm(), {303, 403});
  auto domain = scalar(ContinuationKind::Domain);
  domain.family = 8;
  domain.capacity = 3;
  auto config = configuration();
  config.type =
      product({"time", "step", "a", "b"},
              {scalar(ContinuationKind::Float64), scalar(ContinuationKind::Int64), domain, domain});
  Store store;
  store.configure(config);
  auto closure = std::make_shared<ContinuationClosure<D>>();
  closure->replicated_realm = parent.realm();
  const auto value = V::product({V::real(0), V::integer(0), V::domain(first.issue(1, 3, 8)),
                                 V::domain(second.issue(1, 3, 8))});
  store.publish(store.prepare(value, none(), true, {}, closure));
  EXPECT_EQ(store.domain_realm(), parent.realm());
  EXPECT_EQ(store.domain_high_water(), 0);
  const auto restored = store.prepare_restart(store.checkpoint(none()), none());
  EXPECT_EQ(restored.realm->transport_identity(), parent.realm()->transport_identity());
  EXPECT_NE(restored.realm, restored.value->members()[2]->domain_ref()->realm());
  EXPECT_NE(restored.realm, restored.value->members()[3]->domain_ref()->realm());
  auto scalar_store = initial_store();
  scalar_store.publish(scalar_store.prepare(root(), none(), false, {}, closure));
  const auto scalar_restored =
      scalar_store.prepare_restart(scalar_store.checkpoint(none()), none());
  EXPECT_EQ(scalar_restored.realm->transport_identity(), parent.realm()->transport_identity());
}

TEST(ContinuationAuthority, RegridDescendantsKeepDistinctTargetAllocatorsAndSourceBase) {
  FiniteDomainRegistry source_domains, first_domains, second_domains;
  source_domains.synchronize_identity(source_domains.realm(), {511, 611});
  first_domains.synchronize_identity(first_domains.realm(), {512, 612});
  second_domains.synchronize_identity(second_domains.realm(), {513, 613});
  auto domain_type = scalar(ContinuationKind::Domain);
  domain_type.family = 31;
  domain_type.capacity = 8;
  auto config = configuration();
  config.type = product({"time", "step", "domain", "source"},
                        {scalar(ContinuationKind::Float64), scalar(ContinuationKind::Int64),
                         domain_type, domain_type});
  const auto image = [](FiniteDomainRef own, FiniteDomainRef from) {
    return V::product(
        {V::real(.375), V::integer(9), V::domain(std::move(own)), V::domain(std::move(from))});
  };
  const auto closure = [](FiniteDomainRealmRef primary) {
    auto value = std::make_shared<ContinuationClosure<D>>();
    value->replicated_realm = std::move(primary);
    return value;
  };
  Store source, first, second;
  source.configure(config);
  first.configure(config);
  second.configure(config);
  const auto source_domain = source_domains.issue(2, 8, 31);
  const auto source_image = image(source_domain, source_domain);
  source.publish(source.prepare(source_image, none(), true, {}, closure(source_domains.realm())));
  source.publish(source.prepare(source_image, none(), false, {}, closure(source_domains.realm())));
  source.publish(source.prepare(source_image, none(), false, {}, closure(source_domains.realm())));
  const auto source_bytes = source.checkpoint(none());
  const auto first_old = first_domains.issue(2, 8, 31);
  first.publish(first.prepare(image(first_old, source_domain), none(), true, {},
                              closure(first_domains.realm())));
  auto rollback_first = first;
  auto rollback_second = second;

  PreparedResourceCache cache;
  const auto attempt = cache.begin_attempt();
  Authority a, b;
  a.begin(attempt);
  b.begin(attempt);
  const auto base_a = a.read(source), base_b = b.read(source);
  EXPECT_THROW(a.read(first), std::logic_error);
  const auto first_new = first_domains.issue(4, 8, 31);
  const auto second_new = second_domains.issue(4, 8, 31);
  const auto candidate_a =
      a.freeze(base_a, V::integer(0), envelope(image(first_new, source_domain)));
  const auto candidate_b =
      b.freeze(base_b, V::integer(1), envelope(image(second_new, source_domain)));
  const auto accepted_a =
      a.accept(candidate_a, a.attest(candidate_a, V::boolean(true), true, "regrid/a"));
  const auto accepted_b =
      b.accept(candidate_b, b.attest(candidate_b, V::boolean(true), true, "regrid/b"));
  EXPECT_THROW(a.require_root_publication(accepted_a, first), std::logic_error);
  auto prepared_a = first.prepare(a.require_root_publication(accepted_a, source), none(), false, {},
                                  closure(first_domains.realm()));
  auto prepared_b = second.prepare(b.require_root_publication(accepted_b, source), none(), true, {},
                                   closure(second_domains.realm()));
  EXPECT_EQ(prepared_a.publication, 1U);
  EXPECT_EQ(prepared_b.publication, 0U);
  EXPECT_EQ(source.publication(), 2U);
  EXPECT_EQ(prepared_a.realm, first_domains.realm());
  EXPECT_EQ(prepared_b.realm, second_domains.realm());
  EXPECT_NE(prepared_a.realm, prepared_b.realm);
  EXPECT_NE(prepared_a.realm, source.domain_realm());
  EXPECT_EQ(prepared_a.value->members()[3]->domain_ref(), source_domain);
  EXPECT_EQ(prepared_b.value->members()[3]->domain_ref(), source_domain);
  first.publish(std::move(prepared_a));
  second.publish(std::move(prepared_b));
  first.require_frontier(.375, 9);
  second.require_frontier(.375, 9);
  EXPECT_EQ(source.checkpoint(none()), source_bytes);
  EXPECT_THROW(first.prepare(first.read(), none(), false, {}, closure(source_domains.realm())),
               std::invalid_argument);

  first.swap(rollback_first);
  second.swap(rollback_second);
  cache.reject_attempt();
  EXPECT_THROW(accepted_a.value(), std::logic_error);
  EXPECT_THROW(accepted_b.value(), std::logic_error);
  EXPECT_EQ(first.read()->members()[2]->domain_ref(), first_old);
  EXPECT_FALSE(second.initialized());
  EXPECT_EQ(source.checkpoint(none()), source_bytes);
  EXPECT_GT(first_domains.issue(1, 8, 31)->version, first_new->version);
  EXPECT_GT(second_domains.issue(1, 8, 31)->version, second_new->version);
}

TEST(ContinuationAuthority, PromotionPreparationIsPureAndCommitKeepsExactAssessment) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto scope = authority.begin_scope();
  auto candidate = authority.freeze(base, V::real(.2), envelope(root(.2, 1, 3), 11));
  auto assessment = authority.attest(candidate, V::boolean(true), true, "graph/check");
  auto accepted = authority.accept(candidate, assessment);
  const auto before = authority.logical_state();
  auto prepared = authority.prepare_promotion(scope, accepted);
  EXPECT_EQ(authority.logical_state(), before);
  EXPECT_THROW(authority.require_root_publication(accepted, store), std::logic_error);
  authority.validate_promotion(prepared, scope);
  EXPECT_EQ(authority.logical_state(), before);
  static_assert(noexcept(authority.commit_promotion(std::move(prepared), scope)));
  auto promoted = authority.commit_promotion(std::move(prepared), scope);
  EXPECT_EQ(promoted.value(), accepted.value());
  EXPECT_TRUE(assessment.accepted());
  EXPECT_EQ(authority.require_root_publication(promoted, store), promoted.value()->members()[0]);
  EXPECT_THROW(authority.validate_promotion(prepared, scope), std::logic_error);
  EXPECT_EQ(authority.consume_increment(promoted)->integer(), 11);
  EXPECT_THROW(authority.consume_increment(promoted), std::logic_error);
}

TEST(ContinuationAuthority, LateRejectedSiblingLeavesEveryPromotionUncommitted) {
  auto first_store = initial_store(), second_store = initial_store();
  PreparedResourceCache first_cache, second_cache;
  Authority first, second;
  first.begin(first_cache.begin_attempt());
  second.begin(second_cache.begin_attempt());
  auto first_base = first.read(first_store), second_base = second.read(second_store);
  auto first_scope = first.begin_scope(), second_scope = second.begin_scope();
  auto a = first.freeze(first_base, V::integer(1), envelope(root(.1, 1, 1)));
  auto b = second.freeze(second_base, V::integer(2), envelope(root(.1, 1, 2)));
  auto accepted_a = first.accept(a, first.attest(a, V::boolean(true), true, "first/check"));
  auto accepted_b = second.accept(b, second.attest(b, V::boolean(true), true, "second/check"));
  auto pending_a = first.prepare_promotion(first_scope, accepted_a);
  auto pending_b = second.prepare_promotion(second_scope, accepted_b);
  const auto before_a = first.logical_state(), before_b = second.logical_state();
  second.supersede(b);
  EXPECT_NO_THROW(first.validate_promotion(pending_a, first_scope));
  EXPECT_THROW(second.validate_promotion(pending_b, second_scope), std::logic_error);
  EXPECT_EQ(first.logical_state(), before_a);
  EXPECT_EQ(second.logical_state(), before_b);
  EXPECT_THROW(first.require_root_publication(accepted_a, first_store), std::logic_error);
  // Abandoning prepared proposals did not promote or detach either Scope.
  first_scope = Authority::Scope{};
  second_scope = Authority::Scope{};
  EXPECT_THROW(accepted_a.value(), std::logic_error);
  EXPECT_THROW(accepted_b.value(), std::logic_error);
  first_store.require_frontier(0, 0);
  second_store.require_frontier(0, 0);
}

TEST(ContinuationAuthority, PromotionRejectsMovedHandleForeignIssuerAndRevokedAttempt) {
  auto store = initial_store();
  PreparedResourceCache cache;
  Authority authority, foreign;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto scope = authority.begin_scope();
  auto candidate = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 1)));
  auto accepted =
      authority.accept(candidate, authority.attest(candidate, V::boolean(true), true, "check"));
  auto prepared = authority.prepare_promotion(scope, accepted);
  auto moved = std::move(scope);
  EXPECT_THROW(authority.validate_promotion(prepared, moved), std::logic_error);
  EXPECT_THROW(authority.validate_promotion(prepared, scope), std::logic_error);
  PreparedResourceCache foreign_cache;
  foreign.begin(foreign_cache.begin_attempt());
  EXPECT_THROW(foreign.validate_promotion(prepared, moved), std::logic_error);
  auto valid = authority.prepare_promotion(moved, accepted);
  cache.reject_attempt();
  EXPECT_THROW(authority.validate_promotion(valid, moved), std::logic_error);
}

static_assert(!std::is_assignable_v<
              decltype((std::declval<ContinuationFieldSupport&>().field_provider)), std::string>);
static_assert(!std::is_assignable_v<
              decltype((std::declval<ContinuationFieldSupport&>().producer_level)), int>);
static_assert(!std::is_assignable_v<
              decltype((std::declval<FieldRepresentationObligation&>().contract)), std::string>);
static_assert(!std::is_assignable_v<
              decltype((std::declval<FieldRepresentationObligation&>().origin)), std::string>);

TEST(ContinuationImage, IndependentProviderAndRepresentationHaveExactCONT6Authority) {
  Extent<D> cells{}, ranks{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 2;
    ranks[d] = 1;
  }
  const auto layout = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), cells);
  const auto distribution =
      mesh::Distribution<D>::replicated(layout, mesh::RankSpace<D>(Index<D>{}, ranks));
  MultiFab<D> field(layout, distribution, Index<D>{}, 1, Extent<D>{});
  field.set_val(Real(7));
  auto type = scalar(ContinuationKind::Field);
  type.field_components = 1;
  type.scientific_contract = "retained/role";
  type.field_provider = "problem:equation-A/unknown:potential/components:1";
  auto support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      "program", "time:0.125/invocation:4", -1, 2, 7, 19, type.field_provider});
  auto representation = std::make_shared<const FieldRepresentationObligation>(
      FieldRepresentationObligation{"transform:exact-a*x*x/a=IEEE:3ff0000000000000", "producer:4"});
  int requests = 0;
  Codec codec(type, "test/independent-field", 1ULL << 20,
              [&](const ContinuationType& selected,
                  const ContinuationFieldSupport::Ref& origin) -> const MultiFab<D>& {
                if (selected != type || !origin || *origin != *support)
                  throw std::invalid_argument("foreign provider or historical support");
                ++requests;
                return field;
              });
  const auto value = V::field(field, -1, type.scientific_contract, {}, support, type.field_provider,
                              representation);
  const auto bytes = codec.encode(value, 23);
  continuation_detail::Reader wire(bytes);
  EXPECT_EQ(wire.text(), "POPSCONT6");
  const auto decoded = codec.decode(bytes);
  ASSERT_TRUE(decoded.root->field_support());
  ASSERT_TRUE(decoded.root->field_representation());
  EXPECT_EQ(decoded.root->field_provider(), type.field_provider);
  EXPECT_EQ(*decoded.root->field_support(), *support);
  EXPECT_EQ(*decoded.root->field_representation(), *representation);
  EXPECT_EQ(codec.encode(decoded.root, 23), bytes);
  EXPECT_EQ(codec.collective_identity(value, 23), codec.collective_identity(decoded.root, 23));
  const auto archived = codec.decode_archive(bytes, 0, 1, [](std::uint64_t) {});
  EXPECT_EQ(codec.archive_collective_identity(archived, 1ULL << 20, [](std::uint64_t) {}),
            codec.collective_identity(value, 23));
  EXPECT_GT(requests, 0);

  for (const std::string provider : {"problem:equation-B/unknown:potential/components:1",
                                     "problem:equation-A/unknown:other/components:1"}) {
    auto foreign = type;
    foreign.field_provider = provider;
    int allocations = 0;
    Codec rejected(
        foreign, "test/independent-field", 1ULL << 20,
        [&](const ContinuationType&, const ContinuationFieldSupport::Ref&) -> const MultiFab<D>& {
          ++allocations;
          return field;
        });
    EXPECT_THROW(rejected.decode(bytes), std::invalid_argument);
    EXPECT_EQ(allocations, 0);
  }
  auto legacy = Codec(type, "test/independent-field", 1ULL << 20,
                      [&](int) -> const MultiFab<D>& { return field; });
  EXPECT_THROW(legacy.encode(value, 23), std::invalid_argument);
  EXPECT_THROW(V::field(field, 0, type.scientific_contract, {}, support, type.field_provider),
               std::invalid_argument);
  auto wrong_support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      support->program_identity, support->origin_point, support->program_owner,
      support->producer_level, support->topology_epoch, support->materialization_generation,
      support->field_provider + "/forged"});
  EXPECT_THROW(
      V::field(field, -1, type.scientific_contract, {}, wrong_support, type.field_provider),
      std::invalid_argument);
  auto wrong_owner = type;
  wrong_owner.field_owner = 0;
  EXPECT_THROW(wrong_owner.validate(), std::invalid_argument);
}

TEST(ContinuationImage, RepresentationWithoutFluxSurvivesCopyVariantAndCodec) {
  Extent<D> cells{}, ranks{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 2;
    ranks[d] = 1;
  }
  const auto layout = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), cells);
  const auto distribution =
      mesh::Distribution<D>::replicated(layout, mesh::RankSpace<D>(Index<D>{}, ranks));
  MultiFab<D> field(layout, distribution, Index<D>{}, 1, Extent<D>{});
  field.set_val(Real(3));
  auto type = scalar(ContinuationKind::Field);
  type.field_owner = 0;
  type.field_components = 1;
  type.scientific_contract = "same-field-type";
  auto variant = product({"left", "right"}, {type, type});
  variant.kind = ContinuationKind::Variant;
  Codec codec(variant, "test/representation-only", 1ULL << 20,
              [&](int owner) -> const MultiFab<D>& {
                if (owner != 0)
                  throw std::invalid_argument("owner");
                return field;
              });
  const auto a = std::make_shared<const FieldRepresentationObligation>(
      FieldRepresentationObligation{"graph:K/capture:a=1", "evaluation:7"});
  const auto b = std::make_shared<const FieldRepresentationObligation>(
      FieldRepresentationObligation{"graph:K/capture:a=2", "evaluation:8"});
  const auto first = V::field(field, 0, type.scientific_contract, {}, {}, {}, a);
  const auto second = V::with_field_representation(first, b);
  EXPECT_EQ(first->field_storage_owner(), second->field_storage_owner());
  EXPECT_NE(codec.collective_identity(V::variant(0, first), 1),
            codec.collective_identity(V::variant(0, second), 1));
  for (std::uint32_t arm = 0; arm < 2; ++arm) {
    const auto selected = arm ? second : first;
    const auto bytes = codec.encode(V::variant(arm, selected), 1);
    const auto decoded = codec.decode(bytes).root;
    EXPECT_EQ(decoded->arm(), arm);
    ASSERT_TRUE(decoded->members()[0]->field_representation());
    EXPECT_EQ(*decoded->members()[0]->field_representation(), *selected->field_representation());
    EXPECT_FALSE(decoded->members()[0]->field_exchange_expression());
    EXPECT_EQ(codec.encode(decoded, 1), bytes);
    const auto archive = codec.decode_archive(bytes, 0, 1, [](std::uint64_t) {});
    EXPECT_EQ(codec.archive_collective_identity(archive, 1ULL << 20, [](std::uint64_t) {}),
              codec.collective_identity(decoded, 1));
  }
  const auto copied = V::with_field_metadata(first, first->field_metadata());
  EXPECT_EQ(copied->field_representation(), a);
  EXPECT_THROW(
      V::with_field_representation(first, std::make_shared<const FieldRepresentationObligation>(
                                              FieldRepresentationObligation{"graph", ""})),
      std::invalid_argument);
}

TEST(ContinuationImage, IndependentFieldStructuralPreflightPrecedesCopyPublication) {
  Extent<D> cells{}, ranks{};
  for (int d = 0; d < D; ++d) {
    cells[d] = 2;
    ranks[d] = 1;
  }
  const auto layout = mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells), cells);
  const auto distribution =
      mesh::Distribution<D>::replicated(layout, mesh::RankSpace<D>(Index<D>{}, ranks));
  MultiFab<D> field(layout, distribution, Index<D>{}, 1, Extent<D>{});
  field.set_val(Real(7));
  auto type = scalar(ContinuationKind::Field);
  type.field_components = 1;
  type.scientific_contract = "retained/role";
  type.field_provider = "problem:A/unknown:p";
  const auto support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
      "program", "time:0.125/invocation:4", -1, 2, 7, 19, type.field_provider});
  const ContinuationFieldMetadata metadata{{}, support, {}};
  const finite_structural::Limits limits{1ULL << 20, 1ULL << 20, 1ULL << 20};
  int preflights = 0, copies = 0;
  auto copied = [&](const auto&, const auto&, const auto&, const auto&) {
    ++copies;
    return metadata;
  };
  auto preflight = [&](const auto& source, const auto& schema, const auto&...) {
    ++preflights;
    if (&source != &field || schema.field_provider != type.field_provider)
      throw std::invalid_argument("stale source incarnation");
  };
  const std::any borrowed = std::cref(field);
  EXPECT_THROW(finite_structural::pack<D>(type, borrowed, limits, copied), std::invalid_argument);
  EXPECT_EQ(copies, 0);
  auto reject = [](const auto&, const auto&, const auto&...) {
    throw std::invalid_argument("revoked source incarnation");
  };
  EXPECT_THROW(finite_structural::pack<D>(type, borrowed, limits, copied, reject),
               std::invalid_argument);
  EXPECT_EQ(copies, 0);
  const auto retained = finite_structural::pack<D>(type, borrowed, limits, copied, preflight);
  EXPECT_EQ(preflights, 1);
  EXPECT_EQ(copies, 1);
  EXPECT_EQ(retained->field_support(), support);
  EXPECT_THROW(finite_structural::unpack<D>(type, retained, limits, copied, reject),
               std::invalid_argument);
  EXPECT_EQ(copies, 1);
  const auto accepted =
      finite_structural::unpack<D>(type, retained, limits, copied,
                                   [&](const auto& source, const auto& schema, const auto& proof) {
                                     ++preflights;
                                     EXPECT_EQ(&source, &retained->field());
                                     EXPECT_EQ(schema.field_provider, type.field_provider);
                                     EXPECT_EQ(proof.support, support);
                                   });
  EXPECT_EQ(preflights, 2);
  EXPECT_EQ(copies, 2);
  EXPECT_TRUE(std::any_cast<std::shared_ptr<MultiFab<D>>>(accepted));
}
