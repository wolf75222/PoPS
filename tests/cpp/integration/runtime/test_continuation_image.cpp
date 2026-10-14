#include <gtest/gtest.h>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/program/continuation_authority.hpp>
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
ContinuationType scalar(ContinuationKind kind) { ContinuationType t; t.kind=kind; return t; }
ContinuationType product(std::vector<std::string> names,std::vector<ContinuationType> children) {
  ContinuationType t;t.names=std::move(names);t.children=std::move(children);return t;
}
Codec::Prototypes none() {
  return [](int)->const MultiFab<D>& { throw std::logic_error("test schema has no field"); };
}
ContinuationConfiguration configuration() {
  ContinuationConfiguration c;c.type=product({"time","step","counter"},
      {scalar(ContinuationKind::Float64),scalar(ContinuationKind::Int64),scalar(ContinuationKind::Int64)});
  c.identity="test/continuation/structural";c.byte_capacity=4096;c.time_path={0};c.step_path={1};return c;
}
R root(double time=0,std::int64_t step=0,std::int64_t counter=0) {
  return V::product({V::real(time),V::integer(step),V::integer(counter)});
}
R envelope(R next,std::int64_t increment=7) {
  return V::product({std::move(next),V::integer(increment),V::product({}),V::boolean(true)});
}
Store initial_store() {
  Store store;store.configure(configuration());store.publish(store.prepare(root(),none(),true));return store;
}
}

TEST(ContinuationImage, ExactInt64NanPayloadAndVariantRoundTrip) {
  auto sum=product({"integer","bits"},{scalar(ContinuationKind::Int64),scalar(ContinuationKind::Float64)});
  sum.kind=ContinuationKind::Variant;
  const auto type=product({"sum","huge"},{sum,scalar(ContinuationKind::Int64)});
  Codec codec(type,"test/typed-bits",4096,none());
  const auto nan=std::bit_cast<double>(UINT64_C(0x7ff8000000000005));
  auto value=V::product({V::variant(1,V::real(nan)),V::integer(INT64_C(9007199254740997))});
  const auto restored=codec.decode(codec.encode(value,41));
  EXPECT_EQ(restored.publication,41u);
  EXPECT_EQ(restored.root->members()[1]->integer(),INT64_C(9007199254740997));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(restored.root->members()[0]->members()[0]->real()),UINT64_C(0x7ff8000000000005));
}

TEST(ContinuationImage, DomainAliasesSurviveOneTableAndRealmIsRenewed) {
  FiniteDomainRegistry registry;auto domain=registry.issue(2,8,5);
  auto dt=scalar(ContinuationKind::Domain);dt.family=5;dt.capacity=8;
  auto it=dt;it.kind=ContinuationKind::Index;
  auto at=dt;at.kind=ContinuationKind::Array;at.children={scalar(ContinuationKind::Int64)};
  Codec codec(product({"domain","index","array"},{dt,it,at}),"test/aliases",4096,none());
  auto image=V::product({V::domain(domain),V::index(domain,1),V::array(domain,{V::integer(3),V::integer(9)})});
  auto restored=codec.decode(codec.encode(image,2));
  const auto a=restored.root->members()[0]->domain_ref();
  EXPECT_EQ(a,restored.root->members()[1]->domain_ref());
  EXPECT_EQ(a,restored.root->members()[2]->domain_ref());
  EXPECT_NE(a,domain);EXPECT_NE(a->realm(),domain->realm());
  EXPECT_EQ(a->version,domain->version);
  registry.import_high_water(restored.max_domain_version);
  EXPECT_GT(registry.issue(0,8,5,restored.realm)->version,a->version);
}

TEST(ContinuationImage, RejectsForeignRealmBoundsWrongTypeAndCorruption) {
  FiniteDomainRegistry first,second;
  auto dt=scalar(ContinuationKind::Domain);dt.family=3;dt.capacity=4;
  Codec codec(product({"a","b"},{dt,dt}),"test/realm",4096,none());
  EXPECT_THROW(codec.encode(V::product({V::domain(first.issue(0,4,3)),V::domain(second.issue(0,4,3))}),0),std::invalid_argument);
  auto store=initial_store();auto bytes=store.checkpoint(none());bytes.push_back(0);
  EXPECT_THROW(store.prepare_restart(bytes,none()),std::invalid_argument);
  bytes.pop_back();bytes.pop_back();EXPECT_THROW(store.prepare_restart(bytes,none()),std::invalid_argument);
  EXPECT_THROW(store.prepare(V::integer(1),none()),std::invalid_argument);
  Codec tiny(configuration().type,"test/continuation/structural",32,none());
  EXPECT_THROW(tiny.encode(root(),0),std::length_error);
}

TEST(ContinuationImage, ExtendedLocalRealmsRoundTripAndAmbiguousIdentitiesAreRejected) {
  FiniteDomainRegistry replicated, local_a(false), local_b(false);
  auto dt=scalar(ContinuationKind::Domain);dt.family=3;dt.capacity=4;
  const auto a=local_a.issue(1,4,3),b=local_b.issue(2,4,3);
  Codec codec(product({"global","local_a","local_b","alias_a"},{dt,dt,dt,dt}),
              "test/local-realms",4096,none());
  auto original=V::product({V::domain(replicated.issue(0,4,3)),V::domain(a),
                            V::domain(b),V::domain(a)});
  const auto restored=codec.decode(codec.encode(original,7));
  ASSERT_EQ(restored.realms.size(),3u);
  ASSERT_EQ(restored.root->members().size(),4u);
  const auto global=restored.root->members()[0]->domain_ref();
  const auto ra=restored.root->members()[1]->domain_ref();
  const auto rb=restored.root->members()[2]->domain_ref();
  EXPECT_EQ(restored.publication,7u);
  EXPECT_EQ(restored.realm,global->realm());
  EXPECT_TRUE(global->realm()->replicated());
  EXPECT_FALSE(ra->realm()->replicated());EXPECT_FALSE(rb->realm()->replicated());
  EXPECT_NE(ra->realm(),rb->realm());EXPECT_NE(ra->realm(),a->realm());
  EXPECT_EQ(ra,restored.root->members()[3]->domain_ref());
  EXPECT_EQ(ra->version,rb->version); // equal ordinals from distinct issuers remain distinct
  EXPECT_EQ(ra->size,1);EXPECT_EQ(rb->size,2);

  Codec pair(product({"a","b"},{dt,dt}),"test/extended-authority-rejection",4096,none());
  ContinuationTransportSession<D> sender, first_receiver, second_receiver;
  const auto wire=sender.encode(dt,V::domain(a),4096);
  const auto first=first_receiver.decode(dt,wire,4096);
  const auto second=second_receiver.decode(dt,wire,4096);
  ASSERT_NE(first->domain_ref()->realm(),second->domain_ref()->realm());
  ASSERT_EQ(first->domain_ref()->realm()->unexported_identity(),
            second->domain_ref()->realm()->unexported_identity());
  EXPECT_THROW(pair.encode(V::product({first,second}),0),std::invalid_argument);

  FiniteDomainRegistry another_local(false);
  const auto repeated=another_local.issue(1,4,3,a->realm());
  ASSERT_NE(a,repeated);ASSERT_EQ(a->realm(),repeated->realm());
  ASSERT_EQ(a->version,repeated->version);
  EXPECT_THROW(pair.encode(V::product({V::domain(a),V::domain(repeated)}),0),std::invalid_argument);
}

TEST(ContinuationImage, StoreSnapshotRollbackAndRestartFrontier) {
  auto store=initial_store();auto before=store;
  store.publish(store.prepare(root(.25,1,4),none()));store.require_frontier(.25,1);
  EXPECT_THROW(store.require_frontier(0,0),std::logic_error);
  store.swap(before);store.require_frontier(0,0);
  auto archived=before.checkpoint(none());
  store.publish(store.prepare_restart(archived,none()));store.require_frontier(.25,1);
  EXPECT_EQ(store.read()->members()[2]->integer(),4);
}

TEST(ContinuationAuthority, RepeatedProducerCannotReuseAssessmentAndTransformInvalidates) {
  auto store=initial_store();PreparedResourceCache cache;Authority authority;
  authority.begin(cache.begin_attempt());auto base=authority.read(store);
  auto first=authority.freeze(base,V::real(.1),envelope(root(.1,1,1)));
  auto assessment=authority.attest(first,V::boolean(true),true,"graph/check");
  auto second=authority.freeze(base,V::real(.2),envelope(root(.2,1,2)));
  EXPECT_THROW(authority.accept(second,assessment),std::logic_error);
  auto accepted=authority.accept(first,assessment);
  authority.supersede(first);
  EXPECT_THROW(accepted.value(),std::logic_error);
  EXPECT_THROW(assessment.accepted(),std::logic_error);
  EXPECT_NO_THROW(second.value());
}

TEST(ContinuationAuthority, ChildCannotRevokeParentAndFailureRevokesChild) {
  auto store=initial_store();PreparedResourceCache cache;Authority authority;
  authority.begin(cache.begin_attempt());auto base=authority.read(store);
  auto parent=authority.freeze(base,V::real(.1),envelope(root(.1,1,1)));
  Authority::Candidate child;
  {
    auto scope=authority.begin_scope();
    EXPECT_THROW(authority.supersede(parent),std::logic_error);
    child=authority.freeze(base,V::real(.1),envelope(root(.1,1,2)));
  }
  EXPECT_THROW(child.value(),std::logic_error);EXPECT_NO_THROW(parent.value());
}

TEST(ContinuationAuthority, PromotionKeepsWholeCandidateAndConsumesDeltaOnce) {
  auto store=initial_store();PreparedResourceCache cache;Authority authority;
  authority.begin(cache.begin_attempt());auto base=authority.read(store);
  auto scope=authority.begin_scope();auto candidate=authority.freeze(base,V::real(.2),envelope(root(.2,1,3),11));
  auto assessment=authority.attest(candidate,V::boolean(true),true,"graph/check");
  auto accepted=authority.promote(std::move(scope),authority.accept(candidate,assessment));
  EXPECT_EQ(accepted.value(),candidate.value());
  EXPECT_TRUE(assessment.accepted());
  EXPECT_EQ(accepted.value()->members().size(),4u);
  EXPECT_EQ(authority.consume_increment(accepted)->integer(),11);
  EXPECT_THROW(authority.consume_increment(accepted),std::logic_error);
  EXPECT_EQ(authority.require_root_publication(accepted,store),accepted.value()->members()[0]);
  auto provisional=authority.provisional_base(accepted);EXPECT_EQ(provisional.root(),accepted.value()->members()[0]);
  authority.supersede(candidate);EXPECT_THROW(provisional.root(),std::logic_error);
}

TEST(ContinuationAuthority, NewAttemptRevokesAllOldCapabilities) {
  auto store=initial_store();PreparedResourceCache cache;Authority authority;
  authority.begin(cache.begin_attempt());auto base=authority.read(store);
  auto candidate=authority.freeze(base,V::real(.1),envelope(root(.1,1,1)));
  auto accepted=authority.accept(candidate,authority.attest(candidate,V::boolean(true),true,"graph/check"));
  authority.begin(cache.begin_attempt());
  EXPECT_THROW(base.root(),std::logic_error);
  EXPECT_THROW(accepted.value(),std::logic_error);
  EXPECT_NO_THROW(authority.read(store));
}

TEST(ContinuationImage, FieldSnapshotIsOwnedAndRestoresExactLayout) {
  Extent<D> cells{}, ghosts{};for(int d=0;d<D;++d){cells[d]=4;ghosts[d]=1;}
  const auto box=Box<D>::from_extents(cells);
  const auto layout=mesh::BoxArray<D>::from_domain(box,cells);
  Extent<D> ranks_extent{};for(int d=0;d<D;++d)ranks_extent[d]=1;
  const mesh::RankSpace<D> ranks(Index<D>{},ranks_extent);
  const auto distribution=mesh::Distribution<D>::replicated(layout,ranks);
  MultiFab<D> field(layout,distribution,Index<D>{},2,ghosts);field.set_val(Real(7));
  auto snapshot=V::field(field,0,"role/read-write/owner0");field.set_val(Real(9));
  auto type=scalar(ContinuationKind::Field);type.field_owner=0;type.field_components=2;
  type.scientific_contract="role/read-write/owner0";
  Codec codec(type,"test/owned-field",1024*1024,[&](int owner)->const MultiFab<D>&{
    if(owner!=0)throw std::logic_error("foreign owner");return field;
  });
  auto decoded=codec.decode(codec.encode(snapshot,0));
  auto host=decoded.root->field().fab(0).create_host_mirror();
  decoded.root->field().fab(0).copy_to_host(host);
  for(std::size_t i=0;i<host.size();++i)EXPECT_EQ(host(i),Real(7));
  EXPECT_THROW(require_continuation_projection<D>(decoded.root->field(),field),std::logic_error);
  field.set_val(Real(7));EXPECT_NO_THROW(require_continuation_projection<D>(decoded.root->field(),field));
}

TEST(ContinuationAuthority, ParentRejectionRevokesPromotedDescendants) {
  auto store=initial_store();PreparedResourceCache cache;Authority authority;
  authority.begin(cache.begin_attempt());auto base=authority.read(store);
  Authority::Accepted accepted;Authority::Base derived;
  {
    auto parent=authority.begin_scope();auto child=authority.begin_scope();
    auto candidate=authority.freeze(base,V::real(.2),envelope(root(.2,1,3)));
    accepted=authority.promote(std::move(child),
        authority.accept(candidate,authority.attest(candidate,V::boolean(true),true,"graph/check")));
    derived=authority.provisional_base(accepted);EXPECT_NO_THROW(derived.root());
  }
  EXPECT_THROW(accepted.value(),std::logic_error);
  EXPECT_THROW(derived.root(),std::logic_error);
  EXPECT_NO_THROW(base.root());
}

TEST(ContinuationImage, ExactRankFieldShardsIncludeEmptyRankWithoutRankZeroSubstitution) {
  Extent<D> cells{},ghosts{},ranks_extent{};
  for(int d=0;d<D;++d){cells[d]=2;ghosts[d]=0;ranks_extent[d]=1;}ranks_extent[0]=2;
  const auto layout=mesh::BoxArray<D>::from_domain(Box<D>::from_extents(cells),cells);
  const mesh::RankSpace<D> ranks(Index<D>{},ranks_extent);
  const auto distribution=mesh::Distribution<D>::partitioned(layout,ranks,{Index<D>{}});
  Index<D> rank1{};rank1[0]=1;
  MultiFab<D> owned(layout,distribution,Index<D>{},1,ghosts),empty(layout,distribution,rank1,1,ghosts);
  owned.set_val(Real(3));EXPECT_EQ(empty.local_size(),0u);
  auto type=scalar(ContinuationKind::Field);type.field_owner=0;type.field_components=1;type.scientific_contract="field/role";
  Codec owner_codec(type,"test/shards",65536,[&](int)->const MultiFab<D>&{return owned;});
  Codec empty_codec(type,"test/shards",65536,[&](int)->const MultiFab<D>&{return empty;});
  auto owner_bytes=owner_codec.encode(V::field(owned,0,"field/role"),0);
  auto empty_bytes=empty_codec.encode(V::field(empty,0,"field/role"),0);
  EXPECT_EQ(empty_codec.decode(empty_bytes).root->field().local_size(),0u);
  EXPECT_THROW(empty_codec.decode(owner_bytes),std::invalid_argument);
  EXPECT_THROW(owner_codec.decode(empty_bytes),std::invalid_argument);
}

TEST(ContinuationImage, RestartRevokesOldAliasesButPreservesDecodedRealm) {
  FiniteDomainRegistry registry;
  const auto old = registry.issue(2, 4, 7);
  auto dt = scalar(ContinuationKind::Domain); dt.family = 7; dt.capacity = 4;
  auto it = dt; it.kind = ContinuationKind::Index;
  auto at = dt; at.kind = ContinuationKind::Array; at.children = {scalar(ContinuationKind::Int64)};
  Codec codec(product({"d", "i", "a"}, {dt, it, at}), "restart/realm", 4096, none());
  const auto old_index = V::index(old, 1);
  const auto old_array = V::array(old, {V::integer(3), V::integer(8)});
  const auto original = V::product({V::domain(old), old_index, old_array});
  auto decoded = codec.decode(codec.encode(original, 1));
  const auto restored = decoded.root->members()[0]->domain_ref();
  registry.adopt(decoded.realm);
  registry.revoke_on_restart(decoded.realm);
  EXPECT_FALSE(old->visible()); EXPECT_TRUE(restored->visible());
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
  EXPECT_FALSE(old->visible()); EXPECT_FALSE(registry.realm());
  auto next = registry.issue(1, 4, 2);
  EXPECT_TRUE(next->visible()); EXPECT_NE(next->realm(), old->realm());
  EXPECT_GT(next->version, version);
  EXPECT_THROW(registry.issue(1, 4, 2, old->realm()), std::logic_error);
}

TEST(ContinuationImage, RejectingTrialDoesNotRevokeAcceptedDomainData) {
  FiniteDomainRegistry registry;
  auto domain = registry.issue(1, 2, 1);
  auto store = initial_store(); PreparedResourceCache cache; Authority authority;
  authority.begin(cache.begin_attempt());
  auto base = authority.read(store);
  auto child = authority.begin_scope();
  auto candidate = authority.freeze(base, V::real(.1), envelope(root(.1, 1, 1)));
  authority.revoke();
  EXPECT_THROW(candidate.value(), std::logic_error);
  EXPECT_TRUE(domain->visible());
  EXPECT_EQ(V::index(domain, 0)->index_offset(), 0);
}
