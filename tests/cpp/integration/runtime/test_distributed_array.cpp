// PREPARED ONLY. Native PoPS execution-lane/codec test; compile and run under MPI2 and MPI3.
// This is not a replacement runtime and does not qualify the pending full Program dispatch bridge.
#include <pops/runtime/program/distributed_array.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
namespace d=pops::runtime::program::distributed;
namespace p=pops::runtime::program;
using V=p::ContinuationValue<2>;using T=p::ContinuationType;using K=p::ContinuationKind;
static constexpr std::int64_t large=(std::int64_t(1)<<60)+17;
void check(const pops::ExecutionLane& lane,bool valid){d::phase(lane,[&]{d::local_require(valid,d::Code::internal);});}
template<class F> void rejects(const pops::ExecutionLane& lane,d::Code expected,F&& operation) {
  bool caught=false;try{operation();}catch(const d::Failure& e){caught=e.code==expected;}
  check(lane,caught);
}
int run() {
  auto lane=pops::ExecutionLane::world();const int n=lane.size(),rank=lane.rank();
  check(lane,n==2||n==3);d::Registry registry(lane);
  std::vector<std::int64_t> owners(n+1,1),requested(n+1,2);
  owners.front()=requested.front()=0;owners.back()=2;requested.back()=3;
  auto D=registry.issue(10,4,owners),Q=registry.issue(11,4,requested);
  T integer{K::Int64},real{K::Float64},boolean{K::Boolean};T tag,element,error;
  d::phase(lane,[&]{tag=T{K::Variant,{boolean,integer},{"boolean","integer"}};
    element=T{K::Product,{integer,tag},{"id","tag"}};
    error=T{K::Product,{integer,real},{"code","defect"}};});
  auto value=[](std::int64_t gid){return V::product({V::integer(large+10*gid),
    V::variant(unsigned(gid%2),gid%2?V::integer(99):V::boolean(true))});};
  auto made=d::map<2>(registry,D,element,error,4096,[&](d::Index i){return V::variant(1,value(i.gid));});
  check(lane,made.ok());const auto source=made.value;
  std::vector<std::vector<std::uint8_t>> before;
  d::phase(lane,[&]{for(const auto& x:source->values)before.push_back(p::encode_continuation_transport<2>(element,x,4096));});
  auto req=d::requests(registry,Q,D,[&](d::Index i){return d::index(D,i.gid<2?1:0);});
  auto imported=d::import_values(registry,source,req,65536);
  bool exact=imported->values.size()==std::size_t(Q->local_size());
  for(std::size_t j=0;j<imported->values.size();++j){const auto wanted=(Q->begin()+std::int64_t(j))<2?1:0;
    const auto& x=imported->values[j];exact=exact&&x->members()[0]->integer()==large+10*wanted;
    exact=exact&&x->members()[1]->arm()==unsigned(wanted%2);
    exact=exact&&(wanted%2?x->members()[1]->members()[0]->integer()==99:x->members()[1]->members()[0]->boolean());}
  check(lane,exact); // independent global formula, exact Int64, duplicate positions, empty rank

  auto C=d::csr_offsets(registry,D,D,req,4,[&](std::int64_t row){return row==0?0:Q->local_size();});
  auto via_csr=d::import_csr(registry,source,C,65536);
  check(lane,via_csr->values.size()==imported->values.size());
  for(std::size_t j=0;j<via_csr->values.size();++j)
    exact=exact&&via_csr->values[j]->members()[0]->integer()==imported->values[j]->members()[0]->integer();
  check(lane,exact);

  // Full E is selected at the first failure of the lowest failing rank, including >2^53.
  auto failed=d::map<2>(registry,Q,element,error,4096,[&](d::Index i){
    if(i.gid==1)return V::variant(0,V::product({V::integer(large),V::real(.25)}));
    return V::variant(1,value(i.gid));});
  check(lane,!failed.ok()&&failed.rank==0&&failed.gid==1&&failed.code==d::Code::user&&
    failed.payload->members()[0]->integer()==large&&failed.payload->members()[1]->real()==.25);
  auto arithmetic=d::map<2>(registry,D,element,error,4096,[&](d::Index i){
    if(i.gid==1)(void)d::integer(3,1,0);return V::variant(1,value(i.gid));});
  check(lane,!arithmetic.ok()&&arithmetic.code==d::Code::arithmetic&&arithmetic.rank==n-1&&arithmetic.gid==1);

  // Error only on one rank must be observed before any peer enters the next communication.
  rejects(lane,d::Code::index,[&]{auto bad=req;if(rank==n-1)bad.indices[0].gid=100;
    (void)d::import_values(registry,source,bad,65536);});
  rejects(lane,d::Code::capacity,[&]{(void)d::import_values(registry,source,req,rank==0?1:65536);});
  rejects(lane,d::Code::csr,[&]{(void)d::csr_offsets(registry,D,D,req,4,[&](std::int64_t row){
    return row==0?0:Q->local_size()+(rank==n-1?1:0);});});
  rejects(lane,d::Code::allocation,[&]{d::phase(lane,[&]{if(rank==(n==3?1:0))throw std::bad_alloc();});});

  // A new same-family/same-shape instance cannot authenticate old indices.
  const auto issued=registry.issued();auto fresh=registry.issue(10,4,owners);
  check(lane,registry.issued()==issued+1&&fresh->identity().ordinal>D->identity().ordinal);
  rejects(lane,d::Code::index,[&]{auto forged=req;if(rank==n-1)forged.indices[0].domain=fresh;
    (void)d::import_values(registry,source,forged,65536);});
  d::Registry restarted(lane);
  rejects(lane,d::Code::domain,[&]{d::phase(lane,[&]{restarted.authenticate(D);});});
  rejects(lane,d::Code::domain,[&]{auto divergent=owners;if(rank==n-1)divergent.back()=3;
    (void)registry.issue(19,4,divergent);});

  // No attempted failure changed an already published immutable array.
  exact=source==made.value;
  d::phase(lane,[&]{for(std::size_t j=0;j<before.size();++j)
    exact=exact&&before[j]==p::encode_continuation_transport<2>(element,source->values[j],4096);});
  check(lane,exact);
  // All-zero sizes still enter the count exchanges and produce a complete empty result.
  std::vector<std::int64_t> zeros(n+1,0);auto Z=registry.issue(20,1,zeros);
  auto z=d::map<2>(registry,Z,element,error,4096,[&](d::Index i){return V::variant(1,value(i.gid));});
  auto zr=d::requests(registry,Z,Z,[&](d::Index i){return i;});
  auto zi=d::import_values(registry,z.value,zr,4096);check(lane,zi->values.empty());
  if(rank==0)std::cout<<"distributed MPI protocol prepared checks passed on "<<n<<" ranks\n";
  return 0;
}
TEST(DistributedArray, ExactNativeProtocolWithAsymmetricOwnership) {
  const auto lane = pops::ExecutionLane::world();
  if (lane.size() != 2 && lane.size() != 3) GTEST_SKIP() << "requires MPI2 or MPI3";
  EXPECT_EQ(run(), 0);
}

TEST(DistributedArray, RetainedIssuerIsRevokedAndRankLocalRevocationConverges) {
  auto lane = pops::ExecutionLane::world();
  d::Registry registry(lane);
  std::vector<std::int64_t> offsets(lane.size() + 1, 1); offsets.front() = 0;
  auto domain = registry.issue(61, 2, offsets);
  const auto retained_index = d::index(domain, 0);
  auto local = std::make_shared<d::Array<2>>();
  local->domain = domain; local->element = T{K::Int64};
  local->values.resize(std::size_t(domain->local_size()), V::integer(3));
  d::ArrayRef<2> array = local;
  if (lane.rank() == 0) registry.revoke();
  // The revocation is deliberately inconsistent here. Registry.common must converge the error
  // before any peer performs a domain-sized operation or communication.
  rejects(lane, d::Code::domain, [&]{ registry.common(domain); });
  registry.revoke();
  EXPECT_FALSE(domain->visible());
  EXPECT_THROW(domain->size(), d::Failure);
  EXPECT_THROW(d::index(domain, 0), d::Failure);
  EXPECT_THROW(d::at_local<2>(array, retained_index), d::Failure);
  rejects(lane, d::Code::domain, [&]{ (void)registry.issue(61, 2, offsets); });
}

TEST(DistributedArray, DestroyingRegistryRevokesSurvivingDomainWitness) {
  auto lane = pops::ExecutionLane::world();
  d::DomainRef old;
  {
    d::Registry registry(lane);
    std::vector<std::int64_t> offsets(lane.size() + 1, 1); offsets.front() = 0;
    old = registry.issue(62, 2, offsets);
  }
  EXPECT_FALSE(old->visible());
  EXPECT_THROW(old->local_size(), d::Failure);
}
