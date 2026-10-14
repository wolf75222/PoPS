#include <pops/runtime/program/distributed_array.hpp>
#include <gtest/gtest.h>

namespace {
namespace d=pops::runtime::program::distributed;
namespace p=pops::runtime::program;
using V=p::ContinuationValue<2>;using Ref=p::ContinuationValueRef<2>;
using A=d::Array<2>;using AR=d::ArrayRef<2>;
using T=p::ContinuationType;using K=p::ContinuationKind;
std::vector<std::int64_t> offsets(const pops::ExecutionLane& lane) {
  std::vector<std::int64_t> out(lane.size()+1,1);out.front()=0;out.back()=4;return out;
}
AR make(d::Registry& registry,const d::DomainRef& domain,std::int64_t shift=0) {
  T integer{K::Int64};auto result=d::map<2>(registry,domain,integer,integer,4096,
    [&](d::Index i){return V::variant(1,V::integer(shift+i.gid));});
  d::phase(registry.lane(),[&]{d::local_require(result.ok(),d::Code::internal);});
  return std::move(result.value);
}

TEST(DistributedGcV14, RepeatedStepsReuseBudgetAndProtectRollbackAndFailure) {
  auto lane=pops::ExecutionLane::world();d::Registry registry(lane);registry.configure(65536,3);
  auto domain=registry.issue(741,4,offsets(lane));AR accepted;Ref failure;
  std::uint64_t previous=0;
  for(int step=0;step<32;++step) {
    d::collect_garbage<2>(registry);
    {
      auto rollback=accepted;
      auto next=make(registry,domain,100*step);
      EXPECT_GT(next->version.ordinal,previous);previous=next->version.ordinal;
      accepted=next;
      if(step==0)failure=V::distributed_array(accepted);
      if(step==8)failure.reset();
      // Collect with the rollback snapshot and new value simultaneously live.
      d::collect_garbage<2>(registry);
      if(rollback)for(std::size_t i=0;i<rollback->values.size();++i)
        EXPECT_EQ(rollback->values[i]->integer(),100*(step-1)+domain->begin()+std::int64_t(i));
    }
    d::collect_garbage<2>(registry);
    EXPECT_EQ(registry.retained_values(),std::uint64_t(step>0&&step<8?2:1));
    const auto expected=d::Registry::domain_charge(domain->partition_offsets().size())+
      registry.value_charge(accepted->version,d::array_type_tag<2>())+
      (step>0&&step<8?registry.value_charge(failure->distributed_array_ref()->version,d::array_type_tag<2>()):0);
    EXPECT_EQ(registry.retained_bytes(),expected);
  }
  accepted.reset();failure.reset();d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(),0);EXPECT_EQ(registry.retained_domains(),1);
  domain.reset();d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_domains(),0);EXPECT_EQ(registry.retained_bytes(),0);
}

TEST(DistributedGcV14, RemoteOnlyRootAndFailureRetainEveryShard) {
  auto lane=pops::ExecutionLane::world();d::Registry registry(lane);registry.configure(65536,4);
  auto domain=registry.issue(742,4,offsets(lane));auto source=make(registry,domain,30);
  const auto source_id=source->version;T integer{K::Int64};T array{K::DistributedArray,{integer},{},742,4};
  T element{K::Variant,{integer,array},{"empty","reference"}};
  p::ContinuationTransportSession<2> session;
  session.bind_distributed_registry(registry,[&](auto id,const auto& dom,const auto& t){return d::resolve_array<2>(registry,id,dom,t);});
  auto made=d::map<2>(registry,domain,element,integer,4096,[&](d::Index i){
    return V::variant(1,i.gid==3?V::variant(1,V::distributed_array(source)):V::variant(0,V::integer(i.gid)));
  },&session);
  d::phase(lane,[&]{d::local_require(made.ok(),d::Code::internal);});
  Ref root=lane.rank()==0?V::distributed_array(made.value):Ref{};
  made.value.reset();source.reset();d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(),2);
  Ref failure;
  if(lane.rank()==lane.size()-1)failure=V::distributed_array(d::resolve_array<2>(registry,source_id,domain,integer));
  root.reset();d::collect_garbage<2>(registry);
  EXPECT_EQ(registry.retained_values(),1);
  {
    const auto value=d::resolve_array<2>(registry,source_id,domain,integer);
    for(std::size_t i=0;i<value->values.size();++i)EXPECT_EQ(value->values[i]->integer(),30+domain->begin()+std::int64_t(i));
  }
  failure.reset();d::collect_garbage<2>(registry);EXPECT_EQ(registry.retained_values(),0);
}

TEST(DistributedGcV14, SharedProductEdgesAreCountedOncePerOwnershipEdge) {
  auto lane=pops::ExecutionLane::world();d::Registry registry(lane);registry.configure(65536,4);
  auto domain=registry.issue(743,4,offsets(lane));auto source=make(registry,domain,40);
  T integer{K::Int64};T array{K::DistributedArray,{integer},{},743,4};T product{K::Product,{array,array},{"left","right"}};
  auto shared=V::distributed_array(source);auto tuple=V::product({shared,shared});
  p::ContinuationTransportSession<2> session;
  session.bind_distributed_registry(registry,[&](auto id,const auto& dom,const auto& t){return d::resolve_array<2>(registry,id,dom,t);});
  auto made=d::map<2>(registry,domain,product,integer,4096,[&](d::Index){return V::variant(1,tuple);},&session);
  d::phase(lane,[&]{d::local_require(made.ok(),d::Code::internal);});
  source.reset();shared.reset();made.value.reset();if(lane.rank()!=0)tuple.reset();
  d::collect_garbage<2>(registry);EXPECT_EQ(registry.retained_values(),1);
  if(tuple)EXPECT_EQ(tuple->members()[0],tuple->members()[1]);
  tuple.reset();d::collect_garbage<2>(registry);EXPECT_EQ(registry.retained_values(),0);
}

TEST(DistributedGcV14, CorruptNativeCycleIsRejectedBeforeAnySweep) {
  auto lane=pops::ExecutionLane::world();d::Registry registry(lane);registry.configure(65536,4);
  auto domain=registry.issue(744,4,offsets(lane));
  auto pending=std::make_shared<A>(A{domain,T{K::Int64},{}});
  for(auto gid=domain->begin();gid<domain->end();++gid)pending->values.push_back(V::integer(gid));
  auto published=d::publish_array<2>(registry,pending,4096,nullptr);
  // Only raw C++ can mutate a published object into a recursive graph. The
  // finite typed frontend cannot construct this; collection still fails closed.
  if(!pending->values.empty())pending->values[0]=V::distributed_array(published);
  const auto bytes=registry.retained_bytes();
  EXPECT_THROW(d::collect_garbage<2>(registry),d::Failure);
  EXPECT_EQ(registry.retained_values(),1);EXPECT_EQ(registry.retained_bytes(),bytes);
  if(!pending->values.empty())pending->values[0]=V::integer(domain->begin());
  pending.reset();published.reset();d::collect_garbage<2>(registry);EXPECT_EQ(registry.retained_values(),0);
}

TEST(DistributedGcV14, OneRankPreparationFailureLeavesEveryVaultUnchanged) {
  auto lane=pops::ExecutionLane::world();d::Registry registry(lane);registry.configure(65536,4);
  auto domain=registry.issue(745,4,offsets(lane));auto dead=make(registry,domain,50);dead.reset();
  auto pending=std::make_shared<A>(A{domain,T{K::Int64},{}});
  for(auto gid=domain->begin();gid<domain->end();++gid)pending->values.push_back(V::integer(gid));
  pending->version=registry.publish_value(pending,lane.rank()==0?"unsupported-test-value":d::array_type_tag<2>(),1024);
  const auto bytes=registry.retained_bytes();const auto count=registry.retained_values();
  EXPECT_THROW(d::collect_garbage<2>(registry),d::Failure);
  EXPECT_EQ(registry.retained_values(),count);EXPECT_EQ(registry.retained_bytes(),bytes);
}
} // namespace
