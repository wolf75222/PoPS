#pragma once
#include <pops/runtime/program/continuation_closure.hpp>
#include <map>
#include <set>

namespace pops::runtime::program::distributed {
namespace gc_detail {
// A borrowed census of the exact strong-ownership graph. No shared_ptr copy is
// permitted until every count is complete. Values are immutable and the caller
// is a quiescent collective Program frontier, never a destructor/local region.
template<int Dim> class Census {
  using V=ContinuationValue<Dim>; using Ref=ContinuationValueRef<Dim>;
  using A=ContinuationDistributedArray<Dim>; using AR=ContinuationDistributedArrayRef<Dim>;
  struct Count {long total=0,internal=0;unsigned color=0;};
  struct ValueNode:Count {const Ref* handle=nullptr;};
  std::map<const Domain*,Count> domains_;
  std::map<const A*,Count> arrays_;
  std::map<const V*,ValueNode> values_;
  std::uint64_t remaining_;
  void charge_(){local_require(remaining_>0,Code::capacity);--remaining_;}
  void depth_(unsigned depth){local_require(depth<=256,Code::capacity);}
  void domain_edge_(const DomainRef& handle) {
    charge_();local_require(bool(handle),Code::domain);handle->require_live();
    auto found=domains_.find(handle.get());local_require(found!=domains_.end(),Code::domain);
    ++found->second.internal;
  }
  void array_edge_(const AR& handle,unsigned depth) {
    charge_();local_require(bool(handle),Code::domain);
    auto found=arrays_.find(handle.get());local_require(found!=arrays_.end(),Code::domain);
    ++found->second.internal;array_(found,depth);
  }
  void value_edge_(const Ref& handle,unsigned depth) {
    charge_();depth_(depth);local_require(bool(handle),Code::domain);
    auto [found,created]=values_.try_emplace(handle.get());
    auto& count=found->second;
    if(created){charge_();count.total=handle.use_count();count.handle=&handle;}
    ++count.internal;
    local_require(count.color!=1,Code::domain); // cyclic graphs are outside finite immutable values
    if(count.color==2)return;
    count.color=1;
    if(const auto& domain=handle->distributed_domain_ref())domain_edge_(domain);
    if(handle->kind()==ContinuationKind::DistributedArray) {
      // members() forwards to Array::values: these are owned by the Array, not
      // by each CV alias. Counting them here would silently collect live roots.
      array_edge_(handle->distributed_array_ref(),depth+1);
    } else for(const auto& member:handle->members())value_edge_(member,depth+1);
    count.color=2;
  }
  void array_(typename std::map<const A*,Count>::iterator found,unsigned depth) {
    depth_(depth);auto& count=found->second;
    local_require(count.color!=1,Code::domain);
    if(count.color==2)return;
    count.color=1;const auto* array=found->first;
    domain_edge_(array->domain);
    if(array->index_domain)domain_edge_(array->index_domain);
    for(const auto& value:array->values)value_edge_(value,depth+1);
    count.color=2;
  }
  static void valid_count_(const Count& count) {
    local_require(count.total>=count.internal&&count.internal>0,Code::internal);
  }
 public:
  // The serialized resource policy also bounds the census: each actual pointer
  // edge/node consumes one unit, less than its charged transport representation.
  explicit Census(std::uint64_t bytes):remaining_(bytes){}
  void register_domain(std::uint64_t ordinal,const DomainRef& domain) {
    charge_();local_require(domain&&domain->visible()&&domain->identity().ordinal==ordinal,Code::domain);
    local_require(domains_.emplace(domain.get(),Count{domain.use_count(),1,0}).second,Code::domain);
  }
  void register_array(std::uint64_t ordinal,const std::shared_ptr<const void>& handle,const std::string& type) {
    charge_();local_require(handle&&type==array_type_tag<Dim>(),Code::domain);
    const auto* array=static_cast<const A*>(handle.get());
    local_require(array->version.ordinal==ordinal&&array->domain&&
      array->version.issuer==array->domain->identity().issuer&&
      array->values.size()==std::size_t(array->domain->local_size()),Code::domain);
    local_require(arrays_.emplace(array,Count{handle.use_count(),1,0}).second,Code::domain);
  }
  void complete() {
    for(auto found=arrays_.begin();found!=arrays_.end();++found)array_(found,0);
    for(const auto& [_,count]:domains_)valid_count_(count);
    for(const auto& [_,count]:arrays_)valid_count_(count);
    for(const auto& [_,count]:values_)valid_count_(count);
  }
  Ref roots(Registry& registry)const {
    // Counts are final. Owning copies may now be taken without changing roots.
    std::vector<Ref> roots;
    for(const auto& [domain,count]:domains_)if(count.total>count.internal)
      roots.push_back(V::distributed_domain(registry.resolve_domain(domain->identity())));
    for(const auto& [array,count]:arrays_)if(count.total>count.internal)
      roots.push_back(V::distributed_array(std::static_pointer_cast<const A>(
        registry.resolve_value(array->version,array_type_tag<Dim>()))));
    for(const auto& [_,count]:values_)if(count.total>count.internal)roots.push_back(*count.handle);
    return V::product(std::move(roots));
  }
};
} // namespace gc_detail

// Stop-the-world for this Program's native host-control thread only. It must not
// race another host thread that acquires a new strong reference from a weak one.
// Accepted continuation roots, rollback snapshots, Prepared closures, temporaries
// and Failure payloads are ordinary external strong roots; none is subtracted as
// a disposable cache. The union reaches remote-only references and empty shards.
template<int Dim> void collect_garbage(Registry& registry) {
  const auto& lane=registry.lane();
  std::optional<gc_detail::Census<Dim>> census;
  ContinuationValueRef<Dim> root;
  phase(lane,[&]{
    census.emplace(registry.storage_bytes());
    registry.inspect_domains([&](auto id,const auto& domain){census->register_domain(id,domain);});
    registry.inspect_values([&](auto id,const auto& value,const auto& type,auto){census->register_array(id,value,type);});
    census->complete();root=census->roots(registry);
  });
  // Every operation above and every allocation in closure/plan converges before
  // any mutation. On failure all tables and exact byte charges remain unchanged.
  auto live=continuation_collective_closure<Dim>(root,&registry,lane,registry.storage_bytes());
  std::optional<Registry::SweepPlan> plan;
  phase(lane,[&]{
    std::set<std::uint64_t> arrays,domains;
    for(const auto& value:live->arrays)arrays.insert(value->version.ordinal);
    for(const auto& domain:live->domains)domains.insert(domain->identity().ordinal);
    plan.emplace(registry.prepare_sweep(arrays,domains));
  });
  // This path has no allocation, communication, authority mint or revocation.
  // Monotone issued IDs are deliberately not reused after collection.
  registry.apply_sweep(std::move(*plan));
}
} // namespace pops::runtime::program::distributed
