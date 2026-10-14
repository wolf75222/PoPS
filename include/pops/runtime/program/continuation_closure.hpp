#pragma once
#include <pops/runtime/program/continuation_image.hpp>

namespace pops::runtime::program {
// Collective graph closure before rank-local codecs. This retains every rank's shard of
// every reachable immutable Array, including a reference occurring on only one rank.
template<int Dim>
std::shared_ptr<const ContinuationClosure<Dim>> continuation_collective_closure(
    const ContinuationValueRef<Dim>& root, distributed::Registry* registry,
    const ExecutionLane& lane, std::uint64_t capacity, FiniteDomainRealmRef replicated_realm = {}) {
  using Ref=ContinuationValueRef<Dim>; using A=ContinuationDistributedArrayRef<Dim>;
  using Key=std::array<std::uint64_t,3>;
  auto key=[](distributed::Identity id){return Key{id.issuer[0],id.issuer[1],id.ordinal};};
  auto identity=[](const Key& k){return distributed::Identity{{k[0],k[1]},k[2]};};
  std::map<Key,distributed::DomainRef> domains; std::map<Key,A> arrays;
  const auto present=registry?1L:0L;
  if(all_reduce_min(present,lane)!=all_reduce_max(present,lane))
    throw std::logic_error("continuation registry presence differs across ranks");
  auto collect=[&](auto&& self,const Ref& value,unsigned depth)->void {
    if(!value || depth>128) throw std::invalid_argument("invalid continuation closure depth");
    if(const auto& d=value->distributed_domain_ref()) {
      if(!registry) throw std::logic_error("distributed continuation has no installed registry");
      registry->authenticate(d); domains.emplace(key(d->identity()),d);
    }
    if(const auto& a=value->distributed_array_ref()) {
      auto published=registry->resolve_value(a->version,distributed::array_type_tag<Dim>());
      if(published.get()!=a.get()) throw std::logic_error("continuation contains an unpublished Array version");
      if(a->index_domain) {registry->authenticate(a->index_domain);domains.emplace(key(a->index_domain->identity()),a->index_domain);}
      if(!arrays.emplace(key(a->version),a).second) return;
    }
    for(const auto& child:value->members()) self(self,child,depth+1);
  };
  distributed::phase(lane,[&]{collect(collect,root,0);});
  if(registry) {
    bool complete=false;
    for(std::uint64_t round=0;round<=registry->storage_versions();++round) {
      const auto old_arrays=arrays.size(),old_domains=domains.size();
      std::vector<std::uint8_t> local;
      distributed::phase(lane,[&]{
        continuation_detail::Writer out(capacity);out.word(domains.size());out.word(arrays.size());
        for(const auto& [id,d]:domains) for(auto word:id) out.word(word);
        for(const auto& [id,a]:arrays) for(auto word:id) out.word(word);
        local=std::move(out.bytes);
      });
      for(int owner=0;owner<lane.size();++owner) {
        std::uint64_t length=lane.rank()==owner?local.size():0;
        const auto words=all_reduce_max(static_cast<long>(length),lane);
        std::vector<std::uint8_t> received;
        distributed::phase(lane,[&]{
          if(words<0 || static_cast<std::uint64_t>(words)>capacity || words>INT_MAX)
            throw std::length_error("continuation closure metadata exceeds capacity");
          if(lane.rank()==owner) received=local;else received.resize(static_cast<std::size_t>(words));
        });
        distributed::broadcast(received,owner,lane);
        distributed::phase(lane,[&]{
          continuation_detail::Reader in(received);const auto nd=in.word(),na=in.word();
          if(nd>(received.size()-in.at)/24 || na>(received.size()-in.at-nd*24)/24)
            throw std::invalid_argument("continuation closure metadata is truncated");
          for(std::uint64_t i=0;i<nd;++i) {
            const Key id{in.word(),in.word(),in.word()};
            domains.emplace(id,registry->resolve_domain(identity(id)));
          }
          for(std::uint64_t i=0;i<na;++i) {
            const Key id{in.word(),in.word(),in.word()};
            auto a=std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
                registry->resolve_value(identity(id),distributed::array_type_tag<Dim>()));
            if(!a || key(a->version)!=id) throw std::logic_error("continuation closure version differs");
            collect(collect,ContinuationValue<Dim>::distributed_array(a),0);
          }
          if(in.at!=received.size()) throw std::invalid_argument("continuation closure metadata has trailing bytes");
        });
      }
      const long changed=arrays.size()!=old_arrays || domains.size()!=old_domains;
      if(all_reduce_max(changed,lane)==0) {complete=true;break;}
    }
    if(!complete) throw std::length_error("continuation closure did not fit its version budget");
  }
  std::shared_ptr<ContinuationClosure<Dim>> result;
  std::string descriptor;
  distributed::phase(lane,[&]{
    result=std::make_shared<ContinuationClosure<Dim>>();continuation_detail::Writer out(capacity);
    if(replicated_realm && replicated_realm->synchronized()) result->replicated_realm=replicated_realm;
    out.word(bool(result->replicated_realm));
    if(result->replicated_realm) for(auto word:result->replicated_realm->transport_identity()) out.word(word);
    out.word(domains.size());
    for(const auto& [id,d]:domains) {
      for(auto w:id)out.word(w);out.word(d->family());out.word(d->capacity());
      out.word(d->partition_offsets().size());for(auto x:d->partition_offsets())out.word(x);
      result->domains.push_back(d);
    }
    out.word(arrays.size());
    for(const auto& [id,a]:arrays) {
      for(auto w:id)out.word(w);for(auto w:key(a->domain->identity()))out.word(w);
      out.text(continuation_transport_type_identity(a->element,true));
      out.word(bool(a->index_domain));if(a->index_domain)for(auto w:key(a->index_domain->identity()))out.word(w);
      result->arrays.push_back(a);
    }
    descriptor.assign(reinterpret_cast<const char*>(out.bytes.data()),out.bytes.size());
  });
  if(!all_ranks_agree_exact_ordered_byte_pairs({{"continuation-closure",descriptor}},lane))
    throw std::invalid_argument("continuation closure identities/types/ownership differ across ranks");
  return result;
}
} // namespace pops::runtime::program
