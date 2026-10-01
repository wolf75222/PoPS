// Bounded host oracle: no PoPS native runtime, Kokkos, MPI, build or JIT.
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <cassert>
#include <functional>
#include <iostream>
using namespace pops::runtime::checkpoint;
void rejects(const std::function<void()>& f) { bool rejected=false; try { f(); } catch(const std::invalid_argument&) { rejected=true; } assert(rejected); }
int main() {
  StateCarrierArchive<2> a; a.real_bits=64; a.ranks=2; a.levels=1; a.blocks={"opaque-A","opaque-B"};
  for(unsigned block=0;block<2;++block) for(unsigned patch=0;patch<2;++patch) {
    StateCarrierPatch<2> p; p.block=block;p.patch=patch;p.components=block+1;p.owner=patch;
    p.lo={std::int64_t(2*patch),0};p.hi={std::int64_t(2*patch+1),1};
    p.grown_lo={p.lo[0]-1,-1};p.grown_hi={p.hi[0]+1,2};
    p.bits.assign(16*p.components,0x8000000000000000ULL); // signed-zero ghosts
    p.bits[5]=0x3ff0000000000000ULL; p.bits[6]=0x7ff8000000000042ULL; // exact NaN payload
    a.patches.push_back(p);
  }
  validate_complete_state_carriers(a);
  auto bytes=encode_state_carriers(a);auto b=decode_state_carriers<2>(bytes);
  assert(b.patches==a.patches);assert(encode_state_carriers(b)==bytes);
  auto changed=a;changed.patches[0].bits[0]=0;assert(encode_state_carriers(changed)!=bytes); // ghost bit matters
  std::vector<std::string> shards;
  for(int rank=0;rank<2;++rank){auto s=a;s.shard=rank;s.patches.clear();for(auto p:a.patches)if(p.owner==rank)s.patches.push_back(p);auto v=encode_state_carriers(s);shards.emplace_back(reinterpret_cast<const char*>(v.data()),v.size());}
  assert(merge_state_carrier_shards<2>(shards).patches==a.patches);
  std::vector<std::string> replicas;
  auto replicated=a;for(auto& p:replicated.patches)p.owner=-1;
  for(int rank=0;rank<2;++rank){replicated.shard=rank;auto v=encode_state_carriers(replicated);replicas.emplace_back(reinterpret_cast<const char*>(v.data()),v.size());}
  auto merged_replicas=merge_state_carrier_shards<2>(replicas);
  assert(merged_replicas.patches.size()==a.patches.size());
  replicated.patches[0].bits[0]=0;auto divergent=encode_state_carriers(replicated);
  replicas[1].assign(reinterpret_cast<const char*>(divergent.data()),divergent.size());
  rejects([&]{merge_state_carrier_shards<2>(replicas);});
  replicated.patches.erase(replicated.patches.begin());auto missing=encode_state_carriers(replicated);
  replicas[1].assign(reinterpret_cast<const char*>(missing.data()),missing.size());
  rejects([&]{merge_state_carrier_shards<2>(replicas);});
  std::swap(shards[0],shards[1]);rejects([&]{merge_state_carrier_shards<2>(shards);});
  changed=a;changed.patches.insert(changed.patches.begin(),changed.patches[0]);rejects([&]{encode_state_carriers(changed);});
  changed=a;changed.patches[0].grown_hi[0]++;rejects([&]{encode_state_carriers(changed);});
  changed=a;changed.levels=UINT64_MAX;rejects([&]{validate_complete_state_carriers(changed);});
  changed=a;changed.patches[0].owner=2;rejects([&]{encode_state_carriers(changed);});
  changed=a;changed.patches.erase(changed.patches.begin());rejects([&]{validate_complete_state_carriers(changed);});
  changed=a;changed.patches.back().lo[0]++;rejects([&]{validate_complete_state_carriers(changed);});
  auto trailing=bytes;trailing.push_back(0);rejects([&]{decode_state_carriers<2>(trailing);});
  for(std::size_t cut=0;cut<bytes.size();++cut)rejects([&]{decode_state_carriers<2>({bytes.data(),cut});});
  rejects([&]{decode_state_carriers<3>(bytes);});
  std::cout<<"SOURCE_ONLY codec: ghost bits, signed zero/NaN, N components, shards and refusal cases passed\n";
}
