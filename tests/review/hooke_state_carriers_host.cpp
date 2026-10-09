// Independent host-only codec authority review. No native runtime or Kokkos/MPI.
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <cassert>
#include <iostream>
using namespace pops::runtime::checkpoint;
template<class F> void refuse(F f) { bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}assert(rejected); }
std::string pack(const StateCarrierArchive<1>& a){auto b=encode_state_carriers(a);return {reinterpret_cast<const char*>(b.data()),b.size()};}
int main(){
 StateCarrierArchive<1> a;a.real_bits=32;a.ranks=2;a.levels=1;a.blocks={"carrier"};
 StateCarrierPatch<1> p;p.owner=-1;p.components=1;p.lo={0};p.hi={0};p.grown_lo={-1};p.grown_hi={1};p.bits={0x80000000,0x7fc00042,0};a.patches={p};
 std::vector<std::string> shards;for(int r=0;r<2;r++){a.shard=r;shards.push_back(pack(a));}
 auto merged=merge_state_carrier_shards<1>(shards);validate_complete_state_carriers(merged);
 assert(merged.patches[0].bits==p.bits);
 auto mixed=a;mixed.shard=1;mixed.patches[0].owner=1;auto hostile=shards;hostile[1]=pack(mixed);refuse([&]{merge_state_carrier_shards<1>(hostile);});
 auto missing=a;missing.patches.clear();hostile=shards;hostile[1]=pack(missing);refuse([&]{merge_state_carrier_shards<1>(hostile);});
 auto width=a;width.patches[0].bits[0]=0x100000000ULL;refuse([&]{pack(width);});
 auto overlap=merged;overlap.patches.push_back(overlap.patches[0]);overlap.patches[1].patch=1;refuse([&]{validate_complete_state_carriers(overlap);});
 auto extreme=merged;extreme.patches[0].grown_lo[0]=INT64_MIN;extreme.patches[0].grown_hi[0]=INT64_MAX;refuse([&]{pack(extreme);});
 auto empty_block=merged;empty_block.blocks.push_back("absent");refuse([&]{validate_complete_state_carriers(empty_block);});
 std::cout<<"HOST_ONLY independent: mixed ownership, missing replica, 32-bit payload, overlap, extent overflow, missing block refused; signed-zero/NaN bits preserved\n";
}
