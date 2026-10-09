// SOURCE_ONLY: actual codec + actual ExactContractBuilder, without Kokkos/native runtime.
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/sha256.hpp>
#include <bit>
#include <fstream>
#include <iostream>
#include <string>
using namespace pops::runtime::checkpoint;
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 bool replicated=std::string(argv[2])=="replicated";
 StateCarrierArchive<2> a;a.real_bits=64;a.ranks=2;a.levels=1;a.blocks={"opaque-A","opaque-B"};
 for(unsigned block=0;block<2;++block)for(unsigned patch=0;patch<2;++patch){StateCarrierPatch<2> p;
  p.block=block;p.patch=patch;p.components=block+1;p.owner=replicated?-1:int(patch);
  p.lo={2*int(patch),0};p.hi={2*int(patch)+1,3};p.grown_lo={p.lo[0]-1,-1};p.grown_hi={p.hi[0]+1,4};
  for(unsigned component=0;component<p.components;++component)for(int y=-1;y<=4;++y)for(int x=int(p.grown_lo[0]);x<=p.grown_hi[0];++x){
   double value=100*block+10*component+y+.125*x;p.bits.push_back(std::bit_cast<std::uint64_t>(value));}
  p.bits[0]=0x8000000000000000ULL;p.bits[1]=0x7ff8000000000042ULL;
  a.patches.push_back(std::move(p));}
 validate_complete_state_carriers(a);auto wire=encode_state_carriers(a);
 std::ofstream raw(std::string(argv[1])+".bin",std::ios::binary);raw.write(reinterpret_cast<const char*>(wire.data()),wire.size());
 std::ofstream hashes(std::string(argv[1])+".hashes");
 for(int rank=0;rank<2;++rank)for(const auto& p:a.patches){if(p.owner!=-1&&p.owner!=rank)continue;
  std::uint64_t local=replicated?p.patch:0;
  pops::ExactContractBuilder payload;
  payload.text("pops.amr.rank-local-carrier-payload").scalar(std::uint32_t{1}).scalar(std::int32_t{2})
   .text("state").text(a.blocks[p.block]).scalar(std::int32_t(p.level)).scalar(local).scalar(p.patch)
   .scalar(std::int32_t(p.components)).scalar(std::uint64_t(p.bits.size()));
  for(int d=0;d<2;++d)payload.scalar(std::int32_t(p.lo[d])).scalar(std::int32_t(p.hi[d]));
  for(int d=0;d<2;++d)payload.scalar(std::int32_t(p.grown_lo[d])).scalar(std::int32_t(p.grown_hi[d]));
  for(auto bits:p.bits)payload.scalar(std::bit_cast<double>(bits));
  auto value=payload.str();std::vector<std::uint8_t> bytes(value.begin(),value.end());
  hashes<<rank<<" "<<p.block<<" "<<p.patch<<" "<<local<<" pops.amr.rank-local-carrier.v1:sha256:"<<pops::identity::sha256_hex(bytes)<<"\n";
 }
}
