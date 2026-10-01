// Pure STD host oracle and comparative benchmark. OLD_HEADER is the frozen 86a71b6 header
// instrumented only with one uint64 pair counter; no production native TU is compiled.
#include <pops/runtime/checkpoint/state_carriers.hpp>
#ifdef OLD_HEADER
#include OLD_HEADER
#endif
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace pops::runtime::checkpoint;
template<int D> StateCarrierArchive<D> fixture(std::size_t count) {
  StateCarrierArchive<D> a;a.real_bits=64;a.ranks=1;a.levels=1;a.blocks={"opaque-one","opaque-two"};
  auto side=std::size_t(std::ceil(std::pow(double(count),1./D)));
  while(std::pow(double(side),D)<count)++side;
  for(unsigned block=0;block<2;++block)for(std::size_t i=0;i<count;++i){StateCarrierPatch<D> p;p.block=block;p.patch=i;p.components=2;
    auto index=i;std::size_t cells=2;for(int d=0;d<D;++d){p.lo[d]=p.hi[d]=2*std::int64_t(index%side);index/=side;p.grown_lo[d]=p.lo[d]-1;p.grown_hi[d]=p.hi[d]+1;cells*=3;}
    p.bits.assign(cells,0x8000000000000000ULL);a.patches.push_back(std::move(p));}
  return a;
}
template<class F> bool refuses(F&& f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
template<int D> void oracles(){
  auto a=fixture<D>(137);validate_complete_state_carriers(a);
  auto b=a;std::int64_t maximum=0;for(auto& p:a.patches)for(int d=0;d<D;++d)maximum=std::max(maximum,p.grown_hi[d]);
  auto shift=INT64_MAX-maximum;for(auto& p:b.patches)for(int d=0;d<D;++d){p.lo[d]+=shift;p.hi[d]+=shift;p.grown_lo[d]+=shift;p.grown_hi[d]+=shift;}validate_complete_state_carriers(b);
  for(auto& p:b.patches)for(int d=0;d<D;++d){p.lo[d]-=shift;p.hi[d]-=shift;p.grown_lo[d]-=shift;p.grown_hi[d]-=shift;}
  auto lowshift=INT64_MIN+1;for(auto& p:b.patches)for(int d=0;d<D;++d){p.lo[d]+=lowshift;p.hi[d]+=lowshift;p.grown_lo[d]+=lowshift;p.grown_hi[d]+=lowshift;}validate_complete_state_carriers(b);
  b=a;for(auto& q:b.patches)for(int d=0;d<D;++d){auto location=q.patch%2 ? INT64_MAX-4-2*std::int64_t(q.patch) : INT64_MIN+4+2*std::int64_t(q.patch);q.lo[d]=q.hi[d]=location;q.grown_lo[d]=location-1;q.grown_hi[d]=location+1;}validate_complete_state_carriers(b);
  auto many=fixture<D>(4096);StateCarrierSpatialValidationStats complexity;validate_complete_state_carriers(many,&complexity);
  if(complexity.node_tests>=4096*100 || complexity.leaf_pairs!=0)throw std::runtime_error("regular AMR geometry candidate growth regressed");
  auto levels=a;levels.levels=3;levels.patches.clear();for(unsigned block=0;block<2;++block)for(unsigned level=0;level<3;++level)for(std::size_t i=0;i<137;++i){auto q=a.patches[block*137+i];q.level=level;levels.patches.push_back(q);}validate_complete_state_carriers(levels);
  levels.patches.back().owner=-1;if(!refuses([&]{validate_complete_state_carriers(levels);}))throw std::runtime_error("cached source range missed owner mismatch");
  b=a;for(unsigned block=0;block<2;++block){auto& q=b.patches[block*137+136];const auto& first=b.patches[block*137];q.lo=first.lo;q.hi=first.hi;q.grown_lo=first.grown_lo;q.grown_hi=first.grown_hi;}
  if(!refuses([&]{validate_complete_state_carriers(b);}))throw std::runtime_error("missed duplicate physical patch");
  // Closed-cell touching is overlap, even when centre ordering differs.
  b=a;for(auto& q:b.patches)for(int d=0;d<D;++d){q.hi[d]++;q.grown_hi[d]++;} // update payload to shape
  for(auto& q:b.patches){std::size_t size=q.components;for(int d=0;d<D;++d)size*=std::size_t(q.grown_hi[d]-q.grown_lo[d]+1);q.bits.resize(size);}
  validate_complete_state_carriers(b); // neighbouring valid [0,1],[2,3] stay disjoint
  for(unsigned block=0;block<2;++block){auto& q=b.patches[block*137+1];q.lo=b.patches[block*137].hi;for(int d=0;d<D;++d){q.hi[d]=q.lo[d]+1;q.grown_lo[d]=q.lo[d]-1;q.grown_hi[d]=q.hi[d]+1;}}
  if(!refuses([&]{validate_complete_state_carriers(b);}))throw std::runtime_error("missed closed boundary overlap");
  std::mt19937 gen(1729);for(int trial=0;trial<100;++trial){a=fixture<D>(31);for(auto& q:a.patches){for(int d=0;d<D;++d){q.lo[d]=q.hi[d]=int(gen()%51)-25;q.grown_lo[d]=q.lo[d]-1;q.grown_hi[d]=q.hi[d]+1;}}for(std::size_t i=31;i<a.patches.size();++i){auto block=a.patches[i].block;a.patches[i]=a.patches[i-31];a.patches[i].block=block;}
    bool overlap=false;for(std::size_t i=0;i<31;++i)for(std::size_t j=0;j<i;++j){bool hit=true;for(int d=0;d<D;++d)hit=hit&&a.patches[i].lo[d]<=a.patches[j].hi[d]&&a.patches[j].lo[d]<=a.patches[i].hi[d];overlap|=hit;}
    if(refuses([&]{validate_complete_state_carriers(a);})!=overlap)throw std::runtime_error("BVH differs from exact all-pairs oracle");}
}
template<int D>void bench(std::size_t count){
#ifdef OLD_HEADER
  auto a=fixture<D>(count);auto bytes=encode_state_carriers(a);
  auto old=pops::runtime::checkpoint_old::decode_state_carriers<D>(bytes);
  if(pops::runtime::checkpoint_old::encode_state_carriers(old)!=bytes)throw std::runtime_error("benchmark payloads differ");
  using Clock=std::chrono::steady_clock;
  pops::runtime::checkpoint_old::measured_pair_checks=0;auto start=Clock::now();pops::runtime::checkpoint_old::validate_complete_state_carriers(old);auto old_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  StateCarrierSpatialValidationStats stats;start=Clock::now();validate_complete_state_carriers(a,&stats);auto new_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  std::cout<<D<<","<<count<<","<<a.patches.size()<<",2,1,"<<bytes.size()<<","<<old_ms<<","<<new_ms<<","<<pops::runtime::checkpoint_old::measured_pair_checks<<","<<stats.node_tests<<","<<stats.leaf_pairs<<","<<stats.peak_index_bytes<<"\n";
#else
  (void)count;
#endif
}
int main(){oracles<1>();oracles<2>();oracles<3>();std::cout<<"dim,patches_per_block,rows,components,ghost_depth,archive_bytes,old_ms,new_ms,old_pairs,new_node_tests,new_leaf_pairs,index_bytes\n";for(auto n:{4096u,16384u,65536u}){bench<1>(n);bench<2>(n);bench<3>(n);}}
