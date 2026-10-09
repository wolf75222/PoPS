// Independent arbitrary-sized closed-box oracle; standard-library host only.
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <random>
#include <iostream>
using namespace pops::runtime::checkpoint;
template<int D> void check(){std::mt19937 gen(90421);for(int trial=0;trial<250;++trial){
 std::vector<StateCarrierPatch<D>> rows(40);for(auto& r:rows)for(int d=0;d<D;++d){r.lo[d]=int(gen()%201)-100;r.hi[d]=r.lo[d]+gen()%17;}
 bool pair=false;for(std::size_t i=0;i<rows.size();++i)for(std::size_t j=0;j<i;++j){bool hit=true;for(int d=0;d<D;++d)hit=hit&&rows[i].lo[d]<=rows[j].hi[d]&&rows[j].lo[d]<=rows[i].hi[d];pair|=hit;}
 bool bvh=false;try{state_carrier_detail::SpatialIndex<D>(rows).require_disjoint();}catch(const std::invalid_argument&){bvh=true;}
 if(pair!=bvh)throw std::runtime_error("independent variable-sized all-pairs mismatch");
}}
int main(){check<1>();check<2>();check<3>();std::cout<<"HOST_ONLY BVH: 750 variable-sized closed-box geometries agree with independent all-pairs\n";}
