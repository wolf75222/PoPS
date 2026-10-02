"""Independent real-header host probes; no Native system or physics receipt."""
from pathlib import Path
import subprocess

def test_initial_point_and_provider_state_authority(tmp_path):
    root=Path("/Users/romaindespoulain/dev/tmp/PoPS-sol61-initial-field-ghost")
    source=tmp_path/"probe.cpp";binary=tmp_path/"probe"
    source.write_text(r'''#include <pops/runtime/dynamic/component_consumers.hpp>
#include <cassert>
#include <limits>
using namespace pops::component;
int main(){
 PopsLogicalTimeV1 p{sizeof(PopsLogicalTimeV1),"nonzero-origin-clock",0,0,0,0,0,1,0.,-7.5};
 validate_accepted_initial_ghost_point(p,1);
 for(double dt:{-1.,1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
  auto q=p;q.dt=dt;bool refused=false;try{validate_accepted_initial_ghost_point(q,1);}catch(const std::invalid_argument&){refused=true;}assert(refused);
 }
 PopsGhostBoundaryApiV1 ordinary{};PopsAcceptedInitialGhostApiV1 initial{};
 initial.apply_initial_region_batch=+[](void*,const PopsAcceptedInitialGhostRequestV1*,PopsComponentStatusV1*)->int32_t{return 0;};
 require_initial_ghost_provider_lifecycle(initial,ordinary);
 auto mismatch=initial;mismatch.header.prepare=+[](const PopsComponentPrepareRequestV1*,void**,PopsComponentStatusV1*)->int32_t{return 0;};
 bool refused=false;try{require_initial_ghost_provider_lifecycle(mismatch,ordinary);}catch(const std::invalid_argument&){refused=true;}assert(refused);
 initial.apply_initial_region_batch=nullptr;refused=false;try{require_initial_ghost_provider_lifecycle(initial,ordinary);}catch(const std::exception&){refused=true;}assert(refused);
}
''')
    subprocess.run(["clang++","-std=c++20","-DPOPS_NATIVE_DIM=2","-I",str(root/"include"),str(source),"-o",str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)
