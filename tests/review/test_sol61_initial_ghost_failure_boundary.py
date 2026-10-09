"""Actual public C++ consumer failure boundary; not Native AMR/MPI rollback."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_initial_callback_failure_and_preflight_nonmutation(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler, 'real C++20 compiler required'
    source=tmp_path/'initial-failure.cpp'; binary=tmp_path/'initial-failure'
    source.write_text(r'''#include <pops/runtime/dynamic/component_consumers.hpp>
#include "tests/cpp/support/component_abi_test_helpers.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
using namespace pops::component;
struct State { int calls=0; bool throws=false; };
int main(){
 namespace abi=pops::component::test_support;
 double q[2]{1.,2.},phi[2]{2.,2.},ghosts[2]{-0.,3.},coordinates[2]{};
 double before[2];std::memcpy(before,ghosts,sizeof ghosts);int axis=0,side=-1;
 PopsQualifiedConstFieldV1 dependency{sizeof(PopsQualifiedConstFieldV1),1,"case::phi",abi::const_field_view(phi,1,1,2)};
 PopsGhostBoundaryRequestV1 region{sizeof(PopsGhostBoundaryRequestV1),"case::provider","case::Q","case::ghost",
 abi::const_field_view(q,1,1,2),abi::field_view(ghosts,1,1,2),abi::const_field_view(coordinates,1,1,2),
 {sizeof(PopsBoundaryRegionV1),POPS_BOUNDARY_FACE_V1,2,1,1,&axis,&side,"x-low"},1,&dependency,0,nullptr,
 {sizeof(PopsLogicalTimeV1),"primary",0,0,0,0,0,1,0.,7.5},abi::noncollective_host_execution_context()};
 PopsAcceptedInitialGhostApiV1 api{};
 api.apply_initial_region_batch=+[](void* ptr,const PopsAcceptedInitialGhostRequestV1* request,PopsComponentStatusV1*){
  auto& state=*static_cast<State*>(ptr);++state.calls;
  assert(request->region_request.logical_time.physical_time==7.5);
  auto* out=static_cast<double*>(request->region_request.ghosts.data);out[0]=42.;
  if(state.throws)throw std::runtime_error("independent initial callback failure");
  return 71;
 };
 State state;PopsComponentStatusV1 status{};
 PopsAcceptedInitialGhostRequestV1 request{sizeof(PopsAcceptedInitialGhostRequestV1),1,region};
 for(int which=0;which<6;++which){auto bad=request;
  switch(which){case 0:bad.region_request.logical_time.tick=1;break;
   case 1:bad.region_request.logical_time.dt=-0.;break;
   case 2:bad.region_request.logical_time.stage=1;break;
   case 3:bad.region_request.logical_time.clock_identity="";break;
   case 4:bad.point_contract_version=2;break;
   case 5:bad.region_request.dependencies=nullptr;break;}
  bool refused=false;try{apply_accepted_initial_ghost(api,&state,bad,status);}catch(const std::invalid_argument&){refused=true;}
  assert(refused && state.calls==0 && std::memcmp(before,ghosts,sizeof ghosts)==0);
 }
 // Nonzero provider status must be preserved for the outer collective owner.
 assert(apply_accepted_initial_ghost(api,&state,request,status)==71 && state.calls==1);
 std::memcpy(ghosts,before,sizeof ghosts);state.throws=true;bool threw=false;
 try{apply_accepted_initial_ghost(api,&state,request,status);}catch(const std::runtime_error& e){threw=std::strcmp(e.what(),"independent initial callback failure")==0;}
 assert(threw && state.calls==2 && ghosts[0]==42.);
 // The dispatcher deliberately cannot claim rollback: outer AMR transaction owns it.
}
''')
    subprocess.run([compiler,'-std=c++20','-DPOPS_NATIVE_DIM=2','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),'-I',str(ROOT),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)
