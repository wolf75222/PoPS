"""Full actual test DSO source host probe; no Native owner or MPI execution."""
import shutil
import subprocess
from pathlib import Path
from tests.python.support.initial_ghost_failure_component import package_data

def test_callback_transport_refusals_carry_reason_without_writes(tmp_path):
    _,source=package_data()
    probe=r'''
int main(){
 State state{0,2}; double out=7.;
 PopsAcceptedInitialGhostRequestV1 request{};request.point_contract_version=1;request.region_request.ghosts.data=&out;
 const char* targets[]={nullptr,"garbage","-1","2",""};
 for(const char* target:targets){
  if(target)setenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK",target,1);else unsetenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK");
  PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1),0,POPS_COMPONENT_CONTINUE_V1,nullptr};
  int code=initial(&state,&request,&status);
  if(code==0||out!=7.)return 1;
  if(!status.reason||!*status.reason)return 2;
  if(status.struct_size!=sizeof(PopsComponentStatusV1)||status.code!=code)return 3;
 }
 return 0;
}
'''
    cpp=tmp_path/'probe.cpp';cpp.write_bytes(source+b'\n#include <initializer_list>\n'+probe.encode())
    root=Path(__file__).resolve().parents[2]
    env=Path('/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17')
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    binary=tmp_path/'probe'
    subprocess.run([compiler,'-std=c++20','-DPOPS_NATIVE_DIM=2','-I',str(root/'include'),'-I',str(env/'include'),str(cpp),'-L',str(env/'lib'),'-Wl,-rpath,'+str(env/'lib'),'-lmpi','-o',str(binary)],check=True,capture_output=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,(result.returncode,result.stderr)
