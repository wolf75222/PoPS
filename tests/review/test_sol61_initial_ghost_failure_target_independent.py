"""Actual test-owned C++ callback target adversaries, HOST_ONLY, not Native/MPI."""
import shutil
import subprocess
from pathlib import Path
from tests.python.support.initial_ghost_failure_component import package_data


def test_actual_callback_refuses_forged_target_before_tentative_write(tmp_path):
    _,source=package_data()
    probe=r'''
int main(){
 State state{0,2};double value=2.,out=7.;
 PopsQualifiedConstFieldV1 field{};field.present=1;field.values.component_count=1;field.values.data=&value;field.values.dimension=2;
 PopsAcceptedInitialGhostRequestV1 request{};request.point_contract_version=1;request.region_request.dependencies=&field;request.region_request.dependency_count=1;request.region_request.ghosts.data=&out;
 PopsComponentStatusV1 status{};
 for(const char* target:{"garbage","-1","2","0trailing","","00","+0"," 0","false","999999999999999999999999999999999999"}){
  setenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK",target,1);out=7.;
  if(initial(&state,&request,&status)==0)return 1;
  if(out!=7.)return 2;
 }
 unsetenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK");out=7.;
 if(initial(&state,&request,&status)==0||out!=7.)return 3;
 setenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK","0",1);out=7.;
 if(initial(&state,&request,&status)!=0||status.code!=73||out!=-1234.)return 4;
 setenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK","1",1);out=7.;
 if(initial(&state,&request,&status)!=0||status.code!=0||out!=-1234.)return 5;
 return 0;
}
'''
    cpp=tmp_path/"actual.cpp";cpp.write_bytes(source+b"\n#include <initializer_list>\n"+probe.encode())
    root=Path(__file__).resolve().parents[2]
    env=Path("/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17")
    compiler=shutil.which("clang++") or shutil.which("c++");assert compiler
    binary=tmp_path/"probe"
    result=subprocess.run([compiler,"-std=c++20","-DPOPS_NATIVE_DIM=2","-I",str(root/"include"),"-I",str(env/"include"),str(cpp),"-L",str(env/"lib"),"-Wl,-rpath,"+str(env/"lib"),"-lmpi","-o",str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,(result.returncode,result.stderr)
