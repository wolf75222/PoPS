"""Actual test-component callback and ABI dispatch, explicit host MPI stand-ins only."""
from pathlib import Path
import shutil,subprocess
from tests.python.support.initial_ghost_failure_component import package_data
ROOT=Path(__file__).resolve().parents[2]

def test_actual_callback_missing_arm_is_logged_and_refused_without_write(tmp_path):
    _,source=package_data()
    stub=tmp_path/'mpi.h';stub.write_text('typedef int MPI_Comm;typedef int MPI_Fint; inline MPI_Comm MPI_Comm_f2c(MPI_Fint x){return x;} inline int MPI_Comm_rank(MPI_Comm,int* x){*x=0;return 0;} inline int MPI_Comm_size(MPI_Comm,int* x){*x=1;return 0;}')
    main=r'''
#include <pops/runtime/dynamic/component_consumers.hpp>
#include "tests/cpp/support/component_abi_test_helpers.hpp"
#include <cassert>
int main(int argc,char** argv){assert(argc==2);setenv("POPS_TEST_INITIAL_GHOST_LOG",argv[1],1);unsetenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK");
 namespace abi=pops::component::test_support;double q[2]{1.,2.},phi[2]{2.,2.},out[2]{-0.,3.},coords[2]{};int axis=0,side=-1;
 PopsQualifiedConstFieldV1 dependency{sizeof(PopsQualifiedConstFieldV1),1,"qualified::phi",abi::const_field_view(phi,1,1,2)};
 PopsGhostBoundaryRequestV1 region{sizeof(PopsGhostBoundaryRequestV1),"qualified::producer","qualified::state","qualified::ghost",abi::const_field_view(q,1,1,2),abi::field_view(out,1,1,2),abi::const_field_view(coords,1,1,2),{sizeof(PopsBoundaryRegionV1),POPS_BOUNDARY_FACE_V1,2,1,1,&axis,&side,"xmin"},1,&dependency,0,nullptr,{sizeof(PopsLogicalTimeV1),"clock-auth",0,0,0,0,0,1,0.,7.5},abi::noncollective_host_execution_context()};
 PopsAcceptedInitialGhostRequestV1 request{sizeof(PopsAcceptedInitialGhostRequestV1),1,region};State state{0,1};PopsComponentStatusV1 status{};
 assert(pops::component::apply_accepted_initial_ghost(init,&state,request,status)==55);
 assert(status.code==55);assert(std::strcmp(status.reason,"initial Ghost fault target is not armed before callback")==0);assert(std::signbit(out[0])&&out[1]==3.);
}
'''
    cpp=tmp_path/'probe.cpp';cpp.write_bytes(source+main.encode());binary=tmp_path/'probe'
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    subprocess.run([compiler,'-std=c++20','-DPOPS_NATIVE_DIM=2','-Wall','-Wextra','-Werror','-I',str(tmp_path),'-I',str(ROOT/'include'),'-I',str(ROOT),str(cpp),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary),str(tmp_path/'callback')],check=True,capture_output=True)
    log=(tmp_path/'callback-rank0.log').read_text();assert 'initial-entry' in log and 'time=0x1.ep+2' in log
    assert 'field-before-write' not in log and 'tentative-write' not in log and 'injected-failure' not in log
