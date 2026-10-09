"""Independent Host MPI communicator/type adversaries; no PoPS Native provider."""
from pathlib import Path
import os,sys,subprocess,shutil
import pytest
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def actual_helper(tmp_path_factory):
 d=tmp_path_factory.mktemp('nonauthor-collective');cpp=d/'probe.cpp';exe=d/'probe'
 cpp.write_text(r"""
#include <pops/parallel/collective_exception.hpp>
#include <iostream>
#include <string>
struct Novel final : std::exception {
 int code; std::string text;
 Novel(int c,std::string s):code(c),text(std::move(s)){}
 const char* what()const noexcept override{return text.c_str();}
};
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);int wr=0,n=0;MPI_Comm_rank(MPI_COMM_WORLD,&wr);MPI_Comm_size(MPI_COMM_WORLD,&n);
 int mode=std::stoi(argv[1]),result=0;MPI_Comm supplied=MPI_COMM_NULL;
 if(mode==0)MPI_Comm_split(MPI_COMM_WORLD,wr,0,&supplied);
 else if(mode==1)MPI_Comm_split(MPI_COMM_WORLD,0,n-1-wr,&supplied);
 else MPI_Comm_dup(MPI_COMM_WORLD,&supplied);
 {
 pops::CommunicatorView view{supplied};std::exception_ptr error;std::string original="novel α\nworld="+std::to_string(wr);
 bool fail=mode!=3 || wr==n-1;
 if(fail)try{if(mode==3)throw 93;throw Novel(701+wr,original);}catch(...){error=std::current_exception();}
 std::string captured;int typed=0;
 try{pops::collectively_rethrow_exception(error,view,"independent cause boundary");result=1;}
 catch(const Novel&e){captured=e.what();typed=e.code;if(view.size()!=1||typed!=701+wr||captured!=original)result=2;}
 catch(int x){typed=x;if(mode!=3||view.size()!=1||x!=93)result=3;captured="non-standard exception";}
 catch(const std::runtime_error&e){captured=e.what();if(view.size()==1)result=4;}
 catch(...){result=5;}
 if(view.size()>1){
  int selected_world=mode==1?n-1:mode==3?n-1:0;
  int selected_rank=mode==3?n-1:0;
  std::string expected="independent cause boundary; rank "+std::to_string(selected_rank)+": "+(mode==3?"non-standard exception":"novel α\nworld="+std::to_string(selected_world));
  if(captured!=expected)result=6;
  if(!pops::all_ranks_agree_exact_ordered_byte_pairs({{"cause",captured}},view))result=7;
 }
 // A successful retry boundary has no residual diagnostic or request state.
 pops::collectively_rethrow_exception(nullptr,view,"retry must return");
 std::cout<<"world="<<wr<<" comm="<<view.rank()<<" mode="<<mode<<" typed="<<typed<<" result="<<result<<'\n';
 }
 MPI_Comm_free(&supplied);int collective=0;MPI_Allreduce(&result,&collective,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);MPI_Finalize();return collective;
}
""")
 cmd=[str(Path(sys.prefix)/'bin/mpicxx'),'-std=c++20','-DPOPS_HAS_MPI','-I'+str(ROOT/'include'),str(cpp),'-o',str(exe)]
 environment=dict(os.environ);environment.setdefault("MPICH_CXX",shutil.which("c++") or "c++")
 p=subprocess.run(cmd,capture_output=True,text=True,env=environment);assert p.returncode==0,p.stdout+p.stderr
 return exe
@pytest.mark.parametrize('ranks',(1,2))
@pytest.mark.parametrize('mode',(0,1,2,3))
def test_exact_communicator_priority_type_nonstandard_retry(actual_helper,ranks,mode):
 p=subprocess.run([str(Path(sys.prefix)/'bin/mpiexec'),'-n',str(ranks),str(actual_helper),str(mode)],capture_output=True,text=True,timeout=20)
 assert p.returncode==0,p.stdout+p.stderr
 assert p.stdout.count('result=0')==ranks
