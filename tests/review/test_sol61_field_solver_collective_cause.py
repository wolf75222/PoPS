"""Actual C++ boundary bodies and real MPI Host ranks; no PoPS engine/provider build."""
from pathlib import Path
import os
import sys
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
PREFIX=Path(sys.prefix)

def source(path):
    revision=os.environ.get('POPS_CAUSE_BASELINE')
    if revision:
        return subprocess.check_output(['git','-C',str(ROOT),'show',revision+':'+path],text=True)
    return (ROOT/path).read_text()

def body(text, marker):
    start=text.index('{',text.index(marker)); depth=1;end=start+1
    while depth:
        depth += (text[end]=='{')-(text[end]=='}');end+=1
    return text[start+1:end-1]

@pytest.fixture(scope='module')
def executable(tmp_path_factory):
    directory=tmp_path_factory.mktemp('actual-collective-cause')
    prepared=source('include/pops/runtime/system/prepared_field_solver_component.hpp')
    preflight=body(prepared,'static void collective_preflight_(')
    system=body(source('src/runtime/system/system_fields.cpp'),'SolveOutcome System<Dim>::run_field_publication_outcome_(')
    named=body(source('include/pops/runtime/system/exact_named_field.hpp'),'static void collective_rethrow_(')
    start=prepared.index('    if (all_reduce_max(solve_error ? 1L : 0L)') if '    if (all_reduce_max(solve_error ? 1L : 0L)' in prepared else prepared.index('    collectively_rethrow_exception(solve_error,')
    end=prepared.index('\n\n    std::string exact_report;',start)
    vote=prepared[start:end]
    cpp=r"""
#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/collective_step_rejection.hpp>
#include <functional>
#include <iostream>
using namespace pops;
using namespace pops::runtime::program;
struct SolveOutcome { int value; };
struct Facade {
 ExecutionLane lane=ExecutionLane::world();
 int rollbacks=0, staged=0, beginnings=0;
 const ExecutionLane& prepared_boundary_execution_lane(){return lane;}
 void begin_field_publication_outcome_(){++beginnings;}
 void rollback_field_publication_transaction(){++rollbacks;}
 SolveOutcome stage_field_publication_outcome_(SolveReport){++staged;return {0};}
 SolveOutcome run(const std::function<SolveReport()>& solve) {
"""+system+r"""
 }
};
template<class Function> void preflight(Function&& function,const char* collective_message) {
"""+preflight+r"""
}
void named_failure(const std::exception_ptr& error,const char* message,const ExecutionLane& lane) {
"""+named+r"""
}
void callback_vote(std::exception_ptr solve_error) {
"""+vote+r"""
}
int main(int argc,char**argv) {
 int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
 if(provided<MPI_THREAD_MULTIPLE) return 91;
 int result=0;
 {
 int mode=std::stoi(argv[1]); Facade system; int calls=0,writes=0;
 const int failing=mode==3 ? 0 : system.lane.size()-1;
 int exception_kind=0;
 std::string captured;
 try {
   collective_step_rejection_phase(system.lane.communicator(),{"outer-v1","outer",false,false},"System step failed collectively",[&]{
    collective_step_rejection_phase(system.lane.communicator(),{"cadence-v1","cadence",false,false},"Program cadence phase failed collectively",[&]{
      system.run([&] {
        if(mode==0) preflight([&]{if(system.lane.rank()==failing)throw std::invalid_argument("table size truncated: original diagnostic");},"callback preparation phase");
        if(mode==2) {
          std::exception_ptr error;
          if(system.lane.rank()==failing)try{throw std::logic_error("arbitrary dependency: original diagnostic");}catch(...){error=std::current_exception();}
          named_failure(error,"named-field preparation",system.lane);
        }
        ++calls;++writes;
        std::exception_ptr error;
        if((mode==1 && system.lane.rank()==failing) || mode==3)try{throw std::range_error("callback after work: original diagnostic");}catch(...){error=std::current_exception();}
        callback_vote(error);
        return SolveReport{};
      });
    });
   });
 }catch(const std::invalid_argument& error){captured=error.what();exception_kind=1;}
 catch(const std::range_error& error){captured=error.what();exception_kind=2;}
 catch(const std::logic_error& error){captured=error.what();exception_kind=3;}
 catch(const std::exception& error){captured=error.what();}
 if(captured.find("original diagnostic")==std::string::npos || system.rollbacks!=1 || system.staged!=0 || (mode==0 && calls!=0) || (mode==1 && writes!=1))result=1;
 if(system.lane.size()==1 && exception_kind!=(mode==0 ? 1 : mode==2 ? 3 : 2))result=5;
 if(system.lane.size()>1 && captured.find("rank "+std::to_string(failing))==std::string::npos)result=2;
 if(!all_ranks_agree_exact_ordered_byte_pairs({{"cause",captured}},system.lane))result=3;
 if(system.run([]{return SolveReport{};}).value!=0 || system.staged!=1)result=4;
 std::cout<<"rank="<<system.lane.rank()<<" mode="<<mode<<" cause="<<captured<<" result="<<result<<'\n';
 result=static_cast<int>(all_reduce_max(static_cast<long>(result),system.lane));
 }
 MPI_Finalize(); return result;
}
"""
    path=directory/'cause.cpp';path.write_text(cpp);exe=directory/'cause'
    environment=dict(os.environ)
    environment.setdefault('MPICH_CXX',shutil.which('c++') or 'c++')
    compiled=subprocess.run([str(PREFIX/'bin/mpicxx'),'-std=c++20','-DPOPS_HAS_MPI','-I'+str(ROOT/'include'),str(path),'-o',str(exe)],capture_output=True,text=True,env=environment)
    assert compiled.returncode==0,compiled.stdout+compiled.stderr
    return exe

@pytest.mark.parametrize('mode',(0,1,2,3))
@pytest.mark.parametrize('ranks',(1,2))
def test_actual_mpi_host_preserves_remote_cause_and_rollback(executable,mode,ranks):
    result=subprocess.run([str(PREFIX/'bin/mpiexec'),'-n',str(ranks),str(executable),str(mode)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('original diagnostic')>=ranks
