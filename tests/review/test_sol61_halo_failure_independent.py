"""Independent actual header host adversaries, no Native/MPI execution claim."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

def test_actual_consensus_argument_no_allocation_and_request_matrix(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('c++')
    if compiler is None:pytest.skip('host C++20 compiler unavailable')
    source=tmp_path/'actual.cpp'; binary=tmp_path/'actual'
    source.write_text(r'''
#include <pops/parallel/comm.hpp>
#include <pops/runtime/accepted_halo_test_failure.hpp>
#include <cassert>
#include <cstdlib>
#include <new>
#include <type_traits>
bool deny=false;
void* operator new(std::size_t n){if(deny)throw std::bad_alloc();if(auto p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete[](void* p) noexcept {::operator delete(p);}
void inspect(std::initializer_list<pops::ExactOrderedBytePair> pairs){
 assert(pairs.size()==1);assert(pairs.begin()->second=="actual contract");
}
int main(){
 static_assert(std::is_same_v<pops::ExactOrderedBytePair,std::pair<std::string_view,std::string_view>>);
 const std::string contract="actual contract";
 deny=true;inspect({{"accepted-halo-test-failure",contract}});deny=false;
 assert(pops::all_ranks_agree_exact_ordered_byte_pairs({{"request",contract}}));
 deny=true;bool voted=false;
 try{(void)pops::all_ranks_agree_exact_ordered_byte_pairs({{"request",contract}});}
 catch(const std::bad_alloc&){voted=true;}deny=false;assert(voted);
 for(int block:{0,1})for(int level:{0,1,2,3,4})for(int rank:{0,1,2}){
  pops::AcceptedHaloTestFailureRequest r{1,pops::AcceptedHaloTestFailurePhase::after_block_level_preparation,block,level,rank};
  pops::validate_accepted_halo_test_failure_request(r,2,5,3);
  for(int axis=0;axis<3;++axis)for(int invalid:{-1,99}){
   auto bad=r;if(axis==0)bad.block=invalid;if(axis==1)bad.level=invalid;if(axis==2)bad.rank=invalid;
   bool refused=false;try{pops::validate_accepted_halo_test_failure_request(bad,2,5,3);}
   catch(const std::invalid_argument&){refused=true;}assert(refused);assert(r==r);
  }
 }
}
''')
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)
