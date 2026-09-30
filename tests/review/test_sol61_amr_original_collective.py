"""Exact Newton dot/local-phase methods; MPI/kernel substitutions are explicit traces."""

import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def method(text, signature):
    start = text.index(signature)
    opening = text.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@pytest.fixture(scope="module")
def reductions(tmp_path_factory):
    folder = tmp_path_factory.mktemp("original-amr-vote-host")
    text = (
        ROOT / "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp"
    ).read_text()
    dot = method(text, "  Real dot_(const hierarchy_type& left,")
    phase = method(text, "  template <class Operation>\n  void local_phase_(")
    authenticate = ""
    if "  void authenticate_owned_(" in text:
        authenticate = method(text, "  void authenticate_owned_(")
        authenticate += method(text, "  static bool same_layout_(")
    cpp = (
        r"""
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Real=double;using ExecutionLane=int;
std::string events;bool fail_kernel=false;
namespace Kokkos{void fence(){}}
void collectively_rethrow_exception(std::exception_ptr e,const ExecutionLane&,const char*){
 events+='V';if(e)std::rethrow_exception(e);
}
double all_reduce_sum(double x,const ExecutionLane&){events+='S';return x;}
struct Distribution{bool replica=false;bool replicated()const{return replica;}
 friend bool operator==(const Distribution&,const Distribution&)=default;};
struct RankSpace{int coordinate(int)const{return 0;}};
struct Field{
 double value=0;int rank=0;Distribution dist;int width=1,shape=1;
 int ncomp()const{return width;}int local_rank()const{return rank;}int layout()const{return shape;}
 Distribution distribution()const{return dist;}RankSpace rank_space()const{return {};}
};
double dot_active_local(const Field&a,const Field&b,int,const Field*mask){
 if(fail_kernel)throw std::runtime_error("injected kernel failure");
 return mask->value>=.5?a.value*b.value:0;
}
struct Probe{
 using field_type=Field;using hierarchy_type=std::vector<Field>;
 hierarchy_type iterate_=hierarchy_type(2);
 std::vector<const Field*>active_cells_;std::vector<double>cell_measures_{.25,.125};
 ExecutionLane lane=0;const ExecutionLane*local_lane_=&lane;
 DOT
 PHASE
 AUTHENTICATE
};
int main(){
 Field covered{0},owned{1};Probe p;p.active_cells_={&covered,&owned};
 std::cout<<"[";bool comma=false;
 for(int mode=0;mode<6;++mode){
  std::vector<Field>a{{1},{2}},b{{3},{4}};
  if(mode==1)a.pop_back();if(mode==2)b.pop_back();fail_kernel=mode==3;
  if(mode==4)a[0].width=2;if(mode==5)a[0].shape=42;
  events.clear();bool refused=false;double result=0;
  try{result=p.dot_(a,b,p.lane);}catch(const std::exception&){refused=true;}
  if(comma)std::cout<<",";comma=true;
  std::cout<<"["<<mode<<","<<(refused?"true":"false")<<",\""<<events<<"\","<<result<<"]";
 }std::cout<<"]\n";
}
""".replace("DOT", dot)
        .replace("PHASE", phase)
        .replace("AUTHENTICATE", authenticate)
    )
    source, binary = folder / "vote.cpp", folder / "vote"
    source.write_text(cpp)
    subprocess.run(
        ["/usr/bin/clang++", "-std=c++20", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(subprocess.check_output([str(binary)], text=True))


def test_actual_dot_weights_only_active_physical_measure(reductions):
    mode, refused, events, value = reductions[0]
    assert (mode, refused, value) == (0, False, 1)
    assert events.endswith("S") and events[:-1] and set(events[:-1]) == {"V"}


def test_actual_kernel_refusal_votes_before_sum(reductions):
    assert reductions[3][1] is True
    assert reductions[3][2] and set(reductions[3][2]) == {"V"}


@pytest.mark.parametrize("mode", (1, 2, 4, 5))
def test_actual_hierarchy_size_refusal_votes_before_peer_collective(reductions, mode):
    assert reductions[mode][1] is True
    assert reductions[mode][2] and set(reductions[mode][2]) == {"V"}
