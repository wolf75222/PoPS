"""Host execution of authentic dot_all kernels/providers; storage and MPI are substitutes.

No installed native provider or collective execution is qualified by this fixture.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _extract(path, signature):
    source = (ROOT / path).read_text()
    start = source.index(signature)
    opening = source.index("{", start)
    # Struct bodies can contain default initializers before their opening brace;
    # every requested signature ends at the declaration, before any members.
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + (";" if signature.startswith(("template <int Dim>\nstruct", "struct ")) else "")


def test_authentic_dot_all_finite_coverage_and_collective_control_flow(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    mesh = "include/pops/mesh/storage/mf_arith.hpp"
    uniform = "include/pops/runtime/program/program_context.hpp"
    amr = "include/pops/runtime/program/amr_program_context_spatial_operations.inc"
    contracts = "include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc"
    pieces = {
        "accumulator": _extract("include/pops/mesh/execution/for_each.hpp", "struct FiniteCompensatedSum"),
        "kernel": _extract(mesh, "template <int Dim>\nstruct FiniteOwnedDotKernel"),
        "layout": _extract(mesh, "template <int Dim, class LeftSpace, class RightSpace>\nvoid require_same_layout"),
        "measure": _extract(mesh, "template <int Dim, class MemorySpace>\nvoid validate_measure"),
        "local": _extract(mesh, "template <int Dim, class MemorySpace>\nReal dot_owned_active_all_finite_local"),
        "local_sum": _extract(mesh, "template <int Dim, class MemorySpace>\nFiniteCompensatedSum dot_owned_active_all_finite_sum_local"),
        "uniform": _extract(uniform, "Real dot_all(int program_block,"),
        "uniform_contract": _extract(uniform, "static void require_same_field_contract_"),
        "amr": _extract(amr, "Real dot_all(int program_block,"),
        "visitor": _extract(amr, "template <class Visitor>\nvoid for_each_owner_active_level_"),
        "finest": _extract(amr, "template <class Visitor>\nvoid for_each_owner_finest_active_level_"),
        "amr_contract": _extract(contracts, "static void require_same_field_contract_"),
        "amr_layout": _extract(contracts, "static void require_same_layout_"),
    }
    source = SCAFFOLD
    for key, value in pieces.items():
        token = "@" + key.upper() + "@"
        assert source.count(token) == 1
        source = source.replace(token, value)
    cpp, binary = tmp_path / "authentic-dot-all.cpp", tmp_path / "authentic-dot-all"
    cpp.write_text(source)
    build = subprocess.run([compiler, "-std=c++20", "-O0", str(cpp), "-o", str(binary)],
                           text=True, capture_output=True, timeout=30)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "checks=24" in result.stdout


SCAFFOLD = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define POPS_HD
using Real=double;
namespace Kokkos {using std::isfinite;}
template<int Dim>using Index=std::array<int,Dim>;
template<class T,int Dim>struct FieldView {
 const std::vector<std::vector<Real>>* cells=nullptr;
 const Real& operator()(const Index<Dim>& i,int c)const{return cells->at(i[0]).at(c);}
};
struct Distribution {
 bool replica=false;int id=1;
 bool replicated()const{return replica;}
 bool operator==(const Distribution&)const=default;
};
template<int Dim,class Memory>struct MultiFab {
 std::vector<std::vector<Real>> cells;int owner=7,width=2,layout_id=1,rank_id=0,ghosts_id=0;
 Distribution distribution_id;
 int ncomp()const{return width;}int layout()const{return layout_id;}
 Distribution distribution()const{return distribution_id;}int local_rank()const{return rank_id;}
 int ghosts()const{return ghosts_id;}size_t local_size()const{return cells.empty()?0:1;}
 size_t box(size_t)const{return cells.size();}
 struct Fab{const std::vector<std::vector<Real>>* cells;FieldView<const Real,Dim>view()const{return {cells};}};
 Fab fab(size_t)const{return {&cells};}
};
template<class Kernel>Real for_each_cell_reduce_sum(size_t n,Kernel kernel){
 Real x=0;for(size_t i=0;i<n;++i)x+=kernel(Index<1>{static_cast<int>(i)});return x;
}
using Field=MultiFab<1,int>;
namespace pops {
@ACCUMULATOR@
template<class Kernel>FiniteCompensatedSum for_each_cell_reduce_finite_sum(size_t n,Kernel kernel){
 FiniteCompensatedSum x;for(size_t i=0;i<n;++i)x.add(kernel(Index<1>{static_cast<int>(i)}));return x;
}
template<int Dim,class Memory>struct RelativeCellMeasure {
 const MultiFab<Dim,Memory>* active_cells;const MultiFab<Dim,Memory>* inverse_volume_fraction;
};
namespace mf_arith_detail {
@KERNEL@
@LAYOUT@
@MEASURE@
}
@LOCAL_SUM@
@LOCAL@
}
using pops::FiniteCompensatedSum;
struct ExecutionLane {int r=0,n=1;int rank()const{return r;}int size()const{return n;}
 const ExecutionLane& communicator()const{return *this;}};
struct Identity {enum class Family{State,History,Scratch,Direct};Family family=Family::State;int direct_level=0;};
using Family=Identity::Family;
std::string events;bool global_overflow=false;
double all_reduce_sum(double x,const ExecutionLane&){events+='S';return global_overflow?x*2:x;}
// This fixture substitutes transport; real collective summaries are tested in C++.
namespace pops {
Real collective_finite_compensated_sum(const FiniteCompensatedSum& local,const ExecutionLane& lane){
 const Real result=all_reduce_sum(local.value(),lane);
 if(!local.finite()||!std::isfinite(result))throw std::overflow_error("collective overflow");
 return result;
}
}
struct Base {
 using field_type=Field;ExecutionLane lane;
 const ExecutionLane& prepared_execution_lane()const{return lane;}
 void converge_owner_reduction_(std::exception_ptr e,const ExecutionLane&,const char*)const {
   events+='V';if(e)std::rethrow_exception(e);
 }
};
struct Uniform:Base {
 Field mask{{{1}}};Uniform(){mask.width=1;}
 const Field* pointwise_active_mask(int,const Field&)const{events+='P';return &mask;}
 @UNIFORM_CONTRACT@
 @UNIFORM@
};
struct Facade {
 std::vector<Field> levels,active,coverage;int coverage_queries=0;
 const Field& prepared_amr_block_state(int,int l)const{return levels.at(l);}
 const Field* prepared_amr_block_level_active_mask(int,int l)const{return active.empty()?nullptr:&active.at(l);}
 const Field& prepared_amr_block_level_coverage_mask(int,int l){++coverage_queries;return coverage.at(l);}
};
struct AMR:Base {
 Facade* facade_;Identity selected;explicit AMR(Facade&f):facade_(&f){}
 void refresh_resources_()const{}
 int sys_block(int b)const{if(b!=7)throw std::invalid_argument("owner");return b;}
 int nlev()const{return static_cast<int>(facade_->levels.size());}
 Identity classify_owner_field_(int owner,const Field&f,const char*)const {
   if(owner!=f.owner)throw std::invalid_argument("foreign owner");
   Identity result=selected;
   for(int l=0;l<nlev();++l)if(&f==&facade_->levels[l])result.direct_level=l;
   return result;
 }
 const Field& resolve_owner_field_(int,const Field&,const Identity&,int l,const char*)const{return facade_->levels.at(l);}
 @AMR_LAYOUT@
 @AMR_CONTRACT@
 @VISITOR@
 @FINEST@
 @AMR@
};
int checks=0;void ok(bool b){if(!b)throw std::runtime_error("host assertion "+std::to_string(checks));++checks;}
template<class F>void refuses(F fn,const std::string&expected){events.clear();bool r=false;try{fn();}catch(const std::exception&){r=true;}ok(r&&events==expected);}
int main(){
 const double nan=std::numeric_limits<double>::quiet_NaN();
 Field q{{{2,3}}};Uniform u;
 events.clear();ok(u.dot_all(7,q,q)==13&&events=="PVS");
 for(int fault=0;fault<5;++fault){Field bad=q;
   if(fault==0)bad.width=1;if(fault==1)bad.layout_id=2;
   if(fault==2)bad.distribution_id.id=2;if(fault==3)bad.rank_id=1;if(fault==4)bad.ghosts_id=1;
   refuses([&]{u.dot_all(7,q,bad);},"PV");}
 Field poison{{{2,nan}}};refuses([&]{u.dot_all(7,poison,poison);},"PV");
 Field huge{{{1e308,1e308}}},ones{{{1,1}}};
 refuses([&]{u.dot_all(7,huge,ones);},"PV");
 refuses([&]{u.dot_all(7,huge,huge);},"PV");
 Field global{{{1e308,0}}};global_overflow=true;
 refuses([&]{u.dot_all(7,global,ones);},"PVS");global_overflow=false;
 u.mask.cells={{0}};ok(u.dot_all(7,poison,poison)==0);
 u.mask.cells={{nan}};refuses([&]{u.dot_all(7,q,q);},"PV");u.mask.cells={{1}};
 q.distribution_id.replica=true;u.mask.distribution_id.replica=true;u.lane.n=2;u.lane.r=1;
 ok(u.dot_all(7,q,q)==0);
 poison.distribution_id.replica=true;refuses([&]{u.dot_all(7,poison,poison);},"PV");
 u.lane.r=0;ok(u.dot_all(7,q,q)==13);
 q.distribution_id.replica=false;u.mask.distribution_id.replica=false;u.lane={};
 Facade f;f.levels={q,q};f.coverage={Field{{{0}}},Field{{{1}}}};for(auto& m:f.coverage)m.width=1;AMR a(f);
 events.clear();ok(a.dot_all(7,f.levels[0],f.levels[0])==13&&events=="VS");
 ok(f.coverage_queries==2);
 f.active={Field{{{1}}},Field{{{0}}}};for(auto& m:f.active)m.width=1;ok(a.dot_all(7,f.levels[0],f.levels[0])==0);f.active.clear();
 f.levels[0].cells={{nan,nan}};ok(a.dot_all(7,f.levels[0],f.levels[0])==13);
 f.levels[1].cells={{nan,3}};refuses([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");
 f.levels={q,q};f.coverage[0].layout_id=2;
 refuses([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");f.coverage[0].layout_id=1;
 a.selected.family=Family::Direct;a.selected.direct_level=1;
 ok(a.dot_all(7,f.levels[1],f.levels[1])==13);
 a.selected.family=Family::State;for(auto& field:f.levels)field.distribution_id.replica=true;
 for(auto& mask:f.coverage)mask.distribution_id.replica=true;a.lane={1,2};
 ok(a.dot_all(7,f.levels[0],f.levels[0])==0);
 a.lane.r=0;ok(a.dot_all(7,f.levels[0],f.levels[0])==13);
 std::cout<<"checks="<<checks<<"\n";
}
'''
