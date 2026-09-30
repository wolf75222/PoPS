"""Actual dot_all and complete AMR owner-level visitor, host substitutions explicit.

Records the exact candidate's covered-coarse double counting and nonfinite results.
No MPI, native storage or reconstructed provider is claimed.
"""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SHA = "21a56b910c421eee465afcb6f16123dd309360a9"
OUT = ROOT / "outputs/sol61-dot-all-21a56b9-host"
OUT.mkdir(parents=True, exist_ok=True)


def extract(path, signature):
    source = subprocess.check_output(("git", "show", SHA + ":" + path), cwd=ROOT, text=True)
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


uniform_path = "include/pops/runtime/program/program_context.hpp"
amr_path = "include/pops/runtime/program/amr_program_context_spatial_operations.inc"
contract_path = "include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc"
parts = {
    "uniform": extract(uniform_path, "Real dot_all(int program_block,"),
    "amr": extract(amr_path, "Real dot_all(int program_block,"),
    "visitor": extract(amr_path, "template <class Visitor>\nvoid for_each_owner_active_level_"),
    "uniform_contract": extract(uniform_path, "static void require_same_field_contract_"),
    "amr_contract": extract(contract_path, "static void require_same_field_contract_"),
    "amr_layout": extract(contract_path, "static void require_same_layout_"),
}
scaffold = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
using Real=double;using ExecutionLane=int;
struct Field {
 std::vector<std::vector<double>> cells;
 int owner=7, width=2, layout_id=1, distribution_id=1, rank_id=0, ghosts_id=0;
 int ncomp()const{return width;}int layout()const{return layout_id;}
 int distribution()const{return distribution_id;}int local_rank()const{return rank_id;}
 int ghosts()const{return ghosts_id;}size_t local_size()const{return cells.size();}
};
struct Identity {enum class Family{State,History,Scratch,Direct};Family family=Family::State;int direct_level=0;};
using Family=Identity::Family;
std::string events;bool fail_local=false;bool global_overflow=false;int calls=0;
namespace pops {
double dot_active_local(const Field&a,const Field&b,int c,const Field*mask) {
 ++calls;if(fail_local)throw std::runtime_error("local kernel fault");
 double sum=0;for(size_t i=0;i<a.cells.size();++i)
   if(!mask || mask->cells[i][0]>=.5)sum+=a.cells[i][c]*b.cells[i][c];return sum;
}}
double all_reduce_sum(double x,const ExecutionLane&){events+='S';return global_overflow?x*2:x;}
struct Base {
 using field_type=Field;int lane=17;
 const ExecutionLane& prepared_execution_lane()const{return lane;}
 void converge_owner_reduction_(std::exception_ptr e,const ExecutionLane&,const char*)const {
   events+='V';if(e)std::rethrow_exception(e);
 }
};
struct Uniform:Base {
 Field mask;bool fail_mask=false;
 const Field* pointwise_active_mask(int,const Field&)const {
   // Substitute records a collectively completed preflight; real System getter
   // separately authenticates route/layout/mode/generation with convergence.
   events+='P';if(fail_mask)throw std::runtime_error("collective mask refusal");return &mask;
 }
 @UNIFORM_CONTRACT@
 @UNIFORM@
};
struct Facade {
 std::vector<Field> levels,active,coverage;int coverage_queries=0;
 const Field& prepared_amr_block_state(int,int l)const{return levels.at(l);}
 const Field* prepared_amr_block_level_active_mask(int,int l)const {
   // The actual System getter returns nullptr in Cartesian mode. No coverage merge.
   return active.empty()?nullptr:&active.at(l);
 }
 const Field& prepared_amr_block_level_coverage_mask(int,int l){++coverage_queries;return coverage.at(l);}
};
struct AMR:Base {
 Facade* facade_;int active_level_=0;Identity selected;bool fail_refresh=false;
 explicit AMR(Facade&f):facade_(&f){}
 void refresh_resources_()const{if(fail_refresh)throw std::runtime_error("local refresh fault");}
 int sys_block(int b)const{if(b!=7)throw std::invalid_argument("owner");return b;}
 int nlev()const{return static_cast<int>(facade_->levels.size());}
 Identity classify_owner_field_(int owner,const Field&f,const char*)const {
   if(owner!=f.owner)throw std::invalid_argument("foreign owner");return selected;
 }
 const Field& resolve_owner_field_(int,const Field&,const Identity&,int l,const char*)const {
   return facade_->levels.at(l);
 }
 @AMR_LAYOUT@
 @AMR_CONTRACT@
 @VISITOR@
 @AMR@
};
int checks=0;void ok(bool b){if(!b)throw std::runtime_error("probe assertion");++checks;}
template<class F>void refuses(F fn,const std::string&expected){events.clear();bool r=false;try{fn();}catch(const std::exception&){r=true;}ok(r&&events==expected);}
int main(){
 Field q{{{2,3,5,7,11}}};q.width=5;Uniform u;u.mask={{{1}}};
 events.clear();ok(u.dot_all(7,q,q)==208 && events=="PVS");
 Field rotated{{{-3,2,5,7,11}}};rotated.width=5;ok(u.dot_all(7,rotated,rotated)==208);
 Field tail{{{2,30,50,70,110}}};tail.width=5;ok(u.dot_all(7,tail,tail)==20404);
 for(int attr=0;attr<5;++attr){Field bad=q;
   if(attr==0)bad.width=4;if(attr==1)bad.layout_id=2;if(attr==2)bad.distribution_id=2;
   if(attr==3)bad.rank_id=1;if(attr==4)bad.ghosts_id=1;
   refuses([&]{u.dot_all(7,q,bad);},"PV");}
 fail_local=true;refuses([&]{u.dot_all(7,q,q);},"PV");fail_local=false;
 u.fail_mask=true;refuses([&]{u.dot_all(7,q,q);},"P");u.fail_mask=false;
 double nan=std::numeric_limits<double>::quiet_NaN();Field masked{{{2,3},{nan,nan}}};
 u.mask={{{1},{0}}};ok(u.dot_all(7,masked,masked)==13);
 Field finite{{{1e308,1e308}}},ones{{{1,1}}};u.mask={{{1}}};
 events.clear();ok(std::isinf(u.dot_all(7,finite,ones))&&events=="PVS");
 Field onefinite{{{1e308}}},one{{{1}}};onefinite.width=one.width=1;global_overflow=true;
 events.clear();ok(std::isinf(u.dot_all(7,onefinite,one))&&events=="PVS");global_overflow=false;
 Field poisoned{{{2,nan}}};ok(std::isnan(u.dot_all(7,poisoned,poisoned)));
 Facade f;f.levels={{{{2,3}}},{{{2,3}}}};f.coverage={{{{0}}},{{{1}}}};AMR a(f);
 events.clear();double actual=a.dot_all(7,f.levels[0],f.levels[0]);
 ok(actual==26 && f.coverage_queries==0 && events=="VS"); // Correct finest-owned answer is 13.
 f.active={{{{1}}},{{{1}}}};ok(a.dot_all(7,f.levels[0],f.levels[0])==26); // EB mask does not fix coverage.
 f.active={{{{0}}},{{{1}}}};ok(a.dot_all(7,f.levels[0],f.levels[0])==13); // Explicit combined mask oracle.
 a.fail_refresh=true;refuses([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");a.fail_refresh=false;
 Field foreign=f.levels[0];foreign.owner=8;refuses([&]{a.dot_all(7,foreign,foreign);},"V");
 a.selected.family=Family::Scratch;a.selected.direct_level=1;calls=0;
 ok(a.dot_all(7,f.levels[0],f.levels[0])==13 && calls==2);
 Facade empty;AMR e(empty);events.clear();ok(e.dot_all(7,q,q)==0&&events=="VS");
 std::cout<<"checks="<<checks<<" covered_actual="<<actual<<" covered_expected=13 coverage_queries="<<f.coverage_queries<<"\n";
}
'''
for key, value in parts.items():
    token = "@" + key.upper() + "@"
    assert scaffold.count(token) == 1
    scaffold = scaffold.replace(token, value)
source, binary = OUT / "probe.cpp", OUT / "probe"
source.write_text(scaffold)
compiler = shutil.which("clang++") or shutil.which("c++")
assert compiler
build = subprocess.run((compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)), capture_output=True, text=True)
assert build.returncode == 0, build.stderr
run = subprocess.run((str(binary),), capture_output=True, text=True)
assert run.returncode == 0, run.stderr
receipt = {
    "candidate": SHA,
    "header_archive_sha256": sha256(subprocess.check_output(("git", "archive", SHA, "include", "src/runtime/system/system_embedded_boundary.cpp", "src/runtime/amr/amr_system.cpp"), cwd=ROOT)).hexdigest(),
    "extracted_parts_sha256": {key: sha256(value.encode()).hexdigest() for key, value in parts.items()},
    "host_scaffold_sha256": sha256(scaffold.encode()).hexdigest(),
    "compiler": compiler, "result": run.stdout.strip(),
    "confirmed_defect": "AMR covered coarse contribution counted with fine level:26 vs13; no coverage query",
    "nonfinite_observation": "finite local and simulated global sum overflow plus active NaN return nonfinite",
    "scope": "actual provider methods and complete AMR visitor; mocked storage, kernels, owner classifier, prepared lane and collectives",
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
