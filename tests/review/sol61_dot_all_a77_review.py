"""Independent real-kernel/local-helper/provider reception on exact a77e1b1.

Native FieldView, Index, Box and owner-identity helpers are real headers. Field
allocation, local serial reduction and prepared hierarchy/collectives are substitutes.
"""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SHA = "a77e1b1ce09b84171fd3f63ffc64f6995614b8e5"
OLD = "21a56b910c421eee465afcb6f16123dd309360a9"
OUT = ROOT / "outputs/sol61-dot-all-a77-independent"
OUT.mkdir(parents=True, exist_ok=True)


def extract(path, signature, revision=SHA):
    source = subprocess.check_output(("git", "show", revision + ":" + path), cwd=ROOT, text=True)
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + (";" if "\nstruct " in signature else "")


uniform = "include/pops/runtime/program/program_context.hpp"
amr = "include/pops/runtime/program/amr_program_context_spatial_operations.inc"
mesh = "include/pops/mesh/storage/mf_arith.hpp"
contracts = "include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc"
legacy = {}
for path in (uniform, amr):
    for name in ("dot", "norm2", "norm_inf"):
        signature = f"Real {name}(int program_block,"
        old, new = extract(path, signature, OLD), extract(path, signature)
        assert old == new
        legacy[path + "::" + name] = sha256(new.encode()).hexdigest()
signature = "template <class Visitor>\nvoid for_each_owner_active_level_"
assert extract(amr, signature, OLD) == extract(amr, signature)
legacy["legacy_visitor"] = sha256(extract(amr, signature).encode()).hexdigest()
pieces = {
    "kernel": extract(mesh, "template <int Dim>\nstruct FiniteOwnedDotKernel"),
    "local": extract(mesh, "template <int Dim, class MemorySpace>\nReal dot_owned_active_all_finite_local"),
    "layout": extract(mesh, "template <int Dim, class LeftSpace, class RightSpace>\nvoid require_same_layout"),
    "measure": extract(mesh, "template <int Dim, class MemorySpace>\nvoid validate_measure"),
    "uniform": extract(uniform, "Real dot_all(int program_block,"),
    "uniform_contract": extract(uniform, "static void require_same_field_contract_"),
    "amr": extract(amr, "Real dot_all(int program_block,"),
    "amr_layout": extract(contracts, "static void require_same_layout_"),
    "amr_contract": extract(contracts, "static void require_same_field_contract_"),
    "visitor": extract(amr, signature),
    "finest": extract(amr, "template <class Visitor>\nvoid for_each_owner_finest_active_level_"),
}
for path in (uniform, amr, mesh):
    assert (ROOT / path).read_bytes() == subprocess.check_output(("git", "show", SHA + ":" + path), cwd=ROOT)
scaffold = r'''
#include <pops/mesh/storage/field_view.hpp>
#include <pops/runtime/program/program_owner_field_identity.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <map>
#include <vector>
#include <optional>
#include <exception>
#include <limits>
#include <cmath>
#include <iostream>
using pops::Real;using pops::Index;using pops::Box;using pops::FieldView;
struct Distribution {bool replica=false;int id=1;bool replicated()const{return replica;}bool operator==(const Distribution&)const=default;};
struct Patch {
 Box<1> bounds;int width;std::vector<Real> values;
 Patch(int lo,std::vector<std::vector<Real>> rows,int n):bounds(Index<1>{lo},Index<1>{lo+static_cast<int>(rows.size())-1}),width(n),values(rows.size()*n){
  for(size_t i=0;i<rows.size();++i)for(int c=0;c<n;++c)values[c*rows.size()+i]=rows[i][c];
 }
 FieldView<const Real,1> view()const{
  FieldView<const Real,1>v{};v.data=values.data();v.origin=bounds.lo;v.extents=bounds.extent();v.strides[0]=1;v.ncomp=width;v.component_stride=bounds.numPts();return v;
 }
 void set(int row,int c,Real value){values.at(c*bounds.numPts()+row)=value;}
};
namespace pops {
template<int Dim,class Memory>struct MultiFab {
 std::vector<Patch>patches;int width=2,layout_id=1,rank_id=0,ghosts_id=0;Distribution dist;
 MultiFab()=default;
 MultiFab(std::vector<std::vector<Real>>rows):width(rows.empty()?2:static_cast<int>(rows[0].size())){if(!rows.empty())patches.emplace_back(0,rows,width);}
 int ncomp()const{return width;}int layout()const{return layout_id;}Distribution distribution()const{return dist;}
 int local_rank()const{return rank_id;}int ghosts()const{return ghosts_id;}size_t local_size()const{return patches.size();}
 const Box<1>&box(size_t i)const{return patches.at(i).bounds;}const Patch&fab(size_t i)const{return patches.at(i);}
};
template<int D,class K>Real for_each_cell_reduce_sum(const Box<D>&b,K k){Real v=0;for(int i=b.lo[0];i<=b.hi[0];++i)v+=k(Index<D>{i});return v;}
template<int D,class M>struct RelativeCellMeasure{const MultiFab<D,M>*active_cells;const MultiFab<D,M>*inverse_volume_fraction;};
namespace mf_arith_detail{
@KERNEL@
@LAYOUT@
@MEASURE@
}
@LOCAL@
}
using Field=pops::MultiFab<1,int>;
using Identity=pops::runtime::program::ProgramOwnerFieldIdentity;using Family=Identity::Family;
struct ExecutionLane {int r=0,n=1;int rank()const{return r;}int size()const{return n;}};
std::string events;bool global_overflow=false;double remote_sum=0;
double all_reduce_sum(double x,const ExecutionLane&){events+='S';return global_overflow?x*2:x+remote_sum;}
struct Base {
 using field_type=Field;ExecutionLane lane;
 const ExecutionLane&prepared_execution_lane()const{return lane;}
 void converge_owner_reduction_(std::exception_ptr e,const ExecutionLane&,const char*)const{events+='V';if(e)std::rethrow_exception(e);}
};
struct Uniform:Base {
 Field mask;bool use_mask=false;
 const Field*pointwise_active_mask(int,const Field&)const{events+='P';return use_mask?&mask:nullptr;}
 @UNIFORM_CONTRACT@
 @UNIFORM@
};
struct Facade {
 std::vector<Field>levels,active,coverage;int coverage_queries=0;
 const Field&prepared_amr_block_state(int,int l)const{return levels.at(l);}
 const Field*prepared_amr_block_level_active_mask(int,int l)const{return active.empty()?nullptr:&active.at(l);}
 const Field&prepared_amr_block_level_coverage_mask(int,int l){++coverage_queries;return coverage.at(l);}
};
struct History {std::map<std::string,std::vector<Field>>histories;std::map<std::string,int>owner;};
struct AMR:Base {
 Facade*facade_;int active_level_=0;std::map<std::tuple<int,int,int,int,int>,Field>scratch;History history;
 explicit AMR(Facade&f):facade_(&f){}
 void refresh_resources_()const{}
 int sys_block(int b)const{if(b!=7)throw std::invalid_argument("owner");return b;}
 int nlev()const{return static_cast<int>(facade_->levels.size());}
 static auto decode(const std::string&k){return std::optional<std::pair<int,std::string>>{{std::stoi(k.substr(1)),"h"}};}
 Identity classify_owner_field_(int owner,const Field&seed,const char*role)const {
  return pops::runtime::program::classify_program_owner_field(owner,seed,nlev(),
    [&](int l)->const Field&{return facade_->levels.at(l);},[](int)->const Field*{return nullptr;},scratch,history,decode,role);
 }
 const Field&resolve_owner_field_(int owner,const Field&seed,const Identity&id,int level,const char*role)const {
  return pops::runtime::program::resolve_program_owner_field(owner,seed,id,level,
   [&](int l)->const Field&{return facade_->levels.at(l);},[](int)->const Field*{return nullptr;},scratch,history,
   [](const std::string&,int l){return "h"+std::to_string(l);},role);
 }
 @AMR_LAYOUT@
 @AMR_CONTRACT@
 @VISITOR@
 @FINEST@
 @AMR@
};
int checks=0;void ok(bool b){if(!b)throw std::runtime_error("check "+std::to_string(checks));++checks;}
template<class F>void refuse(F f,std::string sequence){events.clear();bool r=false;try{f();}catch(const std::exception&){r=true;}ok(r&&events==sequence);}
Field field(std::initializer_list<std::initializer_list<double>>rows){std::vector<std::vector<double>>v;for(auto row:rows)v.emplace_back(row);return Field(v);}
int main(){
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 Uniform u;Field q=field({{2,3,5,7,11}}),rotation=field({{-3,2,5,7,11}}),corrupt=field({{2,30,50,70,110}});
 events.clear();ok(u.dot_all(7,q,q)==208&&events=="PVS");ok(u.dot_all(7,rotation,rotation)==208);ok(u.dot_all(7,corrupt,corrupt)==20404);
 for(int mode=0;mode<5;++mode){auto bad=q;if(mode==0)bad.width=4;if(mode==1)bad.layout_id=2;if(mode==2)bad.dist.id=2;if(mode==3)bad.rank_id=1;if(mode==4)bad.ghosts_id=1;refuse([&]{u.dot_all(7,q,bad);},"PV");}
 for(double poison:{nan,inf,-inf}){Field bad=field({{0,poison}});refuse([&]{u.dot_all(7,bad,bad);},"PV");}
 Field huge=field({{1e308,1e308}}),ones=field({{1,1}});refuse([&]{u.dot_all(7,huge,huge);},"PV");refuse([&]{u.dot_all(7,huge,ones);},"PV");
 Field patchsum=field({{1e308,0}});patchsum.patches.emplace_back(3,std::vector<std::vector<Real>>{{1e308,0}},2);Field patchones=patchsum;for(auto&p:patchones.patches)p.values={1,1};refuse([&]{u.dot_all(7,patchsum,patchones);},"PV");
 Field global=field({{1e308,0}});global_overflow=true;refuse([&]{u.dot_all(7,global,ones);},"PVS");global_overflow=false;
 Field bad=field({{nan,nan}});u.use_mask=true;u.mask=field({{0}});ok(u.dot_all(7,bad,bad)==0);
 u.mask=field({{nan}});refuse([&]{u.dot_all(7,bad,bad);},"PV");u.mask=field({{inf}});refuse([&]{u.dot_all(7,ones,ones);},"PV");
 u.mask=field({{1,1}});refuse([&]{u.dot_all(7,ones,ones);},"PV");u.mask=field({{1}});u.mask.layout_id=2;refuse([&]{u.dot_all(7,ones,ones);},"PV");u.use_mask=false;
 q.dist.replica=true;u.lane={1,2};ok(u.dot_all(7,q,q)==0);bad.dist.replica=true;refuse([&]{u.dot_all(7,bad,bad);},"PV");
 u.lane.r=0;ok(u.dot_all(7,q,q)==208);q.dist.replica=false;u.lane={};
 Field empty;empty.width=5;empty.rank_id=1;u.lane={1,2};events.clear();remote_sum=208;ok(u.dot_all(7,empty,empty)==208&&events=="PVS");remote_sum=0;u.lane={};
 Facade f;f.levels={field({{2,3}}),field({{2,3}})};f.levels[1].layout_id=2;
 f.coverage={field({{0}}),field({{1}})};f.coverage[1].layout_id=2;AMR a(f);
 events.clear();ok(a.dot_all(7,f.levels[0],f.levels[0])==13&&events=="VS");ok(f.coverage_queries==2);
 f.active={field({{1}}),field({{0}})};f.active[1].layout_id=2;ok(a.dot_all(7,f.levels[0],f.levels[0])==0);f.active.clear();
 f.levels[0].patches[0].values={nan,nan};ok(a.dot_all(7,f.levels[0],f.levels[0])==13);
 f.levels[1].patches[0].set(0,1,nan);refuse([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");f.levels[1].patches[0].set(0,1,3);
 f.coverage[0].patches[0].set(0,0,nan);refuse([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");f.coverage[0].patches[0].set(0,0,0);
 for(int mode=0;mode<4;++mode){auto save=f.coverage[1];if(mode==0)f.coverage[1].width=2;if(mode==1)f.coverage[1].layout_id=3;if(mode==2)f.coverage[1].dist.id=2;if(mode==3)f.coverage[1].rank_id=1;refuse([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");f.coverage[1]=save;}
 // Actual registry identity resolves Scratch and History at level1, not callback ordinal0.
 a.scratch[{0,1,7,42,0}]=f.levels[1];auto&scratch=a.scratch.begin()->second;ok(a.dot_all(7,scratch,scratch)==13);
 a.scratch[{0,1,8,99,0}]=f.levels[1];auto&foreign=a.scratch.at({0,1,8,99,0});refuse([&]{a.dot_all(7,foreign,foreign);},"V");
 for(int l=0;l<2;++l){a.history.histories["h"+std::to_string(l)]={f.levels[l]};a.history.owner["h"+std::to_string(l)]=7;}
 auto&h=a.history.histories["h1"][0];ok(a.dot_all(7,h,h)==13);a.history.owner["h1"]=8;refuse([&]{a.dot_all(7,h,h);},"V");a.history.owner["h1"]=7;
 Field direct=f.levels[1];ok(a.dot_all(7,direct,direct)==13);direct.ghosts_id=99;ok(a.dot_all(7,f.levels[1],direct)==13);direct.layout_id=9;refuse([&]{a.dot_all(7,direct,direct);},"V");
 f.coverage[0].patches[0].set(0,0,1);for(auto&v:f.levels)v.patches[0].values={1e154,0};refuse([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");
 for(auto&v:f.levels)v.patches[0].values={2,3};f.coverage[0].patches[0].set(0,0,0);global_overflow=true;f.levels[1].patches[0].values={1e154,0};refuse([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"VS");global_overflow=false;
 f.levels[1].patches[0].values={2,3};for(auto&v:f.levels)v.dist.replica=true;for(auto&m:f.coverage)m.dist.replica=true;a.lane={1,2};ok(a.dot_all(7,f.levels[0],f.levels[0])==0);
 f.levels[1].patches[0].set(0,1,nan);refuse([&]{a.dot_all(7,f.levels[0],f.levels[0]);},"V");f.levels[1].patches[0].set(0,1,3);a.lane.r=0;ok(a.dot_all(7,f.levels[0],f.levels[0])==13);
 Facade blank;AMR e(blank);events.clear();remote_sum=13;ok(e.dot_all(7,empty,empty)==13&&events=="VS");remote_sum=0;
 std::cout<<checks<<" independent actual-kernel/provider/identity checks PASS\n";
}
'''
for key, value in pieces.items():
    token = "@" + key.upper() + "@"
    assert scaffold.count(token) == 1
    scaffold = scaffold.replace(token, value)
cpp, binary = OUT / "probe.cpp", OUT / "probe"
cpp.write_text(scaffold)
compiler = shutil.which("clang++") or shutil.which("c++")
assert compiler
command = (compiler, "-std=c++20", "-O0", "-I" + str(ROOT / "include"),
           "-I/Users/romaindespoulain/miniforge3/envs/pops-api040/include", str(cpp), "-o", str(binary))
built = subprocess.run(command, capture_output=True, text=True)
assert built.returncode == 0, built.stderr
run = subprocess.run((str(binary),), capture_output=True, text=True)
assert run.returncode == 0, run.stderr
receipt = {
    "candidate": SHA, "legacy_parent": OLD, "legacy_function_sha256": legacy,
    "extracted_function_sha256": {key: sha256(value.encode()).hexdigest() for key, value in pieces.items()},
    "archive_sha256": sha256(subprocess.check_output(("git", "archive", SHA, "include", "python"), cwd=ROOT)).hexdigest(),
    "compiler_command": command, "host_scaffold_sha256": sha256(scaffold.encode()).hexdigest(), "result": run.stdout.strip(),
    "scope": "actual kernel/helper/providers/visitors, FieldView/Box/Index and ownership identity header; host storage and serial kernel loop, prepared hierarchy and collectives are substitutes",
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
