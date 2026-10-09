"""SOURCE_ONLY actual IR19 budget phase with explicit protocol/layout substitutes."""
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
HEADER="include/pops/runtime/program/amr_program_context_spatial_interaction.inc"


def body(text, marker):
    start=text.index(marker)
    opened=text.index("{",start)
    depth,end=1,opened+1
    while depth:
        depth+=(text[end]=="{")-(text[end]=="}")
        end+=1
    return text[opened+1:end-1]


@pytest.mark.parametrize("historical",(False,True))
def test_actual_tower_budget_phase_grows_all_levels_and_collectively_wraps_getters(tmp_path,historical):
    text=(subprocess.check_output(["git","show","18117299ba49d13845948ebccce7100c5429da75:"+HEADER],cwd=ROOT,text=True)
          if historical else (ROOT/HEADER).read_text())
    prepare=text[text.index("void prepare_closed_original_interaction("):]
    start=prepare.index("  closed_interaction_phase_([&] {",prepare.index("  std::size_t retained = 0;"))
    phase=body(prepare[start:],"closed_interaction_phase_([&]")
    native=(ROOT/"include/pops/mesh/index/box.hpp").read_text()
    box="\n".join(line for line in native.splitlines() if not line.startswith("#include") and line!="#pragma once")
    helper=(ROOT/"include/pops/runtime/program/spatial_direct_interaction.hpp").read_text()
    arithmetic=helper[helper.index("inline std::size_t interaction_add("):helper.index("template <class Function>")]
    code=r'''
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <iostream>
#define POPS_HD
namespace pops {template<int D>using Index=std::array<int,D>;template<int D>using Extent=std::array<int64_t,D>;}
namespace Kokkos {int fences=0;void fence(){++fences;}}
'''+box+'\n'+arithmetic+r'''
using namespace pops;using Real=double;constexpr int Dim=2;
struct Field{using fab_type=int;std::vector<Box<2>> boxes;int dist=7;Index<2> rank{};Extent<2> halo{};int width=3;
 const auto& layout()const{return boxes;}int distribution()const{return dist;}auto local_rank()const{return rank;}
 auto ghosts()const{return halo;}size_t local_size()const{return boxes.size();}auto box(size_t p)const{return boxes.at(p);}};
using field_type=Field;struct MemorySpace{};
template<int,class>struct InteractionLevelView{const Field* field;};
struct Source{size_t retained_bytes=80;size_t accepted_generation=9;};
struct ClosedInteractionTower{std::shared_ptr<Source> source;int storage_owner=0;std::string identity;std::vector<Field> output;static int objects;ClosedInteractionTower(){++objects;}};
int ClosedInteractionTower::objects=0;
struct Provider{bool stale=false;size_t accepted_original_candidate_generation(const std::vector<Field>*,int){if(stale)throw std::invalid_argument("named stale candidate authority");return 9;}};
struct Facade{std::vector<Field> storage;bool inside=false;bool getter_fault=false;int calls=0;
 const Field& prepared_amr_block_state(int owner,size_t level){++calls;assert(inside&&owner==4);if(getter_fault)throw std::invalid_argument("named getter fault");return storage.at(level);}};
int checks=0;void check(bool b){++checks;if(!b)throw std::runtime_error("assertion"+std::to_string(checks));}
struct Probe{std::shared_ptr<Source> source=std::make_shared<Source>();Provider provider;std::vector<Field> candidate;
 Facade facade;Facade* facade_=&facade;int votes=0;bool succeeded=false;
 int nlev(){return 2;}int sys_block(int program){assert(program==5);return 4;}
 template<class Fn>void closed_interaction_phase_(Fn fn){facade.inside=true;try{fn();}catch(...){facade.inside=false;++votes;throw;}facade.inside=false;++votes;}
 size_t run(uint64_t max_bytes){int lane=0,storage_block=5;std::string identity="issued";std::array<int,1> components{1};
 std::shared_ptr<ClosedInteractionTower> result;std::vector<const Field*>output_prototypes;std::vector<InteractionLevelView<Dim,MemorySpace>>levels;size_t retained=0;int storage_owner=-1;
 struct ProviderAdapter{Provider* p;size_t accepted_original_candidate_generation(const std::vector<Field>*x,int lane){return p->accepted_original_candidate_generation(x,lane);}};
 // Actual budget phase accesses source->provider; this member is added by the generated wrapper.
 closed_interaction_phase_([&]{
'''
    # Source provider is a protocol seam, not an actual Native core authority.
    code=code.replace("struct Source{size_t", "struct Provider;struct Source{Provider* provider=nullptr;size_t")
    code+=phase+r'''
 });
 check(output_prototypes.size()==2);for(size_t level=0;level<2;++level)check(output_prototypes[level]==&facade.storage[level]);
 check(result->output.capacity()>=2);check(levels.capacity()>=2);succeeded=true;return retained;
 }};
int main(){Probe p;Field src;src.boxes={Box<2>({0,0},{1,0})};src.width=3;
 Field storage=src;storage.width=5;storage.halo={1,2};p.candidate={src,src};p.facade.storage={storage,storage};p.source->provider=&p.provider;
 const size_t expected=80+sizeof(ClosedInteractionTower)+6+2*(sizeof(Field)+sizeof(InteractionLevelView<2,MemorySpace>)+sizeof(Field*))
 +2*(4*5*sizeof(Real)+sizeof(size_t)+2*sizeof(Box<2>)+sizeof(Index<2>)+sizeof(Field::fab_type)+sizeof(size_t));
 check(p.run(expected+1)==expected);check(p.succeeded);check(p.facade.calls==GETTER_CALLS);check(p.votes==1);check(Kokkos::fences==1);
 auto bad=[&](auto mutation){Probe q;q.candidate={src,src};q.facade.storage={storage,storage};q.source->provider=&q.provider;mutation(q);int votes=q.votes;bool refused=false;try{q.run(expected+1);}catch(const std::exception&){refused=true;}check(refused);check(q.votes==votes+1);check(!q.succeeded);};
 bad([](auto&q){q.facade.storage[1].dist=8;});bad([](auto&q){q.facade.storage[1].rank[0]=1;});bad([](auto&q){q.facade.storage[1].halo[0]=-1;});
 bad([](auto&q){q.facade.storage[1].halo[0]=INT32_MAX;});bad([](auto&q){q.facade.storage[1].halo={INT32_MAX-2,INT32_MAX-2};});bad([](auto&q){q.facade.getter_fault=true;});
 int old=ClosedInteractionTower::objects;p.provider.stale=true;try{p.run(expected+1);check(false);}catch(const std::invalid_argument&){}check(ClosedInteractionTower::objects==old);p.provider.stale=false;
 // Historical scope: metadata object creation currently precedes the final budget refusal.
 old=ClosedInteractionTower::objects;bool budget=false;try{p.run(expected);}catch(const std::length_error&){budget=true;}check(budget);check(ClosedInteractionTower::objects==old+HISTORICAL_OBJECTS);
 std::cout<<"bridge assertions "<<checks<<"\n";
}
'''
    code="#define GETTER_CALLS "+str(2 if historical else 4)+"\n#define HISTORICAL_OBJECTS "+str(int(historical))+"\n"+code
    path=tmp_path/"bridge.cpp"
    path.write_text(code)
    exe=tmp_path/"bridge"
    subprocess.run(["/usr/bin/clang++","-std=c++20","-fsanitize=undefined","-fno-sanitize-recover=all",str(path),"-o",str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert "bridge assertions" in result.stdout


def test_bridge_output_workspace_receives_every_exact_owner_prototype():
    text=(ROOT/HEADER).read_text()
    prepare=text[text.index("void prepare_closed_original_interaction("):text.index("const ClosedInteractionTower& require_closed_interaction_")]
    assert "max_bytes - reserved, identity, lane, kernel, output_prototypes.at(level)" in prepare
    assert "sizeof(field_type*)" in prepare
    assert "allocated_box = allocated_box.grow(axis, ghosts[axis])" in prepare
    assert prepare.index("if (retained >= max_bytes)")<prepare.index("std::make_shared<ClosedInteractionTower>()")<prepare.index("output_prototypes.reserve(")<prepare.index("result->output.reserve(")<prepare.index("auto output = direct_spatial_interaction<")
    assert prepare.index("(void)source->candidate();")<prepare.index("staged.emplace(result_id")
