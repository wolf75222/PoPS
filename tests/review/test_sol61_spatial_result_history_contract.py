"""Actual helper/guard bodies with explicit host storage/lane substitutes."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "dbc9e1cd"
HEADER = "include/pops/runtime/program/spatial_direct_interaction.hpp"


def body(source):
    return "\n".join(line for line in source.splitlines()
                     if not line.startswith("#include") and line != "#pragma once")


def guard(source):
    start = source.index("  static void require_same_field_contract_(")
    opened = source.index("{", start)
    depth = 1
    index = opened+1
    while depth:
        depth += (source[index] == "{")-(source[index] == "}")
        index += 1
    return source[start:index]


@pytest.mark.parametrize("real", ("float", "double"))
@pytest.mark.parametrize("historical", (False, True))
def test_actual_result_and_history_guard_closed_contract(tmp_path, real, historical):
    header = (ROOT/HEADER).read_text()
    original = subprocess.run(["git", "show", f"{BASE}:{HEADER}"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout
    context_path = "include/pops/runtime/program/program_context.hpp"
    context = (ROOT/context_path).read_text()
    original_context = subprocess.run(["git", "show", f"{BASE}:{context_path}"], cwd=ROOT,
                                      check=True, capture_output=True, text=True).stdout
    assert guard(context) == guard(original_context)  # Guard is never relaxed.
    reduction = (ROOT/"include/pops/mesh/execution/for_each.hpp").read_text()
    start = reduction.index("struct FiniteCompensatedSum {")
    stop = reduction.index("\n};", start)+3
    cpp = (ROOT/"tests/review/sol61_spatial_interaction_host.cpp").read_text()
    cpp += "\nnamespace pops {"+reduction[start:stop]+"}\n"
    cpp += body(original if historical else header)
    cpp += "\nstruct ExactHistoryGuard { using field_type = pops::MultiFab<2,Kokkos::HostSpace>;\n"
    cpp += guard(context)+"\n};\n"
    cpp += r'''
using namespace pops; using namespace pops::runtime::program;
using F=MultiFab<2,Kokkos::HostSpace>;
int checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("host contract assertion "+std::to_string(checks));}
template<class Work> void refuse(Work work){bool refused=false;try{work();}catch(const std::exception&){refused=true;}require(refused);}
int main(){
  Box<2> box({0,0},{1,0});Geometry<2> geometry{box,{0,0},{2,3}};
  F source({box},{},{},3,{}),owner({box},{},{},3,{1,2});
  auto value=source.fab(0).view();value({0,0},0)=2;value({1,0},0)=4;
  value({0,0},2)=3;value({1,0},2)=5;
  const std::array<InteractionLevelView<2,Kokkos::HostSpace>,1> levels{{{&source,nullptr,nullptr,nullptr,geometry}}};
  const std::array<int,2> components{2,0};ExecutionLane lane;
  auto kernel=[](auto,auto){return Real(1);};
  auto calculate=[&](uint64_t budget,const F* prototype){
#ifdef HISTORICAL
    (void)prototype;return direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,components,budget,"real-fields",lane,kernel);
#else
    return direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,components,budget,"real-fields",lane,kernel,prototype);
#endif
  };
  F keeper({box},{},{},2,owner.ghosts());
  auto result=calculate(100000,&owner);
#ifdef HISTORICAL
  require(result.ghosts()!=keeper.ghosts());
  refuse([&]{ExactHistoryGuard::require_same_field_contract_(keeper,result,"ProgramContext history store");});
  std::cout<<"historical exact ranked refusal received\n";
#else
  require(result.ghosts()==keeper.ghosts());
  ExactHistoryGuard::require_same_field_contract_(keeper,result,"ProgramContext history store");++checks;
  auto out=result.fab(0).view();require(out({0,0},0)==24);require(out({1,0},1)==18);
  // Grown cells carry initialized storage, never additional quadrature targets.
  require(out({-1,-2},0)==0);require(out({2,2},1)==0);
  require(result.fab(0).values.data->size()==40);
  F foreign=owner;foreign.rank[0]=1;refuse([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.boxes[0].hi[0]=2;refuse([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.dist.replica=true;refuse([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.halo[0]=-1;refuse([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.halo[0]=INT_MAX;refuse([&]{calculate(100000,&foreign);});
  auto valid_only=calculate(100000,nullptr);
  require(valid_only.ghosts()==Extent<2>{});
  require(valid_only.fab(0).view()({0,0},0)==out({0,0},0));
  refuse([&]{ExactHistoryGuard::require_same_field_contract_(keeper,valid_only,"ProgramContext history store");});
  auto minimum=[&](const F* prototype){uint64_t low=1,high=100000;while(low<high){auto mid=(low+high)/2;try{calculate(mid,prototype);high=mid;}catch(const std::length_error&){low=mid+1;}}return low;};
  const auto old_bytes=minimum(nullptr),grown_bytes=minimum(&owner);
  require(grown_bytes-old_bytes==(40-4)*sizeof(Real));
  refuse([&]{calculate(grown_bytes-1,&owner);});
  ExactHistoryGuard::require_same_field_contract_(keeper,calculate(grown_bytes,&owner),"ProgramContext history store");++checks;
  std::cout<<"corrected owner ghosts, budget and strict guard received "<<checks<<"\n";
#endif
}
'''
    source = tmp_path/"contract.cpp"
    source.write_text(cpp)
    executable = tmp_path/"contract"
    args = ["/usr/bin/clang++", "-std=c++20", "-O0", "-DPOPS_REAL_TYPE="+real,
            str(source), "-o", str(executable)]
    if historical:
        args.insert(3, "-DHISTORICAL")
    subprocess.run(args, check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert ("historical exact ranked refusal" if historical else "corrected owner ghosts") in result.stdout


def test_all_providers_select_the_actual_owner_before_kernel():
    uniform = (ROOT/"include/pops/runtime/program/program_context.hpp").read_text()
    start = uniform.index("  field_type spatial_interaction(int program_block")
    end = uniform.index("  /// Reduce one generated", start)
    method = uniform[start:end]
    assert "output_prototype = &system_->block_state(owner);" in method
    assert method.index("output_prototype =") < method.index("auto result = direct_spatial_interaction")
    assert "authority, lane, kernel, output_prototype);" in method
    amr = (ROOT/"include/pops/runtime/program/amr_program_context_spatial_interaction.inc").read_text()
    assert amr.count("output_prototype = &facade_->prepared_amr_block_state(owner, active_level_);") == 2
    assert amr.count("lane, kernel, output_prototype);") == 2
