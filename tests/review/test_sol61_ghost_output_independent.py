"""Independent inputs/expectations; actual C++ helper, explicit shared host storage seams."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "dbc9e1cd725c894f615b76a459d47986a8febd03"
SHIM = "15f741b78f9bbd7531a5e0b8eb510944ee81a279"
HEADER = "include/pops/runtime/program/spatial_direct_interaction.hpp"


def block(text, marker):
    start = text.index(marker)
    opened = text.index("{", start)
    depth, end = 1, opened + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def git(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True)


@pytest.mark.parametrize("real", ("float", "double"))
def test_actual_complete_kernel_owner_width_masks_grown_budget_and_refusals(tmp_path, real):
    # Shared author storage seam is explicit. Consumer, compensated accumulator and strict
    # Uniform history guard are authentic. Independent cases below are not author test outputs.
    shim = git(SHIM, "tests/review/sol61_spatial_interaction_host.cpp")
    shim = shim.replace("namespace Kokkos {", "int allocations=0; bool agreement=true;\nnamespace Kokkos {")
    shim = shim.replace("View(int,size_t n,size_t m):", "View(int,size_t n,size_t m):")
    shim = shim.replace("width(m){}", "width(m){++allocations;}")
    shim = shim.replace("halo(ghosts){for(auto box:b)", "halo(ghosts){++allocations;for(auto box:b)")
    shim = shim.replace("const ExecutionLane&){return true;}", "const ExecutionLane&){return agreement;}")
    header = (ROOT / HEADER).read_text()
    consumer = "\n".join(line for line in header.splitlines() if not line.startswith("#include") and line != "#pragma once")
    accumulator = block((ROOT / "include/pops/mesh/execution/for_each.hpp").read_text(), "struct FiniteCompensatedSum {")+";"
    guard = block((ROOT / "include/pops/runtime/program/program_context.hpp").read_text(), "  static void require_same_field_contract_(")
    code = shim + "\nnamespace pops {" + accumulator + "}\n" + consumer
    code += "\nstruct Guard {using field_type=pops::MultiFab<2,Kokkos::HostSpace>;\n"+guard+"\n};\n"
    code += r'''
using namespace pops; using namespace pops::runtime::program;
using F=MultiFab<2,Kokkos::HostSpace>; int assertions=0;
void check(bool b){++assertions;if(!b)throw std::runtime_error("independent assertion "+std::to_string(assertions));}
template<class W> void refusal_before_allocation(W work){allocations=0;bool failed=false;try{work();}catch(const std::exception&){failed=true;}check(failed);check(allocations==0);}
int main(){
  Box<2> left({0,0},{0,0}),right({1,0},{1,0}),domain({0,0},{1,0});
  Geometry<2> geom{domain,{0,0},{2,1}};
  // Source tuple 2,0 has width2, owner has width5, source has width3. History width2.
  F source({left,right},{},{},3,{}),owner({left,right},{},{},5,{2,1});
  F mask({left,right},{},{},1,{}),covered({left,right},{},{},1,{});
  mask.set_val(1);covered.set_val(1);
  auto a=source.fab(0).view(),b=source.fab(1).view();
  a({0,0},0)=2;a({0,0},1)=999;a({0,0},2)=-3;
  b({1,0},0)=7;b({1,0},1)=888;b({1,0},2)=5;
  // Second level contributes a distinct signed value, then is coverage-excluded.
  F fine({left,right},{},{},3,{}); fine.set_val(0);
  fine.fab(0).view()({0,0},0)=11;fine.fab(0).view()({0,0},2)=-13;
  F fine_active({left,right},{},{},1,{}),fine_coverage({left,right},{},{},1,{});
  fine_active.set_val(1);fine_coverage.set_val(0);
  const std::array<int,2> components{2,0}; ExecutionLane lane;
  auto kernel=[](auto x,auto y){return Real(1)+x[0]-Real(2)*y[0];};
  const std::array<InteractionLevelView<2,Kokkos::HostSpace>,2> levels{{
      {&source,&mask,&covered,nullptr,geom},{&fine,&fine_active,&fine_coverage,nullptr,geom}}};
  auto calculate=[&](uint64_t cap,const F* output){return direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,components,cap,"independent-tuple",lane,kernel,output);};
  F ring({left,right},{},{},2,owner.ghosts()); auto result=calculate(100000,&owner);
  Guard::require_same_field_contract_(ring,result,"actual history guard");check(true);
  check(result.ncomp()==2);check(result.ghosts()==owner.ghosts());
  // Midpoints .5/1.5 and unit measures: weights at x=.5 are .5,-1.5.
  check(result.fab(0).view()({0,0},0)==Real(-9));
  check(result.fab(0).view()({0,0},1)==Real(-9.5));
  check(result.fab(1).view()({1,0},0)==Real(-7));
  check(result.fab(1).view()({1,0},1)==Real(-.5));
  check(result.fab(0).view()({-2,-1},0)==0);check(result.fab(1).view()({3,1},1)==0);
  auto valid=calculate(100000,nullptr);check(valid.ghosts()==Extent<2>{});
  check(valid.fab(0).view()({0,0},0)==result.fab(0).view()({0,0},0));
  bool strict=false;try{Guard::require_same_field_contract_(ring,valid,"actual history guard");}catch(const std::invalid_argument&){strict=true;}check(strict);
  const size_t baseline=2*(2+1+2)*2*sizeof(Real)+2*sizeof(InteractionLevelView<2,Kokkos::HostSpace>)+2*sizeof(int)
      +2*(sizeof(size_t)+2*sizeof(Box<2>)+sizeof(Index<2>))+2*(sizeof(F::fab_type)+sizeof(size_t));
  // Each 1x1 patch grows to5x3, output width2; budget uses result width, not owner width5.
  const size_t grown=baseline+2*15*2*sizeof(Real),zero=baseline+2*1*2*sizeof(Real);
  check(calculate(grown,&owner).ncomp()==2);check(calculate(zero,nullptr).ncomp()==2);
  refusal_before_allocation([&]{calculate(grown-1,&owner);});
  refusal_before_allocation([&]{calculate(zero-1,nullptr);});
  F foreign=owner;foreign.rank[0]=1;refusal_before_allocation([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.dist.replica=true;refusal_before_allocation([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.boxes[0].lo[0]=-1;refusal_before_allocation([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.halo[1]=-1;refusal_before_allocation([&]{calculate(100000,&foreign);});
  foreign=owner;foreign.halo[0]=INT_MAX;refusal_before_allocation([&]{calculate(100000,&foreign);});
  agreement=false;refusal_before_allocation([&]{calculate(grown,&owner);});agreement=true;
  refusal_before_allocation([&]{interaction_product(std::numeric_limits<size_t>::max(),2);});
  refusal_before_allocation([&]{interaction_add(std::numeric_limits<size_t>::max(),1);});
  // A coverage-excluded fine NaN remains excluded; selected active source NaN refuses.
  fine.fab(0).view()({0,0},0)=NAN;check(calculate(grown,&owner).fab(0).view()({0,0},0)==Real(-9));
  source.fab(0).view()({0,0},2)=NAN;bool finite=false;try{calculate(grown,&owner);}catch(const std::overflow_error&){finite=true;}check(finite);
  std::cout<<"independent assertions "<<assertions<<"\n";
}
'''
    source=tmp_path/"receive.cpp"
    source.write_text(code)
    exe=tmp_path/"receive"
    subprocess.run(["/usr/bin/clang++", "-std=c++20", "-O0", "-DPOPS_REAL_TYPE="+real, str(source), "-o", str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert "independent assertions 36" in result.stdout


def test_owner_route_is_exact_and_physical_loop_and_guards_unchanged():
    context="include/pops/runtime/program/program_context.hpp"
    now=(ROOT/context).read_text()
    old=git(BASE,context)
    assert block(now,"  static void require_same_field_contract_(") == block(old,"  static void require_same_field_contract_(")
    assert "output_prototype = &system_->block_state(owner);" in now
    amr=(ROOT/"include/pops/runtime/program/amr_program_context_spatial_interaction.inc").read_text()
    assert amr.count("output_prototype = &facade_->prepared_amr_block_state(owner, active_level_);")==2
    # Full quadrature, component selection, masks, finite guards and owner transport unchanged.
    before=git(BASE,HEADER)
    after=(ROOT/HEADER).read_text()
    assert before[before.index("  std::size_t row = 0;"):] == after[after.index("  std::size_t row = 0;"):]
    assert "pops.direct-spatial-interaction-output-owner@1" in after
    assert after.index("if (bytes > max_bytes)") < after.index("snapshot = Snapshot(")
    native_box=(ROOT/"include/pops/mesh/index/box.hpp").read_text()
    assert "detail::checked_box_index(static_cast<std::int64_t>(lo[axis]) - amount" in native_box
    assert "detail::checked_box_index(static_cast<std::int64_t>(hi[axis]) + amount" in native_box


def test_actual_box_extreme_grow_and_volume_overflow_under_ubsan(tmp_path):
    native=(ROOT/"include/pops/mesh/index/box.hpp").read_text()
    native="\n".join(line for line in native.splitlines() if not line.startswith("#include") and line!="#pragma once")
    # Alias seams are explicit; entire actual Box and its checked helpers are retained unchanged.
    code="""#include <array>\n#include <cstdint>\n#include <limits>\n#include <stdexcept>\n#include <cassert>\n#define POPS_HD\nnamespace pops {template<int D> using Index=std::array<int,D>;template<int D> using Extent=std::array<int64_t,D>;}\n"""+native+r'''
int main(){using namespace pops;int checked=0;
  Box<1> full({INT32_MIN},{INT32_MAX});assert(full.numPts()==INT64_C(4294967296));
  try{full.grow(0,1);}catch(const std::overflow_error&){++checked;}
  Box<2> wide({INT32_MIN,INT32_MIN},{INT32_MAX,INT32_MAX});
  try{wide.numPts();}catch(const std::overflow_error&){++checked;}
  Box<3> tall({0,0,0},{INT32_MAX,INT32_MAX,2});
  try{tall.numPts();}catch(const std::overflow_error&){++checked;}
  try{Box<1>::from_extents({INT64_MAX});}catch(const std::overflow_error&){++checked;}
  try{Box<2>({0,0},{1,1}).grow(-1,1);}catch(const std::invalid_argument&){++checked;}
  assert(checked==5);
}
'''
    source=tmp_path/"box.cpp"
    source.write_text(code)
    exe=tmp_path/"box"
    subprocess.run(["/usr/bin/clang++","-std=c++20","-fsanitize=undefined","-fno-sanitize-recover=all",str(source),"-o",str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
