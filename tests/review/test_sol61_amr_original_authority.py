"""Actual authority guard, attempts/cache/point/contract/options; operator and fence substitutes."""

import json
from pathlib import Path
import subprocess

import pytest

from test_sol61_amr_original_collective import method

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def authority(tmp_path_factory):
    folder = tmp_path_factory.mktemp("original-amr-authority-host")
    text = (ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    declaration = method(text, "struct AmrFieldResidualAuthority") + ";"
    require = method(text, "  void require_authority(")
    phase = method(text, "  template <class Operation>\n  static void local_phase_(")
    scaffold = (
        r"""
#include <pops/runtime/program/prepared_resource_cache.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <iostream>
#include <map>
namespace Kokkos{void fence(){}}
namespace pops::runtime::program {
DECLARATION
struct Operator {
 std::uint64_t generation=1;
 std::uint64_t original_field_preparation_generation()const{return generation;}
 std::string original_field_contract()const{return "independent.original-provider@1";}
};
struct Probe{
 static constexpr std::string_view identity="pops.prepared-amr-original-field-residual@1";
 AmrFieldResidualAuthority authority_;Operator*op_;
 std::uint64_t coefficient_generation_=1;double step_=1e-7;FieldNewtonOptions options_;
 REQUIRE
 PHASE
};
}
using namespace pops;using namespace pops::runtime::program;
int main(){
 auto lane=ExecutionLane::world("independent.amr.original.authority");
 PreparedResourceCache parent,level0,level1,foreign;
 int owner=0,other_owner=0;Operator op;Probe p;
 p.op_=&op;p.authority_.owner=&owner;p.authority_.attempt=parent.begin_attempt();
 p.authority_.level_attempts={level0.begin_attempt(),level1.begin_attempt()};
 p.authority_.topology_epoch=3;p.authority_.materialization_generation=8;
 p.authority_.original_equation_identity="independent.original-physical-body@1";
 p.authority_.capture_identities={"independent.capture@1"};p.authority_.capture_components={3};
 for(int level=0;level<2;++level){runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock="macro";point.tick=1;point.level=level;point.dt=.125;point.physical_time=.125;
  point.graph_identity="independent.program@1";point.rate_identity="independent.body@1";
  point.application_identity="independent.solve@1";p.authority_.points.push_back(point);}
 auto refuses=[&](const auto&a){try{p.require_authority(a,lane);}catch(const std::exception&){return true;}return false;};
 std::map<std::string,bool>result;
 result["baseline"]=refuses(p.authority_);
 for(int mode=0;mode<10;++mode){auto a=p.authority_;std::string name;
  if(mode==0){name="owner";a.owner=&other_owner;}
  if(mode==1){name="foreign_parent_same_ordinal";a.attempt=foreign.begin_attempt();}
  if(mode==2){name="foreign_level_same_ordinal";a.level_attempts[1]=a.attempt;}
  if(mode==3){name="equation";a.original_equation_identity+="forged";}
  if(mode==4){name="capture";a.capture_identities[0]+="forged";}
  if(mode==5){name="width";a.capture_components[0]=2;}
  if(mode==6){name="point";a.points[1].physical_time+=.125;}
  if(mode==7){name="topology";++a.topology_epoch;}
  if(mode==8){name="materialization";++a.materialization_generation;}
  if(mode==9){name="coefficient_generation";++op.generation;}
  result[name]=refuses(a);op.generation=1;
  if(refuses(p.authority_))return 2;
 }
 p.authority_.level_attempts[1].reject();result["rejected_level"]=refuses(p.authority_);
 p.authority_.level_attempts[1]=level1.begin_attempt();
 p.authority_.attempt.reject();result["rejected_parent"]=refuses(p.authority_);
 auto retry=p.authority_;retry.attempt=parent.begin_attempt();
 result["retry_is_not_old_invocation"]=refuses(retry);
 bool comma=false;std::cout<<"{";for(const auto&[name,value]:result){
  if(comma)std::cout<<",";comma=true;std::cout<<"\""<<name<<"\":"<<(value?"true":"false");
 }std::cout<<"}\n";
}
""".replace("DECLARATION", declaration)
        .replace("REQUIRE", require)
        .replace("PHASE", phase)
    )
    source, binary = folder / "authority.cpp", folder / "authority"
    source.write_text(scaffold)
    subprocess.run(
        [
            "/usr/bin/clang++",
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(subprocess.check_output([str(binary)], text=True))


def test_actual_authority_preserves_baseline_and_real_cache_attempts(authority):
    assert authority["baseline"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "owner",
        "foreign_parent_same_ordinal",
        "foreign_level_same_ordinal",
        "equation",
        "capture",
        "width",
        "point",
        "topology",
        "materialization",
        "coefficient_generation",
        "rejected_level",
        "rejected_parent",
        "retry_is_not_old_invocation",
    ),
)
def test_actual_authority_refuses_foreign_stale_and_revoked_images(authority, mutation):
    assert authority[mutation] is True
