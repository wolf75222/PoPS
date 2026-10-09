"""Real constructor-prefix guard + real Rational/point; storage/MPI are not received."""

import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"


@pytest.fixture(scope="module")
def observations(tmp_path_factory):
    directory = tmp_path_factory.mktemp("original-amr-point-host")
    source = HEADER.read_text()
    start = source.index("    const int levels = op.original_field_levels();")
    stop = source.index("    candidate_ = allocate_(layouts);", start)
    # Unchanged constructor prefix; numerical storage/solver allocation is excluded.
    guard = source[start:stop]
    scaffold = r"""
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <cmath>
#include <iostream>
#include <vector>
using Real=double;
using Point=pops::runtime::multiblock::BoundaryEvaluationPoint;
struct Authority {
 std::string original_equation_identity="independent.original-body@1";
 std::vector<std::string> capture_identities;
 std::vector<int> capture_components,level_attempts{1,2};
 std::vector<Point> points;
};
struct Shape {
 int original_field_levels()const{return 2;}
 const int& original_field_layout(int)const{static int shape=1;return shape;}
 const int& original_field_active_cells(int)const{static int mask=1;return mask;}
 double original_field_cell_measure(int i)const{return i==0?.25:.125;}
};
struct Guard {
 using field_type=int;
 Authority authority_;
 void check(Shape& op) {
  std::vector<int> captures;
  GUARD
 }
};
int main(){
 bool comma=false;std::cout<<"{";
 for(auto image:std::vector<std::pair<long,long>>{{0,1},{1,2},{1,1},{0,0},{2,1},{2,4},{-1,1}}){
  Guard check;
  for(int level=0;level<2;++level){
   Point p; p.clock="macro";p.tick=1;p.level=level;p.dt=.125;p.physical_time=.125;
   p.graph_identity="independent.program@1";p.rate_identity="independent.body@1";
   p.application_identity="independent.solve@1";
   p.stage_fraction.numerator=image.first;p.stage_fraction.denominator=image.second;
   check.authority_.points.push_back(p);
  }
  bool refused=false;Shape shape;
  try{check.check(shape);}catch(const std::exception&){refused=true;}
  if(comma)std::cout<<",";comma=true;
  std::cout<<"\""<<image.first<<"/"<<image.second<<"\":"<<(refused?"true":"false");
 }
 std::cout<<"}\n";
}
""".replace("GUARD", guard)
    cpp = directory / "point.cpp"
    binary = directory / "point"
    cpp.write_text(scaffold)
    subprocess.run(
        [
            "/usr/bin/clang++",
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(subprocess.check_output([str(binary)], text=True))


@pytest.mark.parametrize("image", ("0/1", "1/2", "1/1"))
def test_real_point_guard_preserves_canonical_images(observations, image):
    assert observations[image] is False


@pytest.mark.parametrize("image", ("0/0", "2/1", "2/4", "-1/1"))
def test_real_point_guard_refuses_malformed_images(observations, image):
    assert observations[image] is True
