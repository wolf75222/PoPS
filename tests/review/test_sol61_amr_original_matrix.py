"""Exact provider composition; scalar backend is a declared modal arithmetic substitute."""

import json
from pathlib import Path
import subprocess

import pytest

from test_sol61_amr_original_collective import method

ROOT = Path(__file__).resolve().parents[2]
D = ((1.4, -0.7, 0.2), (0.3, 1.1, -0.8), (-0.1, 0.6, 0.9))
REACTION = ((0.5, 0.2, 0), (-0.3, 0.4, 0.1), (0.1, 0, 0.7))
STATE = (0.4, -1.2, 2)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    folder = tmp_path_factory.mktemp("original-amr-matrix-host")
    source = (
        ROOT / "include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp"
    ).read_text()
    apply = method(source, "  void apply_impl_(const hierarchy_type& input,")
    cpp = r"""
#include <array>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <vector>
using Real=double;using ExecutionLane=int;
struct Field{std::vector<double>v;void set_val(double x){for(auto&t:v)t=x;}};
struct Scalar{
 std::array<Field,2>phi{{Field{{0}},Field{{0}}}},image=phi;
 double coefficient=0;bool arithmetic=false;
 Field&phi_level(int l){return phi[l];}Field&linear_image_level(int l){return image[l];}
 void apply_linear_composite(bool mean,const ExecutionLane*){
  arithmetic=mean;for(int l=0;l<2;++l)image[l].v[0]=2.5*coefficient*phi[l].v[0];
 }
};
struct Probe{
 using hierarchy_type=std::vector<Field>;
 struct{int components=3,coefficients=9;std::vector<double>reaction;}options_;
 ExecutionLane lane=0;const ExecutionLane*lane_=&lane;
 std::vector<std::unique_ptr<Scalar>>entries_;
 int level_count()const{return 2;}
 void authenticate_(const hierarchy_type&x){if(x.size()!=2)throw std::runtime_error("shape");}
 template<class F>void local_field_phase_(bool,F&&f){f();}
 static void copy_component_(const Field&a,int i,Field&b,int j){b.v[j]=a.v[i];}
 static void add_component_(const Field&a,int i,Field&b,int j,double c){b.v[j]+=c*a.v[i];}
 void mask_(hierarchy_type&){}
 APPLY
};
int main(){
 const double d[3][3]={{1.4,-.7,.2},{.3,1.1,-.8},{-.1,.6,.9}};
 const double r[3][3]={{.5,.2,0},{-.3,.4,.1},{.1,0,.7}},q[3]={.4,-1.2,2};
 std::cout<<std::setprecision(17)<<"[";bool comma=false;
 for(auto order:{std::array<int,3>{0,1,2},std::array<int,3>{2,0,1}}){
  Probe p;for(int i=0;i<3;++i)for(int j=0;j<3;++j){
   auto scalar=std::make_unique<Scalar>();scalar->coefficient=d[order[i]][order[j]];
   p.entries_.push_back(std::move(scalar));p.options_.reaction.push_back(r[order[i]][order[j]]);
  }
  std::vector<Field>input(2,Field{{q[order[0]],q[order[1]],q[order[2]]}}),output(2,Field{{0,0,0}});
  p.apply_impl_(input,output,true);
  if(comma)std::cout<<",";comma=true;std::cout<<"[";
  for(int i=0;i<3;++i){if(i)std::cout<<",";std::cout<<output[0].v[i];}
  std::cout<<"]";
  for(auto&scalar:p.entries_)if(!scalar->arithmetic)return 2;
 }std::cout<<"]\n";
}
""".replace("APPLY", apply)
    source, binary = folder / "matrix.cpp", folder / "matrix"
    source.write_text(cpp)
    subprocess.run(
        ["/usr/bin/clang++", "-std=c++20", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(subprocess.check_output([str(binary)], text=True))


@pytest.mark.parametrize("index,order", ((0, (0, 1, 2)), (1, (2, 0, 1))))
def test_original_provider_composes_signed_nonsymmetric_rows_without_transpose(
    images, index, order
):
    expected = tuple(
        sum((2.5 * D[i][j] + REACTION[i][j]) * STATE[j] for j in range(3)) for i in order
    )
    transposed = tuple(
        sum((2.5 * D[j][i] + REACTION[i][j]) * STATE[j] for j in range(3)) for i in order
    )
    assert images[index] == pytest.approx(expected, abs=1e-14)
    assert max(abs(a - b) for a, b in zip(images[index], transposed, strict=True)) > 0.1
