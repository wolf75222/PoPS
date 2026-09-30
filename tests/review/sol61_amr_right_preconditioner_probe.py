"""Actual C++ phases with explicit host storage/collective substitutes, never AMR.

The source owns Arnoldi/right-apply/update/recheck, diagonal admission and
authority decisions. The scaffold owns vectors, one-cell views and transport.
"""

import hashlib
import json
from pathlib import Path
import subprocess


def method(source, marker):
    start = source.index(marker)
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def run_probe(root, directory, *, workspace_source=None, prepared_source=None):
    root, directory = Path(root), Path(directory)
    paths = (
        root / "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp",
        root / "include/pops/runtime/program/prepared_amr_field_residual.hpp",
    )
    workspace = paths[0].read_text() if workspace_source is None else workspace_source
    prepared = paths[1].read_text() if prepared_source is None else prepared_source
    phase = (
        "enum class LinearFailure"
        + workspace.split("enum class LinearFailure", 1)[1].split("// Covered/EB", 1)[0]
    )
    authority = method(prepared, "void require_authority(")
    invocation = method(prepared, "std::string_view invocation_identity()")
    diagonal = prepared[
        prepared.index("const Real invalid = for_each_cell_reduce_max(singleton,") : prepared.index(
            '"spatial-basis Jacobi requires a finite nonzero actual active spatial diagonal"'
        )
    ]
    diagonal += '"spatial-basis Jacobi requires a finite nonzero actual active spatial diagonal");'
    count = method(prepared, "std::size_t applications = 1;")
    cpp = (
        r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
using Real=double;
namespace Kokkos { void fence() {} }
struct ExecutionLane { int identity; };
void need(bool value) { if(!value) throw std::runtime_error("independent host assertion"); }
struct Engine {
  using hierarchy_type=std::vector<Real>;
  struct { int restart=3,linear_max_iterations=9; } options_;
  std::vector<hierarchy_type> basis_; hierarchy_type linear_residual_,work_,correction_,image_;
  std::vector<Real> hessenberg_,cosine_,sine_,rotated_rhs_,coefficients_,measure_;
  std::vector<bool> active_;
  Engine(int n,int restart=3,int budget=9):measure_(n,1),active_(n,true) {
    options_.restart=restart; options_.linear_max_iterations=budget;
    basis_.resize(restart+1,hierarchy_type(n));
    for(auto* value:{&linear_residual_,&work_,&correction_,&image_}) value->resize(n);
    hessenberg_.resize((restart+1)*restart); cosine_.resize(restart); sine_.resize(restart);
    rotated_rhs_.resize(restart+1); coefficients_.resize(restart);
  }
  void copy_(const hierarchy_type&a,hierarchy_type&b){b=a;}
  void scale_(hierarchy_type&a,Real s){for(auto&v:a)v*=s;}
  void saxpy_(hierarchy_type&a,Real s,const hierarchy_type&b){for(std::size_t i=0;i<a.size();++i)a[i]+=s*b[i];}
  void lincomb_(hierarchy_type&a,Real s,const hierarchy_type&b,Real t,const hierarchy_type&c){
    for(std::size_t i=0;i<a.size();++i)a[i]=s*b[i]+t*c[i];}
  Real dot_(const hierarchy_type&a,const hierarchy_type&b,const ExecutionLane&){
    Real sum=0;for(std::size_t i=0;i<a.size();++i)if(active_[i])sum+=measure_[i]*a[i]*b[i];return sum;}
  Real norm_(const hierarchy_type&a,const ExecutionLane&lane){return std::sqrt(dot_(a,a,lane));}
  void project_unknowns_(hierarchy_type&a){for(std::size_t i=0;i<a.size();++i)if(!active_[i])a[i]=0;}
  template<class Body>void local_phase_(Body body){body();}
  static bool finite_(Real v){return std::isfinite(v);}
  Real&h_(int row,int col){return hessenberg_[col*(options_.restart+1)+row];}
"""
        + phase
        + r"""
};
enum class AmrFieldRightPreconditioner { kIdentity,kSpatialBasisJacobi };
struct Attempt { int id=1;bool alive=true;bool visible()const{return alive;}
  bool same_attempt(const Attempt&b)const{return id==b.id;}int ordinal()const{return id;} };
struct Point {std::string clock="main";int tick=1,level=0,substep=0,stage=0;
  struct Rational{int numerator=0,denominator=1;bool operator==(const Rational&)const=default;}stage_fraction;
  Real dt=.1,physical_time=.2;std::string graph_identity="graph",rate_identity="rate",application_identity="body";
  bool operator==(const Point&)const=default;};
struct AmrFieldResidualAuthority {const void*owner=nullptr;Attempt attempt;
  std::vector<Attempt>level_attempts{{}};int topology_epoch=2,materialization_generation=4;
  std::string original_equation_identity="original";std::vector<std::string>capture_identities{"capture"};
  std::vector<int>capture_components{3};std::vector<Point>points{{}};};
struct ExactContractBuilder {std::string data;
  ExactContractBuilder&text(std::string_view s){data+=std::string(s)+"|";return *this;}
  template<class T>ExactContractBuilder&scalar(T v){data+=std::to_string(v)+"|";return *this;}
  template<class Values,class Emit>ExactContractBuilder&sequence(const Values&vs,Emit emit){for(auto&v:vs)emit(*this,v);return *this;}
  std::string release(){return data;}};
bool agreement=true;
bool all_ranks_agree_exact_ordered_byte_pairs(const std::vector<std::pair<std::string,std::string>>&,const ExecutionLane&){return agreement;}
struct Operator {const ExecutionLane*lane;std::uint64_t generation=7;
  std::uint64_t original_field_preparation_generation()const{return generation;}
  const ExecutionLane&original_field_execution_lane()const{return *lane;}
  std::string original_field_contract()const{return "operator";}};
struct AuthorityHarness {
  static constexpr std::string_view identity="pops.prepared-amr-original-field-residual@1";
  static constexpr std::string_view preconditioned_identity="pops.prepared-amr-original-field-residual@2";
  AmrFieldRightPreconditioner preconditioner_=AmrFieldRightPreconditioner::kSpatialBasisJacobi;
  AmrFieldResidualAuthority authority_;Operator*op_;std::uint64_t coefficient_generation_=7;Real step_=1e-5;
  struct {Real tolerance=1e-7,linear_tolerance=1e-5,armijo=1e-4,minimum_step=1e-8;
    int max_iterations=24,linear_max_iterations=240,restart=80;}options_;
  AuthorityHarness(Operator&op):op_(&op){authority_.owner=this;}
  static inline int last_validation_lane=-1;
  template<class Body>static void local_phase_(const ExecutionLane&lane,Body body){last_validation_lane=lane.identity;body();}
"""
        + invocation
        + "\n"
        + authority
        + r"""
};
template<int Dim>struct Index {int value=0;};
#define POPS_HD
struct View {Real*value;Real&operator()(Index<1>,int)const{return *value;}};
template<class Kernel>Real for_each_cell_reduce_max(Index<1>cell,Kernel kernel){return kernel(cell);}
Real admit(Real response,Real offset,Real mask){
  constexpr int Dim=1;const Index<Dim>singleton{};const int component=0;
  Real reciprocal=0;const View value{&response},zero{&offset},active{&mask},inverse{&reciprocal};
"""
        + diagonal
        + r"""
  return reciprocal;
}
struct CountBox{std::size_t points;std::size_t numPts()const{return points;}};
struct CountField{std::vector<CountBox>boxes;int width;
  const auto&layout()const{return boxes;}int ncomp()const{return width;}};
std::size_t count_applications(const std::vector<CountField>&candidate_){
"""
        + count
        + r"""
  return applications;
}
int main(){
  ExecutionLane lane{1},foreign{2};int checks=0;
  for(auto permutation:{std::array<int,3>{0,1,2},std::array<int,3>{2,0,1},std::array<int,3>{1,2,0}}){
    const Real original[3][3]={{-3,.25,.125},{.5,7,-.25},{-.125,.75,2}};
    const Real forcing[3]={1.25,-.75,.5};Engine e(4,3,9);e.active_[3]=false;
    e.measure_={.5,.125,2,99};std::vector<Real>rhs(4),iterate(4);
    for(int i=0;i<3;++i)rhs[i]=forcing[permutation[i]];
    int right_calls=0,jvp_calls=0;
    auto right=[&](const auto&input,auto&out){++right_calls;
      for(int i=0;i<3;++i)out[i]=input[i]/original[permutation[i]][permutation[i]];
      out[3]=999;};
    auto jvp=[&](const auto&,const auto&direction,auto&out,int){++jvp_calls;
      need(direction[3]==0);for(int i=0;i<3;++i){out[i]=0;for(int j=0;j<3;++j)out[i]+=original[permutation[i]][permutation[j]]*direction[j];}out[3]=123;};
    const auto result=e.solve_linear_(iterate,rhs,1e-12,jvp,0,lane,right);
    need(result.converged && result.columns<=9 && right_calls>result.columns);
    std::vector<Real>image(4);jvp(iterate,e.correction_,image,0);
    Real residual=0;for(int i=0;i<3;++i)residual=std::hypot(residual,image[i]-rhs[i]);
    need(residual<1e-12 && e.correction_[3]==0 && result.evaluations==jvp_calls-1);++checks;
  }
  {
    Engine e(2,2,6);std::vector<Real>rhs{3,4},iterate(2);
    auto right=[](const auto&v,auto&out){out[0]=v[0]/4;out[1]=v[1]/4;};
    auto approximate=[](const auto&,const auto&v,auto&out,int){
      Real factor=1+.2*std::hypot(v[0],v[1]);out[0]=factor*v[0];out[1]=factor*v[1];};
    const auto result=e.solve_linear_(iterate,rhs,1e-10,approximate,0,lane,right);
    std::vector<Real>actual(2);approximate(iterate,e.correction_,actual,0);
    Real norm=std::hypot(rhs[0]-actual[0],rhs[1]-actual[1]);
    need(!result.converged || norm<=1e-10);need(result.columns<=6);
    need(std::abs(result.residual_norm-norm)<1e-12);++checks;
  }
  need(admit(4,7,1)==Real(-1)/3);++checks;
  need(admit(7,7,0)==0);++checks;
  need(std::isfinite(admit(-1e308,0,1)));++checks;
  for(auto pair:{std::pair<Real,Real>{0,0},{std::numeric_limits<Real>::infinity(),0},
      {1e-320,0},{std::numeric_limits<Real>::quiet_NaN(),0}}){
    bool rejected=false;try{admit(pair.first,pair.second,1);}catch(const std::invalid_argument&){rejected=true;}need(rejected);++checks;}
  Operator op{&lane};AuthorityHarness harness(op);auto current=harness.authority_;
  harness.require_authority(current,lane);++checks;
  bool foreign_refused=false;try{harness.require_authority(current,foreign);}catch(const std::logic_error&){foreign_refused=true;}
  const int foreign_validation_lane=AuthorityHarness::last_validation_lane;
  for(int attack=0;attack<20;++attack){auto changed=current;
    if(attack==0)changed.owner=&op; if(attack==1)changed.attempt.alive=false;
    if(attack==2)changed.points[0].dt=.3;if(attack==3)changed.topology_epoch+=1;
    if(attack==4)changed.capture_identities[0]="foreign";if(attack==5)changed.level_attempts[0].id=9;
    if(attack==6)op.generation=8;
    if(attack==7)changed.points[0].clock="retimed";
    if(attack==8)changed.points[0].stage_fraction={1,3};
    if(attack==9)changed.points[0].physical_time=std::nextafter(changed.points[0].physical_time,1.);
    if(attack==10)changed.points[0].level=1;
    if(attack==11)changed.points[0].application_identity="foreign-body";
    if(attack==12)changed.capture_components[0]=2;
    if(attack==13)changed.materialization_generation+=1;
    if(attack==14)changed.attempt.id=2;
    if(attack==15)changed.level_attempts[0].alive=false;
    if(attack==16)changed.points[0].graph_identity="foreign-graph";
    if(attack==17)changed.points[0].rate_identity="foreign-rate";
    if(attack==18)changed.points[0].substep=1;
    if(attack==19)changed.points[0].stage=1;
    bool refused=false;try{harness.require_authority(changed,lane);}catch(const std::logic_error&){refused=true;}
    need(refused);op.generation=7;++checks;
  }
  agreement=false;bool disagreement_refused=false;
  try{harness.require_authority(current,lane);}catch(const std::logic_error&){disagreement_refused=true;}
  need(disagreement_refused);agreement=true;++checks;
  need(count_applications({{{{8},{15}},5},{{{4}},3}})==128);++checks;
  const std::size_t large=std::size_t(1)<<62;
  need(count_applications({{{{large}},3}})==1+3*large);++checks;
  bool count_refused=false;try{count_applications({{{{std::size_t(std::numeric_limits<std::int64_t>::max())}},3}});}
    catch(const std::length_error&){count_refused=true;}need(count_refused);++checks;
  std::cout<<"{\"checks\":"<<checks<<",\"foreign_lane_refused\":"<<(foreign_refused?"true":"false")
    <<",\"foreign_validation_lane\":"<<foreign_validation_lane<<"}\n";
}
"""
    )
    directory.mkdir(parents=True, exist_ok=True)
    source, binary = directory / "right_probe.cpp", directory / "right_probe"
    source.write_text(cpp)
    compilation = subprocess.run(
        ["/usr/bin/clang++", "-std=c++20", "-O0", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if compilation.returncode:
        raise RuntimeError("host probe compilation failed: " + compilation.stderr)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)
    receipt = json.loads(result.stdout)
    receipt["source_headers"] = {
        str(path.relative_to(root)): hashlib.sha256(text.encode()).hexdigest()
        for path, text in zip(paths, (workspace, prepared), strict=True)
    }
    receipt["scaffold_sha256"] = hashlib.sha256(cpp.encode()).hexdigest()
    receipt["source_phase_sha256"] = {
        "workspace": hashlib.sha256(workspace.encode()).hexdigest(),
        "prepared": hashlib.sha256(prepared.encode()).hexdigest(),
    }
    receipt["scope"] = "actual phases; host storage/transport; no Kokkos/MPI/AMR native reception"
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.checkout, args.directory), indent=2))
