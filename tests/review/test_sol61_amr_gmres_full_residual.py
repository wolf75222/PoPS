"""Run unchanged actual GMRES phase over host vector algebra; no AMR emulation."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp"


def test_actual_gmres_requires_full_jvp_and_preserves_budget(tmp_path):
    source = HEADER.read_text()
    phase = source.split("  enum class LinearFailure", 1)[1].split("  // Covered/EB", 1)[0]
    phase = "  enum class LinearFailure" + phase
    # All Arnoldi/rotation/update/recheck statements are the actual header body.
    # The seams implement scalar vector storage/algebra only, never a substitute
    # PDE or AMR solver. The independent native witness remains unchanged.
    cpp = r'''
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <stdexcept>
namespace Kokkos { void fence() {} }
using Real=double; struct ExecutionLane {};
struct Engine {
  using hierarchy_type=std::vector<Real>;
  struct { int restart=2, linear_max_iterations=6; } options_;
  std::vector<hierarchy_type> basis_;
  hierarchy_type linear_residual_, work_, correction_, image_;
  std::vector<Real> hessenberg_, cosine_, sine_, rotated_rhs_, coefficients_;
  Engine(int width,int restart=2,int budget=6) {
    options_.restart=restart; options_.linear_max_iterations=budget;
    basis_.resize(restart+1,hierarchy_type(width));
    linear_residual_.resize(width); work_.resize(width); correction_.resize(width); image_.resize(width);
    hessenberg_.resize((restart+1)*restart); cosine_.resize(restart); sine_.resize(restart);
    rotated_rhs_.resize(restart+1); coefficients_.resize(restart);
  }
  void copy_(const hierarchy_type& a,hierarchy_type& b) { b=a; }
  void scale_(hierarchy_type& a,Real s) { for(auto& x:a) x*=s; }
  void saxpy_(hierarchy_type& a,Real s,const hierarchy_type& b) {
    for(std::size_t i=0;i<a.size();++i) a[i]+=s*b[i]; }
  void lincomb_(hierarchy_type& out,Real a,const hierarchy_type& x,Real b,const hierarchy_type& y) {
    for(std::size_t i=0;i<out.size();++i) out[i]=a*x[i]+b*y[i]; }
  Real dot_(const hierarchy_type& a,const hierarchy_type& b,const ExecutionLane&) {
    Real v=0; for(std::size_t i=0;i<a.size();++i) v+=a[i]*b[i]; return v; }
  Real norm_(const hierarchy_type& a,const ExecutionLane& lane) { return std::sqrt(dot_(a,a,lane)); }
  void project_unknowns_(hierarchy_type&) {}
  template<class Body> void local_phase_(Body body) { body(); }
  static bool finite_(Real v) { return std::isfinite(v); }
  Real& h_(int row,int col) { return hessenberg_[col*(options_.restart+1)+row]; }
''' + phase + r'''
};
void check(bool v) { if(!v) throw std::runtime_error("actual GMRES check"); }
int main() {
  ExecutionLane lane;
  for(int width: {1,3,5}) {
    Engine e(width,3,12); std::vector<Real> rhs(width), iterate(width);
    for(int i=0;i<width;++i) rhs[i]=.3+.1*i;
    int calls=0;
    auto identity=[&](const auto&,const auto& v,auto& out,int) { ++calls; out=v; };
    const auto result=e.solve_linear_(iterate,rhs,1e-12,identity,0,lane);
    check(result.converged && result.columns<=12 && result.evaluations==calls);
    check(calls>result.columns); // Every projected stop gets a complete-correction JVP.
    Real residual=0; for(int i=0;i<width;++i) residual=std::hypot(residual,rhs[i]-e.correction_[i]);
    check(residual<=1e-12);
  }
  {
    Engine e(2,2,6); std::vector<Real> rhs{3,4}, iterate(2);
    int full=0;
    auto approximate=[&](const auto&,const auto& v,auto& out,int) {
      const Real norm=std::hypot(v[0],v[1]); const Real factor=norm>2 ? 2 : 1;
      if(norm>2) ++full; for(int i=0;i<2;++i) out[i]=factor*v[i]; };
    const auto result=e.solve_linear_(iterate,rhs,1e-8,approximate,0,lane);
    check(!result.converged && result.columns==6 && full>0);
    check(result.residual_norm>1e-8);
  }
  {
    Engine e(2,2,6); std::vector<Real> rhs{.3,.7}, iterate(2);
    constexpr Real a=64000000000000.36,b=-47999999999999.52,c=36000000000000.64;
    auto matrix=[&](const auto&,const auto& v,auto& out,int) {
      out[0]=a*v[0]+b*v[1]; out[1]=b*v[0]+c*v[1]; };
    const Real stop=1e-5*std::hypot(rhs[0],rhs[1]);
    const auto result=e.solve_linear_(iterate,rhs,stop,matrix,0,lane);
    std::vector<Real> image(2); matrix(iterate,e.correction_,image,0);
    const Real actual=std::hypot(rhs[0]-image[0],rhs[1]-image[1]);
    check(!result.converged || actual<=stop);
    check(result.columns<=6);
    std::cout<<std::hexfloat<<"stop="<<stop<<" residual="<<actual<<" converged="<<result.converged<<'\n';
  }
}
'''
    path, exe = tmp_path / "actual_gmres.cpp", tmp_path / "actual_gmres"
    path.write_text(cpp)
    subprocess.run(["/usr/bin/clang++", "-std=c++20", "-O0", str(path), "-o", str(exe)],
                   check=True, timeout=30, capture_output=True)
    subprocess.run([str(exe)], check=True, timeout=5, capture_output=True)


def test_current_source_never_accepts_rotated_rhs_without_actual_jvp():
    phase = HEADER.read_text().split("LinearResult solve_linear_", 1)[1].split("bool update_correction_", 1)[0]
    assert "cycle_converged" not in phase
    assert phase.count("result.converged = true;") == 2
    assert phase.index("apply_jvp(iterate, correction_, image_") < phase.rindex("if (beta <= stop)")
    assert "h_(row, column) += correction;" in phase
    assert "completed < options_.linear_max_iterations" in phase
    assert "std::min(options_.restart, options_.linear_max_iterations - completed)" in phase
