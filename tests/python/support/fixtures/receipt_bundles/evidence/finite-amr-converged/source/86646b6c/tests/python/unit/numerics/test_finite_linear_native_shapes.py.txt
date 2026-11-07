"""Actual native finite map templates beyond the W06 widths, host only."""
from pathlib import Path
import shutil
import subprocess
import sys
import pytest


def test_rectangular_native_shapes_and_joint_failure(tmp_path):
    compiler=shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    kokkos=Path(sys.prefix)/"include"
    if not (kokkos/"Kokkos_Array.hpp").exists():
        pytest.skip("Kokkos headers unavailable")
    cpp=tmp_path/"native_shapes.cpp"
    cpp.write_text(r"""
#include <pops/numerics/linalg/finite_linear.hpp>
#include <cmath>
int main() {
  using pops::Real;
  const auto rectangle=pops::detail::finite_linear_apply<2,3>(
      Kokkos::Array<Real,6>{1,2,3,-1,0,1}, Kokkos::Array<Real,3>{2,3,4});
  if(rectangle[0]!=20 || rectangle[1]!=2) return 1;
  Kokkos::Array<Real,10> tall{};
  for(int i=0;i<5;++i){tall[2*i]=i+1;tall[2*i+1]=1-i;}
  const auto applied=pops::detail::finite_linear_apply<5,2>(tall,Kokkos::Array<Real,2>{2,-1});
  for(int i=0;i<5;++i)if(applied[i]!=3*i+1)return 2;
  const auto solved=pops::detail::finite_linear_solve<3,3>(
      Kokkos::Array<Real,9>{3,1,0,-1,4,2,0,-2,5},Kokkos::Array<Real,3>{1,-3,19});
  const Real expected[3]={1,-2,3};
  for(int i=0;i<3;++i)if(std::abs(solved[i]-expected[i])>1e-13)return 3;
  const auto singular=pops::detail::finite_linear_solve<3,3>(
      Kokkos::Array<Real,9>{1,2,3,2,4,6,0,0,0},Kokkos::Array<Real,3>{1,2,0});
  for(int i=0;i<3;++i)if(std::isfinite(singular[i]))return 4;
  // First mathematical row is finite (20); another row overflows. Selecting
  // only the first projection must still observe failure of the joint map.
  const auto joint=pops::detail::finite_linear_apply<2,3>(
      Kokkos::Array<Real,6>{1,2,3,1e308,0,0},Kokkos::Array<Real,3>{2,3,4});
  if(std::isfinite(joint[0]) || std::isfinite(joint[1]))return 5;
  return 0;
}
""")
    executable=tmp_path/"native_shapes"
    result=subprocess.run([compiler,"-std=c++20","-O2","-fno-fast-math",
        "-I"+str(Path(__file__).resolve().parents[4]/"include"),"-I"+str(kokkos),
        str(cpp),"-o",str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(executable)],check=True)
