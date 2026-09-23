"""Execute the native moment map against independent positive particle measures."""
from pathlib import Path
import subprocess

import pytest


SOURCE = r'''
#include <pops/numerics/moments/affine_velocity.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using pops::Real;
using Basis = pops::moments::CartesianMomentBasis<4>;

static void require(bool value, const char* message) {
  if (!value) { std::cerr << message << '\n'; std::exit(1); }
}
static void add(Real (&moments)[15], Real weight, Real vx, Real vy) {
  for (int q = 0; q <= 4; ++q)
    for (int p = 0; p + q <= 4; ++p)
      moments[Basis::index(p,q)] += weight * std::pow(vx,p) * std::pow(vy,q);
}
static Real variance_trace(const Real (&moments)[15]) {
  const Real ux = moments[1]/moments[0], uy = moments[5]/moments[0];
  return (moments[2]+moments[9])/moments[0] - ux*ux - uy*uy;
}

int main() {
  const Real xs[] = {-1.25, 0.125, 1.75};
  const Real ys[] = {-0.875, 0.25, 1.125};
  const Real weights[] = {0.125, 0.25, 0.625};
  Real maximum_error = 0;
  int cases = 0;
  for (Real metric : {Real(1), Real(7.5)}) {
    Real old[15]{};
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
      add(old, metric*weights[i]*weights[j], xs[i], ys[j]);
    for (Real omega : {Real(-6.28318531e12), Real(-2), Real(0), Real(2), Real(1e200)}) {
      for (Real half_dt : {Real(0), Real(1e-8), Real(0.0025), Real(1e200)}) {
        // Independent Cayley angle; atan(infinity) correctly reaches the limiting rotation.
        const Real angle = Real(2)*std::atan(omega*half_dt);
        const Real c = std::cos(angle), s = std::sin(angle);
        // These are first moments from the coupled frozen-field CN solve with
        // E=-Omega*J*u_d, u_d=(0.375,-0.625), hence u+=u_d+R*(u-u_d).
        const Real drift_x=0.375, drift_y=-0.625;
        const Real bx=drift_x-c*drift_x-s*drift_y;
        const Real by=drift_y+s*drift_x-c*drift_y;
        Real expected[15]{};
        for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
          add(expected, metric*weights[i]*weights[j],
              c*xs[i]+s*ys[j]+bx, -s*xs[i]+c*ys[j]+by);
        Real endpoint[15], output[15];
        std::copy(old, old+15, endpoint);
        endpoint[1]=expected[1]; endpoint[5]=expected[5];
        require(pops::moments::affine_velocity_push_forward<4>(
            old, endpoint, omega, half_dt, output), "valid positive measure refused");
        require(output[0]==old[0] && output[1]==endpoint[1] && output[5]==endpoint[5],
                "density or solved first moments changed");
        for (int k=0; k<15; ++k) {
          const Real error=std::fabs(output[k]-expected[k])/std::max(Real(1),std::fabs(expected[k]));
          maximum_error=std::max(maximum_error,error);
          require(error < Real(3e-13), "common affine map differs from positive particle push-forward");
        }
        require(std::fabs(variance_trace(output)-variance_trace(old)) < Real(1e-12),
                "rotation changed central second-moment trace");
        ++cases;
      }
    }
    Real bad[15], output[15];
    std::copy(old,old+15,bad); std::fill(output,output+15,Real(42));
    bad[0] *= Real(2);
    require(!pops::moments::affine_velocity_push_forward<4>(old,bad,1,1,output),
            "changed density was accepted");
    for (Real value : output) require(value==Real(42), "refused map partially published");
    std::copy(old,old+15,bad); bad[7]=std::numeric_limits<Real>::quiet_NaN();
    require(!pops::moments::affine_velocity_push_forward<4>(old,bad,1,1,output),
            "non-finite endpoint was accepted");
    require(!pops::moments::affine_velocity_push_forward<4>(old,old,1,-1,output),
            "negative source interval was accepted");
  }
  std::cout << "cases=" << cases << " max_relative_moment_error=" << maximum_error << '\n';
}
'''


@pytest.mark.compiler
def test_native_affine_velocity_preserves_full_positive_measure_and_solved_means(tmp_path, native_cxx):
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / "affine_velocity.cpp"
    executable = tmp_path / "affine_velocity"
    source.write_text(SOURCE)
    subprocess.run([native_cxx, "-std=c++20", "-O2", "-I", str(root / "include"),
                    str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "cases=40" in result.stdout
    print(result.stdout.strip())
