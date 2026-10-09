"""Independent actual-header authority lifetime/value probe; no Native runtime."""
from pathlib import Path
import shutil
import subprocess


def test_authority_owns_point_and_rejects_each_changed_coordinate(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cpp = r'''#include <pops/runtime/accepted_initial_field_point.hpp>
#include <cassert>
#include <limits>
using pops::runtime::PreparedAcceptedInitialFieldPointV1;
using pops::runtime::multiblock::BoundaryEvaluationPoint;
int main() {
  BoundaryEvaluationPoint p; p.clock="primary.actual"; p.physical_time=3.25; p.level=2; p.dt=0.;
  const auto original=p;
  PreparedAcceptedInitialFieldPointV1 a(p,p.clock,0,p.physical_time,2,true);
  p.clock="mutated.caller"; p.physical_time=100.; // caller storage cannot retarget authority
  assert(a.authenticates(original)); assert(!a.authenticates(p));
  auto q=original; q.level=1; assert(!a.authenticates(q));
  q=original; q.substep=1; assert(!a.authenticates(q));
  q=original; q.stage=1; assert(!a.authenticates(q));
  q=original; q.stage_fraction.numerator=1; assert(!a.authenticates(q));
  q=original; q.stage_fraction.denominator=2; assert(!a.authenticates(q));
  q=original; q.tick=1; assert(!a.authenticates(q));
  q=original; q.dt=std::numeric_limits<double>::quiet_NaN(); assert(!a.authenticates(q));
  q=original; q.physical_time=std::numeric_limits<double>::infinity(); assert(!a.authenticates(q));
  q=original; q.dt=-0.; assert(!a.authenticates(q));
}
'''
    source=tmp_path/"authority.cpp"; source.write_text(cpp)
    binary=tmp_path/"authority"
    compiler=shutil.which("clang++") or shutil.which("c++")
    assert compiler
    subprocess.run([compiler,"-std=c++20","-Wall","-Wextra","-Werror","-DPOPS_NATIVE_DIM=2","-I",str(root/"include"),str(source),"-o",str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True)
