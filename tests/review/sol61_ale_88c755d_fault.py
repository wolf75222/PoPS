"""Bounded fault injection into the exact 88c755d local numeric publication phase.

Run from repository root. This confirms a review finding, not native qualification.
The extracted production body is unmodified; storage and execution are host probes.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ale-sol61-88c755d-source"
OUT.mkdir(parents=True, exist_ok=True)
revision = sys.argv[1] if len(sys.argv) > 1 else "88c755d"
source = subprocess.check_output([
    "git", "show", revision + ":include/pops/runtime/program/program_context_moving_interval.inc"
], cwd=ROOT, text=True)
start = source.index("      Real invalid = 0;")
end = source.index("      stage_exchange_batch", start)
body = source[start:end]
has_local_guard = re.search(r"\bcatch\b", body) is not None
host = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
using Real=double;
constexpr int Dim=1;
#define POPS_HD
template<int> using Index=std::array<int,1>;
namespace Kokkos {
  inline double abs(double x) {return x<0?-x:x;}
  inline void fence() {}
}
struct Box {};
struct View {
  double* data;
  double& operator()(const Index<1>&, int=0) const { return *data; }
};
struct Fab { double value=1.; View view() const {return {const_cast<double*>(&value)};} };
struct Face { Fab value; template<int> const Fab& field() const {return value;} };
struct Field {
  Fab value;
  std::size_t local_size() const {return 1;}
  Box box(std::size_t) const {return {};}
  const Fab& fab(std::size_t) const {return value;}
};
struct Geometry { std::vector<Face> coordinates{Face{}}; Field measures; };
struct SweptInterval {
  static double integrated_amount_update(double q,double l,double r,double fl,double fr,
                                          double s,double sl,double sr) {
    return q-(fr-fl)+r*sr-l*sl+s;
  }
};
enum class SolveStatus { kInvalidEvaluation };
struct StepAttemptRejected:std::runtime_error {
  StepAttemptRejected(SolveStatus,const char*,const char* why):std::runtime_error(why) {}
};
int local_launches=0, numeric_votes=0, error_votes=0;
template<class F> double for_each_cell_reduce_max(Box,F) {
  ++local_launches;
  throw std::runtime_error("injected rank-local execution failure");
}
double all_reduce_max(double value,int) {++numeric_votes; return value;}
void collectively_rethrow_exception(std::exception_ptr error,int,const char*) {
  ++error_votes;
  if (error) std::rethrow_exception(error);
}
void phase() {
  const Field input, integrated_source;
  const Geometry accepted;
  std::optional<Geometry> candidate{accepted};
  std::optional<Field> updated{input};
  const std::vector<Face> proposed_coordinates{Face{}}, swept_volumes{Face{}},
                          integrated_physical_flux{Face{}}, face_density{Face{}};
  const int components=1, lane=0;
  const double geometry_tolerance=1e-13;
  std::exception_ptr error;
'''
host += body
host += r'''
}
int main() {
  try {phase(); return 1;}
  catch(const std::runtime_error&) {}
  assert(local_launches==1 && numeric_votes==0);
}
'''
host = host.replace("assert(local_launches==1 && numeric_votes==0);",
                    "assert(local_launches==1 && numeric_votes==0 && error_votes==%d);"
                    % (1 if has_local_guard else 0))
translation_unit = OUT / "numeric_collective_fault.cpp"
translation_unit.write_text(host)
executable = OUT / "numeric_collective_fault"
compiler = shutil.which("clang++") or shutil.which("c++")
assert compiler, "host C++20 compiler required"
subprocess.run([compiler, "-std=c++20", "-O0", str(translation_unit), "-o", str(executable)],
               check=True, capture_output=True, text=True)
subprocess.run([str(executable)], check=True)
print(revision + " exact numeric phase: " +
      ("rank-local failure reaches collective error vote" if has_local_guard else
       "rank-local exception skips its collective vote (confirmed defect)"))
