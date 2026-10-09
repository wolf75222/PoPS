"""SOURCE_ONLY compile the real retained Field freshness guard against its actual point header."""
from pathlib import Path
import hashlib
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = root / "src/runtime/amr/amr_system.cpp"
text = source.read_text()
start = text.index("[providers = prepared_ghosts]")
end = text.index("\n          prepared_fluxes.reserve", start)
fragment = text[start:end].rstrip()
assert fragment.endswith("});")
fragment = fragment[:-2]  # retain the actual lambda, remove push_back closing parenthesis/semicolon
cpp = r'''#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <optional>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>
namespace runtime = pops::runtime;
using Point = pops::runtime::multiblock::BoundaryEvaluationPoint;
struct FieldPlan {
  std::optional<Point> boundary_point; // A setter value is deliberately present in the adversary.
  std::optional<Point> accepted_halo_producer_point;
};
struct Dependencies { std::vector<const FieldPlan*> field_plans; };
struct Invocation { std::shared_ptr<Dependencies> dependencies; };
int main() {
  Point p{.clock="actual.primary", .tick=7, .level=3, .substep=0, .stage=0,
          .stage_fraction={1,1}, .dt=.125, .physical_time=123.0};
  FieldPlan plan{p, std::nullopt};
  std::vector<Invocation> prepared_ghosts{{std::make_shared<Dependencies>(Dependencies{{&plan}})}};
  auto guard = ''' + fragment + r''';
  int refusals = 0;
  const auto refuse = [&](const Point& target) {
    try { guard(target); } catch (const std::invalid_argument&) { ++refusals; return; }
    throw std::runtime_error("stale or missing producer witness was accepted");
  };
  refuse(p); // Exact old setter is insufficient without a real producer witness.
  plan.accepted_halo_producer_point = p;
  guard(p);
  auto bad=p; bad.clock="other"; refuse(bad);
  bad=p; ++bad.level; refuse(bad);
  bad=p; ++bad.tick; refuse(bad);
  bad=p; ++bad.stage; refuse(bad);
  bad=p; ++bad.substep; refuse(bad);
  bad=p; bad.stage_fraction={0,1}; refuse(bad);
  bad=p; bad.dt=std::nextafter(p.dt, 1.0); refuse(bad);
  bad=p; bad.physical_time=std::nextafter(p.physical_time, 124.0); refuse(bad);
  bad=p; bad.dt=0; plan.accepted_halo_producer_point=bad; refuse(bad);
  prepared_ghosts.clear();
  auto periodic = ''' + fragment + r''';
  periodic(bad); // No GhostBoundary provider; genuine initial periodic dt0 is admissible.
  if (refusals != 10) throw std::runtime_error("wrong refusal inventory");
  std::cout << "SOURCE_ONLY actual guard: 10 refusals + exact witness + periodic initial PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix="pops-halo-field-host-") as directory:
    path = Path(directory)
    (path / "probe.cpp").write_text(cpp)
    subprocess.run(["/usr/bin/clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-I", str(root / "include"), str(path / "probe.cpp"), "-o", str(path / "probe")], check=True)
    subprocess.run([str(path / "probe")], check=True)
print("guard_sha256=" + hashlib.sha256(fragment.encode()).hexdigest())
print("point_header_sha256=" + hashlib.sha256((root / "include/pops/runtime/multiblock/evaluation_point.hpp").read_bytes()).hexdigest())
