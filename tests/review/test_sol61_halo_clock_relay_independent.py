"""Independent Source/actual public C++ primitive review, no Native reception."""
from pathlib import Path
import subprocess
import sys
import json
import re
import pops
from tests.python.support.tag_selection_case import build
from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata

ROOT = Path(__file__).resolve().parents[2]


def test_actual_source_and_public_primary_namespace():
    assert Path(pops.__file__).resolve() == ROOT / "python/pops/__init__.py"
    assert not any(n == "_pops" or n.endswith("._pops") for n in sys.modules)
    case, layout = build((8, 12), 0, "linear")
    pops.resolve(pops.validate(case), layout=layout)
    program = case._time
    manifest = program.temporal_manifest()
    emitted = _emit_checkpoint_shape_metadata(program)
    match = re.search(r'pops_program_checkpoint_primary_clock_identity\(\)\s*\{\s*return\s*("(?:[^"\\]|\\.)*");', emitted)
    assert match is not None
    exported = json.loads(match[1])
    assert exported == program.clock.qualified_id == manifest["primary_clock"]
    assert exported != program.clock.name
    assert "pops.amr.topology-rematerialization.accepted" != exported
    assert "pops.amr.restart-regrid.accepted" != exported


def test_real_public_clock_header_exact_partitions_and_refusals(tmp_path):
    cpp = tmp_path / "clock.cpp"
    cpp.write_text(r'''#include "pops/numerics/time/amr/levels/amr_clock.hpp"
#include <cassert>
#include <cmath>
using namespace pops::amr;
template<class F> void refuses(F f) { bool caught=false; try { f(); } catch(const std::exception&) {caught=true;} assert(caught); }
int main() {
 const Rational ratios[]={{2,1},{5,2},{3,1},{7,3}};
 const Rational independentLastSpans[]={{1,2},{1,10},{1,30},{1,210}};
 const unsigned counts[]={2,3,3,3};
 const double dt=.125; const long tick=37;
 ClockWindow window{{0,tick,{0,1},0},{0,tick,{1,1},dt}};
 for(int level=0;level<4;++level) {
   ParentChildClockRelation relation(level,level+1,ratios[level],RemainderPolicy::ExplicitFinalSubstep);
   auto rows=relation.partition(window); assert(rows.size()==counts[level]);
   Rational sum{0,1}; auto cursor=window.begin.phase;
   for(const auto& row: rows) {
     assert(row.window.begin.phase==cursor); assert(row.window.begin.level==level+1);
     assert(row.window.begin.macro_step==tick && row.window.end.macro_step==tick);
     assert(row.window.begin.phase<row.window.end.phase);
     sum=sum+row.window.end.phase-row.window.begin.phase;cursor=row.window.end.phase;
   }
   assert(cursor==window.end.phase); assert(sum==window.end.phase-window.begin.phase);
   window=rows.back().window;
   assert(window.end.phase-window.begin.phase==independentLastSpans[level]);
   assert(dt*(window.end.phase-window.begin.phase).value()==dt*independentLastSpans[level].value());
   assert(rows.back().is_declared_remainder==!ratios[level].integral());
 }
 // Large absolute time must not be used to infer a small interval: it rounds away.
 assert(1e16+dt==1e16); assert(dt*independentLastSpans[3].value()>0);
 refuses([]{ParentChildClockRelation r(0,2,{2,1},RemainderPolicy::IntegralOnly);});
 refuses([]{ParentChildClockRelation r(0,1,{1,2},RemainderPolicy::IntegralOnly);});
 ParentChildClockRelation r(0,1,{5,2},RemainderPolicy::IntegralOnly);
 ClockWindow good{{0,37,{0,1},0},{0,37,{1,1},.125}};
 refuses([&]{r.partition(good);});
 ParentChildClockRelation exact(0,1,{2,1},RemainderPolicy::IntegralOnly);
 auto wrong=good;wrong.begin.level=1;refuses([&]{exact.partition(wrong);});
 wrong=good;wrong.end.phase={0,1};refuses([&]{exact.partition(wrong);});
 wrong=good;wrong.end.physical_time=0;refuses([&]{exact.partition(wrong);});
 refuses([&]{good.alpha({0,38,{1,2},.0625});});
 refuses([&]{good.alpha({1,37,{1,2},.0625});});
 refuses([&]{good.alpha({0,37,{2,1},.25});});
}
''')
    binary = tmp_path / "clock"
    subprocess.run(["/usr/bin/clang++", "-std=c++20", "-O0", "-I", str(ROOT / "include"), str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
