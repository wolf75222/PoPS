"""Actual non-iterable BoxArray and production capacity loop, not a vector adapter."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('dim',(1,2,3))
def test_actual_layout_capacity_loop(tmp_path,dim):
    cpp=(ROOT/'src/runtime/system/system.cpp').read_text()
    method=cpp[cpp.index('System<Dim>::checkpoint_state_carriers_capacity() const'):]
    begin=method.index('      std::uint64_t count = 0;')
    end=method.index('      cells = std::max(cells, count);',begin)
    body=method[begin:end].replace('field.layout()','input')
    source=tmp_path/'actual-layout.cpp'
    source.write_text('''#include <pops/mesh/layout/box_array.hpp>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
using namespace pops;
using Layout=pops::mesh::BoxArray<TEST_DIM>;
std::uint64_t count(const Layout& input) {
'''+body+'''return count; }
int main() {
  constexpr int D=TEST_DIM;
  Box<D> a{},b{};
  std::uint64_t expected=1;
  for(int axis=0;axis<D;++axis){a.lo[axis]=-3;a.hi[axis]=-2;b.lo[axis]=4;b.hi[axis]=6;expected*=2;}
  std::uint64_t second=1;for(int axis=0;axis<D;++axis)second*=3;
  Layout layout{std::vector<Box<D>>{a,b}};
  assert(count(layout)==expected+second);
  assert(count(Layout{})==0);
}
''')
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    binary=tmp_path/'actual-layout'
    result=subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',f'-DTEST_DIM={dim}','-I',str(ROOT/'include'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(binary)],check=True)
