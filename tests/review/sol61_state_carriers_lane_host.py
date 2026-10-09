"""Compile the exact source transport block against the real ExecutionLane API (serial host)."""
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parents[2]
s = (root / "src/runtime/amr/amr_system.cpp").read_text()
start = s.index("  // ExecutionLane owns the authenticated runtime communicator.")
end = s.index("  std::vector<std::uint8_t> result;", start)
block = s[start:end]
source = r"""#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <cstdint>
#include <iostream>
#include <stdexcept>
using namespace pops;
std::vector<std::string> capture(const ExecutionLane& lane, const std::string& shard) {
  std::exception_ptr error;
""" + block + r"""
  return shards;
}
int main() {
  auto lane = ExecutionLane::world();
  std::string payload;
  for(unsigned i=0;i<70000;++i)payload.push_back(static_cast<char>(i%256));
  for(const auto& value : {std::string{}, std::string("a\0b",3), payload}) {
    auto shards=capture(lane,value);
    if(shards.size()!=1 || shards[0]!=value)throw std::runtime_error("serial lane changed source bytes");
  }
  std::cout << "SOURCE_ONLY exact transport/real ExecutionLane API: empty, NUL, 70000 bytes passed\n";
}
"""
with tempfile.TemporaryDirectory(prefix="pops-state-carriers-lane-host-") as directory:
    cpp=Path(directory)/"probe.cpp";exe=Path(directory)/"probe";cpp.write_text(source)
    subprocess.run(["c++","-std=c++20","-Wall","-Wextra","-Werror","-I"+str(root/"include"),str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
