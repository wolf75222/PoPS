"""Compile the emitted collective vote against the real ExecutionLane overloads."""
from pathlib import Path
import re
import shutil
import subprocess

import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize("operators", (False,True))
def test_product_layout_vote_uses_a_supported_exact_collective_type(tmp_path, operators):
    if operators:
        from tests.python.support.local_product_operator_case import make_case
    else:
        from tests.python.support.local_residual_product_case import make_case
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    case, layout, _ = make_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    declaration = re.search(r"^\s*\w+ product_layout_error_ = 0;",source,re.MULTILINE)
    vote = re.search(r"^\s*if \(pops::all_reduce_max\(product_layout_error_,[^\n]+\n[^\n]+",source,re.MULTILINE)
    assert declaration and vote
    assert "throw std::runtime_error" in vote.group()
    between = source[declaration.end():vote.start()]
    assert ".layout() !=" in between and ".distribution() !=" in between and ".local_rank() !=" in between
    # Use the emitted declaration/call unchanged, with the actual lane API. No
    # mock overload can hide an ambiguous integral conversion in a generated TU.
    cpp = tmp_path/"layout_vote.cpp"
    cpp.write_text("""#include <pops/parallel/execution_lane.hpp>
#include <stdexcept>
struct Context {
  const pops::ExecutionLane& lane;
  const pops::ExecutionLane& prepared_execution_lane() const { return lane; }
};
void layout_vote(const Context& ctx) {
"""+declaration.group()+"\n"+vote.group()+"\n}\n")
    result = subprocess.run([compiler,"-std=c++20","-DPOPS_NATIVE_DIM=2",
        "-I"+str(Path(__file__).resolve().parents[4]/"include"),
        "-fsyntax-only",str(cpp)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
