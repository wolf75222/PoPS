"""Source admission and actual C++ capability header; no Native execution."""
import ast
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
from pops.runtime._mapped_field_capability import (requires_mapped_consumed_field_output, require_mapped_consumed_field_output)
ROOT=Path(__file__).resolve().parents[2]
def artifact(*irs):
    return SimpleNamespace(layout_programs=tuple(SimpleNamespace(program=SimpleNamespace(program=SimpleNamespace(_serialize=lambda ir=ir:ir))) for ir in irs))
@pytest.mark.parametrize("facts",[None,{}, {"abi_version":8,"mapped_consumed_field_output":True},{"abi_version":9},{"abi_version":9,"mapped_consumed_field_output":1},{"abi_version":True,"mapped_consumed_field_output":True},{"abi_version":10,"mapped_consumed_field_output":True}])
def test_incompatible_native_facts_refused(facts):
    with pytest.raises(RuntimeError,match="ABI9"):
        require_mapped_consumed_field_output(artifact({"nodes":[{"op":"layout_map_export","attrs":{"contract":"mapped-consumed-output@1"}}]}), capability_reader=lambda target:facts)
def test_exact_capability_and_all_partitions():
    calls=[]
    require_mapped_consumed_field_output(artifact({"version":23},{"nodes":[{"op":"field_map_pack"}]}),capability_reader=lambda target:calls.append(target) or {"abi_version":9,"mapped_consumed_field_output":True})
    assert calls==["production"]
def test_legacy_state_maps_and_high_ir_do_not_request_capability():
    require_mapped_consumed_field_output(artifact({"version":24,"nodes":[{"op":"layout_map_export","attrs":{"contract":"state-map@1"}}]}),capability_reader=lambda target:pytest.fail("legacy capability read"))
    assert not requires_mapped_consumed_field_output({"version":100})
def test_install_and_codegen_guards_precede_side_effects():
    t=ast.parse((ROOT/'python/pops/runtime/_runtime_executor.py').read_text())
    f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='install_runtime_executor')
    text=ast.unparse(f)
    assert text.index('require_mapped_consumed_field_output(plan.artifact)')<text.index('matches =')<text.index('matches[0].install')
    text=(ROOT/'python/pops/codegen/_compile_drivers.py').read_text()
    assert text.index('require_mapped_field_native_facts()',text.index('def _compile_problem_impl'))<text.index('src = emit_program_graph',text.index('def _compile_problem_impl'))
def test_actual_capability_header_abi9(tmp_path):
    source=tmp_path/'cap.cpp';binary=tmp_path/'cap'
    source.write_text('#include <pops/runtime/module_capabilities.hpp>\n#include <cassert>\nint main(){static_assert(pops::kAbiVersion==9);auto c=pops::module_capabilities();assert(c.abi_version==9&&c.mapped_consumed_field_output);assert(pops::module_capabilities(pops::CapabilityTarget::kProduction).mapped_consumed_field_output);}')
    result=subprocess.run(['clang++','-std=c++20','-DPOPS_NATIVE_DIM=2','-I',str(ROOT/'include'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(binary)],check=True)
