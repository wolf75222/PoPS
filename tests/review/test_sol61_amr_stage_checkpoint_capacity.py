"""Actual public Stage emission; no Native loader, JIT or catalogue substitution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.evolved_stage_amr import build

ROOT = Path(__file__).resolve().parents[2]
BASE = "0abbe25395a44e570d8d5525693b8e2dcbf4d387"
PATH = "python/pops/codegen/program_emit_amr.py"
TOKEN = r'"(?:[^"\\]|\\.)*"|-?\d+'


def old_metadata(program):
    source = subprocess.check_output(["git", "show", BASE+":"+PATH], cwd=ROOT, text=True)
    start = source.index("def _emit_checkpoint_shape_metadata(")
    end = source.index("\ndef ", start+1)
    namespace = {"json": json, "Any": object}
    exec(source[start:end], namespace)
    return namespace["_emit_checkpoint_shape_metadata"](program)


def exported_strings(cpp, symbol):
    match = re.search(r'extern "C" const char\* '+symbol+r'\(int index\) \{(.*?)\n\}', cpp, re.S)
    assert match, symbol
    return {int(i): json.loads(value) for i, value in re.findall(
        r'case (\d+): return ("(?:[^"\\]|\\.)*");', match[1])}


def registered(cpp):
    rows = {}
    for image in re.findall(r'ctx\.register_history\((.*?)\);', cpp):
        tokens = re.findall(TOKEN, image)
        args = [json.loads(token) if token.startswith('"') else int(token) for token in tokens]
        assert len(args) == 8
        row = (args[2], args[3], *args[4:])
        assert args[0] not in rows or rows[args[0]] == row
        rows[args[0]] = row
    return rows


@pytest.mark.parametrize("cells,width", ((8, 1), (16, 1), (8, 2), (16, 2)))
def test_actual_stage_ir16_registration_and_frozen_shape_match(cells, width):
    case, layout, *_ = build(cells, width)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    program = resolved.time
    before = program._serialize(), program._ir_hash()
    cpp = emit_cpp_program(program, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    current = _emit_checkpoint_shape_metadata(program)
    old = old_metadata(program)
    names = exported_strings(current, "pops_program_checkpoint_history_name")
    states = exported_strings(current, "pops_program_checkpoint_history_state_identity")
    old_states = exported_strings(old, "pops_program_checkpoint_history_state_identity")
    live = registered(cpp)
    assert names and len(names) == len(live)
    for index, name in names.items():
        assert states[index] == live[name][2]
        payload = json.loads(states[index])
        assert payload["contract"] == "pops.program.global-field-history-storage@1"
        assert payload["field_unknown"]["local_id"] == name
        assert old_states[index] == "scalar-history:"+name
        assert states[index] != old_states[index]
    # This correction changes only the DSO shape's state descriptor strings.
    old_state_export = re.search(r'extern "C" const char\* pops_program_checkpoint_history_state_identity.*?\n\}', old, re.S)[0]
    new_state_export = re.search(r'extern "C" const char\* pops_program_checkpoint_history_state_identity.*?\n\}', current, re.S)[0]
    assert old.replace(old_state_export, new_state_export) == current
    assert (program._serialize(), program._ir_hash()) == before
    # The compile driver passes the real emitted source into the artifact-spec
    # builder. Exercise its actual content-identity primitive without a Native
    # toolchain lookup; these fixed options are source-proof options only.
    from pops.identity import artifact_spec_identity, semantic_identity
    semantic = semantic_identity(program._serialize())
    def source_spec(source):
        return artifact_spec_identity(semantic, target="amr_system", backend="production",
            precision="source-proof-only", abi="source-proof-only", toolchain="source-proof-only",
            routes={}, components={"generated_source":hashlib.sha256(source.encode()).digest()})
    old_cpp = cpp.replace(new_state_export, old_state_export)
    assert source_spec(old_cpp) != source_spec(cpp)
    driver = (ROOT/"python/pops/codegen/_compile_drivers.py").read_text()
    implementation = (ROOT/"python/pops/codegen/_artifact_identity.py").read_text()
    assert "source=src," in driver
    assert '"generated_source": hashlib.sha256(source.encode("utf-8")).digest()' in implementation


def test_legacy_state_history_shape_is_byte_identical():
    case, _layout, *_ = build(8, 1)
    # Use a genuine issued State from the public fixture without adding an IR16 store.
    original = case._time_registry.program
    reference = next(value for value in original._values if value.op == "state")
    program = pops.Program("legacy-state-history")
    state = program.state(reference.state_ref)
    program.keep_history(state, depth=2)
    program.store_history("derived-state", program.value("derived", 2*state.n, at=state.n.point), depth=1)
    assert not getattr(program, "_global_field_history_issuance", {})
    assert _emit_checkpoint_shape_metadata(program) == old_metadata(program)


@pytest.mark.parametrize("mutation", ("descriptor", "owner", "boolean"))
def test_frozen_shape_rejects_mutated_issued_storage(mutation):
    case, _layout, *_ = build(8, 1)
    program = case._time_registry.program
    node = next(value for value in program._values if value.op == "store_history")
    attrs = dict(node.attrs)
    image = dict(attrs["global_field_storage"])
    if mutation == "descriptor":
        image["sampling"] = "other"
    elif mutation == "owner":
        image["owner_block"] = next(value.block for value in program._values
            if value.op == "state" and value.block is not node.block)
    else:
        image["ncomp"] = True
    attrs["global_field_storage"] = image
    program._replace_value(node, attrs=attrs)
    with pytest.raises(ValueError, match="immutable authority"):
        _emit_checkpoint_shape_metadata(program)


def test_native_capacity_refusal_is_unchanged():
    path = "src/runtime/amr/amr_system.cpp"
    original = subprocess.check_output(["git", "show", BASE+":"+path], cwd=ROOT)
    def guard(source):
        start = source.index(b"    if (live.histories != shape.histories")
        end = source.index(b'"live AMR Program checkpoint state differs from its frozen capacity metadata");', start)
        return source[start:end+len(b'"live AMR Program checkpoint state differs from its frozen capacity metadata");')]
    assert guard((ROOT/path).read_bytes()) == guard(original)


def test_actual_native_descriptor_and_capacity_guard_host(tmp_path):
    source = (ROOT/"src/runtime/amr/amr_system.cpp").read_text()
    start = source.index("    if (live.histories != shape.histories")
    end_text = '"live AMR Program checkpoint state differs from its frozen capacity metadata");'
    end = source.index(end_text, start)+len(end_text)
    guard = source[start:end]
    header = (ROOT/"include/pops/runtime/program/amr_program_checkpoint.hpp").read_text()
    start = header.index("struct AmrProgramHistoryDescriptor {")
    end = header.index("\n};", start)+len("\n};")
    descriptor = header[start:end]
    cpp = tmp_path/"actual-capacity-guard.cpp"
    cpp.write_text("""#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>
"""+descriptor+"""
struct Partition {std::string provider_identity;std::vector<int> cells;};
struct Image {std::vector<AmrProgramHistoryDescriptor> histories;Partition temporal_partition;};
struct Shape {std::vector<AmrProgramHistoryDescriptor> histories;std::vector<std::string> logical_clock_identities;std::string temporal_provider_identity;std::size_t temporal_cell_count;};
void check(const Image& live,const Shape& shape,const std::vector<std::string>& live_clocks,const std::vector<unsigned char>& live_bytes,std::size_t accepted) {
"""+guard+"""
}
int main() {
 const std::string clock="auth-clock", provider="pops.temporal-partition.global@1";
 AmrProgramHistoryDescriptor record{"T0",0,"issued-global-storage-descriptor","field-space",clock,"exact-interpolation",2,1};
 Image live{{record},{provider,{}}}; Shape shape{{record},{clock},provider,0};
 const std::vector<std::string> clocks{clock};std::vector<unsigned char> bytes(16);
 check(live,shape,clocks,bytes,16);
 int rejected=0;
 auto refuse=[&](Image bad,std::vector<std::string> names,std::vector<unsigned char> image,std::size_t budget){
  try{check(bad,shape,names,image,budget);}catch(const std::logic_error&){++rejected;return;}
  assert(false);
 };
 Image bad=live;bad.histories[0].state_identity="scalar-history:T0";refuse(bad,clocks,bytes,16);
 bad=live;bad.histories[0].program_owner=1;refuse(bad,clocks,bytes,16);
 bad=live;bad.histories[0].space_identity="owner-State-space";refuse(bad,clocks,bytes,16);
 bad=live;bad.histories[0].components=3;refuse(bad,clocks,bytes,16);
 bad=live;bad.histories[0].depth=3;refuse(bad,clocks,bytes,16);
 bad=live;bad.temporal_partition.provider_identity="foreign-provider";refuse(bad,clocks,bytes,16);
 bad=live;bad.temporal_partition.cells.push_back(1);refuse(bad,clocks,bytes,16);
 refuse(live,{"foreign-clock"},bytes,16);refuse(live,clocks,bytes,15);
 assert(rejected==9);
}
""")
    binary = tmp_path/"guard"
    subprocess.run(["clang++", "-std=c++20", str(cpp), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True)
