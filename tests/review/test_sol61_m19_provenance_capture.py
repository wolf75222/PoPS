"""Metadata-only genuine classes, labelled synthetic files; no Native receipt."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import pops
from pops.codegen._loader_model import CompiledModel
from pops.codegen.loader import CompiledProblem
from pops.model.manifest import build_module_manifest
from tests.python.integration.runtime.test_m19_freestreaming_runtime import _capture_components, MODULE

def handles(tmp_path):
    so=tmp_path/"synthetic-model.so";so.write_bytes(b"not-native-test-bytes")
    Path(str(so)+".pops-artifact.json").write_text("{}")
    model=CompiledModel(str(so),"production",["population"],["other"],[],1,None,1,{},
                        {"cpu":True},"metadata-only","synthetic",None,"c++20",2,
                        module_manifest=build_module_manifest(MODULE))
    program=pops.Program("metadata-only empty program")
    binary=tmp_path/"synthetic-program.so";binary.write_bytes(b"not-native-test-bytes")
    Path(str(binary)+".pops-artifact.json").write_text("{}")
    compiled=CompiledProblem(str(binary),program,None,None,None,"c++20",generated_cpp="// retained test bytes",native_dimension=2)
    artifact=SimpleNamespace(blocks=[SimpleNamespace(name="kinetic",model=model)],
                             layout_programs=[SimpleNamespace(layout_id="kinetic",program=compiled)])
    return artifact,model,compiled

def test_genuine_model_without_cpp_and_program_retained_exports(tmp_path):
    artifact,model,program=handles(tmp_path)
    assert not hasattr(model,"_generated_cpp")
    binaries,programs,sources,manifests=_capture_components(artifact,tmp_path)
    assert len(binaries)==2 and sources==[{"component":"block-kinetic","cpp":None}]
    assert Path(programs[0]["cpp"]["path"]).read_text()==program._generated_cpp
    assert json.loads(Path(programs[0]["ir"]["path"]).read_text())==program.program._serialize()
    assert programs[0]["program_hash"]==program.program_hash
    assert json.loads(Path(manifests[0]["manifest"]["path"]).read_text())==model.module_manifest.to_dict()

@pytest.mark.parametrize("attack",("program_cpp_absent","program_cpp_wrong","model_cpp_wrong","manifest_absent","sidecar_absent","binary_absent","program_hash_absent"))
def test_missing_mandatory_capture_evidence_refused(tmp_path,attack,monkeypatch):
    artifact,model,program=handles(tmp_path)
    if attack=="program_cpp_absent":program._generated_cpp=None
    if attack=="program_cpp_wrong":program._generated_cpp=42
    if attack=="model_cpp_wrong":model._generated_cpp=42
    if attack=="manifest_absent":model.module_manifest=None
    if attack=="sidecar_absent":Path(str(model.so_path)+".pops-artifact.json").unlink()
    if attack=="binary_absent":Path(model.so_path).unlink()
    if attack=="program_hash_absent":program.program_hash=None
    # A missing retained Program source must be refused before dump_cpp fallback.
    if attack.startswith("program_cpp"):
        monkeypatch.setattr(program,"dump_cpp",lambda *a:pytest.fail("forbidden regeneration/export"))
    with pytest.raises((AssertionError,FileNotFoundError)):_capture_components(artifact,tmp_path)


def test_genuine_remaining_physical_receipt_serialization():
    from tests.python.support.m19_freestreaming import FRAME, build, FINAL_TIME
    _,_,quadrature,dt=build(32,8)
    assert type(FRAME.to_dict()) is dict
    assert type(quadrature.to_data()) is dict
    assert [dt.numerator,dt.denominator]==[1,128]
    assert [FINAL_TIME.numerator,FINAL_TIME.denominator]==[1,8]
