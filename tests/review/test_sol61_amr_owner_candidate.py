"""Synthetic inventory assembly only; admission stubs are never Native receipts."""
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("amr_candidate_test",Path(__file__).with_name("sol61_amr_owner_candidate.py"))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    # This test isolates inventory plumbing; real source/CPP admissions retain their independent reader tests.
    monkeypatch.setattr(a.reader,"declared_source",lambda *args: None)
    monkeypatch.setattr(a.reader,"program_image",lambda *args: None)
    hist=a.load("inventory_historical_test","test_sol61_evolved_stage_amr_saved_reception.py")
    wire=a.load("inventory_wire_test","test_sol61_evolved_stage_amr_saved_reception_v2.py")
    count=0
    def leaf(name,data=b"SYNTHETIC_SOURCE_ONLY"):
        nonlocal count
        count+=1;p=tmp_path/(str(count)+name);p.write_bytes(data)
        return a.existing.leaf(p,tmp_path)
    origins={key:leaf(key) for key in ("native","sdk","package_manifest")}
    identities=leaf("before.json",json.dumps(origins).encode())
    after=leaf("after.json",json.dumps(origins).encode())
    receipts=[]
    suite=ET.Element("testsuite",tests="4",failures="0",errors="0",skipped="0")
    for cells,width in sorted(a.reader.CASES):
        arrays,_=wire.synthetic();arrays.update(t=np.array(.01),macro_step=np.array(1),pops_amr_checkpoint_version=np.array(12))
        # Serial synthetic envelope and shard; no real execution authority.
        arrays["n_ranks"]=np.array(1)
        raw=bytearray(arrays["state_carriers_checkpoint"].tobytes());raw[24:32]=(1).to_bytes(8,"little")
        arrays["state_carriers_checkpoint"]=np.frombuffer(raw,dtype=np.uint8).copy()
        checkpoints={}
        import io
        for phase in ("accepted","continuous","replay"):
            stream=io.BytesIO();np.savez(stream,**hist.source_only_envelope(arrays))
            checkpoints[phase]=leaf(phase+".npz",stream.getvalue())
        artifact="pops.artifact.v1:sha256:"+"a"*64
        receipt=dict(cells=cells,width=width,fixture_schema="pops.evolved-stage-amr-native-fixture@2",rank=0,size=1,dimension=2,
            newton=a.reader.CONTROLS,dt=a.reader.DT,fd_step=1e-6,acceptance=a.reader.TOL,native=origins["native"],
            carrier_registry=leaf("registry.json",json.dumps(dict(schema="sol61.amr.carrier-registry@1",dimension=2,size=1,phases={phase:{"rows_by_rank":[[]]} for phase in a.reader.PHASES})).encode()),artifact=artifact,
            phases={phase:{"levels":[leaf(phase+"-level.npz")]} for phase in a.reader.PHASES},checkpoints=checkpoints,
            compilation=[dict(component="program-X",DSO=leaf("program.so"),sidecar=leaf("sidecar"),
                              **{"ir.json":leaf("ir.json",b"{}"),"cpp":leaf("program.cpp"),"program_hash":"a"*64})])
        receipt_pin=leaf("receipt.json",json.dumps(receipt).encode());receipts.append(receipt_pin["path"])
        case=ET.SubElement(suite,"testcase",name=f"test_public_evolved_stage_amr_checkpoint_and_composite_Q[{width}-{cells}]")
        props=ET.SubElement(case,"properties")
        for key,value in dict(rank=0,size=1,dimension=2,artifact_identity=artifact,evolved_stage_amr_receipt=receipt_pin["path"]).items():
            ET.SubElement(props,"property",name=key,value=str(value))
    xml=leaf("results.xml",ET.tostring(suite))
    value=dict(roots=[str(tmp_path)],mode="serial",ir_version=16,source_commit="b"*40,native_build_source_commit="c"*40,abi_key="SOURCE_ONLY",
        identity_before=identities["path"],identity_after=after["path"],native_build_receipt=leaf("build.json")["path"],
        source_files={key:leaf(key+".py")["path"] for key in ("amr","equations","controls","fixture")},junit=[dict(rank=0,path=xml["path"])],receipts=receipts)
    return value


def test_candidate_inventory_is_pending_and_never_mints_approval(prepared):
    result=a.assemble(prepared)
    assert result["status"]=="candidate_pending_ROOT_audit" and result["scientific_reception"] is False
    assert len(result["owner_pins_candidate"]["cases"])==4
    assert "approval" not in result and "pops" not in sys.modules


@pytest.mark.parametrize("attack",("partial","changedidentity","controls","cpp","resealrank","alias","xmlfailure"))
def test_inventory_refuses_missing_or_changed_origins(prepared,attack):
    path=Path(prepared["receipts"][0]);receipt=json.loads(path.read_text())
    if attack=="partial":prepared["receipts"].pop()
    elif attack=="changedidentity":Path(prepared["identity_after"]).write_text("{}")
    elif attack=="controls":receipt["newton"]["max_iterations"]+=1
    elif attack=="cpp":del receipt["compilation"][0]["cpp"]
    elif attack=="resealrank":
        xml=Path(prepared["junit"][0]["path"]);tree=ET.fromstring(xml.read_bytes())
        tree.find("testcase/properties/property").set("value","1");xml.write_bytes(ET.tostring(tree))
    elif attack=="alias":receipt["checkpoints"]["continuous"]=receipt["checkpoints"]["accepted"]
    else:
        xml=Path(prepared["junit"][0]["path"]);tree=ET.fromstring(xml.read_bytes());ET.SubElement(tree.find("testcase"),"failure");xml.write_bytes(ET.tostring(tree))
    path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError,KeyError)):a.assemble(prepared)
