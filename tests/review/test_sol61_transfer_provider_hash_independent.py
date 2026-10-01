"""Independent small CBOR encoder and route provenance witnesses; SOURCE_ONLY."""
import hashlib
import importlib.util
from pathlib import Path
from urllib.parse import quote
import pytest

spec=importlib.util.spec_from_file_location("counterreview_profile3",Path(__file__).with_name("test_sol61_evolved_stage_amr_saved_reception_v3.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def head(major,n):
    if n<24:return bytes([major*32+n])
    for size,flag in ((1,24),(2,25),(4,26),(8,27)):
        if n < 1<<(8*size):return bytes([major*32+flag])+n.to_bytes(size,"big")
    raise ValueError("too large")

def cb(value):
    if type(value) is str:
        raw=value.encode();return head(3,len(raw))+raw
    if type(value) is int:return head(0,value)
    if type(value) is list:return head(4,len(value))+b"".join(map(cb,value))
    if type(value) is dict:
        rows=sorted(((cb(k),cb(v)) for k,v in value.items()),key=lambda row:(len(row[0]),row[0]))
        return head(5,len(rows))+b"".join(k+v for k,v in rows)
    raise TypeError(type(value))

@pytest.mark.parametrize("operation,route",[("prolongation","prolongation"),("restriction","restriction"),("coarse_fine_fill","coarse_fine"),("temporal_interpolation","temporal")])
def test_provider_canonical_cbor_independent(operation,route):
    subject="pops.handle.v1::case:independent%20case/block:opaque/model_definition:nonphysical::state::arbitrary%2Fslot"
    envelope={"protocol":"pops.identity","domain":"amr-authored-provider","schema_version":1,"payload":{"subjects":[subject],"route":route,"kind":"amr_transfer_provider"}}
    token="pops.amr-authored-provider.v1:sha256:"+hashlib.sha256(cb(envelope)).hexdigest()
    expected="pops.handle.v1::case:independent%20case::amr_transfer_provider::"+quote(route+"_"+token,safe="")
    assert m.r.transfer_provider_identity(subject,operation)==expected

@pytest.mark.parametrize("attack",["restriction","order","representation","wholegroup","injected","operationkey"])
def test_original_counterreview_attacks_now_refused(attack):
    row=m.source_contract();routes=row["transfer_routes"]
    if attack=="restriction":routes.pop(1)
    elif attack=="order":routes[0][10]="1"
    elif attack=="representation":routes[0][7]="foreign"
    elif attack=="wholegroup":del routes[:4]
    elif attack=="injected":routes[0][0]="pops.handle.v1::case:foreign/block:other::state::slot"
    else:routes[0][9]="restriction"
    with pytest.raises(ValueError):m.r.accepted_contract(row,1,m.source_subjects())

@pytest.mark.parametrize("attack",["producer","representation","sampling"])
def test_real_native_history_provenance_independent(attack):
    import json
    root=Path("/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-amr12-variants-serial-dim2/pytest-tmp/test_public_evolved_stage_amr_0/evolved-stage-amr")
    assert root.is_dir(), "this receipt requires actual retained bb416 files"
    receipt=m.r.strict_json((root/"receipt.json").read_bytes())
    ir=m.r.strict_json((root/"program-2.ir.json").read_bytes())
    arrays=m.r.wire.archive(Path(receipt["checkpoints"]["accepted"]["path"]).read_bytes())
    contract=m.r.strict_json(str(arrays["amr_accepted_contract"].item()))
    registry=m.r.program_history_registry(ir)
    m.r.original_history_contract(contract,1,registry)
    row=contract["history_qualifications"][0]
    if attack=="producer":row[1]="program.block.1"
    else:
        descriptor=m.r.strict_json(row[2]);descriptor[attack]="foreign"
        row[2]=json.dumps(descriptor,sort_keys=True,separators=(",",":"))
    with pytest.raises(ValueError,match="original history provenance"):
        m.r.original_history_contract(contract,1,registry)
