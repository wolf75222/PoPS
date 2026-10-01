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
