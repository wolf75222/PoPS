"""SOURCE_ONLY schema/tag/ABI counterexamples, never Native receipts or approvals."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location("independent_amr_v3_test",Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception_v3.py"))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def source_contract(step=1):
    result={key:[] for key in r.CONTRACT_KEYS}
    result.update(schema_version=8,guarantee="bit_identical_accepted_state",program_state="compiled",
        ledger=dict(accepted_entries=0,transaction_depth=0,entries=[]),interface_ledger=dict(accepted_entries=0,transaction_depth=0,entries=[]),
        tag_selection=deepcopy(r.TAG_SELECTION),clocks=[["level",str(level),str(step),"0","1",f"{step*r.DT:.6f}"] for level in (0,1)]+[["logical","SOURCE_ONLY",str(step)]],
        field_providers=[["pops.amr.field-provider-checkpoint-manifest@1","SOURCE_ONLY-slot","2","SOURCE_ONLY-provider","SOURCE_ONLY-plan","SOURCE_ONLY-config","SOURCE_ONLY-owner","Q0","T0","0","0"]])
    kernels=(("prolongation","conservative_linear",2,"1,1"),("restriction","volume_average",1,"0,0"),
        ("coarse_fine_fill","conservative_coarse_fine",2,"2,2"),("temporal_interpolation","linear_time_interpolation",2,"0,0"))
    result["transfer_routes"]=[["SOURCE_ONLY-subject",operation,"SOURCE_ONLY-route","SOURCE_ONLY-provider",kernel,"state","cell","conservative","dense",operation,str(order),ghosts,"2","2,2"] for operation,kernel,order,ghosts in kernels]
    return result


@pytest.mark.parametrize("step",(1,2))
def test_source_only_profile8_tag_provenance_admission(step):
    r.accepted_contract(source_contract(step),step)
    assert "pops" not in sys.modules


@pytest.mark.parametrize("attack",("old7","missingtag","tagversion","tagbuffer","rank","ratio","nesting","lookahead","transition","typed","missingroutes","routeidentity","routekernel","routewidth","routeorder","routeaxis","routeduplicate","missingfields","fielddepth","fieldidentity","clock","provisional","extra"))
def test_source_only_profile8_refuses_old_or_mutated_protocol(attack):
    row=source_contract()
    if attack=="old7":row["schema_version"]=7
    elif attack=="missingtag":del row["tag_selection"]
    elif attack=="tagversion":row["tag_selection"][0][1]="2"
    elif attack=="tagbuffer":row["tag_selection"][1][1]="1"
    elif attack=="rank":row["tag_selection"][1].pop()
    elif attack=="ratio":row["tag_selection"][2][2]="3"
    elif attack=="nesting":row["tag_selection"][2][3]="0"
    elif attack=="lookahead":row["tag_selection"][2][4]="0"
    elif attack=="transition":row["tag_selection"][2][1]="1"
    elif attack=="typed":row["tag_selection"][0][1]=1
    elif attack=="missingroutes":row["transfer_routes"]=[]
    elif attack=="routeidentity":row["transfer_routes"][0][3]=""
    elif attack=="routekernel":row["transfer_routes"][0][4]="foreign_kernel"
    elif attack=="routewidth":row["transfer_routes"][2][11]="1,1"
    elif attack=="routeorder":row["transfer_routes"][2][10]="5"
    elif attack=="routeaxis":row["transfer_routes"][0][12]="3"
    elif attack=="routeduplicate":row["transfer_routes"].append(deepcopy(row["transfer_routes"][0]))
    elif attack=="missingfields":row["field_providers"]=[]
    elif attack=="fielddepth":row["field_providers"][0][2]="1"
    elif attack=="fieldidentity":row["field_providers"][0][3]=""
    elif attack=="clock":row["clocks"][0][2]="0"
    elif attack=="provisional":row["ledger"]["transaction_depth"]=1
    else:row["invented"]=True
    with pytest.raises(ValueError):r.accepted_contract(row,1)


def source_abi():
    return dict(schema="root.api040.native-abi@1",native=dict(path="/SOURCE_ONLY/nonexistent.so",sha256="a"*64),
        header_signature="SOURCE_ONLY-not-native",module_abi_version=6,capability_abi_version=6,release_native_abi_version=6)


@pytest.mark.parametrize("attack",("module5","capability5","release5","typed","dso","missing","extra"))
def test_source_only_abi6_origin_protocol_refuses(attack):
    row=source_abi();native=deepcopy(row["native"])
    r.native_abi_receipt(row,native)
    if attack=="module5":row["module_abi_version"]=5
    elif attack=="capability5":row["capability_abi_version"]=5
    elif attack=="release5":row["release_native_abi_version"]=5
    elif attack=="typed":row["module_abi_version"]=6.
    elif attack=="dso":row["native"]["sha256"]="b"*64
    elif attack=="missing":del row["header_signature"]
    else:row["extra"]=True
    with pytest.raises(ValueError):r.native_abi_receipt(row,native)


def test_checkpoint_private_profile_is_bijective_and_historical_functions_unchanged():
    assert r.checkpoint is not r.v2.checkpoint
    assert r.checkpoint.__code__ is r.v2.checkpoint.__code__
    assert r.checkpoint.__globals__ is not r.v2.checkpoint.__globals__
    assert r.checkpoint.__globals__["accepted_contract"] is r.accepted_contract
    assert r.v2.checkpoint.__globals__["accepted_contract"] is r.v2.accepted_contract
    changed={key for key in r.checkpoint.__globals__ if r.checkpoint.__globals__[key] is not r.v2.checkpoint.__globals__[key]}
    assert changed=={"accepted_contract"}
    historical=source_contract();historical["schema_version"]=7
    r.v2.accepted_contract(historical,1)
    with pytest.raises(ValueError):r.accepted_contract(historical,1)
    assert r.contract()["schema"].endswith("@3") and r.v2.contract()["schema"].endswith("@2")


def test_old_owner_or_approval_never_upcast(tmp_path):
    pins=dict(schema="sol61.evolved-stage-amr.owner-pins@2")
    approval=dict(schema="sol61.evolved-stage-amr.root-approval@2",approved_by="ROOT",qualification=r.v2.QUALIFICATION,pins_sha256="a"*64)
    paths=[];digests=[]
    for name,value in (("pins",pins),("approval",approval)):
        path=tmp_path/(name+".json");raw=json.dumps(value).encode();path.write_bytes(raw);paths.append(path);digests.append(r.digest(raw))
    with pytest.raises(ValueError,match="ROOT approval scope"):
        r.receive(paths[0],digests[0],paths[1],digests[1])
