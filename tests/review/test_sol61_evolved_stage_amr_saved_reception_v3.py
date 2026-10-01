"""SOURCE_ONLY schema/tag/ABI counterexamples, never Native receipts or approvals."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location("independent_amr_v3_test",Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception_v3.py"))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def source_subjects(count=2):
    return frozenset(f"pops.handle.v1::case:SOURCE_ONLY/block:generic{i}/model_definition:arbitrary::state::S{i}" for i in range(count))


def source_contract(step=1, subject_count=2):
    result={key:[] for key in r.CONTRACT_KEYS}
    result.update(schema_version=8,guarantee="bit_identical_accepted_state",program_state="compiled",
        ledger=dict(accepted_entries=0,transaction_depth=0,entries=[]),interface_ledger=dict(accepted_entries=0,transaction_depth=0,entries=[]),
        tag_selection=deepcopy(r.TAG_SELECTION),clocks=[["level",str(level),str(step),"0","1",f"{step*r.DT:.6f}"] for level in (0,1)]+[["logical","SOURCE_ONLY",str(step)]],
        field_providers=[["pops.amr.field-provider-checkpoint-manifest@1","SOURCE_ONLY-slot","2","SOURCE_ONLY-provider","SOURCE_ONLY-plan","SOURCE_ONLY-config","SOURCE_ONLY-owner","Q0","T0","0","0"]])
    kernels=(("prolongation","conservative_linear",2,"1,1"),("restriction","volume_average",1,"0,0"),
        ("coarse_fine_fill","conservative_coarse_fine",2,"2,2"),("temporal_interpolation","linear_time_interpolation",2,"0,0"))
    result["transfer_routes"]=[[subject,operation,"pops.amr-resolved-transfer.v1:sha256:"+"a"*64,r.transfer_provider_identity(subject,operation),kernel,"cell","cell","conservative","dense",operation,str(order),ghosts,"2","2,2"] for subject in sorted(source_subjects(subject_count)) for operation,kernel,order,ghosts in kernels]
    return result


@pytest.mark.parametrize("step",(1,2))
def test_source_only_profile8_tag_provenance_admission(step):
    r.accepted_contract(source_contract(step),step,source_subjects())
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
    with pytest.raises(ValueError):r.accepted_contract(row,1,source_subjects())


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
    bound=r.checkpoint_profile(source_subjects())
    assert bound is not r.v2.checkpoint
    assert bound.__code__ is r.v2.checkpoint.__code__
    assert bound.__globals__ is not r.v2.checkpoint.__globals__
    assert r.v2.checkpoint.__globals__["accepted_contract"] is r.v2.accepted_contract
    changed={key for key in bound.__globals__ if bound.__globals__[key] is not r.v2.checkpoint.__globals__[key]}
    assert changed=={"accepted_contract"}
    historical=source_contract();historical["schema_version"]=7
    r.v2.accepted_contract(historical,1)
    with pytest.raises(ValueError):r.accepted_contract(historical,1,source_subjects())
    assert r.contract()["schema"].endswith("@3") and r.v2.contract()["schema"].endswith("@2")


def test_old_owner_or_approval_never_upcast(tmp_path):
    pins=dict(schema="sol61.evolved-stage-amr.owner-pins@2")
    approval=dict(schema="sol61.evolved-stage-amr.root-approval@2",approved_by="ROOT",qualification=r.v2.QUALIFICATION,pins_sha256="a"*64)
    paths=[];digests=[]
    for name,value in (("pins",pins),("approval",approval)):
        path=tmp_path/(name+".json");raw=json.dumps(value).encode();path.write_bytes(raw);paths.append(path);digests.append(r.digest(raw))
    with pytest.raises(ValueError,match="ROOT approval scope"):
        r.receive(paths[0],digests[0],paths[1],digests[1])


@pytest.mark.parametrize("subject_count",(2,3,4))
@pytest.mark.parametrize("attack",("missingrestriction","orderdowngrade","representation","space","centering","storage","operationkey","haloprojection","halorestriction","halotemporal","providerhash","subjectgone","subjectforeign"))
def test_independent_per_subject_descriptors_and_complete_registry_refuse(subject_count,attack):
    row=source_contract(subject_count=subject_count);expected=source_subjects(subject_count)
    r.accepted_contract(row,1,expected)
    routes=row["transfer_routes"]
    if attack=="missingrestriction":del routes[1]
    elif attack=="orderdowngrade":routes[0][10]="1"
    elif attack=="representation":routes[0][7]="foreign"
    elif attack=="space":routes[0][5]="face"
    elif attack=="centering":routes[0][6]="node"
    elif attack=="storage":routes[0][8]="sparse"
    elif attack=="operationkey":routes[0][9]="restriction"
    elif attack=="haloprojection":routes[0][11]="0,0"
    elif attack=="halorestriction":routes[1][11]="1,1"
    elif attack=="halotemporal":routes[3][11]="1,1"
    elif attack=="providerhash":routes[0][3]=routes[0][3][:-1]+("0" if routes[0][3][-1]!="0" else "1")
    elif attack=="subjectgone":del routes[:4]
    else:
        for item in routes[:4]:item[0]="pops.handle.v1::case:SOURCE_ONLY/block:foreign::state::Z"
    with pytest.raises(ValueError):r.accepted_contract(row,1,expected)


def test_expected_subject_registry_is_from_ir_not_observed_routes():
    expected=source_subjects(3)
    ir={"commits":[{"state":{"kind":"state","qualified_id":subject}} for subject in sorted(expected)]}
    assert r.program_transfer_subjects(ir)==expected
    for attack in ({"commits":[]},{"commits":[{"state":{"kind":"field","qualified_id":next(iter(expected))}}]},
                   {"commits":[ir["commits"][0],ir["commits"][0]]}):
        with pytest.raises(ValueError):r.program_transfer_subjects(attack)


@pytest.fixture
def retained_native_program():
    # Read-only actual bb416 archive; never fabricate Native evidence or import PoPS.
    folder = Path("/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-amr12-variants-serial-dim2/pytest-tmp/test_public_evolved_stage_amr_0/evolved-stage-amr")
    if not folder.is_dir():
        pytest.skip("actual retained bb416 archive unavailable")
    return r.strict_json((folder/"program-2.ir.json").read_bytes()), (folder/"program-2.cpp").read_text()


def retained_hash(ir):
    projected=deepcopy(ir)
    def strip(nodes):
        for node in nodes:
            node.pop("provenance",None)
            for key in ("nodes","residual_block"):
                if key in node.get("attrs",{}):strip(node["attrs"][key])
    strip(projected["nodes"])
    return r.digest(json.dumps(projected,sort_keys=True,separators=(",",":")).encode())


def test_actual_retained_ir_canonical_integer_controls(retained_native_program):
    ir,cpp=retained_native_program
    assert r.program_image(ir,cpp,ir["version"],retained_hash(ir),1)==retained_hash(ir)
    with pytest.raises(ValueError):r.v2.program_image(ir,cpp,ir["version"],retained_hash(ir),1)
    assert "pops" not in sys.modules


@pytest.mark.parametrize("control",("linear_max_iterations","max_iterations","restart"))
@pytest.mark.parametrize("attack",("rawint","bool","delta","leadingzero","dtype","extra","floatvalue"))
def test_actual_retained_ir_integer_control_mutations(retained_native_program,control,attack):
    ir,cpp=retained_native_program
    ir=deepcopy(ir)
    controls=next(n for n in ir["nodes"] if n["op"]=="solve_spatial_field")["attrs"]["newton_controls"]
    value=r.CONTROLS[control]
    replacements={"rawint":value,"bool":True,"delta":{"scalar":{"kind":"integer","value":str(value+1)}},
        "leadingzero":{"scalar":{"kind":"integer","value":"0"+str(value)}},
        "dtype":{"scalar":{"kind":"binary64","value":float(value).hex()}},
        "extra":{"scalar":{"kind":"integer","value":str(value),"target":"int"}},
        "floatvalue":{"scalar":{"kind":"integer","value":float(value)}}}
    controls[control]=replacements[attack]
    # Rehash both images so the actual semantic-control guard, not the hash guard, refuses.
    cpp=cpp.replace(retained_hash(retained_native_program[0]),retained_hash(ir))
    with pytest.raises(ValueError,match="seven controls"):
        r.program_image(ir,cpp,ir["version"],retained_hash(ir),1)
