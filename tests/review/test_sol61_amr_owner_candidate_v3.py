"""SOURCE_ONLY candidate inventories; synthetic metadata is never Native evidence."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("candidate_v3_protocol_test",Path(__file__).with_name("sol61_amr_owner_candidate_v3.py"))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
profile=a.load("candidate_v3_profile_test","test_sol61_evolved_stage_amr_saved_reception_v3.py")
old=a.load("candidate_v3_inventory_fixture","test_sol61_amr_owner_candidate.py")

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    value=old.prepared.__wrapped__(tmp_path,monkeypatch)
    # Inventory test admissions remain explicit stubs, not false source/CPP receipts.
    monkeypatch.setattr(a.reader,"declared_source",lambda *args:None)
    monkeypatch.setattr(a.reader,"program_image",lambda *args:None)
    monkeypatch.setattr(a.reader,"program_history_registry",lambda *args:{"SOURCE_ONLY":[]})
    monkeypatch.setattr(a.reader,"original_history_contract",lambda *args:None)
    value["native_abi_version"]=6
    identity=json.loads(Path(value["identity_before"]).read_text())
    abi=profile.source_abi();abi["native"]=identity["native"]
    path=tmp_path/"abi-source-only.json";path.write_text(json.dumps(abi));value["native_abi_receipt"]=str(path)
    historical=a.load("candidate_v3_historical_envelope","test_sol61_evolved_stage_amr_saved_reception.py")
    for path in value["receipts"]:
        path=Path(path);receipt=json.loads(path.read_text())
        subject_count=receipt["width"]+1
        ir_path=Path(receipt["compilation"][0]["ir.json"]["path"])
        ir={"commits":[{"state":{"kind":"state","qualified_id":subject}} for subject in sorted(profile.source_subjects(subject_count))]}
        ir_path.write_text(json.dumps(ir))
        receipt["compilation"][0]["ir.json"]["sha256"]=a.reader.digest(ir_path.read_bytes())
        for phase,row in receipt["checkpoints"].items():
            with np.load(row["path"],allow_pickle=False) as archive:
                arrays={name:archive[name].copy() for name in archive.files if name not in ("pops_checkpoint_manifest","pops_restart_identity")}
            arrays["field_provider_slots"]=np.array([],dtype=str)
            arrays["field_provider_manifest"]=np.array("[]")
            arrays["amr_accepted_contract"]=np.array(json.dumps(profile.source_contract(a.reader.STEPS[phase],subject_count)))
            np.savez(row["path"],**historical.source_only_envelope(arrays))
            row["sha256"]=a.reader.digest(Path(row["path"]).read_bytes())
        path.write_text(json.dumps(receipt))
    return value


def test_v3_candidate_is_new_profile_pending_without_approval(prepared):
    result=a.assemble(prepared);pins=result["owner_pins_candidate"]
    assert pins["schema"]=="sol61.evolved-stage-amr.owner-pins@3" and pins["native_abi_version"]==6
    assert pins["qualification"]==a.reader.QUALIFICATION
    assert result["scientific_reception"] is False and "approval" not in result
    assert a._inventory.__code__ is a.v2.assemble.__code__ and a._inventory.__globals__ is not a.v2.assemble.__globals__
    assert a._inventory.__globals__["reader"] is a.reader and a.v2.assemble.__globals__["reader"] is a.v2.reader


@pytest.mark.parametrize("attack",("schema7","tagbuffer","routewidth","abi5","missingabi"))
def test_v3_candidate_refuses_old_or_mutated_profile(prepared,attack):
    if attack=="abi5":prepared["native_abi_version"]=5
    elif attack=="missingabi":del prepared["native_abi_receipt"]
    else:
        receipt_path=Path(prepared["receipts"][0]);receipt=json.loads(receipt_path.read_text());row=receipt["checkpoints"]["accepted"]
        with np.load(row["path"],allow_pickle=False) as archive:
            arrays={name:archive[name].copy() for name in archive.files if name not in ("pops_checkpoint_manifest","pops_restart_identity")}
        contract=json.loads(str(arrays["amr_accepted_contract"].item()))
        if attack=="schema7":contract["schema_version"]=7
        elif attack=="tagbuffer":contract["tag_selection"][1][1]="1"
        else:contract["transfer_routes"][2][11]="1,1"
        arrays["amr_accepted_contract"]=np.array(json.dumps(contract))
        historical=a.load("candidate_v3_negative_envelope","test_sol61_evolved_stage_amr_saved_reception.py")
        np.savez(row["path"],**historical.source_only_envelope(arrays));row["sha256"]=a.reader.digest(Path(row["path"]).read_bytes())
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError,KeyError)):a.assemble(prepared)
