"""Reseal explicit negative copies of future authentic M18 receipts; never owner pins."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

_spec=importlib.util.spec_from_file_location("m18_counter_oracle",Path(__file__).with_name("sol61_m18_entropy_offline_oracle.py"))
oracle=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=oracle
_spec.loader.exec_module(oracle)

LABELS=("wrong_accepted_multiplier","readonly_target_changed","transposed_axes","overflow_population",
        "outside_replaced_by_feasible_boundary","rollback_buffer_changed","clock_one_ulp","bool_clock",
        "missing_owned_cell","future_target_capture","relaxed_original_tolerance","negative_quadrature_weight")


def leaves(value):
    if type(value) is dict:
        if set(value)=={"path","sha256"}:
            return [value]
        return [leaf for item in value.values() for leaf in leaves(item)]
    if type(value) is list:
        return [leaf for item in value for leaf in leaves(item)]
    return []


def json_file(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def reseal_checkpoint(path,image):
    manifest=oracle.strict_json(str(image["pops_checkpoint_manifest"].item()))
    manifest["arrays"]={name:oracle.typed_array(array) for name,array in image.items()
                        if name not in {"pops_checkpoint_manifest","pops_restart_identity"}}
    manifest["restart_identity"]["hexdigest"]=oracle.digest(oracle.cbor(dict(
        protocol="pops.identity",domain="restart",schema_version=1,
        payload={key:value for key,value in manifest.items() if key!="restart_identity"})))
    image["pops_checkpoint_manifest"]=np.asarray(json.dumps(manifest,sort_keys=True,separators=(",",":")))
    image["pops_restart_identity"]=np.asarray(oracle.protocol.identity_token(manifest["restart_identity"],"restart"))
    np.savez_compressed(path,**image)


def mutate_phase(pins,phase,operation):
    rows=pins["phases"][phase]
    paths={name:Path(row["path"]) for name,row in rows.items()}
    state=oracle.protocol.archive(paths["state"].read_bytes())
    receipt=oracle.strict_json(paths["receipt"].read_bytes())
    checkpoint=oracle.protocol.archive(paths["checkpoint"].read_bytes())
    operation(state,receipt,checkpoint)
    for block in ("dual","target"):
        checkpoint["state_"+block]=state[block].reshape(checkpoint["state_"+block].shape).copy()
    reseal_checkpoint(paths["checkpoint"],checkpoint)
    np.savez_compressed(paths["state"],**state)
    receipt["checkpoint_sha256"]=oracle.digest(paths["checkpoint"].read_bytes())
    json_file(paths["receipt"],receipt)


def control(label,pins,root):
    if label in {"wrong_accepted_multiplier","readonly_target_changed","transposed_axes","overflow_population"}:
        def operation(state,receipt,checkpoint):
            if label=="wrong_accepted_multiplier":
                state["dual"][0,1,2]+=1.e-4
            elif label=="readonly_target_changed":
                state["target"][2,1,2]+=1.e-4
            elif label=="transposed_axes":
                state["dual"]=state["dual"].transpose(0,2,1).copy()
            else:
                state["dual"][0,1,2]=1000.
        mutate_phase(pins,"accepted",operation)
    elif label=="outside_replaced_by_feasible_boundary":
        for phase in ("outside_before","outside_rejected_0","outside_rejected_1"):
            mutate_phase(pins,phase,lambda state,receipt,checkpoint: state["target"].__setitem__((slice(None),1,2),(1.,1.,1.)))
    elif label=="rollback_buffer_changed":
        mutate_phase(pins,"outside_rejected_1",lambda state,receipt,checkpoint: state["dual"].__setitem__((0,0,0),1.e-4))
    elif label in {"clock_one_ulp","bool_clock"}:
        def operation(state,receipt,checkpoint):
            temporal=copy.deepcopy(receipt["temporal"])
            manifest=oracle.strict_json(str(checkpoint["pops_checkpoint_manifest"].item()))
            if label=="clock_one_ulp":
                time=float(np.nextafter(receipt["time"],np.inf))
                receipt["time"],receipt["time_hex"]=time,time.hex()
                checkpoint["t"]=np.asarray(time)
                temporal["clock"]["time"]=time.hex()
            else:
                temporal["clock"]["macro_step"]=True
            manifest["clock"]=copy.deepcopy(temporal["clock"])
            receipt["temporal"]=temporal
            checkpoint["temporal_restart_state"]=np.asarray(json.dumps(temporal))
            checkpoint["pops_checkpoint_manifest"]=np.asarray(json.dumps(manifest))
        mutate_phase(pins,"accepted",operation)
    elif label=="missing_owned_cell":
        def operation(state,receipt,checkpoint):
            for rank in receipt["local_boxes_by_rank"]:
                if rank["dual"]:
                    rank["dual"].pop()
                    break
        mutate_phase(pins,"accepted",operation)
    elif label in {"future_target_capture","relaxed_original_tolerance"}:
        path=Path(pins["sources"]["example"]["path"])
        source=path.read_text()
        changed=(source.replace('captures={"target": data_state.n}','captures={"target": data_state.next}')
                 if label=="future_target_capture" else source.replace('tolerance=2.e-11','tolerance=2.e-8'))
        oracle.require(changed!=source,"authentic source seam missing for negative injection")
        path.write_text(changed)
        provenance_path=Path(pins["provenance"]["path"])
        provenance=oracle.strict_json(provenance_path.read_bytes())
        provenance["source_sha256"]["example"]=oracle.digest(path.read_bytes())
        json_file(provenance_path,provenance)
    elif label=="negative_quadrature_weight":
        path=Path(pins["provenance"]["path"])
        provenance=oracle.strict_json(path.read_bytes())
        provenance["constants"]["weights"][0]=-.1
        json_file(path,provenance)
    else:
        raise ValueError("unknown explicit countermodel")
    for row in leaves(pins):
        row["sha256"]=oracle.digest(Path(row["path"]).read_bytes())
    path=root/"NOT-OWNER-countermodel-pins.json"
    json_file(path,pins)
    seal=oracle.digest(path.read_bytes())  # Explicit negative harness seal, never root's positive authority.
    for row in leaves(pins):
        oracle.pinned(path.parent,row)
    try:
        oracle.receive(path,seal)
    except (ValueError,AssertionError) as error:
        return dict(countermodel=label,refusal=str(error),all_external_file_pins_verified=True,
                    harness_sha256=seal,NOT_OWNER_AUTHORITY=True)
    raise AssertionError("scientific/protocol negative accepted: "+label)


def receive(path,seal,workspace):
    positive=oracle.receive(path,seal)
    oracle.require(not workspace.exists() or not any(workspace.iterdir()),"countermodel workspace must be empty")
    workspace.mkdir(parents=True,exist_ok=True)
    original=oracle.strict_json(path.read_bytes())
    results=[]
    for label in LABELS:
        root=workspace/label
        root.mkdir()
        pins=copy.deepcopy(original)
        for index,row in enumerate(leaves(pins)):
            source,raw=oracle.pinned(path.parent,row)
            directory=root/("file-%02d"%index)
            directory.mkdir()
            target=directory/source.name
            target.write_bytes(raw)
            row["path"]=str(target.absolute())
        results.append(control(label,pins,root))
    oracle.require(oracle.receive(path,seal)==positive,"authentic original files changed")
    return dict(status="received",authentic_positive=positive,countermodels=results,original_reverified_unchanged=True,
                native_execution_in_harness=False,countermodel_authority="explicit negative reseals, NOT owner/native receipts")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins",type=Path,required=True)
    parser.add_argument("--owner-sha256",required=True)
    parser.add_argument("--countermodels-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=receive(args.pins,args.owner_sha256,args.countermodels_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    json_file(args.output,result)
    print(json.dumps(dict(status=result["status"],refusals=len(result["countermodels"]))))


if __name__=="__main__":
    main()
