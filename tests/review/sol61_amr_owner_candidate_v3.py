"""Prepare @3 inventories only, no seal/approval/scientific qualification."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

v2=load("candidate_v3_historical_inventory","sol61_amr_owner_candidate.py")
reader=load("candidate_v3_strict_reader","sol61_evolved_stage_amr_saved_reception_v3.py")
_private=dict(v2.assemble.__globals__);_private["reader"]=reader
_inventory=FunctionType(v2.assemble.__code__,_private,"inventory",v2.assemble.__defaults__,v2.assemble.__closure__)


def assemble(spec):
    reader.need(type(spec.get("native_abi_version")) is int and spec["native_abi_version"] == 6,
                "candidate @3 requires explicit ROOT-attested NativeABI6")
    result=_inventory(spec)
    pins=result["owner_pins_candidate"]
    origin=Path(spec["native_abi_receipt"])
    roots=[v2.existing.root_directory(path) for path in spec["roots"]]
    matched=[root for root in roots if origin.is_relative_to(root)]
    reader.need(len(matched)==1,"ABI receipt needs one canonical explicit root")
    abi_pin=v2.existing.leaf(origin,matched[0])
    reader.native_abi_receipt(reader.strict_json(origin.read_bytes()),pins["native"])
    # Strictly receive the new accepted schema/tag provenance before naming this @3.
    for case in pins["cases"]:
        receipt=reader.strict_json(Path(case["receipt"]["path"]).read_bytes())
        programs = [row for row in receipt["compilation"] if row["component"].startswith("program-")]
        reader.need(len(programs)==1,"one retained Program registry required")
        ir_pin = programs[0]["ir.json"]
        raw_ir = Path(ir_pin["path"]).read_bytes()
        reader.need(reader.digest(raw_ir)==case["files"][ir_pin["path"]],"Program registry changed during candidate validation")
        ir=reader.strict_json(raw_ir)
        subjects = reader.program_transfer_subjects(ir)
        histories = reader.program_history_registry(ir)
        for phase in ("accepted","continuous","replay"):
            row=receipt["checkpoints"][phase]
            raw=Path(row["path"]).read_bytes()
            reader.need(reader.digest(raw)==case["files"][row["path"]],"checkpoint changed during candidate profile validation")
            arrays=reader.wire.archive(raw)
            contract=reader.strict_json(str(arrays["amr_accepted_contract"].item()))
            reader.accepted_contract(contract,reader.STEPS[phase],subjects)
            reader.original_history_contract(contract,reader.STEPS[phase],histories)
            reader.empty_component_registry(arrays)
    pins.update(schema="sol61.evolved-stage-amr.owner-pins@3",qualification=reader.QUALIFICATION,native_abi_version=6,native_abi_receipt=abi_pin)
    result.update(profile="NativeABI6-accepted8-TagSelection1@3")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("spec");parser.add_argument("candidate")
    args=parser.parse_args()
    result=assemble(reader.strict_json(Path(args.spec).read_bytes()))
    with Path(args.candidate).open("x") as output:output.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
