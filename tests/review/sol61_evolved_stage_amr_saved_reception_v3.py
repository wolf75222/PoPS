"""@3 NativeABI6/schema8 TagSelection receipt; historical @1/@2 immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET
from types import FunctionType
from urllib.parse import quote
import numpy as np

_spec = importlib.util.spec_from_file_location("amr_v3_historical_v2",Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v2
_spec.loader.exec_module(v2)
# Reuse immutable math/wire functions. No object in the historical module is mutated.
need, digest, strict_json = v2.need, v2.digest, v2.strict_json
wire, carriers = v2.wire, v2.carriers
envelope = v2.envelope
DT, TOL, CONTROLS, PHASES, STEPS, CASES = v2.DT, v2.TOL, v2.CONTROLS, v2.PHASES, v2.STEPS, v2.CASES
exact, typed, same = v2.exact, v2.typed, v2.same
science, declared_source, run_identity = v2.science, v2.declared_source, v2.run_identity
QUALIFICATION = "homogeneous-original-composite-Q-tag-selection@3"
TAG_SELECTION = [["pops.amr.tag-selection@1","1"],["tag-buffer","0","0"],
                 ["parent-coverage","0","2","2","1","2","2","1"]]
CONTRACT_KEYS = {"schema_version","guarantee","program_state","ledger","interface_ledger","clocks",
    "temporal_partition","synchronization","history_qualifications","level_relations","transfer_routes","field_providers","tag_selection"}


def program_image(ir, cpp, expected_version, program_hash, width):
    need(type(ir) is dict and type(ir.get("version")) is int and ir["version"] == expected_version
         and expected_version in (16, 17), "qualified Program IR version differs")
    projection = deepcopy(ir)
    def nodes(values):
        for node in values:
            node.pop("provenance", None)
            for key in ("nodes", "residual_block"):
                if key in node.get("attrs", {}):
                    nodes(node["attrs"][key])
    nodes(projection["nodes"])
    actual = digest(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode())
    need(actual == program_hash, "actual saved IR hash differs")
    matches = re.findall(r'pops_program_hash\(\)\s*\{\s*return\s*"([0-9a-f]{64})";', cpp)
    need(matches == [actual], "saved CPP Program hash differs from saved IR")
    text = json.dumps(ir)
    need("pops.program.global-field-history-storage@1" in text and "field_evolved_state" in text
         and "pops.amr.full-residual-basis-lu@1" in text and "store_global_field_history(" in cpp,
         "original Stage/history/realization qualification missing")
    solves = [node for node in ir["nodes"] if node["op"] == "solve_spatial_field"]
    need(len(solves) == 1,"original full-product spatial solve missing/duplicated")
    attrs = solves[0]["attrs"]
    expected_controls = {key:dict(scalar=dict(kind="integer",value=str(value))) if type(value) is int else dict(kind="binary64",value=value.hex()) for key,value in CONTROLS.items()}
    need(typed(attrs["newton_controls"],expected_controls) and attrs["finite_difference_step"] == dict(kind="binary64",value=(1e-6).hex())
         and type(attrs["ncomp"]) is int and attrs["ncomp"] == width+(width == 2)
         and attrs["right_preconditioner"] == "pops.amr.full-residual-basis-lu@1"
         and typed(attrs["right_preconditioner_resources"],dict(version=1,max_dense_bytes=dict(uint64_hex="0000000010000000"),
             scope="per_rank_dense_arrays_active_map_and_numeric_towers"))
         and attrs["source_contract"].get("temporal_tau") is not None and attrs["source_contract"].get("evolved_stage") is not None,
         "actual IR original seven controls/duration/product/realization differ")
    return actual


TRANSFER_DESCRIPTIONS = {
    "prolongation": ("prolongation", "conservative_linear", "2", "1,1"),
    "restriction": ("restriction", "volume_average", "1", "0,0"),
    "coarse_fine_fill": ("coarse_fine", "conservative_coarse_fine", "2", "2,2"),
    "temporal_interpolation": ("temporal", "linear_time_interpolation", "2", "0,0"),
}


def program_transfer_subjects(ir):
    """Expected registry comes from authenticated committed state handles, not routes."""
    commits = ir.get("commits")
    need(type(commits) is list and bool(commits), "Program committed-state provenance missing")
    subjects = []
    for commit in commits:
        state = commit.get("state")
        need(type(state) is dict and state.get("kind") == "state"
             and type(state.get("qualified_id")) is str, "Program committed state identity missing")
        subject = state["qualified_id"]
        need(subject.startswith("pops.handle.v1::case:") and "/block:" in subject
             and "::state::" in subject and subject not in subjects, "Program transfer subject registry differs")
        subjects.append(subject)
    return frozenset(subjects)


def transfer_provider_identity(subject, operation):
    route = TRANSFER_DESCRIPTIONS[operation][0]
    payload = dict(subjects=[subject], route=route, kind="amr_transfer_provider")
    token = "pops.amr-authored-provider.v1:sha256:" + digest(wire.cbor(
        dict(protocol="pops.identity", domain="amr-authored-provider", schema_version=1, payload=payload)))
    case_owner = subject.split("/block:", 1)[0]
    return case_owner + "::amr_transfer_provider::" + quote(route+"_"+token, safe="")


def accepted_contract(contract, step, transfer_subjects):
    exact(contract,CONTRACT_KEYS,"NativeABI6 accepted contract")
    need(type(contract["schema_version"]) is int and contract["schema_version"] == 8
         and contract["guarantee"] == "bit_identical_accepted_state" and contract["program_state"] == "compiled",
         "NativeABI6 requires accepted schema8")
    need(typed(contract["tag_selection"],TAG_SELECTION),"authored Buffer0 TagSelection@1/ranked parent coverage differs")
    providers = contract["field_providers"]
    need(type(providers) is list and bool(providers),"original Q/T/aux field provider provenance missing")
    for row in providers:
        need(type(row) is list and len(row) >= 11 and all(type(v) is str for v in row)
             and row[0] == "pops.amr.field-provider-checkpoint-manifest@1" and row[2] == "2"
             and all(row[index] for index in (1,3,4,5,6,7,8)), "original field provider identities/depth differ")
    need(type(transfer_subjects) is frozenset and bool(transfer_subjects), "authenticated Program transfer registry required")
    expected = {(subject, operation) for subject in transfer_subjects for operation in TRANSFER_DESCRIPTIONS}
    routes = contract["transfer_routes"]
    need(type(routes) is list and len(routes) == len(expected), "complete per-subject interpolation coverage differs")
    seen = set()
    for row in routes:
        need(type(row) is list and len(row) == 14 and all(type(v) is str for v in row), "typed interpolation row differs")
        key = (row[0],row[1])
        need(key in expected and key not in seen, "foreign or duplicate interpolation subject/operation")
        seen.add(key)
        operation = row[1]
        _, kernel, order, halo = TRANSFER_DESCRIPTIONS[operation]
        need(row[4:] == [kernel,"cell","cell","conservative","dense",operation,order,halo,"2","2,2"],
             "exact authored interpolation descriptor differs")
        need(re.fullmatch(r"pops\.amr-resolved-transfer\.v1:sha256:[0-9a-f]{64}", row[2]) is not None,
             "native resolved route identity differs")
        need(row[3] == transfer_provider_identity(row[0],operation), "authored subject/provider provenance differs")
    need(seen == expected, "complete per-subject interpolation coverage differs")
    for name in ("ledger","interface_ledger"):
        row = contract[name]
        need(type(row["accepted_entries"]) is int and row["accepted_entries"] == len(row["entries"])
             and type(row["transaction_depth"]) is int and row["transaction_depth"] == 0,"native AMR ledger is provisional/inconsistent")
    level_rows = [row for row in contract["clocks"] if row[0] == "level"]
    expected = [["level",str(level),str(step),"0","1",f"{step*DT:.6f}"] for level in (0,1)]
    need(level_rows == expected,"native AMR rational level clocks differ from accepted macro barrier")
    logical = [row for row in contract["clocks"] if row[0] == "logical"]
    need(len(logical) == 1 and len(logical[0]) == 3 and logical[0][1] and logical[0][2] == str(step)
         and len(contract["clocks"]) == 3,"native AMR logical clock publication differs")


# Bind each checkpoint to the independent committed-state registry from its retained IR.
# This creates a private function object; no historical globals are modified.
def checkpoint_profile(transfer_subjects):
    private = dict(v2.checkpoint.__globals__)
    def accepted(contract, step):
        return accepted_contract(contract, step, transfer_subjects)
    private["accepted_contract"] = accepted
    return FunctionType(v2.checkpoint.__code__,private,"checkpoint",v2.checkpoint.__defaults__,v2.checkpoint.__closure__)


def checkpoint(arrays, phase, images, n, width, ranks, identities, abi, *, transfer_subjects):
    return checkpoint_profile(transfer_subjects)(arrays,phase,images,n,width,ranks,identities,abi)


def native_abi_receipt(receipt, native):
    exact(receipt,("schema","native","header_signature","module_abi_version","capability_abi_version","release_native_abi_version"),"ROOT NativeABI receipt")
    need(receipt["schema"] == "root.api040.native-abi@1" and typed(receipt["native"],native)
         and type(receipt["header_signature"]) is str and bool(receipt["header_signature"]),"NativeABI DSO/header origin differs")
    for name in ("module_abi_version","capability_abi_version","release_native_abi_version"):
        need(type(receipt[name]) is int and receipt[name] == 6,"NativeABI6 actual module/capability/release provenance differs")
    return receipt


def contract():
    result = v2.contract()
    result.update(schema="sol61.evolved-stage-amr.owner-pins@3",qualification=QUALIFICATION)
    result["native_abi_version"] = 6
    result["native_abi_receipt"] = "actual ROOT runtime ABI receipt path+sha256"
    result["approval"] = dict(schema="sol61.evolved-stage-amr.root-approval@3",approved_by="ROOT",qualification=QUALIFICATION,pins_sha256="external pins SHA256")
    return result


def receive(pins_path, pins_sha, approval_path, approval_sha):
    """No assembler/seal producer. ROOT supplies actual identities and exact inventory."""
    pins_raw, approval_raw = Path(pins_path).read_bytes(), Path(approval_path).read_bytes()
    need(digest(pins_raw) == pins_sha and digest(approval_raw) == approval_sha, "external ROOT seals differ")
    pins, approval = strict_json(pins_raw), strict_json(approval_raw)
    need(approval == dict(schema="sol61.evolved-stage-amr.root-approval@3", approved_by="ROOT",
                         qualification=QUALIFICATION, pins_sha256=pins_sha), "ROOT approval scope differs")
    exact(pins, ("schema", "qualification", "native_evidence", "mode", "ranks", "source_commit", "native_build_source_commit",
                 "abi_key", "native_abi_version", "native_abi_receipt", "ir_version", "roots", "native", "sdk", "package_manifest", "source_files", "junit", "batch_names", "cases"), "owner pins")
    need(pins["schema"] == "sol61.evolved-stage-amr.owner-pins@3" and pins["qualification"] == QUALIFICATION
         and pins["native_evidence"] is True and pins["mode"] in ("serial", "mpi2")
         and type(pins["ranks"]) is int and pins["ranks"] == (1 if pins["mode"] == "serial" else 2), "native owner qualification differs")
    for name in ("source_commit", "native_build_source_commit"):
        need(re.fullmatch("[0-9a-f]{40}", pins[name]) is not None, "source commit pin differs")
    roots = [Path(path).resolve() for path in pins["roots"]]
    def read(row):
        exact(row, ("path", "sha256"), "file pin")
        path = Path(row["path"]).resolve()
        need(any(path.is_relative_to(root) for root in roots), "file outside owner roots")
        need(path.is_file() and path.stat().st_size <= 1024**3, "missing or oversized offline pinned file")
        raw = path.read_bytes()
        need(digest(raw) == row["sha256"], "external file SHA differs: "+str(path))
        return raw
    for key in ("native", "sdk", "package_manifest"):
        read(pins[key])
    need(type(pins["native_abi_version"]) is int and pins["native_abi_version"] == 6,
         "reader @3 requires explicit ROOT-attested NativeABI6 origin")
    native_abi_receipt(strict_json(read(pins["native_abi_receipt"])),pins["native"])
    exact(pins["source_files"], ("amr", "equations", "controls", "fixture"), "source pins")
    source = {key:read(row) for key,row in pins["source_files"].items()}
    declared_source(source["amr"], source["equations"], source["controls"])
    need(len(pins["cases"]) == 4 and {(row["cells"], row["width"]) for row in pins["cases"]} == CASES, "case inventory differs")
    results = []
    for case in pins["cases"]:
        exact(case, ("cells", "width", "receipt", "files", "artifact", "bind", "semantic"), "case pin")
        n, width = case["cells"], case["width"]
        receipt = strict_json(read(case["receipt"]))
        # @1 had reversed physical history slot labels and is not upcast.
        need(receipt["fixture_schema"] == "pops.evolved-stage-amr-native-fixture@2", "requires corrected physical-slot archive @2; @1 is historical")
        need(typed(receipt["history_protocol"],dict(wire="POPSHID1",raw_slots_after_publication=True,
             latest_slot=1,previous_slot=0,depth=2)),"physical history slot policy differs")
        need((receipt["cells"], receipt["width"], receipt["dimension"], receipt["rank"], receipt["size"]) == (n,width,2,0,pins["ranks"])
             and receipt["artifact"] == case["artifact"] and typed(receipt["newton"],CONTROLS)
             and receipt["dt"] == DT and receipt["fd_step"] == 1e-6 and receipt["acceptance"] == TOL
             and receipt["realization"] == "FullResidualBasisLU@1" and receipt["max_dense_bytes"] == 256*1024**2,
             "receipt original controls/provenance differ")
        seen = set()
        def row_file(row, case=case, seen=seen):
            need(case["files"].get(row["path"]) == row["sha256"], "receipt file absent external inventory")
            seen.add(row["path"])
            return read(row)
        registry = strict_json(row_file(receipt["carrier_registry"]))
        exact(registry, ("schema", "dimension", "size", "phases"), "carrier registry")
        need(registry["schema"] == "sol61.amr.carrier-registry@1" and registry["dimension"] == 2
             and registry["size"] == pins["ranks"] and set(registry["phases"]) == set(PHASES), "carrier registry authority differs")
        for phase in PHASES:
            exact(registry["phases"][phase], ("rows_by_rank",), "carrier phase")
            rows = registry["phases"][phase]["rows_by_rank"]
            need(type(rows) is list and len(rows) == pins["ranks"]
                 and all(type(rank) is list and all(type(row) is list and all(type(v) is str for v in row) for row in rank) for rank in rows), "carrier rank row types differ")
            need(typed(rows[0], receipt["phases"][phase]["metadata"][1]), "carrier rank-zero observed registry differs")
        for a,b in (("accepted","reloaded"),("continuous","replay")):
            need(typed(registry["phases"][a], registry["phases"][b]), "full state/history/auxiliary carrier restart hashes differ")
        row_file(receipt["native"])
        need(receipt["native"]["sha256"] == pins["native"]["sha256"], "receipt native DSO differs")
        images = {phase:[wire.archive(row_file(row)) for row in receipt["phases"][phase]["levels"]] for phase in PHASES}
        arrays = {phase:wire.archive(row_file(receipt["checkpoints"][phase])) for phase in ("accepted", "continuous", "replay")}
        need(len({receipt["checkpoints"][p]["path"] for p in arrays}) == 3, "checkpoint overwrite aliases")
        program_hashes, transfer_subjects = [], None
        for component in receipt["compilation"]:
            for name in ("DSO", "sidecar"):
                row_file(component[name])
            if component["component"].startswith("program-"):
                program_ir = strict_json(row_file(component["ir.json"]))
                transfer_subjects = program_transfer_subjects(program_ir)
                program_hashes.append(program_image(program_ir,
                    row_file(component["cpp"]).decode(), pins["ir_version"], component["program_hash"],width))
        need(len(program_hashes) == 1, "this same-layout witness requires one actual Program IR")
        manifests, masks, diagnostics = {}, None, {}
        for phase in arrays:
            need(str(arrays[phase]["program_hash"].item()) == program_hashes[0], "checkpoint installed Program differs")
            manifests[phase], actual_masks, diagnostics[phase] = checkpoint(arrays[phase], phase, images, n,width,pins["ranks"],
                (case["artifact"],case["bind"],case["semantic"]),pins["abi_key"],transfer_subjects=transfer_subjects)
            need("state_carriers_checkpoint" in arrays[phase], "CP12 full carrier image missing")
            archive = carriers.receive_carriers(arrays[phase], registry["phases"][phase]["rows_by_rank"],
                {**{f"Q{i}":1 for i in range(width)}, "forcing":width+1})
            if phase == "accepted":
                carriers.receive_carriers(arrays[phase], registry["phases"]["reloaded"]["rows_by_rank"],
                    {**{f"Q{i}":1 for i in range(width)}, "forcing":width+1})
            if masks is not None:
                for a,b in zip(masks,actual_masks,strict=True):
                    same(a,b,"stationary topology across continuation")
            masks = actual_masks
        for phase in PHASES:
            meta = receipt["phases"][phase]["metadata"]
            need(type(meta) is list and len(meta) == 4,"observed native metadata inventory differs")
            need(typed(meta[3][:2],[STEPS[phase]*DT,STEPS[phase]]),"observed native lifecycle differs")
            if phase == "initial":
                need(meta[0] == [],"initial fixture invents published histories")
                continue
            durable_phase = "accepted" if phase == "reloaded" else phase
            native_boxes = [[int(row[0]),[int(row[1]),int(row[2])],[int(row[3]),int(row[4])]]
                            for row in arrays[durable_phase]["patch_boxes"]]
            need(meta[3][2] == native_boxes,"observed/durable native patch layout differs")
            names = [f"T{i}" for i in range(width)]+(["z"] if width == 2 else [])
            expected_history = [[level,name,2,True,STEPS[phase],[DT.hex(),DT.hex()],
                images[phase][level]["history_sample_identity_"+name].tobytes().hex()]
                for level in (0,1) for name in names]
            need(typed(meta[0],expected_history),"observed history metadata/NPZ point differs")
            observed_diagnostics = {name.encode():struct.pack("<d",value) for name,value in meta[2]}
            need(len(observed_diagnostics) == len(meta[2]) and observed_diagnostics == diagnostics[durable_phase][0],
                 "rank-zero observed/durable diagnostic bits differ")
        need(set(case["files"]) == seen, "surplus or missing externally pinned case files")
        science_result = science(images,masks,n,width)
        for key in arrays["continuous"]:
            if key not in ("pops_checkpoint_manifest", "pops_restart_identity"):
                same(arrays["continuous"][key],arrays["replay"][key],"durable continuation "+key)
        equivalence = receipt["checkpoint_equivalence"]
        need(equivalence["contract"] == "pops.evolved-stage-checkpoint-equivalence@1" and equivalence["exact_payload_and_manifest"] is True,
             "checkpoint replay contract differs")
        provenance = equivalence["provenance"]
        for phase in arrays:
            authority = provenance[phase]
            need(authority["clock"] == manifests[phase]["clock"] and authority["restart"] == wire.identity_token(manifests[phase]["restart_identity"],"restart")
                 and authority["run"] == wire.identity_token(manifests[phase]["run_identity"],"run"), "live creator authority differs")
            for name in ("artifact","bind","semantic"):
                need(authority[name] == case[name], "live creator identity differs")
        need(provenance["accepted"]["last_restart"] is None and provenance["continuous"]["last_restart"] is None
             and provenance["replay"]["last_restart"] == provenance["accepted"]["restart"], "restart lineage differs")
        for phase in arrays:
            payload = run_identity(provenance[phase]["run_manifest"])
            need(payload["bind_identity"] == case["bind"] and payload["run_identity"] == provenance[phase]["run"]
                 and payload["start_time"] == (0. if phase == "accepted" else DT)
                 and payload["start_macro_step"] == (0 if phase == "accepted" else 1)
                 and payload["controls"]["max_steps"] == 1 and payload["controls"]["t_end"] == STEPS[phase]*DT
                 and payload["continuation_identity"] == (provenance["accepted"]["run"] if phase == "replay" else None), "first/continued run request differs")
        results.append(dict(cells=n,width=width,science=science_result,diagnostic_rank_images=len(diagnostics["accepted"])))
    need(len(pins["junit"]) == pins["ranks"] and {row["rank"] for row in pins["junit"]} == set(range(pins["ranks"])), "raw all-rank XML inventory differs")
    for row in pins["junit"]:
        root = ET.fromstring(read(row["file"]))
        tests = root.findall(".//testcase")
        need(len(set(pins["batch_names"])) == len(pins["batch_names"]),"duplicate batch name pin")
        for suite in root.iter("testsuite"):
            need(all(int(suite.attrib.get(key,"0")) == 0 for key in ("failures","errors","skipped")),"raw suite totals not clean")
        need(len(tests) == len(pins["batch_names"]) and sorted(t.attrib["name"] for t in tests) == sorted(pins["batch_names"])
             and not any(t.find(tag) is not None for t in tests for tag in ("failure","error","skipped")), "full raw batch XML is not clean/exact")
        selected = [t for t in tests if t.attrib["name"].startswith("test_public_evolved_stage_amr_checkpoint_and_composite_Q[")]
        need(len(selected) == 4, "actual four-case AMR Stage XML inventory differs")
        for case in pins["cases"]:
            matches = [t for t in selected if t.attrib["name"] == f'test_public_evolved_stage_amr_checkpoint_and_composite_Q[{case["width"]}-{case["cells"]}]']
            need(len(matches) == 1, "missing/duplicate native parameter case")
            properties = matches[0].findall("./properties/property")
            need(len({p.attrib["name"] for p in properties}) == len(properties),"duplicate JUnit property")
            props = {p.attrib["name"]:p.attrib["value"] for p in properties}
            need(props.get("rank") == str(row["rank"]) and props.get("size") == str(pins["ranks"])
                 and props.get("dimension") == "2" and props.get("artifact_identity") == case["artifact"]
                 and props.get("evolved_stage_amr_receipt") == case["receipt"]["path"], "JUnit actual runtime provenance differs")
    return dict(status="received",qualification=QUALIFICATION,mode=pins["mode"],cases=results,
                cases_qualified=4,cpp_to_dso_crypto_link=False,
                limits=["homogeneous zero flux; not nonconstant AMR/restriction/constitutive flux qualification",
                        "not full M06/M13; no native execution performed by this reader",
                        "opaque Program/auxiliary images authenticated and replay-compared, not completely decoded",
                        "source/native/package provenance ROOT-attested; CPP-to-DSO build graph not cryptographically reconstructed",
                        "private capture leases/attempt authority not persisted; do not infer them from saved fields"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("contract")
    accept = sub.add_parser("receive")
    for key in ("pins","pins_sha256","approval","approval_sha256"):
        accept.add_argument("--"+key.replace("_","-"),required=True)
    args = parser.parse_args()
    result = contract() if args.command == "contract" else receive(args.pins,args.pins_sha256,args.approval,args.approval_sha256)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
