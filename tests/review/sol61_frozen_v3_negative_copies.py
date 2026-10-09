"""Negative copies of externally sealed Frozen@3 archives. No PoPS/native call.

Simulated ROOT approvals are permitted ONLY in marked NEGATIVE-TEST-ONLY copies.
The donor is authenticated with the two externally supplied real ROOT hashes.
No copied or modified archive is ever reported as a positive native reception.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

spec = importlib.util.spec_from_file_location("frozen_receiver", Path(__file__).with_name("sol61_captured_diffusion_saved_reception.py"))
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def json_write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def npz_edit(path, edit, *, checkpoint=False):
    arrays = r.protocol.archive(r.read(path)[1])
    edit(arrays)
    if checkpoint:
        manifest = json.loads(str(arrays.pop("pops_checkpoint_manifest").item()))
        arrays.pop("pops_restart_identity")
        manifest["arrays"] = {key: r.wire.typed_array(value) for key, value in arrays.items()}
        manifest["clock"] = dict(time=float(arrays["t"]).hex(), macro_step=int(arrays["macro_step"]))
        payload = {key: value for key, value in manifest.items() if key != "restart_identity"}
        hashed = r.digest(r.protocol.cbor(dict(protocol="pops.identity", domain="restart", schema_version=1, payload=payload)))
        manifest["restart_identity"]["hexdigest"] = hashed
        arrays["pops_checkpoint_manifest"] = np.array(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        arrays["pops_restart_identity"] = np.array("pops.restart.v1:sha256:"+hashed)
    np.savez(path, **arrays)


def relocate(value, replacements):
    if isinstance(value, dict):
        return {key: relocate(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            if value == old or value.startswith(old+"/"):
                return new+value[len(old):]
    return value


def refresh_receipt(directory):
    path = directory/"receipt.json"
    receipt = json.loads(path.read_text())
    receipt["initial_sha256"] = r.leaf(receipt["initial_npz"])["sha256"]
    for row in receipt["phases"].values():
        row["sha256"] = r.leaf(row["npz"])["sha256"]
    for row in list(receipt["checkpoints"].values())+receipt["sources"]+receipt["program_irs"]:
        row["sha256"] = r.leaf(row["path"])["sha256"]
    json_write(path, receipt)


def copy_archive(pins, directory):
    r.need(not directory.exists(), "negative copy output already exists")
    directory.mkdir(parents=True)
    json_write(directory/"NEGATIVE-TEST-ONLY.json", dict(qualification="NEGATIVE-TEST-ONLY",
        simulated_root_seals=True, native_execution=False, positive_native_qualification=False))
    replacements = {row["directory"]: str(directory/key) for key, row in pins["cases"].items()}
    for old, new in replacements.items():
        shutil.copytree(old, new)
        path = Path(new)/"receipt.json"
        json_write(path, relocate(json.loads(path.read_text()), replacements))
    xmls = []
    for rank, row in enumerate(pins["junit"]):
        tree = ET.fromstring(r.read(row["path"])[1])
        for prop in tree.findall(".//property"):
            if "value" in prop.attrib:
                prop.set("value", relocate(prop.get("value"), replacements))
        path = directory/("rank%d.xml" % rank)
        path.write_bytes(ET.tostring(tree))
        xmls.append(path)
    return replacements, xmls


def diagnostic_edit(arrays, attack):
    raw = arrays["program_diagnostics_state"].copy()
    offsets = arrays["program_diagnostics_offsets"].copy()
    if attack == "rank_duplicate":
        raw[int(offsets[1])+16:int(offsets[1])+24] = np.frombuffer(struct.pack("<Q", 0), dtype=np.uint8)
    elif attack == "rank_reorder":
        pieces = [raw[int(lo):int(hi)] for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)]
        raw = np.concatenate(pieces[::-1])
        offsets = np.array([0, len(pieces[1]), len(raw)], dtype=np.int64)
    elif attack == "offset_overlap":
        offsets[1] = len(raw)
    else:
        position = 40
        length, = struct.unpack_from("<Q", raw, position)
        position += 8+length
        if attack == "diagnostic_nonfinite":
            bits = 0x7ff8000000000042
        else:
            old, = struct.unpack_from("<d", raw, position)
            bits, = struct.unpack("<Q", struct.pack("<d", old+1))
        raw[position:position+8] = np.frombuffer(struct.pack("<Q", bits), dtype=np.uint8)
    arrays["program_diagnostics_state"], arrays["program_diagnostics_offsets"] = raw, offsets


def mutate(directory, attack):
    case = directory/"coupled3-201"
    receipt = json.loads((case/"receipt.json").read_text())
    if attack in ("transpose_D_equilibrium", "constant_shift_equilibrium"):
        initial = r.protocol.archive(r.read(receipt["initial_npz"])[1])
        q = initial["target"]
        if attack == "transpose_D_equilibrium":
            d, _ = r.matrices(3)
            forcing, _ = r.original(q, initial["material"][0], diffusion=np.array(d, dtype=float).T)
        else:
            q = q+.04
            forcing, _ = r.original(q, initial["material"][0])
        npz_edit(Path(receipt["initial_npz"]), lambda z: z.update(target=q.copy(), forcing=forcing.copy()))
        for phase, row in receipt["phases"].items():
            step = r.CLOCKS[phase][1]
            npz_edit(Path(row["npz"]), lambda z, step=step: z.update(solution=q.copy(), forcing=forcing.copy(), response=q*(step*r.DT)))
        for phase, row in receipt["checkpoints"].items():
            def fields(z, step=r.CLOCKS[phase][1]):
                z["state_forcing"] = forcing.reshape(z["state_forcing"].shape)
                z["state_response"] = (q*(step*r.DT)).reshape(z["state_response"].shape)
                for i in range(3):
                    for slot in (0, 1):
                        key = "history_q%d_%d" % (i, slot)
                        z[key] = q[i].reshape(z[key].shape)
            npz_edit(Path(row["path"]), fields, checkpoint=True)
    elif attack == "stale_material_capture":
        npz_edit(Path(receipt["phases"]["accepted"]["npz"]), lambda z: z.update(material=z["material"]+.01))
        npz_edit(Path(receipt["checkpoints"]["accepted"]["path"]), lambda z: z.update(state_material=z["state_material"]+.01), checkpoint=True)
    elif attack in ("IR_body", "IR_point"):
        path = Path(receipt["program_irs"][0]["path"])
        ir = json.loads(path.read_text())
        ir["nodes"][0]["op" if attack == "IR_body" else "point"] = "NEGATIVE-TEST-ONLY-stale"
        json_write(path, ir)
    elif attack == "XML_rank_duplicate":
        path = directory/"rank1.xml"
        tree = ET.fromstring(path.read_bytes())
        suite = next(s for s in tree.iter("testsuite") if s.findall("testcase"))
        cases = [node for node in suite.findall("testcase") if node.get("name", "").startswith("test_public_captured_diffusion_")]
        r.need(len(cases) == 2, "actual Frozen XML inventory absent")
        suite.remove(cases[1])
        suite.append(copy.deepcopy(cases[0]))
        path.write_bytes(ET.tostring(tree))
    else:
        phase = "replay" if attack == "diagnostic_replay_bits" else "continuous"
        path = Path(receipt["checkpoints"][phase]["path"])
        def durable(z):
            if attack == "history_ordinal":
                raw = z["history_sample_identity_q0"].copy()
                raw[-8:] = np.frombuffer(struct.pack("<Q", 2), dtype=np.uint8)
                z["history_sample_identity_q0"] = raw
            elif attack == "history_lag":
                temporal = json.loads(str(z["temporal_restart_state"].item()))
                temporal["history_cursors"]["q0"]["oldest_tick"] = 0
                z["temporal_restart_state"] = np.array(json.dumps(temporal))
            elif attack == "checkpoint_clock":
                z["t"] = np.array(.01)
            elif attack == "diagnostics_absence":
                del z["program_diagnostics_state"], z["program_diagnostics_offsets"]
            else:
                diagnostic_edit(z, attack)
        npz_edit(path, durable, checkpoint=True)


ATTACKS = {
    "transpose_D_equilibrium": "initial forcing differs from original",
    "constant_shift_equilibrium": "declared centre-sample recipe",
    "stale_material_capture": "readonly capture material",
    "history_ordinal": "stale captured/history publication point",
    "history_lag": "declared clock/schedule/history cursors",
    "checkpoint_clock": "checkpoint exact phase clock",
    "diagnostics_absence": "durable Program diagnostic images absent",
    "diagnostic_nonfinite": "diagnostic is nonfinite/negative",
    "diagnostic_replay_bits": "replay continuation program_diagnostics_state",
    "rank_duplicate": "diagnostic width/rank/count authority",
    "rank_reorder": "diagnostic width/rank/count authority",
    "offset_overlap": "diagnostic rank offsets/geometry",
    "IR_body": "actual Program IR/CPP/receipt hash link",
    "IR_point": "actual Program IR/CPP/receipt hash link",
    "XML_rank_duplicate": "JUnit receipt absent/duplicate",
}


def run(pins_path, pins_sha, approval_path, approval_sha, output):
    donor_result = r.receive(pins_path, pins_sha, approval_path, approval_sha)
    pins = json.loads(r.read(pins_path)[1])
    r.need(pins["schema"] == "sol61.captured-d-owner-pins@3" and pins["ranks"] == 2, "negative MPI2 suite requires authentic @3 MPI2 donor")
    output = output.resolve()
    r.need("NEGATIVE-TEST-ONLY" in output.parts and not output.exists(), "marked fresh negative output required")
    r.need(not output.is_relative_to(Path(pins["archive_root"])), "negative output cannot modify donor archive")
    results = {}
    for attack, expected in ATTACKS.items():
        directory = output/attack
        _, xmls = copy_archive(pins, directory)
        mutate(directory, attack)
        for key in pins["cases"]:
            refresh_receipt(directory/key)
        forged = copy.deepcopy(pins)
        forged["archive_root"] = str(directory)
        forged["file_roots"] = pins["file_roots"]+[str(directory)]
        forged["cases"] = {key: r.case_inventory(directory/key, fixture_version=3) for key in pins["cases"]}
        forged["junit"] = [r.leaf(path) for path in xmls]
        pin_file, approval_file = directory/"negative-pins.json", directory/"SIMULATED-ROOT-approval.json"
        json_write(pin_file, forged)
        pin_hash = r.leaf(pin_file)["sha256"]
        json_write(approval_file, dict(schema="sol61.captured-d-root-approval@3", approved_by="ROOT", pins_sha256=pin_hash,
                                     qualification="saved-states-original-residual@3"))
        try:
            r.receive(pin_file, pin_hash, approval_file, r.leaf(approval_file)["sha256"])
        except ValueError as error:
            r.need(expected in str(error), "unexpected negative refusal for %s: %s" % (attack, error))
            results[attack] = dict(status="refused", guard=str(error), simulated_root_seals=True,
                                   positive_native_qualification=False)
        else:
            raise AssertionError("negative archive unexpectedly accepted: "+attack)
    # All donor payloads are authenticated again after copy injections.
    r.need(r.receive(pins_path, pins_sha, approval_path, approval_sha) == donor_result, "donor changed during negative-copy tests")
    report = dict(qualification="NEGATIVE-TEST-ONLY", native_execution=False,
                  donor_pins_sha256=pins_sha, donor_approval_sha256=approval_sha,
                  scientific_cases=2, ranks=2, rejected_copies=results)
    json_write(output/"negative-results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "pins-sha256", "approval", "approval-sha256", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    report = run(args.pins, args.pins_sha256, args.approval, args.approval_sha256, Path(args.output))
    print(json.dumps(dict(qualification=report["qualification"], refused=len(report["rejected_copies"]),
                          scientific_cases=2, ranks=2, native_execution=False), sort_keys=True))


if __name__ == "__main__":
    main()
