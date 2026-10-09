"""Real archive negative controls. Copies/reseals are NOT root owner authority."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = load(Path(__file__).with_name("sol61_t5_portable_archive_checker.py"), "independent_archive_checker")
controls = load(Path(__file__).with_name("sol61_integral_feedback_real_countermodels.py"), "independent_t5_real_controls")
oracle = controls.oracle


def reseal_copy_manifest(root, data):
    # Explicit protocol countermodel seal, never root's positive archive seal.
    for row in data["files"]:
        path = root / row["path"]
        row["bytes"], row["sha256"] = path.stat().st_size, checker.digest(path.read_bytes())
    raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    (root / "manifest.json").write_bytes(raw)
    return checker.digest(raw)


def protocol_control(source, destination, label, original_checker):
    shutil.copytree(source, destination)
    manifest = checker.strict_json((destination / "manifest.json").read_bytes())
    expected = checker.ROOT_MANIFEST_SHA
    if label == "internal_symlink_same_hash":
        link = destination / "t5-sdk7b/positive/restart-n8/accepted-state.npz"
        target = link.with_name("restored-state.npz")
        assert link.read_bytes() == target.read_bytes()
        link.unlink()
        link.symlink_to(target.name)
    elif label == "extra_empty_directory":
        (destination / "explicit-countermodel-empty-dir").mkdir()
    elif label == "duplicate_mapping_row":
        manifest["mapping"].append(copy.deepcopy(manifest["mapping"][0]))
        expected = reseal_copy_manifest(destination, manifest)
    elif label == "dangling_mapping_row":
        extra = copy.deepcopy(manifest["mapping"][0])
        extra["source_path"] = "/explicit-countermodel/unused/accepted.npz"
        manifest["mapping"].append(extra)
        expected = reseal_copy_manifest(destination, manifest)
    elif label == "absolute_portable_mapping":
        portable_path = destination / "portable-owner-pins.json"
        portable = checker.strict_json(portable_path.read_bytes())
        name = portable["identity_file"]["path"]
        absolute = str(destination / name)
        portable["identity_file"]["path"] = absolute
        portable_path.write_text(json.dumps(portable, indent=2) + "\n")
        next(row for row in manifest["mapping"] if row["path"] == name)["path"] = absolute
        expected = reseal_copy_manifest(destination, manifest)
    elif label == "duplicate_inventory_row":
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        expected = reseal_copy_manifest(destination, manifest)
    elif label == "duplicate_json_key":
        raw = (destination / "manifest.json").read_text()
        raw = raw.replace('{\n', '{\n  "schema": "pops.native-t5-archive@1",\n', 1)
        (destination / "manifest.json").write_text(raw)
        expected = checker.digest(raw.encode())
    elif label == "changed_historical_source_snapshot":
        path = destination / "source/test_public_integral_feedback.py"
        path.write_bytes(path.read_bytes() + b"\n# Explicit archive countermodel source change.\n")
        expected = reseal_copy_manifest(destination, manifest)
    else:
        raise ValueError("unknown protocol control")
    old = original_checker.receive(destination, expected)
    checker.require(old["status"] == "received", "old checker did not reproduce the admitted countermodel")
    try:
        checker.receive(destination, expected)
    except ValueError as error:
        return dict(countermodel=label, original_checker_status=old["status"], strict_refusal=str(error),
                    external_root_manifest_unchanged=expected == checker.ROOT_MANIFEST_SHA,
                    harness_manifest_sha256=expected, NOT_ROOT_OWNER_AUTHORITY=True)
    raise AssertionError("strict checker accepted protocol countermodel " + label)


def scientific_control(portable, archive, directory, label):
    pins = controls.clone(portable, archive, directory)
    case = next(case for case in pins["cases"] if (case["kind"], case["cells"]) == ("restart", 8))
    first = oracle.archive(Path(case["phases"]["initial"]["state"]["path"]).read_bytes())
    actual = oracle.archive(Path(case["phases"]["accepted"]["state"]["path"]).read_bytes())
    u0, q0 = first["state"].reshape(-1), float(first["q"])
    if label == "source_reads_post_transport_snapshot":
        reacted = u0 - oracle.DT * oracle.GAMMA * q0 * actual["state"].reshape(-1)
    elif label == "capture_reads_endpoint_quantity":
        reacted = u0 * (1. - oracle.DT * oracle.GAMMA * float(actual["q"]))
    elif label == "physical_source_evaluated_twice":
        reacted = u0 * (1. - 2. * oracle.DT * oracle.GAMMA * q0)
    else:
        raise ValueError("unknown scientific control")
    wrong = reacted - oracle.DT * u0.size * (reacted - np.concatenate((reacted[:1], reacted[:-1])))
    wrong_q = q0 + oracle.DT * reacted[-1]

    def mutation(saved, checkpoint, ledgers, receipt):
        saved["state"] = wrong.reshape(saved["state"].shape)
        saved["level_0"] = saved["state"].copy()
        saved["q"] = np.asarray(wrong_q, dtype=np.float64)
        receipt["quantity"] = wrong_q
        for ledger in ledgers:
            ledger["quantities"][case["quantity_identity"]][1] = wrong_q
            for record in ledger["records"]:
                cell_text, axis_text, side_text = record["quadrature"].split("/")
                cell = int(cell_text.split(":")[1])
                axis, side = int(axis_text.split(":")[1]), int(side_text.split(":")[1])
                if axis == 0:
                    record["flux"] = float(reacted[cell if side == 1 else max(0, cell - 1)])

    delta = controls.modify_phase(pins, case, "accepted", mutation)
    pins["countermodel_harness"] = dict(kind=label, source_root_archive_sha256=checker.ROOT_MANIFEST_SHA,
                                       source_owner_pins_sha256=checker.OWNER_SHA,
                                       NOT_OWNER_PINS=True, NOT_NATIVE_POSITIVE_RECEIPTS=True)
    path = directory / "explicit-scientific-countermodel-pins.json"
    path.write_text(json.dumps(pins, indent=2, allow_nan=False) + "\n")
    for row in [pins["identity_file"]] + [row for entry in pins["cases"] for files in entry["phases"].values() for row in files.values()]:
        oracle.pinned_file(directory, row)
    oracle.load_snapshot(directory, pins, case, "accepted")
    try:
        oracle.receive(path)
    except ValueError as error:
        checker.require(str(error) == "reaction/candidate q capture or FV field update mismatch", "unexpected scientific refusal")
        # The wrong model remains self-consistent with its native-style flux
        # ledger; the refusal must come from the original source/capture law.
        source_inventory = float(np.mean(reacted - u0))
        boundary_inventory = oracle.DT * (reacted[0] - reacted[-1])
        residual = float(np.mean(wrong) - np.mean(u0) - source_inventory - boundary_inventory)
        checker.require(abs(residual) <= 3e-13 and abs((wrong_q - q0) - oracle.DT * reacted[-1]) <= 3e-13,
                        "countermodel is not internally balanced")
        return dict(countermodel=label, refusal=str(error), all_40_file_pins_verified=True,
                    authenticated_snapshot_before_scientific_refusal=True, mutation_delta=delta,
                    counterfactual_inventory_residual=residual, counterfactual_source_inventory=source_inventory,
                    original_source_inventory=float(np.mean(-oracle.DT * oracle.GAMMA * q0 * u0)),
                    harness_pins_sha256=checker.digest(path.read_bytes()), NOT_ROOT_OWNER_AUTHORITY=True)
    raise AssertionError("scientific countermodel accepted")


def receive(archive, workspace):
    archive = archive.absolute()
    checker.require(not workspace.exists() or not any(workspace.iterdir()), "countermodel workspace must be empty")
    workspace.mkdir(parents=True, exist_ok=True)
    positive = checker.receive(archive, checker.ROOT_MANIFEST_SHA)
    cold = workspace / "actual-cold-copy"
    shutil.copytree(archive, cold)
    original_pins = checker.strict_json((archive / "provenance/root-owner-pins.json").read_bytes())
    forbidden = [row["path"] for row in checker.owner_leaves(original_pins)]
    forbidden += [str(path.absolute()) for path in archive.rglob("*") if path.is_file()]
    # Actual portable bytes, with Python open-audit refusal of every original
    # donor file and every file in the initial root archive. No fabricated data.
    code = '''import importlib.abc,importlib.util,json,os,sys
from pathlib import Path
class NoPoPS(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=="pops" or name.startswith("pops."): raise AssertionError("PoPS import forbidden")
sys.meta_path.insert(0,NoPoPS())
denied=set(json.loads(sys.argv[3]))
def audit(event,args):
    if event=="open" and isinstance(args[0],(str,bytes)):
        name=str(Path(os.fsdecode(args[0])).absolute())
        if name in denied: raise AssertionError("original donor/archive read forbidden: "+name)
sys.addaudithook(audit)
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("cold_archive_checker",sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
result=module.receive(Path(sys.argv[2]),module.ROOT_MANIFEST_SHA)
print(json.dumps(result,allow_nan=False))
'''
    cold_run = subprocess.run([sys.executable, "-I", "-c", code, checker.__file__, str(cold), json.dumps(forbidden)],
                              capture_output=True, text=True, timeout=30)
    checker.require(cold_run.returncode == 0, "cold relocation failed: " + cold_run.stderr)
    checker.require(json.loads(cold_run.stdout) == positive, "cold relocation changed receipt")
    old = load(archive / "verify_archive.py", "authentic_root_archive_checker")
    protocol_labels = ("internal_symlink_same_hash", "extra_empty_directory", "duplicate_mapping_row", "dangling_mapping_row",
                       "absolute_portable_mapping", "duplicate_inventory_row", "duplicate_json_key", "changed_historical_source_snapshot")
    protocol = [protocol_control(archive, workspace / label, label, old) for label in protocol_labels]
    portable = checker.strict_json((archive / "portable-owner-pins.json").read_bytes())
    original_labels = ("wrong_q_in_original_reaction", "right_face_flux", "omitted_negative_trace_scale",
                       "transpose_physical_axes", "interior_trace_with_right_amount", "duration_one_ulp",
                       "foreign_declared_clock", "restore_physical_state", "rejected_attempt_state")
    scientific = []
    for label in original_labels:
        directory = workspace / label
        directory.mkdir()
        scientific.append(controls.receive_countermodel(portable, archive, directory, label))
    for label in ("source_reads_post_transport_snapshot", "capture_reads_endpoint_quantity", "physical_source_evaluated_twice"):
        directory = workspace / label
        directory.mkdir()
        scientific.append(scientific_control(portable, archive, directory, label))
    checker.require(checker.receive(archive, checker.ROOT_MANIFEST_SHA) == positive, "root archive changed")
    return dict(schema="sol61.t5-portable-archive.countermodel-reception@1", status="received",
                authentic_archive=positive, protocol_countermodels=protocol, scientific_countermodels=scientific,
                root_archive_reverified_unchanged=True, countermodel_workspace=str(workspace),
                cold_relocation_received_with_original_files_denied=len(forbidden), cold_pops_import_forbidden=True,
                countermodel_authority="explicit negative harness reseals, NOT root owner seals or native positive receipts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--countermodels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        result = receive(args.archive, args.countermodels_dir.absolute())
    finally:
        sys.dont_write_bytecode = prior
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], protocol_refusals=len(result["protocol_countermodels"]),
                         scientific_refusals=len(result["scientific_countermodels"]), root_archive_unchanged=True)))


if __name__ == "__main__":
    main()
