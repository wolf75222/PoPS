"""Scientific negative controls on copies of root-pinned authentic SDK7b states.

Every counter-model is fully resealed by this explicitly adversarial harness.
These pins are NOT owner pins and these states are NOT native positive receipts.
No PoPS import/build/JIT. Originals are read-only; scratch copies are temporary.
"""
import argparse
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np

SCRIPT = Path(__file__).with_name("sol61_integral_feedback_offline_oracle.py")
spec = importlib.util.spec_from_file_location("sol61_feedback_oracle_for_negative_controls", SCRIPT)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)

OWNER_SHA = "7a6d07c39e434b256e08792ff6c89d991c8525068eb0df0c0a37e43c12521bc2"
SOURCE = "634cba3511fef957e65a21fcae3ab257f164b360"
NATIVE_SHA = "8a217a0fff5a131729a842dd222e2e08346034eb53b4f461fc17fc8cc291b563"


def encode_image(ledger):
    """Counter-model serializer only; never used to produce a positive receipt."""
    data = bytearray(b"POPSEX02")

    def word(value):
        data.extend((value & ((1 << 64) - 1)).to_bytes(8, "little"))

    def text(value):
        raw = value.encode("utf-8")
        word(len(raw))
        data.extend(raw)

    def number(value):
        data.extend(struct.pack("<d", value))

    word(len(ledger["records"]))
    for record in ledger["records"]:
        for name in ("operation", "occurrence", "context", "quadrature"):
            text(record[name])
        word(record["orientation"])
        for name in ("measure", "flux", "weight"):
            number(record[name])
        word(record["multiplicity"])
        for name in ("axis", "side", "component"):
            word(record[name] + 1)
        word(int(record["exterior"]))
        text(record["evaluation"])
    word(len(ledger["quantities"]))
    for name, (initial, value) in sorted(ledger["quantities"].items()):
        text(name)
        number(initial)
        number(value)
    word(len(ledger["consumed"]))
    for key in sorted(ledger["consumed"]):
        for value in key:
            text(value)
    result = bytes(data)
    oracle.require(oracle.exchange_image(result) == ledger, "negative-control wire roundtrip failed")
    return result


def clone(owner, base, directory):
    pins = copy.deepcopy(owner)
    sources = [(owner["identity_file"], pins["identity_file"], directory / "identity.json")]
    for original_case, copied_case in zip(owner["cases"], pins["cases"], strict=True):
        target = directory / f"{original_case['kind']}-n{original_case['cells']}"
        target.mkdir()
        for phase in original_case["phases"]:
            for kind in ("receipt", "state", "checkpoint"):
                row = original_case["phases"][phase][kind]
                source = Path(row["path"])
                sources.append((row, copied_case["phases"][phase][kind], target / source.name))
    for original, copied, destination in sources:
        path, _ = oracle.pinned_file(base, original)
        shutil.copyfile(path, destination)
        copied["path"] = str(destination)
    oracle.require(len(sources) == 40, "negative controls need exact root campaign inventory")
    return pins


def reseal_checkpoint(payload):
    """Self-consistent adversarial envelope; it has no native/owner authority."""
    manifest = oracle.strict_json(str(payload["pops_checkpoint_manifest"].item()))
    fields = {name: value for name, value in payload.items()
              if name not in {"pops_checkpoint_manifest", "pops_restart_identity"}}
    manifest["arrays"] = {}
    for name, array in fields.items():
        header = oracle.cbor(dict(protocol="pops.array-evidence.v1", dtype=array.dtype.str, shape=list(array.shape)))
        manifest["arrays"][name] = dict(dtype=array.dtype.str, shape=list(array.shape),
                                       content_sha256=oracle.digest(header + array.tobytes(order="C")))
    base = {key: value for key, value in manifest.items() if key != "restart_identity"}
    envelope = dict(protocol="pops.identity", domain="restart", schema_version=1, payload=base)
    manifest["restart_identity"]["hexdigest"] = oracle.digest(oracle.cbor(envelope))
    payload["pops_checkpoint_manifest"] = np.asarray(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    payload["pops_restart_identity"] = np.asarray(oracle.identity_token(manifest["restart_identity"], "restart"))


def modify_phase(pins, case, phase, mutation):
    files = case["phases"][phase]
    receipt = oracle.strict_json(Path(files["receipt"]["path"]).read_text())
    saved = oracle.archive(Path(files["state"]["path"]).read_bytes())
    checkpoint = oracle.archive(Path(files["checkpoint"]["path"]).read_bytes())
    ledgers = copy.deepcopy(receipt["ledgers"])
    original_state, original_q = saved["state"].copy(), float(saved["q"])
    original_ledgers = copy.deepcopy(ledgers)
    mutation(saved, checkpoint, ledgers, receipt)
    checkpoint["state_fluid"] = saved["state"].copy()
    images = [encode_image(ledger) for ledger in ledgers]
    wire = b"".join(images)
    offsets, cursor = [0], 0
    for image in images:
        cursor += len(image)
        offsets.append(cursor)
    checkpoint["program_exchange_state"] = np.frombuffer(wire, dtype=np.uint8).copy()
    checkpoint["program_exchange_offsets"] = np.asarray(offsets, dtype=np.int64)
    reseal_checkpoint(checkpoint)
    np.savez(files["checkpoint"]["path"], **checkpoint)
    np.savez(files["state"]["path"], **saved)
    receipt["checkpoint"] = files["checkpoint"]["path"]
    receipt["checkpoint_sha256"] = oracle.digest(Path(files["checkpoint"]["path"]).read_bytes())
    receipt["ledger_sha256"] = [oracle.digest(image) for image in images]
    receipt["ledgers"] = ledgers
    Path(files["receipt"]["path"]).write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    for row in files.values():
        row["sha256"] = oracle.digest(Path(row["path"]).read_bytes())
    changes = []
    for rank, (old, new) in enumerate(zip(original_ledgers, ledgers, strict=True)):
        for index, (before, after) in enumerate(zip(old["records"], new["records"], strict=True)):
            fields = [name for name in before if before[name] != after[name]]
            if fields:
                changes.append(dict(rank=rank, record=index, fields=fields))
    return dict(state_shape_before=list(original_state.shape), state_shape_after=list(saved["state"].shape),
                state_linf_delta=float(np.max(np.abs(original_state.reshape(-1) - saved["state"].reshape(-1)))),
                q_delta=float(saved["q"]) - original_q, changed_native_record_fields=changes)


def right_trace(ledgers):
    rows = [record for ledger in ledgers for record in ledger["records"] if record["exterior"]
            and (record["axis"], record["side"], record["component"]) == (0, 1, 0)]
    oracle.require(len(rows) == 1, "authentic negative-control seed lacks unique right trace")
    return rows[0]


def key(record):
    return [record[name] for name in ("operation", "occurrence", "context", "quadrature")]


def move_consumed(ledgers, old, new):
    changed = 0
    for ledger in ledgers:
        for index, consumed in enumerate(ledger["consumed"]):
            if consumed == old:
                ledger["consumed"][index] = new
                changed += 1
    oracle.require(changed == 1, "counter-model must move one actual consumed key")


def context_fields(record):
    prefix = "pops.exchange.frame.v1/"
    length, remainder = record["context"][len(prefix):].split(":", 1)
    raw = remainder.encode()
    count = int(length)
    clock = raw[:count].decode()
    fields = raw[count + 1:].decode().split("/", 8)
    oracle.require(len(fields) == 9, "incomplete authentic source context")
    return prefix, clock, fields


def receive_countermodel(owner, base, directory, label):
    pins = clone(owner, base, directory)
    case = next(case for case in pins["cases"] if (case["kind"], case["cells"]) == ("restart", 8))
    phase = "accepted"
    if label == "wrong_q_in_original_reaction":
        initial_path = Path(case["phases"]["initial"]["state"]["path"])
        initial = oracle.archive(initial_path.read_bytes())["state"].reshape(-1)

        def mutation(saved, checkpoint, ledgers, receipt):
            # Counterfactual: read endpoint q instead of the accepted pre-step candidate q.
            wrong_q = float(saved["q"])
            reacted = initial * (1. - oracle.GAMMA * wrong_q * oracle.DT)
            lower = np.concatenate((reacted[:1], reacted[:-1]))
            wrong = reacted - oracle.DT * initial.size * (reacted - lower)
            saved["state"] = wrong.reshape(saved["state"].shape)
            saved["level_0"] = saved["state"].copy()
        expected = "reaction/candidate q capture or FV field update mismatch"
    elif label == "right_face_flux":
        def mutation(saved, checkpoint, ledgers, receipt):
            right_trace(ledgers)["flux"] += .001
        expected = "ledger physical flux/measure/duration/orientation mismatch"
    elif label == "omitted_negative_trace_scale":
        def mutation(saved, checkpoint, ledgers, receipt):
            wrong = oracle.Q0 - (float(saved["q"]) - oracle.Q0)
            saved["q"] = np.asarray(wrong, dtype=np.float64)
            receipt["quantity"] = wrong
            for ledger in ledgers:
                ledger["quantities"][case["quantity_identity"]][1] = wrong
        expected = "reaction/candidate q capture or FV field update mismatch"
    elif label == "transpose_physical_axes":
        def mutation(saved, checkpoint, ledgers, receipt):
            saved["state"] = saved["state"].transpose(0, 2, 1).copy()
            saved["level_0"] = saved["state"].copy()
        expected = "T5 state shape/type/finitude mismatch"
    elif label == "interior_trace_with_right_amount":
        def mutation(saved, checkpoint, ledgers, receipt):
            actual = right_trace(ledgers)
            prior = key(actual)
            interior = next(record for ledger in ledgers for record in ledger["records"]
                            if record["quadrature"] == "cell:6:0/axis:0/side:1")
            actual["exterior"] = False
            interior["exterior"] = True
            interior["flux"] = actual["flux"]  # correct amount does not confer physical endpoint authority
            move_consumed(ledgers, prior, key(interior))
        expected = "wrong physical right endpoint cell"
    elif label in {"duration_one_ulp", "foreign_declared_clock"}:
        def mutation(saved, checkpoint, ledgers, receipt):
            record = right_trace(ledgers)
            prior = key(record)
            prefix, clock, fields = context_fields(record)
            if label == "duration_one_ulp":
                fields[6] = str(int(fields[6]) + 1)
            else:
                clock += "/foreign-countermodel"
            record["context"] = prefix + str(len(clock.encode())) + ":" + clock + "/" + "/".join(fields)
            move_consumed(ledgers, prior, key(record))
        expected = ("wrong runtime interval/duration authority" if label == "duration_one_ulp"
                    else "trace uses a foreign declared clock")
    elif label in {"restore_physical_state", "rejected_attempt_state"}:
        phase = "restored"
        if label == "rejected_attempt_state":
            case = next(case for case in pins["cases"] if case["kind"] == "retry")
            phase = "rejected"

        def mutation(saved, checkpoint, ledgers, receipt):
            saved["state"].reshape(-1)[2] += .001
            saved["level_0"] = saved["state"].copy()
        expected = ("restart restore did not preserve exact state/q/clock/rank-ledger bytes"
                    if label == "restore_physical_state"
                    else "rejected attempt did not preserve exact state/q/clock/rank-ledger bytes")
    else:
        raise ValueError("unknown counter-model")
    delta = modify_phase(pins, case, phase, mutation)
    pins["countermodel_harness"] = dict(kind=label, source_owner_pins_sha256=OWNER_SHA,
                                       NOT_OWNER_PINS=True, NOT_NATIVE_POSITIVE_RECEIPTS=True)
    pins_path = directory / "explicit-countermodel-pins.json"
    pins_path.write_text(json.dumps(pins, indent=2, allow_nan=False) + "\n")
    for row in [pins["identity_file"]] + [row for entry in pins["cases"]
            for files in entry["phases"].values() for row in files.values()]:
        oracle.pinned_file(directory, row)
    # Deliberately authenticate the resealed phase independently of the scientific step.
    # Axis transpose refuses at the source-declared spatial shape, rather than a checksum.
    if label != "transpose_physical_axes":
        oracle.load_snapshot(directory, pins, case, phase)
    try:
        oracle.receive(pins_path)
    except ValueError as error:
        oracle.require(str(error) == expected, f"unexpected counter-model refusal for {label}: {error}")
        return dict(countermodel=label, phase=phase, expected_refusal=str(error),
                    all_40_external_file_pins_verified=True, mutation_delta=delta,
                    resealed_phase_sha256={name: value["sha256"] for name, value in case["phases"][phase].items()},
                    resealed_pins_sha256=oracle.digest(pins_path.read_bytes()),
                    provenance="adversarial harness copies; NOT root-owner pins/native positive receipts")
    raise ValueError("scientific oracle accepted counter-model: " + label)


def receive(owner_path, countermodels_dir=None):
    raw = owner_path.read_bytes()
    oracle.require(oracle.digest(raw) == OWNER_SHA, "owner pin manifest differs from root's explicit SHA")
    owner = oracle.strict_json(raw)
    oracle.require(owner["source_commit"] == SOURCE and owner["native_sha256"] == NATIVE_SHA,
                   "counter-model seed is not the historically received SDK7b run")
    positive = oracle.receive(owner_path)
    labels = ("wrong_q_in_original_reaction", "right_face_flux", "omitted_negative_trace_scale",
              "transpose_physical_axes", "interior_trace_with_right_amount", "duration_one_ulp",
              "foreign_declared_clock", "restore_physical_state", "rejected_attempt_state")
    if countermodels_dir is not None:
        countermodels_dir = countermodels_dir.resolve()
        countermodels_dir.mkdir(parents=True, exist_ok=True)
        oracle.require(not any(countermodels_dir.iterdir()), "countermodel directory must be empty")
    context = (tempfile.TemporaryDirectory(prefix="sol61-t5-negative-controls-")
               if countermodels_dir is None else nullcontext(str(countermodels_dir)))
    with context as scratch:
        rows = []
        for label in labels:
            directory = Path(scratch) / label
            directory.mkdir()
            rows.append(receive_countermodel(owner, owner_path.resolve().parent, directory, label))
    oracle.require(oracle.receive(owner_path) == positive, "authentic originals changed during negative controls")
    return dict(schema="sol61.integral-feedback.countermodel-reception@1",
                status="9 fully resealed scientific countermodels refused",
                authentic_positive=positive, countermodels=rows,
                authentic_owner_files_rechecked_after_controls=True,
                countermodel_artifacts=(None if countermodels_dir is None else str(countermodels_dir)),
                scope="offline authenticated actual SDK7b saves; original files read-only; no PoPS/native/MPI execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--countermodels-dir", type=Path)
    args = parser.parse_args()
    result = receive(args.owner_pins, args.countermodels_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
