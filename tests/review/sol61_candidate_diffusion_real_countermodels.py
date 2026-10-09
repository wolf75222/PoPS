"""Resealed negative-only copies of real ROOT-pinned Candidate@2 archives.

Original positives require external ROOT seals. Locally manufactured approval
files are explicitly negative-test-only, never authenticated ROOT authority.
"""

from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "candidate_countermodel_reader",
    Path(__file__).with_name("sol61_candidate_diffusion_checkpoint_reception.py"),
)
c = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = c
_spec.loader.exec_module(c)
LABELS = (
    "solution-balanced-consumer",
    "frozen-D-manufactured-load",
    "history-issued-interval",
    "diagnostic-nonfinite",
    "clock-issued-grid",
    "documentary-D-beta",
    "rank-diagnostic-ownership",
    "restart-onebit",
    "artifact-binary-binding",
)
EXPECTED = (
    "original science guard",
    "manufactured load differs",
    "history exact issued duration",
    "diagnostic guard differs",
    "fixed dt issued grid differs",
    "documentary coefficient differs",
    "diagnostic width/rank/count differs",
    "accepted/reloaded solution",
    "artifact identity does not bind",
)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def leaves(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for v in value.values():
                yield from leaves(v)
    elif type(value) is list:
        for v in value:
            yield from leaves(v)


def snapshot(pins, pins_path, approval_path):
    rows = {str(c.wire.canonical(v["path"])): v["sha256"] for v in leaves(pins)}
    rows[str(c.wire.canonical(pins_path))] = c.digest(c.read(pins_path)[1])
    rows[str(c.wire.canonical(approval_path))] = c.digest(c.read(approval_path)[1])
    for path, sha in rows.items():
        c.need(c.digest(c.read(path, c.MAX_BYTES)[1]) == sha, "donor leaf changed: " + path)
    return rows


def clone(pins, directory):
    directory.mkdir()
    result = copy.deepcopy(pins)
    mapping = {case["directory"]: str(directory / key) for key, case in pins["cases"].items()}

    def paths(value):
        if type(value) is str:
            for source, target in mapping.items():
                if value == source or value.startswith(source + "/"):
                    return target + value[len(source) :]
            return value
        if type(value) is dict:
            return {k: paths(v) for k, v in value.items()}
        if type(value) is list:
            return [paths(v) for v in value]
        return value

    result = paths(result)
    result["archive_root"] = str(directory)
    result["file_roots"] = list(pins["file_roots"]) + [str(directory)]
    for key, case in pins["cases"].items():
        target = directory / key
        target.mkdir()
        c.need(len(list(Path(case["directory"]).iterdir())) == 11, "donor closed archive differs")
        for source in Path(case["directory"]).iterdir():
            c.need(not source.is_symlink(), "donor symlink")
            raw = c.read(source)[1]
            if source.name == "receipt.json":
                raw = (
                    json.dumps(paths(c.strict_json(raw)), sort_keys=True, indent=2) + "\n"
                ).encode()
            (target / source.name).write_bytes(raw)
    for rank, row in enumerate(pins["junit"]):
        root = ET.fromstring(c.pinned(row, pins["file_roots"])[1])
        for p in root.findall(".//property"):
            if p.get("value") is not None:
                p.set("value", paths(p.get("value")))
        target = directory / ("rank%d.xml" % rank)
        target.write_bytes(ET.tostring(root))
        result["junit"][rank] = c.leaf(target)
    for key in result["cases"]:
        refresh(result, key)
    return result


def arrays(path):
    return c.protocol.archive(c.read(path)[1])


def save(path, payload):
    with open(path, "wb") as f:
        np.savez(f, **payload)


def reseal_cp(payload):
    manifest = c.strict_json(str(payload["pops_checkpoint_manifest"].item()))
    manifest["arrays"] = {
        name: c.wire.typed_array(value)
        for name, value in payload.items()
        if name not in {"pops_checkpoint_manifest", "pops_restart_identity"}
    }
    data = {k: v for k, v in manifest.items() if k != "restart_identity"}
    manifest["restart_identity"]["hexdigest"] = c.digest(
        c.protocol.cbor(
            dict(protocol="pops.identity", domain="restart", schema_version=1, payload=data)
        )
    )
    payload["pops_checkpoint_manifest"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    payload["pops_restart_identity"] = np.asarray(
        c.protocol.identity_token(manifest["restart_identity"], "restart")
    )


def refresh(pins, key):
    case = pins["cases"][key]
    receipt_path = Path(case["receipt"]["path"])
    receipt = c.strict_json(c.read(receipt_path)[1])
    receipt["initial_sha256"] = c.digest(c.read(receipt["initial_npz"])[1])
    for row in receipt["phases"].values():
        row["sha256"] = c.digest(c.read(row["npz"])[1])
    for row in receipt["checkpoints"].values():
        row["sha256"] = c.digest(c.read(row["path"])[1])
    for row in receipt["sources"] + receipt["program_irs"]:
        row["sha256"] = c.digest(c.read(row["path"])[1])
    json_write(receipt_path, receipt)
    pins["cases"][key] = c.case_inventory(case["directory"])
    program = receipt["sources"][0]["component"]
    pins["owner"]["execution_association"][key]["components"][program]["cpp"] = pins["cases"][key][
        "cpp"
    ]


def wrong_frozen_load(q, alpha):
    # Independent wrong equation for a negative: remove only candidate q_c**2.
    width, n, _ = q.shape
    ds, rs = c.base.matrices(width)
    out = np.zeros_like(q)
    for row in range(width):
        for col in range(width):
            out[row] += float(rs[row][col]) * q[col]
            dc = float(ds[row][col]) * (1 + alpha)
            for axis in (0, 1):
                jump = np.roll(q[col], -1, axis=axis) - q[col]
                flux = 0.5 * (dc + np.roll(dc, -1, axis=axis)) * jump * n * n
                out[row] -= flux - np.roll(flux, 1, axis=axis)
        out[row] += 0.2 * q[row] ** 3
    return out


def edit_diagnostic(payload, rank, *, bad_owner=False):
    raw = bytearray(payload["program_diagnostics_state"].tobytes())
    offsets = payload["program_diagnostics_offsets"]
    lo, hi = int(offsets[rank]), int(offsets[rank + 1])
    c.need(raw[lo : lo + 8] == b"POPSDIA1", "actual diagnostic seed missing")
    if bad_owner:
        struct.pack_into("<Q", raw, lo + 16, len(offsets) - 1)
    else:
        count = struct.unpack_from("<Q", raw, lo + 32)[0]
        cursor = lo + 40
        changed = False
        for _ in range(count):
            length = struct.unpack_from("<Q", raw, cursor)[0]
            cursor += 8
            name = bytes(raw[cursor : cursor + length])
            cursor += length
            if name.endswith(b".rel_residual"):
                struct.pack_into("<Q", raw, cursor, 0x7FF8000000000021)
                changed = True
            cursor += 8
        c.need(changed and cursor == hi, "actual required diagnostic seed absent")
    payload["program_diagnostics_state"] = np.frombuffer(raw, dtype=np.uint8).copy()


def mutate(pins, label):
    key = "coupled3-201"
    case = pins["cases"][key]
    if label in ("solution-balanced-consumer", "frozen-D-manufactured-load"):
        if label == "frozen-D-manufactured-load":
            initial = arrays(case["initial"]["path"])
            load = wrong_frozen_load(initial["target"], initial["material"][0])
            initial["forcing"] = load
            save(case["initial"]["path"], initial)
        states = {phase: arrays(row["path"]) for phase, row in case["phases"].items()}
        for phase, data in states.items():
            if label == "solution-balanced-consumer":
                data["solution"][0, 3, 7] += 1e-4
                data["response"] = float(data["time"]) * data["solution"]
            else:
                data["forcing"] = load.copy()
            save(case["phases"][phase]["path"], data)
        for phase, row in case["checkpoints"].items():
            payload = arrays(row["path"])
            if label == "solution-balanced-consumer":
                payload["state_response"] = states[phase]["response"].reshape(
                    payload["state_response"].shape
                )
                for i in range(3):
                    for slot, source in enumerate((states["accepted"], states[phase])):
                        payload[f"history_q{i}_{slot}"] = source["solution"][i].reshape(
                            payload[f"history_q{i}_{slot}"].shape
                        )
            else:
                payload["state_forcing"] = load.reshape(payload["state_forcing"].shape)
            reseal_cp(payload)
            save(row["path"], payload)
    elif label == "restart-onebit":
        row = case["phases"]["reloaded"]
        data = arrays(row["path"])
        data["solution"][0, 3, 7] = np.nextafter(data["solution"][0, 3, 7], np.inf)
        save(row["path"], data)
    elif label == "documentary-D-beta":
        row = case["ir"]
        ir = c.strict_json(c.read(row["path"])[1])
        source = next(n for n in ir["nodes"] if n["op"] == "solve_spatial_field")["attrs"][
            "source_contract"
        ]
        changed = False

        def visit(n):
            nonlocal changed
            if type(n) is list:
                if not changed and len(n) == 2 and n[0] == "literal" and c.base.scalar(n[1]) == 3:
                    n[1] = dict(kind="integer", value="2")
                    changed = True
                else:
                    for x in n:
                        visit(x)

        # Local (row0,column1) corresponds to physical (2,0): nonzero D20.
        # Mutating a beta factor multiplied by literal zero would be a no-op.
        visit(source["diffusion"][1])
        c.need(changed, "actual candidate beta seed missing")
        json_write(row["path"], ir)
    elif label == "artifact-binary-binding":
        program = next(
            n
            for n in pins["owner"]["execution_association"][key]["components"]
            if n.startswith("program-")
        )
        row = pins["owner"]["execution_association"][key]["components"][program]
        data = c.strict_json(c.pinned(row["sidecar"], pins["file_roots"])[1])
        data["artifact_identity"] = "pops.artifact.v1:sha256:" + "0" * 64
        target = Path(pins["archive_root"]) / "counterfeit-program-sidecar.json"
        json_write(target, data)
        row["sidecar"] = c.leaf(target)
    else:
        row = case["checkpoints"]["accepted"]
        payload = arrays(row["path"])
        if label == "history-issued-interval":
            raw = bytearray(payload["history_sample_identity_q0"].tobytes())
            raw[-8:] = struct.pack("<Q", 2)
            payload["history_sample_identity_q0"] = np.frombuffer(raw, dtype=np.uint8).copy()
        elif label == "clock-issued-grid":
            t = c.strict_json(str(payload["temporal_restart_state"].item()))
            t["controller_state"]["fixed_dt_grid"]["time"] = np.nextafter(c.DT, np.inf).hex()
            payload["temporal_restart_state"] = np.asarray(
                json.dumps(t, sort_keys=True, separators=(",", ":"))
            )
        elif label in ("diagnostic-nonfinite", "rank-diagnostic-ownership"):
            edit_diagnostic(
                payload, pins["ranks"] - 1, bad_owner=label == "rank-diagnostic-ownership"
            )
        else:
            raise ValueError("foreign mutation")
        reseal_cp(payload)
        save(row["path"], payload)
    refresh(pins, key)


def run(pins_path, pins_sha, approval_path, approval_sha, output):
    # This is the only positive receive: real untouched originals with explicit
    # external ROOT authorities. A locally manufactured positive is forbidden.
    baseline = c.receive(pins_path, pins_sha, approval_path, approval_sha)
    originals = c.strict_json(c.read(pins_path)[1])
    before = snapshot(originals, pins_path, approval_path)
    output = Path(output)
    c.need(not output.exists(), "fresh output required")
    output.mkdir()
    json_write(output / "actual-root-approved-baseline.json", baseline)
    records = []
    for label, expected in zip(LABELS, EXPECTED, strict=True):
        directory = output / label
        pins = clone(originals, directory)
        mutate(pins, label)
        local = directory / "pins.negative-test-only.json"
        json_write(local, pins)
        sha = c.digest(c.read(local)[1])
        approval = directory / "approval.negative-test-only.json"
        json_write(
            approval,
            dict(
                schema="sol61.candidate-d-root-approval@2",
                approved_by="ROOT",
                pins_sha256=sha,
                qualification=c.QUALIFICATION,
            ),
        )
        json_write(
            directory / "negative-test-only-authority.json",
            dict(
                authority="LOCAL COUNTERMODEL ONLY; NOT AUTHENTICATED ROOT",
                positive_native_qualification=False,
                real_owner_pins_sha256=pins_sha,
                real_approval_sha256=approval_sha,
                local_pins_sha256=sha,
                local_approval_sha256=c.digest(c.read(approval)[1]),
            ),
        )
        try:
            c.receive(str(local), sha, str(approval), c.digest(c.read(approval)[1]))
        except ValueError as e:
            c.need(
                expected in str(e),
                "negative rejected by unexpected guard: " + label + ": " + str(e),
            )
            records.append(
                dict(
                    attack=label,
                    status="refused",
                    guard=str(e),
                    local_pins_sha256=sha,
                    negative_test_only=True,
                )
            )
        else:
            raise AssertionError("resealed negative accepted: " + label)
    after = snapshot(originals, pins_path, approval_path)
    c.need(before == after, "donor inventory changed")
    report = dict(
        schema="sol61.candidate-d-real-countermodels@1",
        status="9 negative-test-only copies refused",
        real_root_pins_sha256=pins_sha,
        real_root_approval_sha256=approval_sha,
        reader_qualification=c.QUALIFICATION,
        ranks=originals["ranks"],
        scientific_baseline=baseline,
        negative_authority="explicit local counterfeit approvals only; checksums are NOT ROOT authentication",
        donor_leaf_count=len(before),
        donors_unchanged=True,
        countermodels=records,
        limits=[
            "no new Native/DSO execution/import",
            "no missing compound semantic/artifact-spec/aggregate payload recomposition",
            "well-formed semantic replacements under counterfeit owner authority cannot be authenticated by newly calculated hashes",
        ],
    )
    json_write(output / "countermodels-reception.json", report)
    json_write(output / "donor-inventory-before-after.json", dict(before=before, after=after))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "pins-sha256", "approval", "approval-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(args.pins, args.pins_sha256, args.approval, args.approval_sha256, args.output),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
