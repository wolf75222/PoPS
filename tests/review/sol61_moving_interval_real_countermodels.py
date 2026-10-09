"""Fully resealed negatives from real owner-pinned ALE files; never owner seals.

NumPy/stdlib only. A positive must first be received from the external root
pins. Countermodels live in a separate empty private tree; originals are read-only.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np

spec = importlib.util.spec_from_file_location(
    "independent_ale_negative_oracle",
    Path(__file__).with_name("sol61_moving_interval_offline_oracle.py"),
)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)
require = oracle.require
OWNER_SHA = "9cf00652fe3de74c437ba19f91fb9687adacde99d4c57cdb0e1d9939a37e8b4b"
LABELS = (
    "fma_one_bit_balanced_face",
    "nonfused_relative_amount",
    "wrong_original_velocity_balanced",
    "wrong_original_source_balanced",
    "wrong_coordinate_law_balanced",
    "legacy_false_temporal_key",
    "foreign_synchronization_cursor",
    "restored_controller_changed",
    "restored_native_history_changed",
)


def leaves(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            return [value]
        return [row for item in value.values() for row in leaves(item)]
    if type(value) is list:
        return [row for item in value for row in leaves(item)]
    return []


class Writer:
    def __init__(self, raw=b""):
        self.raw = bytearray(raw)

    def word(self, value):
        self.raw.extend((int(value) & ((1 << 64) - 1)).to_bytes(8, "little"))

    def real(self, value):
        self.raw.extend(struct.pack("<d", float(value)))

    def blob(self, value):
        self.word(len(value))
        self.raw.extend(value)

    def text(self, value):
        self.blob(value.encode())


def ledger_bytes(ledger):
    w = Writer(b"POPSEX02")
    w.word(len(ledger["records"]))
    for row in ledger["records"]:
        for name in ("operation", "occurrence", "context", "quadrature"):
            w.text(row[name])
        w.word(row["orientation"])
        for name in ("measure", "flux", "weight"):
            w.real(row[name])
        w.word(row["multiplicity"])
        for name in ("axis", "side", "component"):
            w.word(row[name] + 1)
        w.word(int(row["exterior"]))
        w.text(row["evaluation"])
    require(
        not ledger["quantities"] and not ledger["consumed"], "negative ALE scope has no integrals"
    )
    w.word(0)
    w.word(0)
    result = bytes(w.raw)
    require(oracle.common.exchange_image(result) == ledger, "negative ledger serialization differs")
    return result


def topology_bytes(topo, components):
    w = Writer()
    for value in (
        1,
        components,
        len(topo["boxes"]),
        int(topo["replicated"]),
        0,
        topo["size"],
        topo["rank"],
    ):
        w.word(value)
    for lo, hi, owner in topo["boxes"]:
        w.word(lo)
        w.word(hi)
        if not topo["replicated"]:
            w.word(owner)
    w.word(len(topo["local"]))
    for index in topo["local"]:
        w.word(index)
    return bytes(w.raw)


def put_fab(w, value):
    components, cells = value.shape
    w.word(cells)
    w.word(components)
    w.raw.extend(np.ascontiguousarray(value.T, dtype="<f8").tobytes())


def put_field(w, topo, values, components):
    w.blob(topology_bytes(topo, components))
    for value in values:
        put_fab(w, value)


def put_faces(w, values):
    w.word(len(values))
    for value in values:
        put_fab(w, value)


def moving_bytes(carrier, magic):
    """Negative serializer, byte-roundtripped against every real seed image."""
    w = Writer(magic)
    w.blob(ledger_bytes(carrier["ledger"]))
    w.word(1)
    w.text(carrier["identity"])
    w.word(0)
    w.text(carrier["frame"])
    w.text(carrier["clock"])
    w.real(oracle.GEOMETRY_TOL)
    w.word(carrier["generation"])
    receipt, topo = carrier["receipt"], carrier["topology"]
    w.text(oracle.context(receipt["point"], carrier["identity"]) if receipt else "")
    put_field(w, topo, carrier["state"], topo["components"])
    put_field(w, topo, carrier["volumes"], 1)
    put_faces(w, carrier["nodes"])
    put_faces(w, carrier["sweeps"])
    w.word(int(receipt is not None))
    if receipt:
        point = receipt["point"]
        w.text(point["clock"])
        for name in ("tick", "level", "substep", "stage", "numerator", "denominator"):
            w.word(point[name])
        w.real(point["dt"])
        w.real(point["time"])
        for name in ("graph", "rate", "application"):
            w.text(point[name])
        w.text(carrier["frame"])
        w.text(oracle.QUADRATURE)
        w.real(oracle.GEOMETRY_TOL)
        if magic == b"POPSEX04":
            w.word(receipt["convention"])
        for name, components in (
            ("previous", topo["components"]),
            ("old_volumes", 1),
            ("source", topo["components"]),
        ):
            put_field(w, topo, receipt[name], components)
        for name in ("old_nodes", "flux", "density"):
            put_faces(w, receipt[name])
    return bytes(w.raw)


def reseal_checkpoint(checkpoint):
    manifest = oracle.strict_json(str(checkpoint["pops_checkpoint_manifest"].item()))
    manifest["arrays"] = {}
    for name, array in checkpoint.items():
        if name in {"pops_checkpoint_manifest", "pops_restart_identity"}:
            continue
        header = oracle.common.cbor(
            dict(protocol="pops.array-evidence.v1", dtype=array.dtype.str, shape=list(array.shape))
        )
        manifest["arrays"][name] = dict(
            dtype=array.dtype.str,
            shape=list(array.shape),
            content_sha256=oracle.digest(header + array.tobytes()),
        )
    payload = {key: value for key, value in manifest.items() if key != "restart_identity"}
    manifest["restart_identity"]["hexdigest"] = oracle.digest(
        oracle.common.cbor(
            dict(protocol="pops.identity", domain="restart", schema_version=1, payload=payload)
        )
    )
    checkpoint["pops_checkpoint_manifest"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    checkpoint["pops_restart_identity"] = np.asarray(
        oracle.common.identity_token(manifest["restart_identity"], "restart")
    )


def relative(carrier):
    receipt = carrier["receipt"]
    require(
        len(carrier["topology"]["local"]) == 1,
        "counter-equation scoped to real one-patch Serial seed",
    )
    flux, trace, sweep = receipt["flux"][0], receipt["density"][0], carrier["sweeps"][0][0]
    result = np.empty_like(flux)
    for component, face in np.ndindex(flux.shape):
        result[component, face] = float(
            Fraction.from_float(float(flux[component, face]))
            - Fraction.from_float(float(trace[component, face]))
            * Fraction.from_float(float(sweep[face]))
        )
    return result


def sync_records(carrier):
    rel = relative(carrier)
    receipt = carrier["receipt"]
    for row in carrier["ledger"]["records"]:
        operation = row["operation"]
        if operation.startswith("amount:"):
            cell, side = (
                int(value)
                for value in (
                    row["occurrence"].split(":")[1].split("/")[0],
                    row["occurrence"].rsplit(":", 1)[1],
                )
            )
            component = int(operation.rsplit(":", 1)[1])
            row["flux"] = float(rel[component, cell + side])
        elif operation.startswith("geometry:"):
            cell = int(row["occurrence"].split(":")[1].split("/")[0])
            side = int(row["occurrence"].rsplit(":", 1)[1])
            row["flux"] = float(carrier["sweeps"][0][0, cell + side])
        else:
            component = int(operation.rsplit(":", 1)[1])
            cell = int(row["occurrence"].split(":")[1])
            row["flux"] = float(receipt["source"][0][component, cell])


def mirror_output(case, phase, state):
    row = case["outputs"][phase]
    path = Path(row["path"])
    data = oracle.common.archive(oracle.file_bytes(path))
    manifest = oracle.strict_json(str(data["pops_output_manifest"].item()))
    for field in manifest["datasets"]["fields"].values():
        for piece in field["pieces"]:
            lo, hi = piece["lower"][0], piece["upper"][0]
            data[piece["name"]] = state["state"][:, lo:hi].copy()
    # The exact real observer geometry member names are declared in its manifest.
    geometry = next(iter(manifest["datasets"]["geometries"].values()))
    for name in ("node_coordinates", "cell_volumes"):
        target = geometry[name]
        data[target] = state[name].copy()
    for name in manifest["arrays"]:
        value = np.ascontiguousarray(data[name])
        header = value.dtype.str.encode() + b"\0" + ",".join(map(str, value.shape)).encode() + b"\0"
        manifest["arrays"][name] = dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            content_sha256=oracle.digest(header + value.tobytes()),
        )

    def evidence(value):
        value = np.ascontiguousarray(value)
        header = oracle.common.cbor(
            dict(protocol="pops.array-evidence.v1", dtype=value.dtype.str, shape=list(value.shape))
        )
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            content_sha256=oracle.digest(header + value.tobytes()),
        )

    for field, typed in zip(
        manifest["datasets"]["fields"].values(), manifest["snapshot"]["fields"], strict=True
    ):
        for piece, target in zip(field["pieces"], typed["pieces"], strict=True):
            target["array"] = evidence(data[piece["name"]])
    for name, target in geometry.items():
        manifest["snapshot"]["geometries"][0][name] = evidence(data[target])
    base = {key: value for key, value in manifest.items() if key != "output_identity"}
    output = dict(
        domain="scientific-output",
        schema_version=1,
        algorithm="sha256",
        hexdigest=oracle.digest(
            oracle.common.cbor(
                dict(
                    protocol="pops.identity",
                    domain="scientific-output",
                    schema_version=1,
                    payload=base,
                )
            )
        ),
    )
    manifest["output_identity"] = oracle.common.identity_token(output, "scientific-output")
    data["pops_output_manifest"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(path, **data)


def mutate(pins, label):
    fused = label in {"fma_one_bit_balanced_face", "nonfused_relative_amount"}
    components = ["a", "b", "c"] if fused else ["c", "a", "b"]
    case = next(
        case
        for case in pins["cases"]
        if case["kind"] == "restart"
        and case["cells"] == 16
        and case["components"] == components
        and case["source"]
    )
    phase = "accepted" if fused else ("restored" if label.startswith("restored_") else "step1")
    files = case["phases"][phase]
    paths = {name: Path(row["path"]) for name, row in files.items()}
    state = oracle.common.archive(oracle.file_bytes(paths["state"]))
    checkpoint = oracle.common.archive(oracle.file_bytes(paths["checkpoint"]))
    receipt = oracle.strict_json(oracle.file_bytes(paths["receipt"]))
    raw = checkpoint["program_exchange_state"].tobytes()
    carrier = oracle.moving_image(raw, case, 0, 1)
    require(
        moving_bytes(carrier, raw[:8]) == raw, "real carrier negative serializer is not byte-exact"
    )
    diagnostic = {}
    if label in {"fma_one_bit_balanced_face", "nonfused_relative_amount"}:
        candidates = []
        for row in carrier["ledger"]["records"]:
            if not row["operation"].startswith("amount:"):
                continue
            cell = int(row["occurrence"].split(":")[1].split("/")[0])
            side = int(row["occurrence"].rsplit(":", 1)[1])
            component = int(row["operation"].rsplit(":", 1)[1])
            face = cell + side
            native = carrier["receipt"]
            naive = float(native["flux"][0][component, face]) - float(
                native["density"][0][component, face]
            ) * float(carrier["sweeps"][0][0, face])
            if 0 < face < case["cells"] and naive != row["flux"]:
                candidates.append((row["operation"], face, row["flux"], naive))
        require(candidates, "real seed has no discriminating fused face")
        operation, face, value, naive = candidates[0]
        replacement = (
            float(np.nextafter(value, np.inf)) if label == "fma_one_bit_balanced_face" else naive
        )
        rows = [
            row
            for row in carrier["ledger"]["records"]
            if row["operation"] == operation
            and (
                int(row["occurrence"].split(":")[1].split("/")[0])
                + int(row["occurrence"].rsplit(":", 1)[1])
            )
            == face
        ]
        require(
            len(rows) == 2 and sum(row["orientation"] for row in rows) == 0,
            "shared-face incidences are not balanced",
        )
        for row in rows:
            row["flux"] = replacement
        diagnostic = dict(
            face=face,
            paired_incidences=2,
            old=value,
            new=replacement,
            native_bits=struct.pack("<d", value).hex(),
            negative_bits=struct.pack("<d", replacement).hex(),
            global_signed_face_delta=0.0,
        )
    elif label in {
        "wrong_original_velocity_balanced",
        "wrong_original_source_balanced",
        "wrong_coordinate_law_balanced",
    }:
        native = carrier["receipt"]
        q0 = native["previous"][0]
        v0 = native["old_volumes"][0][0]
        if label == "wrong_original_velocity_balanced":
            displacement = carrier["sweeps"][0][0]
            speed = case["velocity"] + 0.2 - displacement / oracle.DT
            left = q0[:, np.arange(case["cells"] + 1) - 1]
            right = q0[:, np.arange(case["cells"] + 1) % case["cells"]]
            frel = 0.5 * speed * (left + right) - 0.5 * np.abs(speed) * (right - left)
            native["flux"][0] = oracle.DT * frel + native["density"][0] * displacement
        elif label == "wrong_original_source_balanced":
            native["source"][0] *= 2.0
        else:
            nodes = state["node_coordinates"][:, 0].copy()
            ref = np.arange(case["cells"] + 1) / case["cells"]
            nodes[1:-1] += 1.0e-5 * np.sin(2 * np.pi * ref[1:-1])
            state["node_coordinates"][:, 0] = nodes
            state["cell_volumes"] = np.diff(nodes)
            old_rel = relative(carrier)
            carrier["nodes"][0][0] = nodes.copy()
            carrier["volumes"][0][0] = state["cell_volumes"].copy()
            carrier["sweeps"][0][0] = nodes - native["old_nodes"][0][0]
            native["flux"][0] = old_rel + native["density"][0] * carrier["sweeps"][0][0]
        rel = relative(carrier)
        state["state"] = (q0 * v0 - np.diff(rel, axis=1) + native["source"][0]) / state[
            "cell_volumes"
        ]
        carrier["state"][0] = state["state"].copy()
        sync_records(carrier)
        residual = (
            state["state"] * state["cell_volumes"]
            - q0 * v0
            + np.diff(rel, axis=1)
            - native["source"][0]
        )
        gcl = (state["cell_volumes"] - v0) - np.diff(carrier["sweeps"][0][0])
        require(
            np.max(abs(residual)) < 1.0e-13 and np.max(abs(gcl)) <= oracle.GEOMETRY_TOL,
            "negative self-balance failed",
        )
        diagnostic = dict(
            self_reynolds_linf=float(np.max(abs(residual))), self_gcl_linf=float(np.max(abs(gcl)))
        )
        mirror_output(case, phase, state)
    elif label == "restored_native_history_changed":
        checkpoint["history_names"] = np.asarray(["foreign-ring"])
    else:
        temporal = oracle.strict_json(str(checkpoint["temporal_restart_state"].item()))
        if label == "legacy_false_temporal_key":
            temporal["synchronization_state"] = temporal.pop("synchronization_cursors")
        elif label == "foreign_synchronization_cursor":
            temporal["synchronization_cursors"] = {"foreign": dict(macro_step=1)}
        else:
            grid = temporal["controller_state"]["fixed_dt_grid"]
            grid["origin"] = float(np.nextafter(float.fromhex(grid["origin"]), np.inf)).hex()
        checkpoint["temporal_restart_state"] = np.asarray(
            json.dumps(temporal, sort_keys=True, separators=(",", ":"))
        )
    image = moving_bytes(carrier, raw[:8])
    checkpoint["state_fluid"] = state["state"].copy()
    checkpoint["program_exchange_state"] = np.frombuffer(image, dtype=np.uint8).copy()
    reseal_checkpoint(checkpoint)
    np.savez_compressed(paths["checkpoint"], **checkpoint)
    np.savez_compressed(paths["state"], **state)
    receipt["checkpoint_sha256"] = oracle.digest(oracle.file_bytes(paths["checkpoint"]))
    receipt["image_sha256"] = [oracle.digest(image)]
    paths["receipt"].write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return diagnostic


def receive(path, owner_sha, workspace):
    require(
        owner_sha == OWNER_SHA and oracle.digest(oracle.file_bytes(path)) == owner_sha,
        "external root owner pins differ",
    )
    owner = oracle.strict_json(oracle.file_bytes(path))
    rows = leaves(owner)
    require(
        len(rows) == 285 and len({row["path"] for row in rows}) == 285,
        "exact real file inventory differs",
    )
    originals = {row["path"]: oracle.pinned_file(path.parent, row)[1] for row in rows}
    positive = oracle.receive(path)
    require(
        not workspace.exists() or not any(workspace.iterdir()), "negative workspace must be empty"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    results = []
    for label in LABELS:
        directory = workspace / label
        directory.mkdir()
        pins = copy.deepcopy(owner)
        for index, row in enumerate(leaves(pins)):
            target = directory / str(index) / Path(row["path"]).name
            target.parent.mkdir()
            target.write_bytes(originals[row["path"]])
            row["path"] = str(target.absolute())
        diagnostic = mutate(pins, label)
        for row in leaves(pins):
            row["sha256"] = oracle.digest(oracle.file_bytes(Path(row["path"])))
            oracle.pinned_file(directory, row)
        local = directory / "NOT-OWNER-countermodel-pins.json"
        local.write_text(json.dumps(pins, indent=2, allow_nan=False) + "\n")
        try:
            oracle.receive(local)
        except ValueError as error:
            results.append(
                dict(
                    countermodel=label,
                    refusal=str(error),
                    all_285_file_pins_verified=True,
                    NOT_OWNER_AUTHORITY=True,
                    harness_manifest_sha256=oracle.digest(oracle.file_bytes(local)),
                    **diagnostic,
                )
            )
        else:
            raise AssertionError("resealed scientific negative accepted: " + label)
    require(oracle.receive(path) == positive, "original root science changed")
    require(
        all(oracle.file_bytes(Path(name)) == raw for name, raw in originals.items()),
        "original root file bytes changed",
    )
    return dict(
        status="received",
        owner_sha256=owner_sha,
        authentic_positive=positive,
        countermodels=results,
        original_285_files_reverified_unchanged=True,
        negative_authority="explicit harness reseals, NOT root owner pins",
        native_execution_here=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--owner-sha256", required=True)
    parser.add_argument("--countermodels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = receive(args.pins, args.owner_sha256, args.countermodels_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], refusals=len(result["countermodels"]))))


if __name__ == "__main__":
    main()
