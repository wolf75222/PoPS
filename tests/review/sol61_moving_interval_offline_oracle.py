"""Independent ALE receipt equations. NumPy/stdlib, external owner pins only.

No source-only probe creates a positive physical receipt. The decoder covers
the declared Uniform1D witness; it does not impose a production dimension cap.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

# Shared independent evidence primitives, not PoPS or its numerical producer.
_spec = importlib.util.spec_from_file_location("sol61_evidence_primitives", Path(__file__).with_name(
    "sol61_integral_feedback_offline_oracle.py"))
common = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = common
_spec.loader.exec_module(common)
require, digest, strict_json = common.require, common.digest, common.strict_json
SCHEMA = "sol61.moving-interval.offline-pins@1"
SOURCE_CONTRACT = "009ed9d2e27d9f41d697ef3c9362878705cf22c0"
DT, SOURCE_COEFFICIENT, GEOMETRY_TOL = .001, .05, 1e-13
EPS = np.finfo(np.float64).eps
QUADRATURE = "left_endpoint@1/exact_endpoint_displacement@1"
PHASES = {"restart": ("initial", "step1", "accepted", "continuous", "restored", "replayed"),
          "retry": ("before", "rejected", "retried")}


class Reader:
    def __init__(self, raw):
        require(len(raw) <= common.MAX_FILE_BYTES, "wire exceeds reception budget")
        self.raw, self.position = raw, 0

    def take(self, count):
        require(type(count) is int and 0 <= count <= len(self.raw) - self.position, "truncated wire")
        value = self.raw[self.position:self.position + count]
        self.position += count
        return value

    def word(self, signed=False):
        return int.from_bytes(self.take(8), "little", signed=signed)

    def real(self):
        value = struct.unpack("<d", self.take(8))[0]
        require(math.isfinite(value), "nonfinite moving wire value")
        return value

    def blob(self):
        return self.take(self.word())

    def text(self):
        return self.blob().decode("utf-8")

    def count(self, minimum):
        value = self.word()
        require(value <= (len(self.raw) - self.position) // minimum, "wire count exceeds remaining image")
        return value

    def finish(self):
        require(self.position == len(self.raw), "trailing moving wire bytes")


def topology(raw, cells, components, rank, size):
    reader = Reader(raw)
    require(reader.word(True) == 1 and reader.word(True) == components, "wire dimension/component mismatch")
    patches = reader.count(16)
    replicated = reader.word(True)
    require(replicated in (0, 1), "invalid replication declaration")
    origin, extent, local = (reader.word(True) for _ in range(3))
    require(origin == 0 and extent == size and local == rank, "wrong rank-space/local-rank authority")
    boxes = []
    for _ in range(patches):
        lo, hi = reader.word(True), reader.word(True)
        owner = None if replicated else reader.word(True)
        require(0 <= lo <= hi < cells and (replicated or 0 <= owner < size), "foreign box/owner")
        boxes.append((lo, hi, owner))
    ordered = sorted(boxes)
    require(ordered and ordered[0][0] == 0 and ordered[-1][1] == cells - 1
            and all(a[1] + 1 == b[0] for a, b in zip(ordered[:-1], ordered[1:], strict=True)),
            "global boxes do not partition physical cells")
    local_indices = tuple(reader.word() for _ in range(reader.count(8)))
    expected = tuple(index for index, (_, _, owner) in enumerate(boxes) if replicated or owner == rank)
    require(local_indices == expected, "wrong local ownership/patch order")
    reader.finish()
    return dict(components=components, replicated=bool(replicated), boxes=boxes,
                local=local_indices, rank=rank, size=size)


def fab(reader, cells, components):
    require(reader.word() == cells and reader.word(True) == components, "wire Fab shape mismatch")
    require(cells * components <= (len(reader.raw) - reader.position) // 8, "Fab allocation exceeds image")
    values = np.frombuffer(reader.take(8 * cells * components), dtype="<f8").copy()
    require(np.isfinite(values).all(), "nonfinite wire Fab")
    return values.reshape(cells, components).T


def field(reader, cells, components, rank, size, expected=None):
    topo = topology(reader.blob(), cells, components, rank, size)
    if expected is not None:
        require({key: value for key, value in topo.items() if key != "components"}
                == {key: value for key, value in expected.items() if key != "components"},
                "receipt field topology differs")
    values = [fab(reader, topo["boxes"][index][1] - topo["boxes"][index][0] + 1, components)
              for index in topo["local"]]
    return topo, values


def faces(reader, topo, components):
    require(reader.word() == len(topo["local"]), "face patch count mismatch")
    return [fab(reader, topo["boxes"][index][1] - topo["boxes"][index][0] + 2, components)
            for index in topo["local"]]


def point(reader):
    value = dict(clock=reader.text(), tick=reader.word(True), level=reader.word(True),
                 substep=reader.word(True), stage=reader.word(True), numerator=reader.word(True),
                 denominator=reader.word(True), dt=reader.real(), time=reader.real(),
                 graph=reader.text(), rate=reader.text(), application=reader.text())
    require(value["clock"] and value["tick"] >= 0 and value["dt"] > 0
            and (value["level"], value["substep"], value["stage"], value["numerator"], value["denominator"])
            == (0, 0, 0, 0, 1) and not any(value[key] for key in ("graph", "rate", "application")),
            "wrong ordinary interval point authority")
    return value


def context(point, identity):
    def bits(value):
        return int.from_bytes(struct.pack("<d", value), "little")
    return ("pops.exchange.frame.v1/" + str(len(point["clock"].encode())) + ":" + point["clock"] + "/"
            + "/".join(str(point[key]) for key in ("tick", "level", "substep", "stage", "numerator", "denominator"))
            + "/" + str(bits(point["dt"])) + "/" + str(bits(point["time"]))
            + "/" + str(len(identity.encode())) + ":" + identity)


def moving_image(raw, case, rank, size):
    reader = Reader(raw)
    require(reader.take(8) == b"POPSEX03", "ALE witness requires POPSEX03")
    ledger = common.exchange_image(reader.blob())
    require(not ledger["quantities"] and not ledger["consumed"], "unexpected integral authority in ALE witness")
    require(reader.word() == 1, "wrong moving carrier count")
    identity, block, frame, clock = reader.text(), reader.word(True), reader.text(), reader.text()
    tolerance, generation, last_interval = reader.real(), reader.word(), reader.text()
    require(identity == case["moving_identity"] and block == 0 and frame == case["physical_frame"]
            and clock and tolerance == GEOMETRY_TOL, "wrong carrier/block/frame/clock/tolerance authority")
    n, nc = case["cells"], len(case["components"])
    topo, state = field(reader, n, nc, rank, size)
    _, volumes = field(reader, n, 1, rank, size, topo)
    nodes, sweeps = faces(reader, topo, 1), faces(reader, topo, 1)
    present = reader.word()
    require(present in (0, 1) and bool(present) == bool(generation)
            and bool(last_interval) == bool(generation), "generation/receipt authority mismatch")
    receipt = None
    if present:
        issued = point(reader)
        receipt_frame, quadrature, receipt_tolerance = reader.text(), reader.text(), reader.real()
        require(receipt_frame == frame and issued["clock"] == clock and receipt_tolerance == tolerance
                and quadrature == QUADRATURE and last_interval == context(issued, identity),
                "wrong receipt declaration/runtime interval")
        _, previous = field(reader, n, nc, rank, size, topo)
        _, old_volumes = field(reader, n, 1, rank, size, topo)
        _, source = field(reader, n, nc, rank, size, topo)
        receipt = dict(point=issued, previous=previous, old_volumes=old_volumes, source=source,
                       old_nodes=faces(reader, topo, 1), flux=faces(reader, topo, nc),
                       density=faces(reader, topo, nc))
    reader.finish()
    return dict(identity=identity, frame=frame, clock=clock, generation=generation, topology=topo,
                state=state, volumes=volumes, nodes=nodes, sweeps=sweeps, receipt=receipt, ledger=ledger)


@dataclass
class Snapshot:
    state: np.ndarray
    nodes: np.ndarray
    volumes: np.ndarray
    time: float
    step: int
    generation: int
    images: tuple
    carriers: list
    temporal: dict
    hashes: dict


def checkpoint_envelope(checkpoint, case, pins, phase, time, step):
    manifest = strict_json(str(checkpoint["pops_checkpoint_manifest"].item()))
    common.identity_token(manifest["semantic_identity"], "semantic")
    common.identity_token(manifest["bind_identity"], "bind")
    require(common.identity_token(manifest["artifact_identity"], "artifact") == case["artifact"]
            and manifest["runtime_kind"] == "uniform"
            and common.cbor(manifest["clock"]) == common.cbor(dict(time=time.hex(), macro_step=step)), "checkpoint identity/clock differs")
    initial = phase in ("initial", "before")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == (2 if initial else 1),
            "wrong initial/run provenance version")
    if initial:
        require(time.hex() == "0x0.0p+0" and step == 0 and manifest["run_identity"] is None
                and common.cbor(manifest["origin"]) == common.cbor(dict(kind="bound_initial", schema_version=1)), "invented initial run")
    else:
        common.identity_token(manifest["run_identity"], "run")
        require("origin" not in manifest, "run invents initial origin")
    require(set(checkpoint) == set(manifest["arrays"]) | {"pops_checkpoint_manifest", "pops_restart_identity"},
            "checkpoint inventory differs from envelope")
    for name, expected in manifest["arrays"].items():
        array = np.ascontiguousarray(checkpoint[name]) if checkpoint[name].shape else checkpoint[name]
        header = common.cbor(dict(protocol="pops.array-evidence.v1", dtype=array.dtype.str, shape=list(array.shape)))
        require(expected == dict(dtype=array.dtype.str, shape=list(array.shape),
                                 content_sha256=digest(header + array.tobytes())), "typed array evidence differs: " + name)
    payload = {key: value for key, value in manifest.items() if key != "restart_identity"}
    restart = common.identity_token(manifest["restart_identity"], "restart")
    require(manifest["restart_identity"]["hexdigest"] == digest(common.cbor(dict(
        protocol="pops.identity", domain="restart", schema_version=1, payload=payload)))
        and str(checkpoint["pops_restart_identity"].item()) == restart, "restart envelope digest differs")
    require(str(checkpoint["abi_key"].item()) == pins["abi_key"], "checkpoint ABI differs")


def equal_array(a, b, label):
    require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), label)


def load_snapshot(base, pins, case, phase):
    row = case["phases"][phase]
    require(set(row) == {"receipt", "state", "checkpoint"}, "incomplete phase pins")
    files = {key: common.pinned_file(base, value) for key, value in row.items()}
    receipt = strict_json(files["receipt"][1])
    require(type(receipt["time"]) is float and all(type(receipt[key]) is int for key in (
        "macro_step", "generation", "dimension", "size", "rank")), "receipt scalar types differ")
    require(receipt["phase"] == phase and receipt["artifact"] == case["artifact"]
            and receipt["platform"] == case["platform"] and receipt["dimension"] == 1
            and receipt["size"] == pins["size"] and receipt["rank"] == 0
            and receipt["moving_identity"] == case["moving_identity"]
            and receipt["physical_frame"] == case["physical_frame"], "phase/declaration/platform differs")
    require(receipt["checkpoint_sha256"] == digest(files["checkpoint"][1])
            and Path(receipt["checkpoint"]).name == files["checkpoint"][0].name, "checkpoint receipt pin differs")
    data, checkpoint = common.archive(files["state"][1]), common.archive(files["checkpoint"][1])
    require(set(data) == {"state", "node_coordinates", "cell_volumes", "generation", "time", "step"},
            "saved ALE phase array inventory differs")
    state, nodes, volumes = data["state"], data["node_coordinates"], data["cell_volumes"]
    n, nc = case["cells"], len(case["components"])
    for value, shape in ((state, (nc, n)), (nodes, (n + 1, 1)), (volumes, (n,))):
        require(value.dtype == np.dtype("float64") and value.shape == shape and np.isfinite(value).all(),
                "saved physical shape/type/finitude differs")
    time, step = common.scalar(data["time"], "real"), common.scalar(data["step"], "int")
    generation = common.scalar(data["generation"], "int")
    require((time, step, generation) == (receipt["time"], receipt["macro_step"], receipt["generation"])
            and step >= 0 and generation == step, "phase lifecycle differs")
    equal_array(checkpoint["state_fluid"], state, "checkpoint physical state differs")
    require(common.scalar(checkpoint["t"], "real") == time
            and common.scalar(checkpoint["macro_step"], "int") == step, "checkpoint lifecycle differs")
    checkpoint_envelope(checkpoint, case, pins, phase, time, step)
    temporal = strict_json(str(checkpoint["temporal_restart_state"].item()))
    require(common.cbor(temporal["clock"]) == common.cbor(dict(time=time.hex(), macro_step=step)), "temporal accepted clock differs")
    schedule = temporal["program_schedule"]
    require(schedule["kind"] == "pops.temporal-program-schedule" and len(schedule["clocks"]) == 1
            and schedule["clocks"][0]["id"] == schedule["primary_clock"]
            and type(schedule["clocks"][0]["ticks_per_macro"]) is int
            and schedule["clocks"][0]["ticks_per_macro"] == 1, "wrong declared root clock schedule")
    wire, offsets = checkpoint["program_exchange_state"], checkpoint["program_exchange_offsets"]
    require(wire.dtype == np.dtype("uint8") and wire.ndim == 1 and offsets.dtype.kind in "iu"
            and offsets.ndim == 1 and len(offsets) == pins["size"] + 1, "wrong exact rank offsets")
    bounds = [int(value) for value in offsets]
    require(bounds[0] == 0 and bounds[-1] == wire.size
            and all(a < b for a, b in zip(bounds[:-1], bounds[1:], strict=True)), "rank image partition differs")
    images = tuple(wire[a:b].tobytes() for a, b in zip(bounds[:-1], bounds[1:], strict=True))
    require([digest(image) for image in images] == receipt["image_sha256"], "rank image receipt digests differ")
    carriers = [moving_image(image, case, rank, pins["size"]) for rank, image in enumerate(images)]
    for carrier in carriers:
        require(carrier["clock"] == schedule["primary_clock"] and carrier["generation"] == generation,
                "wire clock/generation differs from enclosing lifecycle")
        if carrier["receipt"]:
            issued = carrier["receipt"]["point"]
            require(issued["tick"] == step - 1 and issued["time"] + issued["dt"] == time,
                    "receipt end-time/macro-step mismatch")
        topo = carrier["topology"]
        for local, index in enumerate(topo["local"]):
            lo, hi, _ = topo["boxes"][index]
            equal_array(carrier["state"][local], state[:, lo:hi + 1], "wire physical state differs")
            equal_array(carrier["volumes"][local][0], volumes[lo:hi + 1], "wire physical volumes differ")
            equal_array(carrier["nodes"][local][0], nodes[lo:hi + 2, 0], "wire physical nodes differ")
    global_topology = [{key: value for key, value in carrier["topology"].items() if key not in ("rank", "local")}
                       for carrier in carriers]
    require(all(topo == global_topology[0] for topo in global_topology), "rank topology/ownership divergence")
    return Snapshot(state, nodes[:, 0], volumes, time, step, generation, images, carriers, temporal,
                    {key: digest(value[1]) for key, value in files.items()})


def initial(snapshot, case):
    require(snapshot.time.hex() == "0x0.0p+0" and snapshot.step == snapshot.generation == 0, "initial clock differs")
    require(np.array_equal(snapshot.nodes, np.arange(case["cells"] + 1, dtype=float) / case["cells"]),
            "initial projected endpoints differ")
    require(np.array_equal(snapshot.volumes, np.diff(snapshot.nodes)), "initial measure differs from endpoints")
    for component, name in enumerate(case["components"]):
        require(np.all(snapshot.state[component] == {"a": 2., "b": 3., "c": 4.}[name]), "initial projection differs")
    for carrier in snapshot.carriers:
        require(not carrier["ledger"]["records"] and carrier["receipt"] is None
                and all(np.all(values == 0.) for values in carrier["sweeps"]), "initial fabricated accepted interval")


def same_accepted(a, b, label):
    for name in ("state", "nodes", "volumes"):
        equal_array(getattr(a, name), getattr(b, name), label + " physical " + name)
    require(a.time.hex() == b.time.hex() and a.step == b.step and a.generation == b.generation
            and a.images == b.images, label + " clock/geometry/wire changed")
    for name in ("clock", "program_schedule", "synchronization_state", "history_cursors", "cache_generations"):
        require(common.cbor(a.temporal[name]) == common.cbor(b.temporal[name]), label + " accepted temporal " + name)


def small_error(actual, expected, label):
    scale = np.maximum(1., np.maximum(np.abs(actual), np.abs(expected)))
    error = np.abs(actual - expected)
    require(np.isfinite(error).all() and np.all(error <= 64 * EPS * scale), label)
    return float(np.max(error, initial=0.))


def scientific_output(base, pins, case, phase, snapshot):
    """Actual observer publication, independently read without NPZ.reopen/PoPS."""
    _, raw = common.pinned_file(base, case["outputs"][phase])
    data = common.archive(raw)
    manifest = strict_json(str(data["pops_output_manifest"].item()))
    require(manifest["format"] == "npz" and re.fullmatch(
        r"pops\.scientific-output\.v[1-9][0-9]*:sha256:[0-9a-f]{64}", manifest["output_identity"]),
        "scientific publication identity/format differs")
    require(set(data) == set(manifest["arrays"]) | {"pops_output_manifest"}, "scientific output inventory differs")
    for name, expected in manifest["arrays"].items():
        array = np.ascontiguousarray(data[name])
        header = array.dtype.str.encode() + b"\0" + ",".join(str(value) for value in array.shape).encode() + b"\0"
        require(expected == dict(dtype=array.dtype.str, shape=list(array.shape), content_sha256=digest(header + array.tobytes())),
                "scientific output typed array evidence differs")
    clock = manifest["snapshot"]["clock"]
    require(type(clock["macro_step"]) is int and clock["clock_id"] == snapshot.temporal["program_schedule"]["primary_clock"]
            and clock["time"] == snapshot.time.hex() and clock["macro_step"] == snapshot.step,
            "scientific output accepted clock differs")
    checkpoint = common.archive(common.pinned_file(base, case["phases"][phase]["checkpoint"])[1])
    checkpoint_manifest = strict_json(str(checkpoint["pops_checkpoint_manifest"].item()))
    provenance = manifest["snapshot"]["provenance"]
    require(provenance["bind_identity"] == common.identity_token(checkpoint_manifest["bind_identity"], "bind")
            and provenance["run_identity"] == common.identity_token(checkpoint_manifest["run_identity"], "run"),
            "scientific output binding/run provenance differs")
    datasets = manifest["datasets"]
    require(len(datasets["fields"]) == len(datasets["geometries"]) == 1, "wrong observer selection")
    geometry = next(iter(datasets["geometries"].values()))
    equal_array(data[geometry["node_coordinates"]], snapshot.nodes[:, None], "observer physical nodes differ from checkpoint")
    equal_array(data[geometry["cell_volumes"]], snapshot.volumes, "observer physical measures differ from checkpoint")
    require(not np.any(data[geometry["coverage"]]) and np.all(data[geometry["valid_cells"]]), "observer physical coverage differs")
    field = next(iter(datasets["fields"].values()))
    require(field["global_shape"] == [case["cells"]], "observer global field shape differs")
    pieces = sorted(field["pieces"], key=lambda piece: piece["lower"])
    expected_lower = 0
    for piece in pieces:
        lo, hi = piece["lower"], piece["upper"]
        require(len(lo) == len(hi) == 1 and lo[0] == expected_lower and lo[0] < hi[0] <= case["cells"]
                and type(piece["replicated"]) is bool and 0 <= piece["owner_rank"] < pins["size"],
                "observer piece ownership/tiling differs")
        expected_lower = hi[0]
        equal_array(data[piece["name"]], snapshot.state[:, lo[0]:hi[0]], "observer physical field differs from checkpoint")
    require(expected_lower == case["cells"], "observer omits physical cells")
    return dict(sha256=digest(raw), output_identity=manifest["output_identity"], step=snapshot.step, time=snapshot.time)


def step(before, after, case):
    require(after.time == before.time + DT and after.step == before.step + 1, "issued duration or step changed")
    displacement = after.nodes - before.nodes
    equal_array(after.volumes, np.diff(after.nodes), "accepted endpoint measure identity differs")
    require(np.all(after.volumes > 0), "inverted accepted geometry")
    gcl = (after.volumes - before.volumes) - np.diff(displacement)
    require(np.all(np.abs(gcl) <= GEOMETRY_TOL), "saved physical GCL failed")
    reference = np.arange(case["cells"] + 1, dtype=float) / case["cells"]
    expected_nodes = reference + .08 * np.sin(6.283185307179586 * reference) * after.time
    law_error = small_error(after.nodes, expected_nodes, "authored coordinate law differs from saved nodes")
    # Original state/flux/source at left endpoint, periodic physical traces.
    left, right = before.state[:, np.arange(case["cells"] + 1) - 1], before.state[:, np.arange(case["cells"] + 1) % case["cells"]]
    speed = case["velocity"] - displacement / DT
    density = .5 * left + .5 * right
    relative_density = .5 * speed * (left + right) - .5 * np.abs(speed) * (right - left)
    physical = DT * relative_density + density * displacement
    source = DT * (SOURCE_COEFFICIENT if case["source"] else 0.) * before.state * before.volumes
    relative = physical - density * displacement
    expected = (before.state * before.volumes - np.diff(relative, axis=1) + source) / after.volumes
    field_error = small_error(after.state, expected, "original Reynolds/Rusanov/source equation mismatch")
    residual = (after.state * after.volumes - before.state * before.volumes) + np.diff(relative, axis=1) - source
    reynolds_error = small_error(residual, np.zeros_like(residual), "independent Reynolds residual failed")
    record_total, face_error, source_error = 0, 0., 0.
    replicated = after.carriers[0]["topology"]["replicated"]
    for rank, carrier in enumerate(after.carriers):
        receipt, topo = carrier["receipt"], carrier["topology"]
        require(receipt is not None and receipt["point"]["dt"] == DT
                and receipt["point"]["time"] == before.time and receipt["point"]["tick"] == before.step,
                "wrong exact issued receipt interval")
        frame = context(receipt["point"], case["moving_identity"])
        expected_records = {}
        for local, index in enumerate(topo["local"]):
            lo, hi, _ = topo["boxes"][index]
            cell_slice, face_slice = slice(lo, hi + 1), slice(lo, hi + 2)
            equal_array(receipt["previous"][local], before.state[:, cell_slice], "receipt previous physical state differs")
            equal_array(receipt["old_volumes"][local][0], before.volumes[cell_slice], "receipt previous volumes differ")
            equal_array(receipt["old_nodes"][local][0], before.nodes[face_slice], "receipt previous nodes differ")
            equal_array(carrier["sweeps"][local][0], displacement[face_slice], "receipt sweep is not actual endpoint displacement")
            equal_array(receipt["density"][local], density[:, face_slice], "centered physical trace differs")
            face_error = max(face_error, small_error(receipt["flux"][local], physical[:, face_slice], "physical Rusanov face amount differs"))
            source_error = max(source_error, small_error(receipt["source"][local], source[:, cell_slice], "original source evaluated with wrong measure/state/duration"))
            for cell in range(lo, hi + 1):
                for component in range(len(case["components"])):
                    key = ("source:" + case["moving_identity"] + "/component:" + str(component), "cell:" + str(cell))
                    expected_records[key] = (1, receipt["source"][local][component, cell - lo])
                for side in (0, 1):
                    occurrence = "cell:" + str(cell) + "/side:" + str(side)
                    expected_records[("geometry:" + case["moving_identity"], occurrence)] = (
                        -1 if side == 0 else 1, displacement[cell + side])
                    for component in range(len(case["components"])):
                        key = ("amount:" + case["moving_identity"] + "/component:" + str(component), occurrence)
                        amount = receipt["flux"][local][component, cell - lo + side] - density[component, cell + side] * displacement[cell + side]
                        expected_records[key] = (-1 if side == 0 else 1, amount)
        records = carrier["ledger"]["records"]
        require(len(records) == len(expected_records), "missing/extra rank-local accepted exchange")
        for record in records:
            key = (record["operation"], record["occurrence"])
            require(key in expected_records, "foreign physical exchange occurrence")
            orientation, amount = expected_records.pop(key)
            require(record["context"] == frame and record["evaluation"] == case["moving_identity"]
                    and record["quadrature"] == QUADRATURE and record["orientation"] == orientation
                    and record["measure"] == record["weight"] == 1. and record["multiplicity"] == 1
                    and record["flux"] == float(amount) and not record["exterior"]
                    and (record["axis"], record["side"], record["component"]) == (-1, -1, -1),
                    "accepted exchange support/amount/duration authority differs")
        if not replicated or rank == 0:
            record_total += len(records)
    require(record_total == case["cells"] * (2 + 3 * len(case["components"])), "global incidence coverage differs")
    if replicated:
        require(all(carrier["ledger"] == after.carriers[0]["ledger"] for carrier in after.carriers),
                "replicated ledger copies differ")
    inventory = np.array([math.fsum(float(x) for x in row) for row in after.state * after.volumes])
    initial_values = np.array([{"a": 2., "b": 3., "c": 4.}[name] for name in case["components"]])
    inventory_error = small_error(inventory, initial_values * (1 + SOURCE_COEFFICIENT * DT if case["source"] else 1) ** after.step,
                                  "global source/inventory balance failed")
    return dict(time=after.time, step=after.step, field_max_abs=field_error, gcl_max_abs=float(np.max(np.abs(gcl))),
                reynolds_max_abs=reynolds_error, face_max_abs=face_error, source_max_abs=source_error,
                law_max_abs=law_error, inventory_max_abs=inventory_error, inventory=inventory.tolist(),
                incidence_count=record_total, replicated=replicated)


def receive(path):
    base, raw = path.resolve().parent, path.read_bytes()
    pins = strict_json(raw)
    require(pins["schema"] == SCHEMA and type(pins["dimension"]) is int and pins["dimension"] == 1
            and type(pins["size"]) is int and pins["size"] > 0,
            "wrong ALE pin schema/rank/dimension")
    require(re.fullmatch("[0-9a-f]{40}", pins["source_commit"]) and re.fullmatch("[0-9a-f]{64}", pins["native_sha256"])
            and "dim=1" in pins["abi_key"], "missing source/native/SDK pins")
    _, identity_raw = common.pinned_file(base, pins["identity_file"])
    identity = strict_json(identity_raw)
    require(all(identity[key] == pins[key] for key in ("source_commit", "native_sha256", "abi_key")), "owner runtime identity differs")
    expected = {(kind, n, names, source) for kind, n, names, source in campaign()}
    labels = [(case["kind"], case["cells"], tuple(case["components"]), case["source"]) for case in pins["cases"]]
    require(len(labels) == len(expected) and set(labels) == expected, "incomplete/duplicate witness campaign")
    results, fresh = [], {}
    for case in pins["cases"]:
        require(type(case["source"]) is bool and case["velocity"] == (-.3 if case["components"] == ["a", "b", "c"] else .7)
                and set(case["phases"]) == set(PHASES[case["kind"]]), "wrong original model/phase contract")
        snapshots = {name: load_snapshot(base, pins, case, name) for name in PHASES[case["kind"]]}
        output_phases = ("step1", "accepted", "continuous", "replayed") if case["kind"] == "restart" else ("retried",)
        require(set(case["outputs"]) == set(output_phases), "missing/extra accepted scientific publication")
        outputs = {name: scientific_output(base, pins, case, name, snapshots[name]) for name in output_phases}
        first = snapshots["initial" if case["kind"] == "restart" else "before"]
        initial(first, case)
        if case["kind"] == "restart":
            steps = [step(first, snapshots["step1"], case), step(snapshots["step1"], snapshots["accepted"], case),
                     step(snapshots["accepted"], snapshots["continuous"], case)]
            same_accepted(snapshots["accepted"], snapshots["restored"], "restart restore")
            same_accepted(snapshots["continuous"], snapshots["replayed"], "restart replay")
            steps.append(step(snapshots["restored"], snapshots["replayed"], case))
            if case["components"] == ["c", "a", "b"] and case["source"]:
                fresh[case["cells"]] = snapshots["step1"]
        else:
            same_accepted(first, snapshots["rejected"], "rejected attempt")
            steps = [step(first, snapshots["retried"], case)]
        results.append(dict(kind=case["kind"], cells=case["cells"], components=case["components"], source=case["source"],
                            steps=steps, scientific_outputs=outputs,
                            file_sha256={name: saved.hashes for name, saved in snapshots.items()}))
    for case in pins["cases"]:
        if case["kind"] == "retry":
            retried = load_snapshot(base, pins, case, "retried")
            for name in ("state", "nodes", "volumes"):
                equal_array(getattr(fresh[case["cells"]], name), getattr(retried, name), "safe retry differs from fresh public step")
    return dict(schema="sol61.moving-interval.offline-reception@1", scientific_status="PASS", source_contract=SOURCE_CONTRACT,
                source_commit=pins["source_commit"], native_sha256=pins["native_sha256"], abi_key=pins["abi_key"],
                pins_sha256=digest(raw), dimension=1, ranks=pins["size"], cases=results,
                limitation="Offline real saved-state evidence only; no new native/MPI/AMR/GPU execution.")


def campaign():
    return [("restart", n, names, source) for n in (16, 32) for names in (("a",), ("a", "b", "c"), ("c", "a", "b"))
            for source in (False, True)] + [("retry", n, ("c", "a", "b"), True) for n in (16, 32)]


def contract():
    return dict(schema=SCHEMA, scientific_status="pending_receipts", source_contract=SOURCE_CONTRACT,
                consumes_no_pops_package=True, expected_cases=[dict(kind=kind, cells=n, components=names, source=source,
                phases=PHASES[kind]) for kind, n, names, source in campaign()],
                required_external_pins=["source_commit", "native_sha256", "abi_key", "identity_file", "dimension", "size"],
                case_pins=["artifact", "platform", "moving_identity", "physical_frame", "kind", "cells", "components", "source", "velocity", "phases", "outputs"],
                output_pins="step1/accepted/continuous/replayed for restart; retried for retry; each external path+SHA256",
                phase_pins={key: dict(path="actual file path", sha256="external owner SHA256") for key in ("state", "receipt", "checkpoint")},
                equations=["V=diff(saved physical nodes)", "sweep=accepted_nodes-previous_nodes", "GCL=deltaV-diff(sweep)",
                           "Frel=.5*(a-wg)*(UL+UR)-.5*abs(a-wg)*(UR-UL)", "Fphysical_amount=h*Frel+centered_density*sweep",
                           "Qnew=Qold-diff(Fphysical_amount-density*sweep)+h*Vold*.05*Uold"],
                geometry_tolerance=GEOMETRY_TOL, arithmetic_tolerance="64 binary64 eps * max(1,abs(actual),abs(expected))",
                witness_scope="Uniform1D provider, not a production dimensional restriction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--describe-contract", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(args.describe_contract or args.pins is not None, "external owner pins are required")
    result = contract() if args.describe_contract else receive(args.pins)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
