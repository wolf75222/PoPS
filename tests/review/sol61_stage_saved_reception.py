"""Offline original Stage equations and ROOT-sealed evidence, no PoPS import."""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import copy
from pathlib import Path
import re
import shlex
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


files = module("stage_file_protocol", "sol61_captured_diffusion_saved_reception.py")
lineage = module("stage_independent_lineage", "sol61_stage_checkpoint_lineage.py")
wire, protocol = files.wire, files.protocol
need, exact, strict_json = files.need, files.exact, files.strict_json
PHASES = ("accepted", "reloaded", "continuous", "replay")
CP = ("accepted", "continuous", "replay")
CASES = {(n, dt, w) for n in (8, 16) for dt in (.01, .02) for w in (1, 2)}
TOL = 3e-8
QUALIFICATION = "uniform-original-stage-saved-state-equations@1"
CONTROLS = dict(tolerance=1e-10, max_iterations=20, linear_tolerance=1e-8,
                linear_max_iterations=240, restart=60, armijo=1e-4, minimum_step=1/1024)
RECEIPT_KEYS = ("fixture_schema", "artifact", "dimension", "rank", "size", "cells", "dt", "evolved_states",
                "unknowns", "candidate_diffusion", "newton", "fd_step", "acceptance", "projection", "volume_sum",
                "native", "platform", "binaries", "sources", "program_irs", "initial", "phases", "checkpoints",
                "checkpoint_equivalence", "exact_restart_and_replay")


def key(n, dt, width):
    return "N%d-dt%s-width%d" % (n, dt.hex(), width)


def q_of(t):
    if len(t) == 1:
        return t + t * t
    return np.stack((t[0]+t[0]*t[0]+float(Fraction(1, 10))*t[1]*t[1],
                     t[1]+t[1]*t[1]+float(Fraction(1, 5))*t[0]*t[1]))


def spatial(t, candidate, *, transposed=False, harmonic=False):
    """Each oriented periodic face once; equal/opposite physical FV updates."""
    w, ny, nx = t.shape
    d = np.array([[float(Fraction(3, 250)), float(Fraction(1, 500))],
                  [float(Fraction(-1, 1000)), float(Fraction(7, 500))]])[:w, :w]
    if transposed:
        d = d.T
    output = np.zeros_like(t)
    nonzero_faces = [0., 0.]
    for y in range(ny):
        for x in range(nx):
            for axis, (yy, xx, n) in enumerate((((y+1) % ny, x, ny), (y, (x+1) % nx, nx))):
                a = 1 + (float(Fraction(2, 5))*t[:, y, x]**2 if candidate else np.zeros(w))
                b = 1 + (float(Fraction(2, 5))*t[:, yy, xx]**2 if candidate else np.zeros(w))
                mean = 2*a*b/(a+b) if harmonic else (a+b)/2
                contribution = d @ (mean*(t[:, yy, xx]-t[:, y, x]))*n**2
                output[:, y, x] += contribution
                output[:, yy, xx] -= contribution
                nonzero_faces[axis] = max(nonzero_faces[axis], float(np.max(np.abs(contribution))))
    return output, nonzero_faces


def recipe(n, w):
    xx, yy = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n, indexing="xy")
    initial = np.stack([.2+.03*np.sin(2*np.pi*xx)+.02*np.cos(2*np.pi*yy)+.07*i for i in range(w)])
    target = np.stack([.21+.025*np.cos(2*np.pi*(i+1)*xx)+.018*np.sin(2*np.pi*yy)+.07*i for i in range(w)])
    return initial, target


def array(value, shape, name):
    need(type(value) is np.ndarray and value.dtype == np.dtype("float64")
         and value.shape == shape and np.isfinite(value).all(), "invalid finite binary64 array: " + name)


def science(initial, states, n, dt, w):
    need((n, dt, w) in CASES, "foreign witness")
    qnames = ["Q%d" % i for i in range(w)]
    exact(initial, (*qnames, "forcing", "initial_temperature", "first_target", "cell_volumes"), "initial NPZ")
    expected_initial, target = recipe(n, w)
    for name in ("initial_temperature", "first_target", "forcing"):
        array(initial[name], (w, n, n), name)
    array(initial["cell_volumes"], (n, n), "volumes")
    need(np.max(np.abs(initial["initial_temperature"]-expected_initial)) <= 2e-14
         and np.max(np.abs(initial["first_target"]-target)) <= 2e-14, "declared initial/target centre samples differ")
    need(np.array_equal(initial["cell_volumes"], np.full((n, n), 1/(n*n))), "cell volumes differ")
    q0 = q_of(expected_initial)
    for i, name in enumerate(qnames):
        array(initial[name], (1, n, n), name)
        need(np.max(np.abs(initial[name][0]-q0[i])) <= 2e-14, "initial Q differs")
    forcing = (q_of(target)-q0)/dt-spatial(target, w == 2)[0]
    need(np.max(np.abs(forcing-initial["forcing"])) <= 2e-12, "original prescribed forcing differs")
    exact(states, PHASES, "state phase inventory")
    reports = {}
    for phase in PHASES:
        saved = states[phase]
        names = ["T%d" % i for i in range(w)] + (["z"] if w == 2 else [])
        exact(saved, (*qnames, "forcing", *names, *(name+"-previous" for name in names), "time", "step"), "phase NPZ")
        step = 1 if phase in ("accepted", "reloaded") else 2
        need(protocol.scalar(saved["time"], "real").hex() == (step*dt).hex()
             and protocol.scalar(saved["step"], "int") == step, "phase exact clock differs")
        for name in names:
            array(saved[name], (n, n), name)
            array(saved[name+"-previous"], (n, n), name+"-previous")
        array(saved["forcing"], (w, n, n), "forcing")
        files.same(saved["forcing"], initial["forcing"], "readonly forcing")
        t = np.stack([saved[name] for name in names[:w]])
        q = np.concatenate([saved[name] for name in qnames])
        for name in qnames:
            array(saved[name], (1, n, n), name)
        previous = q0 if step == 1 else np.concatenate([states["accepted"][name] for name in qnames])
        action, faces = spatial(t, w == 2)
        residual = float(np.linalg.norm(q-dt*(action+forcing)-previous)
                         / max(float(np.linalg.norm(previous)), float(np.linalg.norm(dt*forcing)), 1e-300))
        projection = float(np.max(np.abs(q-q_of(t))))
        balance = float(np.max(np.abs(np.sum((q-previous-dt*forcing)/(n*n), axis=(1, 2)))))
        need(residual <= TOL, "original F=Q-tauR-Qn exceeds fixed tolerance")
        need(projection <= TOL, "original accumulation projection differs")
        need(balance <= TOL, "global amount law differs")
        need(float(np.max(np.abs(action))) > .005 and all(face > 0 for face in faces), "nonconstant diffusion absent")
        if step == 1:
            need(np.max(np.abs(t-target)) <= TOL, "first physical target differs")
        if w == 2:
            need(np.max(np.abs(saved["z"]-.25*t[0]-.5*t[1])) <= TOL, "auxiliary constraint differs")
        for name in names:
            files.same(saved[name+"-previous"], states["accepted"][name], "bootstrap/previous physical history " + name)
        reports[phase] = dict(original_relative_l2=residual, Q_projection_linf=projection, global_balance_linf=balance)
    for first, second in (("accepted", "reloaded"), ("continuous", "replay")):
        for name in states[first]:
            files.same(states[first][name], states[second][name], "exact phase replay " + name)
    return reports


def npz(row, roots):
    return protocol.archive(files.pinned(row, roots)[1])


def case_inventory(directory):
    directory = wire.canonical(directory)
    receipt = strict_json(files.read(directory / "receipt.json")[1])
    exact(receipt, RECEIPT_KEYS, "fixture receipt")
    local = {directory / "receipt.json"}
    result = dict(directory=str(directory), receipt=files.leaf(directory / "receipt.json"), artifact=receipt["artifact"],
                  witness=dict(cells=receipt["cells"], dt=receipt["dt"], width=receipt["evolved_states"]))
    for label, rows in (("initial", {"initial": receipt["initial"]}),
                        ("phases", receipt["phases"]), ("checkpoints", receipt["checkpoints"])):
        checked = {}
        for phase, row in rows.items():
            record = files.leaf(row["path"])
            need(record["sha256"] == row["sha256"], "receipt leaf digest differs")
            checked[phase] = record
            local.add(Path(record["path"]))
        result[label] = checked["initial"] if label == "initial" else checked
    exact(result["phases"], PHASES, "receipt phases")
    exact(result["checkpoints"], CP, "receipt checkpoints")
    for label in ("sources", "program_irs"):
        need(type(receipt[label]) is list and len(receipt[label]) == 1, "one compiler Program CPP/IR required")
        row = receipt[label][0]
        record = files.leaf(row["path"])
        need(record["sha256"] == row["sha256"], "Program CPP/IR digest differs")
        local.add(Path(record["path"]))
        result[label] = [record]
    need(len(local) == 11 and all(p.parent == directory for p in local)
         and set(directory.iterdir()) == local, "closed Stage case inventory differs")
    result["components"] = receipt["binaries"]
    return result


def checkpoint(payload, saved, accepted, n, dt, w, abi, artifact, program_hash):
    manifest = lineage.authenticate(payload)
    need(manifest["artifact_identity"]["hexdigest"] == artifact.split(":")[-1], "checkpoint artifact differs")
    need(payload["abi_key"].item() == abi and payload["program_hash"].item() == program_hash, "ABI/Program association differs")
    step = int(saved["step"])
    need(manifest["clock"] == dict(time=(dt*step).hex(), macro_step=step), "checkpoint clock differs")
    geometry = strict_json(payload["pops_spatial_contract"].item())
    need(geometry["dimension"] == 2 and geometry["shape"] == [n, n]
         and geometry["lower"] == [0..hex()]*2 and geometry["upper"] == [1..hex()]*2
         and geometry["periodicity"] == [True, True] and geometry["refinement_ratios"] == [], "checkpoint geometry differs")
    need(geometry["identity"] == "pops.checkpoint-spatial-layout.v1:sha256:" + lineage.digest("checkpoint-spatial-layout",
         {k: v for k, v in geometry.items() if k != "identity"}), "geometry identity differs")
    names = ["T%d" % i for i in range(w)] + (["z"] if w == 2 else [])
    blocks = ["Q%d" % i for i in range(w)] + ["forcing"]
    need(list(payload["blocks"]) == blocks and list(payload["history_names"]) == names, "block/history ordering differs")
    for block in blocks:
        files.same(payload["state_"+block], saved[block], "checkpoint state " + block)
        need(list(payload["names_"+block]) == (["amount"] if block != "forcing" else ["f%d" % i for i in range(w)]), "component names differ")
    for name in names:
        for prefix in ("history_depth_", "history_ncomp_", "history_fill_count_"):
            need(payload[prefix+name].dtype == np.dtype("int64") and payload[prefix+name].shape == (), "history typed metadata differs")
        need(payload["history_init_"+name].dtype == np.dtype("bool") and payload["history_init_"+name].shape == (), "history typed initialization differs")
        need(payload["history_storage_mode_"+name].item() == "policy"
             and strict_json(payload["history_policy_"+name].item()) == dict(kind="history-persistence", payload=dict(policy="dense"),
                 protocol="pops.manifest", schema_version=1), "history persistence contract differs")
        need(payload["history_depth_"+name].item() == 2 and payload["history_ncomp_"+name].item() == 1
             and payload["history_init_"+name].item() is True
             and payload["history_fill_count_"+name].item() == min(step, 2), "history ring/fill differs")
        for prefix in ("history_stored_slots_", "history_requested_stored_slots_"):
            files.same(payload[prefix+name], np.array([0, 1], dtype="int64"), "history slots")
        files.same(payload["history_slot_dt_"+name], np.array([dt, dt]), "history duration")
        expected = b"POPSHID1"+struct.pack("<Q", len(name))+name.encode()+struct.pack("<qQ", -1, 2)
        for start in (0., (step-1)*dt):
            expected += struct.pack("<QQQQ", 2, int.from_bytes(struct.pack("<d", start), "little"),
                                    int.from_bytes(struct.pack("<d", dt), "little"), 1)
        files.same(payload["history_sample_identity_"+name], np.frombuffer(expected, dtype="uint8"), "history exact sample point")
        for slot, value in ((0, accepted[name]), (1, saved[name])):
            array(payload["history_"+name+"_%d" % slot], (1, n, n), "checkpoint history slot")
            files.same(payload["history_"+name+"_%d" % slot][0], value, "history field slot")
    temporal = strict_json(payload["temporal_restart_state"].item())
    need(temporal["clock"] == manifest["clock"] and temporal["status"] == "accepted"
         and temporal["synchronized"] is True
         and temporal["controller_state"]["last_accepted_dt"] == dt.hex(), "temporal accepted window differs")


def receipt_provenance(receipt, cps, dt):
    equivalence = receipt["checkpoint_equivalence"]
    exact(equivalence, ("contract", "exact_payload_and_manifest", "provenance"), "checkpoint equivalence")
    need(equivalence["contract"] == "pops.evolved-stage-checkpoint-equivalence@1"
         and equivalence["exact_payload_and_manifest"] is True, "checkpoint equivalence contract differs")
    exact(equivalence["provenance"], CP, "checkpoint provenance phases")
    first = lineage.authenticate(cps["accepted"])
    def token(row):
        return "pops.%s.v%d:sha256:%s" % (row["domain"], row["schema_version"], row["hexdigest"])
    for phase in CP:
        manifest = lineage.authenticate(cps[phase])
        row = equivalence["provenance"][phase]
        exact(row, ("semantic", "artifact", "bind", "run", "run_manifest", "last_restart", "restart", "clock"), "live checkpoint provenance")
        for domain in ("semantic", "artifact", "bind", "run", "restart"):
            need(row[domain] == token(manifest[domain+"_identity"]), "captured runtime identity differs")
        need(row["clock"] == manifest["clock"] and row["last_restart"] == (token(first["restart_identity"]) if phase == "replay" else None),
             "captured runtime clock/restart differs")
        strategy = strict_json(cps[phase]["temporal_restart_state"].item())["strategy"]
        expected = dict(protocol="pops.manifest", kind="run", schema_version=3, payload=dict(
            bind_identity=row["bind"], continuation_identity=token(first["run_identity"]) if phase == "replay" else None,
            start_time=0. if phase == "accepted" else dt, start_macro_step=0 if phase == "accepted" else 1,
            controls=dict(t_end=dt if phase == "accepted" else 2*dt, step_transaction=strategy,
                          max_steps=1, output_mode="current-directory"), run_identity=row["run"]))
        need(row["run_manifest"] == expected, "captured RunManifest request/lineage differs")


def component(binary, roots):
    exact(binary, ("component", "path", "sha256", "sidecar", "compile_command"), "binary component")
    raw = files.pinned({k: binary[k] for k in ("path", "sha256")}, roots, budget=files.MAX_BINARY_BYTES)[1]
    sidecar = strict_json(files.pinned(binary["sidecar"], roots)[1])
    exact(sidecar, ("protocol", "semantic_identity", "artifact_spec_identity", "binary_identity", "artifact_identity"), "DSO sidecar")
    need(sidecar["protocol"] == "pops.artifact-sidecar.v1", "sidecar protocol differs")
    def reference(text, domain):
        match = re.fullmatch(r"pops\."+domain+r"\.v1:sha256:([0-9a-f]{64})", text)
        need(match is not None, "sidecar identity domain differs")
        return dict(domain=domain, schema_version=1, algorithm="sha256", digest=bytes.fromhex(match[1]))
    expected_binary = lineage.digest("binary", dict(algorithm="sha256", content_digest=bytes.fromhex(files.digest(raw)), size=len(raw)))
    need(sidecar["binary_identity"] == "pops.binary.v1:sha256:"+expected_binary, "sidecar binary content identity differs")
    expected_artifact = lineage.digest("artifact", dict(spec=reference(sidecar["artifact_spec_identity"], "artifact-spec"),
                                                        binary=reference(sidecar["binary_identity"], "binary")))
    need(sidecar["artifact_identity"] == "pops.artifact.v1:sha256:"+expected_artifact, "component artifact digest differs")
    return sidecar


def control_scalar(row):
    if "scalar" in row:
        exact(row, ("scalar",), "integer control")
        row = row["scalar"]
    exact(row, ("kind", "value"), "control scalar")
    need(row["kind"] in ("binary64", "integer"), "foreign control kind")
    return float.fromhex(row["value"]) if row["kind"] == "binary64" else int(row["value"])


def add(a, b):
    result = dict(a)
    for term, coefficient in b.items():
        result[term] = result.get(term, Fraction(0))+coefficient
    return {k: v for k, v in result.items() if v}


def multiply(a, b):
    result = {}
    for left, ca in a.items():
        for right, cb in b.items():
            powers = dict(left)
            for name, degree in right:
                powers[name] = powers.get(name, 0)+degree
            term = tuple(sorted(powers.items()))
            result[term] = result.get(term, Fraction(0))+ca*cb
    return {k: v for k, v in result.items() if v}


def constant(value):
    return {(): Fraction(value)} if value else {}


def symbol(name):
    return {((name, 1),): Fraction(1)}


def polynomial(node, names, capture_names, clock):
    """Exact binary64/rational coefficient algebra; no sampling of IR bodies."""
    op = node[0]
    if op == "literal":
        row = node[1]
        if row["kind"] == "binary64":
            return constant(float.fromhex(row["value"]))
        if row["kind"] == "rational":
            return constant(Fraction(int(row["numerator"]), int(row["denominator"])))
        need(row["kind"] == "integer", "foreign polynomial literal")
        return constant(int(row["value"]))
    if op == "unknown":
        need(type(node[1]) is int and node[2]["local_id"] == names[node[1]], "unknown ordering differs")
        return symbol(node[2]["local_id"])
    if op == "input":
        need(type(node[1]) is int and type(node[2]) is int
             and node[3]["local_id"] == capture_names[node[1]], "capture slot/port differs")
        return symbol(node[3]["local_id"]+":"+str(node[2]))
    if op == "temporal_tau":
        row = node[1]
        need(row["kind"] == "issued_frame_duration" and row["window_authority"] == "native_issued_cadence_frame@1"
             and row["factor"] == {"kind": "integer", "value": "1"}
             and row["point"] == dict(schema_version=1, clock=clock, step=1, offset={"kind": "integer", "value": "0"})
             and row["program_owner"] == clock["owner"], "native tau Program/window point differs")
        return symbol("tau")
    left = polynomial(node[1], names, capture_names, clock)
    if op == "neg":
        return multiply(constant(-1), left)
    right = polynomial(node[2], names, capture_names, clock)
    if op == "add":
        return add(left, right)
    if op == "sub":
        return add(left, multiply(constant(-1), right))
    if op == "mul":
        return multiply(left, right)
    if op == "div":
        need(set(right) == {()} and right[()] != 0, "nonconstant polynomial divisor")
        return multiply(left, constant(1/right[()]))
    need(op == "pow" and set(right) == {()} and right[()] in (1, 2), "foreign closed-witness polynomial operation")
    return left if right[()] == 1 else multiply(left, left)


def ir_physics(ir, w, dt):
    nodes = ir["nodes"]
    need(__debug__, "optimized Python disables independent checkpoint assertions")
    need([node["id"] for node in nodes] == list(range(len(nodes))), "IR node ownership/order differs")
    need(ir["version"] == (14 if w == 1 else 15), "Stage IR version differs")
    solve = [n for n in nodes if n["op"] == "solve_spatial_field"]
    coefficients = [n for n in nodes if n["op"] == "field_problem_coefficients"]
    projections = [n for n in nodes if n["op"] == "field_evolved_state"]
    need(len(solve) == len(coefficients) == 1 and len(projections) == w, "Stage operations differ")
    attrs = solve[0]["attrs"]
    need(attrs["coefficient_face_policy"] == "pops.field.face-mean.arithmetic@1" and attrs["physical_boundary"] == "periodic"
         and {k: control_scalar(v) for k, v in attrs["newton_controls"].items()} == CONTROLS
         and float.fromhex(attrs["finite_difference_step"]["value"]) == 1e-6,
         "original method/controls differ")
    observed = [n["attrs"]["field_unknown"]["local_id"] for n in nodes if n["op"] == "field_component"]
    names = ["T%d" % i for i in range(w)] + (["z"] if w == 2 else [])
    handles = attrs["source_contract"]["unknown_components"]
    need([row["local_id"] for row in handles] == names and len({row["qualified_id"] for row in handles}) == len(names),
         "original unknown authorities differ")
    for i, row in enumerate(handles):
        need(row["qualified_id"] == f"pops.handle.v1::case:evolved-original-stage-MMS/descriptor:field%3Aoriginal-evolution/descriptor:unknown%3A{i}::field::{names[i]}",
             "foreign unknown owner")
    for node in nodes:
        need(node["point"]["clock"] == ir["clock"] and node["point"]["offset"] == {"kind": "integer", "value": "0"}
             and node["point"]["step"] == (0 if node["op"] == "state" else 1), "IR exact clock/read point differs")
    need(observed == names, "field/history unknown order differs")
    need([n["state"]["local_id"] for n in projections] == ["Q%d" % i for i in range(w)], "distinct Q port ordering differs")
    need(len({n["state"]["qualified_id"] for n in projections}) == w, "Q ports alias")
    state_nodes = {n["id"]: n for n in nodes if n["op"] == "state"}
    capture_names = [state_nodes[i]["state"]["local_id"] for i in solve[0]["inputs"][2:]]
    need(attrs["capture_count"] == w+1 and attrs["seed_index"] is None
         and len(capture_names) == len(set(capture_names))
         and set(capture_names) == {"Q%d" % i for i in range(w)} | {"forcing"},
         "original capture/seed partition differs")
    for state in state_nodes.values():
        need(state["point"]["step"] == 0, "capture is not an authentic n-read")
        need(state["attrs"]["state"]["handle"] == state["state"]
             and state["state"]["block_ref"]["local_id"] == state["state"]["local_id"], "capture declaration authority differs")
        port = state["state"]
        need(port["block_ref"]["qualified_id"] == "pops.handle.v1::case:evolved-original-stage-MMS::block::"+port["local_id"]
             and port["block_ref"]["owner_path"]["nodes"] == [{"kind": "case", "name": "evolved-original-stage-MMS"}]
             and port["owner_path"]["nodes"][:2] == [{"kind": "case", "name": "evolved-original-stage-MMS"},
                                                        {"kind": "block", "name": port["local_id"]}],
             "foreign capture/block owner")
    capture_handles = [state_nodes[i]["state"] for i in solve[0]["inputs"][2:]]
    def closure(row):
        if type(row) is list:
            if row and row[0] == "unknown":
                need(row[2] == handles[row[1]], "IR unknown handle closure differs")
            if row and row[0] == "input":
                need(row[3] == capture_handles[row[1]], "IR capture handle closure differs")
            for child in row:
                closure(child)
    for encoded in [attrs["local_expressions"], coefficients[0]["attrs"]["expressions"],
                    *(projection["attrs"]["expressions"] for projection in projections)]:
        closure(encoded)
    def parsed(row):
        return polynomial(row, names, capture_names, ir["clock"])
    ts = [symbol(name) for name in names[:w]]
    accumulation = [add(t, multiply(t, t)) for t in ts]
    if w == 2:
        accumulation[0] = add(accumulation[0], multiply(constant(.1), multiply(ts[1], ts[1])))
        accumulation[1] = add(accumulation[1], multiply(constant(.2), multiply(ts[0], ts[1])))
    expected = [add(accumulation[i], multiply(constant(-1), add(symbol("Q%d:0" % i),
                multiply(symbol("tau"), symbol("forcing:%d" % i))))) for i in range(w)]
    if w == 2:
        expected.append(add(symbol("z"), multiply(constant(-1), add(multiply(constant(.25), ts[0]),
                                                                 multiply(constant(.5), ts[1])))))
    need([parsed(row) for row in attrs["local_expressions"]] == expected, "IR exact original local residual differs")
    d = [[.012, .002], [-.001, .014]]
    matrix = []
    for i in range(len(names)):
        for j in range(len(names)):
            coefficient = {} if i >= w or j >= w else multiply(symbol("tau"), constant(d[i][j]))
            if i < w and j < w and w == 2:
                coefficient = multiply(coefficient, add(constant(1), multiply(constant(.4), multiply(ts[j], ts[j]))))
            matrix.append(coefficient)
    need([parsed(row) for row in coefficients[0]["attrs"]["expressions"]] == matrix,
         "IR exact signed per-candidate diffusion differs")
    commits = ir["commits"]
    need(len(commits) == w+1, "IR complete commit partition differs")
    for i, projection in enumerate(projections):
        need(commits[i] == dict(block=projection["state"]["block_ref"], state=projection["state"], value=projection["id"]),
             "IR Q publication target differs")
        need(len(projection["attrs"]["expressions"]) == 1
             and parsed(projection["attrs"]["expressions"][0]) == accumulation[i]
             and projection["attrs"]["previous_capture_index"] == capture_names.index("Q%d" % i),
             "IR exact original Q projection differs")

    forcing = next(state for state in state_nodes.values() if state["state"]["local_id"] == "forcing")
    final = nodes[-1]
    need(final["op"] == "linear_combine" and final["inputs"] == [forcing["id"]]
         and commits[-1] == dict(block=forcing["state"]["block_ref"], state=forcing["state"], value=final["id"]),
         "IR forcing commit ownership differs")
    observations = [node for node in nodes if node["op"] == "field_component"]
    stores = [node for node in nodes if node["op"] == "store_history"]
    need(len(stores) == len(names) and all(store["inputs"] == [observation["id"]]
         and store["attrs"] == dict(history=name, state=None)
         and observation["attrs"]["component"] == i and observation["attrs"]["field_unknown"] == handles[i]
         for i, (store, observation, name) in enumerate(zip(stores, observations, names, strict=True))), "IR history observation route differs")
    need(ir["histories"] == [dict(lag=1, name=name, ncomp=1, state=None) for name in names]
         and ir["history_persistence"] == [dict(depth=2, name=name, policy=dict(kind="history-persistence",
             payload=dict(policy="dense"), protocol="pops.manifest", schema_version=1)) for name in names],
         "IR declared physical history differs")


def program_hash(ir):
    semantic = copy.deepcopy(ir)
    for node in semantic["nodes"]:
        node.pop("provenance", None)
    return files.digest(json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def origins(owner, roots):
    exact(owner, ("schema", "source_commit", "native_build_source_commit", "abi_key", "python_package", "sdk", "native", "sources", "cpp_dso_links"), "execution owner")
    need(owner["schema"] == "sol61.stage-execution-owner@1", "owner schema differs")
    for name in ("source_commit", "native_build_source_commit"):
        need((name == "native_build_source_commit" and owner[name] is None)
             or (type(owner[name]) is str and re.fullmatch("[0-9a-f]{40}", owner[name])), "source commit differs")
    need(type(owner["abi_key"]) is str and owner["abi_key"], "ABI absent")
    for name in ("python_package", "sdk", "native"):
        files.pinned(owner[name], roots, budget=files.MAX_BINARY_BYTES if name == "native" else protocol.MAX_FILE_BYTES)
    exact(owner["sources"], ("fixture", "physical_helper", "controls_helper"), "source owners")
    for row in owner["sources"].values():
        files.pinned(row, roots)
    need(owner["cpp_dso_links"] is None, "unsupported cryptographic CPP/DSO aggregate claim")


def junit(raw, rank, size, cases):
    tree = ET.fromstring(raw)
    entries = tree.findall(".//testcase")
    need(len(entries) == 8 and not tree.findall(".//failure") and not tree.findall(".//error")
         and not tree.findall(".//skipped"), "JUnit is not eight successful real cases")
    found = set()
    for test in entries:
        rows = test.findall("./properties/property")
        props = {n.attrib["name"]: n.attrib["value"] for n in rows}
        need(len(rows) == len(props), "duplicate JUnit property")
        exact(props, ("artifact_identity", "dimension", "rank", "size", "evidence_path", "evolved_stage_receipt"), "JUnit properties")
        need(props.get("dimension") == "2" and props.get("rank") == str(rank)
             and props.get("size") == str(size), "JUnit dimension/rank differs")
        phase = next((k for k, c in cases.items() if props.get("evidence_path") == c["directory"]), None)
        need(phase is not None and phase not in found
             and props.get("evolved_stage_receipt") == cases[phase]["receipt"]["path"], "JUnit case mapping differs")
        witness = cases[phase]["witness"]
        expected_name = "test_public_evolved_original_stage_saved_and_exact_replay[%d-%s-%s-%d]" % (
            witness["width"], str(witness["width"] == 2), str(witness["dt"]), witness["cells"])
        need(test.attrib.get("name") == expected_name
             and test.attrib.get("classname") == "tests.python.integration.runtime.test_public_evolved_original_stage"
             and props["artifact_identity"] == cases[phase]["artifact"], "JUnit physical case/artifact differs")
        found.add(phase)
    need(found == set(cases), "JUnit case coverage differs")


def assemble(root, directories, junits, owner, roots):
    root = wire.canonical(root)
    need(len(directories) == 8 and len(junits) in (1, 2), "eight cases / Serial or MPI2 required")
    need(type(roots) is list and len(roots) == len(set(roots)) and str(root) in roots, "file roots differ")
    origins(owner, roots)
    cases = {}
    for directory in directories:
        case = case_inventory(directory)
        need(Path(case["directory"]).is_relative_to(root), "case escapes archive")
        receipt = strict_json(files.pinned(case["receipt"], roots)[1])
        triple = receipt["cells"], receipt["dt"], receipt["evolved_states"]
        need(triple in CASES and key(*triple) not in cases, "foreign/duplicate Stage case")
        cases[key(*triple)] = case
    need(len(cases) == 8, "case coverage incomplete")
    junit_pins = [files.leaf(p) for p in junits]
    for rank, pin in enumerate(junit_pins):
        junit(files.pinned(pin, roots)[1], rank, len(junits), cases)
    return dict(schema="sol61.stage-owner-pins@1", qualification=QUALIFICATION,
                archive_root=str(root), file_roots=roots, mode="serial" if len(junits) == 1 else "mpi2",
                ranks=len(junits), owner=owner, junit=junit_pins, cases=cases)


def receive(pins_path, pins_sha, approval_path, approval_sha):
    raw, approval_raw = files.read(pins_path)[1], files.read(approval_path)[1]
    need(files.digest(raw) == pins_sha and files.digest(approval_raw) == approval_sha, "external seal differs")
    approval = strict_json(approval_raw)
    need(approval == dict(schema="sol61.stage-root-approval@1", approved_by="ROOT", pins_sha256=pins_sha,
                          qualification=QUALIFICATION), "ROOT approval scope differs")
    pins = strict_json(raw)
    exact(pins, ("schema", "qualification", "archive_root", "file_roots", "mode", "ranks", "owner", "junit", "cases"), "owner pins")
    need(pins["schema"] == "sol61.stage-owner-pins@1" and pins["qualification"] == QUALIFICATION
         and type(pins["ranks"]) is int and pins["ranks"] in (1, 2)
         and pins["mode"] == ("serial" if pins["ranks"] == 1 else "mpi2"), "owner mode differs")
    roots = pins["file_roots"]
    need(type(roots) is list and len(roots) == len(set(roots)) and pins["archive_root"] in roots, "approved roots differ")
    origins(pins["owner"], roots)
    exact(pins["cases"], [key(*case) for case in CASES], "closed witness cases")
    need(len(pins["junit"]) == pins["ranks"] and len({p["path"] for p in pins["junit"]}) == pins["ranks"], "JUnit inventory differs")
    for rank, pin in enumerate(pins["junit"]):
        junit(files.pinned(pin, roots)[1], rank, pins["ranks"], pins["cases"])
    reports = {}
    for label, case in pins["cases"].items():
        need(Path(case["directory"]).is_relative_to(wire.canonical(pins["archive_root"])), "case escapes archive")
        need(case_inventory(case["directory"]) == case, "sealed closed inventory differs")
        receipt = strict_json(files.pinned(case["receipt"], roots)[1])
        n, dt, w = receipt["cells"], receipt["dt"], receipt["evolved_states"]
        need(label == key(n, dt, w) and type(n) is int and type(w) is int and type(dt) is float,
             "typed witness identity differs")
        need(receipt["fixture_schema"] == "pops.evolved-stage-native-fixture@2"
             and receipt["dimension"] == 2 and receipt["rank"] == 0 and receipt["size"] == pins["ranks"]
             and receipt["candidate_diffusion"] is (w == 2) and receipt["unknowns"] == w+(w == 2)
             and receipt["newton"] == CONTROLS and receipt["fd_step"] == 1e-6 and receipt["acceptance"] == TOL
             and receipt["projection"] == "piecewise_constant_cell" and receipt["volume_sum"] == 1.
             and receipt["exact_restart_and_replay"] is True, "fixture contract differs")
        need(receipt["native"] == pins["owner"]["native"], "selected native differs")
        roles = ["block-Q%d" % i for i in range(w)] + ["block-forcing"]
        need(len(receipt["binaries"]) == w+2 and [r["component"] for r in receipt["binaries"][:-1]] == roles,
             "component ownership inventory differs")
        for binary in receipt["binaries"]:
            component(binary, roots)
        program = receipt["binaries"][-1]
        need(program["component"].startswith("program-") and type(program["compile_command"]) is str
             and program["compile_command"], "actual Program compiler command absent")
        need("-DPOPS_NATIVE_DIM=2" in shlex.split(program["compile_command"]), "compiler dimension differs")
        ir_record = receipt["program_irs"][0]
        need(ir_record["component"] == program["component"], "IR/DSO component association differs")
        ir = strict_json(files.pinned(case["program_irs"][0], roots)[1])
        need(program_hash(ir) == ir_record["program_hash"], "Program semantic IR digest differs")
        ir_physics(ir, w, dt)
        images = {phase: npz(row, roots) for phase, row in case["phases"].items()}
        initial = npz(case["initial"], roots)
        reports[label] = science(initial, images, n, dt, w)
        cps = {phase: npz(row, roots) for phase, row in case["checkpoints"].items()}
        lineage.review(*(cps[phase] for phase in CP))
        receipt_provenance(receipt, cps, dt)
        for phase in CP:
            checkpoint(cps[phase], images[phase], images["accepted"], n, dt, w,
                       pins["owner"]["abi_key"], receipt["artifact"], ir_record["program_hash"])
    return dict(schema="sol61.stage-scientific-reception@1", qualification=QUALIFICATION, cases=reports,
                mode=pins["mode"], ranks=pins["ranks"], unique_scientific_witnesses=8,
                pins_sha256=pins_sha, approval_sha256=approval_sha,
                source_commit=pins["owner"]["source_commit"], native_build_source_commit=pins["owner"]["native_build_source_commit"],
                component_associations={label: dict(Program_IR=case["program_irs"][0], Program_CPP=case["sources"][0],
                    Program_DSO=case["components"][-1], artifact=case["artifact"]) for label, case in pins["cases"].items()},
                gaps=["No independently authenticated cryptographic CPP-to-DSO aggregate; explicit component/source association only",
                      "Block DSO source/command records absent in fixture; owner attested binaries only",
                      "Uniform cell-constant finite witnesses, no AMR conservation or nonlinear Marshak/full family qualification"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    a = commands.add_parser("assemble")
    for name in ("archive-root", "owner", "output"):
        a.add_argument("--"+name, required=True)
    for name in ("case", "junit", "file-root"):
        a.add_argument("--"+name, action="append", default=[])
    r = commands.add_parser("receive")
    for name in ("pins", "pins-sha256", "approval", "approval-sha256"):
        r.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if args.mode == "assemble":
        value = assemble(args.archive_root, args.case, args.junit, strict_json(files.read(args.owner)[1]), args.file_root)
        encoded = json.dumps(value, indent=2, sort_keys=True)+"\n"
        target = Path(args.output)
        need(not target.exists() or target.read_text() == encoded, "existing pending inventory differs")
        target.write_text(encoded)
        print("pending two external ROOT seals; no native scientific qualification")
    else:
        print(json.dumps(receive(args.pins, args.pins_sha256, args.approval, args.approval_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
